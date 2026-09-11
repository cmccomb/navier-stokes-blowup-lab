"""Bounded per-run publisher: atomic solver snapshots -> compact media -> main.

Run on the designated controller in a dedicated clean clone. No scheduler,
service installation, credentials, or source-machine paths are published.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from PIL import Image

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.interactive import (
    RESPONSIVE_VOLUME_SCRIPT,
    build_volume_figure,
)
from navier_stokes_sim.volume_series import VolumeSeries

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

OUTPUTS = [
    "site/data/stream.json",
    "site/media/stream-flow.gif",
    "site/media/stream-flow.mp4",
    "site/media/stream-force.gif",
    "site/media/stream-force.mp4",
    "site/media/stream-flow-3d.html",
    "site/media/stream-force-3d.html",
]


def build_playback(times: list[float], slow_motion_after: float | None = None) -> dict:
    """Hold state i over [t_i, t_i+1); quantize transitions, never fluid data.

    The optional final six active intervals use a labeled slower linear clock.
    Total traversal lasts at most 11.5 seconds, followed by a 0.5 s editorial
    end pause. A 50 Hz clock gives both MP4 and
    GIF the same 20 ms timing (two GIF centiseconds, avoiding sub-20 ms delays).
    Cumulative rounding avoids accumulating one rounding error per interval.
    Extremely close samples retain at least one tick; report that distortion.
    """
    values = np.asarray(times, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not np.all(np.diff(values) > 0)
    ):
        raise ValueError("playback needs finite, strictly increasing saved times")
    if len(values) > 24:
        raise ValueError("playback is bounded to 24 source frames")
    fps = 50
    span = float(values[-1] - values[0])
    traversal_ticks = round(np.clip(span / 0.05, 4, 11.5) * fps) if span else 0
    ideal = (values - values[0]) / span * traversal_ticks if span else np.zeros(1)
    segments = []
    if span:
        segments = [
            {
                "label": "Replay",
                "start_index": 0,
                "end_index": len(values) - 1,
                "start_t": float(values[0]),
                "end_t": float(values[-1]),
                "simulation_units_per_playback_second": span / (traversal_ticks / fps),
                "duration_seconds": traversal_ticks / fps,
                "slowdown_factor": 1.0,
            }
        ]
    if span and slow_motion_after is not None:
        split = max(len(values) - 7, int(np.searchsorted(values, slow_motion_after)))
        if split < len(values) - 1:
            prefix_span = float(values[split] - values[0])
            slow_span = float(values[-1] - values[split])
            prefix_ticks = (
                round(np.clip(prefix_span / 0.05, 2, 4) * fps) if split else 0
            )
            base_rate = (
                prefix_span / (prefix_ticks / fps)
                if split
                else max(0.05, 4 * slow_span / 11.5)
            )
            slow_ticks = round(
                np.clip(4 * slow_span / base_rate, 4, 11.5 - prefix_ticks / fps) * fps
            )
            slow_rate = slow_span / (slow_ticks / fps)
            # Never call an accelerated end segment slow motion on irregular input.
            if slow_rate < base_rate:
                segments = []
                if split:
                    ideal[: split + 1] = (
                        (values[: split + 1] - values[0]) / prefix_span * prefix_ticks
                    )
                    segments.append(
                        {
                            "label": "Replay",
                            "start_index": 0,
                            "end_index": split,
                            "start_t": float(values[0]),
                            "end_t": float(values[split]),
                            "simulation_units_per_playback_second": base_rate,
                            "duration_seconds": prefix_ticks / fps,
                            "slowdown_factor": 1.0,
                        }
                    )
                ideal[split:] = (
                    prefix_ticks
                    + (values[split:] - values[split]) / slow_span * slow_ticks
                )
                traversal_ticks = prefix_ticks + slow_ticks
                segments.append(
                    {
                        "label": "Slow motion",
                        "start_index": split,
                        "end_index": len(values) - 1,
                        "start_t": float(values[split]),
                        "end_t": float(values[-1]),
                        "simulation_units_per_playback_second": slow_rate,
                        "reference_simulation_units_per_playback_second": base_rate,
                        "duration_seconds": slow_ticks / fps,
                        "slowdown_factor": base_rate / slow_rate,
                    }
                )
    boundaries = np.rint(ideal).astype(int)
    for i in range(1, len(boundaries) - 1):
        boundaries[i] = np.clip(
            boundaries[i],
            boundaries[i - 1] + 1,
            traversal_ticks - (len(values) - 1 - i),
        )
    end_ticks = 25 if span else 200
    repeats = [*np.diff(boundaries).tolist(), end_ticks]
    return {
        "mode": "piecewise simulation-time-proportional zero-order hold",
        "applies_to": ["flow_gif", "flow_mp4", "force_gif", "force_mp4"],
        "simulation_time_span": span,
        "segments": segments,
        "slow_motion_policy": "last up to six saved intervals starting at or after activation; linear time within each segment; preceding segment capped at 4 s",
        "traversal_seconds": traversal_ticks / fps,
        "terminal_hold_seconds": end_ticks / fps,
        "duration_seconds": (traversal_ticks + end_ticks) / fps,
        "mp4_fps": fps,
        "mp4_frame_repeats": repeats,
        "gif_timing_quantum_ms": 20,
        "source_frame_duration_ms": [int(n * 20) for n in repeats],
        "max_transition_rounding_error_ms": float(np.max(np.abs(boundaries - ideal)))
        * 20,
        "terminal_hold_policy": "editorial pause at the final saved time, not additional simulated time",
        "quantization_policy": "nearest cumulative 20 ms boundary; at least one tick per source state",
    }


def read_frames(folder: Path, window: int = 24) -> tuple[list[dict], int]:
    """Derive native planes and browser volumes from each finalized archive.

    Keep at most one native vector field in memory, independent of clip length.
    The archival file is never rewritten, moved, or reduced.
    """
    if not 1 <= window <= 24:
        raise ValueError("display window must contain 1 to 24 source frames")
    paths = sorted(folder.glob("frame-*.npz"))
    paths = [p for p in paths if not p.name.endswith(".tmp.npz")]
    count = len(paths)
    frames = []
    for path in paths[-window:]:
        with np.load(path, allow_pickle=False) as values:
            frame = {key: values[key].copy() for key in ("time", "axis")}
            if not all(np.isfinite(v).all() for v in frame.values()):
                raise ValueError(f"nonfinite snapshot: {path.name}")
            n = len(frame["axis"])
            if n < 2 or not np.all(np.diff(frame["axis"]) > 0):
                raise ValueError("coordinates must increase")
            if frames and not np.array_equal(frame["axis"], frames[0]["axis"]):
                raise ValueError("archive coordinates changed within the clip")
            index = int(np.argmin(np.abs(frame["axis"])))
            stride = max(1, int(np.ceil(n / 32)))
            frame["volume_axis"] = frame["axis"][::stride].copy()
            for field in ("velocity", "force"):
                vectors = values[field]
                if vectors.shape != (n, n, n, 3):
                    raise ValueError("invalid vector shape")
                if not np.isfinite(vectors).all():
                    raise ValueError(f"nonfinite snapshot: {path.name}")
                frame[f"{field}_planes"] = np.stack(
                    (vectors[:, index, :, :], vectors[:, :, index, :])
                ).astype(np.float32)
                frame[field] = np.array(
                    vectors[::stride, ::stride, ::stride], dtype=np.float32, copy=True
                )
                frame[f"{field}_archive_dtype"] = str(vectors.dtype)
                del vectors
        frames.append(frame)
    if frames and not np.all(np.diff([float(f["time"]) for f in frames]) > 0):
        raise ValueError("snapshot times must increase")
    return frames, count


def plane_vectors(frame: dict, field: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Nearest saved planes to zero, with their actual coordinate retained."""
    index = int(np.argmin(np.abs(frame["axis"])))
    if f"{field}_planes" in frame:
        planes = frame[f"{field}_planes"]
        return planes[0], planes[1], float(frame["axis"][index])
    vectors = frame[field]
    return vectors[:, index, :, :], vectors[:, :, index, :], float(frame["axis"][index])


def render_pair(
    frames: list[dict],
    field: str,
    stem: Path,
    resolution: int,
    half_domain: float = 1.0,
    slow_motion_after: float | None = None,
) -> dict:
    playback = build_playback(
        [float(frame["time"]) for frame in frames], slow_motion_after
    )
    planes = [plane_vectors(frame, field) for frame in frames]
    magnitudes = [
        tuple(np.linalg.norm(p.astype(np.float64), axis=-1) for p in pair[:2])
        for pair in planes
    ]
    peak = max(float(np.max(p)) for pair in magnitudes for p in pair)
    vmax = peak if peak > 0 else 1.0
    norm = SymLogNorm(linthresh=0.02 * vmax, vmin=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "stream", ["#07111f", "#177eab", "#69d2e7", "#fff2c0"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=180, facecolor="#07111f")
    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.23, top=0.80, wspace=0.38)
    coord = frames[0]["axis"]
    spacing = float(coord[1] - coord[0])
    extent = (coord[0] - spacing / 2, coord[-1] + spacing / 2) * 2
    images = []
    for j, (axis, label) in enumerate(zip(axes, ("x–z", "x–y"))):
        axis.set_facecolor("#07111f")
        images.append(
            axis.imshow(
                magnitudes[0][j].T,
                origin="lower",
                extent=extent,
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
        )
        held = "y" if j == 0 else "z"
        axis.set_title(
            f"{label} · {held}={planes[0][2]:.3f}", color="#e9f1f5", fontsize=12.8
        )
        axis.set_xlabel("x", color="#9fb3c2", fontsize=11.2)
        axis.set_ylabel("z" if j == 0 else "y", color="#9fb3c2", fontsize=11.2)
        axis.tick_params(colors="#9fb3c2", labelsize=11.2)
        axis.set_xlim(-half_domain, half_domain)
        axis.set_ylim(-half_domain, half_domain)
    color_axis = fig.add_axes((0.89, 0.24, 0.02, 0.47))
    bar = fig.colorbar(images[0], cax=color_axis)
    bar.ax.tick_params(colors="#9fb3c2", labelsize=11.2)
    bar.set_label(
        "|u|" if field == "velocity" else "|f|", color="#e9f1f5", fontsize=11.2
    )
    fig.text(
        0.11,
        0.92,
        "Velocity" if field == "velocity" else "Applied force",
        color="#e9f1f5",
        fontsize=16,
        ha="left",
    )
    time_label = fig.text(
        0.50,
        0.92,
        "",
        color="#e9f1f5",
        fontsize=12.8,
        family="DejaVu Sans Mono",
        ha="left",
    )
    status_label = fig.text(0.88, 0.92, "", color="#9fb3c2", fontsize=11.2, ha="right")
    fig.text(
        0.5,
        0.035,
        f"{resolution}³ solver · {len(coord)}² planes · actual saved frames · shared scale per clip",
        ha="center",
        color="#9fb3c2",
        fontsize=11.2,
    )

    replay_label = fig.text(
        0.11,
        0.09,
        "",
        ha="left",
        color="#9fb3c2",
        fontsize=11.2,
    )

    def update(index: int):
        for j, image in enumerate(images):
            image.set_data(magnitudes[index][j].T)
        time_label.set_text(f"t = {float(frames[index]['time']):.6f}")
        status_label.set_text(
            "zero field" if all(not np.any(p) for p in magnitudes[index]) else ""
        )
        segments = [s for s in playback["segments"] if s["start_index"] <= index]
        if not segments:
            replay_label.set_text("Single saved state · no simulated-time advance")
        else:
            segment = segments[-1]
            rate = segment["simulation_units_per_playback_second"]
            prefix = (
                f"Slow motion · {segment['slowdown_factor']:.1f}× slower"
                if segment["label"] == "Slow motion"
                else "Replay"
            )
            replay_label.set_text(f"{prefix} · {rate:.4g} simulation units/s")

    stem.parent.mkdir(parents=True, exist_ok=True)
    video = stem.with_suffix(".tmp.mp4")
    gif = stem.with_suffix(".tmp.gif")
    stills = []
    try:
        width, height = fig.canvas.get_width_height()
        # Render each source state once; repetition affects only encoded timing.
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [
                    str(matplotlib.rcParams["animation.ffmpeg_path"]),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(playback["mp4_fps"]),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(video),
                ],
                stdin=subprocess.PIPE,
                stderr=errors,
            )
            try:
                for i, repeats in enumerate(playback["mp4_frame_repeats"]):
                    update(i)
                    fig.canvas.draw()
                    pixels = bytes(fig.canvas.buffer_rgba())
                    stills.append(
                        Image.frombytes("RGBA", (width, height), pixels).convert("RGB")
                    )
                    for _ in range(repeats):
                        process.stdin.write(pixels)
                process.stdin.close()
                if process.wait(timeout=120):
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace"))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        # Fixed palette keeps text colors stable as the data and timestamp change.
        background = np.asarray([7, 17, 31])
        palette_colors = (
            np.concatenate(
                [
                    cmap(np.linspace(0, 1, 128))[:, :3] * 255,
                    np.linspace(background, [233, 241, 245], 64),
                    np.linspace(background, [159, 179, 194], 64),
                ]
            )
            .round()
            .astype(np.uint8)
        )
        palette = Image.new("P", (1, 1))
        palette.putpalette(palette_colors.ravel().tolist())
        indexed = [
            still.quantize(palette=palette, dither=Image.Dither.NONE)
            for still in stills
        ]
        indexed[0].save(
            gif,
            save_all=True,
            append_images=indexed[1:],
            duration=playback["source_frame_duration_ms"]
            if len(indexed) > 1
            else playback["source_frame_duration_ms"][0],
            loop=0,
            disposal=2,
        )
        for still in indexed:
            still.close()
        palette.close()
        video.replace(stem.with_suffix(".mp4"))
        gif.replace(stem.with_suffix(".gif"))
    finally:
        plt.close(fig)
        for still in stills:
            still.close()
    return {
        "color_max": vmax,
        "plane_coordinate": planes[-1][2],
        "scale_policy": "shared across both planes and all frames in this clip; recomputed on publication",
    }


def build_manifest(remote: dict, frames: list[dict], count: int, render: dict) -> dict:
    times = [float(f["time"]) for f in frames]
    revision = hashlib.sha256(
        json.dumps(
            [
                remote.get("id"),
                remote["source_commit"],
                remote["config"],
                times,
                count,
                "renderer-v10",
            ]
        ).encode()
    ).hexdigest()[:12]
    return {
        "schema_version": 1,
        "id": remote.get("id", "best-guess-n192-rest-t0985"),
        "label": "Current best · streaming from rest",
        "status": remote["status"],
        "observed_at": remote["observed_at"],
        "started_at": remote["started_at"],
        "source_commit": remote["source_commit"],
        "config": remote["config"],
        "revision": revision,
        "render_revision": 10,
        "latest_t": times[-1],
        "captured_frames": count,
        "clip_times": times,
        "playback": build_playback(
            times, remote["config"].get("paper_time_cutoff_start")
        ),
        "display_resolution": len(frames[-1]["axis"]),
        "display_resolution_3d": len(frames[-1].get("volume_axis", frames[-1]["axis"])),
        "archive": {
            "resolution": len(frames[-1]["axis"]),
            "fields": ["velocity", "force"],
            "components_per_field": 3,
            "dtype": frames[-1].get("velocity_archive_dtype", "unknown"),
            "policy": "Immutable saved fields; native planes and reduced browser volumes derived after capture",
        },
        "diagnostics": remote.get("diagnostics"),
        "progress": remote.get("progress"),
        "media": {
            "flow_gif": "media/stream-flow.gif",
            "flow_mp4": "media/stream-flow.mp4",
            "force_gif": "media/stream-force.gif",
            "force_mp4": "media/stream-force.mp4",
            "flow_3d": "media/stream-flow-3d.html",
            "force_3d": "media/stream-force-3d.html",
        },
        "render": render,
        "scope_warning": "Finite manufactured-solution surrogate; exploratory and spatially unvalidated. Display subsampling is not solver resolution. Rolling real-frame clips, not wall-clock playback.",
    }


def render_3d(frames: list[dict], field: str, destination: Path, config: dict) -> None:
    """Render the browser copy; native three-component archives remain intact."""
    series = VolumeSeries(
        np.stack([f[field] for f in frames]),
        np.asarray([float(f["time"]) for f in frames]),
        frames[-1].get("volume_axis", frames[-1]["axis"]),
        SimulationConfig(**config),
        field,
    )
    figure = build_volume_figure(series)
    figure.update_layout(
        font={"family": "Arial, Helvetica, sans-serif", "size": 16}, title_font_size=20
    )
    for name in ("xaxis", "yaxis", "zaxis"):
        figure.layout.scene[name].update(
            tickfont={"size": 14}, title={"font": {"size": 14}}
        )
    for annotation in figure.layout.annotations:
        annotation.font.size = 14
        if len(series.axis) < len(frames[-1]["axis"]):
            annotation.text += f"<br>Derived from {len(frames[-1]['axis'])}³ saved fields; archive retained"
    for slider in figure.layout.sliders:
        slider.font.size = 14
        slider.currentvalue.font.size = 16
    temporary = destination.with_suffix(".tmp.html")
    html = figure.to_html(
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        div_id="stream-volume",
        config={"responsive": True, "displaylogo": False},
        post_script=RESPONSIVE_VOLUME_SCRIPT,
    )
    html = html.replace(
        "<head>",
        '<head><meta name="viewport" content="width=device-width, initial-scale=1" />'
        "<style>html,body{margin:0;background:#07111f;color:#e9f1f5;font-family:Arial,Helvetica,sans-serif}"
        ".modebar{top:55px!important}</style>",
        1,
    )
    temporary.write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def command(argv: list[str], **kwargs) -> str:
    return subprocess.check_output(
        argv, text=True, timeout=kwargs.pop("timeout", 120), **kwargs
    ).rstrip("\n")


def read_remote(ssh: list[str], host: str, run: str) -> dict:
    script = f"""import csv, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
p=Path({run!r})
launch=json.loads((p/'launch.json').read_text())
cfg=json.loads((p/'preview-volumes/manifest.json').read_text())['config']
result=subprocess.run(['ps','-p',str(launch['pid']),'-o','command='],capture_output=True,text=True)
alive=result.returncode==0 and str(p) in result.stdout and 'navier_stokes_sim.cli' in result.stdout
progress=json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None
diagnostics=None
if (p/'partial-checkpoint.npz').exists():
 with np.load(p/'partial-checkpoint.npz',allow_pickle=False) as f:
  history=json.loads(str(f['metadata'].item()))['diagnostics']
  diagnostics=history[-1] if history else None
complete=(p/'run.json').exists() and (p/'final-state.npz').exists()
if complete:
 with (p/'diagnostics.csv').open() as handle:
  rows=list(csv.DictReader(handle))
 if rows:
  diagnostics={{k: (v.lower()=='true' if v.lower() in ('true','false') else float(v)) for k,v in rows[-1].items()}}
paths=sorted(f for f in (p/'preview-volumes').glob('frame-*.npz') if not f.name.endswith('.tmp.npz'))
latest=None
if paths:
 with np.load(paths[-1],allow_pickle=False) as f: latest=float(f['time'])
print(json.dumps(dict(id=p.name,status='complete' if complete else ('running' if alive else 'stopped'),observed_at=datetime.now(timezone.utc).isoformat(),started_at=launch['started_at'],source_commit=launch['source_commit'],config=cfg,progress=progress,diagnostics=diagnostics,captured_frames=len(paths),latest_t=latest)))
"""
    # The simulation interpreter, not an ambient system Python lacking NumPy.
    python = str(Path(run).parents[1] / ".venv/bin/python")
    if host == "local":
        return json.loads(command([python, "-c", script]))
    return json.loads(command([*ssh, host, shlex.join([python, "-c", script])]))


def publish(repo: Path, message: str) -> str:
    dirty = command(["git", "status", "--porcelain"], cwd=repo)
    if any(line[3:] not in OUTPUTS for line in dirty.splitlines()):
        raise RuntimeError("publisher clone has changes outside its allowlist")
    command(["git", "add", "--", *OUTPUTS], cwd=repo)
    if command(["git", "diff", "--cached", "--name-only"], cwd=repo):
        command(["git", "commit", "-m", message], cwd=repo)
    for attempt in range(3):
        try:
            command(["git", "pull", "--rebase", "origin", "main"], cwd=repo)
            command(["git", "push", "origin", "HEAD:main"], cwd=repo)
            return command(["git", "rev-parse", "HEAD"], cwd=repo)
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable publish retry state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Source SSH host, or local")
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument(
        "--deadline",
        required=True,
        help="ISO time with timezone; controller stops by this bound",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if args.host != "local" and args.identity is None:
        parser.error("an explicit SSH identity is required for a remote source")
    if args.host == "local" and not args.once:
        parser.error("local publication is a bounded --once operation")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline needs a timezone")
    args.cache.mkdir(parents=True, exist_ok=True)
    lock = (args.cache / "publisher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh = [
        "ssh",
        "-i",
        str(args.identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    last_key = None
    last_render = None
    failures = 0
    while datetime.now(UTC) < deadline:
        try:
            remote = read_remote(ssh, args.host, args.run)
            if args.host == "local":
                folder = Path(args.run) / "preview-volumes"
            else:
                # Dense archives stay on their owning machine. Render there
                # with --host local --once, then transfer only compact outputs.
                if remote["config"]["preview_resolution"] > 32:
                    raise RuntimeError(
                        "render native archives on their source with --host local --once"
                    )
                folder = args.cache / "preview-volumes"
                folder.mkdir(exist_ok=True)
                command(
                    [
                        "rsync",
                        "-az",
                        "--exclude",
                        "*.tmp*",
                        "--exclude",
                        ".*",
                        "-e",
                        shlex.join(ssh),
                        f"{args.host}:{args.run}/preview-volumes/",
                        str(folder) + "/",
                    ]
                )
            frames, count = read_frames(folder)
            if not frames:
                raise RuntimeError("no finalized preview snapshot yet")
            frame_key = (count, float(frames[-1]["time"]))
            key = (*frame_key, remote["status"])
            if key != last_key:
                if frame_key != last_render:
                    rendering = {
                        field: render_pair(
                            frames,
                            field,
                            args.repo / f"site/media/stream-{name}",
                            remote["config"]["resolution"],
                            remote["config"]["half_domain"],
                            remote["config"]["paper_time_cutoff_start"],
                        )
                        for field, name in (("velocity", "flow"), ("force", "force"))
                    }
                    for field, name in (("velocity", "flow"), ("force", "force")):
                        render_3d(
                            frames,
                            field,
                            args.repo / f"site/media/stream-{name}-3d.html",
                            remote["config"],
                        )
                    last_render = frame_key
                manifest = build_manifest(remote, frames, count, rendering)
                destination = args.repo / "site/data/stream.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(manifest, indent=2) + "\n")
                commit = (
                    None
                    if args.no_push
                    else publish(
                        args.repo,
                        f"Stream 192-cubed frame {count}: t={manifest['latest_t']:.6f}",
                    )
                )
                receipt = {
                    "published_at": datetime.now(UTC).isoformat(),
                    "commit": commit,
                    "frames": count,
                    "latest_t": manifest["latest_t"],
                    "status": remote["status"],
                }
                (args.cache / "receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                print(json.dumps(receipt), flush=True)
                last_key = key
            failures = 0
            if args.once or remote["status"] in ("complete", "stopped"):
                return
        except Exception as exc:
            failures += 1
            failure = {
                "failed_at": datetime.now(UTC).isoformat(),
                "consecutive_failures": failures,
                "error": str(exc),
            }
            (args.cache / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            print(json.dumps(failure), flush=True)
            if args.once or failures >= 5:
                raise
        time.sleep(max(5, args.poll_seconds))
    print(
        json.dumps({"status": "deadline-reached", "at": datetime.now(UTC).isoformat()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
