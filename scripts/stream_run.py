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
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

OUTPUTS = [
    "site/data/stream.json",
    "site/media/stream-flow.gif",
    "site/media/stream-flow.mp4",
    "site/media/stream-force.gif",
    "site/media/stream-force.mp4",
]


def read_frames(folder: Path, window: int = 24) -> tuple[list[dict], int]:
    """Read only atomic finalized frames; never blend simulated states."""
    paths = sorted(folder.glob("frame-*.npz"))
    paths = [p for p in paths if not p.name.endswith(".tmp.npz")]
    count = len(paths)
    frames = []
    for path in paths[-window:]:
        with np.load(path, allow_pickle=False) as values:
            frame = {
                key: values[key].copy() for key in ("time", "axis", "velocity", "force")
            }
        if not all(np.isfinite(v).all() for v in frame.values()):
            raise ValueError(f"nonfinite snapshot: {path.name}")
        n = len(frame["axis"])
        if any(frame[k].shape != (n, n, n, 3) for k in ("velocity", "force")):
            raise ValueError("invalid vector shape")
        if not np.all(np.diff(frame["axis"]) > 0):
            raise ValueError("coordinates must increase")
        frames.append(frame)
    if frames and not np.all(np.diff([float(f["time"]) for f in frames]) > 0):
        raise ValueError("snapshot times must increase")
    return frames, count


def plane_vectors(frame: dict, field: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Nearest saved planes to zero, with their actual coordinate retained."""
    index = int(np.argmin(np.abs(frame["axis"])))
    vectors = frame[field]
    return vectors[:, index, :, :], vectors[:, :, index, :], float(frame["axis"][index])


def render_pair(
    frames: list[dict],
    field: str,
    stem: Path,
    resolution: int,
    half_domain: float = 1.0,
) -> dict:
    planes = [plane_vectors(frame, field) for frame in frames]
    magnitudes = [
        tuple(np.linalg.norm(p, axis=-1) for p in pair[:2]) for pair in planes
    ]
    peak = max(float(np.max(p)) for pair in magnitudes for p in pair)
    vmax = max(peak, 1e-12)
    norm = SymLogNorm(linthresh=0.02 * vmax, vmin=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "stream", ["#07111f", "#177eab", "#69d2e7", "#fff2c0"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), facecolor="#07111f")
    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.18, top=0.80, wspace=0.38)
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
            f"{label} · {held}={planes[0][2]:.3f}", color="#e9f1f5", fontsize=12
        )
        axis.set_xlabel("x", color="#9fb3c2")
        axis.set_ylabel("z" if j == 0 else "y", color="#9fb3c2")
        axis.tick_params(colors="#9fb3c2", labelsize=9)
        axis.set_xlim(-half_domain, half_domain)
        axis.set_ylim(-half_domain, half_domain)
    color_axis = fig.add_axes((0.89, 0.24, 0.02, 0.47))
    bar = fig.colorbar(images[0], cax=color_axis)
    bar.ax.tick_params(colors="#9fb3c2", labelsize=9)
    bar.set_label("|u|" if field == "velocity" else "|f|", color="#e9f1f5")
    title = fig.suptitle("", color="#e9f1f5", fontsize=14)
    fig.text(
        0.5,
        0.035,
        f"{resolution}³ solver → {len(coord)}³ display · actual saved frames · shared scale per clip",
        ha="center",
        color="#9fb3c2",
        fontsize=9,
    )

    def update(index: int):
        for j, image in enumerate(images):
            image.set_data(magnitudes[index][j].T)
        zero = " · zero field" if all(not np.any(p) for p in magnitudes[index]) else ""
        title.set_text(
            f"{'Velocity' if field == 'velocity' else 'Applied force'} · t = {float(frames[index]['time']):.6f}{zero}"
        )
        return [*images, title]

    # Hold a single real rest state; repetition does not invent a new fluid time.
    indices = list(range(len(frames))) if len(frames) > 1 else [0, 0]
    movie = FuncAnimation(fig, update, frames=indices, interval=125, blit=False)
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension, writer in (
            ("gif", PillowWriter(fps=8)),
            (
                "mp4",
                FFMpegWriter(
                    fps=8, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
                ),
            ),
        ):
            destination = stem.with_suffix(f".{extension}")
            temporary = stem.with_suffix(f".tmp.{extension}")
            movie.save(temporary, writer=writer, dpi=90)
            temporary.replace(destination)
    finally:
        plt.close(fig)
    return {
        "color_max": vmax,
        "plane_coordinate": planes[-1][2],
        "scale_policy": "shared across both planes and all frames in this clip; recomputed on publication",
    }


def build_manifest(remote: dict, frames: list[dict], count: int, render: dict) -> dict:
    times = [float(f["time"]) for f in frames]
    revision = hashlib.sha256(
        json.dumps([remote["source_commit"], times, count]).encode()
    ).hexdigest()[:12]
    return {
        "schema_version": 1,
        "id": "best-guess-n192-rest-t0985",
        "label": "Current best · streaming from rest",
        "status": remote["status"],
        "observed_at": remote["observed_at"],
        "started_at": remote["started_at"],
        "source_commit": remote["source_commit"],
        "config": remote["config"],
        "revision": revision,
        "latest_t": times[-1],
        "captured_frames": count,
        "clip_times": times,
        "display_resolution": len(frames[-1]["axis"]),
        "diagnostics": remote.get("diagnostics"),
        "progress": remote.get("progress"),
        "media": {
            "flow_gif": "media/stream-flow.gif",
            "flow_mp4": "media/stream-flow.mp4",
            "force_gif": "media/stream-force.gif",
            "force_mp4": "media/stream-force.mp4",
        },
        "render": render,
        "scope_warning": "Finite manufactured-solution surrogate; exploratory and spatially unvalidated. Display subsampling is not solver resolution. Rolling real-frame clips, not wall-clock playback.",
    }


def command(argv: list[str], **kwargs) -> str:
    return subprocess.check_output(
        argv, text=True, timeout=kwargs.pop("timeout", 120), **kwargs
    ).rstrip("\n")


def read_remote(ssh: list[str], host: str, run: str) -> dict:
    script = f"""import json, subprocess
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
print(json.dumps(dict(status='complete' if complete else ('running' if alive else 'stopped'),observed_at=datetime.now(timezone.utc).isoformat(),started_at=launch['started_at'],source_commit=launch['source_commit'],config=cfg,progress=progress,diagnostics=diagnostics)))
"""
    # The simulation interpreter, not an ambient system Python lacking NumPy.
    python = str(Path(run).parents[1] / ".venv/bin/python")
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
    parser.add_argument("--host", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument(
        "--deadline",
        required=True,
        help="ISO time with timezone; controller stops by this bound",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
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
                        )
                        for field, name in (("velocity", "flow"), ("force", "force"))
                    }
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
