"""Render smooth GIF and MP4 media from saved 2D solver checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

plt.switch_backend("Agg")

INK = "#07111f"
PAPER = "#e9f1f5"
MUTED = "#9fb5c5"
ACCENT = "#69d2e7"


def _load(run_dir: Path) -> dict[str, object]:
    with np.load(run_dir / "slices.npz") as data:
        payload = {name: data[name] for name in data.files}
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    payload.update(
        resolution=int(config["resolution"]),
        half_domain=float(config.get("half_domain", 1.0)),
        base_radius=float(config.get("base_radius", 0.52)),
        base_height=float(config.get("base_height", 0.52)),
        h=float(config.get("h", 0.008)),
        t_star=float(config.get("t_star", 1.0)),
        rest_until=float(config["paper_time_cutoff_start"]),
        max_dt=float(config.get("max_dt", 0.02)),
        cfl=float(config.get("cfl", 0.32)),
    )
    return payload


def _speed(velocity: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity, axis=-1)


def _playback_times(
    times: np.ndarray, rest_until: float, movie_frames: int
) -> np.ndarray:
    """Use a short rest hold, then sample evenly in similarity time."""
    active = times[times > rest_until]
    if len(active) == 0:
        return np.linspace(times[0], times[-1], movie_frames)
    rest_frames = max(2, round(0.08 * movie_frames))
    active_frames = movie_frames - rest_frames
    similarity_start = -np.log(1.0 - active[0])
    similarity_end = -np.log(1.0 - times[-1])
    similarity_clock = np.linspace(similarity_start, similarity_end, active_frames)
    active_clock = 1.0 - np.exp(-similarity_clock)
    return np.concatenate((np.full(rest_frames, times[0]), active_clock))


def _interpolate(field: np.ndarray, times: np.ndarray, sample_t: float) -> np.ndarray:
    right = int(np.searchsorted(times, sample_t, side="left"))
    if right == 0:
        return field[0]
    if right >= len(times):
        return field[-1]
    left = right - 1
    weight = (sample_t - times[left]) / (times[right] - times[left])
    return (1.0 - weight) * field[left] + weight * field[right]


def _style_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#29445d")


def render_run(
    run_dir: Path,
    output_dir: Path,
    stem: str,
    movie_frames: int,
    fps: int,
) -> None:
    data = _load(run_dir)
    resolution = int(data["resolution"])
    computed = _speed(np.asarray(data["velocity"]))
    target = _speed(np.asarray(data["target"]))
    times = np.asarray(data["times"])
    playback_times = _playback_times(
        times, float(data["rest_until"]), movie_frames
    )
    vmax = float(np.quantile(np.concatenate([computed.ravel(), target.ravel()]), 0.998))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=INK)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.11, top=0.85, wspace=0.06)
    images = []
    for axis, field, title_text in zip(
        axes, (computed, target), ("PhiFlow", "target"), strict=True
    ):
        image = axis.imshow(
            field[0].T, origin="lower", cmap="magma", vmin=0, vmax=vmax
        )
        axis.set_title(title_text, color=PAPER, fontsize=14, pad=10)
        _style_axis(axis)
        images.append(image)
    title = fig.suptitle("", color=PAPER, fontsize=16, y=0.96)
    note = fig.text(
        0.5,
        0.025,
        (
            f"{len(times)} solver checkpoints · similarity-time playback interpolation · "
            f"CFL {float(data['cfl']):g} · max Δt {float(data['max_dt']):.3g}"
        ),
        color=ACCENT,
        ha="center",
        fontsize=9.5,
    )

    def update(frame: int):
        sample_t = float(playback_times[frame])
        for image, field in zip(images, (computed, target), strict=True):
            image.set_data(_interpolate(field, times, sample_t).T)
        title.set_text(
            f"{resolution}³ from rest  ·  t={sample_t:.4f}  ·  τ={1-sample_t:.4f}"
        )
        return (*images, title, note)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / stem
    update(movie_frames - 1)
    fig.savefig(
        output_stem.with_name(f"{stem}-poster.png"), dpi=150, facecolor=INK
    )

    mp4 = FuncAnimation(
        fig, update, frames=movie_frames, interval=1000 / fps, blit=False
    )
    mp4.save(
        output_stem.with_suffix(".mp4"),
        writer=FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        dpi=120,
    )

    gif_stride = 2
    gif_frames = [*range(0, movie_frames - 1, gif_stride), movie_frames - 1]
    gif = FuncAnimation(
        fig,
        update,
        frames=gif_frames,
        interval=1000 * gif_stride / fps,
        blit=False,
    )
    gif.save(
        output_stem.with_suffix(".gif"),
        writer=PillowWriter(fps=max(1, fps // gif_stride)),
        dpi=105,
    )
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    parser.add_argument("--stem", default="baseline")
    parser.add_argument("--movie-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.movie_frames < 4:
        raise ValueError("--movie-frames must be at least 4")
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    render_run(args.run, args.output, args.stem, args.movie_frames, args.fps)
    from navier_stokes_sim.plotting import plot_snapshots
    from navier_stokes_sim.solver import load_simulation_result

    result = load_simulation_result(args.run)
    args.endpoint_output.parent.mkdir(parents=True, exist_ok=True)
    plot_snapshots(result, args.endpoint_output)
    print(f"wrote smooth media to {args.output} and {args.endpoint_output}")


if __name__ == "__main__":
    main()
