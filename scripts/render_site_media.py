"""Render compact GIF and MP4 result media from saved 2D slice checkpoints."""

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
ACCENT = "#69d2e7"


def _load(run_dir: Path) -> dict[str, object]:
    with np.load(run_dir / "slices.npz") as data:
        payload = {name: data[name] for name in data.files}
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    payload["resolution"] = int(metadata["config"]["resolution"])
    return payload


def _speed(velocity: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity, axis=-1)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#29445d")


def _save(animation: FuncAnimation, stem: Path, poster: plt.Figure) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    poster.savefig(stem.with_name(f"{stem.name}-poster.png"), dpi=150, facecolor=INK)
    animation.save(
        stem.with_suffix(".gif"),
        writer=PillowWriter(fps=4),
        dpi=105,
    )
    animation.save(
        stem.with_suffix(".mp4"),
        writer=FFMpegWriter(
            fps=4,
            codec="libx264",
            bitrate=1800,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        dpi=120,
    )


def render_late_time(run_dir: Path, output_dir: Path) -> None:
    data = _load(run_dir)
    resolution = int(data["resolution"])
    computed = _speed(np.asarray(data["velocity"]))
    target = _speed(np.asarray(data["target"]))
    times = np.asarray(data["times"])
    vmax = float(np.quantile(np.concatenate([computed.ravel(), target.ravel()]), 0.998))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=INK)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.85, wspace=0.06)
    images = []
    for axis, field, title in zip(axes, (computed, target), ("PhiFlow", "target")):
        image = axis.imshow(field[0].T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(title, color=PAPER, fontsize=14, pad=10)
        _style_axis(axis)
        images.append(image)
    title = fig.suptitle("", color=PAPER, fontsize=16, y=0.96)
    scale = fig.text(0.5, 0.02, "shared speed scale", color=ACCENT, ha="center", fontsize=10)

    def update(frame: int):
        for image, field in zip(images, (computed, target)):
            image.set_data(field[frame].T)
        title.set_text(
            f"{resolution}³ late-time concentration  ·  "
            f"t={times[frame]:.3f}  ·  τ={1-times[frame]:.3f}"
        )
        return (*images, title, scale)

    animation = FuncAnimation(fig, update, frames=len(times), interval=250, blit=False)
    update(len(times) - 1)
    _save(animation, output_dir / "current-best", fig)
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--late-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    render_late_time(args.late_run, args.output)
    print(f"wrote GIF, MP4, and poster media to {args.output}")


if __name__ == "__main__":
    main()
