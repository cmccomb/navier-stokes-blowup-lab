"""Render direction-aware GIFs from a completed start-from-rest run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

from scripts.render_site_media import _interpolate, _load, _playback_times

plt.switch_backend("Agg")

INK = "#07111f"
PAPER = "#e9f1f5"
MUTED = "#9fb5c5"
GRID = "#29445d"


def _limit(field: np.ndarray, quantile: float = 0.997) -> float:
    value = float(np.quantile(np.abs(field), quantile))
    return max(value, np.finfo(float).eps)


def _vorticity_z(velocity: np.ndarray, dx: float) -> np.ndarray:
    du_y_dx = (
        np.roll(velocity[..., 1], -1, axis=-2)
        - np.roll(velocity[..., 1], 1, axis=-2)
    ) / (2 * dx)
    du_x_dy = (
        np.roll(velocity[..., 0], -1, axis=-1)
        - np.roll(velocity[..., 0], 1, axis=-1)
    ) / (2 * dx)
    return du_y_dx - du_x_dy


def _render_scalar_gif(
    times: np.ndarray,
    playback_times: np.ndarray,
    scalar: np.ndarray,
    output: Path,
    *,
    title: str,
    scalar_label: str,
    half_domain: float,
    fps: int,
    colors: tuple[str, str],
) -> None:
    resolution = scalar.shape[1]

    vmax = _limit(scalar)
    norm = SymLogNorm(
        linthresh=0.035 * vmax,
        linscale=0.75,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )
    fig, plot_axis = plt.subplots(figsize=(7.2, 6.4), facecolor=INK)
    fig.subplots_adjust(left=0.025, right=0.88, bottom=0.045, top=0.87)
    plot_axis.set_facecolor(INK)
    image = plot_axis.imshow(
        scalar[0].T,
        origin="lower",
        extent=(-half_domain, half_domain, -half_domain, half_domain),
        cmap=LinearSegmentedColormap.from_list(
            "signed_dark", [colors[0], INK, colors[1]], N=257
        ),
        norm=norm,
        interpolation="bilinear",
    )
    plot_axis.set_axis_off()
    frame_title = fig.suptitle("", color=PAPER, fontsize=15, y=0.95)
    colorbar = fig.colorbar(image, ax=plot_axis, fraction=0.047, pad=0.035)
    colorbar.set_label(scalar_label, color=MUTED)
    colorbar.ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=7)
    colorbar.outline.set_visible(False)
    note = fig.text(
        0.5,
        0.025,
        "Fixed color scale · interpolated similarity-time playback",
        color=MUTED,
        ha="center",
        fontsize=9,
    )

    def update(frame_index: int):
        sample_t = float(playback_times[frame_index])
        signed_scalar = _interpolate(scalar, times, sample_t)
        image.set_data(signed_scalar.T)
        frame_title.set_text(
            f"{title}  ·  {resolution}³  ·  t={sample_t:.4f}  ·  τ={1-sample_t:.4f}"
        )
        return image, frame_title, note

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig,
        update,
        frames=len(playback_times),
        interval=1000 / fps,
        blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)


def render_flow_gifs(
    run_dir: Path,
    output_dir: Path,
    stem: str,
    frames: int = 72,
    fps: int = 12,
) -> list[Path]:
    """Write axial-jet, equatorial-swirl, and pulse-force GIFs."""

    data = _load(run_dir)
    times = np.asarray(data["times"])
    playback_times = _playback_times(times, float(data["rest_until"]), frames)
    axial = np.asarray(data["velocity"])
    equatorial = np.asarray(data["equatorial_velocity"])
    force = np.asarray(data["force"])
    resolution = int(data["resolution"])
    half_domain = float(data["half_domain"])
    dx = 2 * half_domain / resolution

    axial_path = output_dir / f"{stem}-axial-jet.gif"
    _render_scalar_gif(
        times,
        playback_times,
        axial[..., 2],
        axial_path,
        title="Axial jet",
        scalar_label="vertical velocity  u_z",
        half_domain=half_domain,
        fps=fps,
        colors=("#58bfff", "#ffab72"),
    )

    swirl_path = output_dir / f"{stem}-equatorial-swirl.gif"
    _render_scalar_gif(
        times,
        playback_times,
        _vorticity_z(equatorial, dx),
        swirl_path,
        title="Equatorial swirl",
        scalar_label="vertical vorticity  ω_z",
        half_domain=half_domain,
        fps=fps,
        colors=("#b49aff", "#72ead2"),
    )

    forcing_path = output_dir / f"{stem}-pulse-forcing.gif"
    _render_scalar_gif(
        times,
        playback_times,
        force[..., 2],
        forcing_path,
        title="Applied forcing",
        scalar_label="vertical force  f_z",
        half_domain=half_domain,
        fps=fps,
        colors=("#61b9fa", "#f58bac"),
    )
    return [axial_path, swirl_path, forcing_path]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", default="best")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.frames < 4:
        raise ValueError("--frames must be at least 4")
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    paths = render_flow_gifs(args.run, args.output, args.stem, args.frames, args.fps)
    print("\n".join(f"wrote {path}" for path in paths))


if __name__ == "__main__":
    main()
