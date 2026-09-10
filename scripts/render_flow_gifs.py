"""Render direction-aware GIFs from a completed start-from-rest run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import SymLogNorm

from scripts.render_site_media import _interpolate, _load, _playback_times

plt.switch_backend("Agg")

INK = "#07111f"
PAPER = "#e9f1f5"
MUTED = "#9fb5c5"
GRID = "#29445d"


def _limit(field: np.ndarray, quantile: float = 0.997) -> float:
    value = float(np.quantile(np.abs(field), quantile))
    return max(value, np.finfo(float).eps)


def _direction_vectors(
    horizontal: np.ndarray, vertical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(horizontal, vertical)
    threshold = 0.04 * max(float(np.quantile(magnitude, 0.98)), 1e-14)
    visible = magnitude >= threshold
    first = np.divide(
        horizontal,
        magnitude,
        out=np.full_like(horizontal, np.nan),
        where=visible,
    )
    second = np.divide(
        vertical,
        magnitude,
        out=np.full_like(vertical, np.nan),
        where=visible,
    )
    return first, second


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


def _style(axis: plt.Axes, horizontal_label: str, vertical_label: str) -> None:
    axis.set_xlabel(horizontal_label, color=MUTED)
    axis.set_ylabel(vertical_label, color=MUTED)
    axis.tick_params(colors=MUTED, length=3)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.set_aspect("equal")


def _render_vector_gif(
    field: np.ndarray,
    times: np.ndarray,
    playback_times: np.ndarray,
    scalar: np.ndarray,
    output: Path,
    *,
    title: str,
    scalar_label: str,
    horizontal_label: str,
    vertical_label: str,
    horizontal_component: int,
    vertical_component: int,
    half_domain: float,
    fps: int,
    cmap: str,
) -> None:
    resolution = field.shape[1]
    axis_values = np.linspace(-half_domain, half_domain, resolution, endpoint=False)
    axis_values += half_domain / resolution
    stride = max(1, resolution // 25)
    sparse_axis = axis_values[::stride]
    grid_x, grid_y = np.meshgrid(sparse_axis, sparse_axis, indexing="xy")

    vmax = _limit(scalar)
    norm = SymLogNorm(
        linthresh=0.035 * vmax,
        linscale=0.75,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )
    fig, plot_axis = plt.subplots(figsize=(7.2, 6.4), facecolor=INK)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.14, top=0.86)
    plot_axis.set_facecolor(INK)
    image = plot_axis.imshow(
        scalar[0].T,
        origin="lower",
        extent=(-half_domain, half_domain, -half_domain, half_domain),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )
    initial_u, initial_v = _direction_vectors(
        field[0, ::stride, ::stride, horizontal_component],
        field[0, ::stride, ::stride, vertical_component],
    )
    arrows = plot_axis.quiver(
        grid_x,
        grid_y,
        initial_u.T,
        initial_v.T,
        color=PAPER,
        alpha=0.66,
        angles="xy",
        scale_units="xy",
        scale=18,
        width=0.0022,
        headwidth=3.4,
        headlength=4.2,
        pivot="mid",
    )
    _style(plot_axis, horizontal_label, vertical_label)
    frame_title = fig.suptitle("", color=PAPER, fontsize=15, y=0.955)
    colorbar = fig.colorbar(image, ax=plot_axis, fraction=0.047, pad=0.035)
    colorbar.set_label(scalar_label, color=MUTED)
    colorbar.ax.tick_params(colors=MUTED)
    note = fig.text(
        0.5,
        0.035,
        "signed field · arrows show local in-plane direction · similarity-time playback",
        color=MUTED,
        ha="center",
        fontsize=9,
    )

    def update(frame_index: int):
        sample_t = float(playback_times[frame_index])
        velocity = _interpolate(field, times, sample_t)
        signed_scalar = _interpolate(scalar, times, sample_t)
        arrow_u, arrow_v = _direction_vectors(
            velocity[::stride, ::stride, horizontal_component],
            velocity[::stride, ::stride, vertical_component],
        )
        image.set_data(signed_scalar.T)
        arrows.set_UVC(arrow_u.T, arrow_v.T)
        frame_title.set_text(
            f"{title}  ·  {resolution}³  ·  t={sample_t:.4f}  ·  τ={1-sample_t:.4f}"
        )
        return image, arrows, frame_title, note

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
    _render_vector_gif(
        axial,
        times,
        playback_times,
        axial[..., 2],
        axial_path,
        title="Axial jet",
        scalar_label="vertical velocity  u_z",
        horizontal_label="x",
        vertical_label="z",
        horizontal_component=0,
        vertical_component=2,
        half_domain=half_domain,
        fps=fps,
        cmap="RdBu_r",
    )

    swirl_path = output_dir / f"{stem}-equatorial-swirl.gif"
    _render_vector_gif(
        equatorial,
        times,
        playback_times,
        _vorticity_z(equatorial, dx),
        swirl_path,
        title="Equatorial swirl",
        scalar_label="vertical vorticity  ω_z",
        horizontal_label="x",
        vertical_label="y",
        horizontal_component=0,
        vertical_component=1,
        half_domain=half_domain,
        fps=fps,
        cmap="PuOr_r",
    )

    forcing_path = output_dir / f"{stem}-pulse-forcing.gif"
    _render_vector_gif(
        force,
        times,
        playback_times,
        force[..., 2],
        forcing_path,
        title="Pulse forcing",
        scalar_label="vertical force  f_z",
        horizontal_label="x",
        vertical_label="z",
        horizontal_component=0,
        vertical_component=2,
        half_domain=half_domain,
        fps=fps,
        cmap="coolwarm",
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
