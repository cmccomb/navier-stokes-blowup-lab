"""Render direction-aware GIFs from a completed start-from-rest run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from scipy.ndimage import map_coordinates

from scripts.render_site_media import _load

plt.switch_backend("Agg")
plt.rcParams["font.family"] = "Arial"

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


def _similarity_resample(
    scalar: np.ndarray,
    times: np.ndarray,
    *,
    half_domain: float,
    base_radius: float,
    base_height: float,
    h: float,
    t_star: float,
    samples: int = 192,
    extent: float = 2.0,
) -> np.ndarray:
    """Resample an axial plane on a fixed ``(x/l_r, z/l_z)`` window."""

    normalized_axis = np.linspace(-extent, extent, samples)
    normalized_x, normalized_z = np.meshgrid(
        normalized_axis, normalized_axis, indexing="ij"
    )
    source_resolution = scalar.shape[1]
    dx = 2 * half_domain / source_resolution
    resampled = []
    for frame, t in zip(scalar, times, strict=True):
        tau = max(t_star - float(t), np.finfo(float).tiny)
        radial_length = base_radius * tau**0.5
        axial_length = base_height * tau ** (0.5 - h)
        physical_x = normalized_x * radial_length
        physical_z = normalized_z * axial_length
        index_x = (physical_x + half_domain) / dx - 0.5
        index_z = (physical_z + half_domain) / dx - 0.5
        resampled.append(
            map_coordinates(
                frame,
                (index_x, index_z),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        )
    return np.stack(resampled)


def _render_scalar_gif(
    times: np.ndarray,
    frame_indices: np.ndarray,
    scalar: np.ndarray,
    output: Path,
    *,
    title: str,
    scalar_label: str,
    half_domain: float,
    fps: int,
    colors: tuple[str, str],
    vmax: float | None = None,
    sequential: bool = False,
    source_resolution: int | None = None,
    note_text: str = "Saved solver frames · fixed color scale",
) -> None:
    vmax = _limit(scalar) if vmax is None else vmax
    norm = SymLogNorm(
        linthresh=0.035 * vmax,
        linscale=0.75,
        vmin=0 if sequential else -vmax,
        vmax=vmax,
        base=10,
    )
    fig, plot_axis = plt.subplots(figsize=(7.2, 6.4), facecolor=INK)
    fig.subplots_adjust(left=0.025, right=0.80, bottom=0.025, top=0.87)
    plot_axis.set_facecolor(INK)
    image = plot_axis.imshow(
        scalar[0].T,
        origin="lower",
        extent=(-half_domain, half_domain, -half_domain, half_domain),
        cmap=LinearSegmentedColormap.from_list(
            "signed_dark", [INK, colors[1]] if sequential else [colors[0], INK, colors[1]], N=257
        ),
        norm=norm,
        interpolation="bilinear",
    )
    plot_axis.set_axis_off()
    frame_title = fig.suptitle("", color=PAPER, fontsize=20, y=0.95)
    colorbar = fig.colorbar(image, ax=plot_axis, fraction=0.047, pad=0.035)
    colorbar.set_label(scalar_label, color=MUTED, fontsize=18)
    colorbar.ax.tick_params(colors=MUTED, labelsize=18, length=0, pad=7)
    colorbar.outline.set_visible(False)
    note = fig.text(
        0.5,
        0.025,
        note_text,
        color=MUTED,
        ha="center",
        fontsize=9,
    )

    def update(frame_index: int):
        saved_index = int(frame_indices[frame_index])
        sample_t = float(times[saved_index])
        image.set_data(scalar[saved_index].T)
        resolution_text = (
            f" · {source_resolution}³" if source_resolution is not None else ""
        )
        frame_title.set_text(
            f"{title}{resolution_text} · t = {sample_t:.4f}"
        )
        return image, frame_title, note

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
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
    fps: int = 8,
) -> list[Path]:
    """Write paired physical-plane GIFs and a core-frame outflow GIF."""

    data = _load(run_dir)
    times = np.asarray(data["times"])
    # Compress the unchanged rest interval, retaining every active checkpoint
    # unless an explicit frame cap requests subsampling. Never blend fields.
    active_indices = np.flatnonzero(times > float(data["rest_until"]))
    if len(active_indices) > frames - 1:
        active_indices = active_indices[
            np.linspace(0, len(active_indices) - 1, frames - 1, dtype=int)
        ]
    frame_indices = np.concatenate((np.zeros(3, dtype=int), active_indices))
    axial = np.asarray(data["velocity"])
    equatorial = np.asarray(data["equatorial_velocity"])
    resolution = int(data["resolution"])
    half_domain = float(data["half_domain"])
    dx = 2 * half_domain / resolution

    outputs = []
    similarity_outflow = _similarity_resample(
        axial[..., 2],
        times,
        half_domain=half_domain,
        base_radius=float(data["base_radius"]),
        base_height=float(data["base_height"]),
        h=float(data["h"]),
        t_star=float(data["t_star"]),
    )
    similarity_output = output_dir / f"{stem}-similarity-outflow.gif"
    _render_scalar_gif(
        times,
        frame_indices,
        similarity_outflow,
        similarity_output,
        title="Axial outflow · similarity core frame",
        scalar_label="u_z",
        half_domain=2.0,
        fps=fps,
        colors=("#58bfff", "#ffab72"),
        source_resolution=resolution,
        note_text="Fixed view in x/ℓᵣ and z/ℓ_z · saved solver frames",
    )
    outputs.append(similarity_output)

    # Plane-normal curl uses only derivatives within each saved slice.
    normal_axial = (
        np.gradient(axial[..., 0], dx, axis=2)
        - np.gradient(axial[..., 2], dx, axis=1)
    )
    measures = (
        ("speed", "Speed", "|u|", np.linalg.norm(axial, axis=-1),
         np.linalg.norm(equatorial, axis=-1), ("#58bfff", "#ffe1a1")),
        ("axial-velocity", "Vertical velocity", "u_z", axial[..., 2],
         equatorial[..., 2], ("#58bfff", "#ffab72")),
        ("normal-vorticity", "Plane-normal vorticity", "ω",
         normal_axial, _vorticity_z(equatorial, dx), ("#b49aff", "#72ead2")),
    )
    for key, title, label, vertical, horizontal, colors in measures:
        shared_limit = max(_limit(vertical), _limit(horizontal))
        for plane, field, component in (
            ("vertical", vertical, "ω_y" if key == "normal-vorticity" else label),
            ("equatorial", horizontal, "ω_z" if key == "normal-vorticity" else label),
        ):
            output = output_dir / f"{stem}-{key}-{plane}.gif"
            _render_scalar_gif(
                times, frame_indices, field, output,
                title=f"{title} · {'x–z' if plane == 'vertical' else 'x–y'}",
                scalar_label=component, half_domain=half_domain,
                fps=fps, colors=colors, vmax=shared_limit,
                sequential=key == "speed",
            )
            outputs.append(output)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", default="best")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=8)
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
