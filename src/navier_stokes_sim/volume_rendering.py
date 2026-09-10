"""Static three-dimensional views of captured vector volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

from .config import SimulationConfig

VectorField = Literal["velocity", "force"]

INK = "#07111f"
PAPER = "#e9f1f5"
MUTED = "#9fb5c5"
GRID = "#29445d"
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "signed_dark", ["#58bfff", INK, "#ffab72"], N=257
)


def _similarity_points(
    vectors: np.ndarray,
    time: float,
    config: SimulationConfig,
    *,
    extent: float,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strong voxels in similarity coordinates and their z component."""

    stride = max(1, int(np.ceil(config.resolution / 64)))
    sampled = vectors[::stride, ::stride, ::stride]
    axis = (
        np.linspace(
            -config.half_domain,
            config.half_domain,
            config.resolution,
            endpoint=False,
        )
        + config.dx / 2
    )[::stride]
    tau = max(config.t_star - float(time), np.finfo(float).tiny)
    radial_length = config.base_radius * tau**0.5
    axial_length = config.base_height * tau ** (0.5 - config.h)
    x, y, z = np.meshgrid(
        axis / radial_length,
        axis / radial_length,
        axis / axial_length,
        indexing="ij",
    )
    component = sampled[..., 2]
    magnitude = np.linalg.norm(sampled, axis=-1)
    inside = (np.abs(x) <= extent) & (np.abs(y) <= extent) & (np.abs(z) <= extent)
    nonzero = magnitude[inside & (magnitude > 0)]
    if not nonzero.size:
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    threshold = float(np.quantile(nonzero, 0.975))
    selected = inside & (magnitude >= threshold)
    points = np.column_stack((x[selected], y[selected], z[selected]))
    values = component[selected]
    strengths = magnitude[selected]
    if len(points) > maximum_points:
        keep = np.argpartition(strengths, -maximum_points)[-maximum_points:]
        points = points[keep]
        values = values[keep]
        strengths = strengths[keep]
    return points, values, strengths


def write_orbit_gif(
    vectors: np.ndarray,
    time: float,
    config: SimulationConfig,
    destination: Path,
    *,
    field: VectorField,
    frames: int = 36,
    fps: int = 12,
    extent: float = 2.0,
    maximum_points: int = 18_000,
) -> bool:
    """Render an orbiting strong-voxel view of one 3D checkpoint."""

    points, values, strengths = _similarity_points(
        vectors,
        time,
        config,
        extent=extent,
        maximum_points=maximum_points,
    )
    if not len(points):
        return False
    vmax = max(float(np.quantile(np.abs(values), 0.995)), np.finfo(float).eps)
    norm = SymLogNorm(
        linthresh=0.035 * vmax,
        linscale=0.75,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )
    scaled = strengths / max(float(np.max(strengths)), np.finfo(float).eps)
    sizes = 2.0 + 10.0 * np.sqrt(scaled)

    fig = plt.figure(figsize=(7.2, 6.8), facecolor=INK)
    axis = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.89)
    axis.set_facecolor(INK)
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap=SIGNED_CMAP,
        norm=norm,
        s=sizes,
        alpha=0.58,
        depthshade=False,
        linewidths=0,
    )
    axis.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        zlim=(-extent, extent),
        xlabel=r"$x/\ell_r$",
        ylabel=r"$y/\ell_r$",
        zlabel=r"$z/\ell_z$",
    )
    axis.set_box_aspect((1, 1, 1))
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    for item in (axis.xaxis, axis.yaxis, axis.zaxis):
        item.label.set_color(MUTED)
        item.pane.set_facecolor(INK)
        item.pane.set_edgecolor(GRID)
        item._axinfo["grid"]["color"] = GRID
        item._axinfo["grid"]["linewidth"] = 0.4
    symbol = "u" if field == "velocity" else "f"
    label = "Velocity" if field == "velocity" else "Applied force"
    fig.suptitle(
        f"{label} volume · {config.mesh_preset} · "
        f"{config.resolution}³ · t = {time:.4f}",
        color=PAPER,
        fontsize=16,
        y=0.95,
    )
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label(rf"${symbol}_z$", color=MUTED, fontsize=13)
    colorbar.ax.tick_params(colors=MUTED, labelsize=10, length=0)
    colorbar.outline.set_visible(False)
    fig.text(
        0.5,
        0.025,
        "Strongest 2.5% of nonzero voxels · fixed similarity-coordinate cube",
        ha="center",
        color=MUTED,
        fontsize=9,
    )

    def update(frame: int):
        axis.view_init(elev=24, azim=25 + 360 * frame / frames)
        return (scatter,)

    destination.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )
    animation.save(destination, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    return True
