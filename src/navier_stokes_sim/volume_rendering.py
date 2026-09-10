"""Static three-dimensional views of captured vector volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .config import SimulationConfig
from .volume_series import Component, VolumeSeries, scalar_values

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


def write_time_animation(
    series: VolumeSeries,
    destination: Path,
    *,
    component: Component = "magnitude",
    orbit: bool = True,
    fps: int = 12,
    holds: int = 6,
    maximum_points: int = 7_000,
    vmax: float | None = None,
    vector_max: float | None = None,
    style: Literal["isosurface", "cloud"] = "isosurface",
) -> None:
    """Animate measured states; camera motion never interpolates fluid fields.

    Axes, scalar limits, opacity, voxel selection and vector length scaling are
    fixed over the sequence. Vectors show all three physical components.
    """

    if fps < 1 or holds < 1 or maximum_points < 1:
        raise ValueError("fps, holds and maximum_points must be positive")
    if destination.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("animation destination must end in .gif or .mp4")
    if style not in {"isosurface", "cloud"}:
        raise ValueError("style must be isosurface or cloud")
    peak = series.peak(component) if vmax is None else float(vmax)
    vector_peak = series.peak("magnitude") if vector_max is None else float(vector_max)
    if not np.isfinite([peak, vector_peak]).all() or min(peak, vector_peak) <= 0:
        raise ValueError("color and vector limits must be finite and positive")
    magnitude = component == "magnitude"
    surface_levels = peak * np.asarray(
        [0.02, 0.10, 0.40] if magnitude else [-0.4, -0.1, 0.1, 0.4]
    )
    meshes = []
    spacing = float(series.axis[1] - series.axis[0])
    if not np.allclose(np.diff(series.axis), spacing):
        raise ValueError("isosurfaces require uniformly spaced sample coordinates")
    if style == "isosurface":
        for frame in series.vectors:
            scalar = scalar_values(frame, component)
            frame_meshes = []
            for level in surface_levels:
                if scalar.min() < level < scalar.max():
                    vertices, faces, _, _ = marching_cubes(
                        scalar,
                        level=float(level),
                        spacing=(spacing,) * 3,
                        step_size=2,
                        allow_degenerate=False,
                    )
                    vertices += series.axis[0]
                    frame_meshes.append((vertices[faces], float(level)))
            meshes.append(frame_meshes)
    norm = SymLogNorm(
        linthresh=0.0001 * peak,
        linscale=0.7,
        vmin=0 if magnitude else -peak,
        vmax=peak,
        base=10,
    )
    cmap = plt.get_cmap("magma") if magnitude else SIGNED_CMAP
    x, y, z = np.meshgrid(series.axis, series.axis, series.axis, indexing="ij")
    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    occupancy = np.zeros(x.size, dtype=bool)
    for frame in series.vectors:
        occupancy |= np.linalg.norm(frame, axis=-1).ravel() >= 0.0001 * vector_peak
    indices = np.flatnonzero(occupancy)
    if len(indices) > maximum_points:
        indices = np.sort(
            np.random.default_rng(0).choice(indices, maximum_points, replace=False)
        )
    points = positions[indices]
    vector_stride = max(1, int(np.ceil(len(series.axis) / 9)))
    selection = (slice(None, None, vector_stride),) * 3
    vector_positions = np.column_stack(
        (x[selection].ravel(), y[selection].ravel(), z[selection].ravel())
    )
    fig = plt.figure(figsize=(8, 7.2), facecolor=INK)
    axis = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.09, top=0.84)
    axis.set_facecolor(INK)
    half = series.config.half_domain
    axis.set(
        xlim=(-half, half),
        ylim=(-half, half),
        zlim=(-half, half),
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    axis.set_box_aspect((1, 1, 1))
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    for setter in (axis.set_xticks, axis.set_yticks, axis.set_zticks):
        setter([-half, 0, half])
    for item in (axis.xaxis, axis.yaxis, axis.zaxis):
        item.label.set_color(MUTED)
        item.pane.fill = False
        item.pane.set_edgecolor(GRID)
        item._axinfo["grid"]["color"] = GRID
    scatter = axis.scatter(
        *points.T,
        s=7,
        c=np.zeros(len(points)),
        cmap=cmap,
        norm=norm,
        depthshade=False,
        linewidths=0,
    )
    scatter.set_visible(style == "cloud")
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.035, pad=0.025)
    scalar_label = f"|{series.symbol}|" if magnitude else f"{series.symbol}_{component}"
    colorbar.set_label(scalar_label, color=MUTED, fontsize=13)
    colorbar.ax.tick_params(colors=MUTED, labelsize=10, length=0)
    colorbar.outline.set_visible(False)
    fig.suptitle(
        f"{series.title} · {scalar_label} + 3D directions\n"
        f"{series.config.mesh_preset} · {series.config.resolution}³ source grid",
        color=PAPER,
        fontsize=16,
        y=0.97,
    )
    time_label = fig.text(0.5, 0.87, "", color=PAPER, ha="center", fontsize=13)
    representation = (
        "fixed isosurfaces at 2%, 10%, 40% of sequence peak"
        if magnitude
        else "fixed isosurfaces at ±10%, ±40% of sequence peak"
    )
    fig.text(
        0.5,
        0.035,
        "Saved states · fixed physical axes and color scale · arrows use x, y, z\n"
        + (representation if style == "isosurface" else "fixed voxel sample")
        + f"\n{len(series.axis)}³ input samples · equal checkpoint holds; rest compressed",
        color=MUTED,
        ha="center",
        fontsize=9,
    )
    nonzero = [i for i, v in enumerate(series.vectors) if np.any(v)]
    saved = ([0] if not nonzero or nonzero[0] != 0 else []) + nonzero
    if not saved:
        saved = [0]
    frame_indices = np.repeat(saved, holds)
    quiver = None
    surface_artists = []
    last_saved_index = None

    def update(animation_frame: int):
        nonlocal quiver, surface_artists, last_saved_index
        i = int(frame_indices[animation_frame])
        v = series.vectors[i]
        values = scalar_values(v, component).ravel()[indices]
        strengths = np.linalg.norm(v.reshape(-1, 3)[indices], axis=-1)
        rgba = cmap(norm(values))
        rgba[:, 3] = np.where(strengths >= 0.0001 * vector_peak, 0.30, 0.0)
        # Explicit RGBA is needed so exact rest remains genuinely empty.
        scatter.set_array(None)
        scatter.set_facecolors(rgba)
        scatter.set_edgecolors(rgba)
        if style == "isosurface" and i != last_saved_index:
            for artist in surface_artists:
                artist.remove()
            surface_artists = []
            for triangles, level in meshes[i]:
                artist = Poly3DCollection(
                    triangles,
                    facecolors=cmap(norm(level)),
                    linewidths=0,
                    alpha=0.14 + 0.7 * abs(level) / peak,
                    shade=True,
                )
                axis.add_collection3d(artist)
                surface_artists.append(artist)
            last_saved_index = i
        if quiver is not None:
            quiver.remove()
        arrows = v[selection].reshape(-1, 3)
        active = np.linalg.norm(arrows, axis=-1) > 0.002 * vector_peak
        quiver = axis.quiver(
            *vector_positions[active].T,
            *arrows[active].T,
            length=0.38 * half / vector_peak,
            normalize=False,
            color="#b8e5f2",
            linewidth=0.65,
            alpha=0.9,
            arrow_length_ratio=0.35,
        )
        axis.view_init(
            elev=24,
            azim=(-55 + 180 * animation_frame / max(len(frame_indices) - 1, 1))
            if orbit
            else -55,
        )
        time_label.set_text(
            f"t = {series.times[i]:.6f}  ·  checkpoint {i + 1}/{len(series.times)}"
        )
        return scatter, quiver, time_label

    destination.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )
    writer = (
        FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        if destination.suffix.lower() == ".mp4"
        else PillowWriter(fps=fps)
    )
    animation.save(destination, writer=writer, dpi=100)
    plt.close(fig)
