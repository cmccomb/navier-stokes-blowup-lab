"""Time-resolved 3D vector fields with fixed scales and component controls."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from .config import SimulationConfig
from .solver import SimulationResult
from .volume_series import COMPONENTS, VolumeSeries, scalar_values


def build_volume_figure(series: VolumeSeries) -> go.Figure:
    """One trace per scalar keeps component selection across time changes."""

    stride = max(1, int(np.ceil(len(series.axis) / 32)))
    axis = series.axis[::stride]
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coordinates = {"x": x.ravel(), "y": y.ravel(), "z": z.ravel()}
    vectors = series.vectors[:, ::stride, ::stride, ::stride]
    limits = {component: series.peak(component) for component in COMPONENTS}
    cone_stride = max(1, int(np.ceil(len(axis) / 9)))
    cone_axis = axis[::cone_stride]
    cx, cy, cz = np.meshgrid(cone_axis, cone_axis, cone_axis, indexing="ij")
    half = series.config.half_domain

    def traces(i: int) -> list:
        result = []
        for component in COMPONENTS:
            maximum = limits[component]
            magnitude = component == "magnitude"
            label = (
                f"|{series.symbol}|" if magnitude else f"{series.symbol}_{component}"
            )
            result.append(
                go.Isosurface(
                    **coordinates,
                    value=scalar_values(vectors[i], component).ravel(),
                    isomin=0.02 * maximum if magnitude else -0.8 * maximum,
                    isomax=0.8 * maximum,
                    cmin=0 if magnitude else -maximum,
                    cmax=maximum,
                    cauto=False,
                    surface_count=3 if magnitude else 6,
                    colorscale="Magma" if magnitude else "RdBu_r",
                    opacity=0.25,
                    caps={a: {"show": False} for a in "xyz"},
                    colorbar={"title": label, "thickness": 14, "len": 0.65},
                    name=f"{series.field} {component} isosurfaces",
                    showlegend=False,
                    hovertemplate=f"{label}=%{{value:.4g}}<extra></extra>",
                )
            )
        cones = vectors[i, ::cone_stride, ::cone_stride, ::cone_stride]
        norm = np.linalg.norm(cones, axis=-1)
        active = norm > 0.005 * limits["magnitude"]
        result.append(
            go.Cone(
                x=cx[active],
                y=cy[active],
                z=cz[active],
                u=cones[..., 0][active],
                v=cones[..., 1][active],
                w=cones[..., 2][active],
                sizemode="raw",
                sizeref=0.35 * half / limits["magnitude"],
                cmin=0,
                cmax=limits["magnitude"],
                cauto=False,
                colorscale="Viridis",
                showscale=False,
                showlegend=False,
                anchor="tail",
                name=f"{series.field} vectors",
                hovertemplate=(
                    f"{series.symbol}_x=%{{u:.4g}}<br>"
                    f"{series.symbol}_y=%{{v:.4g}}<br>"
                    f"{series.symbol}_z=%{{w:.4g}}<extra></extra>"
                ),
            )
        )
        return result

    def title(i: int) -> str:
        return (
            f"{series.title} volume · {series.config.mesh_preset}<br>"
            f"{series.config.resolution}³ · t = {series.times[i]:.6f}"
        )

    frames = [
        go.Frame(
            name=f"frame-{i}",
            data=traces(i),
            traces=list(range(5)),
            layout={"title": {"text": title(i)}},
        )
        for i in range(len(series.times))
    ]
    initial = traces(0)
    for i, trace in enumerate(initial):
        trace.visible = i in (0, 4)
    figure = go.Figure(data=initial, frames=frames)
    figure.update_layout(
        title={"text": title(0), "font": {"size": 16}, "x": 0.04, "y": 0.98},
        height=740,
        template="plotly_dark",
        paper_bgcolor="#07111f",
        margin={"l": 15, "r": 20, "t": 155, "b": 135},
        uirevision="preserve-camera",
        scene={
            **{
                f"{a}axis": {"title": a, "range": [-half, half], "autorange": False}
                for a in "xyz"
            },
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": 1.4, "z": 1.0}},
            "uirevision": "preserve-camera",
        },
        sliders=[
            {
                "currentvalue": {"prefix": "Saved time: "},
                "steps": [
                    {
                        "method": "animate",
                        "label": f"{t:.6f}",
                        "args": [
                            [f"frame-{i}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for i, t in enumerate(series.times)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "bgcolor": "#dbe9f0",
                "font": {"color": "#07111f"},
                "direction": "left",
                "x": 0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play time",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 650, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            },
            {
                "type": "dropdown",
                "bgcolor": "#dbe9f0",
                "font": {"color": "#07111f"},
                "x": 0.55,
                "y": 1.1,
                "buttons": [
                    {
                        "label": "Magnitude"
                        if c == "magnitude"
                        else f"{series.symbol}_{c}",
                        "method": "restyle",
                        "args": [{"visible": [i == j for j in range(4)] + [True]}],
                    }
                    for i, c in enumerate(COMPONENTS)
                ],
            },
        ],
        annotations=[
            {
                "text": (
                    "All three vector components · fixed scales and physical axes"
                    f"<br>{len(axis)}³ display samples · "
                    "isosurface levels fixed over time · saved checkpoints only"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.22,
                "showarrow": False,
                "font": {"size": 11},
            }
        ],
    )
    return figure


def write_series_explorer(series: VolumeSeries, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_volume_figure(series).write_html(
        destination,
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        config={"responsive": True, "displaylogo": False},
    )
    return True


def write_interactive_vector_volume(
    volumes: np.ndarray,
    times: np.ndarray,
    config: SimulationConfig,
    destination: Path,
    *,
    field: Literal["velocity", "force"],
) -> bool:
    if not len(times) or not len(volumes):
        return False
    axis = (np.arange(config.resolution) + 0.5) * config.dx - config.half_domain
    return write_series_explorer(
        VolumeSeries(volumes, times, axis, config, field), destination
    )


def write_interactive_volume(
    result: SimulationResult,
    destination: Path,
    field: Literal["auto", "velocity", "force"] = "auto",
) -> bool:
    if field == "auto":
        field = "velocity" if result.volume_velocity is not None else "force"
    volumes = result.volume_velocity if field == "velocity" else result.volume_force
    if volumes is None or not len(result.volume_times):
        return False
    return write_interactive_vector_volume(
        volumes, result.volume_times, result.config, destination, field=field
    )
