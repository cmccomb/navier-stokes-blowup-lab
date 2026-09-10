"""Interactive three-dimensional rendering of captured vector fields."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from .config import SimulationConfig
from .solver import SimulationResult

VectorField = Literal["velocity", "force"]


def _field_labels(field: VectorField) -> tuple[str, str, str]:
    if field == "velocity":
        return "Velocity", "|u|", "u"
    return "Applied force", "|f|", "f"


def _frame_traces(
    vectors: np.ndarray, axis: np.ndarray, field: VectorField = "velocity"
) -> tuple[go.Isosurface, go.Cone]:
    resolution = vectors.shape[0]
    volume_stride = max(1, resolution // 32)
    vector_stride = max(1, resolution // 10)
    _, magnitude_label, symbol = _field_labels(field)

    sampled_axis = axis[::volume_stride]
    x, y, z = np.meshgrid(sampled_axis, sampled_axis, sampled_axis, indexing="ij")
    sampled_vectors = vectors[::volume_stride, ::volume_stride, ::volume_stride]
    magnitude = np.linalg.norm(sampled_vectors, axis=-1)
    maximum = max(float(np.max(magnitude)), 1e-12)
    surface = go.Isosurface(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        value=magnitude.ravel(),
        isomin=0.16 * maximum,
        isomax=0.92 * maximum,
        surface_count=4,
        colorscale="Magma",
        opacity=0.34,
        caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        colorbar={"title": magnitude_label},
        name=f"{field} magnitude isosurfaces",
        showscale=True,
    )

    vector_axis = axis[::vector_stride]
    vx, vy, vz = np.meshgrid(vector_axis, vector_axis, vector_axis, indexing="ij")
    sampled_cones = vectors[::vector_stride, ::vector_stride, ::vector_stride]
    vector_magnitude = np.linalg.norm(sampled_cones, axis=-1)
    active = vector_magnitude > 0.08 * maximum
    cone = go.Cone(
        x=vx[active],
        y=vy[active],
        z=vz[active],
        u=sampled_cones[..., 0][active],
        v=sampled_cones[..., 1][active],
        w=sampled_cones[..., 2][active],
        colorscale="Viridis",
        sizemode="absolute",
        sizeref=0.13,
        showscale=False,
        name=f"{field} vectors",
        hovertemplate=(
            f"{symbol}_x=%{{u:.3g}}<br>{symbol}_y=%{{v:.3g}}"
            f"<br>{symbol}_z=%{{w:.3g}}<extra></extra>"
        ),
    )
    return surface, cone


def write_interactive_volume(
    result: SimulationResult,
    destination: Path,
    field: Literal["auto", "velocity", "force"] = "auto",
) -> bool:
    """Write a self-contained Plotly vector-field explorer with a time slider.

    ``auto`` preserves the original velocity-first behavior but falls back to the
    captured manufactured force.  The boolean return value lets callers skip
    links cleanly when the requested volume was not captured.
    """

    selected: VectorField
    if field == "auto":
        selected = "velocity" if result.volume_velocity is not None else "force"
    else:
        selected = field
    volumes = (
        result.volume_velocity if selected == "velocity" else result.volume_force
    )
    if volumes is None or not len(result.volume_times):
        return False
    return write_interactive_vector_volume(
        volumes,
        result.volume_times,
        result.config,
        destination,
        field=selected,
    )


def write_interactive_vector_volume(
    volumes: np.ndarray,
    times: np.ndarray,
    config: SimulationConfig,
    destination: Path,
    *,
    field: VectorField,
) -> bool:
    """Write an explorer directly from captured vector-volume arrays."""

    if not len(times) or not len(volumes):
        return False
    if len(times) != len(volumes):
        raise ValueError("volume times and vector volumes must have equal length")
    title, magnitude_label, _ = _field_labels(field)
    axis = (
        np.linspace(
            -config.half_domain,
            config.half_domain,
            config.resolution,
            endpoint=False,
        )
        + config.dx / 2
    )
    plotly_frames: list[go.Frame] = []
    labels: list[str] = []
    for t, vectors in zip(times, volumes, strict=True):
        traces = _frame_traces(vectors, axis, field)
        label = f"t={t:.3f}"
        labels.append(label)
        plotly_frames.append(go.Frame(name=label, data=list(traces)))

    figure = go.Figure(data=plotly_frames[0].data, frames=plotly_frames)
    steps = [
        {
            "method": "animate",
            "label": label,
            "args": [
                [label],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for label in labels
    ]
    figure.update_layout(
        title=(
            f"{title} volume · {config.resolution}³ · "
            f"{config.mesh_preset}"
        ),
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 55, "b": 10},
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 1.15}},
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Captured time: "},
                "pad": {"t": 35},
                "steps": steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Play",
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
                            },
                        ],
                    },
                ],
            }
        ],
        annotations=[
            {
                "text": (
                    f"Isosurfaces show {magnitude_label}; cones show {field} direction. "
                    "Captured solver checkpoints only; fixed physical coordinates."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.01,
                "showarrow": False,
            }
        ],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        destination,
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        config={"responsive": True, "displaylogo": False},
    )
    return True
