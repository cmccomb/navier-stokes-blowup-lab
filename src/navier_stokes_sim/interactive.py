"""Interactive three-dimensional rendering of captured velocity fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .solver import SimulationResult


def _frame_traces(
    velocity: np.ndarray, axis: np.ndarray
) -> tuple[go.Isosurface, go.Cone]:
    resolution = velocity.shape[0]
    volume_stride = max(1, resolution // 32)
    vector_stride = max(1, resolution // 10)

    sampled_axis = axis[::volume_stride]
    x, y, z = np.meshgrid(sampled_axis, sampled_axis, sampled_axis, indexing="ij")
    sampled_velocity = velocity[::volume_stride, ::volume_stride, ::volume_stride]
    speed = np.linalg.norm(sampled_velocity, axis=-1)
    maximum = max(float(np.max(speed)), 1e-12)
    surface = go.Isosurface(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        value=speed.ravel(),
        isomin=0.16 * maximum,
        isomax=0.92 * maximum,
        surface_count=4,
        colorscale="Magma",
        opacity=0.34,
        caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        colorbar={"title": "|u|"},
        name="speed isosurfaces",
        showscale=True,
    )

    vector_axis = axis[::vector_stride]
    vx, vy, vz = np.meshgrid(vector_axis, vector_axis, vector_axis, indexing="ij")
    vectors = velocity[::vector_stride, ::vector_stride, ::vector_stride]
    vector_speed = np.linalg.norm(vectors, axis=-1)
    active = vector_speed > 0.08 * maximum
    cone = go.Cone(
        x=vx[active],
        y=vy[active],
        z=vz[active],
        u=vectors[..., 0][active],
        v=vectors[..., 1][active],
        w=vectors[..., 2][active],
        colorscale="Viridis",
        sizemode="absolute",
        sizeref=0.13,
        showscale=False,
        name="velocity vectors",
        hovertemplate="u=%{u:.3f}<br>v=%{v:.3f}<br>w=%{w:.3f}<extra></extra>",
    )
    return surface, cone


def write_interactive_volume(result: SimulationResult, destination: Path) -> None:
    """Write a self-contained Plotly field explorer with a time slider."""

    if result.volume_velocity is None or not len(result.volume_times):
        return
    axis = (
        np.linspace(
            -result.config.half_domain,
            result.config.half_domain,
            result.config.resolution,
            endpoint=False,
        )
        + result.config.dx / 2
    )
    plotly_frames: list[go.Frame] = []
    labels: list[str] = []
    for t, velocity in zip(result.volume_times, result.volume_velocity, strict=True):
        traces = _frame_traces(velocity, axis)
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
        title=f"OpenAI Navier–Stokes {result.config.profile_model} explorer",
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
                    "Isosurfaces show speed; cones show velocity direction. "
                    "Annular waves are finite-resolution surrogates, not the exact pulse hierarchy."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.01,
                "showarrow": False,
            }
        ],
    )
    figure.write_html(
        destination,
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        config={"responsive": True, "displaylogo": False},
    )
