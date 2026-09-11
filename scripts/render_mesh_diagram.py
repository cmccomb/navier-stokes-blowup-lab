"""Render the published uniform solver mesh in two and three dimensions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
INK, PAPER, MUTED, CYAN, AMBER = "#07111f", "#e9f1f5", "#9fb3c2", "#69d2e7", "#ffb35c"


def main() -> None:
    record = json.loads((ROOT / "site/data/stream.json").read_text())
    cfg = record["config"]
    if cfg["mesh_preset"] != "full-domain":
        raise ValueError("This diagram requires the uniform full-domain mesh")
    n, half = cfg["resolution"], cfg["half_domain"]
    dx = 2 * half / n
    edges = np.linspace(-half, half, n + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    middle = n // 2
    plane = centers[middle]
    local = edges[middle - 2 : middle + 3]
    lo, hi = local[0], local[-1]
    cells = centers[middle - 2 : middle + 2]
    output = ROOT / "site/media"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "solver-mesh",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.8), facecolor=INK)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.23, wspace=0.43)
    for axis, values, limits in zip(axes, (edges, local), ((-half, half), (lo, hi))):
        low, high = limits
        lines = [[(v, low), (v, high)] for v in values]
        lines += [[(low, v), (high, v)] for v in values]
        axis.add_collection(
            LineCollection(
                lines, colors="#466d85", linewidths=0.35 if len(values) > 10 else 0.8
            )
        )
        axis.set(xlim=limits, ylim=limits, aspect="equal", facecolor=INK)
        axis.set_xlabel("x", color=MUTED, fontsize=11.2)
        axis.tick_params(colors=MUTED, labelsize=11.2)
        for spine in axis.spines.values():
            spine.set_color(CYAN)
    axes[0].set_title(
        f"x–z plane · y = {plane:.4f}", color=PAPER, fontsize=12.8, pad=10
    )
    axes[0].set_ylabel("z", color=MUTED, fontsize=11.2)
    axes[0].set_xticks([-half, 0, half])
    axes[0].set_yticks([-half, 0, half])
    axes[0].add_patch(
        Rectangle(
            (lo, lo), hi - lo, hi - lo, facecolor="none", edgecolor=AMBER, linewidth=1.5
        )
    )
    axes[1].set_title(
        "x–y plane · four-cell close-up", color=PAPER, fontsize=12.8, pad=10
    )
    axes[1].set_ylabel("y", color=MUTED, fontsize=11.2)
    axes[1].set_xticks([lo, 0, hi], [f"{lo:.3f}", "0", f"{hi:.3f}"])
    axes[1].set_yticks([lo, 0, hi], [f"{lo:.3f}", "0", f"{hi:.3f}"])
    cx, cy = np.meshgrid(cells, cells)
    axes[1].scatter(cx, cy, color=AMBER, s=15, zorder=3)
    fig.text(
        0.10, 0.92, f"{n}³ uniform Cartesian mesh", color=PAPER, fontsize=16, ha="left"
    )
    fig.text(0.10, 0.085, f"Δx = Δy = Δz = {dx:.7f}", color=PAPER, fontsize=11.2)
    fig.text(0.55, 0.085, "● cell centers", color=AMBER, fontsize=11.2)
    fig.savefig(output / "mesh-planes.svg", facecolor=INK, metadata={"Date": None})
    svg_path = output / "mesh-planes.svg"
    svg = svg_path.read_text().replace(
        "font-family: 'DejaVu Sans'", "font-family: Arial, Helvetica, sans-serif"
    )
    svg_path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(fig)

    def wire(segments: list, color: str, width: float) -> go.Scatter3d:
        points = [
            point
            for start, end in segments
            for point in (start, end, (None, None, None))
        ]
        return go.Scatter3d(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            z=[p[2] for p in points],
            mode="lines",
            line={"color": color, "width": width},
            hoverinfo="skip",
            showlegend=False,
        )

    slices = []
    for fixed in range(3):
        a, b = [i for i in range(3) if i != fixed]
        for v in edges:
            for along, across in ((a, b), (b, a)):
                start = [plane, plane, plane]
                end = start.copy()
                start[across] = end[across] = float(v)
                start[along], end[along] = -half, half
                slices.append((start, end))
    boundary = []
    detail = []
    for along in range(3):
        a, b = [i for i in range(3) if i != along]
        for values, limits, segments in (
            ([-half, half], (-half, half), boundary),
            (local, (lo, hi), detail),
        ):
            for first in values:
                for second in values:
                    start = [0.0, 0.0, 0.0]
                    start[a], start[b] = float(first), float(second)
                    end = start.copy()
                    start[along], end[along] = map(float, limits)
                    segments.append((start, end))
    x, y, z = np.meshgrid(cells, cells, cells, indexing="ij")
    figure = go.Figure(
        [
            wire(slices, "#466d85", 1),
            wire(boundary, CYAN, 3),
            wire(detail, CYAN, 2),
            go.Scatter3d(
                x=x.ravel(),
                y=y.ravel(),
                z=z.ravel(),
                mode="markers",
                marker={"color": AMBER, "size": 4},
                hovertemplate="Cell center<br>x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>",
                showlegend=False,
            ),
        ]
    )
    figure.data[2].visible = figure.data[3].visible = False
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=INK,
        plot_bgcolor=INK,
        font={"family": "Arial, Helvetica, sans-serif", "size": 16, "color": PAPER},
        title={
            "text": f"{n}³ uniform Cartesian mesh",
            "font": {"size": 20},
            "x": 0.04,
            "y": 0.97,
            "yanchor": "top",
        },
        height=700,
        margin={"l": 16, "r": 16, "t": 140, "b": 110},
        scene={
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.8, "y": 2.0, "z": 1.45}},
            **{
                f"{a}axis": {
                    "title": {"text": a, "font": {"size": 14}},
                    "tickfont": {"size": 14},
                    "range": [-half, half],
                    "tickvals": [-half, 0, half],
                    "ticktext": [f"{-half:g}", "0", f"{half:g}"],
                    "backgroundcolor": INK,
                    "gridcolor": "#29445d",
                }
                for a in "xyz"
            },
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.04,
                "y": 1.04,
                "xanchor": "left",
                "yanchor": "bottom",
                "bgcolor": "#d7e4ec",
                "font": {"size": 16, "color": INK},
                "buttons": [
                    {
                        "label": "Whole domain",
                        "method": "update",
                        "args": [
                            {"visible": [True, True, False, False]},
                            {
                                **{
                                    f"scene.{a}axis.{key}": value
                                    for a in "xyz"
                                    for key, value in {
                                        "range": [-half, half],
                                        "tickvals": [-half, 0, half],
                                        "ticktext": [f"{-half:g}", "0", f"{half:g}"],
                                    }.items()
                                },
                                "annotations[0].text": f"All {n} cells per axis<br>Three central planes · periodic boundaries",
                            },
                        ],
                    },
                    {
                        "label": "Cell detail",
                        "method": "update",
                        "args": [
                            {"visible": [False, False, True, True]},
                            {
                                **{
                                    f"scene.{a}axis.{key}": value
                                    for a in "xyz"
                                    for key, value in {
                                        "range": [lo, hi],
                                        "tickvals": [lo, 0, hi],
                                        "ticktext": [f"{lo:.3f}", "0", f"{hi:.3f}"],
                                    }.items()
                                },
                                "annotations[0].text": f"4 × 4 × 4 cells · Δx = {dx:.7f}<br>Dots mark cell centers",
                            },
                        ],
                    },
                ],
            }
        ],
        annotations=[
            {
                "text": f"All {n} cells per axis<br>Three central planes · periodic boundaries",
                "x": 0.5,
                "y": -0.18,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    html = figure.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
        auto_play=False,
        div_id="solver-mesh",
        post_script="""
const meshPlot = document.getElementById('{plot_id}');
function fitMesh() {
  const width = window.innerWidth;
  const height = Math.min(700, Math.max(520, width + 180));
  meshPlot.style.height = height + 'px';
  meshPlot.parentElement.style.height = height + 'px';
  if (meshPlot.layout.height !== height || meshPlot.layout.width !== width) Plotly.relayout(meshPlot, {height, width});
}
fitMesh();
window.addEventListener('resize', fitMesh);
""",
    )
    html = html.replace(
        "<head>",
        '<head><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="../favicon.svg" type="image/svg+xml"><title>Interactive solver mesh</title><style>html,body{margin:0;background:#07111f;color:#e9f1f5;font:16px Arial,Helvetica,sans-serif}.modebar{top:50px!important}</style>',
    )
    (output / "mesh-3d.html").write_text(html)
    metadata = {
        "resolution": n,
        "half_domain": half,
        "spacing": dx,
        "cell_count": n**3,
        "central_slice_coordinate": float(plane),
        "source_commit": record["source_commit"],
        "mesh_preset": cfg["mesh_preset"],
        "grid_lines_per_axis": len(edges),
        "cell_centers_shown_in_detail": len(cells) ** 3,
    }
    (ROOT / "site/data/mesh.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
