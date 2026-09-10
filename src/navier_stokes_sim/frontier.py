"""Compare equal-window high-resolution frontier runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .solver import SimulationResult, load_simulation_result


@dataclass(frozen=True)
class FrontierCase:
    path: Path
    result: SimulationResult
    label: str


def _column(result: SimulationResult, name: str) -> np.ndarray:
    return np.asarray([row[name] for row in result.diagnostics], dtype=float)


def _case_label(result: SimulationResult) -> str:
    pulse_label = "pulses" if result.config.pulses_enabled else "no pulses"
    return f"{result.config.resolution}³ {pulse_label}"


def _load_cases(paths: list[Path]) -> list[FrontierCase]:
    cases = [
        FrontierCase(path, result, _case_label(result))
        for path in paths
        for result in [load_simulation_result(path)]
    ]
    cases.sort(key=lambda case: (case.result.config.pulses_enabled, case.result.config.resolution))
    return cases


def _write_summary(cases: list[FrontierCase], destination: Path) -> None:
    rows = []
    for case in cases:
        final = case.result.diagnostics[-1]
        rows.append(
            {
                "label": case.label,
                "path": str(case.path),
                "resolution": case.result.config.resolution,
                "pulses_enabled": case.result.config.pulses_enabled,
                "t_start": case.result.times[0],
                "t_end": case.result.times[-1],
                "peak_speed": final["peak_speed"],
                "target_peak_speed": final["target_peak_speed"],
                "peak_vorticity": final["peak_vorticity"],
                "tracking_relative_l2": final["tracking_relative_l2"],
                "divergence_linf": final["divergence_linf"],
                "cells_per_radial_scale": final["cells_per_radial_scale"],
                "cells_per_axial_scale": final["cells_per_axial_scale"],
                "spectral_k95": final["spectral_k95"],
                "spectral_tail_fraction": final["spectral_tail_fraction"],
                "resolved": final["resolved"],
            }
        )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _endpoint_order(cases: list[FrontierCase], diagnostic: str) -> float | None:
    pulse_cases = [case for case in cases if case.result.config.pulses_enabled]
    end_times = {round(float(case.result.times[-1]), 12) for case in pulse_cases}
    if len(pulse_cases) < 3 or len(end_times) != 1:
        return None
    spacing = np.asarray([case.result.config.dx for case in pulse_cases])
    values = np.asarray(
        [float(case.result.diagnostics[-1][diagnostic]) for case in pulse_cases]
    )
    if np.any(values <= 0):
        return None
    return float(np.polyfit(np.log(spacing), np.log(values), 1)[0])


def _plot(cases: list[FrontierCase], destination: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(cases)))

    for case, color in zip(cases, colors, strict=True):
        result = case.result
        tau = 1.0 - result.times
        line = "-" if result.config.pulses_enabled else "--"
        axes[0, 0].loglog(
            tau, _column(result, "peak_speed"), marker="o", linestyle=line,
            color=color, label=case.label,
        )
        axes[0, 1].loglog(
            tau, _column(result, "peak_vorticity"), marker="o", linestyle=line,
            color=color, label=case.label,
        )
        axes[0, 2].semilogy(
            result.times,
            np.maximum(_column(result, "tracking_relative_l2"), 1e-16),
            marker="o", linestyle=line, color=color, label=case.label,
        )
        axes[1, 0].plot(
            result.times, _column(result, "cells_per_radial_scale"),
            marker="o", linestyle=line, color=color, label=case.label,
        )
        normalized_k95 = _column(result, "spectral_k95") / (
            result.config.resolution / 3
        )
        axes[1, 1].plot(
            result.times, normalized_k95, marker="o", linestyle=line,
            color=color, label=case.label,
        )
        axes[1, 2].semilogy(
            result.times,
            np.maximum(_column(result, "spectral_tail_fraction"), 1e-16),
            marker="o", linestyle=line, color=color, label=case.label,
        )

    target_case = max(cases, key=lambda case: case.result.config.resolution)
    axes[0, 0].loglog(
        1.0 - target_case.result.times,
        _column(target_case.result, "target_peak_speed"),
        color="black", linestyle=":", linewidth=2, label="target",
    )
    axes[0, 0].invert_xaxis()
    axes[0, 1].invert_xaxis()
    axes[1, 0].axhline(4.0, color="black", linestyle=":", label="4-cell limit")
    axes[1, 1].axhline(1.0, color="black", linestyle=":", label="top third")

    tracking_order = _endpoint_order(cases, "tracking_relative_l2")
    tail_order = _endpoint_order(cases, "spectral_tail_fraction")
    tracking_title = "Target tracking"
    tail_title = "Spectral tail"
    if tracking_order is not None:
        tracking_title += f" (endpoint order {tracking_order:.2f})"
    if tail_order is not None:
        tail_title += f" (endpoint order {tail_order:.2f})"
    settings = (
        (axes[0, 0], r"time to singularity $\tau$", "peak speed", "Velocity concentration"),
        (axes[0, 1], r"time to singularity $\tau$", "peak vorticity", "Gradient amplification"),
        (axes[0, 2], "time t", "relative L2 error", tracking_title),
        (axes[1, 0], "time t", "radial scale / grid spacing", "Geometric resolution"),
        (axes[1, 1], "time t", "k95 / top-third mode", "Spectral occupancy"),
        (axes[1, 2], "time t", "top-third energy fraction", tail_title),
    )
    for axis, xlabel, ylabel, title in settings:
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("High-resolution Navier–Stokes frontier comparison", fontsize=16)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def compare_frontier(paths: list[Path], output_dir: Path) -> list[FrontierCase]:
    """Load completed runs and write a compact comparison artifact."""

    if len(paths) < 2:
        raise ValueError("frontier comparison requires at least two completed runs")
    cases = _load_cases(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(cases, output_dir / "frontier-summary.csv")
    _plot(cases, output_dir / "frontier.png")
    metadata = {
        "cases": [
            {
                "label": case.label,
                "path": str(case.path),
                "resolution": case.result.config.resolution,
                "pulses_enabled": case.result.config.pulses_enabled,
            }
            for case in cases
        ],
        "warning": (
            "Equal-window late-time runs initialize from the manufactured target; "
            "they are resolution tests, not start-from-rest histories."
        ),
        "endpoint_orders": {
            "tracking_relative_l2": _endpoint_order(
                cases, "tracking_relative_l2"
            ),
            "spectral_tail_fraction": _endpoint_order(
                cases, "spectral_tail_fraction"
            ),
        },
    }
    (output_dir / "frontier.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return cases


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="completed run directories")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/frontier-comparison")
    )
    args = parser.parse_args(argv)
    cases = compare_frontier(args.runs, args.output)
    print(f"compared {len(cases)} runs in {args.output}")


if __name__ == "__main__":
    main()
