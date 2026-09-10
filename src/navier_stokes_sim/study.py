"""Resolution and force-release study orchestration."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from .config import SimulationConfig
from .solver import SimulationResult, load_matching_checkpoint, run_simulation


def _column(result: SimulationResult, name: str, dtype=float) -> np.ndarray:
    return np.asarray([row[name] for row in result.diagnostics], dtype=dtype)


def _plot_segmented(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    resolved: np.ndarray,
    *,
    label: str,
    color,
) -> None:
    ax.plot(x, y, "o-", color=color, label=label, markersize=3)
    if np.any(~resolved):
        ax.plot(x[~resolved], y[~resolved], "x", color=color, markersize=6)


def plot_study(
    forced: dict[int, SimulationResult],
    released: SimulationResult,
    destination: Path,
) -> None:
    """Create a combined resolution and causal-control audit."""

    resolutions = sorted(forced)
    colors = plt.colormaps["viridis"](np.linspace(0.12, 0.9, len(resolutions)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ax = axes[0, 0]
    for color, resolution in zip(colors, resolutions, strict=True):
        result = forced[resolution]
        _plot_segmented(
            ax,
            _column(result, "tau")[1:],
            _column(result, "peak_speed")[1:],
            _column(result, "resolved", bool)[1:],
            label=f"N={resolution}",
            color=color,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set(
        xlabel=r"$\tau=1-t$",
        ylabel="peak speed",
        title="Resolution sweep · × is under-resolved",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for color, resolution in zip(colors, resolutions, strict=True):
        result = forced[resolution]
        cells = np.minimum(
            _column(result, "cells_per_radial_scale"),
            _column(result, "cells_per_axial_scale"),
        )
        scale = _column(result, "similarity_velocity")
        peak = _column(result, "target_peak_speed")
        mask = scale > 0
        ax.plot(
            cells[mask],
            peak[mask] / scale[mask],
            "o-",
            color=color,
            label=f"N={resolution}",
        )
    ax.axvline(4, color="black", linestyle=":", label="four-cell threshold")
    ax.set(
        xscale="log",
        xlabel="cells across smaller core scale",
        ylabel="sampled peak / similarity amplitude",
        title="Grid-sampling convergence",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[0, 2]
    last_resolved = []
    theoretical = []
    for resolution in resolutions:
        result = forced[resolution]
        mask = _column(result, "resolved", bool)
        last_resolved.append(
            float(result.times[np.flatnonzero(mask)[-1]]) if np.any(mask) else 0.0
        )
        dx = 2 * result.config.half_domain / resolution
        tau_limit = (
            result.config.min_cells_per_scale * dx / result.config.base_radius
        ) ** 2
        theoretical.append(
            max(0.0, min(result.config.t_end, result.config.t_star - tau_limit))
        )
    ax.plot(resolutions, last_resolved, "o-", label="observed saved-frame limit")
    ax.plot(resolutions, theoretical, "s--", label="radial-scale prediction")
    ax.set(
        xlabel="grid resolution N",
        ylabel="last resolved time",
        title="How refinement moves the cutoff",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    reference = forced[max(resolutions)]
    release_time = released.config.forcing_end
    ax = axes[1, 0]
    ax.plot(
        reference.times,
        _column(reference, "peak_speed"),
        "o-",
        label="forced reference",
    )
    ax.plot(
        released.times, _column(released, "peak_speed"), "s-", label="force released"
    )
    ax.plot(
        reference.times,
        _column(reference, "target_peak_speed"),
        ":",
        color="black",
        label="target",
    )
    if release_time is not None:
        ax.axvline(release_time, color="tab:red", linestyle="--", label="force off")
    ax.set(
        xlabel="time t",
        ylabel="peak speed",
        title=f"Causal test at N={max(resolutions)}",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.semilogy(
        reference.times,
        np.maximum(_column(reference, "enstrophy"), 1e-16),
        "o-",
        label="forced enstrophy",
    )
    ax.semilogy(
        released.times,
        np.maximum(_column(released, "enstrophy"), 1e-16),
        "s-",
        label="released enstrophy",
    )
    ax.semilogy(
        released.times,
        np.maximum(_column(released, "tracking_relative_l2"), 1e-16),
        "^-",
        label="released tracking error",
    )
    if release_time is not None:
        ax.axvline(release_time, color="tab:red", linestyle="--")
    ax.set(
        xlabel="time t",
        ylabel="enstrophy / relative error",
        title="Response after force release",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 2]
    ax.semilogy(
        reference.times[1:],
        _column(reference, "force_l2")[1:],
        "o-",
        label=r"$\|f\|_{L^2}$",
    )
    ax.semilogy(
        reference.times[1:],
        _column(reference, "momentum_cancellation_ratio")[1:],
        "s-",
        label="cancellation ratio",
    )
    ax.semilogy(
        reference.times[1:],
        _column(reference, "bkm_integral")[1:],
        "^-",
        label="BKM integral",
    )
    ax.set(
        xlabel="time t",
        ylabel="norm / ratio / integral",
        title="Cost of maintaining the target",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.suptitle("OpenAI Navier–Stokes proof-informed numerical study")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _write_summary(
    forced: dict[int, SimulationResult], released: SimulationResult, destination: Path
) -> None:
    rows: list[dict[str, float | int | str]] = []
    for resolution, result in sorted(forced.items()):
        final = result.diagnostics[-1]
        resolved = _column(result, "resolved", bool)
        rows.append(
            {
                "case": "forced",
                "resolution": resolution,
                "last_resolved_time": float(result.times[np.flatnonzero(resolved)[-1]])
                if np.any(resolved)
                else 0.0,
                "final_peak_speed": float(final["peak_speed"]),
                "final_peak_vorticity": float(final["peak_vorticity"]),
                "final_energy": float(final["kinetic_energy"]),
                "final_enstrophy": float(final["enstrophy"]),
                "final_tracking_error": float(final["tracking_relative_l2"]),
                "final_resolved": bool(final["resolved"]),
                "final_minimum_core_cells": min(
                    float(final["cells_per_radial_scale"]),
                    float(final["cells_per_axial_scale"]),
                ),
                "final_spectral_tail_fraction": float(final["spectral_tail_fraction"]),
            }
        )
    final = released.diagnostics[-1]
    rows.append(
        {
            "case": f"force-off-at-{released.config.forcing_end:g}",
            "resolution": released.config.resolution,
            "last_resolved_time": "",
            "final_peak_speed": float(final["peak_speed"]),
            "final_peak_vorticity": float(final["peak_vorticity"]),
            "final_energy": float(final["kinetic_energy"]),
            "final_enstrophy": float(final["enstrophy"]),
            "final_tracking_error": float(final["tracking_relative_l2"]),
            "final_resolved": bool(final["resolved"]),
            "final_minimum_core_cells": min(
                float(final["cells_per_radial_scale"]),
                float(final["cells_per_axial_scale"]),
            ),
            "final_spectral_tail_fraction": float(final["spectral_tail_fraction"]),
        }
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_study(
    base_config: SimulationConfig,
    resolutions: list[int],
    release_time: float,
    output_dir: Path,
    *,
    resume: bool = True,
) -> tuple[dict[int, SimulationResult], SimulationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forced: dict[int, SimulationResult] = {}
    for resolution in resolutions:
        config = replace(
            base_config,
            resolution=resolution,
            forcing_end=None,
            capture_volumes=False,
        )
        case_dir = output_dir / f"forced-n{resolution}"
        checkpoint = load_matching_checkpoint(case_dir, config) if resume else None
        if checkpoint is not None:
            print(f"reusing forced checkpoint at {resolution}^3", flush=True)
            forced[resolution] = checkpoint
        else:
            print(f"running forced reference at {resolution}^3", flush=True)
            forced[resolution] = run_simulation(config, case_dir)

    finest = max(resolutions)
    released_config = replace(
        base_config,
        resolution=finest,
        forcing_end=release_time,
        capture_volumes=False,
    )
    released_dir = output_dir / f"released-n{finest}"
    released = (
        load_matching_checkpoint(released_dir, released_config) if resume else None
    )
    if released is not None:
        print(f"reusing force-release checkpoint at {finest}^3", flush=True)
    else:
        print(f"running force-release case at {finest}^3", flush=True)
        released = run_simulation(released_config, released_dir)
    plot_study(forced, released, output_dir / "study.png")
    _write_summary(forced, released, output_dir / "summary.csv")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "resolutions": resolutions,
        "release_time": release_time,
        "base_config": base_config.to_dict(),
        "interpretation": (
            "Resolution refinement delays the grid cutoff; the force-release comparison tests "
            "whether the concentrating target persists when the manufactured control is removed."
        ),
    }
    (output_dir / "study.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return forced, released


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the resolution and force-release study."
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[16, 24, 32, 48, 64]
    )
    parser.add_argument("--t-end", type=float, default=0.9)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--release-time", type=float, default=0.55)
    parser.add_argument("--integrator", choices=("euler", "rk2"), default="rk2")
    parser.add_argument("--output", type=Path, default=Path("outputs/study"))
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore matching completed checkpoints and rerun every CFD case",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if len(set(args.resolutions)) != len(args.resolutions):
        raise ValueError("resolutions must be unique")
    base = SimulationConfig(
        resolution=max(args.resolutions),
        t_end=args.t_end,
        frames=args.frames,
        integrator=args.integrator,
        capture_volumes=False,
    )
    run_study(
        base,
        sorted(args.resolutions),
        args.release_time,
        args.output,
        resume=not args.no_resume,
    )
    print(f"study complete: {args.output}")


if __name__ == "__main__":
    main()
