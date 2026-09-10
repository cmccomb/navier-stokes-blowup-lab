"""Temporal-order, similarity-exponent, force, and spectral validation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from .config import SimulationConfig
from .solver import (
    SimulationResult,
    load_matching_checkpoint,
    load_simulation_result,
    run_simulation,
)


@dataclass(frozen=True)
class PowerLawFit:
    quantity: str
    expected_exponent: float | None
    measured_exponent: float
    exponent_error: float | None
    r_squared: float
    samples: int
    first_time: float
    last_time: float


def _column(result: SimulationResult, name: str, dtype=float) -> np.ndarray:
    return np.asarray([row[name] for row in result.diagnostics], dtype=dtype)


def fit_power_law(
    result: SimulationResult,
    quantity: str,
    expected_exponent: float | None,
) -> PowerLawFit:
    """Fit ``quantity = C * tau**p`` over resolved, fully active frames."""

    values = _column(result, quantity)
    tau = _column(result, "tau")
    mask = (
        (result.times >= result.config.similarity_fit_start)
        & _column(result, "resolved", bool)
        & np.isfinite(values)
        & (values > 0)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"not enough resolved samples to fit {quantity}")
    x = np.log(tau[mask])
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    times = result.times[mask]
    return PowerLawFit(
        quantity=quantity,
        expected_exponent=expected_exponent,
        measured_exponent=float(slope),
        exponent_error=None
        if expected_exponent is None
        else float(slope - expected_exponent),
        r_squared=r_squared,
        samples=int(np.count_nonzero(mask)),
        first_time=float(times[0]),
        last_time=float(times[-1]),
    )


def _load_study(
    study_dir: Path,
) -> tuple[dict[int, SimulationResult], SimulationResult]:
    forced = {
        int(path.name.removeprefix("forced-n")): load_simulation_result(path)
        for path in study_dir.glob("forced-n*")
        if path.is_dir()
    }
    if not forced:
        raise FileNotFoundError(f"no forced checkpoints found under {study_dir}")
    finest = max(forced)
    released = load_simulation_result(study_dir / f"released-n{finest}")
    return forced, released


def _temporal_study(
    output_dir: Path, resume: bool
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    steps = (0.05, 0.025, 0.0125, 0.00625, 0.003125)
    base = SimulationConfig(
        resolution=24,
        t_end=0.5,
        frames=2,
        capture_volumes=False,
    )
    for integrator in ("euler", "rk2"):
        for max_dt in steps:
            cfg = replace(base, integrator=integrator, max_dt=max_dt)
            case_dir = output_dir / f"{integrator}-dt-{max_dt:.4f}"
            result = load_matching_checkpoint(case_dir, cfg) if resume else None
            if result is None:
                print(
                    f"running {integrator} temporal check at dt={max_dt:g}", flush=True
                )
                result = run_simulation(cfg, case_dir)
            else:
                print(
                    f"reusing {integrator} temporal check at dt={max_dt:g}", flush=True
                )
            final = result.diagnostics[-1]
            rows.append(
                {
                    "integrator": integrator,
                    "max_dt": max_dt,
                    "tracking_relative_l2": float(final["tracking_relative_l2"]),
                    "divergence_linf": float(final["divergence_linf"]),
                }
            )
    orders = {}
    for integrator in ("euler", "rk2"):
        selected = [row for row in rows if row["integrator"] == integrator]
        selected = sorted(selected, key=lambda row: float(row["max_dt"]))[:3]
        dt = np.asarray([row["max_dt"] for row in selected], dtype=float)
        error = np.asarray(
            [row["tracking_relative_l2"] for row in selected], dtype=float
        )
        orders[integrator] = float(np.polyfit(np.log(dt), np.log(error), 1)[0])
    return rows, orders


def _write_temporal_csv(rows: list[dict[str, float | str]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_validation(
    forced: dict[int, SimulationResult],
    temporal_rows: list[dict[str, float | str]],
    temporal_orders: dict[str, float],
    similarity_fits: list[PowerLawFit],
    force_fits: list[PowerLawFit],
    destination: Path,
) -> None:
    finest = forced[max(forced)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    styles = {"euler": "o-", "rk2": "s-"}
    for integrator in ("euler", "rk2"):
        selected = [row for row in temporal_rows if row["integrator"] == integrator]
        dt = np.asarray([row["max_dt"] for row in selected], dtype=float)
        error = np.asarray(
            [row["tracking_relative_l2"] for row in selected], dtype=float
        )
        ax.loglog(
            dt,
            error,
            styles[integrator],
            label=f"{integrator.upper()} · fitted order {temporal_orders[integrator]:.2f}",
        )
    ax.set(
        xlabel="maximum time step",
        ylabel="final relative tracking error",
        title="Temporal convergence at 24³, t=0.5",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    labels = [
        fit.quantity.replace("peak_", "").replace("measured_", "").replace("_", " ")
        for fit in similarity_fits
    ]
    measured = np.asarray([fit.measured_exponent for fit in similarity_fits])
    expected = np.asarray(
        [fit.expected_exponent for fit in similarity_fits], dtype=float
    )
    positions = np.arange(len(labels))
    width = 0.38
    ax.barh(positions - width / 2, expected, width, label="paper scaling")
    ax.barh(positions + width / 2, measured, width, label="measured fit")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        yticks=positions,
        yticklabels=labels,
        xlabel=r"power $p$ in quantity $\propto\tau^p$",
        title=f"Resolved similarity exponents at {max(forced)}³",
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    tau = _column(finest, "tau")
    force_exponents = {fit.quantity: fit.measured_exponent for fit in force_fits}
    for key, label, style in (
        ("force_l2", r"$\|f\|_{L^2}$", "o-"),
        ("force_gradient_l2", r"$\|\nabla f\|_{L^2}$", "s-"),
        ("force_laplacian_l2", r"$\|\Delta f\|_{L^2}$", "^-"),
        ("force_time_derivative_l2", r"$\|\partial_t f\|_{L^2}$", "d-"),
    ):
        ax.loglog(
            tau[1:],
            _column(finest, key)[1:],
            style,
            label=f"{label} · p={force_exponents[key]:.2f}",
        )
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau$",
        ylabel="manufactured-force norm",
        title="Manufactured-force derivative growth",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    for resolution, result in sorted(forced.items()):
        cells = np.minimum(
            _column(result, "cells_per_radial_scale"),
            _column(result, "cells_per_axial_scale"),
        )
        ax.semilogy(
            cells[1:],
            np.maximum(_column(result, "spectral_tail_fraction")[1:], 1e-16),
            "o-",
            label=f"N={resolution}",
        )
    ax.axvline(4, color="black", linestyle=":", label="four-cell cutoff")
    ax.set(
        xscale="log",
        xlabel="cells across smaller core scale",
        ylabel="energy fraction in top Fourier third",
        title="Independent spectral-resolution audit",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.suptitle("Numerical validation of the proof-informed flow")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def run_validation(
    study_dir: Path,
    output_dir: Path,
    *,
    resume: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forced, released = _load_study(study_dir)
    finest = forced[max(forced)]
    h = finest.config.h
    expected = {
        "peak_radial_speed": -0.5,
        "peak_azimuthal_speed": -0.5 - h,
        "peak_axial_speed": -0.5 - h,
        "measured_radial_width": 0.5,
        "measured_axial_width": 0.5 - h,
        "target_kinetic_energy": 0.5 - 3 * h,
        "peak_vorticity": -1.0 - h,
    }
    similarity_fits = [
        fit_power_law(finest, quantity, exponent)
        for quantity, exponent in expected.items()
    ]
    force_fits = [
        fit_power_law(finest, quantity, None)
        for quantity in (
            "force_l2",
            "force_gradient_l2",
            "force_laplacian_l2",
            "force_time_derivative_l2",
        )
    ]
    temporal_rows, temporal_orders = _temporal_study(
        output_dir / "temporal-cases", resume
    )
    _write_temporal_csv(temporal_rows, output_dir / "temporal-summary.csv")
    _plot_validation(
        forced,
        temporal_rows,
        temporal_orders,
        similarity_fits,
        force_fits,
        output_dir / "validation.png",
    )
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "study_dir": str(study_dir),
        "finest_resolution": max(forced),
        "resolved_fit_interval": [
            similarity_fits[0].first_time,
            similarity_fits[0].last_time,
        ],
        "similarity_fits": [asdict(fit) for fit in similarity_fits],
        "force_norm_fits": [asdict(fit) for fit in force_fits],
        "temporal_orders": temporal_orders,
        "temporal_cases": temporal_rows,
        "released_final_tracking_error": float(
            released.diagnostics[-1]["tracking_relative_l2"]
        ),
        "interpretation": (
            "Negative force-norm exponents indicate that this simple manufactured residual "
            "grows toward the singular time. The exact theorem avoids this through its annular "
            "pulse hierarchy and all-order corrections."
        ),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate temporal order, similarity exponents, force regularity, and spectrum."
    )
    parser.add_argument("--study", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = run_validation(args.study, args.output, resume=not args.no_resume)
    orders = report["temporal_orders"]
    print(
        f"validation complete: Euler order={orders['euler']:.3f}, "
        f"RK2 order={orders['rk2']:.3f}; {args.output}"
    )


if __name__ == "__main__":
    main()
