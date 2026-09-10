"""Paper-structure audit for the upgraded similarity target."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .profile import (
    cell_centers,
    cylindrical_components,
    paper_similarity_coordinates,
    paper_structure_diagnostics,
    target_velocity_components,
)

PAPER_URL = (
    "https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/"
    "navier-stokes.pdf"
)
ANNOUNCEMENT_URL = "https://openai.com/index/navier-stokes-solution/"


def _legacy_coordinate_error(t: float, cfg: SimulationConfig) -> float:
    """Residual incurred by the old separable choice q=tau in the paper map."""

    tau = cfg.t_star - t
    x, y, z = cell_centers(cfg)
    radius = np.sqrt(x * x + y * y)
    zeta = z / cfg.base_height
    q = np.full_like(zeta, tau)
    eta = zeta / q ** (0.5 - cfg.h)
    similarity_x = (radius / cfg.base_radius) ** 2 / (2 * q)
    core = (
        (np.abs(eta) < cfg.paper_eta_support)
        & (similarity_x < cfg.paper_annulus_xb)
    )
    residual = q - zeta * zeta * q ** (2 * cfg.h) - tau
    return float(np.max(np.abs(residual[core])) / tau)


def _exterior_stationarity(
    cfg: SimulationConfig, early: float, late: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    early_background, _ = target_velocity_components(early, cfg)
    late_background, _ = target_velocity_components(late, cfg)
    early_coordinates = paper_similarity_coordinates(early, cfg)
    late_coordinates = paper_similarity_coordinates(late, cfg)
    _, early_swirl, _ = cylindrical_components(early_background, cfg)
    _, late_swirl, _ = cylindrical_components(late_background, cfg)
    _, _, z = cell_centers(cfg)
    common_exterior = (
        (early_coordinates.x_similarity > cfg.paper_annulus_xb + 0.05)
        & (late_coordinates.x_similarity > cfg.paper_annulus_xb + 0.05)
        & (early_coordinates.radius < cfg.localization_inner)
        & (np.abs(z) < 1.1 * cfg.dx)
    )
    relative_change = float(
        np.linalg.norm((late_swirl - early_swirl)[common_exterior])
        / max(np.linalg.norm(early_swirl[common_exterior]), 1e-30)
    )

    radial_bins = np.linspace(
        float(np.min(early_coordinates.radius[common_exterior])),
        float(np.max(early_coordinates.radius[common_exterior])),
        12,
    )
    centers = 0.5 * (radial_bins[1:] + radial_bins[:-1])
    early_profile = np.zeros_like(centers)
    late_profile = np.zeros_like(centers)
    for index in range(len(centers)):
        shell = (
            common_exterior
            & (early_coordinates.radius >= radial_bins[index])
            & (early_coordinates.radius < radial_bins[index + 1])
        )
        early_profile[index] = np.mean(np.abs(early_swirl[shell]))
        late_profile[index] = np.mean(np.abs(late_swirl[shell]))
    valid = (early_profile > 0) & np.isfinite(early_profile)
    slope = float(
        np.polyfit(np.log(centers[valid]), np.log(early_profile[valid]), 1)[0]
    )
    return relative_change, centers, early_profile, late_profile, slope


def run_fidelity_audit(cfg: SimulationConfig, output_dir: Path) -> dict[str, object]:
    """Measure the paper-specific invariants absent from the original target."""

    cfg.validate()
    cfg = replace(cfg, profile_model="paper-surrogate")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_time = min(max(0.80, cfg.similarity_fit_start), cfg.t_end)
    exterior_early = min(cfg.similarity_fit_start, evaluation_time)
    exterior_late = min(0.85, cfg.t_end)
    if exterior_late <= exterior_early:
        exterior_early = max(cfg.t_start + 0.05, 0.45 * cfg.t_end)
        exterior_late = cfg.t_end

    structure = paper_structure_diagnostics(evaluation_time, cfg)
    legacy_error = _legacy_coordinate_error(evaluation_time, cfg)
    paper_error = float(structure["paper_coordinate_residual_linf"]) / (
        cfg.t_star - evaluation_time
    )
    stationarity, radii, early_profile, late_profile, exterior_slope = (
        _exterior_stationarity(cfg, exterior_early, exterior_late)
    )
    improvement = legacy_error / max(paper_error, np.finfo(float).eps)

    background, pulses = target_velocity_components(evaluation_time, cfg)
    total = background + pulses
    coordinates = paper_similarity_coordinates(evaluation_time, cfg)
    speed_slice = np.linalg.norm(total[:, cfg.resolution // 2, :, :], axis=-1)
    x_slice, _, z_slice = cell_centers(cfg)
    x_slice = x_slice[:, cfg.resolution // 2, :]
    z_slice = z_slice[:, cfg.resolution // 2, :]
    similarity_slice = coordinates.x_similarity[:, cfg.resolution // 2, :]
    eta_slice = coordinates.eta[:, cfg.resolution // 2, :]

    report: dict[str, object] = {
        "model": "paper-surrogate-v1",
        "evaluation_time": evaluation_time,
        "resolution": cfg.resolution,
        "coordinate_audit": {
            "legacy_separable_relative_residual_linf": legacy_error,
            "paper_coordinate_relative_residual_linf": paper_error,
            "accuracy_improvement_factor": improvement,
        },
        "exterior_audit": {
            "early_time": exterior_early,
            "late_time": exterior_late,
            "relative_stationarity_error": stationarity,
            "measured_radial_power_law": exterior_slope,
            "paper_radial_power_law": -1.0 - 2 * cfg.h,
        },
        "structure_audit": structure,
        "implemented_features": [
            "implicit q-eta-X similarity coordinates",
            "inner, annular, and stationary exterior regions",
            "zero radial-moment meridional closure",
            "nonzero axial midplane bias",
            "two complete-ring zero-mean oscillatory wave families",
            "N log(X) radial oscillation and dyadic-time pulse gates",
            "fixed physical compact-support cutoff",
            "an exact initial rest interval and C-infinity temporal cutoff",
            "discrete-curl incompressibility",
        ],
        "not_implemented": [
            "the paper's infinite pulse hierarchy",
            "profile-cone proof and exact stress matching",
            "all-order q^(2nh) corrections",
            "a proof that the residual force extends smoothly through t=1",
        ],
        "sources": [ANNOUNCEMENT_URL, PAPER_URL],
    }
    (output_dir / "paper-fidelity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    residuals = [legacy_error, max(paper_error, np.finfo(float).eps)]
    ax.bar(["old separable map", "implicit paper map"], residuals)
    ax.set_yscale("log")
    ax.set(ylabel="relative equation residual", title="Similarity-coordinate closure")
    ax.text(
        0.5,
        0.93,
        f">{improvement:.2e}x more accurate",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)

    ax = axes[0, 1]
    ax.loglog(radii, early_profile, "o-", label=f"t={exterior_early:.2f}")
    ax.loglog(radii, late_profile, "s--", label=f"t={exterior_late:.2f}")
    reference = early_profile[len(early_profile) // 2] * (
        radii / radii[len(radii) // 2]
    ) ** (-1 - 2 * cfg.h)
    ax.loglog(radii, reference, ":", label=r"paper $r^{-1-2h}$")
    ax.set(
        xlabel="fixed physical radius r",
        ylabel=r"mean $|u_\theta|$",
        title=f"Stationary exterior (change {stationarity:.2%})",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    labels = ["annulus\nenergy", "axisymmetric\nmean", "exterior\nmeridional"]
    values = [
        float(structure["pulse_annulus_energy_fraction"]),
        float(structure["pulse_axisymmetric_mean_fraction"]),
        float(structure["background_exterior_meridional_fraction"]),
    ]
    ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set(
        ylabel="fraction",
        title="Support, zero-mean, and moment audits",
        ylim=(0, max(1.0, 1.08 * max(values))),
    )
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    image = ax.pcolormesh(x_slice, z_slice, speed_slice, shading="auto", cmap="magma")
    ax.contour(
        x_slice,
        z_slice,
        similarity_slice,
        levels=[cfg.paper_annulus_xa, cfg.paper_annulus_xb],
        colors=["cyan", "white"],
        linewidths=1.0,
    )
    ax.contour(
        x_slice,
        z_slice,
        np.abs(eta_slice),
        levels=[cfg.paper_eta_support],
        colors=["lime"],
        linewidths=0.8,
        linestyles="--",
    )
    ax.set(
        xlabel="x",
        ylabel="z",
        title=f"Three-region target and pulse annulus at t={evaluation_time:.2f}",
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, label="speed")
    fig.suptitle("OpenAI Navier–Stokes paper-fidelity audit", fontsize=16)
    fig.savefig(output_dir / "paper-fidelity.png", dpi=180)
    plt.close(fig)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper-specific target structure.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--t-end", type=float, default=0.90)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/paper-fidelity")
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = SimulationConfig(
        resolution=args.resolution,
        t_end=args.t_end,
        frames=2,
        capture_volumes=False,
    )
    report = run_fidelity_audit(cfg, args.output)
    coordinate = report["coordinate_audit"]
    exterior = report["exterior_audit"]
    print(f"paper-fidelity audit complete: {args.output}")
    print(
        "coordinate accuracy improvement: "
        f"{coordinate['accuracy_improvement_factor']:.3e}x"
    )
    print(
        "exterior stationarity error: "
        f"{exterior['relative_stationarity_error']:.3%}"
    )


if __name__ == "__main__":
    main()
