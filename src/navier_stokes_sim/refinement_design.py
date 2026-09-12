"""Similarity extrapolation and sampled 3D phase coverage; not an accuracy certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil, expm1, isfinite, log, log2, pi, sqrt
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .time_stepping import forcing_log_rate_bound


def extrapolate(
    reference_t: float, reference_speed: float, multiplier: float, cfg: SimulationConfig
) -> dict:
    """Anchor the implemented leading similarity exponent to a measured peak.

    This is a planning model, not a fitted or independently verified peak law.
    Keep tau explicitly: subtracting it from t_star loses precision near t_star.
    """
    cfg.validate()
    if (
        not (0 <= reference_t < cfg.t_star)
        or any(not isfinite(v) or v <= 0 for v in (reference_speed, multiplier))
        or multiplier < 1
    ):
        raise ValueError(
            "require a finite positive peak, multiplier >= 1, and t < t_star"
        )
    tau_ref = cfg.t_star - reference_t
    tau = tau_ref * multiplier ** (-1 / (0.5 + cfg.h))
    if tau <= 0 or cfg.t_star - tau >= cfg.t_star:
        raise ValueError("requested endpoint cannot be represented safely")
    alpha = cfg.pulse_radial_frequency * (1 + 0.65 * (cfg.pulse_hierarchy_levels - 1))
    midpoint = (cfg.paper_annulus_xa + cfg.paper_annulus_xb) / 2
    inner_half = midpoint - (cfg.paper_annulus_xb - cfg.paper_annulus_xa) / 2 * sqrt(
        1 - 1 / (1 - log(0.5))
    )
    wavelength = pi * cfg.base_radius * sqrt(2 * inner_half * tau) / alpha
    # Eight points across the doubled radial phase, at the inner half-height bump.
    dx = wavelength / 16
    rate = forcing_log_rate_bound(cfg)
    phase_dt = (
        -tau * expm1(-cfg.forcing_phase_step / rate)
        if rate and cfg.forcing_phase_step
        else None
    )
    return {
        "speed_multiplier": multiplier,
        "projected_peak_model_units": reference_speed * multiplier,
        "t": cfg.t_star - tau,
        "tau": tau,
        "radial_scale": cfg.base_radius * sqrt(tau),
        "axial_scale": cfg.base_height * tau ** (0.5 - cfg.h),
        "radial_shrink_factor": sqrt(tau_ref / tau),
        "inner_half_bump_velocity_wavelength": wavelength,
        "radial_design_dx": dx,
        "radial_design_equivalent_n": 2 * ceil(cfg.half_domain / dx),
        "forcing_phase_dt_limit": phase_dt,
        "force_difference_half_window_preserving_epsilon_over_tau": cfg.derivative_epsilon
        * tau
        / tau_ref,
        "additional_phase_only_steps": ceil(
            rate / cfg.forcing_phase_step * log(tau_ref / tau)
        )
        if rate and cfg.forcing_phase_step
        else None,
    }


def phase_geometry(
    t: float,
    similarity_x: np.ndarray,
    eta: np.ndarray,
    theta: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates and maximum |grad phase| of the retained potential waves.

    q=tau/(1-eta²), r=R sqrt(2Xq), z=H eta q^D.
    q_z/q=2eta/(H q^D L); eta_z=(1-eta²)/(H q^D L), L=1-2h eta².
    Norm includes radial, angular, and axial derivatives, not just radial waves.
    Temporal gates are deliberately ignored, so all retained families are audited.
    """
    if not 0 <= t < cfg.t_star:
        raise ValueError("phase geometry requires 0 <= t < t_star")
    X, e, angle = np.broadcast_arrays(similarity_x, eta, theta)
    if (
        not (np.isfinite(X).all() and np.isfinite(e).all() and np.isfinite(angle).all())
        or np.any(X <= 0)
        or np.any(np.abs(e) >= 1)
    ):
        raise ValueError("require finite X>0 and |eta|<1")
    q = (cfg.t_star - t) / (1 - e * e)
    d = 0.5 - cfg.h
    radius = cfg.base_radius * np.sqrt(2 * X * q)
    z = cfg.base_height * e * q**d
    xyz = np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)
    denominator = cfg.base_height * q**d * (1 - 2 * cfg.h * e * e)
    qz_over_q = 2 * e / denominator
    eta_z = (1 - e * e) / denominator
    maximum = np.zeros_like(radius)
    for level in range(cfg.pulse_hierarchy_levels):
        scale = 1 + 0.65 * level
        alpha = scale * cfg.pulse_radial_frequency
        mode = cfg.pulse_azimuthal_mode + level * cfg.pulse_mode_stride
        for family in (1, 2):
            if (family == 1 and cfg.pulse_rtheta_strength == 0) or (
                family == 2 and cfg.pulse_rz_strength == 0
            ):
                continue
            m = mode + family - 1
            beta = 0 if family == 1 else scale * cfg.pulse_axial_frequency
            kz = -alpha * qz_over_q + beta * eta_z
            magnitude = np.sqrt((2 * alpha / radius) ** 2 + (m / radius) ** 2 + kz * kz)
            maximum = np.maximum(maximum, magnitude)
    return xyz, maximum


def mesh_cells(base_n: int, levels: int, patch_fraction: float) -> tuple[int, int]:
    """First refined cube side / domain side; higher cubes halve their side length."""
    refined_n = round(2 * patch_fraction * base_n)
    if (
        type(base_n) is not int
        or base_n < 16
        or base_n % 8
        or type(levels) is not int
        or not 1 <= levels <= 12
        or not 0.5 <= patch_fraction < 1
        or abs(refined_n - 2 * patch_fraction * base_n) > 1e-10
        or refined_n % 8
    ):
        raise ValueError("require nested aligned cubes, N multiple of 8, 1..12 levels")
    stored = base_n**3 + (levels - 1) * refined_n**3
    active = stored - (levels - 1) * (refined_n // 2) ** 3
    return stored, active


def sampled_coverage(
    t_end: float,
    cfg: SimulationConfig,
    candidates: list[tuple[int, int, float]],
    time_samples: int = 257,
    spatial_samples: int = 25,
) -> list[dict]:
    """Sample the entire nonzero pulse support; reject sparse/invalid sampling.

    Samples are not a proof of a global minimum. No amplitude/gate threshold is
    used; only points outside the exact fixed localization support are excluded.
    Two fine-cell widths of interface clearance are required to credit fine spacing.
    The doubled-phase rule bounds sums of the retained phase gradients, but does
    not bound envelopes, basis variation, corrections, background or solution error.
    """
    cfg.validate()
    if (
        not cfg.paper_time_cutoff_start < t_end < cfg.t_star
        or time_samples < 3
        or spatial_samples < 5
    ):
        raise ValueError(
            "require active endpoint and at least 3 time / 5 spatial samples"
        )
    if (
        not cfg.pulses_enabled
        or max(cfg.pulse_rtheta_strength, cfg.pulse_rz_strength) <= 0
    ):
        raise ValueError("phase audit requires enabled retained pulses")
    # Midpoint samples stay strictly inside the open support, even in thin tails.
    unit = (np.arange(spatial_samples) + 0.5) / spatial_samples
    X, eta, theta = np.meshgrid(
        cfg.paper_annulus_xa + unit * (cfg.paper_annulus_xb - cfg.paper_annulus_xa),
        (2 * unit - 1) * cfg.paper_eta_support,
        np.linspace(0, pi / 4, 9),
        indexing="ij",
    )
    X, eta, theta = (a.ravel() for a in (X, eta, theta))
    times = cfg.t_star - np.geomspace(
        cfg.t_star - cfg.paper_time_cutoff_start, cfg.t_star - t_end, time_samples
    )
    rows = []
    for n, levels, fraction in candidates:
        stored, active = mesh_cells(n, levels, fraction)
        rows.append(
            {
                "base_n": n,
                "levels": levels,
                "patch_fraction": fraction,
                "finest_effective_n": n * 2 ** (levels - 1),
                "stored_cells": stored,
                "active_cells": active,
                "minimum_sampled_doubled_phase_points": float("inf"),
                "worst_sample": None,
            }
        )
    retained_samples = 0
    for t in times:
        xyz, wave = phase_geometry(float(t), X, eta, theta, cfg)
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        keep = (radius < cfg.localization_outer) & (
            np.abs(xyz[:, 2]) < cfg.localization_outer
        )
        retained_samples += int(keep.sum())
        xyz, wave = xyz[keep], wave[keep]
        max_coord = np.max(np.abs(xyz), axis=1)
        for row in rows:
            n, levels, fraction = row["base_n"], row["levels"], row["patch_fraction"]
            selected = np.zeros(len(wave), dtype=int)
            for lev in range(1, levels):
                halfwidth = cfg.half_domain * fraction / 2 ** (lev - 1)
                dx = 2 * cfg.half_domain / (n * 2**lev)
                selected[max_coord + 2 * dx < halfwidth] = lev
            dx = 2 * cfg.half_domain / (n * 2**selected)
            points = pi / (dx * wave)  # 2 pi / (dx * 2 |grad phase|)
            index = int(np.argmin(points))
            if points[index] < row["minimum_sampled_doubled_phase_points"]:
                row["minimum_sampled_doubled_phase_points"] = float(points[index])
                row["worst_sample"] = {
                    "t": float(t),
                    "xyz": xyz[index].tolist(),
                    "level": int(selected[index]),
                    "dx": float(dx[index]),
                    "phase_gradient_norm": float(wave[index]),
                }
    for row in rows:
        row["sampled_phase_screen_passed"] = (
            row["minimum_sampled_doubled_phase_points"] >= 8
        )
        row["production_ready"] = False
        row["time_samples"] = time_samples
        row["spatial_samples_per_X_eta"] = spatial_samples
        row["retained_sample_count"] = retained_samples
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="JSON containing config and final diagnostics",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.reference.read_text())
    cfg = SimulationConfig(**source["config"])
    diag = source["diagnostics"]
    forecasts = [
        extrapolate(diag["t"], diag["peak_speed"], factor, cfg)
        for factor in (1, 10, 100)
    ]
    rows = sampled_coverage(
        forecasts[1]["t"],
        cfg,
        [
            (128, 4, 0.5),
            (128, 8, 0.5),
            (128, 8, 0.75),
            (128, 8, 0.875),
            (160, 8, 0.75),
            (192, 8, 0.75),
        ],
    )
    candidate_n = 2 ** ceil(log2(forecasts[1]["radial_design_equivalent_n"]))
    explicit_dt = 0.12 * (2 * cfg.half_domain / candidate_n) ** 2 / cfg.viscosity
    report = {
        "schema_version": 1,
        "scope": "Conditional similarity extrapolation and sampled phase-only mesh screen, not a run or convergence certificate",
        "reference": source,
        "design_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "forecasts": forecasts,
        "explicit_diffusion_warning": {
            "scope": "Applying the existing global uniform-grid explicit rule to a fixed fine spacing, not a derived AMR stability limit",
            "candidate_core_equivalent_n": candidate_n,
            "dt_limit": explicit_dt,
            "active_interval_step_lower_bound": ceil(
                (forecasts[1]["t"] - cfg.paper_time_cutoff_start) / explicit_dt
            ),
        },
        "tenfold_mesh_candidates": rows,
        "limitations": [
            "Anchored to a spatially unvalidated peak; no fitted peak-growth law.",
            "All speeds and lengths remain in model units. No Mach number or acoustic transition is defined.",
            "Samples can miss minima and small cells do not establish solution accuracy.",
            "Envelope, Cartesian-basis variation, numerical correction, background, taper and momentum transport still need verification.",
            "The finite surrogate does not establish the paper's infinite hierarchy or a singularity.",
        ],
    }
    if args.output.exists():
        parser.error("output exists; preserve previous design records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"forecasts": forecasts, "candidates": rows}, indent=2))


if __name__ == "__main__":
    main()
