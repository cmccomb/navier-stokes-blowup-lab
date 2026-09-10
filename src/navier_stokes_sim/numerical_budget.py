"""Source-backed local-wave design estimates, not a convergence certificate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from math import ceil, expm1, isfinite, log, pi, sin, sqrt
from pathlib import Path

from .config import SimulationConfig
from .time_stepping import forcing_log_rate_bound, forcing_step_limit


def centered_symbols(points_per_wavelength: float) -> dict[str, float | bool]:
    """Amplitude response of the centered first derivative and 3-point Laplacian."""
    if not isfinite(points_per_wavelength) or points_per_wavelength <= 0:
        raise ValueError("points per wavelength must be positive")
    kh = 2 * pi / points_per_wavelength
    first = sin(kh) / kh
    second = (sin(kh / 2) / (kh / 2)) ** 2
    return {
        "above_nyquist": points_per_wavelength > 2,
        "first_derivative_ratio": first,
        "laplacian_ratio": second,
        "first_derivative_relative_error": abs(1 - first),
        "laplacian_relative_error": abs(1 - second),
    }


def radial_wavelength(t: float, cfg: SimulationConfig, x_similarity: float) -> float:
    """Finest retained radial phase at z=0: d(alpha log X)/dr = 2 alpha/r."""
    if not 0 <= t < cfg.t_star or x_similarity <= 0:
        raise ValueError("require positive X and time before t_star")
    alpha = cfg.pulse_radial_frequency * (1 + 0.65 * (cfg.pulse_hierarchy_levels - 1))
    radius = cfg.base_radius * sqrt(2 * x_similarity * (cfg.t_star - t))
    return pi * radius / alpha


def phase_time_grid(cfg: SimulationConfig) -> list[float]:
    """Predetermined phase-only clock; actual integration may subdivide it.

    This design clock does not replace advective/diffusive stability limits,
    the early ramp limit, or exact output/event alignment in the solver.
    """
    cfg.validate()
    rate = forcing_log_rate_bound(cfg)
    if rate <= 0 or cfg.forcing_phase_step is None:
        raise ValueError("a phase clock requires enabled oscillatory forcing")
    tau_start = cfg.t_star - cfg.t_start
    log_interval = log(tau_start / (cfg.t_star - cfg.t_end))
    ds = cfg.forcing_phase_step / rate
    count = ceil(log_interval / ds)
    times = [
        cfg.t_start - tau_start * expm1(-index * ds)
        for index in range(count)
    ]
    return times + [cfg.t_end]


def resolution_budget(t: float = 0.985, *, points: float = 8) -> dict:
    """Budget one known demanding feature while keeping its limitations explicit."""
    if not isfinite(points) or points <= 2:
        raise ValueError("design points per wavelength must exceed Nyquist")
    cfg = SimulationConfig(t_end=t)
    cfg.validate()
    middle = (cfg.paper_annulus_xa + cfg.paper_annulus_xb) / 2
    # Inner edge of the region where the radial bump is >= half its peak.
    half_bump_coordinate = sqrt(1 - 1 / (1 - log(0.5)))
    inner_half = (
        middle
        - (cfg.paper_annulus_xb - cfg.paper_annulus_xa) / 2 * half_bump_coordinate
    )
    wavelength = radial_wavelength(t, cfg, middle)
    half_wavelength = radial_wavelength(t, cfg, inner_half)
    rows = []
    for n in (128, 160, 192, 256, 512, 640, 1024):
        local = replace(cfg, resolution=n)
        ppw = wavelength / local.dx
        # lambda(t) = lambda(0) sqrt(tau/t_star) on this reference plane.
        tau_required = (
            points * local.dx / radial_wavelength(0, cfg, middle)
        ) ** 2 * cfg.t_star
        rows.append(
            {
                "resolution": n,
                "dx": local.dx,
                "midpoint_points_per_wavelength": ppw,
                "inner_half_bump_points_per_wavelength": half_wavelength / local.dx,
                "midpoint_last_time_meeting_design_points": max(
                    0, cfg.t_star - tau_required
                ),
                **centered_symbols(ppw),
                "extrapolated_peak_footprint_GiB": 11069710928
                / 1024**3
                * (n / 192) ** 3,
            }
        )
    phase = cfg.forcing_phase_step
    alpha = cfg.pulse_radial_frequency * (1 + 0.65 * (cfg.pulse_hierarchy_levels - 1))
    harmonics = []
    for multiplier in (1, 2):
        harmonics.append(
            {
                "phase_wavenumber_multiplier": multiplier,
                "minimum_even_uniform_N_midpoint": 2
                * ceil(points * 2 * cfg.half_domain * multiplier / wavelength / 2),
                "minimum_even_uniform_N_inner_half_bump": 2
                * ceil(points * 2 * cfg.half_domain * multiplier / half_wavelength / 2),
                "maximum_radial_log_spacing": pi / (alpha * multiplier * points),
                "maximum_radial_relative_spacing": expm1(
                    pi / (alpha * multiplier * points)
                ),
            }
        )
    return {
        "scope": "Local radial-wave estimate at z=0, not an error bound for the full field. Other locations, directions, envelopes, background structure and nonlinear products may require more resolution. Temporal gates change which modes contribute.",
        "t": t,
        "tau": cfg.t_star - t,
        "design_points_per_wavelength": points,
        "config": cfg.to_dict(),
        "midpoint_X": middle,
        "inner_half_bump_X": inner_half,
        "midpoint_wavelength": wavelength,
        "inner_half_bump_wavelength": half_wavelength,
        "minimum_even_uniform_N_midpoint": 2
        * ceil(points * 2 * cfg.half_domain / wavelength / 2),
        "minimum_even_uniform_N_inner_half_bump": 2
        * ceil(points * 2 * cfg.half_domain / half_wavelength / 2),
        "forcing_dt_limit": forcing_step_limit(t, cfg),
        "forcing_log_rate_bound": forcing_log_rate_bound(cfg),
        "maximum_log_time_step": phase / forcing_log_rate_bound(cfg),
        "phase_only_time_grid": phase_time_grid(cfg),
        "radial_harmonic_designs": harmonics,
        "mesh_warning": "Candidate radial phase metric only, not a generated mesh. Quadratic advection can double phase wavenumbers; envelopes are not bandlimited. Axial/angular structure, axis regularity, the background and exterior taper still need independent checks. The current uniform-grid FFT projector cannot use a stretched mesh.",
        "constant_frequency_midpoint_quadrature_relative_error": phase
        / (2 * sin(phase / 2))
        - 1,
        "memory_warning": "N^3 extrapolation from one 192^3 macOS preflight (10.31 GiB peak footprint), not a measurement or a memory cap. Fleet RAM is not pooled by this solver.",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=0.985)
    parser.add_argument("--points-per-wavelength", type=float, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = resolution_budget(args.t_end, points=args.points_per_wavelength)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
