"""Time resolution of the retained oscillatory forcing phases."""

from __future__ import annotations

from math import expm1, inf, log, pi

from .config import SimulationConfig


def forcing_log_rate_bound(cfg: SimulationConfig) -> float:
    """Bound |d phase / d(-log(tau))| in the manufactured forcing.

    X is proportional to r²/q and q*L >= tau on the physical branch, so
    |d log X/dt| <= 1/tau. Also |d eta/dt| <= D*eta_support/tau on
    the pulse support. The gate bound resolves a full bump half-width with
    at least 2*pi/forcing_phase_step samples.
    """

    if cfg.profile_model != "paper-surrogate" or not cfg.pulses_enabled:
        return 0.0
    if not (cfg.pulse_rtheta_strength > 0 or cfg.pulse_rz_strength > 0):
        return 0.0
    log_frequency = 2 * pi * abs(cfg.pulse_log_frequency) / log(2)
    rates = [2 * pi / (cfg.pulse_time_width * log(2))]
    for level in range(cfg.pulse_hierarchy_levels):
        frequency_scale = 1 + 0.65 * level
        radial = cfg.pulse_radial_frequency * frequency_scale
        if cfg.pulse_rtheta_strength > 0:
            rates.append(radial + (1 + 0.17 * level) * log_frequency)
        if cfg.pulse_rz_strength > 0:
            axial = (
                frequency_scale
                * cfg.pulse_axial_frequency
                * (0.5 - cfg.h)
                * cfg.paper_eta_support
            )
            rates.append(radial + axial + (0.73 + 0.09 * level) * log_frequency)
    # The advective term in the manufactured force is quadratic in velocity:
    # products of two retained waves can oscillate at the sum of their rates.
    return 2 * max(rates)


def forcing_step_limit(t: float, cfg: SimulationConfig) -> float:
    """Bound accumulated phase change across the whole proposed step."""

    rate = forcing_log_rate_bound(cfg)
    if cfg.forcing_phase_step is None or rate == 0:
        return inf
    tau = cfg.t_star - t
    if tau <= 0:
        raise ValueError("forcing timestep is defined only before t_star")
    return -tau * expm1(-cfg.forcing_phase_step / rate)
