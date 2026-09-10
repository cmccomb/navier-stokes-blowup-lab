from dataclasses import replace
from math import log, pi

import pytest

from navier_stokes_sim.cli import _parser
from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.time_stepping import forcing_log_rate_bound, forcing_step_limit


def test_forcing_clock_bounds_phase_integrated_across_step() -> None:
    cfg = SimulationConfig()
    for t in (0.55, 0.90, 0.985, 0.9999):
        dt = forcing_step_limit(t, cfg)
        phase_change = forcing_log_rate_bound(cfg) * log((1 - t) / (1 - t - dt))
        assert phase_change == pytest.approx(cfg.forcing_phase_step)
    assert forcing_step_limit(0.985, cfg) < 0.000053
    assert 2 * pi / cfg.forcing_phase_step > 40


def test_more_modes_and_narrower_gates_tighten_forcing_step() -> None:
    cfg = SimulationConfig()
    baseline = forcing_step_limit(0.98, cfg)
    assert forcing_step_limit(0.98, replace(cfg, pulse_hierarchy_levels=8)) < baseline
    assert forcing_step_limit(0.98, replace(cfg, pulse_time_width=0.1)) < baseline
    assert forcing_step_limit(0.98, replace(cfg, pulses_enabled=False)) == float("inf")
    assert forcing_step_limit(0.98, replace(cfg, forcing_phase_step=None)) == float(
        "inf"
    )


def test_cli_can_capture_velocities_while_deferring_graphics() -> None:
    args = _parser().parse_args(["--no-3d", "--capture-velocity-volumes"])
    assert args.no_3d and args.capture_velocity_volumes
    assert args.forcing_phase_step == 0.15
    assert _parser().parse_args(["--no-forcing-phase-limit"]).forcing_phase_step is None
