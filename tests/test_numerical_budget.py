import numpy as np
import pytest

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.numerical_budget import (
    centered_symbols,
    phase_time_grid,
    resolution_budget,
)
from navier_stokes_sim.time_stepping import forcing_log_rate_bound, forcing_step_limit


def test_modified_wavenumber_matches_the_actual_centered_stencil() -> None:
    n, mode = 128, 8
    dx = 2 / n
    x = np.arange(n) * dx
    wave = np.sin(mode * np.pi * x)
    discrete = (np.roll(wave, -1) - np.roll(wave, 1)) / (2 * dx)
    exact = mode * np.pi * np.cos(mode * np.pi * x)
    response = centered_symbols(n / mode)
    np.testing.assert_allclose(
        discrete, exact * response["first_derivative_ratio"], atol=1e-12
    )
    assert not centered_symbols(2)["above_nyquist"]
    assert centered_symbols(8)["first_derivative_relative_error"] == pytest.approx(
        0.0996836838
    )


def test_resolution_budget_distinguishes_core_length_from_wave_accuracy() -> None:
    budget = resolution_budget()
    assert 510 < budget["minimum_even_uniform_N_midpoint"] < 520
    assert (
        budget["minimum_even_uniform_N_inner_half_bump"]
        > budget["minimum_even_uniform_N_midpoint"]
    )
    row = next(r for r in budget["rows"] if r["resolution"] == 192)
    assert 2.9 < row["midpoint_points_per_wavelength"] < 3.1
    assert 0.58 < row["first_derivative_relative_error"] < 0.61
    assert row["midpoint_last_time_meeting_design_points"] < 0.9
    earlier = resolution_budget(0.94)
    assert earlier["midpoint_wavelength"] == pytest.approx(
        2 * budget["midpoint_wavelength"]
    )
    velocity, quadratic = budget["radial_harmonic_designs"]
    assert quadratic["minimum_even_uniform_N_inner_half_bump"] == 1200
    assert quadratic["maximum_radial_log_spacing"] == pytest.approx(
        velocity["maximum_radial_log_spacing"] / 2
    )


def test_predefined_phase_clock_matches_runtime_ceiling() -> None:
    cfg = SimulationConfig(t_start=0.55, t_end=0.985)
    times = np.array(phase_time_grid(cfg))
    assert times[0] == cfg.t_start
    assert times[-1] == cfg.t_end
    assert np.all(np.diff(times) > 0)
    phase_steps = -np.diff(np.log(cfg.t_star - times)) * forcing_log_rate_bound(cfg)
    assert phase_steps.max() <= cfg.forcing_phase_step + 1e-12
    np.testing.assert_allclose(phase_steps[:-1], cfg.forcing_phase_step, atol=1e-12)
    for t, dt in zip(times[:-2], np.diff(times)[:-1]):
        assert dt == pytest.approx(forcing_step_limit(t, cfg), abs=1e-14)
    with pytest.raises(ValueError, match="enabled oscillatory"):
        phase_time_grid(SimulationConfig(pulses_enabled=False))
