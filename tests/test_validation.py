import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.solver import SimulationResult
from navier_stokes_sim.validation import fit_power_law


def test_power_law_fit_recovers_known_exponent(tmp_path) -> None:
    cfg = SimulationConfig(
        resolution=8,
        t_end=0.6,
        frames=5,
        ramp_time=0.05,
        paper_time_cutoff_start=0.0,
        paper_time_cutoff_end=0.05,
        capture_volumes=False,
    )
    times = np.linspace(0.1, 0.6, 5)
    exponent = -0.75
    diagnostics = [
        {"t": t, "tau": 1 - t, "resolved": True, "quantity": (1 - t) ** exponent}
        for t in times
    ]
    empty_slices = np.empty((5, 8, 8, 3))
    result = SimulationResult(
        config=cfg,
        output_dir=tmp_path,
        diagnostics=diagnostics,
        times=times,
        velocity_slices=empty_slices,
        equatorial_slices=empty_slices,
        target_slices=empty_slices,
        force_slices=empty_slices,
        volume_times=np.empty(0),
        volume_velocity=None,
    )
    fit = fit_power_law(result, "quantity", exponent)
    assert np.isclose(fit.measured_exponent, exponent)
    assert np.isclose(fit.r_squared, 1.0)
