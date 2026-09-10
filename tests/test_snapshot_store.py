from dataclasses import replace

import numpy as np
import pytest

from navier_stokes_sim import solver
from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.snapshot_store import preview_times
from navier_stokes_sim.time_stepping import forcing_log_rate_bound
from scripts.export_volume_preview import read_preview


def test_preview_clock_resolves_active_forcing_and_keeps_rest() -> None:
    cfg = SimulationConfig(t_end=0.985, preview_phase_step=0.3)
    times = preview_times(cfg)
    assert times[0] == 0
    assert times[1] == cfg.paper_time_cutoff_start
    assert times[-1] == cfg.t_end
    phases = forcing_log_rate_bound(cfg) * np.log((1 - times[1:-1]) / (1 - times[2:]))
    assert np.max(phases) <= 0.3 + 1e-12
    assert 480 < len(times) < 500
    assert np.array_equal(preview_times(replace(cfg, t_end=0.1)), [0, 0.1])


def test_slices_own_memory_and_streams_do_not_accumulate(tmp_path, monkeypatch) -> None:
    original = solver._write_partial_checkpoint

    def checked_writer(*args, **kwargs):
        for slices in args[5:9]:
            assert all(a.flags.owndata for a in slices)
        assert args[10] == [] and args[11] == []
        original(*args, **kwargs)

    monkeypatch.setattr(solver, "_write_partial_checkpoint", checked_writer)
    cfg = SimulationConfig(
        resolution=16,
        t_end=0.03,
        frames=3,
        pressure_projection="fft",
        paper_time_cutoff_start=0.01,
        paper_time_cutoff_end=0.02,
        stream_volumes=True,
        capture_force_volumes=True,
        preview_phase_step=0.3,
        preview_resolution=8,
        volume_frames=3,
    )
    result = solver.run_simulation(cfg, tmp_path, save_final_state=True)
    assert result.volume_velocity is None and result.volume_force is None
    paths = sorted((tmp_path / "full-volumes").glob("frame-*.npz"))
    assert len(paths) == 3
    with np.load(paths[-1]) as arrays:
        assert arrays["force"].shape == (16, 16, 16, 3)
        assert arrays["force"].dtype == np.float64
    preview = read_preview(tmp_path, "velocity")
    assert len(preview["times"]) > len(result.times)
    assert preview["vectors"].shape[1:] == (8, 8, 8, 3)
    assert np.count_nonzero(preview["vectors"][0]) == 0
    full = read_preview(tmp_path / "full-volumes", "velocity", max_resolution=8)
    assert np.array_equal(preview["vectors"][-1], full["vectors"][-1])
    restored = solver.load_simulation_result(tmp_path)
    assert np.array_equal(restored.volume_times, result.volume_times)


def test_streamed_resume_matches_uninterrupted_run(tmp_path, monkeypatch) -> None:
    cfg = SimulationConfig(
        resolution=8,
        t_end=0.03,
        frames=4,
        pressure_projection="fft",
        paper_time_cutoff_start=0.005,
        paper_time_cutoff_end=0.015,
        stream_volumes=True,
        capture_force_volumes=True,
        preview_phase_step=0.3,
        preview_resolution=8,
    )
    original = solver._write_partial_checkpoint

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        if len(args[4]) == 2:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(solver, "_write_partial_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        solver.run_simulation(cfg, tmp_path / "resumed")
    monkeypatch.setattr(solver, "_write_partial_checkpoint", original)
    resumed = solver.run_simulation(cfg, tmp_path / "resumed")
    reference = solver.run_simulation(cfg, tmp_path / "reference")
    assert np.array_equal(resumed.velocity_slices, reference.velocity_slices)
    for field in ("velocity", "force"):
        actual = read_preview(tmp_path / "resumed", field)
        expected = read_preview(tmp_path / "reference", field)
        assert np.array_equal(actual["times"], expected["times"])
        assert np.array_equal(actual["vectors"], expected["vectors"])
