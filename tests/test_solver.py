import csv
from dataclasses import replace

import numpy as np
import pytest
from phi.flow import Solve, math

from navier_stokes_sim import solver
from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.frontier import compare_frontier
from navier_stokes_sim.profile import discrete_divergence
from navier_stokes_sim.solver import (
    _as_numpy,
    _frame_times,
    _project,
    _to_field,
    load_matching_checkpoint,
    run_simulation,
)


def test_similarity_frame_spacing_concentrates_samples_near_singular_time() -> None:
    linear = SimulationConfig(t_end=0.99, frames=9)
    similarity = replace(linear, frame_spacing="similarity")

    linear_times = _frame_times(linear)
    similarity_times = _frame_times(similarity)

    assert similarity_times[0] == 0
    assert similarity_times[-1] == pytest.approx(0.99)
    assert np.all(np.diff(similarity_times) > 0)
    assert np.diff(similarity_times)[-1] < np.diff(linear_times)[-1]
    assert np.count_nonzero(similarity_times > 0.9) > np.count_nonzero(
        linear_times > 0.9
    )


def test_fft_projection_matches_centered_discrete_divergence() -> None:
    cfg = SimulationConfig(
        resolution=16,
        frames=2,
        t_end=0.1,
        pressure_projection="fft",
    )
    math.set_global_precision(64)
    rng = np.random.default_rng(19)
    velocity = rng.standard_normal((16, 16, 16, 3))
    energy_before = np.sum(velocity * velocity)
    solve = Solve("CG", rel_tol=1e-7, abs_tol=1e-9)

    projected, pressure = _project(_to_field(velocity, cfg), solve, cfg)
    projected_np = _as_numpy(projected)
    projected_twice, _ = _project(projected, solve, cfg)

    assert pressure is None
    assert np.max(np.abs(discrete_divergence(projected_np, cfg.dx))) < 1e-11
    assert np.sum(projected_np * projected_np) <= energy_before
    assert np.allclose(_as_numpy(projected_twice), projected_np, atol=1e-12)


def test_tiny_solver_run(tmp_path) -> None:
    cfg = SimulationConfig(
        resolution=8,
        t_end=0.04,
        frames=3,
        paper_time_cutoff_start=0.0,
        paper_time_cutoff_end=0.01,
        max_dt=0.02,
        pressure_rel_tol=1e-5,
        pressure_abs_tol=1e-7,
        capture_volumes=False,
    )
    result = run_simulation(cfg, tmp_path, save_final_state=True)
    assert len(result.diagnostics) == 3
    assert (tmp_path / "diagnostics.csv").exists()
    assert (tmp_path / "slices.npz").exists()
    assert (tmp_path / "run.json").exists()
    assert (tmp_path / "final-state.npz").exists()
    with np.load(tmp_path / "final-state.npz", allow_pickle=False) as final_state:
        assert float(final_state["time"]) == pytest.approx(cfg.t_end)
        assert final_state["velocity"].shape == (8, 8, 8, 3)
    with (tmp_path / "diagnostics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert float(rows[-1]["divergence_linf"]) < 1e-5
    assert "peak_vorticity" in rows[-1]
    assert "energy_balance_residual" in rows[-1]
    assert "force_laplacian_l2" in rows[-1]
    assert "force_time_derivative_l2" in rows[-1]
    assert "spectral_tail_fraction" in rows[-1]
    assert "paper_coordinate_residual_linf" in rows[-1]
    assert "pulse_annulus_energy_fraction" in rows[-1]
    assert "kinetic_energy_core" in rows[-1]
    assert result.volume_velocity is None

    restored = load_matching_checkpoint(tmp_path, cfg)
    assert restored is not None
    assert restored.config == cfg
    assert np.array_equal(restored.times, result.times)
    assert np.array_equal(restored.velocity_slices, result.velocity_slices)

    changed = SimulationConfig(**(cfg.to_dict() | {"cfl": 0.3}))
    assert load_matching_checkpoint(tmp_path, changed) is None

    resumed = run_simulation(cfg, tmp_path, save_final_state=True)
    assert np.array_equal(resumed.velocity_slices, result.velocity_slices)
    assert not (tmp_path / "partial-checkpoint.npz").exists()


def test_solver_preserves_exact_initial_rest_interval(tmp_path) -> None:
    cfg = SimulationConfig(
        resolution=8,
        t_end=0.08,
        frames=5,
        paper_time_cutoff_start=0.04,
        paper_time_cutoff_end=0.06,
        max_dt=0.01,
        pressure_projection="fft",
        capture_volumes=False,
    )
    result = run_simulation(cfg, tmp_path)
    for index in (0, 1, 2):
        assert result.times[index] <= cfg.paper_time_cutoff_start
        assert np.count_nonzero(result.velocity_slices[index]) == 0
        assert result.diagnostics[index]["force_linf"] == 0
    assert result.diagnostics[-1]["peak_speed"] > 0


def test_partial_checkpoint_resumes_after_interruption(tmp_path, monkeypatch) -> None:
    cfg = SimulationConfig(
        resolution=8,
        t_end=0.06,
        frames=4,
        paper_time_cutoff_start=0.0,
        paper_time_cutoff_end=0.01,
        max_dt=0.02,
        pressure_rel_tol=1e-5,
        pressure_abs_tol=1e-7,
        capture_volumes=False,
    )
    interrupted_dir = tmp_path / "interrupted"
    reference_dir = tmp_path / "reference"
    original_writer = solver._write_partial_checkpoint

    def interrupt_after_second_frame(*args, **kwargs) -> None:
        original_writer(*args, **kwargs)
        diagnostics = args[5]
        if len(diagnostics) == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(solver, "_write_partial_checkpoint", interrupt_after_second_frame)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_simulation(cfg, interrupted_dir)
    assert (interrupted_dir / "partial-checkpoint.npz").is_file()

    monkeypatch.setattr(solver, "_write_partial_checkpoint", original_writer)
    resumed = run_simulation(cfg, interrupted_dir)
    reference = run_simulation(cfg, reference_dir)
    assert np.allclose(resumed.velocity_slices, reference.velocity_slices)
    assert not (interrupted_dir / "partial-checkpoint.npz").exists()


def test_frontier_comparison_writes_summary_and_figure(tmp_path) -> None:
    cfg = SimulationConfig(
        resolution=8,
        t_start=0.4,
        t_end=0.42,
        frames=2,
        paper_time_cutoff_start=0.0,
        paper_time_cutoff_end=0.05,
        max_dt=0.02,
        pressure_projection="fft",
        capture_volumes=False,
    )
    first_dir = tmp_path / "n8"
    second_dir = tmp_path / "n10"
    output_dir = tmp_path / "comparison"
    run_simulation(cfg, first_dir)
    run_simulation(replace(cfg, resolution=10), second_dir)

    cases = compare_frontier([first_dir, second_dir], output_dir)

    assert [case.result.config.resolution for case in cases] == [8, 10]
    assert (output_dir / "frontier-summary.csv").is_file()
    assert (output_dir / "frontier.json").is_file()
    assert (output_dir / "frontier.png").stat().st_size > 10_000
