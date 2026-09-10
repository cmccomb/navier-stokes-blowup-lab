from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.interactive import write_interactive_volume
from navier_stokes_sim.solver import SimulationResult


def _result(tmp_path: Path, *, with_velocity: bool, with_force: bool) -> SimulationResult:
    cfg = SimulationConfig(resolution=8, frames=2, t_end=0.1)
    shape = (2, 8, 8, 8, 3)
    velocity = np.zeros(shape)
    force = np.zeros(shape)
    velocity[1, ..., 2] = 1.0
    force[1, ..., 0] = -2.0
    return SimulationResult(
        config=cfg,
        output_dir=tmp_path,
        diagnostics=[],
        times=np.array([0.0, 0.1]),
        velocity_slices=np.empty((0,)),
        equatorial_slices=np.empty((0,)),
        target_slices=np.empty((0,)),
        force_slices=np.empty((0,)),
        volume_times=np.array([0.0, 0.1]),
        volume_velocity=velocity if with_velocity else None,
        volume_force=force if with_force else None,
    )


def test_interactive_volume_can_render_force_only(tmp_path: Path) -> None:
    destination = tmp_path / "force.html"
    result = _result(tmp_path, with_velocity=False, with_force=True)

    assert write_interactive_volume(result, destination)
    text = destination.read_text(encoding="utf-8")
    assert "Applied force volume" in text
    assert "force magnitude isosurfaces" in text


def test_interactive_volume_respects_explicit_field(tmp_path: Path) -> None:
    destination = tmp_path / "velocity.html"
    result = _result(tmp_path, with_velocity=True, with_force=True)

    assert write_interactive_volume(result, destination, field="velocity")
    assert "Velocity volume" in destination.read_text(encoding="utf-8")


def test_interactive_volume_reports_missing_field(tmp_path: Path) -> None:
    destination = tmp_path / "missing.html"
    result = _result(tmp_path, with_velocity=False, with_force=True)

    assert not write_interactive_volume(result, destination, field="velocity")
    assert not destination.exists()
