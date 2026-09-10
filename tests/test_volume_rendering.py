import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.volume_rendering import _similarity_points


def test_similarity_points_select_nonzero_axial_structure() -> None:
    config = SimulationConfig(resolution=8, frames=2, t_end=0.6)
    vectors = np.zeros((8, 8, 8, 3))
    vectors[2:6, 2:6, 2:6, 2] = 3.0

    points, values, strengths = _similarity_points(
        vectors,
        0.6,
        config,
        extent=2.0,
        maximum_points=1_000,
    )

    assert len(points) > 0
    assert points.shape[1] == 3
    assert np.all(values == 3.0)
    assert np.all(strengths == 3.0)


def test_similarity_points_handle_exact_rest() -> None:
    config = SimulationConfig(resolution=8)

    points, values, strengths = _similarity_points(
        np.zeros((8, 8, 8, 3)),
        0.0,
        config,
        extent=2.0,
        maximum_points=1_000,
    )

    assert points.shape == (0, 3)
    assert not len(values)
    assert not len(strengths)
