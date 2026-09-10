import numpy as np

from scripts.render_site_media import _interpolate, _playback_times


def test_playback_clock_compresses_rest_and_densifies_late_time() -> None:
    checkpoints = np.linspace(0.0, 0.99, 51)
    playback = _playback_times(checkpoints, rest_until=0.55, movie_frames=120)

    assert len(playback) == 120
    assert playback[0] == 0
    np.testing.assert_allclose(playback[-1], 0.99)
    assert np.count_nonzero(playback == 0) == 10
    assert playback[-1] - playback[-2] < playback[20] - playback[19]


def test_playback_interpolates_between_solver_checkpoints() -> None:
    times = np.array([0.0, 0.5, 1.0])
    field = np.array([[[0.0]], [[2.0]], [[4.0]]])

    assert _interpolate(field, times, 0.25).item() == 1.0
    assert _interpolate(field, times, 0.75).item() == 3.0
