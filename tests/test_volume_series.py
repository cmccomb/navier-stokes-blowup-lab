import json

import numpy as np
import pytest
from PIL import Image

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.interactive import build_volume_figure
from navier_stokes_sim.volume_rendering import write_time_animation
from navier_stokes_sim.volume_series import VolumeSeries, scalar_values
from scripts.export_volume_preview import read_preview


def sample_series() -> VolumeSeries:
    cfg = SimulationConfig(resolution=8)
    axis = (np.arange(8) + 0.5) * cfg.dx - cfg.half_domain
    vectors = np.zeros((3, 8, 8, 8, 3))
    vectors[1, 2:6, 2:6, 2:6] = [3, 4, 0]
    vectors[2, 2:6, 2:6, 2:6] = [-6, -8, 0]
    return VolumeSeries(vectors, np.array([0, 0.6, 0.8]), axis, cfg, "force")


def test_non_axial_motion_is_visible_and_scales_stay_fixed() -> None:
    series = sample_series()
    assert series.peak("magnitude") == 10
    assert series.peak("x") == 6
    assert np.max(np.abs(scalar_values(series.vectors, "z"))) == 0
    figure = build_volume_figure(series)
    assert len(figure.frames) == 3
    assert all(len(trace.x) == len(series.axis) ** 3 for trace in figure.data[:4])
    assert all(
        trace.x is None and trace.y is None and trace.z is None
        for frame in figure.frames
        for trace in frame.data[:4]
    )
    assert all(
        len(trace.value) == len(series.axis) ** 3
        for frame in figure.frames
        for trace in frame.data[:4]
    )
    assert all(f.data[0].cmax == 10 for f in figure.frames)
    assert all(f.data[1].cmin == -6 for f in figure.frames)
    assert all(f.data[4].sizeref == figure.data[4].sizeref for f in figure.frames)
    assert np.any(np.asarray(figure.frames[1].data[4].u) != 0)
    assert np.any(np.asarray(figure.frames[1].data[4].v) != 0)
    assert all(t.visible is None for f in figure.frames for t in f.data)
    assert figure.layout.scene.xaxis.range == (-1, 1)


def test_slider_frame_ids_do_not_collide_at_close_times() -> None:
    series = sample_series()
    series.times = np.array([0.98, 0.98001, 0.98002])
    names = [f.name for f in build_volume_figure(series).frames]
    assert len(set(names)) == len(names)


def test_preview_stream_preserves_sample_coordinates_and_values(tmp_path) -> None:
    cfg = SimulationConfig(resolution=16)
    rng = np.random.default_rng(14)
    vectors = rng.normal(size=(2, 16, 16, 16, 3))
    np.savez(
        tmp_path / "partial-checkpoint.npz",
        metadata=np.asarray(json.dumps({"config": cfg.to_dict()})),
        volume_times=[0, 0.7],
        volume_force=vectors,
        volume_velocity=np.empty(0),
    )
    preview = read_preview(tmp_path, "force", max_resolution=8)
    assert np.array_equal(
        preview["vectors"], vectors[:, ::2, ::2, ::2].astype(np.float32)
    )
    assert np.allclose(preview["axis"], ((np.arange(16) + 0.5) * cfg.dx - 1)[::2])
    with pytest.raises(ValueError, match="no captured velocity history"):
        read_preview(tmp_path, "velocity")


def test_time_animation_includes_changing_states(tmp_path) -> None:
    destination = tmp_path / "time.gif"
    write_time_animation(sample_series(), destination, orbit=False, fps=2, holds=1)
    with Image.open(destination) as gif:
        assert gif.n_frames == 3
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"))
        gif.seek(2)
        last = np.asarray(gif.convert("RGB"))
        assert not np.array_equal(first, last)


def test_nonfinite_or_misaligned_series_is_rejected() -> None:
    series = sample_series()
    series.vectors[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        VolumeSeries(series.vectors, series.times, series.axis, series.config, "force")
