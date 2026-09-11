import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageSequence

from navier_stokes_sim.config import SimulationConfig
from scripts.stream_run import (
    CachedFrames,
    build_manifest,
    build_playback,
    plane_vectors,
    read_frames,
    render_3d,
    render_pair,
    verify_gif,
)


def test_playback_keeps_all_saved_frames_on_a_shared_gif_mp4_clock():
    plan = build_playback([0, 0.1, 0.3])
    assert plan["mp4_frame_repeats"] == [50, 10, 25]
    assert plan["source_frame_duration_ms"] == [1000, 200, 500]
    assert plan["duration_seconds"] == 1.7
    assert plan["simulation_time_proportional"] is False
    assert plan["saved_frames_per_second"] == 5


def test_playback_grows_instead_of_dropping_history():
    times = [0, *np.linspace(0.55, 0.985, 486)]
    plan = build_playback(times)
    assert len(plan["source_frame_duration_ms"]) == 487
    assert plan["source_frame_duration_ms"] == [1000, *([200] * 485), 500]
    assert plan["duration_seconds"] == 98.5
    skewed = build_playback([0, 1e-12, 1e-11, 1])
    assert min(skewed["mp4_frame_repeats"]) == 10
    still = build_playback([0.55])
    assert still["source_frame_duration_ms"] == [4000]
    for invalid in ([], [0, 0], [1, 0], [0, np.nan], [0, np.inf]):
        with pytest.raises(ValueError):
            build_playback(invalid)


def test_activation_does_not_change_saved_frame_playback_speed():
    plan = build_playback([0, 0.55, 0.553, 0.556, 0.559], slow_motion_after=0.55)
    assert plan["mp4_frame_repeats"] == [50, 10, 10, 10, 25]
    assert plan == build_playback([0, 0.55, 0.553, 0.556, 0.559])


def test_full_history_cache_is_incremental_and_not_a_24_frame_window(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    cache = tmp_path / "cache"
    axis = np.linspace(-0.75, 0.75, 4)
    for i in range(30):
        vectors = np.full((4, 4, 4, 3), i, dtype=np.float32)
        np.savez(
            source / f"frame-{i:06d}.npz",
            time=i / 100,
            axis=axis,
            velocity=vectors,
            force=-vectors,
        )
    frames, count = read_frames(source, cache=cache)
    assert isinstance(frames, CachedFrames)
    assert len(frames) == count == 30
    assert float(frames[0]["time"]) == 0
    assert float(frames[-1]["time"]) == 0.29
    assert len(frames[-24:]) == 24
    real_load = np.load

    def cached_only(path, **kwargs):
        assert path.parent == cache
        return real_load(path, **kwargs)

    monkeypatch.setattr(np, "load", cached_only)
    again, count = read_frames(source, cache=cache)
    assert len(again) == count == 30
    np.testing.assert_array_equal(again[0]["velocity"], 0)
    (source / "frame-000001.npz").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        read_frames(source, cache=cache)


def test_stream_reads_only_finalized_real_frames_and_bounds_window(tmp_path):
    axis = np.linspace(-0.75, 0.75, 4)
    for i in range(5):
        vectors = np.zeros((4, 4, 4, 3)) + i
        np.savez(
            tmp_path / f"frame-{i:06}.npz",
            time=i / 10,
            axis=axis,
            velocity=vectors,
            force=2 * vectors,
        )
    (tmp_path / "frame-000005.tmp.npz").write_bytes(b"incomplete")
    frames, count = read_frames(tmp_path, window=3)
    assert count == 5
    assert [float(f["time"]) for f in frames] == [0.2, 0.3, 0.4]
    vertical, equatorial, coordinate = plane_vectors(frames[-1], "force")
    assert vertical.shape == equatorial.shape == (4, 4, 3)
    np.testing.assert_array_equal(vertical, 8)
    assert coordinate == -0.25
    remote = {
        "source_commit": "abc",
        "status": "running",
        "observed_at": "now",
        "started_at": "then",
        "config": {"resolution": 192},
        "diagnostics": {"t": 0.3},
    }
    result = build_manifest(remote, frames, count, {})
    assert result["latest_t"] == 0.4
    assert result["diagnostics"]["t"] == 0.3
    assert result["clip_times"] == [0.2, 0.3, 0.4]
    assert result["display_resolution"] == 4
    assert "host" not in json.dumps(result)


def test_stream_preserves_plane_orientation_and_rejects_nonfinite(tmp_path):
    axis = np.linspace(-0.75, 0.75, 4)
    vectors = np.arange(4**3 * 3).reshape(4, 4, 4, 3).astype(float)
    frame = {"axis": axis, "velocity": vectors}
    vertical, equatorial, _ = plane_vectors(frame, "velocity")
    np.testing.assert_array_equal(vertical, vectors[:, 1, :, :])
    np.testing.assert_array_equal(equatorial, vectors[:, :, 1, :])
    vectors[0, 0, 0, 0] = np.nan
    np.savez(
        tmp_path / "frame-000000.npz",
        axis=axis,
        time=0,
        velocity=vectors,
        force=vectors,
    )
    with pytest.raises(ValueError, match="nonfinite"):
        read_frames(tmp_path)


def test_native_planes_are_taken_before_browser_reduction(tmp_path):
    n = 64
    axis = (np.arange(n) + 0.5) * 2 / n - 1
    vectors = np.arange(n**3 * 3, dtype=np.float32).reshape(n, n, n, 3)
    path = tmp_path / "frame-000000.npz"
    np.savez_compressed(path, time=0.56, axis=axis, velocity=vectors, force=-vectors)
    original = path.read_bytes()
    frames, _ = read_frames(tmp_path)
    frame = frames[0]
    vertical, equatorial, coordinate = plane_vectors(frame, "velocity")
    assert coordinate == axis[31]
    np.testing.assert_array_equal(vertical, vectors[:, 31, :, :])
    np.testing.assert_array_equal(equatorial, vectors[:, :, 31, :])
    np.testing.assert_array_equal(frame["velocity"], vectors[::2, ::2, ::2])
    assert frame["velocity"].shape == (32, 32, 32, 3)
    assert len(frame["volume_axis"]) == 32
    assert frame["velocity"].flags.owndata
    assert frame["velocity_planes"].flags.owndata
    assert path.read_bytes() == original
    remote = {
        "id": "native-run",
        "source_commit": "abc",
        "status": "running",
        "observed_at": "now",
        "started_at": "then",
        "config": {"resolution": n},
    }
    manifest = build_manifest(remote, frames, 1, {})
    assert manifest["id"] == "native-run"
    assert manifest["archive"]["resolution"] == manifest["display_resolution"] == 64
    assert manifest["archive"]["dtype"] == "float32"
    assert manifest["display_resolution_3d"] == 32


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="video tools unavailable",
)
@pytest.mark.parametrize(
    ("times", "activation"),
    [
        ((0.0, 0.55), None),
        ((0.0, 0.55, 0.56), 0.55),
        ((0.55,), 0.55),
        ((0, *np.linspace(0.55, 0.85, 29)), 0.55),
    ],
)
def test_stream_media_can_encode_real_rest_frames(tmp_path, times, activation):
    axis = np.linspace(-0.75, 0.75, 4)
    frames = [
        {"time": np.asarray(t), "axis": axis, "velocity": np.zeros((4, 4, 4, 3))}
        for t in times
    ]
    rendering = render_pair(
        frames, "velocity", tmp_path / "rest", 192, slow_motion_after=activation
    )
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,nb_frames,duration,avg_frame_rate",
                "-of",
                "json",
                str(tmp_path / "rest.mp4"),
            ],
            text=True,
        )
    )["streams"][0]
    assert info["width"] % 2 == info["height"] % 2 == 0
    assert (info["width"], info["height"]) == (1440, 792)
    timing = build_playback(list(times), activation)
    assert int(info["nb_frames"]) == sum(timing["mp4_frame_repeats"])
    assert info["avg_frame_rate"] == "50/1"
    assert float(info["duration"]) == pytest.approx(timing["duration_seconds"])
    assert (tmp_path / "rest.gif").stat().st_size > 0
    with Image.open(tmp_path / "rest.gif") as gif:
        assert gif.n_frames == len(times)
        duration = []
        labels = []
        for frame in ImageSequence.Iterator(gif):
            duration.append(frame.info["duration"])
            labels.append(np.asarray(frame.convert("RGB"))[35:90, 145:640])
        assert duration == timing["source_frame_duration_ms"]
        # Title pixels stay fixed while the separately anchored timestamp changes.
        for label in labels[1:]:
            np.testing.assert_array_equal(labels[0], label)
    assert rendering["gif_verification"] == verify_gif(
        tmp_path / "rest.gif", timing["source_frame_duration_ms"]
    )
    with pytest.raises(ValueError, match="every saved frame"):
        verify_gif(tmp_path / "rest.gif", [200] * (len(times) + 1))


def test_stream_3d_is_self_contained_with_real_time_controls(tmp_path):
    axis = np.linspace(-0.875, 0.875, 8)
    frames = [
        {"time": np.asarray(t), "axis": axis, "velocity": np.zeros((8, 8, 8, 3))}
        for t in (0.0, 0.55)
    ]
    destination = tmp_path / "rest.html"
    render_3d(frames, "velocity", destination, SimulationConfig(resolution=8).to_dict())
    html = destination.read_text()
    assert "Play time" in html and "Saved time:" in html
    assert "0.550000" in html
    assert json.dumps("8³ display samples")[1:-1] in html
    assert "Arial, Helvetica, sans-serif" in html
    assert ".modebar{top:55px!important}" in html
    assert "resizeVolume" in html and "autoexpand: false" in html
    assert "{plot_id}" not in html
    assert "document.getElementById('stream-volume')" in html
    assert '<script src="https://cdn.plot.ly' not in html
