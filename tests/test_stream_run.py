import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageSequence

from navier_stokes_sim.config import SimulationConfig
from scripts.stream_run import (
    build_manifest,
    build_playback,
    plane_vectors,
    read_frames,
    render_3d,
    render_pair,
)


def test_playback_holds_are_proportional_and_share_gif_mp4_clock():
    plan = build_playback([0, 0.1, 0.3])
    assert plan["mp4_frame_repeats"] == [100, 200, 25]
    assert plan["source_frame_duration_ms"] == [2000, 4000, 500]
    assert plan["duration_seconds"] == 6.5
    assert plan["segments"][0]["simulation_units_per_playback_second"] == pytest.approx(
        0.05
    )
    assert plan["max_transition_rounding_error_ms"] < 1e-10


def test_playback_rounds_cumulative_edges_and_bounds_duration():
    times = np.linspace(0.55, 0.57, 24).tolist()
    plan = build_playback(times)
    assert plan["traversal_seconds"] == 4
    assert plan["max_transition_rounding_error_ms"] <= 10
    assert sum(plan["source_frame_duration_ms"]) == 4500
    assert all(n >= 1 for n in plan["mp4_frame_repeats"])
    assert build_playback([0, 1, 100])["duration_seconds"] == 12
    skewed = build_playback([0, 1e-12, 1e-11, 1])
    assert min(skewed["mp4_frame_repeats"]) >= 1
    assert skewed["max_transition_rounding_error_ms"] >= 39
    still = build_playback([0.55])
    assert still["source_frame_duration_ms"] == [4000]
    assert still["segments"] == []
    for invalid in ([], [0, 0], [1, 0], [0, np.nan], [0, np.inf], list(range(25))):
        with pytest.raises(ValueError):
            build_playback(invalid)


def test_labeled_bullet_time_does_not_let_rest_dominate_early_clip():
    plan = build_playback([0, 0.55, 0.553, 0.556, 0.559], slow_motion_after=0.55)
    assert plan["duration_seconds"] == 8.5
    normal, slow = plan["segments"]
    assert normal["end_t"] == slow["start_t"] == 0.55
    assert normal["duration_seconds"] / plan["duration_seconds"] < 0.5
    assert slow["label"] == "Slow motion"
    assert slow["slowdown_factor"] > 60
    assert plan["mp4_frame_repeats"] == [200, 67, 66, 67, 25]
    # A rolling active clip slows only the final six actual intervals.
    rolling = build_playback(
        np.linspace(0.6, 0.66, 24).tolist(), slow_motion_after=0.55
    )
    assert rolling["segments"][-1]["start_index"] == 17
    assert rolling["segments"][-1]["slowdown_factor"] > 1
    assert rolling["duration_seconds"] <= 12
    assert (
        build_playback([0, 0.5, 0.55], slow_motion_after=0.55)["segments"][0]["label"]
        == "Replay"
    )


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
    [((0.0, 0.55), None), ((0.0, 0.55, 0.56), 0.55), ((0.55,), 0.55)],
)
def test_stream_media_can_encode_real_rest_frames(tmp_path, times, activation):
    axis = np.linspace(-0.75, 0.75, 4)
    frames = [
        {"time": np.asarray(t), "axis": axis, "velocity": np.zeros((4, 4, 4, 3))}
        for t in times
    ]
    render_pair(
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
        duration = []
        labels = []
        for frame in ImageSequence.Iterator(gif):
            duration.append(frame.info["duration"])
            labels.append(np.asarray(frame.convert("RGB"))[35:90, 145:640])
        assert duration == timing["source_frame_duration_ms"]
        # Title pixels stay fixed while the separately anchored timestamp changes.
        for label in labels[1:]:
            np.testing.assert_array_equal(labels[0], label)


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
