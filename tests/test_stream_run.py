import json
import shutil
import subprocess

import numpy as np
import pytest

from scripts.stream_run import build_manifest, plane_vectors, read_frames, render_pair


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


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="video tools unavailable",
)
def test_stream_media_can_encode_real_rest_frames(tmp_path):
    axis = np.linspace(-0.75, 0.75, 4)
    frames = [
        {"time": np.asarray(t), "axis": axis, "velocity": np.zeros((4, 4, 4, 3))}
        for t in (0.0, 0.55)
    ]
    render_pair(frames, "velocity", tmp_path / "rest", 192)
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "json",
                str(tmp_path / "rest.mp4"),
            ],
            text=True,
        )
    )["streams"][0]
    assert info["width"] % 2 == info["height"] % 2 == 0
    assert int(info["nb_frames"]) == 2
    assert (tmp_path / "rest.gif").stat().st_size > 0
