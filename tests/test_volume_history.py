import gzip
import hashlib

import numpy as np
import pytest

from scripts.volume_history import (
    decode_chunk,
    embedded_history,
    render_3d,
    validate_history,
)


def test_lossless_signed_browser_vectors_and_fixed_full_history_scales(tmp_path):
    axis = np.linspace(-0.9, 0.9, 8)
    vectors = np.zeros((8, 8, 8, 3), dtype=np.float32)
    vectors[3, 4, 5] = [-1e-30, 2e-30, -3e-30]
    frames = [
        {"time": t, "axis": axis, "velocity": vectors * i}
        for i, t in enumerate((0, 0.55, 0.8))
    ]
    destination = tmp_path / "site/media/stream-flow-3d.html"
    history = render_3d(
        frames, "velocity", destination, {"resolution": 192, "half_domain": 1}
    )
    assert embedded_history(destination) == history
    validate_history(tmp_path, history, "velocity", [0, 0.55, 0.8])
    for source, frame in zip(frames, history["frames"], strict=True):
        np.testing.assert_array_equal(
            decode_chunk(tmp_path / frame["path"], frame, history["shape"]),
            source["velocity"],
        )
    assert 0 < history["limits"]["magnitude"] < 1e-28
    # Adding a later frame reuses the exact immutable earlier chunk paths.
    again = render_3d(
        frames[:2], "velocity", destination, {"resolution": 192, "half_domain": 1}
    )
    assert again["frames"] == history["frames"][:2]


@pytest.mark.parametrize("bad", ["shape", "nan", "not_rest"])
def test_semantic_chunk_validation_not_just_metadata(tmp_path, bad):
    vectors = np.zeros((4, 4, 4, 3), dtype=np.float32)
    if bad == "nan":
        vectors[0, 0, 0, 0] = np.nan
    elif bad == "not_rest":
        vectors[0, 0, 0, 0] = 1
    raw = vectors.tobytes() + (b"extra" if bad == "shape" else b"")
    payload = gzip.compress(raw, mtime=0)
    digest = hashlib.sha256(payload).hexdigest()
    name = f"site/media/stream-volumes/velocity-{digest}.f32.gz"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    history = {
        "schema_version": 1,
        "field": "velocity",
        "encoding": "gzip",
        "dtype": "<f4",
        "axis": [-0.75, -0.25, 0.25, 0.75],
        "shape": [4, 4, 4, 3],
        "half_domain": 1,
        "cache_frames": 3,
        "limits": dict.fromkeys(["magnitude", "x", "y", "z"], 1.0),
        "frames": [
            {
                "index": 0,
                "time": 0,
                "path": name,
                "sha256": digest,
                "bytes": len(payload),
            }
        ],
    }
    with pytest.raises(ValueError):
        validate_history(tmp_path, history, "velocity", [0])
