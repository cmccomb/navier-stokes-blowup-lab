"""Lossless browser-sample chunks and a bounded-memory full-history explorer."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

CHUNK_PATH = re.compile(
    r"site/media/stream-volumes/(velocity|force)-([0-9a-f]{64})\.f32\.gz"
)
MANIFEST_TAG = '<script id="volume-history" type="application/json">'


def validate_descriptor(history: dict, field: str, times: list[float]) -> list[str]:
    """Resolve a closed derived-output allowlist before any file transfer."""
    axis = np.asarray(history["axis"], dtype=float)
    n = len(axis)
    if (
        history["schema_version"] != 1
        or history["field"] != field
        or history["encoding"] != "gzip"
        or history["dtype"] != "<f4"
        or axis.ndim != 1
        or not 2 <= n <= 32
        or history["shape"] != [n, n, n, 3]
        or not np.isfinite(axis).all()
        or not np.all(np.diff(axis) > 0)
        or not math.isfinite(history["half_domain"])
        or history["half_domain"] <= 0
        or np.max(np.abs(axis)) > history["half_domain"]
        or history["cache_frames"] != 3
        or set(history["limits"]) != {"magnitude", "x", "y", "z"}
        or not all(math.isfinite(v) and v > 0 for v in history["limits"].values())
        or len(history["frames"]) != len(times)
        or not times
        or not all(math.isfinite(t) for t in times)
        or any(b <= a for a, b in pairwise(times))
    ):
        raise ValueError("invalid full-history 3D descriptor")
    paths = []
    for i, (frame, t) in enumerate(zip(history["frames"], times, strict=True)):
        match = CHUNK_PATH.fullmatch(frame["path"])
        if (
            not match
            or match[1] != field
            or match[2] != frame["sha256"]
            or frame["index"] != i
            or frame["time"] != t
            or not 0 < frame["bytes"] < 2 * n**3 * 3 * 4
        ):
            raise ValueError("invalid 3D frame coverage or derived asset path")
        paths.append(frame["path"])
    return paths


def decode_chunk(path: Path, frame: dict, shape: list[int]) -> np.ndarray:
    """Check compressed bytes and bound decompression to one browser volume."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size != frame["bytes"]:
        raise ValueError("missing or partial 3D frame asset")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != frame["sha256"]:
        raise ValueError("3D frame checksum mismatch")
    expected = math.prod(shape) * 4
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
        raw = handle.read(expected + 1)
    if len(raw) != expected:
        raise ValueError("3D frame shape mismatch")
    vectors = np.frombuffer(raw, dtype="<f4").reshape(shape)
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite 3D vectors")
    return vectors


def validate_history(
    folder: Path, history: dict, field: str, times: list[float]
) -> None:
    validate_descriptor(history, field, times)
    peaks = np.zeros(4)
    for frame in history["frames"]:
        vectors = decode_chunk(folder / frame["path"], frame, history["shape"])
        if frame["index"] == 0 and times[0] == 0 and np.any(vectors):
            raise ValueError(
                "3D from-rest history must begin with the actual zero field"
            )
        peaks = np.maximum(peaks, vector_peaks(vectors))
    limits = dict(zip(("magnitude", "x", "y", "z"), (float(p) or 1.0 for p in peaks)))
    if history["limits"] != limits:
        raise ValueError("3D fixed scales disagree with saved vectors")


def vector_peaks(vectors):
    values = vectors.astype(np.float64)
    return [
        float(np.linalg.norm(values, axis=-1).max()),
        *np.max(np.abs(values), axis=(0, 1, 2)),
    ]


def render_3d(frames, field: str, destination: Path, config: dict) -> dict:
    """Write all saved times without embedding all vector arrays in the HTML."""
    axis = frames[0].get("volume_axis", frames[0]["axis"])
    n = len(axis)
    history = {
        "schema_version": 1,
        "field": field,
        "encoding": "gzip",
        "dtype": "<f4",
        "shape": [n, n, n, 3],
        "axis": axis.tolist(),
        "half_domain": config["half_domain"],
        "source_resolution": config["resolution"],
        "cache_frames": 3,
        "frames": [],
    }
    peaks = np.zeros(4)
    folder = destination.parent / "stream-volumes"
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        vectors = np.asarray(frame[field], dtype="<f4")
        if vectors.shape != (n, n, n, 3) or not np.isfinite(vectors).all():
            raise ValueError("invalid browser vector field")
        if not np.array_equal(frame.get("volume_axis", frame["axis"]), axis):
            raise ValueError("3D coordinates changed during history")
        peaks = np.maximum(peaks, vector_peaks(vectors))
        payload = gzip.compress(vectors.tobytes(order="C"), compresslevel=6, mtime=0)
        digest = hashlib.sha256(payload).hexdigest()
        name = f"{field}-{digest}.f32.gz"
        path = folder / name
        if not path.exists() or path.read_bytes() != payload:
            pending = path.with_suffix(".tmp")
            pending.write_bytes(payload)
            pending.replace(path)
        history["frames"].append(
            {
                "index": i,
                "time": float(frame["time"]),
                "path": f"site/media/stream-volumes/{name}",
                "sha256": digest,
                "bytes": len(payload),
            }
        )
    history["limits"] = dict(
        zip(("magnitude", "x", "y", "z"), (float(p) or 1.0 for p in peaks))
    )
    validate_descriptor(history, field, [f["time"] for f in history["frames"]])
    template = Path(__file__).with_name("volume_history.html").read_text()
    script = Path(__file__).with_name("volume_history.js").read_text()
    html = template.replace("__HISTORY__", json.dumps(history).replace("<", "\\u003c"))
    html = html.replace("__PLOTLY__", get_plotlyjs()).replace("__VIEWER__", script)
    pending = destination.with_suffix(".tmp.html")
    pending.write_text(html)
    pending.replace(destination)
    return history


def embedded_history(path: Path) -> dict:
    text = path.read_text()
    if text.count(MANIFEST_TAG) != 1:
        raise ValueError("3D viewer is missing its exact history descriptor")
    return json.loads(text.split(MANIFEST_TAG, 1)[1].split("</script>", 1)[0])
