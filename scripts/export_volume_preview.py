"""Export a compact vector-volume history without loading the whole archive.

This standalone NumPy-only script can run on a solver host via stdin. The
archive stays open during extraction, so atomic checkpoint replacement cannot
mix metadata from one saved state with arrays from another.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def _small_array(archive: ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(archive.read(name + ".npy")), allow_pickle=False)


def read_preview(
    run: Path, field: str, max_resolution: int = 48
) -> dict[str, np.ndarray]:
    if field not in {"velocity", "force"}:
        raise ValueError("field must be velocity or force")
    if max_resolution < 8:
        raise ValueError("preview resolution must be at least 8")
    partial = run / "partial-checkpoint.npz"
    if partial.is_file():
        source = partial
        key = "volume_" + field
        time_key = "volume_times"
    else:
        source = run / ("volumes.npz" if field == "velocity" else "forces.npz")
        key = field
        time_key = "times"
    with ZipFile(source) as archive:
        if source == partial:
            metadata = json.loads(str(_small_array(archive, "metadata").item()))
        else:
            metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
        times = _small_array(archive, time_key)
        with archive.open(key + ".npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported NPY version: {version}")
            if len(shape) != 5 or not shape[0]:
                raise ValueError(
                    f"no captured {field} history; enable volume capture for future runs"
                )
            if fortran or dtype.hasobject or shape[-1] != 3:
                raise ValueError("expected C-ordered numeric vector volumes")
            if shape[0] != len(times) or len(set(shape[1:4])) != 1:
                raise ValueError("inconsistent volume clock or spatial shape")
            n = shape[1]
            if n != metadata["config"]["resolution"]:
                raise ValueError("source resolution differs from metadata")
            stride = max(1, int(np.ceil(n / max_resolution)))
            frame_size = int(np.prod(shape[1:])) * dtype.itemsize
            sampled_frames = []
            for _ in times:
                raw = handle.read(frame_size)
                if len(raw) != frame_size:
                    raise ValueError("truncated vector-volume checkpoint")
                frame = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
                sampled_frames.append(
                    frame[::stride, ::stride, ::stride].astype(np.float32)
                )
    half = float(metadata["config"]["half_domain"])
    axis = ((np.arange(n) + 0.5) * (2 * half / n) - half)[::stride]
    metadata = {
        **metadata,
        "preview_version": 1,
        "field": field,
        "source_archive": source.name,
        "display_sampling": f"every {stride}th source cell; float32",
    }
    return {
        "times": times,
        "vectors": np.stack(sampled_frames),
        "axis": axis,
        "metadata": np.asarray(json.dumps(metadata)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", choices=("velocity", "force"), required=True)
    parser.add_argument("--max-resolution", type=int, default=48)
    args = parser.parse_args()
    data = read_preview(args.run, args.field, args.max_resolution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "times": data["times"].tolist(),
                "shape": list(data["vectors"].shape),
                "bytes": args.output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
