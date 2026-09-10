"""Atomic per-frame volume files: capture history without retaining it in RAM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .time_stepping import forcing_log_rate_bound


def preview_times(cfg: SimulationConfig) -> np.ndarray:
    """Sample the active interval uniformly in bounded forcing phase, plus rest."""
    if cfg.preview_phase_step is None:
        return np.empty(0)
    start = cfg.t_start
    if cfg.profile_model == "paper-surrogate":
        start = min(max(start, cfg.paper_time_cutoff_start), cfg.t_end)
    span = np.log((cfg.t_star - start) / (cfg.t_star - cfg.t_end))
    rate = max(forcing_log_rate_bound(cfg), 1.0)
    intervals = max(1, int(np.ceil(rate * span / cfg.preview_phase_step)))
    times = cfg.t_star - np.exp(
        np.linspace(
            np.log(cfg.t_star - start), np.log(cfg.t_star - cfg.t_end), intervals + 1
        )
    )
    times[0], times[-1] = start, cfg.t_end
    return np.unique(np.concatenate(([cfg.t_start], times)))


def initialize_store(path: Path, cfg: SimulationConfig, *, resume: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    metadata = {"snapshot_store_version": 1, "config": cfg.to_dict()}
    if manifest.exists():
        if not resume or json.loads(manifest.read_text()) != metadata:
            raise ValueError(
                f"existing snapshot store in {path}; use a new output directory"
            )
        return
    if any(path.iterdir()):
        raise ValueError(f"unrecognized snapshot store in {path}")
    temporary = path / "manifest.tmp.json"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)


def write_snapshot(
    path: Path,
    index: int,
    t: float,
    cfg: SimulationConfig,
    *,
    velocity: np.ndarray | None,
    force: np.ndarray | None,
    preview: bool = False,
) -> None:
    stride = (
        max(1, int(np.ceil(cfg.resolution / cfg.preview_resolution))) if preview else 1
    )
    dtype = np.float32 if preview else np.float64
    axis = ((np.arange(cfg.resolution) + 0.5) * cfg.dx - cfg.half_domain)[::stride]
    arrays = {"time": np.asarray(t), "axis": axis}
    for name, values in (("velocity", velocity), ("force", force)):
        if values is not None:
            arrays[name] = np.asarray(values[::stride, ::stride, ::stride], dtype=dtype)
    destination = path / f"frame-{index:06d}.npz"
    temporary = destination.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)
