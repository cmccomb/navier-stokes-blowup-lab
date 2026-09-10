"""Render interactive and orbiting 3D media from complete or partial runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.interactive import write_interactive_vector_volume
from navier_stokes_sim.volume_rendering import write_orbit_gif

VectorField = Literal["velocity", "force"]


def _load_volumes(
    run_dir: Path, field: VectorField
) -> tuple[np.ndarray, np.ndarray, SimulationConfig]:
    partial = run_dir / "partial-checkpoint.npz"
    if partial.is_file():
        with np.load(partial, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"].item()))
            key = "volume_velocity" if field == "velocity" else "volume_force"
            volumes = arrays[key].copy()
            times = arrays["volume_times"].copy()
        config = SimulationConfig(**metadata["config"])
        return volumes, times, config

    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = SimulationConfig(**metadata["config"])
    filename = "volumes.npz" if field == "velocity" else "forces.npz"
    key = "velocity" if field == "velocity" else "force"
    with np.load(run_dir / filename, allow_pickle=False) as arrays:
        return arrays[key].copy(), arrays["times"].copy(), config


def render_volume_media(
    run_dir: Path,
    output_dir: Path,
    *,
    field: VectorField,
    stem: str,
) -> list[Path]:
    volumes, times, config = _load_volumes(run_dir, field)
    if not len(times) or not volumes.size:
        raise ValueError(f"no captured {field} volumes in {run_dir}")
    nonzero = np.flatnonzero(np.any(volumes != 0, axis=(1, 2, 3, 4)))
    selected = int(nonzero[-1]) if len(nonzero) else len(times) - 1
    output_dir.mkdir(parents=True, exist_ok=True)
    interactive = output_dir / f"{stem}-{field}-interactive.html"
    orbit = output_dir / f"{stem}-{field}-orbit.gif"
    write_interactive_vector_volume(
        volumes, times, config, interactive, field=field
    )
    if not write_orbit_gif(
        volumes[selected], float(times[selected]), config, orbit, field=field
    ):
        raise ValueError(f"selected {field} volume is identically zero")
    return [interactive, orbit]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", choices=("velocity", "force"), required=True)
    parser.add_argument("--stem", default="volume")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    paths = render_volume_media(
        args.run, args.output, field=args.field, stem=args.stem
    )
    print("\n".join(f"wrote {path}" for path in paths))


if __name__ == "__main__":
    main()
