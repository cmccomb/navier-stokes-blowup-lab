"""Render 3D time histories with magnitude, x, y, z and vector directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.interactive import write_series_explorer
from navier_stokes_sim.volume_rendering import write_time_animation
from navier_stokes_sim.volume_series import COMPONENTS, VolumeSeries
from scripts.export_volume_preview import read_preview


def load_series(
    source: Path, *, field: str | None = None, t_end: float | None = None
) -> VolumeSeries:
    if source.is_dir():
        if field is None:
            raise ValueError("--field is required when reading a run directory")
        data = read_preview(source, field)
    else:
        with np.load(source, allow_pickle=False) as arrays:
            data = {k: arrays[k] for k in ("vectors", "times", "axis", "metadata")}
    metadata = json.loads(str(data["metadata"].item()))
    captured_field = metadata["field"]
    if field is not None and field != captured_field:
        raise ValueError("requested field differs from the captured preview")
    mask = (
        np.ones(len(data["times"]), dtype=bool)
        if t_end is None
        else data["times"] <= t_end + 1e-12
    )
    return VolumeSeries(
        data["vectors"][mask],
        data["times"][mask],
        data["axis"],
        SimulationConfig(**metadata["config"]),
        captured_field,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--preview", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", choices=("velocity", "force"))
    parser.add_argument("--stem", default="volume")
    parser.add_argument("--component", choices=COMPONENTS, default="magnitude")
    parser.add_argument("--camera", choices=("orbit", "fixed"), default="orbit")
    parser.add_argument("--format", choices=("gif", "mp4", "both"), default="both")
    parser.add_argument("--t-end", type=float, help="last allowed saved time")
    parser.add_argument(
        "--vmax", type=float, help="fixed scalar limit for paired views"
    )
    parser.add_argument(
        "--vector-max", type=float, help="fixed vector scale for paired views"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    series = load_series(args.preview or args.run, field=args.field, t_end=args.t_end)
    prefix = f"{args.stem}-{series.field}"
    interactive = args.output / f"{prefix}-interactive.html"
    write_series_explorer(series, interactive)
    print(f"wrote {interactive}")
    formats = ("gif", "mp4") if args.format == "both" else (args.format,)
    for extension in formats:
        destination = args.output / f"{prefix}-{args.component}-time.{extension}"
        write_time_animation(
            series,
            destination,
            component=args.component,
            orbit=args.camera == "orbit",
            vmax=args.vmax,
            vector_max=args.vector_max,
        )
        print(f"wrote {destination}")
    manifest = {
        "field": series.field,
        "component": args.component,
        "times": series.times.tolist(),
        "source_resolution": series.config.resolution,
        "display_samples_per_axis": len(series.axis),
        "mesh": series.config.mesh_preset,
        "coordinates": "physical",
        "camera": args.camera,
        "scalar_limit": args.vmax or series.peak(args.component),
        "vector_limit": args.vector_max or series.peak("magnitude"),
        "temporal_interpolation": False,
        "checkpoint_hold_seconds": 0.5,
    }
    (args.output / f"{prefix}-{args.component}-time.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
