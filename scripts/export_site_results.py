"""Export completed start-from-rest simulations as the public comparison set."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DIAGNOSTIC_FIELDS = (
    "t",
    "peak_speed",
    "target_peak_speed",
    "kinetic_energy",
    "kinetic_energy_core",
    "peak_vorticity",
    "tracking_relative_l2",
    "divergence_linf",
    "cells_per_radial_scale",
    "cells_per_axial_scale",
    "spectral_k95",
    "spectral_tail_fraction",
    "force_l2",
    "resolved",
)


def _clean_value(field: str, value: str) -> Any:
    if field == "resolved":
        return value.strip().lower() == "true"
    return float(value)


def load_run(run_dir: Path, run_id: str, label: str) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with (run_dir / "diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no diagnostics in {run_dir}")

    config = metadata["config"]
    final = {
        field: _clean_value(field, rows[-1][field]) for field in DIAGNOSTIC_FIELDS
    }
    return {
        "id": run_id,
        "label": label,
        "created_at": metadata["created_at"],
        "solver": metadata["solver"],
        "scope_warning": metadata["scope_warning"],
        "resolution": int(config["resolution"]),
        "t_start": float(config["t_start"]),
        "t_end": float(config["t_end"]),
        "frames": len(rows),
        "rest_until": float(config["paper_time_cutoff_start"]),
        "fully_active_from": float(config["paper_time_cutoff_end"]),
        "integrator": config["integrator"],
        "pressure_projection": config["pressure_projection"],
        "frame_spacing": config.get("frame_spacing", "linear"),
        "cfl": float(config.get("cfl", 0.32)),
        "max_dt": float(config.get("max_dt", 0.02)),
        "media": {
            "mp4": f"media/{run_id}.mp4",
            "gif": f"media/{run_id}.gif",
            "poster": f"media/{run_id}-poster.png",
            "endpoint": f"figures/{run_id}-endpoint.png",
        },
        **final,
    }


def build_payload(
    runs: list[dict[str, Any]], featured_id: str, generated_at: str
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one public run is required")
    ids = [run["id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("public run IDs must be unique")
    if featured_id not in ids:
        raise ValueError(f"featured run {featured_id!r} is not in the public set")
    for run in runs:
        if run["t_start"] != 0:
            raise ValueError(f"public run {run['id']!r} must start at t=0")
        if not run["resolved"]:
            raise ValueError(
                f"public run {run['id']!r} endpoint must pass the resolution gate"
            )
    return {
        "schema_version": 3,
        "generated_at": generated_at,
        "scope": (
            "Finite-grid manufactured-solution evidence; not an independent proof "
            "and not yet the paper's exact infinite construction."
        ),
        "featured_id": featured_id,
        "runs": runs,
    }


def _pairs(values: list[str], option: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        try:
            key, item = value.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{option} must use ID=VALUE, got {value!r}") from error
        if not key or not item:
            raise ValueError(f"{option} must use non-empty ID=VALUE")
        pairs[key] = item
    return pairs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="completed run to publish; repeat for each start-from-rest case",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="ID=TEXT",
        help="display label for a run ID",
    )
    parser.add_argument("--featured", required=True, help="run ID shown first")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--generated-at",
        default=None,
        help="ISO-8601 timestamp; defaults to the current UTC time",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run_paths = _pairs(args.run, "--run")
    labels = _pairs(args.label, "--label")
    unknown_labels = labels.keys() - run_paths.keys()
    if unknown_labels:
        raise ValueError(f"labels without runs: {sorted(unknown_labels)}")
    runs = [
        load_run(
            Path(path),
            run_id,
            labels.get(run_id, run_id.replace("-", " ").title()),
        )
        for run_id, path in run_paths.items()
    ]
    generated_at = args.generated_at or datetime.now(UTC).isoformat()
    payload = build_payload(runs, args.featured, generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"published {len(runs)} start-from-rest run(s); featured={args.featured}")


if __name__ == "__main__":
    main()
