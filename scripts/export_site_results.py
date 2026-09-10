"""Export one completed simulation as the canonical public result."""

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


def load_run(run_dir: Path) -> dict[str, Any]:
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
        "id": run_dir.name,
        "label": "Complete start-from-rest trajectory",
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
        **final,
    }


def build_payload(run: dict[str, Any], generated_at: str) -> dict[str, Any]:
    if run["t_start"] != 0:
        raise ValueError("the canonical public run must start at t=0")
    if not run["resolved"]:
        raise ValueError("the canonical public endpoint must pass the resolution gate")
    return {
        "schema_version": 2,
        "generated_at": generated_at,
        "scope": (
            "Finite-grid manufactured-solution evidence; not an independent proof "
            "and not yet the paper's exact infinite construction."
        ),
        "run": run,
        "media": {
            "mp4": "media/current-best.mp4",
            "gif": "media/current-best.gif",
            "poster": "media/current-best-poster.png",
            "endpoint": "figures/current-best-endpoint.png",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--generated-at",
        default=None,
        help="ISO-8601 timestamp; defaults to the current UTC time",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    generated_at = args.generated_at or datetime.now(UTC).isoformat()
    payload = build_payload(load_run(args.run), generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    run = payload["run"]
    print(f"published {run['resolution']}^3 from-rest run through t={run['t_end']}")


if __name__ == "__main__":
    main()
