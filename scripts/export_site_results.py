"""Export a compact, path-scrubbed fleet summary for the results site."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PUBLIC_FIELDS = (
    "case",
    "variant",
    "resolution",
    "t_end",
    "max_dt",
    "pulses_enabled",
    "forcing_end",
    "peak_speed",
    "target_peak_speed",
    "peak_vorticity",
    "tracking_relative_l2",
    "divergence_linf",
    "cells_per_radial_scale",
    "cells_per_axial_scale",
    "spectral_k95",
    "spectral_tail_fraction",
    "resolved",
)

FLOAT_FIELDS = {
    "t_end",
    "max_dt",
    "forcing_end",
    "peak_speed",
    "target_peak_speed",
    "peak_vorticity",
    "tracking_relative_l2",
    "divergence_linf",
    "cells_per_radial_scale",
    "cells_per_axial_scale",
    "spectral_k95",
    "spectral_tail_fraction",
}
INT_FIELDS = {"resolution"}
BOOL_FIELDS = {"pulses_enabled", "resolved"}


def _clean_value(field: str, value: str) -> Any:
    if value == "":
        return None
    if field in FLOAT_FIELDS:
        return float(value)
    if field in INT_FIELDS:
        return int(value)
    if field in BOOL_FIELDS:
        return value.strip().lower() in {"1", "true", "yes"}
    return value


def load_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = []
        for source in csv.DictReader(stream):
            row = {
                field: _clean_value(field, source.get(field, ""))
                for field in PUBLIC_FIELDS
            }
            rows.append(row)
    return rows


def build_payload(
    rows: list[dict[str, Any]], highlight: str, generated_at: str
) -> dict[str, Any]:
    matching = [row for row in rows if row["case"] == highlight]
    if not matching:
        raise ValueError(f"highlight case {highlight!r} is absent from the summary")
    headline = matching[0]
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "scope": (
            "Finite-grid manufactured-solution evidence; not an independent proof "
            "and not yet the paper's exact infinite construction."
        ),
        "headline": headline,
        "history_count": len(rows),
        "media": [
            {
                "title": "Late-time concentration at 224³",
                "kind": "video",
                "mp4": "media/current-best.mp4",
                "gif": "media/current-best.gif",
                "poster": "media/current-best-poster.png",
                "caption": (
                    "Computed and target axial speed slices from t=0.94 to 0.995. "
                    "The endpoint still passes the four-cell geometric gate."
                ),
            },
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--highlight", required=True)
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
    payload = build_payload(load_summary(args.summary), args.highlight, generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote current-best result from {payload['history_count']} records to {args.output}")


if __name__ == "__main__":
    main()
