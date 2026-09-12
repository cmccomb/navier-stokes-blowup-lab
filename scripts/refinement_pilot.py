"""Run bounded AMReX projection checks; never launch a production NS run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

MARKER = "NS_PILOT_RESULT "
AMREX_COMMIT = "e60cdc18711ccf7fc0616d7a2fdf062021376976"
HYDRO_COMMIT = "e49df248aabd2cc11865eb5be734a2f5f2f65ee5"


def parse_result(stdout: str) -> dict:
    records = [
        line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)
    ]
    if len(records) != 1:
        raise ValueError("expected exactly one machine-readable pilot result")
    result = json.loads(records[0])
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("unsupported pilot result schema")
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite pilot result")
    return result


def check_result(row: dict, expected: dict | None = None) -> list[str]:
    failures = []
    for field in (
        "base_n",
        "levels",
        "max_grid_size",
        "finest_effective_n",
        "stored_cells",
        "active_cells",
    ):
        if type(row.get(field)) is not int or row[field] <= 0:
            return [f"invalid integer field: {field}"]
    for field in (
        "composite_volume",
        "velocity_linf",
        "velocity_l2_error",
        "velocity_relative_l2_error",
        "velocity_linf_error",
        "interface_velocity_linf_error",
        "divergence_before_linf",
        "divergence_after_linf",
        "divergence_after_l2",
        "coarse_fine_flux_mismatch",
        "projection_idempotence_l2",
        "elapsed_seconds",
        "peak_rss_mib",
    ):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return [f"invalid nonnegative finite field: {field}"]
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("pilot_source_sha256", ""))):
        failures.append("missing pilot source provenance")
    if expected and any(row.get(key) != value for key, value in expected.items()):
        failures.append("executable result does not match the requested case")
    if row.get("test_case") not in {"rest", "solenoidal", "mixed"}:
        failures.append("unknown test case")
    if (
        not 1 <= row["levels"] <= 4
        or not 16 <= row["base_n"] <= 128
        or row["base_n"] % 8
    ):
        failures.append("mesh outside supported pilot bounds")
    if row["finest_effective_n"] != row["base_n"] * 2 ** (row["levels"] - 1):
        failures.append("incorrect finest spacing label")
    if row["amrex_commit"] != AMREX_COMMIT or row["hydro_commit"] != HYDRO_COMMIT:
        failures.append("unrecognized upstream source pins")
    if row.get("passed") is not True:
        failures.append("executable's operator gate failed")
    if row["divergence_after_linf"] >= 1e-8:
        failures.append("post-projection divergence >= 1e-8")
    if row["coarse_fine_flux_mismatch"] >= 1e-12:
        failures.append("coarse/fine flux mismatch >= 1e-12")
    if row["projection_idempotence_l2"] >= 1e-9:
        failures.append("projection is not idempotent to tolerance")
    if abs(row["composite_volume"] - 8) >= 1e-10:
        failures.append("composite mesh does not represent the complete box")
    if row["stored_cells"] != row["levels"] * row["base_n"] ** 3:
        failures.append("unexpected fixed mesh allocation")
    expected_active = row["base_n"] ** 3 * (1 + (row["levels"] - 1) * 7 / 8)
    if row["active_cells"] != expected_active:
        failures.append("covered coarse cells were not excluded correctly")
    if row["test_case"] == "rest" and row["velocity_linf"] != 0:
        failures.append("rest is not preserved exactly")
    if row["test_case"] == "solenoidal" and row["velocity_l2_error"] >= 1e-9:
        failures.append("an already solenoidal field changed")
    return failures


def convergence(rows: list[dict]) -> list[dict]:
    """Known analytic errors, not differences relative to an unverified finest grid."""
    records = []
    for coarse, fine in pairwise(rows):
        if fine["base_n"] != 2 * coarse["base_n"]:
            raise ValueError("convergence levels must double the base resolution")
        a, b = coarse["velocity_l2_error"], fine["velocity_l2_error"]
        if a <= 0 or b <= 0 or not math.isfinite(a + b):
            raise ValueError("positive finite errors required for observed order")
        ratio = a / b
        records.append(
            {
                "coarse_base_n": coarse["base_n"],
                "fine_base_n": fine["base_n"],
                "error_ratio": ratio,
                "observed_order": math.log2(ratio),
                "passed": 2.5 <= ratio <= 6,
            }
        )
    return records


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex/ns_projection_pilot")
    )
    parser.add_argument("--resolutions", type=int, nargs=3, default=[16, 32, 64])
    parser.add_argument("--levels", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = args.resolutions
    if any(n < 16 or n > 128 or n % 8 for n in sizes) or sizes != [
        sizes[0],
        2 * sizes[0],
        4 * sizes[0],
    ]:
        parser.error("use three doubling resolutions, multiples of 8 in [16,128]")
    executable = args.executable.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("output directory must be empty; previous results are preserved")
    report = {
        "schema_version": 1,
        "scope": "static 3D projection validation, not NS evolution or paper reproduction",
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cases": [],
        "failures": [],
        "convergence": {},
        "passed": False,
    }
    refined_levels = args.levels
    cases = [
        ("rest", sizes[0], refined_levels, 32),
        ("solenoidal", sizes[0], refined_levels, 32),
    ]
    cases += [("mixed", n, levels, 32) for levels in [1, refined_levels] for n in sizes]
    cases += [("mixed", sizes[1], refined_levels, 16)]
    for test_case, n, levels, grid in cases:
        name = f"{test_case}-n{n}-l{levels}-b{grid}"
        command = [
            str(executable),
            f"n_cell={n}",
            f"levels={levels}",
            f"max_grid_size={grid}",
            f"test_case={test_case}",
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=600, check=False
            )
            (args.output / f"{name}.log").write_text(
                completed.stdout + completed.stderr
            )
            row = parse_result(completed.stdout)
            row["command"] = command[1:]
            row["exit_code"] = completed.returncode
            if completed.returncode != 0:
                report["failures"].append(
                    f"{name}: nonzero exit {completed.returncode}"
                )
            expected = {
                "test_case": test_case,
                "base_n": n,
                "levels": levels,
                "max_grid_size": grid,
            }
            issues = check_result(row, expected)
            report["failures"].extend(f"{name}: {why}" for why in issues)
            if issues:
                # Do not use invalid or misidentified records in convergence calculations.
                write_json(args.output / "summary.json", report)
                continue
            report["cases"].append(row)
            print(
                f"{name}: L2={row['velocity_l2_error']:.6g}, div={row['divergence_after_linf']:.3g}",
                flush=True,
            )
        except (subprocess.SubprocessError, ValueError, KeyError, OSError) as exc:
            report["failures"].append(f"{name}: {exc}")
        write_json(args.output / "summary.json", report)
    for levels in [1, refined_levels]:
        rows = [
            r
            for r in report["cases"]
            if r["test_case"] == "mixed"
            and r["levels"] == levels
            and r["max_grid_size"] == 32
        ]
        if len(rows) == 3:
            try:
                orders = convergence(rows)
                report["convergence"][str(levels)] = orders
                if not all(r["passed"] for r in orders):
                    report["failures"].append(
                        f"{levels}-level spatial order outside gate"
                    )
            except ValueError as exc:
                report["failures"].append(str(exc))
        else:
            report["failures"].append(f"missing {levels}-level convergence cases")
    decompositions = [
        r
        for r in report["cases"]
        if r["test_case"] == "mixed"
        and r["base_n"] == sizes[1]
        and r["levels"] == refined_levels
    ]
    if (
        len(decompositions) != 2
        or abs(
            decompositions[0]["velocity_l2_error"]
            - decompositions[1]["velocity_l2_error"]
        )
        > 1e-10
    ):
        report["failures"].append("box-decomposition invariance gate failed")
    report["passed"] = not report["failures"]
    report["completed_at"] = datetime.now(UTC).isoformat()
    write_json(args.output / "summary.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "convergence": report["convergence"],
                "failures": report["failures"],
                "summary": str(args.output / "summary.json"),
            }
        )
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
