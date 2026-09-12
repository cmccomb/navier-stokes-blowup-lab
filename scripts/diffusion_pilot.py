"""Validate multilevel Crank-Nicolson vector diffusion, not a full NS trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from itertools import pairwise
from pathlib import Path

from scripts.refinement_pilot import AMREX_COMMIT, write_json

MARKER = "NS_DIFFUSION_RESULT "


def parse_and_check(stdout: str, expected: dict) -> dict:
    lines = [
        line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)
    ]
    if len(lines) != 1:
        raise ValueError("missing or duplicate diffusion result")
    row = json.loads(lines[0])
    if (
        not isinstance(row, dict)
        or row.get("schema_version") != 1
        or row.get("passed") is not True
    ):
        raise ValueError("invalid or failed diffusion result")
    if row.get("amrex_commit") != AMREX_COMMIT or any(
        row.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("wrong source or requested case")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", ""))):
        raise ValueError("missing source provenance")
    for key in (
        "dt",
        "t_end",
        "initial_linf",
        "final_linf",
        "l2_error",
        "mass_balance_defect",
        "max_linear_residual",
        "composite_volume",
        "peak_rss_mib",
        "elapsed_seconds",
        "initial_energy",
        "final_energy",
        "max_energy_increase",
    ):
        if (
            type(row.get(key)) not in (int, float)
            or not math.isfinite(row[key])
            or row[key] < 0
        ):
            raise ValueError(f"invalid {key}")
    if (
        row["mass_balance_defect"] >= 1e-9
        or row["max_linear_residual"] >= 1e-9
        or abs(row["composite_volume"] - 8) >= 1e-10
    ):
        raise ValueError("conservation or solver gate failed")
    if row.get("components") != 3 or not math.isclose(
        row["dt"] * row["steps"], row["t_end"], rel_tol=1e-12
    ):
        raise ValueError("inconsistent vector count or clock")
    if row["test_case"] in ("rest", "spatial", "temporal") and row["initial_linf"] != 0:
        raise ValueError("forced case did not start from exact rest")
    if row["test_case"] == "rest" and row["final_linf"] != 0:
        raise ValueError("rest changed")
    if row["test_case"] == "constant" and row["l2_error"] >= 1e-10:
        raise ValueError("constant changed")
    if row["test_case"] == "decay" and row["max_energy_increase"] >= 1e-10:
        raise ValueError("unforced diffusion increased energy")
    return row


def orders(rows: list[dict], parameter: str) -> list[dict]:
    result = []
    for a, b in pairwise(rows):
        if b[parameter] != 2 * a[parameter] or not (
            a["l2_error"] > 0 and b["l2_error"] > 0
        ):
            raise ValueError("convergence requires doubling and positive errors")
        ratio = a["l2_error"] / b["l2_error"]
        result.append(
            {
                "coarse": a[parameter],
                "fine": b[parameter],
                "ratio": ratio,
                "observed_order": math.log2(ratio),
                "passed": 2.5 <= ratio <= 6,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex/ns_diffusion_pilot")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("preserve previous results: output directory must be empty")
    executable = args.executable.resolve(strict=True)
    report = {
        "schema_version": 1,
        "scope": __doc__,
        "platform": platform.platform(),
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cases": [],
        "failures": [],
        "convergence": {},
        "passed": False,
    }
    cases = [(m, 16, 3, 8, 32) for m in ("rest", "constant", "decay")]
    cases += [("temporal", 16, 4, steps, 32) for steps in (16, 32, 64)]
    cases += [("spatial", n, 3, 32, 32) for n in (16, 32, 64)]
    cases += [("spatial", 32, 3, 32, 16), ("temporal", 16, 8, 32, 32)]
    for mode, n, levels, steps, box in cases:
        expected = {
            "test_case": mode,
            "base_n": n,
            "levels": levels,
            "steps": steps,
            "max_grid_size": box,
        }
        name = f"{mode}-n{n}-l{levels}-s{steps}-b{box}"
        command = [
            str(executable),
            f"test_case={mode}",
            f"n_cell={n}",
            f"levels={levels}",
            f"steps={steps}",
            f"max_grid_size={box}",
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=600
            )
            (args.output / f"{name}.log").write_text(process.stdout + process.stderr)
            if process.returncode:
                raise ValueError(f"exit {process.returncode}; see raw log")
            row = parse_and_check(process.stdout, expected)
            row["command"] = command[1:]
            report["cases"].append(row)
            print(
                f"{name}: L2={row['l2_error']:.7g}; mass defect={row['mass_balance_defect']:.3g}",
                flush=True,
            )
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            report["failures"].append(f"{name}: {error}")
        write_json(args.output / "summary.json", report)
    for mode, param in (("temporal", "steps"), ("spatial", "base_n")):
        rows = [
            r
            for r in report["cases"]
            if r["test_case"] == mode
            and r["levels"] == (4 if mode == "temporal" else 3)
            and r["max_grid_size"] == 32
        ]
        if len(rows) != 3:
            report["failures"].append(f"missing {mode} cases")
            continue
        report["convergence"][mode] = orders(rows, param)
        if not all(r["passed"] for r in report["convergence"][mode]):
            report["failures"].append(f"{mode} second-order gate failed")
    decompositions = [
        r for r in report["cases"] if r["test_case"] == "spatial" and r["base_n"] == 32
    ]
    if (
        len(decompositions) != 2
        or abs(decompositions[0]["l2_error"] - decompositions[1]["l2_error"]) > 1e-10
    ):
        report["failures"].append("decomposition invariance failed")
    report["passed"] = not report["failures"]
    write_json(args.output / "summary.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "convergence", "failures")}, indent=2
        )
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
