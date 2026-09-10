"""Build the morning endpoint and full-field comparison after fleet collection."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _variant(path: Path) -> str:
    name = path.name
    if "half-dt" in name:
        return "half timestep"
    if "no-pulses" in name:
        return "no pulses"
    if "strong-pulses" in name:
        return "strong pulses"
    if "force-release" in name:
        return "force release at 0.98"
    if "maximum-safe" in name:
        return "maximum safe resolution"
    if "full-pulse" in name or "frontier-pulses" in name:
        return "full pulses"
    if "main-resume" in name:
        return "full pulses"
    return name


def _find_runs(project: Path) -> list[Path]:
    candidates = [project / "outputs/n192-frontier-pulses"]
    collected = project / "outputs/fleet-overnight-collected"
    if collected.exists():
        candidates.extend(path.parent for path in collected.rglob("run.json"))
    unique = {
        path.resolve()
        for path in candidates
        if (path / "run.json").is_file() and "capacity-probe" not in path.name
    }
    return sorted(unique)


def _endpoint_rows(paths: list[Path]) -> list[dict[str, object]]:
    from navier_stokes_sim.solver import load_simulation_result

    rows: list[dict[str, object]] = []
    for path in paths:
        result = load_simulation_result(path)
        final = result.diagnostics[-1]
        rows.append(
            {
                "case": path.name,
                "variant": _variant(path),
                "path": str(path),
                "resolution": result.config.resolution,
                "t_end": result.times[-1],
                "max_dt": result.config.max_dt,
                "pulses_enabled": result.config.pulses_enabled,
                "pulse_rtheta_strength": result.config.pulse_rtheta_strength,
                "pulse_rz_strength": result.config.pulse_rz_strength,
                "forcing_end": result.config.forcing_end,
                "peak_speed": final["peak_speed"],
                "target_peak_speed": final["target_peak_speed"],
                "peak_vorticity": final["peak_vorticity"],
                "tracking_relative_l2": final["tracking_relative_l2"],
                "divergence_linf": final["divergence_linf"],
                "cells_per_radial_scale": final["cells_per_radial_scale"],
                "cells_per_axial_scale": final["cells_per_axial_scale"],
                "spectral_k95": final["spectral_k95"],
                "spectral_tail_fraction": final["spectral_tail_fraction"],
                "resolved": final["resolved"],
            }
        )
    return rows


def _full_state_rows(paths: list[Path]) -> list[dict[str, object]]:
    reference_path = next(
        (
            path
            for path in paths
            if path.name == "n224-main-resume"
            and (path / "final-state.npz").is_file()
        ),
        None,
    )
    if reference_path is None:
        return []
    with np.load(reference_path / "final-state.npz", allow_pickle=False) as arrays:
        reference_time = float(arrays["time"])
        reference = arrays["velocity"].copy()
    denominator = float(np.linalg.norm(reference.ravel()))
    rows = []
    for path in paths:
        state_path = path / "final-state.npz"
        if path == reference_path or not state_path.is_file():
            continue
        with np.load(state_path, allow_pickle=False) as arrays:
            comparison_time = float(arrays["time"])
            velocity = arrays["velocity"]
            if velocity.shape != reference.shape or not np.isclose(
                comparison_time, reference_time
            ):
                continue
            difference = velocity - reference
            rows.append(
                {
                    "reference": reference_path.name,
                    "case": path.name,
                    "variant": _variant(path),
                    "resolution": reference.shape[0],
                    "time": reference_time,
                    "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
                    "linf": float(np.max(np.abs(difference))),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    endpoint_rows: list[dict[str, object]],
    full_state_rows: list[dict[str, object]],
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    standard_names = {
        "n192-frontier-pulses",
        "n224-main-resume",
        "n240-full-pulse",
    }
    standard = [
        row
        for row in endpoint_rows
        if row["case"] in standard_names or "maximum-safe" in str(row["case"])
    ]
    variants = [row for row in endpoint_rows if int(row["resolution"]) == 224]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    if standard:
        standard.sort(key=lambda row: int(row["resolution"]))
        resolutions = [int(row["resolution"]) for row in standard]
        axes[0, 0].semilogy(
            resolutions,
            [float(row["tracking_relative_l2"]) for row in standard],
            marker="o",
            linewidth=2,
        )
        axes[0, 1].semilogy(
            resolutions,
            [float(row["spectral_tail_fraction"]) for row in standard],
            marker="o",
            linewidth=2,
            color="#7c3aed",
        )
    axes[0, 0].set(title="Endpoint target tracking", xlabel="grid N", ylabel="relative L2")
    axes[0, 1].set(title="Endpoint top-third spectral energy", xlabel="grid N", ylabel="fraction")

    if variants:
        labels = [str(row["variant"]) for row in variants]
        positions = np.arange(len(labels))
        axes[1, 0].bar(
            positions,
            [float(row["tracking_relative_l2"]) for row in variants],
            color="#0891b2",
        )
        axes[1, 0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1, 0].set(title="224³ sensitivity cases", ylabel="endpoint relative L2")

    if full_state_rows:
        labels = [str(row["variant"]) for row in full_state_rows]
        positions = np.arange(len(labels))
        axes[1, 1].bar(
            positions,
            [float(row["relative_l2"]) for row in full_state_rows],
            color="#ea580c",
        )
        axes[1, 1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1, 1].set(title="Full-field difference from 224³ main", ylabel="relative L2")

    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    fig.suptitle("Overnight Navier–Stokes frontier: resolution and controls", fontsize=16)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--wait-until", type=datetime.fromisoformat, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--return-host")
    parser.add_argument("--return-project", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    coordinator_status = (
        project / "outputs/overnight-20260909/coordinator/status.json"
    )
    while datetime.now(args.wait_until.tzinfo) < args.wait_until:
        if coordinator_status.is_file():
            state = json.loads(coordinator_status.read_text(encoding="utf-8"))
            if state.get("state") == "complete":
                break
        time.sleep(30)

    output_dir = project / "outputs/fleet-overnight-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _find_runs(project)
    endpoint_rows = _endpoint_rows(paths)
    full_state_rows = _full_state_rows(paths)
    _write_csv(endpoint_rows, output_dir / "endpoint-summary.csv")
    _write_csv(full_state_rows, output_dir / "full-state-comparisons.csv")
    if endpoint_rows:
        _plot(endpoint_rows, full_state_rows, output_dir / "overnight-comparison.png")
    metadata = {
        "created_at": _timestamp(),
        "completed_runs": len(endpoint_rows),
        "full_state_comparisons": len(full_state_rows),
        "paths": [str(path) for path in paths],
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if args.identity and args.return_host and args.return_project:
        return_dir = args.return_project / "outputs/fleet-overnight-analysis"
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                (
                    f"ssh -i {args.identity.resolve()} "
                    "-o IdentitiesOnly=yes -o BatchMode=yes"
                ),
                str(output_dir) + "/",
                f"{args.return_host}:{return_dir}/",
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
