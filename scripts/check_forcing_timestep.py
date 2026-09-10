"""From-rest timestep refinement with the force-difference window held fixed."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.profile import discrete_curl
from navier_stokes_sim.solver import run_simulation
from navier_stokes_sim.time_stepping import forcing_step_limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--t-end", type=float, default=0.985)
    parser.add_argument(
        "--profile-interpolation", choices=("linear", "cubic"), default="cubic"
    )
    parser.add_argument("--derivative-epsilon", type=float, default=1e-6)
    parser.add_argument("--no-pulses", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig(
        resolution=args.resolution,
        t_end=args.t_end,
        frames=81,
        frame_spacing="similarity",
        pressure_projection="fft",
        cfl=0.2,
        max_dt=0.001,
        derivative_epsilon=args.derivative_epsilon,
        profile_interpolation=args.profile_interpolation,
        pulses_enabled=not args.no_pulses,
        capture_volumes=False,
    )
    fields, rows = [], []
    for level in range(3):
        cfg = replace(
            base, max_dt=base.max_dt / 2**level, forcing_phase_step=0.15 / 2**level
        )
        if (
            min(
                0.2 * (cfg.t_star - cfg.t_end), 0.1 * forcing_step_limit(cfg.t_end, cfg)
            )
            < cfg.derivative_epsilon
        ):
            raise ValueError("force derivative window would change; reduce epsilon")
        output = args.output / f"level-{level}"
        started = time.monotonic()
        print(
            json.dumps({"event": "start", "level": level, "config": cfg.to_dict()}),
            flush=True,
        )
        result = run_simulation(cfg, output, save_final_state=True)
        with np.load(output / "final-state.npz") as arrays:
            fields.append(arrays["velocity"])
        row = {
            "level": level,
            "elapsed_seconds": time.monotonic() - started,
            "max_dt": cfg.max_dt,
            "forcing_phase_step": cfg.forcing_phase_step,
            "derivative_epsilon": cfg.derivative_epsilon,
            "tracking_relative_l2": result.diagnostics[-1]["tracking_relative_l2"],
        }
        rows.append(row)
        print(json.dumps({"event": "complete", **row}), flush=True)
    denominator = max(float(np.linalg.norm(fields[-1])), np.finfo(float).tiny)
    differences = [
        float(np.linalg.norm(a - b)) / denominator for a, b in pairwise(fields)
    ]
    curls = [discrete_curl(v, base.dx) for v in fields]
    curl_scale = max(float(np.linalg.norm(curls[-1])), np.finfo(float).tiny)
    curl_differences = [
        float(np.linalg.norm(a - b)) / curl_scale for a, b in pairwise(curls)
    ]
    ratio = differences[0] / differences[1] if differences[1] else None
    summary = {
        "scope": "from-rest temporal check on a spatially underresolved grid; not spatial convergence or a singularity proof",
        "resolution": base.resolution,
        "t_start": 0,
        "t_end": base.t_end,
        "profile_interpolation": base.profile_interpolation,
        "pulses_enabled": base.pulses_enabled,
        "runs": rows,
        "relative_velocity_differences": differences,
        "relative_vorticity_differences": curl_differences,
        "successive_difference_ratio": ratio,
        "pilot_gate": bool(
            ratio is not None and 2.5 < ratio < 6 and differences[-1] < 1e-3
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
