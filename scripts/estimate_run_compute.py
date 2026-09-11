"""Publish a transparent planning estimate, not a measured hardware FLOP count."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.snapshot_store import preview_times
from navier_stokes_sim.time_stepping import forcing_step_limit

ROOT = Path(__file__).resolve().parents[1]


def estimate(cfg: SimulationConfig) -> dict:
    """Count clock-limited RK2 steps; future velocity-dependent CFL is unknown."""
    if cfg.integrator != "rk2" or cfg.pressure_projection != "fft":
        raise ValueError("This operation model requires RK2 with FFT projection")
    if cfg.profile_model != "paper-surrogate" or cfg.forcing_end is not None:
        raise ValueError("This clock estimate assumes the forced surrogate run")
    if cfg.frame_spacing == "similarity":
        diagnostic_times = cfg.t_star - np.exp(
            np.linspace(
                np.log(cfg.t_star - cfg.t_start),
                np.log(cfg.t_star - cfg.t_end),
                cfg.frames,
            )
        )
    else:
        diagnostic_times = np.linspace(cfg.t_start, cfg.t_end, cfg.frames)
    diagnostic_times[0], diagnostic_times[-1] = cfg.t_start, cfg.t_end
    events = np.unique(np.concatenate((diagnostic_times, preview_times(cfg))))
    t, steps = cfg.t_start, 0
    for event in events:
        while t < event - 1e-13:
            if t < cfg.paper_time_cutoff_start - 1e-13:
                t = min(event, cfg.paper_time_cutoff_start)
                continue
            t += min(
                cfg.max_dt,
                0.12 * cfg.dx**2 / cfg.viscosity,
                forcing_step_limit(t, cfg),
                event - t,
            )
            steps += 1
    cells = cfg.resolution**3
    # Two projections, each with 3 forward + 3 inverse complex transforms.
    fft_work = steps * 12 * 5 * cells * math.log2(cells)
    low, high = 3 * fft_work / 1e12, 10 * fft_work / 1e12
    low_unit, high_unit = (
        10 ** math.floor(math.log10(low)),
        10 ** math.floor(math.log10(high)),
    )
    display_low = int(math.floor(low / low_unit) * low_unit)
    display_high = int(math.ceil(high / high_unit) * high_unit)
    return {
        "clock_limited_steps": steps,
        "complex_ffts_per_step": 12,
        "fft_equivalent_flop": fft_work,
        "assumed_total_to_fft_multiplier": [3, 10],
        "estimated_total_flop": [low * 1e12, high * 1e12],
        "display_range_tflop": f"{display_low:,}–{display_high:,}",
        "method": "5 N log2(N) per complex FFT; 3–10x planning allowance for other kernels",
        "limitations": "Not profiled or a bound. Excludes extra CFL-limited steps, restarts, validation runs and media rendering.",
    }


def main() -> None:
    run = json.loads((ROOT / "site/data/stream.json").read_text())
    result = {
        "source_commit": run["source_commit"],
        "config": run["config"],
        **estimate(SimulationConfig(**run["config"])),
    }
    (ROOT / "site/data/compute-estimate.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
