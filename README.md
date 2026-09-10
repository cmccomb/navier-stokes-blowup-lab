# Navier–Stokes Blow-up Lab

[![CI](https://github.com/cmccomb/navier-stokes-blowup-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/navier-stokes-blowup-lab/actions/workflows/ci.yml)
[![Results site](https://github.com/cmccomb/navier-stokes-blowup-lab/actions/workflows/pages.yml/badge.svg)](https://cmccomb.com/navier-stokes-blowup-lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-69d2e7.svg)](LICENSE)

An open, reproducible 3D incompressible-flow experiment informed by OpenAI's
[*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf).
The solver uses [PhiFlow 3.4](https://github.com/tum-pbs/PhiFlow), projected RK2,
and an exact periodic FFT pressure projection.

The experiment starts from rest, smoothly activates a compactly supported
manufactured force, and follows an inward-spiraling, axially stretching vortex
toward the normalized singular time `t*=1`.

> [!IMPORTANT]
> This is a finite surrogate, not an independent proof and not yet the paper's
> exact infinite pulse hierarchy. Numerical instability, overflow, or a
> grid-dependent peak is not evidence of blow-up.

## Current best

The [results site](https://cmccomb.com/navier-stokes-blowup-lab/) publishes the
current-best GIF, MP4, endpoint image, and interpretation checks.

| Late-window endpoint | Value |
|---|---:|
| Grid | `288^3` |
| Final time | `0.995` |
| Peak vorticity | `492.51` |
| Relative target-tracking error | `0.7438%` |
| Top-third spectral energy | `0.4631%` |
| Radial scale coverage | `5.29` cells |
| Divergence `L-infinity` | `1.03e-13` |

The `288^3` run initializes from the manufactured target at `t=0.94`. It is the
resolution frontier, not the full start-from-rest trajectory.

## Quick start

Requirements: Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --no-editable --extra dev
uv run ns-blowup --resolution 20 --t-end 0.85 --frames 16 --output outputs/smoke
uv run pytest -q
```

Results are written under `outputs/`, which is intentionally excluded from Git.
See the [experiment runbook](docs/running.md) for start-from-rest, refinement,
control, validation, and media commands.

## Documentation

- [Documentation index](docs/index.md)
- [Model and numerical method](docs/model.md)
- [Experiment runbook and outputs](docs/running.md)
- [Finite-truncation reproduction target](docs/reproduction.md)
- [Results and validation record](docs/results.md)
- [Contribution guide](CONTRIBUTING.md)

## Contributing

Contributions are welcome in the mathematical model, discretization,
validation, visualization, and documentation. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) and propose experiments with explicit
resolution, convergence, and falsification criteria.

Large checkpoints and raw fleet output stay outside Git. Compact current-best
media and result metadata are published automatically through GitHub Pages.

## Primary sources

- OpenAI, [announcement](https://openai.com/index/navier-stokes-solution/),
  September 8, 2026.
- OpenAI, [*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf).
- Holl and Thuerey, [PhiFlow](https://github.com/tum-pbs/PhiFlow), ICML 2024.
