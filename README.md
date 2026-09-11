# Navier–Stokes Singularity Simulation

[![CI](https://github.com/cmccomb/navier-stokes-singularity-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/navier-stokes-singularity-simulation/actions/workflows/ci.yml)
[![Results site](https://github.com/cmccomb/navier-stokes-singularity-simulation/actions/workflows/pages.yml/badge.svg)](https://cmccomb.com/navier-stokes-singularity-simulation/)
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
> exact infinite pulse hierarchy or all-order correction cycle. Numerical instability, overflow, or a
> grid-dependent peak is not evidence of blow-up.

## Current best

The [results site](https://cmccomb.com/navier-stokes-singularity-simulation/)
features the current **192³ start-from-rest run while it is still in progress**.
The [stream record](site/data/stream.json) is the authoritative published
configuration and latest saved time; it is not a completed endpoint claim.

| Current streaming configuration | Value |
|---|---:|
| Grid / domain | `192^3`, full `[-1,1]^3` box |
| Planned time interval | `0` to `0.985` |
| Exact rest interval | `0` to `0.55` |
| Profile interpolation | Cubic, source `9d2afb8` |
| Maximum timestep / phase advance | `0.00025` / `0.0375` radians |
| Force-difference half-window | `2e-7` |

The from-rest 32³ cubic-profile temporal pilot recovered a successive-difference
ratio of 3.989, consistent with second-order time convergence at that coarse
resolution. Spatial convergence is not established. This remains a best-guess
finite-surrogate experiment, not a singularity demonstration.

With noninteractive GitHub authentication configured, the publisher detects
finalized phase-clock snapshots every 30 seconds and commits the latest
manifest plus compact rolling GIF/MP4 clips to `main`.
GitHub Pages adds build and cache latency. Each clip uses up to 24 actual
saved frames, with perpendicular velocity/forcing views; full-volume archives
stay outside Git. The manifest labels the display resolution, actual plane
coordinates, color scale, clip times, and the separate diagnostic timestamp.
No fluid states are interpolated. An initial rest-only clip is real data, not
a placeholder for an older run.

GIF/MP4 playback holds each saved state until the next sampled time. The last
six active intervals play in labeled slow motion; earlier clips slow all
available active intervals. Time remains proportional within each segment.
The preceding segment lasts at most 4 seconds, so initial rest does not dominate
the emerging flow. Clips last at most 12 seconds, including a 0.5-second final
pause (4 seconds total for a single state). Replay speed and slowdown factor
are printed in a fixed position and recorded in `stream.json`. Transitions are
rounded to 20 ms boundaries: GIF uses two-centisecond delays; MP4 repeats frames
at 50 fps.
Every saved state remains visible for at least one tick. The manifest records
durations, repeat counts, and the maximum transition-rounding error. Exports
are 1440×792 pixels for sharp 720-pixel display, with fixed label positions.
These timing rules apply to the movies; the interactive 3D explorer steps
through checkpoints independently.

### Saving the full field

Use `--preview-resolution 192` with `--resolution 192` and
`--preview-phase-step 0.3` to save every cell of both three-component fields
at the existing phase-clock events. The historical directory name is
`preview-volumes/`, but these files contain native 192³ data in float32.
This setting changes only the stored copies: solver arithmetic, restart
checkpoints, and the sparse `full-volumes/` snapshots remain float64.
Use a fresh output directory when changing capture settings; configuration
checks deliberately prevent mixing different capture histories.

For the current interval, 487 dense snapshots require about **82.7 GB
(77.0 GiB)** before compression: `487 × 192³ × 6 × 4` bytes. Budget another
4.1 GB for 12 sparse float64 snapshots, checkpoint/temporary-file space,
and a disk reserve. Compression can reduce usage, but capacity planning
should not depend on it. Keep one full-resolution production run active
at a time and retain its archive outside Git. Extract native 192² planes
and reduce browser 3D samples only after the full snapshot is finalized.

To refresh compact media once on the archive's machine, run
`python -m scripts.stream_run --host local --once --no-push --run <run-directory>
--repo <checkout> --cache <receipt-directory> --deadline <ISO-time-with-timezone>`.
The exporter reads one native field at a time, retaining only native 2D planes
and a 32³ browser copy for up to 24 frames. It never rewrites the archive.
Publish the compact outputs from the authorized publishing checkout. Remote
publication rejects dense-archive transfers; raw histories are not rsynced to
another machine or committed to Git.

The separate `288^3` late-window calculation remains the resolution maximum and
is documented in the [full numerical record](https://cmccomb.com/navier-stokes-singularity-simulation/results.html).

## Quick start

Requirements: Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --no-editable --extra dev
uv run ns-blowup --resolution 20 --t-end 0.85 --frames 16 --output outputs/smoke
uv run pytest -q
```

Results are written under `outputs/`, which is intentionally excluded from Git.
See the [experiment runbook](https://cmccomb.com/navier-stokes-singularity-simulation/running.html) for start-from-rest, refinement,
control, validation, and media commands.

## Documentation

- [Documentation index](https://cmccomb.com/navier-stokes-singularity-simulation/documentation.html)
- [Model and numerical method](https://cmccomb.com/navier-stokes-singularity-simulation/method.html)
- [Profile derivation and mesh calculation](https://cmccomb.com/navier-stokes-singularity-simulation/derivation.html)
- [Experiment runbook and outputs](https://cmccomb.com/navier-stokes-singularity-simulation/running.html)
- [Finite-truncation reproduction target](https://cmccomb.com/navier-stokes-singularity-simulation/reproduction.html)
- [Results and validation record](https://cmccomb.com/navier-stokes-singularity-simulation/results.html)
- [Contribution guide](CONTRIBUTING.md)

## Contributing

Contributions are welcome in the mathematical model, discretization,
validation, visualization, and documentation. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) and propose experiments with explicit
resolution, convergence, and falsification criteria.

Large checkpoints and raw fleet output stay outside Git. Compact featured media
and result metadata are published automatically through GitHub Pages.

## Primary sources

- OpenAI, [announcement](https://openai.com/index/navier-stokes-solution/),
  September 8, 2026.
- OpenAI, [*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf).
- Holl and Thuerey, [PhiFlow](https://github.com/tum-pbs/PhiFlow), ICML 2024.
