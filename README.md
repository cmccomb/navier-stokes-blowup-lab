# OpenAI Navier–Stokes blow-up: numerical experiment

[![CI](https://github.com/cmccomb/navier-stokes-blowup-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/navier-stokes-blowup-lab/actions/workflows/ci.yml)
[![Results site](https://github.com/cmccomb/navier-stokes-blowup-lab/actions/workflows/pages.yml/badge.svg)](https://cmccomb.github.io/navier-stokes-blowup-lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-69d2e7.svg)](LICENSE)

This is a reproducible, proof-informed 3D incompressible-flow experiment based on OpenAI's September 8, 2026 paper, [*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf). It uses [PhiFlow 3.4](https://github.com/tum-pbs/PhiFlow), an established open-source CFD framework, for centered-grid advection, diffusion, and pressure projection.

The experiment starts from rest, forms an inward-spiraling and axially stretching vortex, and approaches the normalized singular time `t*=1`. Version 0.4 replaces the earlier separable vortex with the paper's implicit similarity map, three-region flow structure, stationary exterior swirl, and two resolved annular wave families. Version 0.5 adds an exact discrete Fourier pressure projection for periodic high-resolution runs and an equal-window frontier comparison. Version 0.6 implements Proposition 10.1's stronger zero-data condition: the paper profile and residual force vanish identically through `t=0.55`, transition with a compact-flat `C-infinity` cutoff, and are fully active from `t=0.775`. The solver advances the exact zero state across this interval analytically rather than spending CFD substeps on zero. The PhiFlow layer retains projected RK2 integration, adaptive time steps, a 3D field explorer, restartable checkpoints, force-release controls, and resolution, spectrum, balance, and force-regularity audits.

## Live results and collaboration

The [public results site](https://cmccomb.github.io/navier-stokes-blowup-lab/)
shows only the current best: a verified `288^3` late-window endpoint, the best
available `224^3` time animation in both MP4 and GIF form, and the numerical
checks needed to interpret them. The `288^3` run initializes from the
manufactured target at `t=0.94`; it is the resolution frontier, not the
start-from-rest trajectory.

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[solver reproduction target](REPRODUCTION.md) before proposing an experiment.
The complete experimental record stays in [RESULTS.md](RESULTS.md); large raw
checkpoints remain outside Git.

## What is—and is not—being reproduced

OpenAI's result constructs, for every positive viscosity, a smooth compactly supported force and a smooth finite-energy solution on `0 <= t < 1` whose velocity is unbounded as `t -> 1`. Its leading core obeys

```text
tau = 1 - t
radial length  l_r ~ tau^(1/2)
axial length   l_z ~ tau^(1/2-h)
peak velocity  U   ~ tau^(-1/2-h)
core energy    E   ~ tau^(1/2-3h),    0 < h < 1/100.
```

The default target now solves the paper's implicit coordinate equation at every grid cell:

```text
tau = q(1 - eta^2),    z/base_height = q^(1/2-h) eta,
X = (r/base_radius)^2 / (2q).
```

Fixed profiles in `(X, eta)` generate an inner concentrating core, an annular transition, and a purely azimuthal exterior with the paper's `r^(-1-2h)` decay. A zero radial moment makes the meridional streamfunction vanish outside the annulus. Two complete-ring wave families carry radial-angular and radial-axial quadratic momentum flux; integer angular modes give zero angular mean, `N log(X)` supplies the radial oscillation, and smooth dyadic-time gates mimic the pulse sequence. A slight axial bias keeps the midplane shear nonzero. A centered periodic discrete curl preserves incompressibility to roundoff. PhiFlow advances

```text
du/dt + (u . grad)u - nu Laplacian(u) + grad(p) = f,
div(u) = 0.
```

This is **not an independent numerical proof of blow-up and not yet the paper's exact constructed solution**. The initial rest interval and its temporal localization now follow the paper, but the implemented waves are finite-resolution surrogates rather than a controlled truncation of its infinite pulse hierarchy. The profile-cone argument, exact stress matching, successive `q^(2nh)` corrections, and smooth extension of the residual force through `t=1` remain analytical ingredients rather than solver-ready data. The experiment computes a continuous-time manufactured force from the target's numerical time derivative and PhiFlow's discrete momentum operators, then measures its cost rather than assuming it is benign. See [REPRODUCTION.md](REPRODUCTION.md) for the criterion and staged path to a faithful finite-truncation reproduction.

## Numerical method

- Cell-centered periodic Cartesian grid with exact numerical solution of the implicit `q` map.
- Inner, annular, and fixed-exterior profiles, followed by a smooth compact-support cutoff.
- Two divergence-free, zero-mean annular wave families with independently adjustable covariance directions.
- Second-order centered finite-difference advection and viscous diffusion from PhiFlow.
- Projected midpoint RK2 by default, with a projected Euler option for comparison.
- CFL- and viscosity-limited adaptive substeps, split exactly at a requested force-release time.
- Three pressure paths: PhiFlow's sparse and matrix-free conjugate-gradient solvers, plus an exact periodic FFT Helmholtz projection matched to the centered discrete divergence operator.
- Double precision throughout.
- A four-cell minimum across both shrinking core scales plus an independent Fourier-tail audit; later frames are retained but explicitly marked under-resolved.

Diagnostics include cylindrical component maxima, energy-weighted radial and axial widths, velocity and vorticity maxima, energy, enstrophy, helicity, viscous dissipation, spatial and temporal force-derivative norms, force work, advection/diffusion norms, momentum cancellation, energy-budget residual, Reynolds numbers, divergence, target-tracking error, Fourier occupancy, the accumulated BKM quantity, and grid-scale coverage.

Paper-specific diagnostics additionally track the implicit-coordinate residual, exterior meridional leakage, exterior swirl energy, pulse energy and annular support, coarse-grained angular mean, and pulse covariance. Core widths and core energy exclude the fixed exterior so they can be compared directly with the similarity laws.

## Run it

Requirements: Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run python run.py
```

The default run uses a `32 x 32 x 32` grid, viscosity `nu=0.01`, the paper-compatible exponent `h=0.008`, and stops at `t=0.94`. It intentionally continues past the resolution limit so the audit shows where finite-grid interpretation must stop.

The practical `128^3` refinement profile keeps the core resolved through the same endpoint while suppressing the large full-volume export:

```bash
uv run python run.py \
  --resolution 128 --t-start 0.84 --t-end 0.94 --frames 6 \
  --output outputs/paper-surrogate-n128-late \
  --no-animation --no-3d
```

This late-time window initializes directly from the manufactured target at `t=0.84`; it tests solver tracking in the concentrated regime but does not replay formation from rest. The checked-in run completed through `t=0.94` with 0.439% target-tracking error, `1.18e-10` divergence, 0.151% top-third spectral energy, and 8.15 cells across the radial similarity scale. For the full trajectory, omit `--t-start` and use `outputs/paper-surrogate-n128`. On the tested 16 GB Apple M1 host, a cold benchmark used 1.74 GB peak resident memory with zero swaps. The run is compute-bound because explicit viscosity limits substeps to about `0.00293`; the completed late window took about 49 minutes, while a full start-from-rest run is projected at roughly three hours. Each saved frame writes an atomic `partial-checkpoint.npz`; repeating the same command resumes after the latest frame. Use `--no-resume` only when a clean recomputation is intentional.

Useful variants:

```bash
# Faster smoke run
uv run python run.py --resolution 20 --t-end 0.85 --frames 16 --output outputs/smoke

# Higher resolution with PhiFlow's matrix-free pressure path
uv run python run.py --resolution 48 --t-end 0.90 --frames 30 --output outputs/n48

# High-resolution periodic run with the direct FFT pressure path
uv run python run.py --resolution 192 --t-start 0.94 --t-end 0.99 --frames 11 \
  --pressure-projection fft --output outputs/n192-frontier --no-animation --no-3d

# Paper-localized trajectory: exact rest interval, smooth activation, then concentration
uv run python run.py --resolution 192 --t-start 0 --t-end 0.99 --frames 51 \
  --paper-time-cutoff 0.55 0.775 --pressure-projection fft \
  --output outputs/n192-paper-rest --no-animation --no-3d

# Release the manufactured force and continue freely
uv run python run.py --forcing-end 0.55 --output outputs/released

# Reproduce the v0.3 separable baseline
uv run python run.py --profile-model separable --no-pulses --output outputs/separable

# Isolate the new three-region background without annular waves
uv run python run.py --no-pulses --output outputs/paper-background

# Skip either expensive renderer
uv run python run.py --no-animation
uv run python run.py --no-3d
```

Run the full convergence and force-release study:

```bash
uv run python study.py \
  --resolutions 16 24 32 48 64 \
  --t-end 0.90 --frames 33 \
  --release-time 0.55 \
  --output outputs/study
```

Completed cases are loaded from matching `run.json`, `diagnostics.csv`, and `slices.npz` checkpoints. Incomplete cases resume from the latest atomic per-frame state. Use `--no-resume` to force a clean recomputation. Run the tests with `uv run pytest -q`.

Run the fast, target-level paper-fidelity audit:

```bash
uv run python fidelity.py --resolution 64 --output outputs/paper-fidelity
```

It compares the old separable coordinate proxy with the implicit map, checks that the exterior is stationary at fixed physical radius, measures its power-law slope, and audits annular support, angular mean, covariance, and exterior moment closure. The checked-in audit improves the coordinate-equation residual from `5.97e-1` to `8.88e-16`, while the exterior changes by `0.009%` between `t=0.60` and `0.85`.

After the study, run the validation suite:

```bash
uv run python validation.py --study outputs/study --output outputs/validation
```

This fits resolved similarity exponents, measures force-norm growth, performs a three-step Euler/RK2 temporal refinement test, and cross-checks the geometric cutoff against Fourier-tail energy.

Compare completed equal-window frontier runs:

```bash
uv run python frontier.py \
  outputs/paper-surrogate-n128-frontier-fft \
  outputs/paper-surrogate-n160-frontier \
  outputs/fleet-frontier/n192-frontier-pulses \
  outputs/fleet-frontier/n192-frontier-no-pulses \
  --output outputs/frontier-comparison
```

This writes an endpoint summary and a six-panel comparison of concentration, vorticity, tracking, geometric coverage, spectral occupancy, and spectral-tail energy.

## Outputs

Each standalone run writes:

- `diagnostics.csv`: the full physics and numerical time history.
- `diagnostics.png`: a twelve-panel audit of component scaling, core geometry, force regularity, balances, constraints, and spectral resolution.
- `snapshots.png`: axial and equatorial velocity slices at rest, the last resolved frame, and the final frame.
- `blowup.gif`: animated axial slice.
- `interactive-3d.html`: self-contained Plotly speed isosurfaces and velocity cones with a time slider.
- `slices.npz` and `volumes.npz`: compressed arrays for reproducible follow-on analysis.
- `run.json`: solver, parameters, timestamp, and the scope warning.

The study additionally writes per-case checkpoints, `summary.csv`, `study.json`, and the six-panel `study.png`. The validation suite writes `validation.json`, `temporal-summary.csv`, and `validation.png`. The frontier comparator writes `frontier-summary.csv`, `frontier.json`, and `frontier.png`. See [RESULTS.md](RESULTS.md) for the results already generated in this workspace.

The fidelity audit writes `paper-fidelity.json` and `paper-fidelity.png`. A resolved v0.4 reference run is included under `outputs/paper-surrogate-n48`; it reaches `t=0.85` on a `48^3` grid with 2.78% tracking error, `2.05e-10` divergence, and 1.29% top-third spectral energy. The v0.5 frontier set includes equal-window `128^3`, `160^3`, and `192^3` pulse runs through `t=0.99`, plus a `192^3` pulse-free control. Large volume, animation, and interactive exports were deliberately disabled for these production runs.

## Interpretation guardrails

The informative region ends when either `l_r / dx` or `l_z / dx` drops below four cells. Beyond that point, peak values can oscillate or flatten as the compact core moves between cell centers. That is expected numerical under-resolution, not evidence for or against the analytical theorem.

The earlier resolution sweep shows the cutoff moving toward `t=1` as expected, but it does not show a grid-independent late-time peak. It used the retained `separable` baseline and should not be read as validation of the new annular-wave model. The force-release comparison establishes that the old manufactured concentration relaxes after control is removed; it does not test persistence of the paper's exact smooth-force construction.

## Sources

- OpenAI, [“On the Navier–Stokes Millennium Prize Problem”](https://openai.com/index/navier-stokes-solution/), September 8, 2026.
- OpenAI, [*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf), especially Sections 3–4 for the similarity variables and profiles, Section 6 for annular stress, and Section 7 for oscillatory pulses.
- Holl and Thuerey, [PhiFlow repository and citation](https://github.com/tum-pbs/PhiFlow) (ICML 2024).
