# Experiment runbook

## Install

Requirements: Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --no-editable --extra dev
uv run ns-blowup --help
```

The default command uses a `32^3` grid, viscosity `nu=0.01`, exponent `h=0.008`,
and stops at `t=0.94`. It intentionally continues beyond the four-cell limit so
the resolution audit demonstrates where interpretation must stop.

```bash
uv run ns-blowup
```

## Recommended runs

Fast smoke run:

```bash
uv run ns-blowup \
  --resolution 20 --t-end 0.85 --frames 16 \
  --output outputs/smoke
```

Paper-localized trajectory from exact rest:

```bash
uv run ns-blowup \
  --resolution 192 --t-start 0 --t-end 0.99 --frames 51 \
  --paper-time-cutoff 0.55 0.775 --pressure-projection fft \
  --output outputs/n192-paper-rest --no-animation --no-3d
```

Publication-quality temporal refinement from exact rest:

```bash
uv run ns-blowup \
  --resolution 192 --t-start 0 --t-end 0.99 \
  --frames 121 --frame-spacing similarity \
  --cfl 0.20 --max-dt 0.00065 \
  --paper-time-cutoff 0.55 0.775 --pressure-projection fft \
  --output outputs/n192-paper-rest-fine --no-animation --no-3d
```

Similarity-time frame spacing does not invent intermediate solver states. It
saves more real checkpoints near `t*=1`, where the physical evolution
accelerates. The media renderer may additionally interpolate playback between
saved checkpoints and labels that interpolation in the resulting figure.

Late-window refinement initialized from the manufactured target:

```bash
uv run ns-blowup \
  --resolution 192 --t-start 0.94 --t-end 0.99 --frames 11 \
  --pressure-projection fft --output outputs/n192-frontier \
  --no-animation --no-3d
```

Force-release and pulse-ablation controls:

```bash
uv run ns-blowup --forcing-end 0.55 --output outputs/released
uv run ns-blowup --no-pulses --output outputs/no-pulses
```

The default pulse surrogate retains three scales and one bounded grid-
correction pass. For finite-hierarchy studies, use `--pulse-levels`,
`--pulse-scale-ratio`, `--pulse-correction-strength`, and
`--pulse-correction-passes`.

Each saved frame writes an atomic checkpoint. Repeating the same command resumes
matching work. Use `--no-resume` only when a clean recomputation is intentional.

## Studies and validation

Run the convergence and force-release study:

```bash
uv run ns-study \
  --resolutions 16 24 32 48 64 \
  --t-end 0.90 --frames 33 --release-time 0.55 \
  --output outputs/study
```

Run the target-level paper-fidelity audit:

```bash
uv run ns-fidelity --resolution 64 --output outputs/paper-fidelity
```

Validate the study and compare equal-window frontier runs:

```bash
uv run ns-validation --study outputs/study --output outputs/validation

uv run ns-frontier \
  outputs/paper-surrogate-n128-frontier-fft \
  outputs/paper-surrogate-n160-frontier \
  outputs/n192-frontier \
  --output outputs/frontier-comparison
```

## Output contract

A standalone run writes:

- `run.json`: solver version, parameters, timestamp, and scope warning.
- `diagnostics.csv`: complete physics and numerical time history.
- `diagnostics.png`: balance, force, constraint, and resolution audit.
- `snapshots.png`: axial and equatorial velocity slices.
- `blowup.gif`: axial-slice animation, unless disabled.
- `interactive-3d.html`: self-contained Plotly field explorer, unless disabled.
- `slices.npz` and optional `volumes.npz`: compressed follow-on arrays.
- `partial-checkpoint.npz` and optional `final-state.npz`: restart and comparison
state.

Direction-aware GIFs can be rendered from any completed checkpoint without
re-running the solver:

```bash
uv run python -m scripts.render_flow_gifs \
  --run outputs/n192-paper-rest --output artifacts --stem best
```

This writes separate axial-jet, equatorial-swirl, and pulse-forcing GIFs. None
of the three views uses a target-comparison panel.

The study, validation, fidelity, and frontier commands write compact JSON/CSV
summaries and comparison figures into their requested output directories.

`outputs/` is ignored by Git. Do not commit full arrays or raw run directories.
Follow [CONTRIBUTING.md](../CONTRIBUTING.md) when publishing a new featured
start-from-rest result.

## Verification

```bash
uv run ruff check .
uv run pytest -q
```

A result is interpretable only while both similarity scales span at least four
cells and the Fourier-tail audit remains acceptable. A numerical failure is not
a positive result.
