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

The study, validation, fidelity, and frontier commands write compact JSON/CSV
summaries and comparison figures into their requested output directories.

`outputs/` is ignored by Git. Do not commit full arrays or raw run directories.
Follow [CONTRIBUTING.md](../CONTRIBUTING.md) when publishing a new current-best
result.

## Verification

```bash
uv run ruff check .
uv run pytest -q
```

A result is interpretable only while both similarity scales span at least four
cells and the Fourier-tail audit remains acceptable. A numerical failure is not
a positive result.
