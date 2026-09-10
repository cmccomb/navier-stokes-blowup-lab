# Contributing

Thanks for helping turn the OpenAI Navier–Stokes construction into a more
faithful solver experiment. Contributions are welcome in the mathematical
model, numerical method, validation suite, visualization, and documentation.

## Start here

1. Read [the reproduction target](https://cmccomb.com/navier-stokes-singularity-simulation/reproduction.html). It defines the fidelity gates and
   distinguishes a controlled finite-truncation reproduction from a numerical
   instability.
2. Install Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).
3. Run `uv sync --no-editable --extra dev` and `uv run pytest -q`.
4. Create a focused branch and open a pull request using the repository
   template.

## Good first contributions

- Add independent tests for an existing diagnostic or discrete operator.
- Reproduce a completed case on another machine and report agreement.
- Improve the paper-profile ODE or finite correction hierarchy.
- Add a temporal, spatial, or pulse-frequency refinement comparison.
- Improve accessible result visualization without weakening the caveats.

## Results and large artifacts

The `outputs/` directory is intentionally ignored. Do not commit checkpoints,
full 3D arrays, or raw fleet output. In an issue or pull request, report the
complete command, commit SHA, platform, wall time, peak memory, final
diagnostics, and whether the geometric and spectral resolution gates passed.

To refresh the public record from completed start-from-rest runs:

```bash
uv run python scripts/export_site_results.py \
  --run baseline=outputs/current-best-from-rest \
  --label "baseline=Baseline trajectory" \
  --featured baseline \
  --output site/data/results.json
```

To render publishable media from local slice checkpoints:

```bash
uv run python scripts/render_site_media.py \
  --run outputs/current-best-from-rest \
  --output site/media \
  --stem baseline \
  --endpoint-output site/figures/baseline-endpoint.png \
  --movie-frames 120 --fps 20
```

Commit only compact, representative site media. Put substantially larger
movies and data archives in a GitHub Release and link them from the result
record.

## Scientific claims

Every claim must say which grid, timestep, interval, forcing regime, and
resolution checks support it. A NaN, overflow, CFL failure, or grid-dependent
peak is not evidence of blow-up. This project is not affiliated with OpenAI,
and no numerical run is an independent proof of the theorem.

## Pull-request checks

Before opening a pull request, run:

```bash
uv run ruff check .
uv run pytest -q
```

If the featured start-from-rest set changes, update `site/results.html`, the
compact Pages data, and the representative media in the same pull request. The
public homepage excludes late-window initializations; `site/results.html` keeps the
fuller experimental record.
