# Static 3D refinement pilot

This backend validates the pressure-projection operator needed for full-domain,
fixed local refinement. It uses the established AMReX-Hydro MAC projector and
AMReX multilevel multigrid, with double precision and immutable upstream source
pins in `CMakeLists.txt`. It does **not** yet integrate Navier–Stokes, apply the
project's force, or produce a new flow animation.

The [first-class documentation page](https://cmccomb.com/navier-stokes-singularity-simulation/refinement.html)
contains the mesh derivation, measured results, limitations, and next gates.
The [machine-readable record](../../site/data/refinement-pilot.json) retains all
nine cases, source/binary hashes, errors, timings, and peak process RSS.

## Build and reproduce

Requires Python 3.11+, a C++20 compiler, and network access for the two pinned
source archives. The pilot runner uses only Python's standard library. No
PhiFlow, MPI, GPU, system-wide installation, or fleet service is required.

```sh
python3 -m venv .venv-amrex
.venv-amrex/bin/python -m pip install -r backends/amrex/requirements-build.txt
.venv-amrex/bin/cmake -S backends/amrex -B build/amrex -DCMAKE_BUILD_TYPE=Release
.venv-amrex/bin/cmake --build build/amrex --parallel 2
.venv-amrex/bin/ctest --test-dir build/amrex --output-on-failure
.venv-amrex/bin/python -m scripts.refinement_pilot \
  --levels 4 --resolutions 32 64 128 --output outputs/projection-32-128
```

Use a fresh output directory. The full nine-case suite checks exact rest,
preservation of an analytic solenoidal field, uniform and nested-grid spatial
convergence, and box-decomposition invariance. Each process is capped at ten
minutes. Output and source identities are checked before computing convergence;
any missing case or failed gate produces exit code 2. Partial summaries and raw
process logs remain available after failures. The output directory must be empty
to avoid overwriting a previous experiment.

The smaller `--resolutions 16 32 64` suite is the default. `--levels` can be 2, 3,
or 4; each suite includes a one-level control. For one operator test:

```sh
build/amrex/ns_projection_pilot n_cell=128 levels=4 test_case=mixed
```

The largest recorded case has local 1024³-equivalent spacing only within
`[-0.125,0.125]³`, with a 128³ base grid across the full periodic `[-1,1]³`
domain. The 1.55 GiB measured peak RSS is for this operator pilot, including
two projections and diagnostics, **not** a full NS memory budget. It must not
be compared directly with a macOS physical-footprint measurement.

## Launch boundary

There is deliberately no production-run option. Before a refined from-rest
trajectory can replace the existing 192³ result, implement and verify the
multilevel momentum/viscous update, ghost-cell interpolation and synchronization,
force evaluation, and multilevel archive/export path. Audit the entire active
support and all relevant phase gradients, then separate spatial, temporal, and
force-difference convergence. The pilot's convenient nested cubes are not yet
an accepted forcing-informed production mesh.

CI builds the actual C++ backend and runs the four-level convergence suite when
backend-related files change. The Python unit tests alone do not validate the
numerical operator.
