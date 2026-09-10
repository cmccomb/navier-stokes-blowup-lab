# Model and numerical method

## Analytical target

OpenAI's result constructs, for every positive viscosity, a smooth compactly
supported force and a smooth finite-energy solution on `0 <= t < 1` whose
velocity is unbounded as `t -> 1`. Its leading core obeys

```text
tau = 1 - t
radial length  l_r ~ tau^(1/2)
axial length   l_z ~ tau^(1/2-h)
peak velocity  U   ~ tau^(-1/2-h)
core energy    E   ~ tau^(1/2-3h),    0 < h < 1/100.
```

The numerical target solves the paper's implicit coordinate equation at every
grid cell:

```text
tau = q(1 - eta^2),    z/base_height = q^(1/2-h) eta,
X = (r/base_radius)^2 / (2q).
```

Fixed profiles in `(X, eta)` generate an inner concentrating core, an annular
transition, and a purely azimuthal exterior with the paper's `r^(-1-2h)` decay.
A zero radial moment makes the meridional streamfunction vanish outside the
annulus. A slight axial bias keeps the midplane shear nonzero.

Two complete-ring wave families carry radial-angular and radial-axial quadratic
momentum flux. Each family now retains three staggered scales by default:
successive levels use higher integer angular modes, faster radial and axial
phases, smaller amplitudes, and shifted smooth dyadic-time gates. Integer modes
give zero angular mean. A centered periodic discrete curl preserves
incompressibility to roundoff.

A bounded deconvolution pass on the pulse vector potential pre-emphasizes
features attenuated by the centered grid. Because the correction is applied to
the potential before taking its curl, the corrected velocity remains exactly
discretely divergence-free. This is a numerical leading corrector for retained
grid scales; it is not the paper's all-order analytical correction cycle.

## Governing equations

PhiFlow advances

```text
du/dt + (u . grad)u - nu Laplacian(u) + grad(p) = f,
div(u) = 0.
```

The experiment computes a continuous-time manufactured force from the target's
numerical time derivative and the same discrete momentum operators used by the
solver. It measures the force and its derivatives rather than assuming they are
benign.

## Discretization

- Cell-centered periodic Cartesian grid with numerical solution of the implicit
  `q` map.
- Inner, annular, and fixed-exterior profiles followed by a smooth compact
  support cutoff.
- Two divergence-free, zero-mean annular wave families, each with a configurable
  finite scale hierarchy and independently adjustable covariance directions.
- A configurable bounded grid-deconvolution corrector applied at the vector-
  potential level.
- Second-order centered finite-difference advection and viscous diffusion from
  PhiFlow.
- Projected midpoint RK2 by default, with projected Euler for comparison.
- CFL- and viscosity-limited adaptive substeps, split exactly at a requested
  force-release time.
- PhiFlow sparse and matrix-free conjugate-gradient pressure paths plus an exact
  periodic FFT Helmholtz projection matched to the centered discrete divergence.
- Double precision throughout.

Version 0.6 enforces the paper's stronger zero-data condition. The velocity and
residual force vanish identically through `t=0.55`, transition through a
compact-flat smooth cutoff, and are fully active from `t=0.775`. The solver
advances the exact zero interval analytically.

## Diagnostics

Core diagnostics include cylindrical component maxima, energy-weighted radial
and axial widths, velocity and vorticity maxima, energy, enstrophy, helicity,
viscous dissipation, force derivative norms, force work, momentum-term norms,
cancellation, energy-budget residual, Reynolds numbers, divergence, target
tracking, Fourier occupancy, the accumulated BKM quantity, and grid coverage.

Paper-specific diagnostics include the implicit-coordinate residual, exterior
meridional leakage, exterior swirl energy, pulse energy and annular support,
coarse-grained angular mean, pulse covariance, retained hierarchy depth, and
corrector energy. Core widths and energy exclude the fixed exterior so they can
be compared directly with the similarity laws.

## Interpretation boundary

This is not an independent numerical proof and not yet the paper's exact
constructed solution. The initial rest interval and temporal localization
follow the paper, but the implemented waves remain finite-resolution
surrogates. The profile-cone argument, exact stress matching, successive
`q^(2nh)` corrections, and smooth extension of the residual force through
`t=1` remain analytical ingredients rather than solver-ready data.

The informative region ends when either `l_r / dx` or `l_z / dx` falls below
four cells. Beyond that point, peak values can oscillate or flatten as the core
moves between cell centers. That is numerical under-resolution, not evidence
for or against the theorem. The independent Fourier-tail audit must agree with
the geometric check before an endpoint is called resolved.

See the [reproduction target](reproduction.md) for the staged path from the
current manufactured model to a controlled finite-truncation reproduction.

## Sources

- OpenAI, [“On the Navier–Stokes Millennium Prize Problem”](https://openai.com/index/navier-stokes-solution/),
  September 8, 2026.
- OpenAI, [*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf),
  especially Sections 3–7 and 10.
- Holl and Thuerey, [PhiFlow](https://github.com/tum-pbs/PhiFlow), ICML 2024.
