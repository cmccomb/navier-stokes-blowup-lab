# Solver reproduction target

The target is a solver-resolved sequence of finite truncations of OpenAI's
construction in *Finite Time Blowup for Navier–Stokes*, not merely a flow with
the same headline scaling. A successful numerical reproduction must start with
zero velocity, apply a prescribed smooth compactly supported force, preserve a
uniform kinetic-energy bound, and show velocity growth that strengthens under
both spatial and temporal refinement as `t -> 1`.

## Required fidelity gates

1. **Initial and global geometry.** The computed field is identically zero on
   an initial time interval. Spatial localization is applied to a vector
   potential before curl so incompressibility and fixed compact support survive
   the cutoff. The implicit `(q, eta, X)` coordinates and anisotropic scales are
   used directly.
2. **Leading local field.** Numerically construct the paper's profiles `E`,
   `U`, and `Pi`, including the slightly asymmetric axial datum, the annular
   transition, radial moment constraints, and the exact heat exterior.
3. **Finite all-order hierarchy.** Implement the background corrections through
   order `N` in powers of `q^(2h)`, with each moment constraint and residual
   order checked independently.
4. **Oscillatory stress realization.** Generate both labelled pulse families on
   the auxiliary torus, solve their phase and amplitude equations, apply the
   temporal cutoffs only in exponentially small tails, and verify that their
   averaged covariance matches the prescribed annular stress.
5. **Mean and residual corrections.** Add compactly supported mean corrections
   and iterate residual improvement through the same truncation order `N`.
6. **Convergence evidence.** As grid size, timestep refinement, angular/radial
   pulse frequency, and truncation order increase together, verify decreasing
   solution differences and residual-force derivatives while the resolved peak
   speed grows and kinetic energy stays bounded.

## Current status

- Implemented: exact discrete incompressibility, implicit similarity map,
  inner/annular/exterior decomposition, zero radial-moment closure, fixed
  spatial localization, exact initial rest interval, smooth temporal cutoff,
  two resolved pulse surrogates, prescribed manufactured forcing, RK2/FFT
  projection, and resolution/force/balance diagnostics.
- Next mathematical implementation: replace the hand-shaped leading profiles
  and fixed-strength pulses with the paper's profile ODEs and stress-matched
  finite hierarchy.
- Not a valid success signal: a CFL failure, overflow, NaN, or a peak that moves
  when the timestep or grid changes.

No finite computation can instantiate an actually infinite hierarchy or reach
`t=1`. The reproducible solver claim is therefore convergence of controlled
finite truncations on every resolved interval `t <= 1-epsilon`; the analytical
paper supplies the limiting construction.

Sources: OpenAI, [announcement](https://openai.com/index/navier-stokes-solution/)
and [paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf),
especially Sections 3.5, 5–10, and Proposition 10.1.
