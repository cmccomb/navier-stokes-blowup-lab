# Results from the expanded numerical study

## Current canonical run: `192^3` from rest

The public result is one continuous `192^3` trajectory from exact rest at
`t=0` through `t=0.99`. The velocity and manufactured force vanish identically
through `t=0.55`, transition with the compact-flat temporal cutoff, and are
fully active from `t=0.775`. All 51 requested frames completed.

| Endpoint diagnostic at `t=0.99` | Result |
|---|---:|
| Peak speed, solver / target | 2.83834 / 2.75790 |
| Relative target-tracking error | 1.5661% |
| Divergence `L-infinity` | `4.44e-14` |
| Peak vorticity | 215.52 |
| Total / core kinetic energy | 0.007288 / 0.000204 |
| Radial / axial scale coverage | 4.99 / 5.18 cells |
| 95%-energy Fourier mode | 56.87 |
| Top-third spectral energy | 0.7301% |
| Manufactured-force `L2` norm | 28.48 |
| Geometric resolution gate | Pass |

This is the most complete run because formation, activation, and late-time
concentration occur within one solver trajectory. It is the sole run displayed
on the [results site](https://cmccomb.com/navier-stokes-blowup-lab/). The force
norm still grows, and the finite wave model still omits the analytical infinite
pulse hierarchy and all-order corrections.

## Version 0.6 resolution maximum: `288^3`

The current maximum verified late-window calculation advances the
paper-surrogate target from `t=0.94` through `t=0.995` on a uniform `288^3`
grid. It is a concentrated-regime resolution test initialized from the target,
not the full start-from-rest trajectory.

| Endpoint diagnostic at `t=0.995` | Result |
|---|---:|
| Peak speed, solver / target | 4.54551 / 4.49267 |
| Relative target-tracking error | 0.7438% |
| Divergence `L-infinity` | `1.03e-13` |
| Peak vorticity | 492.51 |
| Radial / axial scale coverage | 5.29 / 5.52 cells |
| 95%-energy Fourier mode | 78.04 |
| Top-third spectral energy | 0.4631% |
| Geometric resolution gate | Pass |

Independent `224^3` and `240^3` half-timestep runs differ from their standard
timestep counterparts by 1.095% and 0.743% in the full terminal velocity field.
Force-release and pulse-ablation controls separate maintained concentration
from free relaxation and test the finite wave surrogate. These checks improve
confidence in the finite-grid trajectory, but they do not supply the paper's
missing infinite pulse hierarchy or prove grid-independent blow-up.

The full fleet comparison remains in `outputs/fleet-overnight-analysis` locally.

## Version 0.5 high-resolution frontier

Version 0.5 replaces iterative pressure solves in the largest periodic runs with a direct FFT Helmholtz projection matched to the same centered discrete derivative used by the solver diagnostics. A small-grid trajectory comparison differs from PhiFlow's sparse conjugate-gradient path by only `4.93e-8` in relative `L2`, while reducing projected divergence from `5.27e-11` to machine precision.

Three full-pulse runs use the same manufactured-target initialization at `t=0.94`, the same RK2 evolution, and the same endpoint at `t=0.99`:

| Grid | Radial cells | Tracking error | Spectral `k95` / top-third | Top-third energy | Peak vorticity | Resolved? |
|---:|---:|---:|---:|---:|---:|:---:|
| `128^3` | 3.33 | 3.760% | 48.96 / 42.67 | 3.042% | 122.90 | No |
| `160^3` | 4.16 | 2.122% | 53.38 / 53.33 | 1.379% | 166.43 | Marginal |
| `192^3` | 4.99 | 1.431% | 56.87 / 64.00 | 0.714% | 217.78 | Yes |

The endpoint tracking error decreases with observed order 2.39 across the three grids. Top-third spectral energy decreases with order 3.57. The `160^3` endpoint passes the four-cell geometric rule but places its 95%-energy mode just beyond the top-third threshold; the `192^3` refinement clears both criteria. This agreement between independent geometric and Fourier audits is stronger evidence than either threshold alone.

A `192^3` pulse-free control finishes with 1.252% tracking error, 0.708% top-third energy, and peak vorticity 212.18. The close full-pulse and pulse-free aggregate errors show that the grid resolves the background at comparable quality; the difference between their fields remains available for the annular-wave ablation.

All four tested Macs have eight CPU cores and 16 GB unified memory. A `192^3`
one-step benchmark used 5.71 GB peak resident memory. Local probes completed at
`208^3` and `224^3` with 3.93 GB and 5.30 GB peak resident memory, respectively,
and zero process swaps. Later scheduling on an otherwise available node
completed the `288^3` late-window trajectory, establishing a higher empirical
frontier than the first local probes suggested. `192^3` remains the conservative
unattended size on a machine that must also support interactive work.

Multi-host spatial decomposition was rejected after measurement rather than
assumption. Peer transfer ranged from about 1 to 4.5 MB/s, with 12 ms average
latency even on the faster link. A distributed 3D FFT would exchange hundreds
of megabytes multiple times per timestep, so communication would dominate. The
fleet was instead used for independent causal trajectories, with compact
outputs collected afterward.

The comparison artifacts are in `outputs/frontier-comparison`; the full remote results are collected under `outputs/fleet-frontier`.

## Version 0.4 paper-surrogate result

The current default model replaces the separable leading vortex with the paper's implicit `q, eta, X` coordinates, inner/annular/exterior regions, zero-moment meridional closure, and two resolved annular wave families. At `64^3`, the target-level fidelity audit gives:

| Paper-specific check | Result |
|---|---:|
| Old separable coordinate residual | 0.59696 |
| Implicit-coordinate residual | `8.88e-16` |
| Coordinate accuracy improvement | `6.72e14 x` |
| Exterior change, `t=0.60` to `0.85` | 0.0086% |
| Exterior radial exponent | -1.002 measured, -1.016 target |
| Exterior meridional energy fraction | `3.63e-8` |
| Pulse energy inside the strict annulus | 98.30% |

A resolved `48^3` PhiFlow run through `t=0.85` finishes with 2.78% relative target-tracking error, divergence `L-infinity = 2.05e-10`, 1.29% top-third spectral energy, and 4.83 cells per radial similarity scale. The manufactured force and its derivatives still grow. This is expected because the implementation does not reproduce the exact stress matching, infinite pulse hierarchy, or all-order corrections that make the theorem's force smooth through `t=1`.

Post-ramp fits over the resolved trajectory recover the core geometry directly: the radial and axial RMS-width exponents are 0.5028 and 0.4961, compared with paper values 0.500 and 0.492. The target peak-speed exponent is -0.5537 versus -0.508, the core-energy exponent is 0.5267 versus 0.476, and the peak-vorticity exponent is -1.1316 versus -1.008. The finite pulse gates produce visible local departures from a single power law; the fits are therefore reported with, rather than used to hide, their oscillations.

The artifacts are in `outputs/paper-fidelity` and `outputs/paper-surrogate-n48`.

## `128^3` late-time refinement

The v0.4.1 production refinement advances the paper-surrogate target from `t=0.84` through `t=0.94` on a uniform `128^3` grid. It initializes directly from the manufactured target at `t=0.84`, so it is a stringent concentrated-regime tracking test rather than a start-from-rest formation history.

| Endpoint diagnostic at `t=0.94` | Result |
|---|---:|
| Peak speed, solver / target | 0.82138 / 0.81707 |
| Relative target-tracking error | 0.4387% |
| Divergence `L-infinity` | `1.18e-10` |
| Peak vorticity | 37.4465 |
| Radial / axial scale coverage | 8.15 / 8.34 cells |
| 95%-energy Fourier mode / top-third threshold | 28.23 / 42.67 |
| Top-third spectral energy | 0.1506% |
| Pulse energy inside the strict annulus | 97.91% |
| Exterior meridional energy fraction | `1.66e-8` |
| Manufactured-force `L2` norm | 6.1664 |

Across the short late-time window, the measured radial- and axial-width exponents are 0.5004 and 0.4946, compared with target values 0.500 and 0.492. The target core-energy exponent is 0.5027 versus 0.476. The interval is too narrow and too strongly modulated by the finite pulse gates for its pointwise peak-speed fit to be a meaningful global similarity estimate.

This refinement improves the numerical margin substantially despite reaching closer to `t=1`: it ends with 8.15 radial cells per similarity scale and 0.151% top-third spectral energy, compared with 4.83 cells and 1.29% for the earlier `48^3` endpoint at `t=0.85`. The comparison is a resolution audit, not a convergence rate, because the time intervals and initializations differ.

The remaining limitation is visible rather than hidden: the manufactured-force `L2` norm grows from 2.19 at `t=0.84` to 6.17 at `t=0.94`. The run therefore approximates the paper's concentrating velocity geometry and annular organization much more sharply, but still does not reproduce the analytical cancellations that keep the exact constructed force smooth.

On the tested 16 GB Apple M1 host, a cold `128^3` benchmark used 1.74 GB peak resident memory with zero swaps. The completed six-frame late window took about 49 minutes. Atomic per-frame checkpoints now make longer runs restartable; a full start-from-rest trajectory is projected at roughly three hours. The artifacts are in `outputs/paper-surrogate-n128-late`.

## Version 0.3 separable-baseline study

The earlier generated output set uses projected RK2, `nu=0.01`, `h=0.008`, 33 saved frames through `t=0.9`, and five grids from `16^3` to `64^3`. The finest-grid control turns off the manufactured force at `t=0.55`. These results use the retained `separable` profile and provide a reproducible baseline, not validation of the v0.4 paper surrogate.

## Resolution study

| Grid | Last resolved time | Final peak speed | Final peak vorticity | Final relative tracking error |
|---:|---:|---:|---:|---:|
| `16^3` | 0.0563 | 0.3760 | 4.4974 | 0.573% |
| `24^3` | 0.5625 | 0.6463 | 12.3565 | 0.584% |
| `32^3` | 0.7594 | 0.9223 | 18.2618 | 0.946% |
| `48^3` | 0.8719 | 1.1372 | 22.0400 | 0.640% |
| `64^3` | 0.9000 | 1.3050 | 25.1414 | 0.353% |

The observed cutoffs follow the radial-scale prediction to within one saved-frame interval. At `64^3`, the final core spans 5.26 radial and 5.36 axial cells, so `t=0.9` is above the configured four-cell threshold. The independent spectral audit agrees: the 95%-energy mode is 16.03, below the top-third threshold of 21.33, and the top-third contains 0.697% of kinetic energy.

This is the first study in the project whose finest-grid endpoint remains inside its declared geometric resolution envelope. It is still a modest uniform-grid calculation, not an asymptotic convergence proof.

## Resolved force-release control

At `64^3` and `t=0.9`, both trajectories pass the geometric threshold:

| Quantity | Forced target | Force released at 0.55 | Ratio, forced/released |
|---|---:|---:|---:|
| Peak speed | 1.3050 | 0.3128 | 4.17 |
| Peak radial speed | 0.2673 | 0.1163 | 2.30 |
| Peak azimuthal speed | 1.2862 | 0.2962 | 4.34 |
| Peak axial speed | 0.5267 | 0.2008 | 2.62 |
| Peak vorticity | 25.1414 | 3.3325 | 7.54 |
| Kinetic energy | 0.005976 | 0.005491 | 1.09 |
| Enstrophy | 3.4154 | 0.5332 | 6.41 |

The comparable total energies alongside strongly separated maxima show that the manufactured forcing primarily sustains spatial concentration. The released field's relative target error reaches 125%, while the forced reference remains at 0.35%. Its top-third spectral energy falls to `8.0e-12`, compared with 0.697% for the concentrating reference.

## Similarity-exponent audit

Power laws were fitted as `quantity ~ tau^p` over all post-ramp, resolved `64^3` frames from `t=0.16875` through `t=0.9`.

| Quantity | Paper/target exponent | Measured exponent | R-squared |
|---|---:|---:|---:|
| Radial RMS width | 0.500 | 0.507 | 0.999997 |
| Axial RMS width | 0.492 | 0.499 | 0.999984 |
| Azimuthal peak speed | -0.508 | -0.457 | 0.998933 |
| Axial peak speed | -0.508 | -0.474 | 0.999111 |
| Radial peak speed | -0.500 | -0.325 | 0.962472 |
| Target kinetic energy | 0.476 | 0.553 | 0.999037 |
| Peak vorticity | -1.008 | -0.592 | 0.992290 |

The core geometry is reproduced exceptionally closely, and the dominant azimuthal and axial speed exponents are reasonably close. Pointwise radial and vorticity maxima remain more sensitive to cell-center sampling and discrete derivatives. Those discrepancies are retained as validation findings, not hidden by fitting only a favorable subinterval.

## Force-regularity stress test

The analytical theorem's difficult step is making the complete momentum residual smooth even while its individual terms diverge. This simplified manufactured leading field does not do that:

| Norm | Fitted exponent `p` | R-squared |
|---|---:|---:|
| `||f|| L2` | -0.512 | 0.998747 |
| `||grad f|| L2` | -0.629 | 0.999240 |
| `||Laplacian f|| L2` | -0.759 | 0.998597 |
| `||d_t f|| L2` | -1.065 | 0.998228 |

All four negative exponents indicate growth as `tau -> 0`. This directly quantifies why the simulation is a leading-order manufactured model rather than the paper's exact construction: the omitted annular pulse hierarchy and all-order corrections are precisely what cancel these singular residuals.

## Temporal verification

At `24^3` through `t=0.5`, five maximum timesteps from `0.05` to `0.003125` give these fitted orders over the three finest steps:

- Projected Euler: 1.013.
- Projected midpoint RK2: 2.176.

This confirms the intended first- and second-order time integration regimes. The RK2 value is reported as a finite-range fit, not as a claim of accuracy above second order.

## Numerical checks

- The resolved `64^3` forced endpoint has divergence `L-infinity = 7.09e-11` and tracking error 0.35%.
- The `48^3` endpoint lies just below the geometric cutoff at 3.95 radial cells and has 2.06% top-third spectral energy; the `64^3` refinement reduces that fraction to 0.697%.
- The default interactive `32^3` run through `t=0.94` remains intentionally under-resolved at its endpoint and is retained for quick exploration.
- Seven automated tests cover rest initial data, discrete incompressibility, divergence-of-curl consistency, paper scaling, cylindrical/width diagnostics, solver output, checkpoint round-tripping, and power-law fitting.

These are diagnostics of a manufactured leading-order flow, not a numerical verification of the OpenAI theorem.
