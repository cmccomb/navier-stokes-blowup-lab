"""PhiFlow-backed incompressible Navier–Stokes experiment."""

from __future__ import annotations

import csv
import json
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from phi.flow import (
    Box,
    CenteredGrid,
    Solve,
    advect,
    channel,
    diffuse,
    extrapolation,
    field,
    fluid,
    math,
    spatial,
)
from scipy import fft

from .config import SimulationConfig
from .profile import (
    axial_outflow_diagnostics,
    cylindrical_components,
    discrete_curl,
    energy_weighted_core_widths,
    paper_similarity_coordinates,
    paper_structure_diagnostics,
    similarity_scales,
    target_velocity,
    temporal_activation,
)

DIAGNOSTIC_SCHEMA_VERSION = 6


@dataclass
class SimulationResult:
    config: SimulationConfig
    output_dir: Path
    diagnostics: list[dict[str, float | bool]]
    times: np.ndarray
    velocity_slices: np.ndarray
    equatorial_slices: np.ndarray
    target_slices: np.ndarray
    force_slices: np.ndarray
    volume_times: np.ndarray
    volume_velocity: np.ndarray | None


_BOOLEAN_DIAGNOSTICS = {"resolved", "forcing_active"}
_PARTIAL_CHECKPOINT_VERSION = 1


def load_simulation_result(output_dir: Path) -> SimulationResult:
    """Load a completed run without repeating its CFD integration."""

    metadata_path = output_dir / "run.json"
    diagnostics_path = output_dir / "diagnostics.csv"
    slices_path = output_dir / "slices.npz"
    missing = [
        path.name
        for path in (metadata_path, diagnostics_path, slices_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete simulation checkpoint in {output_dir}: missing {', '.join(missing)}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema_version = metadata.get("diagnostic_schema_version")
    if schema_version not in {2, 3, 4, 5, DIAGNOSTIC_SCHEMA_VERSION}:
        raise ValueError(f"outdated diagnostics checkpoint in {output_dir}")
    config_values = dict(metadata["config"])
    if schema_version == 2:
        # v0.3 checkpoints predate the paper-coordinate model and therefore
        # unambiguously refer to the original separable target.
        config_values["profile_model"] = "separable"
        config_values["pulses_enabled"] = False
    if schema_version in {2, 3}:
        # v0.5 and earlier began their ramp immediately after t=0. Preserve
        # those checkpoint semantics rather than silently relabeling them as
        # the paper's initial rest interval.
        config_values["paper_time_cutoff_start"] = 0.0
        config_values["paper_time_cutoff_end"] = config_values.get(
            "ramp_time", 0.15
        )
    if schema_version <= 5:
        # Preserve the target that produced the checkpoint.  Without an
        # explicit revision, old results could otherwise compare equal to the
        # Appendix-B default and be mistaken for resumable new-profile work.
        config_values["paper_profile_revision"] = "legacy-hand-shaped"
    cfg = SimulationConfig(**config_values)
    cfg.validate()
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        diagnostics = []
        for source_row in csv.DictReader(handle):
            diagnostics.append(
                {
                    key: value.strip().lower() == "true"
                    if key in _BOOLEAN_DIAGNOSTICS
                    else float(value)
                    for key, value in source_row.items()
                }
            )
    if not diagnostics:
        raise ValueError(f"empty diagnostics checkpoint in {output_dir}")

    with np.load(slices_path, allow_pickle=False) as arrays:
        times = arrays["times"].copy()
        velocity_slices = arrays["velocity"].copy()
        equatorial_slices = arrays["equatorial_velocity"].copy()
        target_slices = arrays["target"].copy()
        force_slices = arrays["force"].copy()
    if len(diagnostics) != len(times):
        raise ValueError(
            f"checkpoint frame mismatch in {output_dir}: "
            f"{len(diagnostics)} diagnostics versus {len(times)} slices"
        )

    volume_path = output_dir / "volumes.npz"
    if volume_path.is_file():
        with np.load(volume_path, allow_pickle=False) as arrays:
            volume_times = arrays["times"].copy()
            volume_velocity = arrays["velocity"].copy()
    else:
        volume_times = np.empty(0, dtype=float)
        volume_velocity = None
    return SimulationResult(
        config=cfg,
        output_dir=output_dir,
        diagnostics=diagnostics,
        times=times,
        velocity_slices=velocity_slices,
        equatorial_slices=equatorial_slices,
        target_slices=target_slices,
        force_slices=force_slices,
        volume_times=volume_times,
        volume_velocity=volume_velocity,
    )


def load_matching_checkpoint(
    output_dir: Path, expected_config: SimulationConfig
) -> SimulationResult | None:
    """Return a complete checkpoint only when its configuration matches exactly."""

    try:
        result = load_simulation_result(output_dir)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return result if result.config == expected_config else None


def _to_field(values: np.ndarray, cfg: SimulationConfig):
    tensor = math.tensor(
        values,
        spatial(x=cfg.resolution, y=cfg.resolution, z=cfg.resolution),
        channel(vector="x,y,z"),
    )
    bounds = Box(
        x=(-cfg.half_domain, cfg.half_domain),
        y=(-cfg.half_domain, cfg.half_domain),
        z=(-cfg.half_domain, cfg.half_domain),
    )
    return CenteredGrid(
        tensor,
        extrapolation.PERIODIC,
        bounds=bounds,
        x=cfg.resolution,
        y=cfg.resolution,
        z=cfg.resolution,
    )


def _as_numpy(grid) -> np.ndarray:
    return np.asarray(grid.values.numpy("x,y,z,vector"), dtype=np.float64)


def _momentum_without_pressure(velocity, viscosity: float):
    return advect.finite_difference(
        velocity, velocity, order=2
    ) + diffuse.finite_difference(velocity, viscosity, order=2)


def _norms(velocity: np.ndarray, cfg: SimulationConfig) -> tuple[float, float]:
    speed_sq = np.sum(velocity * velocity, axis=-1)
    peak = float(np.sqrt(np.max(speed_sq)))
    energy = float(0.5 * np.sum(speed_sq) * cfg.dx**3)
    return peak, energy


def _vector_l2(values: np.ndarray, cfg: SimulationConfig) -> float:
    return float(np.sqrt(np.sum(values * values) * cfg.dx**3))


def _central_difference(values: np.ndarray, axis: int, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2 * dx)


def _spatial_derivative_norms(
    values: np.ndarray, cfg: SimulationConfig
) -> tuple[float, float]:
    gradient_sq = np.zeros(values.shape[:-1], dtype=np.float64)
    laplacian = np.zeros_like(values)
    for axis in range(3):
        derivative = _central_difference(values, axis, cfg.dx)
        gradient_sq += np.sum(derivative * derivative, axis=-1)
        laplacian += (
            np.roll(values, -1, axis=axis) - 2 * values + np.roll(values, 1, axis=axis)
        ) / cfg.dx**2
    gradient_l2 = float(np.sqrt(np.sum(gradient_sq) * cfg.dx**3))
    return gradient_l2, _vector_l2(laplacian, cfg)


@lru_cache(maxsize=8)
def _spectral_grid(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache resolution-only Fourier radii, ordering, and tail mask."""

    modes = np.fft.fftfreq(resolution) * resolution
    kx, ky, kz = np.meshgrid(modes, modes, modes, indexing="ij")
    magnitude = np.sqrt(kx * kx + ky * ky + kz * kz)
    order = np.argsort(magnitude.ravel())
    top_third = (
        (np.abs(kx) >= resolution / 3)
        | (np.abs(ky) >= resolution / 3)
        | (np.abs(kz) >= resolution / 3)
    )
    return magnitude, order, top_third


def _spectral_diagnostics(velocity: np.ndarray) -> tuple[float, float, float]:
    """Return energy-weighted mode, 95% mode, and top-third energy fraction."""

    resolution = velocity.shape[0]
    transformed = fft.fftn(
        velocity, axes=(0, 1, 2), norm="ortho", workers=-1
    )
    modal_energy = np.sum(np.abs(transformed) ** 2, axis=-1)
    total = float(np.sum(modal_energy))
    if total == 0:
        return 0.0, 0.0, 0.0
    magnitude, order, top_third = _spectral_grid(resolution)
    mean_mode = float(np.sum(magnitude * modal_energy) / total)
    cumulative = np.cumsum(modal_energy.ravel()[order])
    k95 = float(magnitude.ravel()[order[np.searchsorted(cumulative, 0.95 * total)]])
    tail_fraction = float(np.sum(modal_energy[top_third]) / total)
    return mean_mode, k95, tail_fraction


def _forcing_is_active(t: float, cfg: SimulationConfig) -> bool:
    return cfg.forcing_end is None or t < cfg.forcing_end - 1e-13


def _manufactured_force(t: float, target, cfg: SimulationConfig):
    """Continuous-time manufactured force for the discrete target field."""

    epsilon = min(cfg.derivative_epsilon, 0.2 * (cfg.t_star - t))
    if (
        cfg.profile_model == "paper-surrogate"
        and t <= cfg.paper_time_cutoff_start
    ):
        # Proposition 10.1 leaves a genuine open interval on which u=p=f=0.
        # Avoid constructing similarity coordinates for an identically zero
        # residual, especially in high-resolution from-rest trajectories.
        return target * 0
    if t - epsilon >= 0:
        sample_times = (t - epsilon, t + epsilon)
        denominator = 2 * epsilon
    else:
        sample_times = (t, t + epsilon)
        denominator = epsilon
    # The two target samples are independent and NumPy releases the GIL for
    # their array kernels.  Two workers use otherwise-idle CPU cores without
    # changing the centered derivative or its deterministic values.
    with ThreadPoolExecutor(max_workers=2) as executor:
        before, after = executor.map(
            lambda sample_t: target_velocity(sample_t, cfg), sample_times
        )
    derivative = (after - before) / denominator
    return _to_field(derivative, cfg) - _momentum_without_pressure(
        target, cfg.viscosity
    )


def _project(velocity, pressure_solve, cfg: SimulationConfig):
    # A periodic pressure is defined only up to a constant. PhiML emits a
    # rank-deficiency warning even when that null mode is declared on the
    # Solve object, so suppress only that known gauge warning.
    mode = cfg.pressure_projection
    if mode == "auto":
        mode = "matrix-free" if cfg.resolution > 32 else "sparse"
    if mode == "fft":
        # The centered periodic derivative has Fourier symbol i*sin(k*dx)/dx.
        # Project against that exact discrete symbol so the result is
        # divergence-free under the same operator used by the diagnostics.
        values = _as_numpy(velocity)
        frequency = np.fft.fftfreq(cfg.resolution)
        symbol = np.sin(2 * np.pi * frequency) / cfg.dx
        sx = symbol[:, None, None]
        sy = symbol[None, :, None]
        sz = symbol[None, None, :]
        denominator = sx * sx + sy * sy + sz * sz
        transformed = fft.fftn(values, axes=(0, 1, 2), workers=-1)
        longitudinal = (
            sx * transformed[..., 0]
            + sy * transformed[..., 1]
            + sz * transformed[..., 2]
        )
        factor = np.divide(
            longitudinal,
            denominator,
            out=np.zeros_like(longitudinal),
            where=denominator > 0,
        )
        transformed[..., 0] -= sx * factor
        transformed[..., 1] -= sy * factor
        transformed[..., 2] -= sz * factor
        projected = fft.ifftn(
            transformed, axes=(0, 1, 2), workers=-1
        ).real
        return _to_field(projected, cfg), None
    if mode == "matrix-free":
        divergence = field.divergence(velocity, order=2)
        pressure_guess = CenteredGrid(
            0.0,
            extrapolation.PERIODIC,
            bounds=velocity.bounds,
            x=cfg.resolution,
            y=cfg.resolution,
            z=cfg.resolution,
        )
        matrix_free_solve = Solve(
            "CG",
            rel_tol=cfg.pressure_rel_tol,
            abs_tol=cfg.pressure_abs_tol,
            x0=pressure_guess,
            max_iterations=cfg.pressure_max_iterations,
            rank_deficiency=0,
        )
        pressure = math.solve_linear(
            fluid.masked_laplace.f,
            divergence,
            matrix_free_solve,
            velocity.boundary,
            None,
            None,
            wide_stencil=True,
            order=2,
            implicit=None,
            upwind=None,
            correct_skew=False,
        )
        gradient = field.spatial_gradient(
            pressure,
            velocity.extrapolation,
            at=velocity.sampled_at,
            order=2,
            scheme="green-gauss",
        )
        return (velocity - gradient).with_extrapolation(
            velocity.extrapolation
        ), pressure

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Rank deficiency >= .* detected in linear solve.*",
            category=RuntimeWarning,
        )
        projected, pressure_increment = fluid.make_incompressible(
            velocity,
            solve=pressure_solve,
            order=2,
        )
    return projected, pressure_increment


def _diagnostic_row(
    t: float,
    velocity: np.ndarray,
    target: np.ndarray,
    force: np.ndarray,
    cfg: SimulationConfig,
) -> dict[str, float | bool]:
    scales = similarity_scales(t, cfg)
    peak, energy = _norms(velocity, cfg)
    target_peak, target_energy = _norms(target, cfg)
    force_l2 = _vector_l2(force, cfg)
    force_linf = float(np.max(np.linalg.norm(force, axis=-1)))
    force_gradient_l2, force_laplacian_l2 = _spatial_derivative_norms(force, cfg)
    error_l2 = _vector_l2(velocity - target, cfg)
    target_l2 = _vector_l2(target, cfg)
    vorticity = discrete_curl(velocity, cfg.dx)
    vorticity_sq = np.sum(vorticity * vorticity, axis=-1)
    peak_vorticity = float(np.sqrt(np.max(vorticity_sq)))
    enstrophy = float(0.5 * np.sum(vorticity_sq) * cfg.dx**3)
    dissipation = 2 * cfg.viscosity * enstrophy
    helicity = float(np.sum(velocity * vorticity) * cfg.dx**3)
    force_work = float(np.sum(velocity * force) * cfg.dx**3)
    radial, azimuthal, axial = cylindrical_components(velocity, cfg)
    radial_width, axial_width = energy_weighted_core_widths(velocity, cfg, t)
    spectral_mean_mode, spectral_k95, spectral_tail_fraction = _spectral_diagnostics(
        velocity
    )
    structure = paper_structure_diagnostics(t, cfg)
    axial_structure = axial_outflow_diagnostics(velocity, t, cfg)
    if cfg.profile_model == "paper-surrogate":
        coordinates = paper_similarity_coordinates(t, cfg)
        core_mask = (
            (coordinates.x_similarity < cfg.paper_annulus_xa)
            & (np.abs(coordinates.eta) < cfg.paper_eta_support)
        )
        kinetic_energy_core = float(
            0.5 * np.sum(np.sum(velocity * velocity, axis=-1)[core_mask]) * cfg.dx**3
        )
        target_kinetic_energy_core = float(
            0.5 * np.sum(np.sum(target * target, axis=-1)[core_mask]) * cfg.dx**3
        )
    else:
        kinetic_energy_core = energy
        target_kinetic_energy_core = target_energy

    phi_velocity = _to_field(velocity, cfg)
    advective = _as_numpy(advect.finite_difference(phi_velocity, phi_velocity, order=2))
    viscous = _as_numpy(diffuse.finite_difference(phi_velocity, cfg.viscosity, order=2))
    advective_l2 = _vector_l2(advective, cfg)
    viscous_l2 = _vector_l2(viscous, cfg)
    net_rhs_l2 = _vector_l2(advective + viscous + force, cfg)
    cancellation_ratio = (advective_l2 + viscous_l2 + force_l2) / max(net_rhs_l2, 1e-14)

    divergence = np.asarray(field.divergence(phi_velocity).values.numpy("x,y,z"))
    cells_r = scales.radial_length / cfg.dx
    cells_z = scales.axial_length / cfg.dx
    return {
        "t": t,
        "tau": scales.tau,
        "radial_length": scales.radial_length,
        "axial_length": scales.axial_length,
        "similarity_velocity": scales.velocity,
        "similarity_radial_velocity": scales.radial_velocity,
        "similarity_vorticity": scales.velocity / scales.radial_length,
        "similarity_energy_law": scales.energy_power_law,
        "temporal_activation": temporal_activation(t, cfg),
        "similarity_enstrophy_law": (scales.energy_power_law / scales.radial_length**2),
        "peak_speed": peak,
        "target_peak_speed": target_peak,
        "kinetic_energy": energy,
        "target_kinetic_energy": target_energy,
        "kinetic_energy_core": kinetic_energy_core,
        "target_kinetic_energy_core": target_kinetic_energy_core,
        "peak_radial_speed": float(np.max(np.abs(radial))),
        "peak_azimuthal_speed": float(np.max(np.abs(azimuthal))),
        "peak_axial_speed": float(np.max(np.abs(axial))),
        "measured_radial_width": radial_width,
        "measured_axial_width": axial_width,
        "force_l2": force_l2,
        "force_linf": force_linf,
        "force_gradient_l2": force_gradient_l2,
        "force_laplacian_l2": force_laplacian_l2,
        "force_work": force_work,
        "advective_l2": advective_l2,
        "viscous_l2": viscous_l2,
        "momentum_cancellation_ratio": cancellation_ratio,
        "peak_vorticity": peak_vorticity,
        "enstrophy": enstrophy,
        "viscous_dissipation": dissipation,
        "helicity": helicity,
        "angular_reynolds": scales.velocity * scales.radial_length / cfg.viscosity,
        "radial_reynolds": (
            scales.velocity
            * scales.radial_length**2
            / (scales.axial_length * cfg.viscosity)
        ),
        "spectral_mean_mode": spectral_mean_mode,
        "spectral_k95": spectral_k95,
        "spectral_tail_fraction": spectral_tail_fraction,
        "divergence_linf": float(np.max(np.abs(divergence))),
        "divergence_l2": float(np.sqrt(np.mean(divergence**2))),
        "tracking_relative_l2": error_l2 / max(target_l2, 1e-14),
        "cells_per_radial_scale": cells_r,
        "cells_per_axial_scale": cells_z,
        "resolved": min(cells_r, cells_z) >= cfg.min_cells_per_scale,
        "forcing_active": _forcing_is_active(t, cfg),
        **axial_structure,
        **structure,
    }


def _finalize_time_diagnostics(rows: list[dict[str, float | bool]]) -> None:
    """Add time-integrated and energy-budget quantities after the run."""

    times = np.asarray([row["t"] for row in rows], dtype=float)
    energy = np.asarray([row["kinetic_energy"] for row in rows], dtype=float)
    peak_vorticity = np.asarray([row["peak_vorticity"] for row in rows], dtype=float)
    force_work = np.asarray([row["force_work"] for row in rows], dtype=float)
    dissipation = np.asarray([row["viscous_dissipation"] for row in rows], dtype=float)
    edge_order = 2 if len(rows) > 2 else 1
    energy_rate = np.gradient(energy, times, edge_order=edge_order)
    residual = energy_rate - force_work + dissipation
    scale = np.abs(energy_rate) + np.abs(force_work) + dissipation + 1e-14
    bkm_integral = np.zeros_like(times)
    force_impulse = np.zeros_like(times)
    force_l2 = np.asarray([row["force_l2"] for row in rows], dtype=float)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        bkm_integral[i] = bkm_integral[i - 1] + 0.5 * dt * (
            peak_vorticity[i] + peak_vorticity[i - 1]
        )
        force_impulse[i] = force_impulse[i - 1] + 0.5 * dt * (
            force_l2[i] + force_l2[i - 1]
        )
    for i, row in enumerate(rows):
        row["energy_rate"] = float(energy_rate[i])
        row["energy_balance_residual"] = float(residual[i])
        row["energy_balance_relative"] = float(residual[i] / scale[i])
        row["bkm_integral"] = float(bkm_integral[i])
        row["force_l2_time_integral"] = float(force_impulse[i])


def _write_diagnostics(rows: list[dict[str, float | bool]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_partial_checkpoint(
    output_dir: Path,
    cfg: SimulationConfig,
    t: float,
    velocity: np.ndarray,
    diagnostics: list[dict[str, float | bool]],
    velocity_slices: list[np.ndarray],
    equatorial_slices: list[np.ndarray],
    target_slices: list[np.ndarray],
    force_slices: list[np.ndarray],
    volume_times: list[float],
    volume_velocity: list[np.ndarray],
) -> None:
    """Atomically persist enough state to resume after the latest saved frame."""

    arrays_path = output_dir / "partial-checkpoint.npz"
    arrays_temporary = output_dir / "partial-checkpoint.tmp.npz"
    metadata = {
        "partial_checkpoint_version": _PARTIAL_CHECKPOINT_VERSION,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "config": cfg.to_dict(),
        "diagnostics": diagnostics,
    }
    with arrays_temporary.open("wb") as handle:
        np.savez(
            handle,
            metadata=np.asarray(json.dumps(metadata)),
            t=np.asarray(t),
            velocity=velocity,
            velocity_slices=np.stack(velocity_slices),
            equatorial_slices=np.stack(equatorial_slices),
            target_slices=np.stack(target_slices),
            force_slices=np.stack(force_slices),
            volume_times=np.asarray(volume_times),
            volume_velocity=(
                np.stack(volume_velocity)
                if volume_velocity
                else np.empty(0, dtype=np.float64)
            ),
        )
    arrays_temporary.replace(arrays_path)


def _load_partial_checkpoint(
    output_dir: Path, cfg: SimulationConfig
) -> tuple[
    float,
    np.ndarray,
    list[dict[str, float | bool]],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[np.ndarray],
] | None:
    arrays_path = output_dir / "partial-checkpoint.npz"
    if not arrays_path.exists():
        return None
    if not arrays_path.is_file():
        raise ValueError(f"incomplete partial checkpoint in {output_dir}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
        if metadata.get("partial_checkpoint_version") != _PARTIAL_CHECKPOINT_VERSION:
            raise ValueError(f"unsupported partial checkpoint in {output_dir}")
        if metadata.get("diagnostic_schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
            raise ValueError(f"outdated partial checkpoint in {output_dir}")
        if metadata.get("config") != cfg.to_dict():
            raise ValueError(
                f"partial checkpoint configuration differs in {output_dir}; "
                "choose another output directory or pass resume=False"
            )
        volume_array = arrays["volume_velocity"]
        return (
            float(arrays["t"]),
            arrays["velocity"].copy(),
            list(metadata["diagnostics"]),
            [item.copy() for item in arrays["velocity_slices"]],
            [item.copy() for item in arrays["equatorial_slices"]],
            [item.copy() for item in arrays["target_slices"]],
            [item.copy() for item in arrays["force_slices"]],
            arrays["volume_times"].astype(float).tolist(),
            [item.copy() for item in volume_array] if volume_array.size else [],
        )


def _frame_times(cfg: SimulationConfig) -> np.ndarray:
    """Return the requested output clock without changing the physical interval."""
    if cfg.frame_spacing == "linear":
        return np.linspace(cfg.t_start, cfg.t_end, cfg.frames)
    similarity_start = -np.log(cfg.t_star - cfg.t_start)
    similarity_end = -np.log(cfg.t_star - cfg.t_end)
    similarity_times = np.linspace(similarity_start, similarity_end, cfg.frames)
    times = cfg.t_star - np.exp(-similarity_times)
    times[0] = cfg.t_start
    times[-1] = cfg.t_end
    return times


def run_simulation(
    cfg: SimulationConfig,
    output_dir: Path,
    resume: bool = True,
    save_final_state: bool = False,
) -> SimulationResult:
    """Run the manufactured-solution experiment and persist raw outputs."""

    cfg.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        completed = load_matching_checkpoint(output_dir, cfg)
        if completed is not None and (
            not save_final_state or (output_dir / "final-state.npz").is_file()
        ):
            return completed
    math.set_global_precision(64)

    save_times = _frame_times(cfg)
    pressure_solve = Solve(
        "CG",
        rel_tol=cfg.pressure_rel_tol,
        abs_tol=cfg.pressure_abs_tol,
        max_iterations=cfg.pressure_max_iterations,
        rank_deficiency=1,
    )

    partial = _load_partial_checkpoint(output_dir, cfg) if resume else None
    if partial is None:
        t = cfg.t_start
        target_np = target_velocity(t, cfg)
        target = _to_field(target_np, cfg)
        velocity = target
        diagnostics: list[dict[str, float | bool]] = []
        velocity_slices: list[np.ndarray] = []
        equatorial_slices: list[np.ndarray] = []
        target_slices: list[np.ndarray] = []
        force_slices: list[np.ndarray] = []
        volume_times: list[float] = []
        volume_velocity: list[np.ndarray] = []
    else:
        (
            t,
            velocity_np,
            diagnostics,
            velocity_slices,
            equatorial_slices,
            target_slices,
            force_slices,
            volume_times,
            volume_velocity,
        ) = partial
        target_np = target_velocity(t, cfg)
        target = _to_field(target_np, cfg)
        velocity = _to_field(velocity_np, cfg)
    recent_forces: deque[tuple[float, np.ndarray]] = deque(maxlen=3)
    if partial is not None:
        for row in diagnostics[-2:]:
            force_time = float(row["t"])
            force_target_np = target_velocity(force_time, cfg)
            force_target = _to_field(force_target_np, cfg)
            force_np = (
                _as_numpy(_manufactured_force(force_time, force_target, cfg))
                if _forcing_is_active(force_time, cfg)
                else np.zeros_like(force_target_np)
            )
            recent_forces.append((force_time, force_np))
    volume_indices = set(
        np.linspace(1, cfg.frames - 1, min(6, cfg.frames - 1), dtype=int).tolist()
    )
    center_y = cfg.resolution // 2

    def save_frame(save_t: float) -> None:
        velocity_np = _as_numpy(velocity)
        if _forcing_is_active(save_t, cfg):
            force_np = _as_numpy(_manufactured_force(save_t, target, cfg))
        else:
            force_np = np.zeros_like(target_np)
        row = _diagnostic_row(save_t, velocity_np, target_np, force_np, cfg)
        diagnostics.append(row)
        recent_forces.append((save_t, force_np.copy()))
        if len(diagnostics) == 2:
            first_t, first_force = recent_forces[0]
            second_t, second_force = recent_forces[1]
            diagnostics[0]["force_time_derivative_l2"] = _vector_l2(
                (second_force - first_force) / (second_t - first_t), cfg
            )
        if len(recent_forces) == 3:
            first_t, first_force = recent_forces[0]
            third_t, third_force = recent_forces[2]
            diagnostics[-2]["force_time_derivative_l2"] = _vector_l2(
                (third_force - first_force) / (third_t - first_t), cfg
            )
        velocity_slices.append(velocity_np[:, center_y, :, :])
        equatorial_slices.append(velocity_np[:, :, center_y, :])
        target_slices.append(target_np[:, center_y, :, :])
        force_slices.append(force_np[:, center_y, :, :])
        frame_index = len(diagnostics) - 1
        if cfg.capture_volumes and frame_index in volume_indices:
            volume_times.append(save_t)
            volume_velocity.append(velocity_np.copy())
        _write_partial_checkpoint(
            output_dir,
            cfg,
            save_t,
            velocity_np,
            diagnostics,
            velocity_slices,
            equatorial_slices,
            target_slices,
            force_slices,
            volume_times,
            volume_velocity,
        )

    if partial is None:
        save_frame(t)
    for frame_time in save_times[len(diagnostics) :]:
        while t < frame_time - 1e-13:
            if (
                cfg.profile_model == "paper-surrogate"
                and t < cfg.paper_time_cutoff_start - 1e-13
                and not np.any(_as_numpy(velocity))
            ):
                # The target and its residual force vanish identically before
                # the temporal cutoff begins. Advancing the exact zero state
                # to the next frame/cutoff is lossless and avoids empty CFD
                # substeps.
                next_t = min(frame_time, cfg.paper_time_cutoff_start)
                next_target_np = target_velocity(next_t, cfg)
                velocity = _to_field(next_target_np, cfg)
                target = velocity
                target_np = next_target_np
                t = next_t
                continue
            current_peak = max(
                _norms(target_np, cfg)[0], _norms(_as_numpy(velocity), cfg)[0], 1e-10
            )
            advective_dt = cfg.cfl * cfg.dx / current_peak
            diffusive_dt = 0.12 * cfg.dx**2 / cfg.viscosity
            dt = min(cfg.max_dt, advective_dt, diffusive_dt, frame_time - t)
            if cfg.forcing_end is not None and t < cfg.forcing_end < t + dt:
                dt = cfg.forcing_end - t

            next_t = t + dt

            force_t = (
                _manufactured_force(t, target, cfg)
                if _forcing_is_active(t, cfg)
                else target * 0
            )
            rhs_t = _momentum_without_pressure(velocity, cfg.viscosity) + force_t
            if cfg.integrator == "euler":
                velocity, _ = _project(velocity + dt * rhs_t, pressure_solve, cfg)
            else:
                midpoint_t = t + 0.5 * dt
                midpoint_trial = velocity + 0.5 * dt * rhs_t
                midpoint_velocity, _ = _project(midpoint_trial, pressure_solve, cfg)
                midpoint_target_np = target_velocity(midpoint_t, cfg)
                midpoint_target = _to_field(midpoint_target_np, cfg)
                midpoint_force = (
                    _manufactured_force(midpoint_t, midpoint_target, cfg)
                    if _forcing_is_active(midpoint_t, cfg)
                    else midpoint_target * 0
                )
                midpoint_rhs = (
                    _momentum_without_pressure(midpoint_velocity, cfg.viscosity)
                    + midpoint_force
                )
                velocity, _ = _project(
                    velocity + dt * midpoint_rhs, pressure_solve, cfg
                )

            next_target_np = target_velocity(next_t, cfg)
            next_target = _to_field(next_target_np, cfg)
            target = next_target
            target_np = next_target_np
            t = next_t

        save_frame(float(frame_time))

    previous_t, previous_force = recent_forces[-2]
    final_t, final_force = recent_forces[-1]
    diagnostics[-1]["force_time_derivative_l2"] = _vector_l2(
        (final_force - previous_force) / (final_t - previous_t), cfg
    )

    _finalize_time_diagnostics(diagnostics)

    result = SimulationResult(
        config=cfg,
        output_dir=output_dir,
        diagnostics=diagnostics,
        times=save_times,
        velocity_slices=np.stack(velocity_slices),
        equatorial_slices=np.stack(equatorial_slices),
        target_slices=np.stack(target_slices),
        force_slices=np.stack(force_slices),
        volume_times=np.asarray(volume_times),
        volume_velocity=np.stack(volume_velocity) if volume_velocity else None,
    )

    _write_diagnostics(diagnostics, output_dir / "diagnostics.csv")
    np.savez_compressed(
        output_dir / "slices.npz",
        times=result.times,
        velocity=result.velocity_slices,
        equatorial_velocity=result.equatorial_slices,
        target=result.target_slices,
        force=result.force_slices,
    )
    if result.volume_velocity is not None:
        np.savez_compressed(
            output_dir / "volumes.npz",
            times=result.volume_times,
            velocity=result.volume_velocity,
        )
    if save_final_state:
        final_state_temporary = output_dir / "final-state.tmp.npz"
        with final_state_temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                time=np.asarray(t),
                velocity=_as_numpy(velocity),
            )
        final_state_temporary.replace(output_dir / "final-state.npz")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "solver": (
            "PhiFlow 3.4.0 centered-grid finite differences with "
            f"projected {cfg.integrator} time integration; "
            f"pressure projection={cfg.pressure_projection}"
        ),
        "equations": "du/dt + (u dot grad)u - nu Laplacian(u) + grad(p) = f; div(u)=0",
        "experiment": (
            f"zero-initial-data manufactured {cfg.profile_model} similarity target"
            if cfg.forcing_end is None
            else f"force-release experiment; manufactured force disabled at t={cfg.forcing_end}"
        ),
        "scope_warning": (
            "This is not the exact OpenAI construction or a numerical proof of singularity. "
            "The paper-surrogate implements the explicit Appendix-B axis data, the exact initial "
            "rest interval, and smooth temporal localization, then uses a recorded finite "
            "off-axis continuation and multiscale approximation of both annular wave families. "
            "It does not reproduce the infinite pulse hierarchy or analytical all-order "
            "corrections; the manufactured force need not remain smooth as t approaches 1."
        ),
        "config": cfg.to_dict(),
        "final_state": "final-state.npz" if save_final_state else None,
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "partial-checkpoint.npz").unlink(missing_ok=True)
    return result
