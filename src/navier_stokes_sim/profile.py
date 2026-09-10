"""Divergence-free numerical surrogates for the OpenAI blow-up construction.

``paper-surrogate`` follows the paper's implicit similarity coordinates,
inner/annular/exterior decomposition, axial bias, and two oscillatory annular
wave families.  The waves are resolved surrogates for the paper's pulse
hierarchy, not its all-order convex-integration construction.  ``separable``
retains the earlier compact leading-vortex baseline for controlled comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import SimulationConfig


@dataclass(frozen=True)
class SimilarityScales:
    tau: float
    radial_length: float
    axial_length: float
    velocity: float
    radial_velocity: float
    energy_power_law: float


@dataclass(frozen=True)
class PaperCoordinates:
    """The paper's exact implicit similarity coordinates on the CFD grid."""

    q: np.ndarray
    eta: np.ndarray
    x_similarity: np.ndarray
    radius: np.ndarray
    theta: np.ndarray


def smooth_ramp(t: float, ramp_time: float) -> float:
    """C-infinity transition from zero to one over ``(0, ramp_time)``."""

    if t <= 0:
        return 0.0
    if t >= ramp_time:
        return 1.0
    s = t / ramp_time
    left = np.exp(-1.0 / s)
    right = np.exp(-1.0 / (1.0 - s))
    return float(left / (left + right))


def smooth_transition(t: float, start: float, end: float) -> float:
    """C-infinity step that is identically zero/one outside ``(start, end)``."""

    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return smooth_ramp(t - start, end - start)


def temporal_activation(t: float, cfg: SimulationConfig) -> float:
    """Return the profile's temporal localization factor.

    The paper-localized construction vanishes on an initial time interval and
    is unchanged near the singular time (Proposition 10.1).  The separable
    legacy baseline retains its original immediate ramp for reproducibility.
    """

    if cfg.profile_model == "paper-surrogate":
        return smooth_transition(
            t, cfg.paper_time_cutoff_start, cfg.paper_time_cutoff_end
        )
    return smooth_ramp(t, cfg.ramp_time)


def similarity_scales(t: float, cfg: SimulationConfig) -> SimilarityScales:
    """Return the leading-order scales stated in Section 2.1 of the paper."""

    tau = cfg.t_star - t
    if tau <= 0:
        raise ValueError("the similarity profile is defined only for t < t_star")
    ramp = temporal_activation(t, cfg)
    radial = cfg.base_radius * tau**0.5
    axial = cfg.base_height * tau ** (0.5 - cfg.h)
    velocity = ramp * cfg.velocity_scale * tau ** (-0.5 - cfg.h)
    radial_velocity = ramp * cfg.velocity_scale * tau**-0.5
    energy_law = ramp**2 * tau ** (0.5 - 3 * cfg.h)
    return SimilarityScales(tau, radial, axial, velocity, radial_velocity, energy_law)


def _bump(s: np.ndarray) -> np.ndarray:
    """Normalized compact C-infinity bump supported on ``abs(s) < 1``."""

    out = np.zeros_like(s, dtype=np.float64)
    mask = np.abs(s) < 1.0
    sm = s[mask]
    out[mask] = np.exp(1.0 - 1.0 / (1.0 - sm * sm))
    return out


def _smooth_step(s: np.ndarray) -> np.ndarray:
    """C-infinity step equal to zero/one outside the unit interval."""

    out = np.zeros_like(s, dtype=np.float64)
    out[s >= 1] = 1.0
    mask = (s > 0) & (s < 1)
    sm = s[mask]
    left = np.exp(-1.0 / sm)
    right = np.exp(-1.0 / (1.0 - sm))
    out[mask] = left / (left + right)
    return out


def _plateau_cutoff(
    coordinate: np.ndarray, inner: float, outer: float
) -> np.ndarray:
    return 1.0 - _smooth_step((np.abs(coordinate) - inner) / (outer - inner))


@lru_cache(maxsize=8)
def cell_centers(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached Cartesian cell-center coordinate arrays."""

    axis = (
        np.arange(cfg.resolution, dtype=np.float64) + 0.5
    ) * cfg.dx - cfg.half_domain
    return np.meshgrid(axis, axis, axis, indexing="ij")


def _central_difference(values: np.ndarray, axis: int, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2 * dx)


def paper_similarity_coordinates(t: float, cfg: SimulationConfig) -> PaperCoordinates:
    """Solve ``q - zeta^2 q^(2h) = tau`` and return ``(q, eta, X)``.

    Here ``zeta=z/base_height``, ``eta=zeta/q^(1/2-h)``, and
    ``X=(r/base_radius)^2/(2q)``.  These are equations (3.2) and (4.1) of
    the paper, rather than the separable ``r/tau^1/2, z/tau^D`` proxy.
    """

    tau = cfg.t_star - t
    if tau <= 0:
        raise ValueError("the similarity coordinates are defined only for t < t_star")
    x, y, z = cell_centers(cfg)
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    zeta = z / cfg.base_height
    d = 0.5 - cfg.h

    # The tau=0 root gives a sharp, positive initial guess.  The derivative
    # is 1-2h*eta^2 > 0 in the physical branch, so Newton is well conditioned.
    q = tau + np.abs(zeta) ** (1.0 / d)
    zeta_sq = zeta * zeta
    for _ in range(7):
        power = q ** (2 * cfg.h)
        residual = q - zeta_sq * power - tau
        derivative = 1.0 - 2 * cfg.h * zeta_sq * power / q
        q = np.maximum(tau, q - residual / derivative)

    eta = zeta / q**d
    x_similarity = (radius / cfg.base_radius) ** 2 / (2 * q)
    return PaperCoordinates(q, eta, x_similarity, radius, theta)


def _cumulative_trapezoid(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(coordinate)
    return np.concatenate(([0.0], np.cumsum(increments)))


@lru_cache(maxsize=16)
def _paper_radial_tables(
    xa: float, xb: float, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate fixed-profile antiderivatives used by the vector potential."""

    grid = np.linspace(0.0, 80.0, 32769)
    core = _bump(np.sqrt(grid / (0.78 * xa)))
    annulus = _bump((2 * grid - xa - xb) / (xb - xa))
    core_mass = np.trapezoid(core, grid)
    annulus_mass = np.trapezoid(annulus, grid)
    # This zero moment makes the meridional streamfunction vanish outside Xb,
    # the numerical analogue of the paper's exterior moment identity.
    u_radial = core - (core_mass / annulus_mass) * annulus
    u_integral = _cumulative_trapezoid(u_radial, grid)

    blend = _smooth_step((grid - xa) / (xb - xa))
    inner_phi = np.exp(-grid / (0.55 * xa)) * (1.0 - blend)
    return (
        grid,
        u_integral,
        _cumulative_trapezoid(inner_phi, grid),
    )


@lru_cache(maxsize=16)
def _exterior_tail_table(
    inner: float, outer: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-coordinate potential for the stationary exterior swirl."""

    radius = np.linspace(0.0, outer, 16385)
    cutoff = _plateau_cutoff(radius, inner, outer)
    swirl = np.zeros_like(radius)
    positive = radius > 0
    swirl[positive] = radius[positive] ** (-1.0 - 2 * h) * cutoff[positive]
    integral = _cumulative_trapezoid(swirl, radius)
    return radius, integral[-1] - integral


def _interp_table(x: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(x, grid, values, left=values[0], right=values[-1])


def _curl_vector_potential(
    a_x: np.ndarray,
    a_y: np.ndarray,
    a_z: np.ndarray,
    dx: float,
) -> np.ndarray:
    u_x = _central_difference(a_z, 1, dx) - _central_difference(a_y, 2, dx)
    u_y = _central_difference(a_x, 2, dx) - _central_difference(a_z, 0, dx)
    u_z = _central_difference(a_y, 0, dx) - _central_difference(a_x, 1, dx)
    return np.stack((u_x, u_y, u_z), axis=-1)


def _separable_target_velocity(t: float, cfg: SimulationConfig) -> np.ndarray:
    """Retain the original compact manufactured target as a baseline."""

    scales = similarity_scales(t, cfg)
    if scales.velocity == 0:
        return np.zeros((cfg.resolution,) * 3 + (3,), dtype=np.float64)

    x, y, z = cell_centers(cfg)
    r = np.sqrt(x * x + y * y)
    radial_similarity = r / scales.radial_length
    axial_similarity = z / scales.axial_length
    radial_bump = _bump(radial_similarity)
    axial_bump = _bump(axial_similarity)

    # For axisymmetric A = A_theta e_theta + A_z e_z, curl(A) combines a
    # meridional incompressible circulation with an azimuthal swirl.
    a_theta = (
        scales.radial_length
        * scales.velocity
        * radial_similarity
        * radial_bump
        * axial_similarity
        * axial_bump
    )
    a_z = (
        cfg.swirl_ratio
        * scales.radial_length
        * scales.velocity
        * radial_bump
        * axial_bump
    )

    cos_theta = np.divide(x, r, out=np.zeros_like(x), where=r > 0)
    sin_theta = np.divide(y, r, out=np.zeros_like(y), where=r > 0)
    a_x = -sin_theta * a_theta
    a_y = cos_theta * a_theta

    return _curl_vector_potential(a_x, a_y, a_z, cfg.dx)


def _paper_vector_potentials(
    t: float, cfg: SimulationConfig
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return background and two-family pulse vector potentials."""

    scales = similarity_scales(t, cfg)
    shape = (cfg.resolution,) * 3
    zeros = np.zeros(shape, dtype=np.float64)
    if scales.velocity == 0:
        return (zeros, zeros, zeros), (zeros, zeros, zeros)

    coordinates = paper_similarity_coordinates(t, cfg)
    q = coordinates.q
    eta = coordinates.eta
    similarity_x = coordinates.x_similarity
    radius = coordinates.radius
    theta = coordinates.theta
    x, y, z = cell_centers(cfg)
    cosine = np.divide(x, radius, out=np.zeros_like(x), where=radius > 0)
    sine = np.divide(y, radius, out=np.zeros_like(y), where=radius > 0)

    table_x, u_integral_table, phi_inner_table = (
        _paper_radial_tables(cfg.paper_annulus_xa, cfg.paper_annulus_xb, cfg.h)
    )
    u_integral = _interp_table(similarity_x, table_x, u_integral_table)
    # Use the tail-integral gauge.  Subtracting a spatial constant does not
    # change curl(A), but it makes A_z vanish before the fixed cutoff and avoids
    # generating a spurious velocity sheet where that cutoff transitions.
    phi_inner_integral = (
        _interp_table(similarity_x, table_x, phi_inner_table) - phi_inner_table[-1]
    )

    eta_window = _bump(eta / cfg.paper_eta_support)
    axial_profile = (eta + cfg.paper_axial_bias) * eta_window
    swirl_profile = (1.0 + cfg.paper_swirl_bias * eta) * eta_window
    ramp = temporal_activation(t, cfg)
    a_theta = (
        cfg.velocity_scale
        * ramp
        * cfg.base_radius**2
        * q ** (1.0 - (0.5 + cfg.h))
        * u_integral
        * axial_profile
        / np.maximum(radius, 0.25 * cfg.dx)
    )
    a_z_core = (
        -cfg.swirl_ratio
        * cfg.velocity_scale
        * ramp
        * cfg.base_radius
        * q ** (0.5 - (0.5 + cfg.h))
        * phi_inner_integral
        * swirl_profile
    )

    # Once X crosses the annulus, q-scaling cancels exactly and the paper's
    # exterior is a fixed r^(-1-2h) swirl.  Building its potential directly in
    # physical radius both enforces this stationarity and lets the compact
    # cutoff decay without differentiating a large gauge constant.
    exterior_radius, exterior_tail = _exterior_tail_table(
        cfg.localization_inner, cfg.localization_outer, cfg.h
    )
    exterior_potential = _interp_table(radius, exterior_radius, exterior_tail)
    exterior_blend = _smooth_step(
        (similarity_x - cfg.paper_annulus_xa)
        / (cfg.paper_annulus_xb - cfg.paper_annulus_xa)
    )
    exterior_coefficient = (
        cfg.swirl_ratio
        * cfg.velocity_scale
        * ramp
        * np.exp(-1.0 / 0.55)
        * cfg.paper_annulus_xa ** (1.0 + cfg.h)
        * 2.0 ** (1.0 + cfg.h)
        * cfg.base_radius ** (1.0 + 2 * cfg.h)
    )
    axial_localization = _plateau_cutoff(
        z, cfg.localization_inner, cfg.localization_outer
    )
    a_z = a_z_core + (
        exterior_coefficient
        * exterior_blend
        * exterior_potential
        * axial_localization
    )

    # A fixed spatial cutoff is applied to the potential, so its discrete curl
    # stays divergence-free and exactly periodic while leaving the core intact.
    localization = _plateau_cutoff(
        radius, cfg.localization_inner, cfg.localization_outer
    ) * axial_localization
    a_theta *= localization
    background = (-sine * a_theta, cosine * a_theta, a_z)

    pulse_x = np.zeros_like(a_theta)
    pulse_y = np.zeros_like(a_theta)
    pulse_z = np.zeros_like(a_theta)
    if cfg.pulses_enabled and (
        cfg.pulse_rtheta_strength > 0 or cfg.pulse_rz_strength > 0
    ):
        annulus = _bump(
            (2 * similarity_x - cfg.paper_annulus_xa - cfg.paper_annulus_xb)
            / (cfg.paper_annulus_xb - cfg.paper_annulus_xa)
        )
        pulse_eta = _bump(eta / cfg.paper_eta_support)
        envelope = annulus * pulse_eta * localization
        mode = cfg.pulse_azimuthal_mode
        log_phase = (
            2
            * np.pi
            * cfg.pulse_log_frequency
            * (-np.log2(max(scales.tau, np.finfo(float).tiny)))
        )
        log_slot = -np.log2(max(scales.tau, np.finfo(float).tiny))

        def pulse_gate(offset: float) -> float:
            distance = (log_slot - offset + 0.5) % 1.0 - 0.5
            return float(_bump(np.asarray(distance / cfg.pulse_time_width)))

        local_velocity = (
            cfg.velocity_scale
            * ramp
            * q ** (-0.5 - cfg.h + 0.5 * cfg.h)
        )
        local_length = cfg.base_radius * np.sqrt(q)

        # curl(A_z e_z) carries (r,theta) covariance; curl(A_theta e_theta)
        # carries (r,z) covariance.  Integer angular modes close around the
        # complete ring and have zero angular mean, as required in Section 7.
        radial_phase = cfg.pulse_radial_frequency * np.log(
            np.maximum(similarity_x, 0.25 * cfg.paper_annulus_xa)
            / cfg.paper_annulus_xa
        )
        phase_one = mode * theta + radial_phase + log_phase
        phase_two = (
            (mode + 1) * theta
            + radial_phase
            + cfg.pulse_axial_frequency * eta
            - 0.73 * log_phase
            + np.pi / 3
        )
        pulse_z = (
            cfg.pulse_rtheta_strength
            * pulse_gate(0.18)
            * local_velocity
            * local_length
            * envelope
            * np.sin(phase_one)
            / mode
        )
        pulse_theta = (
            cfg.pulse_rz_strength
            * pulse_gate(0.68)
            * local_velocity
            * local_length
            * envelope
            * np.cos(phase_two)
            / (mode + 1)
        )
        pulse_x = -sine * pulse_theta
        pulse_y = cosine * pulse_theta
    return background, (pulse_x, pulse_y, pulse_z)


def target_velocity_components(
    t: float, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return paper-surrogate background and oscillatory pulse velocities."""

    if cfg.profile_model == "separable":
        background = _separable_target_velocity(t, cfg)
        return background, np.zeros_like(background)
    background_potential, pulse_potential = _paper_vector_potentials(t, cfg)
    return (
        _curl_vector_potential(*background_potential, cfg.dx),
        _curl_vector_potential(*pulse_potential, cfg.dx),
    )


def target_velocity(t: float, cfg: SimulationConfig) -> np.ndarray:
    """Sample the selected target as an exactly divergence-free discrete curl."""

    background, pulses = target_velocity_components(t, cfg)
    return background + pulses


def paper_annulus_mask(t: float, cfg: SimulationConfig) -> np.ndarray:
    """Return the exact support region of the resolved pulse surrogates."""

    coordinates = paper_similarity_coordinates(t, cfg)
    return (
        (coordinates.x_similarity > cfg.paper_annulus_xa)
        & (coordinates.x_similarity < cfg.paper_annulus_xb)
        & (np.abs(coordinates.eta) < cfg.paper_eta_support)
    )


def paper_structure_diagnostics(
    t: float,
    cfg: SimulationConfig,
) -> dict[str, float]:
    """Audit exact coordinates, exterior closure, and resolved wave properties."""

    if cfg.profile_model != "paper-surrogate":
        return {
            "paper_coordinate_residual_linf": float("nan"),
            "paper_eta_linf": float("nan"),
            "background_exterior_meridional_fraction": float("nan"),
            "background_exterior_swirl_fraction": float("nan"),
            "pulse_energy_fraction": 0.0,
            "pulse_annulus_energy_fraction": 0.0,
            "pulse_axisymmetric_mean_fraction": 0.0,
            "pulse_covariance_rms": 0.0,
        }

    coordinates = paper_similarity_coordinates(t, cfg)
    tau = cfg.t_star - t
    d = 0.5 - cfg.h
    zeta = coordinates.eta * coordinates.q**d
    coordinate_residual = (
        coordinates.q
        - zeta * zeta * coordinates.q ** (2 * cfg.h)
        - tau
    )
    background, pulses = target_velocity_components(t, cfg)
    background_energy = np.sum(background * background, axis=-1)
    pulse_energy = np.sum(pulses * pulses, axis=-1)
    total_energy = float(np.sum((background + pulses) ** 2))

    background_radial, background_theta, background_axial = cylindrical_components(
        background, cfg
    )
    exterior_margin = max(0.15, 2 * cfg.dx / max(similarity_scales(t, cfg).radial_length, cfg.dx))
    exterior = (
        (coordinates.x_similarity > cfg.paper_annulus_xb + exterior_margin)
        & (coordinates.radius < cfg.localization_inner)
        & (np.abs(coordinates.eta) < cfg.paper_eta_support)
    )
    background_total = max(float(np.sum(background_energy)), 1e-30)
    exterior_meridional = float(
        np.sum(
            background_radial[exterior] ** 2
            + background_axial[exterior] ** 2
        )
        / background_total
    )
    exterior_swirl = float(
        np.sum(background_theta[exterior] ** 2) / background_total
    )

    strict_annulus = paper_annulus_mask(t, cfg)
    pulse_total = float(np.sum(pulse_energy))
    annulus_fraction = (
        float(np.sum(pulse_energy[strict_annulus]) / pulse_total)
        if pulse_total > 0
        else 0.0
    )

    pulse_radial, pulse_theta, pulse_axial = cylindrical_components(pulses, cfg)
    # Coarse (X,eta) bins leave enough angular samples to measure the ring
    # average instead of mistaking Cartesian sampling imbalance for m=0 flow.
    bins = 3
    ix = np.floor(
        (coordinates.x_similarity - cfg.paper_annulus_xa)
        / (cfg.paper_annulus_xb - cfg.paper_annulus_xa)
        * bins
    ).astype(int)
    ie = np.floor(
        (coordinates.eta + cfg.paper_eta_support)
        / (2 * cfg.paper_eta_support)
        * bins
    ).astype(int)
    valid = (
        strict_annulus
        & (ix >= 0)
        & (ix < bins)
        & (ie >= 0)
        & (ie < bins)
    )
    bin_id = ix + bins * ie
    counts = np.bincount(bin_id[valid], minlength=bins * bins)
    mean_energy = 0.0
    means: list[np.ndarray] = []
    for component in (pulse_radial, pulse_theta, pulse_axial):
        sums = np.bincount(
            bin_id[valid], weights=component[valid], minlength=bins * bins
        )
        component_mean = np.divide(
            sums, counts, out=np.zeros_like(sums), where=counts > 0
        )
        means.append(component_mean)
        mean_energy += float(np.sum(component_mean**2 * counts))
    covariance_sq = 0.0
    for product in (pulse_radial * pulse_theta, pulse_radial * pulse_axial):
        sums = np.bincount(
            bin_id[valid], weights=product[valid], minlength=bins * bins
        )
        covariance = np.divide(
            sums, counts, out=np.zeros_like(sums), where=counts > 0
        )
        covariance_sq += float(np.sum(covariance**2 * counts))
    sampled_pulse_energy = float(
        np.sum(
            pulse_radial[valid] ** 2
            + pulse_theta[valid] ** 2
            + pulse_axial[valid] ** 2
        )
    )
    mean_fraction = np.sqrt(mean_energy / max(sampled_pulse_energy, 1e-30))
    covariance_rms = np.sqrt(covariance_sq / max(float(np.sum(counts)), 1.0))
    return {
        "paper_coordinate_residual_linf": float(
            np.max(np.abs(coordinate_residual))
        ),
        "paper_eta_linf": float(np.max(np.abs(coordinates.eta))),
        "background_exterior_meridional_fraction": exterior_meridional,
        "background_exterior_swirl_fraction": exterior_swirl,
        "pulse_energy_fraction": pulse_total / max(total_energy, 1e-30),
        "pulse_annulus_energy_fraction": annulus_fraction,
        "pulse_axisymmetric_mean_fraction": float(mean_fraction),
        "pulse_covariance_rms": float(covariance_rms),
    }


def discrete_divergence(velocity: np.ndarray, dx: float) -> np.ndarray:
    """Divergence using the same centered, periodic stencil as the target curl."""

    return sum(_central_difference(velocity[..., i], i, dx) for i in range(3))


def discrete_curl(velocity: np.ndarray, dx: float) -> np.ndarray:
    """Curl using centered, periodic differences."""

    u_x, u_y, u_z = (velocity[..., i] for i in range(3))
    omega_x = _central_difference(u_z, 1, dx) - _central_difference(u_y, 2, dx)
    omega_y = _central_difference(u_x, 2, dx) - _central_difference(u_z, 0, dx)
    omega_z = _central_difference(u_y, 0, dx) - _central_difference(u_x, 1, dx)
    return np.stack((omega_x, omega_y, omega_z), axis=-1)


def cylindrical_components(
    velocity: np.ndarray, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial, azimuthal, and axial velocity components."""

    x, y, _ = cell_centers(cfg)
    radius = np.sqrt(x * x + y * y)
    cosine = np.divide(x, radius, out=np.zeros_like(x), where=radius > 0)
    sine = np.divide(y, radius, out=np.zeros_like(y), where=radius > 0)
    radial = cosine * velocity[..., 0] + sine * velocity[..., 1]
    azimuthal = -sine * velocity[..., 0] + cosine * velocity[..., 1]
    return radial, azimuthal, velocity[..., 2]


def energy_weighted_core_widths(
    velocity: np.ndarray, cfg: SimulationConfig, t: float | None = None
) -> tuple[float, float]:
    """Measure radial and axial RMS widths of the kinetic-energy density."""

    x, y, z = cell_centers(cfg)
    weight = np.sum(velocity * velocity, axis=-1)
    if t is not None and cfg.profile_model == "paper-surrogate":
        coordinates = paper_similarity_coordinates(t, cfg)
        core = (
            (coordinates.x_similarity < cfg.paper_annulus_xa)
            & (np.abs(coordinates.eta) < cfg.paper_eta_support)
        )
        weight = np.where(core, weight, 0.0)
    total = float(np.sum(weight))
    if total == 0:
        return 0.0, 0.0
    radial_width = np.sqrt(float(np.sum((x * x + y * y) * weight)) / total)
    axial_width = np.sqrt(float(np.sum(z * z * weight)) / total)
    return radial_width, axial_width
