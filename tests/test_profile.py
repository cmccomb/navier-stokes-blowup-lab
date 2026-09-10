from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicSpline

from navier_stokes_sim.config import SimulationConfig, mesh_preset_parameters
from navier_stokes_sim.profile import (
    _interp_table,
    _paper_interpolants,
    axial_outflow_diagnostics,
    cylindrical_components,
    discrete_curl,
    discrete_divergence,
    energy_weighted_core_widths,
    paper_annulus_mask,
    paper_axis_profiles,
    paper_similarity_coordinates,
    paper_structure_diagnostics,
    similarity_scales,
    target_velocity,
    target_velocity_components,
    target_velocity_decomposition,
    temporal_activation,
)


def test_cubic_table_eliminates_slope_jumps_and_clamps_exterior() -> None:
    grid = np.linspace(0, 1, 9)
    values = grid**3
    spline = CubicSpline(grid, values)
    x = np.linspace(0, 1, 73)
    np.testing.assert_allclose(_interp_table(x, grid, values, spline), x**3, atol=1e-14)
    np.testing.assert_array_equal(
        _interp_table(np.array([-1, 2]), grid, values, spline), [0, 1]
    )
    knot, eps = 0.5, 1e-7

    def slopes(curve):
        samples = _interp_table(
            np.array([knot - eps, knot, knot + eps]), grid, values, curve
        )
        return np.diff(samples) / eps

    assert abs(np.diff(slopes(None))[0]) > 0.1
    assert abs(np.diff(slopes(spline))[0]) < 1e-5


def test_cubic_target_preserves_rest_and_discrete_solenoidality() -> None:
    cfg = SimulationConfig(resolution=16)
    assert _paper_interpolants(cfg) is _paper_interpolants(cfg)
    assert _paper_interpolants(replace(cfg, profile_interpolation="linear")) == {}
    assert not np.any(target_velocity(0, cfg))
    cubic = target_velocity(0.9, cfg)
    linear = target_velocity(0.9, replace(cfg, profile_interpolation="linear"))
    assert np.max(np.abs(discrete_divergence(cubic, cfg.dx))) < 1e-11
    assert 0 < np.linalg.norm(cubic - linear) / np.linalg.norm(cubic) < 1e-3


def test_starts_from_rest() -> None:
    cfg = SimulationConfig(resolution=12, frames=2, t_end=0.2)
    velocity = target_velocity(0.0, cfg)
    assert np.count_nonzero(velocity) == 0


def test_late_coarse_grid_handles_empty_pulse_bins() -> None:
    cfg = SimulationConfig(resolution=24, t_start=0.98, t_end=0.985, frames=3)
    row = paper_structure_diagnostics(0.9825, cfg)
    assert all(np.isfinite(value) for value in row.values())


def test_paper_target_has_an_initial_rest_interval() -> None:
    cfg = SimulationConfig(
        resolution=12,
        frames=2,
        t_end=0.9,
        paper_time_cutoff_start=0.4,
        paper_time_cutoff_end=0.6,
    )
    assert temporal_activation(0.4, cfg) == 0.0
    assert np.count_nonzero(target_velocity(0.3, cfg)) == 0
    assert 0.0 < temporal_activation(0.5, cfg) < 1.0
    assert temporal_activation(0.6, cfg) == 1.0
    assert np.max(np.linalg.norm(target_velocity(0.7, cfg), axis=-1)) > 0


def test_target_is_discretely_divergence_free() -> None:
    cfg = SimulationConfig(resolution=16, frames=2, t_end=0.2)
    velocity = target_velocity(0.5, cfg)
    divergence = discrete_divergence(velocity, cfg.dx)
    assert np.max(np.abs(divergence)) < 1e-11


def test_discrete_divergence_of_discrete_curl_is_zero() -> None:
    cfg = SimulationConfig(resolution=12, frames=2, t_end=0.2)
    rng = np.random.default_rng(7)
    vector_potential = rng.standard_normal((12, 12, 12, 3))
    velocity = discrete_curl(vector_potential, cfg.dx)
    assert np.max(np.abs(discrete_divergence(velocity, cfg.dx))) < 1e-12


def test_similarity_exponents_match_paper() -> None:
    cfg = SimulationConfig(
        resolution=12,
        frames=2,
        t_end=0.8,
        h=0.008,
        paper_time_cutoff_start=0.1,
        paper_time_cutoff_end=0.2,
    )
    a = similarity_scales(0.5, cfg)
    b = similarity_scales(0.75, cfg)
    tau_ratio = b.tau / a.tau
    assert np.isclose(b.radial_length / a.radial_length, tau_ratio**0.5)
    assert np.isclose(b.axial_length / a.axial_length, tau_ratio ** (0.5 - cfg.h))
    assert np.isclose(b.velocity / a.velocity, tau_ratio ** (-0.5 - cfg.h))
    assert np.isclose(b.radial_velocity / a.radial_velocity, tau_ratio**-0.5)
    assert np.isclose(
        b.energy_power_law / a.energy_power_law, tau_ratio ** (0.5 - 3 * cfg.h)
    )


def test_component_and_width_diagnostics_are_finite() -> None:
    cfg = SimulationConfig(resolution=16, frames=2, t_end=0.8)
    velocity = target_velocity(0.8, cfg)
    components = cylindrical_components(velocity, cfg)
    widths = energy_weighted_core_widths(velocity, cfg)
    assert all(component.shape == velocity.shape[:-1] for component in components)
    assert all(np.isfinite(width) and width > 0 for width in widths)


def test_paper_similarity_coordinates_close_implicit_equation() -> None:
    cfg = SimulationConfig(resolution=20, frames=2, t_end=0.8)
    t = 0.8
    coordinates = paper_similarity_coordinates(t, cfg)
    zeta = coordinates.eta * coordinates.q ** (0.5 - cfg.h)
    residual = (
        coordinates.q - zeta * zeta * coordinates.q ** (2 * cfg.h) - (cfg.t_star - t)
    )
    assert np.max(np.abs(residual)) < 1e-13
    assert np.max(np.abs(coordinates.eta)) < 1


def test_paper_axis_profiles_use_explicit_appendix_b_data() -> None:
    cfg = SimulationConfig(resolution=16, frames=2, t_end=0.8)
    eta = np.linspace(-0.8, 0.8, 17)
    axial, phi = paper_axis_profiles(eta, cfg)
    np.testing.assert_allclose(
        axial, cfg.paper_axial_slope * eta + cfg.paper_axis_offset
    )
    assert np.all(phi > 0)
    assert np.isclose(phi[len(phi) // 2], 1.0)


def test_appendix_b_revision_is_distinct_from_legacy_profile() -> None:
    cfg = SimulationConfig(resolution=16, frames=2, t_end=0.8)
    legacy = SimulationConfig(
        **(cfg.to_dict() | {"paper_profile_revision": "legacy-hand-shaped"})
    )
    assert not np.allclose(target_velocity(0.8, cfg), target_velocity(0.8, legacy))


def test_axial_outflow_diagnostics_detect_opposed_lobes() -> None:
    cfg = SimulationConfig(resolution=32, frames=2, t_end=0.85)
    t = 0.85
    diagnostics = axial_outflow_diagnostics(target_velocity(t, cfg), t, cfg)
    assert diagnostics["axial_outflow_alignment"] > 0.85
    assert diagnostics["upper_axial_flux"] > 0
    assert diagnostics["lower_axial_flux"] > 0
    assert diagnostics["similarity_core_aspect_ratio"] > 1
    assert diagnostics["measured_core_aspect_ratio"] > 0
    assert diagnostics["mesh_spacing_gain"] == 1
    assert diagnostics["cutoff_boundary_clearance_cells"] > 0


def test_core_refined_mesh_spends_more_cells_on_the_fixed_core() -> None:
    mesh = mesh_preset_parameters("core-refined")
    cfg = SimulationConfig(
        resolution=32,
        frames=2,
        t_end=0.85,
        mesh_preset="core-refined",
        **mesh,
    )
    diagnostics = axial_outflow_diagnostics(target_velocity(0.85, cfg), 0.85, cfg)
    assert np.isclose(diagnostics["mesh_spacing_gain"], 1 / 0.68)
    assert diagnostics["cutoff_boundary_clearance_cells"] > 0
    assert diagnostics["activation_annulus_clearance_cells"] > 0


def test_paper_wave_families_are_localized_and_divergence_free() -> None:
    cfg = SimulationConfig(resolution=32, frames=2, t_end=0.75)
    t = 0.75
    background, pulses = target_velocity_components(t, cfg)
    assert np.max(np.abs(discrete_divergence(background, cfg.dx))) < 1e-11
    assert np.max(np.abs(discrete_divergence(pulses, cfg.dx))) < 1e-11
    assert np.count_nonzero(paper_annulus_mask(t, cfg)) > 0
    diagnostics = paper_structure_diagnostics(t, cfg)
    assert diagnostics["paper_coordinate_residual_linf"] < 1e-13
    # A centered curl spreads the sampled support by one cell at each edge.
    assert diagnostics["pulse_annulus_energy_fraction"] > 0.85
    assert diagnostics["pulse_covariance_rms"] > 0


def test_pulse_hierarchy_and_correction_are_separately_divergence_free() -> None:
    cfg = SimulationConfig(resolution=32, frames=2, t_end=0.85)
    background, pulses, correction = target_velocity_decomposition(0.85, cfg)
    for component in (background, pulses, correction):
        assert np.max(np.abs(discrete_divergence(component, cfg.dx))) < 1e-11
    assert np.linalg.norm(pulses) > 0
    assert np.linalg.norm(correction) > 0
    diagnostics = paper_structure_diagnostics(0.85, cfg)
    assert diagnostics["pulse_hierarchy_levels"] == cfg.pulse_hierarchy_levels
    assert diagnostics["pulse_correction_energy_fraction"] > 0


def test_multiple_pulse_levels_change_the_resolved_wave_field() -> None:
    common = {"resolution": 32, "frames": 2, "t_end": 0.85}
    one_level = SimulationConfig(**common, pulse_hierarchy_levels=1)
    three_levels = SimulationConfig(**common, pulse_hierarchy_levels=3)
    _, one = target_velocity_components(0.85, one_level)
    _, three = target_velocity_components(0.85, three_levels)
    assert not np.allclose(one, three)


def test_legacy_profile_remains_available() -> None:
    cfg = SimulationConfig(
        resolution=16,
        frames=2,
        t_end=0.5,
        profile_model="separable",
        pulses_enabled=False,
    )
    velocity = target_velocity(0.5, cfg)
    assert np.max(np.linalg.norm(velocity, axis=-1)) > 0
    assert np.max(np.abs(discrete_divergence(velocity, cfg.dx))) < 1e-11
