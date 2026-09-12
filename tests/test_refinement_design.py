from dataclasses import replace
from math import log, pi

import numpy as np
import pytest

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.refinement_design import (
    extrapolate,
    mesh_cells,
    phase_geometry,
    sampled_coverage,
)


def test_similarity_extrapolation_preserves_reference_and_has_no_physical_units():
    cfg = SimulationConfig(forcing_phase_step=0.0375)
    same = extrapolate(0.985, 4.179807916147603, 1, cfg)
    ten = extrapolate(0.985, 4.179807916147603, 10, cfg)
    assert same["t"] == pytest.approx(0.985)
    assert same["additional_phase_only_steps"] == 0
    assert ten["t"] == pytest.approx(0.9998387174720215)
    assert ten["radial_shrink_factor"] == pytest.approx(9.64388379154)
    assert ten["radial_design_equivalent_n"] == 11564
    assert ten["forcing_phase_dt_limit"] / same[
        "forcing_phase_dt_limit"
    ] == pytest.approx(ten["tau"] / same["tau"])
    assert "mach" not in ten
    for factor in (0, -1, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            extrapolate(0.985, 4.18, factor, cfg)


def test_phase_norm_matches_cartesian_finite_differences():
    cfg = SimulationConfig()
    t, X, eta, theta = 0.93, np.array(1.02), np.array(-0.61), np.array(0.27)
    xyz, norm = phase_geometry(t, X, eta, theta, cfg)

    def phases(point):
        x, y, z = point
        tau, d, zeta = 1 - t, 0.5 - cfg.h, z / cfg.base_height
        q = tau + abs(zeta) ** (1 / d)
        for _ in range(15):
            q -= (q - zeta * zeta * q ** (2 * cfg.h) - tau) / (
                1 - 2 * cfg.h * zeta * zeta * q ** (2 * cfg.h - 1)
            )
        e = zeta / q**d
        xx = (x * x + y * y) / (2 * cfg.base_radius**2 * q)
        angle = np.arctan2(y, x)
        values = []
        for level in range(3):
            scale, mode = 1 + 0.65 * level, 4 + level
            values += [
                mode * angle + scale * 4 * log(xx / 0.6),
                (mode + 1) * angle + scale * 4 * log(xx / 0.6) + scale * 2.5 * e,
            ]
        return np.array(values)

    epsilon = 1e-7
    gradients = np.stack(
        [
            (
                phases(xyz + epsilon * np.eye(3)[d])
                - phases(xyz - epsilon * np.eye(3)[d])
            )
            / (2 * epsilon)
            for d in range(3)
        ],
        axis=-1,
    )
    assert float(norm) == pytest.approx(
        np.linalg.norm(gradients, axis=-1).max(), rel=1e-8
    )
    assert phase_geometry(t, X, eta, theta + pi / 2, cfg)[1] == pytest.approx(norm)


def test_deep_nesting_does_not_hide_coarse_forcing_underresolution():
    cfg = SimulationConfig()
    endpoint = extrapolate(0.985, 4.18, 10, cfg)["t"]
    shallow, deep = sampled_coverage(
        endpoint, cfg, [(128, 4, 0.5), (128, 8, 0.5)], 65, 13
    )
    assert shallow["minimum_sampled_doubled_phase_points"] < 1
    assert deep["minimum_sampled_doubled_phase_points"] < 4
    assert not deep["sampled_phase_screen_passed"]
    assert not deep["production_ready"]
    assert mesh_cells(128, 4, 0.5) == (8388608, 7602176)
    assert mesh_cells(128, 8, 0.5)[0] == 16777216
    with pytest.raises(ValueError):
        mesh_cells(128, 8, 0.99)
    with pytest.raises(ValueError):
        sampled_coverage(endpoint, replace(cfg, pulses_enabled=False), [(128, 4, 0.5)])
