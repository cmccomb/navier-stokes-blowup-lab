import json

import pytest

from scripts.refinement_pilot import (
    AMREX_COMMIT,
    HYDRO_COMMIT,
    MARKER,
    check_result,
    convergence,
    parse_result,
)


def valid_result():
    return {
        "schema_version": 1,
        "amrex_commit": AMREX_COMMIT,
        "hydro_commit": HYDRO_COMMIT,
        "pilot_source_sha256": "a" * 64,
        "passed": True,
        "base_n": 16,
        "levels": 3,
        "max_grid_size": 32,
        "finest_effective_n": 64,
        "stored_cells": 12288,
        "active_cells": 11264,
        "composite_volume": 8,
        "test_case": "rest",
        "velocity_linf": 0,
        "velocity_l2_error": 0,
        "velocity_relative_l2_error": 0,
        "velocity_linf_error": 0,
        "interface_velocity_linf_error": 0,
        "divergence_before_linf": 0,
        "divergence_after_linf": 1e-12,
        "divergence_after_l2": 1e-13,
        "coarse_fine_flux_mismatch": 0,
        "projection_idempotence_l2": 1e-13,
        "elapsed_seconds": 1,
        "peak_rss_mib": 100,
    }


def test_result_parser_rejects_missing_duplicate_and_nonfinite_records():
    row = valid_result()
    line = MARKER + json.dumps(row)
    assert parse_result("AMReX startup\n" + line) == row
    for text in [
        "",
        line + "\n" + line,
        MARKER + '{"schema_version":1,"error":NaN}',
        MARKER + "[]",
    ]:
        with pytest.raises(ValueError):
            parse_result(text)


def test_operator_gate_counts_only_uncovered_cells_and_requires_exact_rest():
    row = valid_result()
    assert not check_result(row)
    for key, value in [
        ("active_cells", 12288),
        ("composite_volume", 9),
        ("velocity_linf", 1e-20),
        ("divergence_after_linf", 1e-7),
        ("coarse_fine_flux_mismatch", 1e-8),
        ("passed", False),
        ("passed", "true"),
        ("levels", True),
        ("velocity_l2_error", float("nan")),
        ("velocity_l2_error", -1),
        ("pilot_source_sha256", "unknown"),
        ("finest_effective_n", 128),
    ]:
        assert check_result({**row, key: value})
    assert check_result(row, {"test_case": "mixed"})
    assert not check_result(row, {"test_case": "rest", "base_n": 16})


def test_convergence_gate_does_not_accept_small_first_order_errors():
    rows = [{"base_n": n, "velocity_l2_error": 1e-5 / n**2} for n in [16, 32, 64]]
    assert all(r["passed"] for r in convergence(rows))
    assert all(r["observed_order"] == pytest.approx(2) for r in convergence(rows))
    rows = [{"base_n": n, "velocity_l2_error": 1e-12 / n} for n in [16, 32, 64]]
    assert not any(r["passed"] for r in convergence(rows))
    with pytest.raises(ValueError):
        convergence([rows[0], {"base_n": 24, "velocity_l2_error": 1e-15}])
