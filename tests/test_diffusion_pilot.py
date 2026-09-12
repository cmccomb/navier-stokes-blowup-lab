import json

import pytest

from scripts.diffusion_pilot import MARKER, orders, parse_and_check
from scripts.refinement_pilot import AMREX_COMMIT


def test_diffusion_validator_rejects_invalid_or_misidentified_results():
    row = {
        "schema_version": 1,
        "components": 3,
        "steps": 20,
        "initial_energy": 0,
        "final_energy": 0,
        "max_energy_increase": 0,
        "passed": True,
        "amrex_commit": AMREX_COMMIT,
        "source_sha256": "a" * 64,
        "test_case": "rest",
        "dt": 0.01,
        "t_end": 0.2,
        "initial_linf": 0,
        "final_linf": 0,
        "l2_error": 0,
        "mass_balance_defect": 0,
        "max_linear_residual": 0,
        "composite_volume": 8,
        "peak_rss_mib": 20,
        "elapsed_seconds": 1,
    }
    assert parse_and_check(MARKER + json.dumps(row), {"test_case": "rest"}) == row
    for key, value in [
        ("passed", False),
        ("passed", "true"),
        ("final_linf", 1e-25),
        ("mass_balance_defect", 1e-5),
        ("l2_error", float("nan")),
        ("source_sha256", "unknown"),
    ]:
        with pytest.raises(ValueError):
            parse_and_check(
                MARKER + json.dumps({**row, key: value}), {"test_case": "rest"}
            )
    with pytest.raises(ValueError):
        parse_and_check(MARKER + json.dumps(row), {"test_case": "temporal"})


def test_diffusion_order_gate_requires_second_order_not_just_small_errors():
    for exponent, passed in [(1, False), (2, True)]:
        rows = [{"steps": s, "l2_error": 1e-9 / s**exponent} for s in (16, 32, 64)]
        assert all(r["passed"] is passed for r in orders(rows, "steps"))
