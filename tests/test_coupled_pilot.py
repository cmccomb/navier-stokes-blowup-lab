import json
from copy import deepcopy

import pytest

from backends.amrex.incflo_overlay import replace
from scripts.coupled_pilot import MARKER, parse_history


def fixture_history():
    base = {
        "adapter_sha256": "a" * 64,
        "schema_version": 1,
        "case": "rest",
        "problem": 0,
        "levels": 3,
        "volume": 8,
        "stored_cells": 12288,
        "active_cells": 11264,
        "l2_error": 0,
        "linf_error": 0,
        "peak_speed": 0,
        "energy": 0,
        "peak_rss_mib": 20,
        "velocity_integral": [0, 0, 0],
    }
    return [{**base, "time": i * 0.01, "step": i, "dt": 0.01} for i in range(3)]


def check(rows):
    return parse_history(
        "\n".join(MARKER + json.dumps(r) for r in rows),
        mode="rest",
        levels=3,
        end=0.02,
        max_dt=0.01,
    )


def test_coupled_history_clock_and_exact_rest():
    rows = fixture_history()
    assert check(rows) == rows
    for index, key, value in [
        (0, "time", 0.01),
        (1, "step", 2),
        (1, "dt", 0.02),
        (2, "time", 0.019),
        (1, "peak_speed", 1e-30),
        (1, "volume", 7.9),
        (1, "l2_error", float("nan")),
        (1, "levels", 2),
        (2, "case", "shear"),
    ]:
        altered = deepcopy(rows)
        altered[index][key] = value
        with pytest.raises(ValueError):
            check(altered)
    with pytest.raises(ValueError):
        check(rows[:1])
    with pytest.raises(ValueError):
        check(rows[1:])


def test_overlay_fails_closed_on_upstream_drift():
    assert replace("old old", "old", "new", 2) == "new new"
    for source, count in [("missing", 1), ("old old", 1), ("old", 2)]:
        with pytest.raises(ValueError, match="upstream drift"):
            replace(source, "old", "new", count)
