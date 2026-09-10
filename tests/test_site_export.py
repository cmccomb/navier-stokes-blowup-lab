import csv
import json

import pytest

from scripts.export_site_results import build_payload, load_run


def _write_run(tmp_path, *, t_start=0.0, resolved=True):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": "2026-09-10T00:00:00+00:00",
                "solver": "test solver",
                "scope_warning": "finite surrogate",
                "config": {
                    "resolution": 64,
                    "t_start": t_start,
                    "t_end": 0.9,
                    "paper_time_cutoff_start": 0.55,
                    "paper_time_cutoff_end": 0.775,
                    "integrator": "rk2",
                    "pressure_projection": "fft",
                },
            }
        ),
        encoding="utf-8",
    )
    fields = [
        "t",
        "peak_speed",
        "target_peak_speed",
        "kinetic_energy",
        "kinetic_energy_core",
        "peak_vorticity",
        "tracking_relative_l2",
        "divergence_linf",
        "cells_per_radial_scale",
        "cells_per_axial_scale",
        "spectral_k95",
        "spectral_tail_fraction",
        "force_l2",
        "resolved",
    ]
    with (run_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: resolved if field == "resolved" else 1 for field in fields})
    return run_dir


def test_site_export_publishes_one_complete_run(tmp_path):
    run = load_run(_write_run(tmp_path))
    payload = build_payload(run, "2026-09-10T00:00:00+00:00")

    assert payload["run"]["resolution"] == 64
    assert payload["run"]["frames"] == 1
    assert payload["run"]["resolved"] is True
    assert "path" not in payload["run"]


@pytest.mark.parametrize("t_start,resolved", [(0.1, True), (0.0, False)])
def test_site_export_rejects_incomplete_public_claim(tmp_path, t_start, resolved):
    run = load_run(_write_run(tmp_path, t_start=t_start, resolved=resolved))
    with pytest.raises(ValueError):
        build_payload(run, "2026-09-10T00:00:00+00:00")
