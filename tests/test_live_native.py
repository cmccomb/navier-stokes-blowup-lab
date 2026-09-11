import copy
import json
import subprocess
import sys

import pytest

from scripts import stream_run
from scripts.live_native import observation_key, validate_package


def package(tmp_path):
    run = {
        "id": "native-run",
        "source_commit": "solver-sha",
        "revision": "render-sha",
        "config": {"resolution": 192, "t_start": 0, "t_end": 0.985},
        "archive": {
            "resolution": 192,
            "dtype": "float32",
            "fields": ["velocity", "force"],
            "components_per_field": 3,
        },
        "captured_frames": 3,
        "clip_times": [0, 0.55, 0.56],
        "latest_t": 0.56,
        "status": "running",
        "diagnostics": {"t": 0.55, "peak_speed": 0.0},
    }
    for name in stream_run.OUTPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete output")
    (tmp_path / "site/data/stream.json").write_text(json.dumps(run))
    return run


@pytest.mark.parametrize(
    "bad", ["run", "config", "precision", "nan", "time_order", "partial", "regression"]
)
def test_publication_rejects_mixed_invalid_or_regressing_packages(tmp_path, bad):
    observed = package(tmp_path)
    assert validate_package(tmp_path, observed, None) == observed
    run = copy.deepcopy(observed)
    previous = None
    if bad == "run":
        run["id"] = "old-run"
    elif bad == "config":
        run["config"]["resolution"] = 32
    elif bad == "precision":
        run["archive"]["dtype"] = "float16"
    elif bad == "nan":
        run["diagnostics"]["peak_speed"] = float("nan")
    elif bad == "time_order":
        run["clip_times"] = [0.55, 0, 0.56]
    elif bad == "partial":
        (tmp_path / stream_run.OUTPUTS[-1]).unlink()
    elif bad == "regression":
        previous = dict(observed, latest_t=0.57, captured_frames=4)
    (tmp_path / "site/data/stream.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        validate_package(tmp_path, observed, previous)


def test_new_diagnostic_and_completion_trigger_updates_without_new_movie(tmp_path):
    run = package(tmp_path)
    key = observation_key(run)
    run["observed_at"] = "another poll"
    assert observation_key(run) == key
    run["diagnostics"]["t"] = 0.556
    assert observation_key(run) != key
    key = observation_key(run)
    run["status"] = "complete"
    assert observation_key(run) != key


def test_finished_run_still_supplies_final_diagnostics(tmp_path, monkeypatch):
    root = tmp_path / "outputs/native-run"
    store = root / "preview-volumes"
    store.mkdir(parents=True)
    (root / "launch.json").write_text(
        json.dumps({"pid": 999999999, "started_at": "then", "source_commit": "abc"})
    )
    (store / "manifest.json").write_text(json.dumps({"config": {"resolution": 192}}))
    (root / "run.json").write_text("{}")
    (root / "final-state.npz").touch()
    (root / "diagnostics.csv").write_text("t,peak_speed,resolved\n0.985,2.5,True\n")

    def execute(argv):
        assert argv[1] == "-c"
        return subprocess.check_output([sys.executable, *argv[1:]], text=True)

    monkeypatch.setattr(stream_run, "command", execute)
    result = stream_run.read_remote([], "local", str(root))
    assert result["status"] == "complete"
    assert result["diagnostics"] == {"t": 0.985, "peak_speed": 2.5, "resolved": True}
    assert result["captured_frames"] == 0 and result["latest_t"] is None
