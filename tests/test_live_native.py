import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from scripts import stream_run
from scripts.live_native import observation_key, validate_package
from scripts.volume_history import render_3d


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
        "clip_times_3d": [0, 0.55, 0.56],
        "display_resolution_3d": 4,
        "latest_t": 0.56,
        "status": "running",
        "diagnostics": {"t": 0.55, "peak_speed": 0.0},
    }
    for name in stream_run.OUTPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete output")
    run["playback"] = stream_run.build_playback(run["clip_times"])
    run["render"] = {}
    for field, name in (("velocity", "flow"), ("force", "force")):
        path = tmp_path / f"site/media/stream-{name}.gif"
        frames = [Image.new("RGB", (4, 4), (i * 50, 0, 0)) for i in range(3)]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=run["playback"]["source_frame_duration_ms"],
        )
        run["render"][field] = {
            "gif_verification": stream_run.verify_gif(
                path, run["playback"]["source_frame_duration_ms"]
            )
        }
        run["render"][field]["volume_history"] = render_3d(
            [
                {
                    "time": t,
                    "axis": np.linspace(-0.75, 0.75, 4),
                    field: np.full((4, 4, 4, 3), i, dtype=np.float32),
                }
                for i, t in enumerate(run["clip_times"])
            ],
            field,
            tmp_path / f"site/media/stream-{name}-3d.html",
            {"resolution": 192, "half_domain": 1},
        )
    (tmp_path / "site/data/stream.json").write_text(json.dumps(run))
    return run


@pytest.mark.parametrize(
    "bad",
    [
        "run",
        "config",
        "precision",
        "nan",
        "time_order",
        "partial",
        "regression",
        "missing_history",
        "missing_rest",
        "wrong_gif",
        "short_3d",
        "wrong_3d_time",
        "wrong_3d_hash",
        "missing_3d",
        "unsafe_3d_path",
        "wrong_3d_html",
        "wrong_3d_scale",
    ],
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
    elif bad == "missing_history":
        run["captured_frames"] = 30
    elif bad == "missing_rest":
        run["clip_times"][0] = 0.54
    elif bad == "wrong_gif":
        run["render"]["velocity"]["gif_verification"]["sha256"] = "wrong"
    elif bad == "short_3d":
        run["clip_times_3d"] = run["clip_times"][-2:]
    elif bad == "wrong_3d_time":
        run["render"]["force"]["volume_history"]["frames"][1]["time"] = 0.551
    elif bad == "wrong_3d_hash":
        frame = run["render"]["force"]["volume_history"]["frames"][1]
        path = tmp_path / frame["path"]
        path.write_bytes(bytes([1]) + path.read_bytes()[1:])
    elif bad == "missing_3d":
        (
            tmp_path / run["render"]["force"]["volume_history"]["frames"][1]["path"]
        ).unlink()
    elif bad == "unsafe_3d_path":
        run["render"]["force"]["volume_history"]["frames"][1]["path"] = (
            "site/media/../../native.npz"
        )
    elif bad == "wrong_3d_html":
        (tmp_path / "site/media/stream-force-3d.html").write_text("older viewer")
    elif bad == "wrong_3d_scale":
        run["render"]["force"]["volume_history"]["limits"]["x"] = 1
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
