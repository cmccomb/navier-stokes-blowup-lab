"""Read back every native vector frame and compare a restarted full field."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


def native_roundtrip(
    executable: Path, checker: Path, inputs: Path, output: Path
) -> dict:
    output.mkdir()
    command = [
        str(executable),
        str(inputs),
        "ns.force=shear",
        "stop_time=0.025",
        "ns.max_dt=0.00625",
        "amr.max_level=2",
        "ns.refine_half_width=0.5 0.25",
        "amr.plot_int=1",
        "amr.check_int=2",
    ]

    def run(argv, cwd, log):
        p = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
        )
        log.write_text(p.stdout + p.stderr)
        if p.returncode:
            raise ValueError(f"native archive exit {p.returncode}: {log}")
        return p.stdout

    def result(stdout):
        rows = [
            json.loads(s.removeprefix("NS_ARCHIVE_RESULT "))
            for s in stdout.splitlines()
            if s.startswith("NS_ARCHIVE_RESULT ")
        ]
        if len(rows) != 1:
            raise ValueError("missing native frame check")
        return rows[0]

    run(command, output, output / "run.log")
    plots = sorted(output.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    if len(plots) != 5:
        raise ValueError("native archive did not preserve all frames from zero")
    rows = []
    for step, plot in enumerate(plots):
        stdout = run(
            [str(checker), f"plot={plot.resolve()}", "ns.force=shear"],
            output,
            output / f"read-{step}.log",
        )
        row = result(stdout)
        if (
            row["step"] != step
            or row["levels"] != 3
            or not math.isclose(row["time"], step * 0.00625, abs_tol=1e-15)
        ):
            raise ValueError("native frame has wrong clock or mesh")
        rows.append(row)
    restart = output / "restart"
    restart.mkdir()
    run(
        command + [f"amr.restart={(output / 'chk00002').resolve()}"],
        restart,
        restart / "run.log",
    )
    stdout = run(
        [
            str(checker),
            f"plot={(restart / 'plt00004').resolve()}",
            f"compare={(output / 'plt00004').resolve()}",
            "ns.force=shear",
        ],
        restart,
        restart / "compare.log",
    )
    comparison = result(stdout)
    if comparison["compared"] is not True:
        raise ValueError("missing full-field restart comparison")
    if rows[0]["peak_speed"] != 0:
        raise ValueError("archived initial velocity is not exact rest")
    return {"passed": True, "frames": rows, "restart": comparison}
