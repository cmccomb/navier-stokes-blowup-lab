"""Run one checkpointed overnight Navier--Stokes queue on a fleet node."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUN_NAME = "overnight-20260909"
MAX_PROBE_RSS_GIB = 10.5
MIN_FREE_DISK_GIB = 4.0

_stop_requested = False
_child: subprocess.Popen[bytes] | None = None


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _signal_handler(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    if _child is not None and _child.poll() is None:
        try:
            os.killpg(_child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _deadline_reached(deadline: datetime) -> bool:
    return _stop_requested or datetime.now(deadline.tzinfo) >= deadline


def _base_command(
    project: Path,
    output: Path,
    resolution: int,
    *,
    t_start: float = 0.94,
    t_end: float = 0.995,
    frames: int = 12,
) -> list[str]:
    return [
        str(project / ".venv/bin/python"),
        str(project / "run.py"),
        "--resolution",
        str(resolution),
        "--t-start",
        str(t_start),
        "--t-end",
        str(t_end),
        "--frames",
        str(frames),
        "--profile-model",
        "paper-surrogate",
        "--integrator",
        "rk2",
        "--pressure-projection",
        "fft",
        "--output",
        str(output),
        "--no-animation",
        "--no-3d",
    ]


def _parse_resource_use(log_path: Path) -> tuple[int | None, int | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rss_matches = re.findall(r"(\d+)\s+maximum resident set size", text)
    swap_matches = re.findall(r"(\d+)\s+swaps", text)
    rss = int(rss_matches[-1]) if rss_matches else None
    swaps = int(swap_matches[-1]) if swap_matches else None
    return rss, swaps


def _run_job(
    state: dict[str, object],
    status_path: Path,
    deadline: datetime,
    name: str,
    command: list[str],
    log_path: Path,
    *,
    benchmark: bool = False,
) -> dict[str, object]:
    global _child
    record: dict[str, object] = {
        "name": name,
        "state": "running",
        "started_at": _timestamp(),
        "command": command,
        "log": str(log_path),
    }
    jobs = state["jobs"]
    assert isinstance(jobs, list)
    jobs.append(record)
    state["active_job"] = name
    _atomic_json(status_path, state)

    actual_command = ["/usr/bin/time", "-l", *command] if benchmark else command
    start = time.monotonic()
    with log_path.open("ab", buffering=0) as log:
        log.write(f"\n[{_timestamp()}] starting {name}\n".encode())
        _child = subprocess.Popen(
            actual_command,
            cwd=status_path.parents[3],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ | {"PYTHONUNBUFFERED": "1"},
        )
        while _child.poll() is None and not _deadline_reached(deadline):
            time.sleep(2)
        if _child.poll() is None:
            record["state"] = "stopping-at-deadline"
            _atomic_json(status_path, state)
            try:
                os.killpg(_child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                _child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(_child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _child.wait()
        return_code = _child.returncode
        log.write(
            f"[{_timestamp()}] finished {name} with return code {return_code}\n".encode()
        )
    _child = None

    record.update(
        {
            "state": "complete" if return_code == 0 else "stopped-or-failed",
            "finished_at": _timestamp(),
            "return_code": return_code,
            "elapsed_seconds": time.monotonic() - start,
        }
    )
    if benchmark:
        rss, swaps = _parse_resource_use(log_path)
        record["maximum_resident_set_bytes"] = rss
        record["swaps"] = swaps
    state["active_job"] = None
    _atomic_json(status_path, state)
    return record


def _probe(
    project: Path,
    run_root: Path,
    state: dict[str, object],
    status_path: Path,
    deadline: datetime,
    resolution: int,
    *,
    t_end: float = 0.941,
) -> bool:
    name = f"n{resolution}-capacity-probe"
    output = run_root / name
    command = _base_command(
        project, output, resolution, t_end=t_end, frames=2
    ) + ["--no-resume"]
    record = _run_job(
        state,
        status_path,
        deadline,
        name,
        command,
        run_root / f"{name}.log",
        benchmark=True,
    )
    rss = record.get("maximum_resident_set_bytes")
    swaps = record.get("swaps")
    passed = (
        record["return_code"] == 0
        and isinstance(rss, int)
        and rss <= MAX_PROBE_RSS_GIB * 1024**3
        and swaps == 0
    )
    record["capacity_gate_passed"] = passed
    _atomic_json(status_path, state)
    return passed


def _full_run(
    project: Path,
    run_root: Path,
    state: dict[str, object],
    status_path: Path,
    deadline: datetime,
    name: str,
    resolution: int,
    extra: list[str] | None = None,
    *,
    output: Path | None = None,
) -> None:
    if _deadline_reached(deadline):
        return
    free_gib = shutil.disk_usage(project).free / 1024**3
    if free_gib < MIN_FREE_DISK_GIB:
        jobs = state["jobs"]
        assert isinstance(jobs, list)
        jobs.append(
            {
                "name": name,
                "state": "skipped-low-disk",
                "free_disk_gib": free_gib,
                "time": _timestamp(),
            }
        )
        _atomic_json(status_path, state)
        return
    target = output or run_root / name
    command = _base_command(project, target, resolution) + ["--save-final-state"]
    if extra:
        command.extend(extra)
    _run_job(
        state,
        status_path,
        deadline,
        name,
        command,
        run_root / f"{name}.log",
    )


def _run_role(
    role: str,
    project: Path,
    run_root: Path,
    state: dict[str, object],
    status_path: Path,
    deadline: datetime,
) -> None:
    if role == "mali":
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n224-main-resume",
            224,
            output=project / "outputs/paper-surrogate-n224-frontier",
        )
        if not _deadline_reached(deadline) and _probe(
            project, run_root, state, status_path, deadline, 240
        ):
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n240-full-pulse",
                240,
            )
        return

    if role == "kay":
        if _probe(project, run_root, state, status_path, deadline, 224):
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n224-half-dt",
                224,
                ["--max-dt", "0.0004783163265306122"],
            )
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n240-half-dt",
                240,
                ["--max-dt", "0.0004166666666666667"],
            )
        return

    if role == "oliver":
        if _probe(project, run_root, state, status_path, deadline, 224):
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n224-no-pulses",
                224,
                ["--no-pulses"],
            )
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n224-strong-pulses",
                224,
                ["--pulse-strengths", "0.12", "0.10"],
            )
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n224-force-release",
                224,
                ["--forcing-end", "0.98"],
            )
        return

    if role == "young":
        selected: int | None = None
        for resolution in (256, 272, 280, 288):
            probe_end = 0.9408 if resolution >= 256 else 0.941
            if not _probe(
                project,
                run_root,
                state,
                status_path,
                deadline,
                resolution,
                t_end=probe_end,
            ):
                break
            selected = resolution
        state["selected_resolution"] = selected
        _atomic_json(status_path, state)
        if selected is not None:
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                f"n{selected}-maximum-safe",
                selected,
            )
        return

    raise ValueError(f"unknown role: {role}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("mali", "kay", "oliver", "young"), required=True
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    run_root = project / "outputs" / RUN_NAME / args.role
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    state: dict[str, object] = {
        "role": args.role,
        "host": os.uname().nodename,
        "project": str(project),
        "deadline": args.deadline.isoformat(),
        "started_at": _timestamp(),
        "state": "running",
        "active_job": None,
        "jobs": [],
    }
    _atomic_json(status_path, state)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        _run_role(
            args.role, project, run_root, state, status_path, args.deadline
        )
        state["state"] = (
            "stopped-at-deadline" if _deadline_reached(args.deadline) else "complete"
        )
    except Exception as error:
        state["state"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state["finished_at"] = _timestamp()
        _atomic_json(status_path, state)


if __name__ == "__main__":
    main()
