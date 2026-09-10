"""Run a proof-aligned daytime queue without using Young or Mali."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
from datetime import datetime
from pathlib import Path

import overnight_frontier as runner

RUN_NAME = "daytime-20260910"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _full_run(
    project: Path,
    run_root: Path,
    state: dict[str, object],
    status_path: Path,
    deadline: datetime,
    name: str,
    resolution: int,
    *,
    t_start: float,
    t_end: float,
    frames: int,
    extra: list[str] | None = None,
) -> None:
    if runner._deadline_reached(deadline):
        return
    free_gib = shutil.disk_usage(project).free / 1024**3
    if free_gib < runner.MIN_FREE_DISK_GIB:
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
        runner._atomic_json(status_path, state)
        return
    command = runner._base_command(
        project,
        run_root / name,
        resolution,
        t_start=t_start,
        t_end=t_end,
        frames=frames,
    ) + ["--save-final-state"]
    if extra:
        command.extend(extra)
    runner._run_job(
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
    rest_only: bool = False,
) -> None:
    if rest_only:
        cutoff = ["--paper-time-cutoff", "0.55", "0.775"]
        if role == "kay":
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n192-paper-rest-to-099",
                192,
                t_start=0.0,
                t_end=0.99,
                frames=51,
                extra=cutoff,
            )
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n224-paper-rest-to-0992",
                224,
                t_start=0.0,
                t_end=0.992,
                frames=63,
                extra=cutoff,
            )
        else:
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n192-paper-rest-half-dt-to-099",
                192,
                t_start=0.0,
                t_end=0.99,
                frames=51,
                extra=cutoff + ["--max-dt", "0.0006510416666666666"],
            )
            _full_run(
                project,
                run_root,
                state,
                status_path,
                deadline,
                "n224-paper-rest-half-dt-to-0992",
                224,
                t_start=0.0,
                t_end=0.992,
                frames=63,
                extra=cutoff + ["--max-dt", "0.0004783163265306122"],
            )
        return

    passed = runner._probe(
        project,
        run_root,
        state,
        status_path,
        deadline,
        256,
        t_end=0.9408,
    )
    if passed and role == "kay":
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n256-forced-to-0996",
            256,
            t_start=0.94,
            t_end=0.996,
            frames=15,
        )
    if passed and role == "oliver":
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n256-half-dt-forced-to-0996",
            256,
            t_start=0.94,
            t_end=0.996,
            frames=15,
            extra=["--max-dt", "0.0003662109375"],
        )

    if role == "kay":
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n160-from-rest-to-0985",
            160,
            t_start=0.0,
            t_end=0.985,
            frames=41,
        )
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n192-from-rest-to-099",
            192,
            t_start=0.0,
            t_end=0.99,
            frames=51,
        )
    else:
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n128-from-rest-to-098",
            128,
            t_start=0.0,
            t_end=0.98,
            frames=41,
        )
        _full_run(
            project,
            run_root,
            state,
            status_path,
            deadline,
            "n192-half-dt-from-rest-to-099",
            192,
            t_start=0.0,
            t_end=0.99,
            frames=51,
            extra=["--max-dt", "0.0006510416666666666"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("kay", "oliver"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument(
        "--rest-only",
        action="store_true",
        help="run the matched paper-localized zero-initial-data queue",
    )
    args = parser.parse_args()

    project = args.project.resolve()
    run_root = project / "outputs" / args.run_name / args.role
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
    runner._atomic_json(status_path, state)
    signal.signal(signal.SIGTERM, runner._signal_handler)
    signal.signal(signal.SIGINT, runner._signal_handler)
    try:
        _run_role(
            args.role,
            project,
            run_root,
            state,
            status_path,
            args.deadline,
            rest_only=args.rest_only,
        )
        state["state"] = (
            "stopped-at-deadline"
            if runner._deadline_reached(args.deadline)
            else "complete"
        )
    except Exception as error:
        state["state"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state["finished_at"] = _timestamp()
        runner._atomic_json(status_path, state)


if __name__ == "__main__":
    main()
