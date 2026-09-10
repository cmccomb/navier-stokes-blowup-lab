"""Kay-owned coordinator for the Kay/Oliver daytime simulation queues."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUN_NAME = "daytime-20260910"
OLIVER_PROJECT = Path("/Users/work/Codex/runs/ns-frontier-20260909")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ssh(identity: Path, command: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "oliver",
        command,
    ]


def _local_launch(
    project: Path, deadline: datetime, run_name: str, rest_only: bool
) -> int:
    run_root = project / "outputs" / run_name / "kay"
    run_root.mkdir(parents=True, exist_ok=True)
    log = (run_root / "worker.log").open("ab", buffering=0)
    process = subprocess.Popen(
        [
            "caffeinate",
            "-dimsu",
            str(project / ".venv/bin/python"),
            str(project / "fleet/daytime_frontier.py"),
            "--role",
            "kay",
            "--project",
            str(project),
            "--deadline",
            deadline.isoformat(),
            "--run-name",
            run_name,
            *(["--rest-only"] if rest_only else []),
        ],
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return process.pid


def _remote_launch(
    identity: Path, deadline: datetime, run_name: str, rest_only: bool
) -> int:
    project = OLIVER_PROJECT
    run_root = project / "outputs" / run_name / "oliver"
    rest_argument = " --rest-only" if rest_only else ""
    command = (
        f"mkdir -p {shlex.quote(str(run_root))} && "
        f"cd {shlex.quote(str(project))} && "
        "nohup caffeinate -dimsu "
        f"{shlex.quote(str(project / '.venv/bin/python'))} "
        f"{shlex.quote(str(project / 'fleet/daytime_frontier.py'))} "
        f"--role oliver --project {shlex.quote(str(project))} "
        f"--deadline {shlex.quote(deadline.isoformat())} "
        f"--run-name {shlex.quote(run_name)}{rest_argument} "
        f">> {shlex.quote(str(run_root / 'worker.log'))} 2>&1 < /dev/null & echo $!"
    )
    result = subprocess.run(
        _ssh(identity, command), check=True, capture_output=True, text=True
    )
    return int(result.stdout.strip().splitlines()[-1])


def _alive(identity: Path, role: str, pid: int) -> bool:
    if role == "kay":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return (
        subprocess.run(
            _ssh(identity, f"kill -0 {pid}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _stop(identity: Path, role: str) -> None:
    pattern = f"daytime_frontier.py --role {role}"
    if role == "kay":
        subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
    else:
        subprocess.run(
            _ssh(identity, f"pkill -TERM -f {shlex.quote(pattern)} || true"),
            check=False,
        )


def _collect(identity: Path, project: Path, run_name: str) -> Path:
    collected = project / "outputs" / f"{run_name}-collected"
    kay_source = project / "outputs" / run_name / "kay"
    if kay_source.exists():
        shutil.copytree(
            kay_source,
            collected / "kay",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("partial-checkpoint*.npz", "*.tmp.npz"),
        )
    (collected / "oliver").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude",
            "partial-checkpoint*.npz",
            "--exclude",
            "*.tmp.npz",
            "-e",
            f"ssh -i {identity} -o IdentitiesOnly=yes -o BatchMode=yes",
            f"oliver:{OLIVER_PROJECT}/outputs/{run_name}/oliver/",
            str(collected / "oliver") + "/",
        ],
        check=False,
    )
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--rest-only", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    identity = args.identity.resolve()
    root = project / "outputs" / args.run_name / "coordinator"
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    state: dict[str, object] = {
        "owner": "kay",
        "started_at": _timestamp(),
        "deadline": args.deadline.isoformat(),
        "state": "launching",
        "workers": {},
        "excluded_hosts": ["young", "mali"],
        "rest_only": args.rest_only,
    }
    workers = state["workers"]
    assert isinstance(workers, dict)
    _atomic_json(status_path, state)
    workers["kay"] = {
        "pid": _local_launch(
            project, args.deadline, args.run_name, args.rest_only
        )
    }
    try:
        workers["oliver"] = {
            "pid": _remote_launch(
                identity, args.deadline, args.run_name, args.rest_only
            )
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        workers["oliver"] = {
            "launch_error": f"{type(error).__name__}: {error}"
        }
    state["state"] = "running"
    _atomic_json(status_path, state)

    while datetime.now(args.deadline.tzinfo) < args.deadline:
        any_alive = False
        for role, record in workers.items():
            if not isinstance(record, dict) or "pid" not in record:
                continue
            alive = _alive(identity, role, int(record["pid"]))
            record["alive"] = alive
            record["checked_at"] = _timestamp()
            any_alive = any_alive or alive
        _atomic_json(status_path, state)
        if not any_alive:
            break
        remaining = (args.deadline - datetime.now(args.deadline.tzinfo)).total_seconds()
        time.sleep(max(1, min(60, remaining)))

    state["state"] = "stopping"
    _atomic_json(status_path, state)
    for role in workers:
        _stop(identity, role)
    time.sleep(5)
    state["state"] = "collecting"
    _atomic_json(status_path, state)
    collected = _collect(identity, project, args.run_name)
    state["state"] = "complete"
    state["finished_at"] = _timestamp()
    state["collected_root"] = str(collected)
    _atomic_json(status_path, state)


if __name__ == "__main__":
    main()
