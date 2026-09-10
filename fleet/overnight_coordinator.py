"""Kay-owned launcher, monitor, hard stop, and collector for the fleet run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

RUN_NAME = "overnight-20260909"
REMOTE_PROJECTS = {
    "mali": Path("/Users/ccm/Documents/Codex/2026-09-09/build-a-simulation-using-an-established"),
    "oliver": Path("/Users/work/Codex/runs/ns-frontier-20260909"),
    "young": Path("/Users/mccomb/Codex/runs/ns-frontier-20260909"),
}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ssh(identity: Path, host: str, command: str) -> list[str]:
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
        host,
        command,
    ]


def _remote_launch(
    identity: Path, host: str, project: Path, deadline: datetime
) -> int:
    run_root = project / "outputs" / RUN_NAME / host
    command = (
        f"mkdir -p {shlex.quote(str(run_root))} && "
        f"cd {shlex.quote(str(project))} && "
        "nohup caffeinate -dimsu "
        f"{shlex.quote(str(project / '.venv/bin/python'))} "
        f"{shlex.quote(str(project / 'fleet/overnight_frontier.py'))} "
        f"--role {shlex.quote(host)} --project {shlex.quote(str(project))} "
        f"--deadline {shlex.quote(deadline.isoformat())} "
        f">> {shlex.quote(str(run_root / 'worker.log'))} 2>&1 < /dev/null & echo $!"
    )
    result = subprocess.run(
        _ssh(identity, host, command),
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[-1])


def _local_launch(project: Path, deadline: datetime) -> int:
    run_root = project / "outputs" / RUN_NAME / "kay"
    run_root.mkdir(parents=True, exist_ok=True)
    log = (run_root / "worker.log").open("ab", buffering=0)
    process = subprocess.Popen(
        [
            "caffeinate",
            "-dimsu",
            str(project / ".venv/bin/python"),
            str(project / "fleet/overnight_frontier.py"),
            "--role",
            "kay",
            "--project",
            str(project),
            "--deadline",
            deadline.isoformat(),
        ],
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return process.pid


def _alive(identity: Path, host: str, pid: int) -> bool:
    if host == "kay":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return (
        subprocess.run(
            _ssh(identity, host, f"kill -0 {pid}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _stop_role(identity: Path, host: str) -> None:
    pattern = f"overnight_frontier.py --role {host}"
    if host == "kay":
        subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
        return
    subprocess.run(
        _ssh(identity, host, f"pkill -TERM -f {shlex.quote(pattern)} || true"),
        check=False,
    )


def _collect(identity: Path, project: Path, host: str, remote_project: Path) -> None:
    destination = project / "outputs" / "fleet-overnight-collected" / host
    destination.mkdir(parents=True, exist_ok=True)
    ssh_transport = f"ssh -i {identity} -o IdentitiesOnly=yes -o BatchMode=yes"
    source = f"{host}:{remote_project}/outputs/{RUN_NAME}/{host}/"
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude",
            "partial-checkpoint*.npz",
            "--exclude",
            "*.tmp.npz",
            "-e",
            ssh_transport,
            source,
            str(destination) + "/",
        ],
        check=False,
    )
    if host == "mali":
        main_destination = destination / "n224-main-resume"
        main_destination.mkdir(exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--exclude",
                "partial-checkpoint*.npz",
                "--exclude",
                "*.tmp.npz",
                "-e",
                ssh_transport,
                f"{host}:{remote_project}/outputs/paper-surrogate-n224-frontier/",
                str(main_destination) + "/",
            ],
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    identity = args.identity.resolve()
    coordinator_root = project / "outputs" / RUN_NAME / "coordinator"
    coordinator_root.mkdir(parents=True, exist_ok=True)
    status_path = coordinator_root / "status.json"
    state: dict[str, object] = {
        "owner": "kay",
        "started_at": _timestamp(),
        "deadline": args.deadline.isoformat(),
        "state": "launching",
        "workers": {},
    }
    _atomic_json(status_path, state)

    workers = state["workers"]
    assert isinstance(workers, dict)
    workers["kay"] = {
        "pid": _local_launch(project, args.deadline),
        "project": str(project),
        "launched_at": _timestamp(),
    }
    for host, remote_project in REMOTE_PROJECTS.items():
        try:
            workers[host] = {
                "pid": _remote_launch(identity, host, remote_project, args.deadline),
                "project": str(remote_project),
                "launched_at": _timestamp(),
            }
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            workers[host] = {
                "launch_error": f"{type(error).__name__}: {error}",
                "time": _timestamp(),
            }
    state["state"] = "running"
    _atomic_json(status_path, state)

    while datetime.now(args.deadline.tzinfo) < args.deadline:
        any_alive = False
        for host, record in workers.items():
            if not isinstance(record, dict) or "pid" not in record:
                continue
            alive = _alive(identity, host, int(record["pid"]))
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
    for host in workers:
        _stop_role(identity, host)
    stop_wait = min(
        args.deadline + timedelta(seconds=40),
        datetime.now(args.deadline.tzinfo) + timedelta(seconds=40),
    )
    while datetime.now(args.deadline.tzinfo) < stop_wait:
        if not any(
            isinstance(record, dict)
            and "pid" in record
            and _alive(identity, host, int(record["pid"]))
            for host, record in workers.items()
        ):
            break
        time.sleep(2)

    state["state"] = "collecting"
    _atomic_json(status_path, state)
    for host, remote_project in REMOTE_PROJECTS.items():
        _collect(identity, project, host, remote_project)

    collected_root = project / "outputs" / "fleet-overnight-collected"
    collected_root.mkdir(parents=True, exist_ok=True)
    local_source = project / "outputs" / RUN_NAME / "kay"
    local_destination = collected_root / "kay"
    if local_source.exists():
        shutil.copytree(
            local_source,
            local_destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("partial-checkpoint*.npz", "*.tmp.npz"),
        )
    state["state"] = "complete"
    state["finished_at"] = _timestamp()
    state["collected_root"] = str(collected_root)
    _atomic_json(status_path, state)


if __name__ == "__main__":
    main()
