"""One bounded controller: render on the archive host, publish compact files.

Run in the controller's authenticated login session. No credentials or dense
archives cross machines. The source host performs only serial one-shot exports.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.stream_run import (
    RENDER_REVISION,
    command,
    package_outputs,
    publish,
    read_remote,
    verify_gif,
)
from scripts.volume_history import embedded_history, validate_history


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def observation_key(record: dict) -> list:
    return [
        record["id"],
        record["captured_frames"],
        record["latest_t"],
        record["status"],
        (record.get("diagnostics") or {}).get("t"),
    ]


def validate_package(folder: Path, observed: dict, previous: dict | None) -> dict:
    """Reject another run, nonfinite output, partial files, or time regression."""
    run = json.loads((folder / "site/data/stream.json").read_text())
    for key in ("id", "source_commit", "config"):
        if run[key] != observed[key]:
            raise ValueError(f"rendered {key} does not match the observed run")
    archive = run["archive"]
    if (
        archive["resolution"] != run["config"]["resolution"]
        or archive["dtype"] != "float32"
        or archive["fields"] != ["velocity", "force"]
        or archive["components_per_field"] != 3
    ):
        raise ValueError("expected both native float32 vector fields")
    times = run["clip_times"]
    if (
        not times
        or not all(math.isfinite(t) for t in times)
        or any(b <= a for a, b in pairwise(times))
        or run["latest_t"] != times[-1]
        or run["captured_frames"] != len(times)
        or times[-1] < observed["latest_t"]
        or times[0] != run["config"]["t_start"]
        or times[-1] > run["config"]["t_end"]
    ):
        raise ValueError("invalid or stale saved frame times")
    if (
        previous
        and previous["id"] == run["id"]
        and (
            run["latest_t"] < previous["latest_t"]
            or run["captured_frames"] < previous["captured_frames"]
        )
    ):
        raise ValueError("publication must not regress")
    diagnostics = run.get("diagnostics") or {}
    if not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in diagnostics.values()
    ):
        raise ValueError("nonfinite diagnostics")
    outputs = package_outputs(run)
    for filename in outputs:
        path = folder / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty compact output: {filename}")
        if path.stat().st_size >= 100 * 1024**2:
            raise ValueError(f"site output exceeds GitHub's per-file limit: {filename}")
    for field, name in (("velocity", "flow"), ("force", "force")):
        verified = verify_gif(
            folder / f"site/media/stream-{name}.gif",
            run["playback"]["source_frame_duration_ms"],
        )
        if verified != run["render"][field]["gif_verification"] or verified[
            "frames"
        ] != len(times):
            raise ValueError("encoded GIF does not match the complete saved history")
        history = run["render"][field]["volume_history"]
        if (
            history["source_resolution"] != run["archive"]["resolution"]
            or len(history["axis"]) != run["display_resolution_3d"]
            or embedded_history(folder / f"site/media/stream-{name}-3d.html") != history
        ):
            raise ValueError("3D viewer does not match the published history")
        validate_history(folder, history, field, times)
    return run


def render_and_collect(
    args, ssh: list[str], observed: dict, previous: dict | None
) -> dict:
    source_repo = Path(args.source_repo)
    argv = [
        str(source_repo / ".venv/bin/python"),
        "-m",
        "scripts.stream_run",
        "--host",
        "local",
        "--once",
        "--no-push",
        "--run",
        args.run,
        "--repo",
        str(source_repo),
        "--cache",
        str(source_repo / ".publish-cache"),
        "--deadline",
        args.deadline,
    ]
    remote_command = f"cd {shlex.quote(str(source_repo))} && {shlex.join(argv)}"
    command([*ssh, args.host, remote_command], timeout=900)
    descriptor = json.loads(
        command(
            [
                *ssh,
                args.host,
                shlex.join(["/bin/cat", str(source_repo / "site/data/stream.json")]),
            ]
        )
    )
    for key in ("id", "source_commit", "config"):
        if descriptor[key] != observed[key]:
            raise ValueError(f"source export {key} changed before collection")
    outputs = package_outputs(descriptor)
    allowlist = args.cache / "compact-files.txt"
    allowlist.write_text("\n".join(outputs) + "\n")
    with tempfile.TemporaryDirectory(dir=args.cache, prefix="incoming-") as temporary:
        incoming = Path(temporary)
        command(
            [
                "rsync",
                "-az",
                "--files-from",
                str(allowlist),
                "-e",
                shlex.join(ssh),
                f"{args.host}:{source_repo}/",
                str(incoming) + "/",
            ],
            timeout=300,
        )
        run = validate_package(incoming, observed, previous)
        if package_outputs(run) != outputs:
            raise ValueError("source export changed during collection")
        for filename in outputs:
            destination = args.repo / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_suffix(destination.suffix + ".publish-tmp")
            shutil.copyfile(incoming / filename, pending)
            pending.replace(destination)
    return run


def check_live(url: str, receipt: dict) -> bool:
    request = Request(
        f"{url.rstrip('/')}/data/stream.json?published={receipt['commit']}",
        headers={"Cache-Control": "no-cache", "User-Agent": "NS-result-publisher"},
    )
    with urlopen(request, timeout=20) as response:
        live = json.load(response)
    return (
        live["id"] == receipt["id"]
        and live["revision"] == receipt["revision"]
        and observation_key(live) == receipt["observation_key"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline <= datetime.now(UTC):
        parser.error("deadline must be a future time with a timezone")
    args.cache.mkdir(parents=True, exist_ok=True)
    lock = (args.cache / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh = [
        "ssh",
        "-i",
        str(args.identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    receipt_path = args.cache / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    expected_id = Path(args.run).name
    if receipt and receipt["id"] != expected_id:
        parser.error("cache belongs to another run")
    failures = 0
    while datetime.now(UTC) < deadline:
        try:
            observed = read_remote(ssh, args.host, args.run)
            if observed["id"] != expected_id:
                raise ValueError("source run identity changed")
            if observed["latest_t"] is None:
                raise RuntimeError("no finalized source frame yet")
            if (
                receipt is None
                or receipt.get("render_revision") != RENDER_REVISION
                or observation_key(observed) != receipt["observation_key"]
            ):
                run = render_and_collect(args, ssh, observed, receipt)
                commit = publish(
                    args.repo,
                    f"Live native frame {run['captured_frames']}: t={run['latest_t']:.6f}",
                )
                receipt = {
                    "id": run["id"],
                    "commit": commit,
                    "revision": run["revision"],
                    "render_revision": run["render_revision"],
                    "captured_frames": run["captured_frames"],
                    "latest_t": run["latest_t"],
                    "status": run["status"],
                    "observation_key": observation_key(run),
                    "pushed_at": datetime.now(UTC).isoformat(),
                    "deployment_status": "pending",
                }
                write_json(receipt_path, receipt)
                print(json.dumps(receipt), flush=True)
            if receipt["deployment_status"] != "live" and check_live(
                args.site_url, receipt
            ):
                receipt.update(
                    deployment_status="live", live_at=datetime.now(UTC).isoformat()
                )
                write_json(receipt_path, receipt)
                print(json.dumps(receipt), flush=True)
            write_json(
                args.cache / "status.json",
                {
                    "checked_at": datetime.now(UTC).isoformat(),
                    "state": "watching",
                    "source": observation_key(observed),
                    "receipt": receipt,
                    "poll_seconds": args.poll_seconds,
                    "deadline": args.deadline,
                },
            )
            failures = 0
            if (
                observed["status"] in ("complete", "stopped")
                and receipt["deployment_status"] == "live"
            ):
                return
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            failures += 1
            failure = {
                "failed_at": datetime.now(UTC).isoformat(),
                "state": "retrying",
                "consecutive_failures": failures,
                "error": str(exc),
            }
            write_json(args.cache / "failure.json", failure)
            write_json(args.cache / "status.json", failure)
            print(json.dumps(failure), flush=True)
        time.sleep(min(300, max(5, args.poll_seconds) * 2 ** min(failures, 3)))
    write_json(
        args.cache / "status.json",
        {"state": "deadline-reached", "at": datetime.now(UTC).isoformat()},
    )


if __name__ == "__main__":
    main()
