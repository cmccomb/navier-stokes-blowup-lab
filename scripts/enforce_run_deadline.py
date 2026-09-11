"""One-shot controller-side deadline for one identified scientific process."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline must include a timezone")
    while True:
        expired = datetime.now(UTC) >= deadline
        script = f"""import json,os,signal,subprocess
from pathlib import Path
p=Path({args.run!r})
launch=json.loads((p/'launch.json').read_text())
pid=launch['pid']
observed=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
matches=observed.returncode==0 and str(p) in observed.stdout and 'scripts.check_forcing_timestep' in observed.stdout
stopped=False
if matches and {expired!r}:
 os.kill(pid,signal.SIGTERM)
 stopped=True
print(json.dumps(dict(matching_process=matches,terminated=stopped,pid=pid)))
"""
        try:
            result = subprocess.check_output(
                [
                    "ssh",
                    "-i",
                    str(args.identity),
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    args.host,
                    shlex.join(["python3", "-c", script]),
                ],
                text=True,
                timeout=30,
            )
            state = json.loads(result)
            state["checked_at"] = datetime.now(UTC).isoformat()
            state["deadline"] = deadline.isoformat()
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state), flush=True)
            if not state["matching_process"]:
                return
            # After SIGTERM, keep checking until the exact process is absent.
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {"checked_at": datetime.now(UTC).isoformat(), "error": str(exc)}
                ),
                flush=True,
            )
        time.sleep(10 if expired else 60)


if __name__ == "__main__":
    main()
