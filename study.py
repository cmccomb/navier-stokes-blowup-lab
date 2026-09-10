"""Workspace entry point for the convergence and force-release study."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
main = import_module("navier_stokes_sim.study").main


if __name__ == "__main__":
    main()
