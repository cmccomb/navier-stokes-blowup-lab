"""Validated spatial samples and scalar measures for 3D time histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import SimulationConfig

Component = Literal["magnitude", "x", "y", "z"]
COMPONENTS = ("magnitude", "x", "y", "z")


def scalar_values(vectors: np.ndarray, component: Component) -> np.ndarray:
    if component == "magnitude":
        # Float32 archives can retain tiny nonzero components whose squares
        # underflow in float32 during a naive norm calculation.
        return np.linalg.norm(vectors.astype(np.float64), axis=-1)
    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    return vectors[..., ("x", "y", "z").index(component)]


@dataclass
class VolumeSeries:
    vectors: np.ndarray
    times: np.ndarray
    axis: np.ndarray
    config: SimulationConfig
    field: str

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors)
        self.times = np.asarray(self.times, dtype=float)
        self.axis = np.asarray(self.axis, dtype=float)
        n = len(self.axis)
        if self.field not in {"velocity", "force"}:
            raise ValueError("field must be velocity or force")
        if self.vectors.shape != (len(self.times), n, n, n, 3):
            raise ValueError("vector shape does not match time and coordinate axes")
        if not len(self.times) or n < 2:
            raise ValueError("at least one volume and two spatial samples are required")
        if not np.all(np.diff(self.times) > 0) or not np.all(np.diff(self.axis) > 0):
            raise ValueError("times and spatial coordinates must increase strictly")
        if not all(np.isfinite(a).all() for a in (self.vectors, self.times, self.axis)):
            raise ValueError("nonfinite volume data cannot be rendered")
        if self.times[0] < 0 or self.times[-1] >= self.config.t_star:
            raise ValueError("volume times must precede t_star")
        if np.max(np.abs(self.axis)) > self.config.half_domain:
            raise ValueError("spatial samples lie outside the source domain")

    @property
    def symbol(self) -> str:
        return "u" if self.field == "velocity" else "f"

    @property
    def title(self) -> str:
        return "Velocity" if self.field == "velocity" else "Applied force"

    def peak(self, component: Component) -> float:
        peak = max(
            float(np.max(np.abs(scalar_values(v, component)))) for v in self.vectors
        )
        return peak if peak > 0 else 1.0
