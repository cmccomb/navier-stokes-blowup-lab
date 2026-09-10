"""Static figures and animation for a completed run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

plt.switch_backend("Agg")

from .interactive import write_interactive_volume
from .solver import SimulationResult


def _column(result: SimulationResult, name: str, dtype=float) -> np.ndarray:
    return np.asarray([row[name] for row in result.diagnostics], dtype=dtype)


def plot_diagnostics(result: SimulationResult, destination: Path) -> None:
    tau = _column(result, "tau")
    resolved = _column(result, "resolved", dtype=bool)
    cutoff = int(np.flatnonzero(~resolved)[0]) if np.any(~resolved) else len(tau)

    fig, axes = plt.subplots(4, 3, figsize=(16, 17), constrained_layout=True)
    ax = axes[0, 0]
    ax.loglog(tau[1:], _column(result, "peak_speed")[1:], "o-", label="PhiFlow")
    ax.loglog(
        tau[1:],
        _column(result, "target_peak_speed")[1:],
        "--",
        label="manufactured target",
    )
    theory = _column(result, "similarity_velocity")
    target_peak = _column(result, "target_peak_speed")
    anchors = np.flatnonzero(
        (result.times >= result.config.similarity_fit_start)
        & resolved
        & (target_peak > 0)
        & (theory > 0)
    )
    if len(anchors):
        anchor = int(anchors[0])
        theory = theory * target_peak[anchor] / theory[anchor]
    ax.loglog(tau[1:], theory[1:], ":", label=r"paper scale $\tau^{-1/2-h}$")
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau=1-t$",
        ylabel="peak speed",
        title="Velocity concentration",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    component_series = (
        ("peak_radial_speed", "radial", "similarity_radial_velocity", "o-"),
        ("peak_azimuthal_speed", "azimuthal", "similarity_velocity", "s-"),
        ("peak_axial_speed", "axial", "similarity_velocity", "^-"),
    )
    for measured_name, label, scale_name, style in component_series:
        measured = _column(result, measured_name)
        scale = _column(result, scale_name)
        anchors = np.flatnonzero(
            (result.times >= result.config.similarity_fit_start)
            & resolved
            & (measured > 0)
            & (scale > 0)
        )
        if len(anchors):
            anchor = int(anchors[0])
            scale = scale * measured[anchor] / scale[anchor]
        ax.loglog(tau[1:], measured[1:], style, label=label)
        ax.loglog(tau[1:], scale[1:], ":", alpha=0.65)
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau$",
        ylabel="component peak speed",
        title="Distinct component scalings",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[0, 2]
    measured_widths = (
        ("measured_radial_width", "radial RMS", "radial_length", "o-"),
        ("measured_axial_width", "axial RMS", "axial_length", "s-"),
    )
    for measured_name, label, scale_name, style in measured_widths:
        measured = _column(result, measured_name)
        scale = _column(result, scale_name)
        anchors = np.flatnonzero(
            (result.times >= result.config.similarity_fit_start)
            & resolved
            & (measured > 0)
        )
        if len(anchors):
            anchor = int(anchors[0])
            scale = scale * measured[anchor] / scale[anchor]
        ax.loglog(tau[1:], measured[1:], style, label=label)
        ax.loglog(tau[1:], scale[1:], ":", alpha=0.65)
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau$",
        ylabel="energy-weighted RMS width",
        title="Anisotropic core contraction",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    vorticity_law = _column(result, "similarity_vorticity")
    vorticity = _column(result, "peak_vorticity")
    if vorticity_law[-1] > 0 and vorticity[-1] > 0:
        vorticity_law = vorticity_law * vorticity[-1] / vorticity_law[-1]
    ax.loglog(tau[1:], vorticity[1:], "o-", label="PhiFlow")
    ax.loglog(tau[1:], vorticity_law[1:], ":", label=r"scaled $\tau^{-1-h}$")
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau$",
        ylabel="peak vorticity",
        title="Gradient amplification",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    energy = _column(result, "kinetic_energy")
    core_energy_name = (
        "target_kinetic_energy_core"
        if "target_kinetic_energy_core" in result.diagnostics[0]
        else "target_kinetic_energy"
    )
    target_energy = _column(result, core_energy_name)
    energy_law = _column(result, "similarity_energy_law")
    if energy_law[-1] > 0 and target_energy[-1] > 0:
        energy_law = energy_law * target_energy[-1] / energy_law[-1]
    simulated_core_name = (
        "kinetic_energy_core"
        if "kinetic_energy_core" in result.diagnostics[0]
        else "kinetic_energy"
    )
    simulated_core_energy = _column(result, simulated_core_name)
    ax.loglog(tau[1:], energy[1:], "o-", alpha=0.45, label="total PhiFlow")
    ax.loglog(tau[1:], simulated_core_energy[1:], "o-", label="core PhiFlow")
    ax.loglog(tau[1:], target_energy[1:], "--", label="core target")
    ax.loglog(tau[1:], energy_law[1:], ":", label=r"scaled $\tau^{1/2-3h}$")
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau$",
        ylabel="kinetic energy",
        title="Total and concentrating-core energy",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 2]
    enstrophy = _column(result, "enstrophy")
    enstrophy_law = _column(result, "similarity_enstrophy_law")
    if enstrophy_law[-1] > 0 and enstrophy[-1] > 0:
        enstrophy_law = enstrophy_law * enstrophy[-1] / enstrophy_law[-1]
    ax.loglog(tau[1:], enstrophy[1:], "o-", label="enstrophy")
    ax.loglog(tau[1:], enstrophy_law[1:], ":", label=r"scaled $\tau^{-1/2-3h}$")
    ax.invert_xaxis()
    ax.set(
        xlabel=r"time to singularity $\tau$",
        ylabel=r"$\frac{1}{2}\int |\omega|^2 dx$",
        title="Enstrophy growth",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.semilogy(
        result.times[1:], _column(result, "advective_l2")[1:], "o-", label="advection"
    )
    ax.semilogy(
        result.times[1:], _column(result, "viscous_l2")[1:], "s-", label="viscosity"
    )
    ax.semilogy(
        result.times[1:], _column(result, "force_l2")[1:], "^-", label="forcing"
    )
    ax.set(xlabel="time t", ylabel=r"term $L^2$ norm", title="Momentum balance")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    ax.semilogy(
        result.times[1:], _column(result, "force_l2")[1:], "o-", label=r"$\|f\|_{L^2}$"
    )
    ax.semilogy(
        result.times[1:],
        _column(result, "force_gradient_l2")[1:],
        "s-",
        label=r"$\|\nabla f\|_{L^2}$",
    )
    ax.semilogy(
        result.times[1:],
        _column(result, "force_laplacian_l2")[1:],
        "^-",
        label=r"$\|\Delta f\|_{L^2}$",
    )
    ax.semilogy(
        result.times[1:],
        _column(result, "force_time_derivative_l2")[1:],
        "d-",
        label=r"$\|\partial_t f\|_{L^2}$",
    )
    ax.set(xlabel="time t", ylabel="derivative norm", title="Force regularity audit")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[2, 2]
    ax.semilogy(
        result.times,
        np.maximum(np.abs(_column(result, "energy_balance_relative")), 1e-16),
        "o-",
        label="relative balance residual",
    )
    ax.semilogy(
        result.times,
        np.maximum(_column(result, "momentum_cancellation_ratio"), 1e-16),
        "s-",
        label="momentum cancellation ratio",
    )
    ax.set(xlabel="time t", ylabel="ratio", title="Balance and cancellation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[3, 0]
    ax.semilogy(
        result.times,
        _column(result, "divergence_linf"),
        "o-",
        label=r"$\|\nabla\cdot u\|_\infty$",
    )
    ax.semilogy(
        result.times,
        np.maximum(_column(result, "tracking_relative_l2"), 1e-16),
        "s-",
        label="relative tracking error",
    )
    ax.set(xlabel="time t", ylabel="error", title="Numerical constraints")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[3, 1]
    cells_r = _column(result, "cells_per_radial_scale")
    cells_z = _column(result, "cells_per_axial_scale")
    ax.plot(result.times, cells_r, "o-", label="radial scale / Δx")
    ax.plot(result.times, cells_z, "s-", label="axial scale / Δx")
    ax.axhline(
        result.config.min_cells_per_scale,
        color="black",
        linestyle=":",
        label="resolution threshold",
    )
    if cutoff < len(tau):
        ax.axvspan(
            result.times[cutoff],
            result.times[-1],
            color="tab:red",
            alpha=0.08,
            label="under-resolved",
        )
    ax.set(xlabel="time t", ylabel="grid cells", title="Resolution audit")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[3, 2]
    ax.plot(
        result.times[1:],
        _column(result, "spectral_mean_mode")[1:],
        "o-",
        label="energy-weighted mode",
    )
    ax.plot(
        result.times[1:],
        _column(result, "spectral_k95")[1:],
        "s-",
        label="95% energy mode",
    )
    ax.axhline(
        result.config.resolution / 3,
        color="black",
        linestyle=":",
        label="top-third threshold",
    )
    ax.set(
        xlabel="time t",
        ylabel="Fourier mode magnitude",
        title="Spectral occupancy",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    experiment = (
        "forced reference"
        if result.config.forcing_end is None
        else f"force released at t={result.config.forcing_end:g}"
    )
    fig.suptitle(
        f"OpenAI Navier–Stokes {result.config.profile_model} experiment · "
        f"{experiment} · {result.config.integrator.upper()}"
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_snapshots(result: SimulationResult, destination: Path) -> None:
    resolved = _column(result, "resolved", dtype=bool)
    resolved_indices = np.flatnonzero(resolved)
    last_resolved = (
        int(resolved_indices[-1]) if len(resolved_indices) else len(result.times) // 2
    )
    indices = sorted({0, last_resolved, len(result.times) - 1})
    extent = (-result.config.half_domain, result.config.half_domain) * 2
    axial_speed = np.linalg.norm(result.velocity_slices, axis=-1)
    equatorial_speed = np.linalg.norm(result.equatorial_slices, axis=-1)
    vmax = max(
        float(np.percentile(axial_speed[-1], 99.5)),
        float(np.percentile(equatorial_speed[-1], 99.5)),
        1e-12,
    )
    fig, axes = plt.subplots(
        2, len(indices), figsize=(13, 8), constrained_layout=True, squeeze=False
    )
    image = None
    axis = (
        np.linspace(
            -result.config.half_domain,
            result.config.half_domain,
            result.config.resolution,
            endpoint=False,
        )
        + result.config.dx / 2
    )
    stride = max(result.config.resolution // 16, 1)
    for column, idx in enumerate(indices):
        ax = axes[0, column]
        image = ax.imshow(
            axial_speed[idx].T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )
        stride = max(result.config.resolution // 16, 1)
        velocity = result.velocity_slices[idx]
        if np.max(np.abs(velocity)) > 0:
            ax.quiver(
                axis[::stride],
                axis[::stride],
                velocity[::stride, ::stride, 0].T,
                velocity[::stride, ::stride, 2].T,
                color="white",
                alpha=0.65,
                angles="xy",
                scale_units="xy",
                scale=5,
                width=0.004,
            )
        row = result.diagnostics[idx]
        state = (
            "at rest"
            if result.times[idx] == 0
            else ("resolved" if row["resolved"] else "under-resolved")
        )
        ax.set(title=f"t={result.times[idx]:.3f} · {state}", xlabel="x", ylabel="z")

        ax = axes[1, column]
        image = ax.imshow(
            equatorial_speed[idx].T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )
        equatorial = result.equatorial_slices[idx]
        if np.max(np.abs(equatorial)) > 0:
            ax.quiver(
                axis[::stride],
                axis[::stride],
                equatorial[::stride, ::stride, 0].T,
                equatorial[::stride, ::stride, 1].T,
                color="white",
                alpha=0.65,
                angles="xy",
                scale_units="xy",
                scale=5,
                width=0.004,
            )
        ax.set(xlabel="x", ylabel="y")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="speed |u|", shrink=0.82)
    fig.suptitle("Central slices: axial stretching (x–z) and inward swirl (x–y)")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_animation(result: SimulationResult, destination: Path) -> None:
    speeds = np.linalg.norm(result.velocity_slices, axis=-1)
    extent = (-result.config.half_domain, result.config.half_domain) * 2
    vmax = max(float(np.percentile(speeds[-1], 99.5)), 1e-12)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(
        speeds[0].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax
    )
    title = ax.set_title("")
    ax.set(xlabel="x", ylabel="z")
    fig.colorbar(image, ax=ax, label="speed |u|")

    def update(frame: int):
        image.set_data(speeds[frame].T)
        row = result.diagnostics[frame]
        state = "resolved" if row["resolved"] else "under-resolved"
        title.set_text(f"t={result.times[frame]:.3f}, τ={row['tau']:.3f} · {state}")
        return image, title

    animation = FuncAnimation(
        fig, update, frames=len(result.times), interval=160, blit=False
    )
    animation.save(destination, writer=PillowWriter(fps=6))
    plt.close(fig)


def create_plots(result: SimulationResult) -> None:
    plot_diagnostics(result, result.output_dir / "diagnostics.png")
    plot_snapshots(result, result.output_dir / "snapshots.png")
    plot_animation(result, result.output_dir / "blowup.gif")
    write_interactive_volume(result, result.output_dir / "interactive-3d.html")
    if result.volume_force is not None:
        write_interactive_volume(
            result, result.output_dir / "interactive-force-3d.html", field="force"
        )
