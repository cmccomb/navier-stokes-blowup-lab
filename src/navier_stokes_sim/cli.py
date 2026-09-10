"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import MESH_PRESETS, SimulationConfig, mesh_preset_parameters
from .interactive import write_interactive_volume
from .plotting import plot_animation, plot_diagnostics, plot_snapshots
from .solver import run_simulation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ns-blowup",
        description="Run a proof-informed PhiFlow experiment for the OpenAI Navier-Stokes result.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="even grid size in each dimension (default: 32)",
    )
    parser.add_argument(
        "--mesh-preset",
        choices=tuple(MESH_PRESETS),
        default="full-domain",
        help=(
            "static periodic box: full-domain or core-refined, which spends "
            "the same cells on a smaller box around the known core"
        ),
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=0.0,
        help="start time; nonzero values initialize directly from the target",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=0.94,
        help="stop before the normalized singular time t*=1",
    )
    parser.add_argument("--frames", type=int, default=25, help="number of saved frames")
    parser.add_argument(
        "--frame-spacing",
        choices=("linear", "similarity"),
        default="linear",
        help=(
            "saved-frame clock; similarity spacing adds samples near t*=1 "
            "for smoother late-time playback"
        ),
    )
    parser.add_argument(
        "--viscosity", type=float, default=0.01, help="kinematic viscosity"
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.008,
        help="anisotropy exponent; must be between 0 and 0.01",
    )
    parser.add_argument(
        "--paper-time-cutoff",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=(0.55, 0.775),
        help=(
            "C-infinity activation window for the paper profile; velocity is "
            "identically zero through START and fully active from END"
        ),
    )
    parser.add_argument(
        "--profile-model",
        choices=("paper-surrogate", "separable"),
        default="paper-surrogate",
        help="paper-coordinate target or the v0.3 separable baseline",
    )
    parser.add_argument(
        "--profile-interpolation",
        choices=("linear", "cubic"),
        default="cubic",
        help="C2 cubic profile tables, or legacy piecewise-linear ablation",
    )
    parser.add_argument(
        "--paper-axis-slope",
        type=float,
        default=4.0,
        help="slope in the paper's explicit U*=slope*eta+j0 axis datum",
    )
    parser.add_argument(
        "--paper-axis-offset",
        type=float,
        default=0.03,
        help="j0 offset in the paper's explicit axis datum (default: 0.03)",
    )
    parser.add_argument(
        "--no-pulses",
        action="store_true",
        help="disable the two annular wave families for an ablation run",
    )
    parser.add_argument(
        "--pulse-strengths",
        nargs=2,
        type=float,
        metavar=("RTHETA", "RZ"),
        default=(0.075, 0.060),
        help="strengths of the radial-angular and radial-axial wave families",
    )
    parser.add_argument(
        "--pulse-levels",
        type=int,
        default=3,
        help="number of retained multiscale pulse levels (default: 3)",
    )
    parser.add_argument(
        "--pulse-scale-ratio",
        type=float,
        default=0.68,
        help="amplitude ratio between successive pulse levels (default: 0.68)",
    )
    parser.add_argument(
        "--pulse-correction-strength",
        type=float,
        default=0.04,
        help="grid-deconvolution correction per pass (default: 0.04)",
    )
    parser.add_argument(
        "--pulse-correction-passes",
        type=int,
        default=1,
        help="number of bounded grid-correction passes (default: 1)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/default"), help="output directory"
    )
    parser.add_argument(
        "--no-animation", action="store_true", help="skip GIF rendering"
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="skip the interactive 3D field explorer"
    )
    parser.add_argument(
        "--capture-velocity-volumes",
        action="store_true",
        help="save 3D velocity checkpoints even when --no-3d defers rendering",
    )
    parser.add_argument(
        "--capture-force-volumes",
        action="store_true",
        help="save full 3D force vectors at selected checkpoints to forces.npz",
    )
    parser.add_argument(
        "--volume-frames",
        type=int,
        default=6,
        help="number of full-volume checkpoints, including both endpoints",
    )
    parser.add_argument(
        "--stream-volumes",
        action="store_true",
        help="write independent full-volume files instead of accumulating 3D history in RAM; use with --no-3d",
    )
    parser.add_argument(
        "--preview-phase-step",
        type=float,
        help="save lightweight 3D velocity and force frames on a separate forcing-phase clock (e.g. 0.3 radians)",
    )
    parser.add_argument(
        "--preview-resolution",
        type=int,
        default=32,
        help="maximum samples per axis for streamed display volumes",
    )
    parser.add_argument(
        "--derivative-epsilon",
        type=float,
        default=2e-4,
        help="maximum force time-difference half-window; hold fixed and small for timestep comparisons",
    )
    parser.add_argument(
        "--integrator",
        choices=("euler", "rk2"),
        default="rk2",
        help="projected time integrator (default: rk2)",
    )
    parser.add_argument(
        "--pressure-projection",
        choices=("auto", "sparse", "matrix-free", "fft"),
        default="auto",
        help="pressure path; fft is an exact periodic-grid Helmholtz projection",
    )
    parser.add_argument(
        "--forcing-end",
        type=float,
        help="disable the manufactured force at this time and continue freely",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.32,
        help="advective CFL limit (default: 0.32)",
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=0.02,
        help="maximum integration timestep (default: 0.02)",
    )
    parser.add_argument(
        "--forcing-phase-step",
        type=float,
        default=0.15,
        help="maximum retained forcing-phase advance in radians (default: 0.15)",
    )
    parser.add_argument(
        "--no-forcing-phase-limit",
        dest="forcing_phase_step",
        action="store_const",
        const=None,
        help="legacy numerical comparison only: disable the forcing clock limit",
    )
    parser.add_argument(
        "--save-final-state",
        action="store_true",
        help="persist the full 3D terminal velocity for cross-run comparisons",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore matching complete or per-frame checkpoints",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.stream_volumes and not args.no_3d:
        raise ValueError(
            "--stream-volumes requires --no-3d; render streamed histories separately"
        )
    mesh_parameters = mesh_preset_parameters(args.mesh_preset)
    cfg = SimulationConfig(
        resolution=args.resolution,
        mesh_preset=args.mesh_preset,
        **mesh_parameters,
        t_start=args.t_start,
        t_end=args.t_end,
        frames=args.frames,
        frame_spacing=args.frame_spacing,
        viscosity=args.viscosity,
        h=args.h,
        paper_time_cutoff_start=args.paper_time_cutoff[0],
        paper_time_cutoff_end=args.paper_time_cutoff[1],
        profile_model=args.profile_model,
        profile_interpolation=args.profile_interpolation,
        paper_axial_slope=args.paper_axis_slope,
        paper_axis_offset=args.paper_axis_offset,
        pulses_enabled=not args.no_pulses and args.profile_model == "paper-surrogate",
        pulse_rtheta_strength=args.pulse_strengths[0],
        pulse_rz_strength=args.pulse_strengths[1],
        pulse_hierarchy_levels=args.pulse_levels,
        pulse_scale_ratio=args.pulse_scale_ratio,
        pulse_correction_strength=args.pulse_correction_strength,
        pulse_correction_passes=args.pulse_correction_passes,
        integrator=args.integrator,
        pressure_projection=args.pressure_projection,
        forcing_end=args.forcing_end,
        cfl=args.cfl,
        max_dt=args.max_dt,
        forcing_phase_step=args.forcing_phase_step,
        capture_volumes=args.capture_velocity_volumes or not args.no_3d,
        capture_force_volumes=args.capture_force_volumes,
        volume_frames=args.volume_frames,
        stream_volumes=args.stream_volumes,
        preview_phase_step=args.preview_phase_step,
        preview_resolution=args.preview_resolution,
        derivative_epsilon=args.derivative_epsilon,
    )
    result = run_simulation(
        cfg,
        args.output,
        resume=not args.no_resume,
        save_final_state=args.save_final_state,
    )
    plot_diagnostics(result, result.output_dir / "diagnostics.png")
    plot_snapshots(result, result.output_dir / "snapshots.png")
    if not args.no_animation:
        plot_animation(result, result.output_dir / "blowup.gif")
    if not args.no_3d:
        write_interactive_volume(
            result, result.output_dir / "interactive-3d.html", field="velocity"
        )
    if args.capture_force_volumes and not args.no_3d:
        write_interactive_volume(
            result, result.output_dir / "interactive-force-3d.html", field="force"
        )
    last = result.diagnostics[-1]
    print(f"completed {len(result.times)} frames in {result.output_dir}")
    print(
        f"final t={last['t']:.4f}, peak speed={last['peak_speed']:.6g}, energy={last['kinetic_energy']:.6g}"
    )
    print(
        f"divergence Linf={last['divergence_linf']:.3e}, "
        f"tracking L2={last['tracking_relative_l2']:.3e}, "
        f"core-scale-covered={last['resolved']}"
    )
    print(
        f"peak vorticity={last['peak_vorticity']:.6g}, "
        f"enstrophy={last['enstrophy']:.6g}, BKM integral={last['bkm_integral']:.6g}"
    )
    print(
        f"component peaks (r,theta,z)=({last['peak_radial_speed']:.6g}, "
        f"{last['peak_azimuthal_speed']:.6g}, {last['peak_axial_speed']:.6g}), "
        f"spectral tail={last['spectral_tail_fraction']:.3%}"
    )


if __name__ == "__main__":
    main()
