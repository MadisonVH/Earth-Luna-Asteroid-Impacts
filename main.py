"""Entry point: show config GUI (no CLI args) or parse args, then run sim."""

from __future__ import annotations
import argparse
import sys

from config import SimConfig
from simulation import run_simulation
from visualization import SimViewer, plot_impact_analysis


def _build_config_from_gui() -> tuple:
    """Show config GUI; return (cfg, viewer, no_viewer, no_analysis)."""
    try:
        from launch_gui import show_config_gui
    except Exception:
        print("Warning: tkinter unavailable — falling back to CLI defaults.")
        return SimConfig(), "matplotlib", False, False

    gui = show_config_gui()
    if gui is None:
        return None, "", False, False     # user cancelled

    viewer      = gui.pop("_viewer", "matplotlib")
    no_analysis = bool(gui.pop("_no_analysis", False))
    no_viewer   = (viewer == "none")
    if no_viewer:
        viewer = "matplotlib"

    return SimConfig(**gui), viewer, no_viewer, no_analysis


def _build_config_from_args(argv) -> tuple:
    """Parse CLI arguments; return (cfg, viewer, no_viewer, no_analysis)."""
    p = argparse.ArgumentParser(description="Earth-Luna asteroid impact sim")
    p.add_argument("-n", "--n-asteroids", type=int, default=500)
    p.add_argument("-d", "--days", type=float, default=30.0,
                   help="Simulation duration in days (default 30)")
    p.add_argument("--dt", type=float, default=60.0, help="Timestep seconds")
    p.add_argument("--mode", choices=("montecarlo", "grid"), default="montecarlo")
    p.add_argument("--integrator", choices=("verlet", "euler", "rk4"),
                   default="verlet")
    p.add_argument("--frame", choices=("inertial", "com", "earth", "luna",
                                       "corotating"), default="corotating")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=None,
                   help="Save state every N steps (default 10; use 500+ "
                        "for large-N runs to keep memory manageable)")
    p.add_argument("--luna-target", type=float, default=None,
                   metavar="FRAC",
                   help="Fraction of asteroids biased toward Luna-crossing orbits "
                        "(default 0.3; set 0 to disable)")
    p.add_argument("--speed-range", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Asteroid speed range in km/s at source distance "
                        "(default 5 25).  Example: --speed-range 5 25")
    p.add_argument("--viewer", choices=("matplotlib", "pygame"), default="matplotlib",
                   help="Viewer backend (default: matplotlib; "
                        "use pygame for large asteroid counts with zoom/pan)")
    p.add_argument("--release-window", type=float, default=None,
                   metavar="DAYS",
                   help="Spread asteroid releases uniformly over this many days "
                        "(default 0=simultaneous). Use 27.32 to cover one full "
                        "lunar orbital period.")
    p.add_argument("--numeric-primaries", action="store_true",
                   help="Integrate Earth-Luna numerically instead of using the "
                        "exact trig solution (slower, accumulates error)")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip interactive viewer; show only impact analysis")
    p.add_argument("--no-analysis", action="store_true",
                   help="Skip impact analysis figure")
    args = p.parse_args(argv)

    cfg_kwargs: dict = dict(
        n_asteroids=args.n_asteroids,
        duration=args.days * 86400.0,
        dt=args.dt,
        init_mode=args.mode,
        integrator=args.integrator,
        reference_frame=args.frame,
        random_seed=args.seed,
    )
    if args.luna_target is not None:
        cfg_kwargs["luna_target_fraction"] = args.luna_target
    if args.speed_range is not None:
        cfg_kwargs["speed_range"] = (args.speed_range[0] * 1000.0,
                                     args.speed_range[1] * 1000.0)
    if args.save_every is not None:
        cfg_kwargs["save_every"] = args.save_every
    if args.release_window is not None:
        cfg_kwargs["release_window"] = args.release_window * 86400.0
    if args.numeric_primaries:
        cfg_kwargs["analytic_primaries"] = False

    return SimConfig(**cfg_kwargs), args.viewer, args.no_viewer, args.no_analysis


def main(argv: list[str] | None = None) -> int:
    # Show GUI when run interactively with no CLI arguments
    if argv is None and len(sys.argv) == 1:
        cfg, viewer, no_viewer, no_analysis = _build_config_from_gui()
        if cfg is None:
            return 0   # user clicked Cancel
    else:
        cfg, viewer, no_viewer, no_analysis = _build_config_from_args(argv)

    results = run_simulation(cfg)

    if not no_viewer:
        if viewer == "pygame":
            from viewer_pygame import PygameViewer
            PygameViewer(results).run()
        else:
            SimViewer(results).show()

    if not no_analysis:
        plot_impact_analysis(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
