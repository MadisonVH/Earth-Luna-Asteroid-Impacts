"""Entry point: run a sim and open the interactive viewer."""

from __future__ import annotations
import argparse
import sys

from config import SimConfig
from simulation import run_simulation
from visualization import SimViewer, plot_impact_analysis


def main(argv: list[str] | None = None) -> int:
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
    p.add_argument("--luna-target", type=float, default=None,
                   metavar="FRAC",
                   help="Fraction of asteroids biased toward Luna-crossing orbits "
                        "(default 0.3; set 0 to disable)")
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
    cfg = SimConfig(**cfg_kwargs)

    results = run_simulation(cfg)

    if not args.no_viewer:
        viewer = SimViewer(results)
        viewer.show()

    if not args.no_analysis:
        plot_impact_analysis(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
