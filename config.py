"""Simulation configuration — tweak here, not in the physics code."""

from dataclasses import dataclass, field
from typing import Tuple
from constants import EARTH_LUNA_DIST


@dataclass
class SimConfig:
    # ── Time ────────────────────────────────────────────────────────────────
    dt: float = 60.0               # integration timestep, seconds
    duration: float = 86400 * 30   # total sim time, seconds (default 30 days)
    save_every: int = 10           # save state every N steps

    # ── Asteroids ───────────────────────────────────────────────────────────
    n_asteroids: int = 500
    # "montecarlo": random (angle, impact-parameter, speed)
    # "grid":       systematic grid of (approach-angle × impact-parameter)
    init_mode: str = "montecarlo"

    # common: starting distance of asteroids from the CoM
    source_distance: float = 4.0 * EARTH_LUNA_DIST   # m
    # common: max transverse offset from the approach centre-line
    b_max: float = 2.0 * EARTH_LUNA_DIST              # m
    # remove asteroid when this far from CoM
    escape_distance: float = 8.0 * EARTH_LUNA_DIST    # m
    # Spread asteroid release times uniformly over this window (seconds).
    # 0 = all released simultaneously at t=0.
    # Set to one lunar period (~2,360,592 s) to sample all orbital phases.
    release_window: float = 0.0

    # Monte Carlo
    speed_range: Tuple[float, float] = (300.0, 4000.0)   # m/s
    angle_range: Tuple[float, float] = (0.0, 360.0)       # degrees (approach direction)
    # Fraction of MC asteroids biased toward Luna-crossing trajectories.
    # For each biased asteroid the impact parameter is drawn near b = -r_L·sin(θ),
    # which is the straight-line aim point for Luna at approach angle θ.
    luna_target_fraction: float = 0.3
    luna_target_sigma: float = 30.0   # spread in units of Luna radii

    # Grid (n_asteroids is overridden to n_angles * n_b_vals)
    n_angles: int = 20        # approach directions
    n_b_vals: int = 25        # impact-parameter samples
    grid_speed: float = 1500.0  # m/s, fixed speed for grid mode

    # ── Integrator ──────────────────────────────────────────────────────────
    # "verlet" (Velocity Verlet, recommended), "euler", "rk4"
    integrator: str = "verlet"
    # Use the exact trig solution for Earth & Luna instead of numerically
    # integrating them.  Valid because our ICs are exactly circular (e=0),
    # so Kepler's parametric equation collapses to:
    #   earth(t) = (-r_E·cos ωt, -r_E·sin ωt)
    #   luna(t)  = (+r_L·cos ωt, +r_L·sin ωt)
    # This gives zero accumulated error in the primary positions for all t.
    analytic_primaries: bool = True

    # ── Reference frame (visualization only) ────────────────────────────────
    # "inertial", "com", "earth", "luna", "corotating"
    reference_frame: str = "corotating"

    # ── Visualization ────────────────────────────────────────────────────────
    show_field_lines: bool = True    # equipotential contours
    show_trails: bool = True
    trail_length: int = 80           # trail length in saved timesteps
    # bodies are drawn at body_scale × their real radius for visibility
    body_scale: float = 8.0
    field_grid_n: int = 60           # NxN grid for potential field

    # ── Output ───────────────────────────────────────────────────────────────
    save_trajectories: bool = True   # False → only impacts saved (large N)
    random_seed: int = 42

    def n_steps(self) -> int:
        return int(self.duration / self.dt)

    def n_saved(self) -> int:
        return self.n_steps() // self.save_every + 1

    def effective_n_asteroids(self) -> int:
        if self.init_mode == "grid":
            return self.n_angles * self.n_b_vals
        return self.n_asteroids
