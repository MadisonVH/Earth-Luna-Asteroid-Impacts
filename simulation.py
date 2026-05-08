"""
Core simulation engine.

Earth and Luna interact bidirectionally (full two-body gravity).
Asteroids are pure test particles: they feel Earth + Luna gravity but exert
none.  All integration is in SI units.  Velocity Verlet is the default
(symplectic, energy-conserving).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time as _time

from constants import G, EARTH_MASS, EARTH_RADIUS, LUNA_MASS, LUNA_RADIUS, EARTH_LUNA_DIST, EARTH_LUNA_OMEGA
from config import SimConfig


# ── Result structures ────────────────────────────────────────────────────────

@dataclass
class ImpactRecord:
    asteroid_idx: int
    time: float           # seconds since t=0
    body: str             # "earth" or "luna"
    pos: np.ndarray       # (2,) inertial-frame position at impact
    # Angle of impact measured from the sub-Earth (luna) or sub-Luna (earth)
    # direction at the moment of impact.
    # For luna: 0 = near-side (Earth-facing), ±π = far-side.
    angle: float          # radians


@dataclass
class SimResults:
    config: SimConfig
    omega: float           # Earth-Luna angular velocity (rad/s)

    # Earth & Luna trajectories — (n_saved, 2), SI metres
    times: np.ndarray
    earth_pos: np.ndarray
    luna_pos: np.ndarray
    earth_vel: np.ndarray
    luna_vel: np.ndarray

    # Asteroid initial conditions — (n_ast, 2)
    ast_init_pos: np.ndarray
    ast_init_vel: np.ndarray

    # Full trajectory history — (n_ast, n_saved, 2), NaN after impact/escape
    # None when config.save_trajectories is False
    ast_pos: Optional[np.ndarray]
    ast_vel: Optional[np.ndarray]

    # Impact records (all of them, unsorted)
    impacts: List[ImpactRecord]

    # Per-asteroid outcome: 0=active-at-end, 1=earth, 2=luna, 3=escaped
    status: np.ndarray   # (n_ast,) int8


# ── Force computation ────────────────────────────────────────────────────────

def _el_acc(el_pos: np.ndarray) -> np.ndarray:
    """Gravitational accelerations for the Earth-Luna two-body system.

    el_pos : (2, 2)  — row 0 = Earth, row 1 = Luna, in metres
    returns : (2, 2) accelerations in m/s²
    """
    r = el_pos[1] - el_pos[0]          # vector Earth → Luna
    d3 = np.linalg.norm(r) ** 3
    acc = np.empty_like(el_pos)
    acc[0] = G * LUNA_MASS * r / d3    # Earth pulled toward Luna
    acc[1] = -G * EARTH_MASS * r / d3  # Luna pulled toward Earth
    return acc


def _ast_acc(ast_pos: np.ndarray,
             earth_pos: np.ndarray,
             luna_pos: np.ndarray) -> np.ndarray:
    """Gravitational acceleration on N asteroids from Earth and Luna.

    ast_pos   : (N, 2)
    earth_pos : (2,)
    luna_pos  : (2,)
    returns   : (N, 2) accelerations in m/s²

    Uses a tiny softening length (1 km) to avoid division-by-zero for
    asteroids extremely close to a body centre; impacts are detected and
    removed before this matters in practice.
    """
    eps2 = 1e6 ** 2   # (1000 m)^2

    dr_e = earth_pos - ast_pos                              # (N, 2)
    dr_l = luna_pos - ast_pos                               # (N, 2)
    d2_e = np.einsum("ij,ij->i", dr_e, dr_e)[:, None] + eps2   # (N, 1)
    d2_l = np.einsum("ij,ij->i", dr_l, dr_l)[:, None] + eps2   # (N, 1)

    a_e = G * EARTH_MASS * dr_e / d2_e ** 1.5              # (N, 2)
    a_l = G * LUNA_MASS * dr_l / d2_l ** 1.5               # (N, 2)
    return a_e + a_l


# ── Integrators ──────────────────────────────────────────────────────────────

def _step_verlet(el_pos, el_vel, el_acc_cur,
                 ast_pos, ast_vel, ast_acc_cur, active, dt):
    """Velocity Verlet — symplectic, O(dt²) global error."""
    # 1. Position half-step
    el_pos_new = el_pos + el_vel * dt + 0.5 * el_acc_cur * dt ** 2
    if active.any():
        ast_pos[active] += ast_vel[active] * dt + 0.5 * ast_acc_cur[active] * dt ** 2

    # 2. New accelerations at new positions
    el_acc_new = _el_acc(el_pos_new)
    ast_acc_new = np.zeros_like(ast_acc_cur)
    if active.any():
        ast_acc_new[active] = _ast_acc(ast_pos[active], el_pos_new[0], el_pos_new[1])

    # 3. Velocity full-step
    el_vel_new = el_vel + 0.5 * (el_acc_cur + el_acc_new) * dt
    if active.any():
        ast_vel[active] += 0.5 * (ast_acc_cur[active] + ast_acc_new[active]) * dt

    return el_pos_new, el_vel_new, el_acc_new, ast_acc_new


def _step_euler(el_pos, el_vel, el_acc_cur,
                ast_pos, ast_vel, ast_acc_cur, active, dt):
    el_pos_new = el_pos + el_vel * dt
    el_vel_new = el_vel + el_acc_cur * dt
    if active.any():
        ast_pos[active] += ast_vel[active] * dt
        ast_vel[active] += ast_acc_cur[active] * dt
    el_acc_new = _el_acc(el_pos_new)
    ast_acc_new = np.zeros_like(ast_acc_cur)
    if active.any():
        ast_acc_new[active] = _ast_acc(ast_pos[active], el_pos_new[0], el_pos_new[1])
    return el_pos_new, el_vel_new, el_acc_new, ast_acc_new


def _step_rk4(el_pos, el_vel, el_acc_cur,
               ast_pos, ast_vel, ast_acc_cur, active, dt):
    """RK4 — O(dt⁴) local error, but not symplectic."""
    def derivs_el(p, v):
        a = _el_acc(p)
        return v, a

    def derivs_ast(p_active, v_active, ep, lp):
        a = _ast_acc(p_active, ep, lp)
        return v_active, a

    # Earth-Luna RK4
    k1_r, k1_v = derivs_el(el_pos, el_vel)
    k2_r, k2_v = derivs_el(el_pos + 0.5*dt*k1_r, el_vel + 0.5*dt*k1_v)
    k3_r, k3_v = derivs_el(el_pos + 0.5*dt*k2_r, el_vel + 0.5*dt*k2_v)
    k4_r, k4_v = derivs_el(el_pos + dt*k3_r, el_vel + dt*k3_v)
    el_pos_new = el_pos + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    el_vel_new = el_vel + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    el_acc_new = _el_acc(el_pos_new)

    # Asteroids RK4 (use mid-point Earth-Luna positions as approximation)
    ast_acc_new = np.zeros_like(ast_acc_cur)
    if active.any():
        pa = ast_pos[active].copy()
        va = ast_vel[active].copy()
        ep_mid = (el_pos[0] + el_pos_new[0]) / 2
        lp_mid = (el_pos[1] + el_pos_new[1]) / 2

        ak1_r, ak1_v = va, _ast_acc(pa, el_pos[0], el_pos[1])
        ak2_r, ak2_v = va + 0.5*dt*ak1_v, _ast_acc(pa + 0.5*dt*ak1_r, ep_mid, lp_mid)
        ak3_r, ak3_v = va + 0.5*dt*ak2_v, _ast_acc(pa + 0.5*dt*ak2_r, ep_mid, lp_mid)
        ak4_r, ak4_v = va + dt*ak3_v, _ast_acc(pa + dt*ak3_r, el_pos_new[0], el_pos_new[1])

        ast_pos[active] += (dt/6) * (ak1_r + 2*ak2_r + 2*ak3_r + ak4_r)
        ast_vel[active] += (dt/6) * (ak1_v + 2*ak2_v + 2*ak3_v + ak4_v)
        ast_acc_new[active] = _ast_acc(ast_pos[active], el_pos_new[0], el_pos_new[1])

    return el_pos_new, el_vel_new, el_acc_new, ast_acc_new


_INTEGRATORS = {
    "verlet": _step_verlet,
    "euler": _step_euler,
    "rk4": _step_rk4,
}


# ── Initial conditions ────────────────────────────────────────────────────────

def _init_earth_luna() -> tuple[np.ndarray, np.ndarray]:
    """Initial conditions for a circular Earth-Luna orbit with CoM at origin.

    Returns el_pos (2,2), el_vel (2,2).
    """
    M = EARTH_MASS + LUNA_MASS
    r_e = LUNA_MASS * EARTH_LUNA_DIST / M    # Earth distance from CoM
    r_l = EARTH_MASS * EARTH_LUNA_DIST / M   # Luna distance from CoM
    omega = EARTH_LUNA_OMEGA

    # Luna at +x, Earth at -x; counter-clockwise orbit
    #   At (-r_e, 0): CCW tangent is (0, -v_e)  [check: r×v = (-r_e)(-v_e)>0 ✓]
    #   At (+r_l, 0): CCW tangent is (0, +v_l)
    el_pos = np.array([
        [-r_e, 0.0],   # Earth
        [+r_l, 0.0],   # Luna
    ])
    el_vel = np.array([
        [0.0, -omega * r_e],   # Earth
        [0.0, +omega * r_l],   # Luna
    ])
    return el_pos, el_vel


def _init_asteroids(config: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_ast, 2) initial positions and velocities for asteroids.

    Asteroids start at distance config.source_distance from the CoM,
    offset transversely by impact parameter b, moving toward the CoM.

    Position = -R * e_approach + b * e_perp
    Velocity  =  v * e_approach

    where e_approach is the unit vector of the approach direction,
    e_perp is the perpendicular unit vector (CCW from e_approach).
    """
    rng = np.random.default_rng(config.random_seed)
    R = config.source_distance

    if config.init_mode == "montecarlo":
        n = config.n_asteroids
        lo, hi = config.angle_range
        theta = np.deg2rad(rng.uniform(lo, hi, n))
        b = rng.uniform(-config.b_max, config.b_max, n)
        v = rng.uniform(*config.speed_range, n)

        # Bias a fraction of asteroids toward Luna-crossing trajectories.
        # For approach direction θ, the impact parameter that aims straight at
        # Luna (at CoM-distance r_L along +x at t=0) is b = −r_L·sin(θ).
        # We scatter around that value with spread luna_target_sigma × R_Luna.
        frac = config.luna_target_fraction
        if frac > 0:
            n_bias = int(n * frac)
            if n_bias > 0:
                r_L = EARTH_MASS * EARTH_LUNA_DIST / (EARTH_MASS + LUNA_MASS)
                b_luna = -r_L * np.sin(theta[:n_bias])
                sigma = config.luna_target_sigma * LUNA_RADIUS
                b[:n_bias] = rng.normal(b_luna, sigma)

    elif config.init_mode == "grid":
        thetas_1d = np.linspace(0, 2 * np.pi, config.n_angles, endpoint=False)
        bs_1d = np.linspace(-config.b_max, config.b_max, config.n_b_vals)
        theta_g, b_g = np.meshgrid(thetas_1d, bs_1d)
        theta = theta_g.ravel()
        b = b_g.ravel()
        v = np.full(len(theta), config.grid_speed)

    else:
        raise ValueError(f"Unknown init_mode: {config.init_mode!r}")

    e_app = np.column_stack([np.cos(theta), np.sin(theta)])     # (N, 2) approach direction
    e_perp = np.column_stack([-np.sin(theta), np.cos(theta)])   # (N, 2) CCW perpendicular

    ast_pos = -R * e_app + b[:, None] * e_perp
    ast_vel = v[:, None] * e_app

    return ast_pos.astype(np.float64), ast_vel.astype(np.float64)


# ── Impact detection ──────────────────────────────────────────────────────────

def _compute_impact_angle(hit_pos: np.ndarray,
                          body_pos: np.ndarray,
                          ref_pos: np.ndarray) -> float:
    """Angle of impact measured from the direction body → ref_pos.

    For a Luna impact: ref_pos = earth_pos  →  0 = near-side, ±π = far-side.
    """
    hit_vec = hit_pos - body_pos
    ref_vec = ref_pos - body_pos
    cross = ref_vec[0] * hit_vec[1] - ref_vec[1] * hit_vec[0]
    dot = np.dot(ref_vec, hit_vec)
    return float(np.arctan2(cross, dot))


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(config: SimConfig) -> SimResults:
    """Run the full Earth-Luna-asteroid simulation and return results."""
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(it, **kw):   # bare fallback if tqdm not installed
            return it

    step_fn = _INTEGRATORS[config.integrator]
    n_steps = config.n_steps()
    n_saved = config.n_saved()
    n_ast = config.effective_n_asteroids()
    dt = config.dt
    analytic = config.analytic_primaries

    # Pre-compute primary orbital radii for the analytic solution
    _M = EARTH_MASS + LUNA_MASS
    _r_E = LUNA_MASS  * EARTH_LUNA_DIST / _M   # Earth distance from CoM
    _r_L = EARTH_MASS * EARTH_LUNA_DIST / _M   # Luna distance from CoM
    _omega = EARTH_LUNA_OMEGA

    print(f"Sim: {n_steps} steps × {dt:.0f} s dt | "
          f"{n_ast} asteroids | integrator={config.integrator} | "
          f"primaries={'analytic' if analytic else 'numeric'}")

    # ── Allocate output arrays ───────────────────────────────────────────────
    times = np.empty(n_saved)
    earth_pos_hist = np.empty((n_saved, 2))
    luna_pos_hist = np.empty((n_saved, 2))
    earth_vel_hist = np.empty((n_saved, 2))
    luna_vel_hist = np.empty((n_saved, 2))

    if config.save_trajectories:
        ast_pos_hist = np.full((n_ast, n_saved, 2), np.nan)
        ast_vel_hist = np.full((n_ast, n_saved, 2), np.nan)
    else:
        ast_pos_hist = None
        ast_vel_hist = None

    impacts: List[ImpactRecord] = []
    status = np.zeros(n_ast, dtype=np.int8)   # 0 = still active

    # ── Initial conditions ───────────────────────────────────────────────────
    el_pos, el_vel = _init_earth_luna()
    ast_init_pos, ast_init_vel = _init_asteroids(config)

    ast_pos = ast_init_pos.copy()
    ast_vel = ast_init_vel.copy()
    active = np.ones(n_ast, dtype=bool)

    el_acc = np.zeros_like(el_pos) if analytic else _el_acc(el_pos)
    ast_acc = _ast_acc(ast_pos, el_pos[0], el_pos[1])

    esc_dist2 = config.escape_distance ** 2

    # Save t=0
    save_idx = 0
    times[0] = 0.0
    earth_pos_hist[0] = el_pos[0]
    luna_pos_hist[0] = el_pos[1]
    earth_vel_hist[0] = el_vel[0]
    luna_vel_hist[0] = el_vel[1]
    if config.save_trajectories:
        ast_pos_hist[:, 0, :] = ast_pos
        ast_vel_hist[:, 0, :] = ast_vel

    # ── Time loop ────────────────────────────────────────────────────────────
    t0_wall = _time.time()
    for step in _tqdm(range(1, n_steps + 1), desc="Simulating", unit="step"):
        t = step * dt

        if analytic:
            # Exact circular solution — zero accumulated error for all t
            _c, _s = np.cos(_omega * t), np.sin(_omega * t)
            el_pos = np.array([[-_r_E * _c, -_r_E * _s],
                               [ _r_L * _c,  _r_L * _s]])
            el_vel = np.array([[ _r_E * _omega * _s, -_r_E * _omega * _c],
                               [-_r_L * _omega * _s,  _r_L * _omega * _c]])
            # Asteroid-only Velocity Verlet
            if active.any():
                ast_pos[active] += (ast_vel[active] * dt
                                    + 0.5 * ast_acc[active] * dt ** 2)
            ast_acc_new = np.zeros_like(ast_acc)
            if active.any():
                ast_acc_new[active] = _ast_acc(
                    ast_pos[active], el_pos[0], el_pos[1])
            if active.any():
                ast_vel[active] += 0.5 * (ast_acc[active]
                                           + ast_acc_new[active]) * dt
            ast_acc = ast_acc_new
        else:
            el_pos, el_vel, el_acc, ast_acc = step_fn(
                el_pos, el_vel, el_acc,
                ast_pos, ast_vel, ast_acc, active, dt,
            )

        # ── Impact & escape detection (vectorised over active set) ───────────
        if active.any():
            active_idx = np.where(active)[0]
            ap = ast_pos[active_idx]             # (k, 2)

            d2_e = np.einsum("ij,ij->i", ap - el_pos[0], ap - el_pos[0])
            d2_l = np.einsum("ij,ij->i", ap - el_pos[1], ap - el_pos[1])
            d2_c = np.einsum("ij,ij->i", ap, ap)

            hit_e = d2_e < EARTH_RADIUS ** 2
            hit_l = (~hit_e) & (d2_l < LUNA_RADIUS ** 2)
            esc = (~hit_e) & (~hit_l) & (d2_c > esc_dist2)

            for li in np.where(hit_e)[0]:
                gi = active_idx[li]
                angle = _compute_impact_angle(ast_pos[gi], el_pos[0], el_pos[1])
                impacts.append(ImpactRecord(int(gi), t, "earth", ast_pos[gi].copy(), angle))
                active[gi] = False
                status[gi] = 1

            for li in np.where(hit_l)[0]:
                gi = active_idx[li]
                angle = _compute_impact_angle(ast_pos[gi], el_pos[1], el_pos[0])
                impacts.append(ImpactRecord(int(gi), t, "luna", ast_pos[gi].copy(), angle))
                active[gi] = False
                status[gi] = 2

            active[active_idx[esc]] = False
            status[active_idx[esc]] = 3

        # ── Save state ───────────────────────────────────────────────────────
        if step % config.save_every == 0:
            save_idx = step // config.save_every
            if save_idx < n_saved:
                times[save_idx] = t
                earth_pos_hist[save_idx] = el_pos[0]
                luna_pos_hist[save_idx] = el_pos[1]
                earth_vel_hist[save_idx] = el_vel[0]
                luna_vel_hist[save_idx] = el_vel[1]
                if config.save_trajectories:
                    ast_pos_hist[active, save_idx, :] = ast_pos[active]
                    ast_vel_hist[active, save_idx, :] = ast_vel[active]

        if not active.any():
            print(f"\nAll asteroids resolved at step {step} / {n_steps}")
            # Trim unused save slots
            last = save_idx + 1
            times = times[:last]
            earth_pos_hist = earth_pos_hist[:last]
            luna_pos_hist = luna_pos_hist[:last]
            earth_vel_hist = earth_vel_hist[:last]
            luna_vel_hist = luna_vel_hist[:last]
            if config.save_trajectories:
                ast_pos_hist = ast_pos_hist[:, :last, :]
                ast_vel_hist = ast_vel_hist[:, :last, :]
            break

    wall = _time.time() - t0_wall
    n_earth = sum(1 for r in impacts if r.body == "earth")
    n_luna = sum(1 for r in impacts if r.body == "luna")
    n_esc = int((status == 3).sum())
    print(f"\nDone in {wall:.1f}s | "
          f"earth hits={n_earth} | luna hits={n_luna} | escaped={n_esc} | "
          f"active at end={(status == 0).sum()}")

    return SimResults(
        config=config,
        omega=EARTH_LUNA_OMEGA,
        times=times,
        earth_pos=earth_pos_hist,
        luna_pos=luna_pos_hist,
        earth_vel=earth_vel_hist,
        luna_vel=luna_vel_hist,
        ast_init_pos=ast_init_pos,
        ast_init_vel=ast_init_vel,
        ast_pos=ast_pos_hist,
        ast_vel=ast_vel_hist,
        impacts=impacts,
        status=status,
    )
