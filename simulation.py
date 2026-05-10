"""
Core simulation engine.

Earth and Luna interact bidirectionally (full two-body gravity).
Asteroids are pure test particles: they feel Earth + Luna gravity but exert
none.  All integration is in SI units.  Velocity Verlet is the default
(symplectic, energy-conserving).

When analytic_primaries=True (default), Earth/Luna positions are computed from
the exact trig solution for a circular orbit — zero accumulated error.  This
also makes each asteroid's trajectory independent, enabling a Numba JIT kernel
that processes all asteroids in parallel with no Python overhead.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import time as _time

from constants import (G, EARTH_MASS, EARTH_RADIUS, LUNA_MASS, LUNA_RADIUS,
                       EARTH_LUNA_DIST, EARTH_LUNA_OMEGA)
from config import SimConfig


# ── Numba optional import ─────────────────────────────────────────────────────

try:
    from numba import njit, prange as _nb_prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False


# ── Result structures ────────────────────────────────────────────────────────

@dataclass
class ImpactRecord:
    asteroid_idx: int
    time: float           # seconds since t=0
    body: str             # "earth" or "luna"
    pos: np.ndarray       # (2,) inertial-frame position at impact
    angle: float          # radians from sub-body reference direction


@dataclass
class SimResults:
    config: SimConfig
    omega: float

    times: np.ndarray
    earth_pos: np.ndarray
    luna_pos: np.ndarray
    earth_vel: np.ndarray
    luna_vel: np.ndarray

    ast_init_pos: np.ndarray
    ast_init_vel: np.ndarray

    # None when save_trajectories is False (large-N / JIT runs)
    ast_pos: Optional[np.ndarray]
    ast_vel: Optional[np.ndarray]

    impacts: List[ImpactRecord]
    status: np.ndarray   # (n_ast,) int8: 0=active, 1=earth, 2=luna, 3=escaped


# ── Force computation (Python/NumPy path) ────────────────────────────────────

def _el_acc(el_pos: np.ndarray) -> np.ndarray:
    r = el_pos[1] - el_pos[0]
    d3 = np.linalg.norm(r) ** 3
    acc = np.empty_like(el_pos)
    acc[0] = G * LUNA_MASS * r / d3
    acc[1] = -G * EARTH_MASS * r / d3
    return acc


def _ast_acc(ast_pos: np.ndarray,
             earth_pos: np.ndarray,
             luna_pos: np.ndarray) -> np.ndarray:
    eps2 = 1e6 ** 2
    dr_e = earth_pos - ast_pos
    dr_l = luna_pos  - ast_pos
    d2_e = np.einsum("ij,ij->i", dr_e, dr_e)[:, None] + eps2
    d2_l = np.einsum("ij,ij->i", dr_l, dr_l)[:, None] + eps2
    a_e = G * EARTH_MASS * dr_e / d2_e ** 1.5
    a_l = G * LUNA_MASS  * dr_l / d2_l ** 1.5
    return a_e + a_l


# ── Integrators (Python/NumPy path) ──────────────────────────────────────────

def _step_verlet(el_pos, el_vel, el_acc_cur,
                 ast_pos, ast_vel, ast_acc_cur, active, dt):
    el_pos_new = el_pos + el_vel * dt + 0.5 * el_acc_cur * dt ** 2
    if active.any():
        ast_pos[active] += ast_vel[active] * dt + 0.5 * ast_acc_cur[active] * dt ** 2
    el_acc_new = _el_acc(el_pos_new)
    ast_acc_new = np.zeros_like(ast_acc_cur)
    if active.any():
        ast_acc_new[active] = _ast_acc(ast_pos[active], el_pos_new[0], el_pos_new[1])
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
    def derivs_el(p, v):
        return v, _el_acc(p)

    k1_r, k1_v = derivs_el(el_pos, el_vel)
    k2_r, k2_v = derivs_el(el_pos + 0.5*dt*k1_r, el_vel + 0.5*dt*k1_v)
    k3_r, k3_v = derivs_el(el_pos + 0.5*dt*k2_r, el_vel + 0.5*dt*k2_v)
    k4_r, k4_v = derivs_el(el_pos + dt*k3_r,     el_vel + dt*k3_v)
    el_pos_new = el_pos + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    el_vel_new = el_vel + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    el_acc_new = _el_acc(el_pos_new)

    ast_acc_new = np.zeros_like(ast_acc_cur)
    if active.any():
        pa = ast_pos[active].copy()
        va = ast_vel[active].copy()
        ep_mid = (el_pos[0] + el_pos_new[0]) / 2
        lp_mid = (el_pos[1] + el_pos_new[1]) / 2
        ak1_r, ak1_v = va, _ast_acc(pa, el_pos[0], el_pos[1])
        ak2_r, ak2_v = va + 0.5*dt*ak1_v, _ast_acc(pa + 0.5*dt*ak1_r, ep_mid, lp_mid)
        ak3_r, ak3_v = va + 0.5*dt*ak2_v, _ast_acc(pa + 0.5*dt*ak2_r, ep_mid, lp_mid)
        ak4_r, ak4_v = va + dt*ak3_v,     _ast_acc(pa + dt*ak3_r, el_pos_new[0], el_pos_new[1])
        ast_pos[active] += (dt/6) * (ak1_r + 2*ak2_r + 2*ak3_r + ak4_r)
        ast_vel[active] += (dt/6) * (ak1_v + 2*ak2_v + 2*ak3_v + ak4_v)
        ast_acc_new[active] = _ast_acc(ast_pos[active], el_pos_new[0], el_pos_new[1])

    return el_pos_new, el_vel_new, el_acc_new, ast_acc_new


_INTEGRATORS = {"verlet": _step_verlet, "euler": _step_euler, "rk4": _step_rk4}


# ── Initial conditions ────────────────────────────────────────────────────────

def _init_earth_luna() -> tuple[np.ndarray, np.ndarray]:
    M   = EARTH_MASS + LUNA_MASS
    r_e = LUNA_MASS  * EARTH_LUNA_DIST / M
    r_l = EARTH_MASS * EARTH_LUNA_DIST / M
    omega = EARTH_LUNA_OMEGA
    el_pos = np.array([[-r_e, 0.0], [+r_l, 0.0]])
    el_vel = np.array([[0.0, -omega * r_e], [0.0, +omega * r_l]])
    return el_pos, el_vel


def _init_asteroids(config: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_seed)
    R   = config.source_distance

    if config.init_mode == "montecarlo":
        n = config.n_asteroids
        lo, hi = config.angle_range
        theta = np.deg2rad(rng.uniform(lo, hi, n))
        b = rng.uniform(-config.b_max, config.b_max, n)
        v = rng.uniform(*config.speed_range, n)

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
        b     = b_g.ravel()
        v     = np.full(len(theta), config.grid_speed)
    else:
        raise ValueError(f"Unknown init_mode: {config.init_mode!r}")

    e_app  = np.column_stack([ np.cos(theta),  np.sin(theta)])
    e_perp = np.column_stack([-np.sin(theta),  np.cos(theta)])
    ast_pos = -R * e_app + b[:, None] * e_perp
    ast_vel =  v[:, None] * e_app
    return ast_pos.astype(np.float64), ast_vel.astype(np.float64)


# ── Impact angle helper ───────────────────────────────────────────────────────

def _compute_impact_angle(hit_pos, body_pos, ref_pos) -> float:
    hit_vec = hit_pos - body_pos
    ref_vec = ref_pos - body_pos
    cross = ref_vec[0] * hit_vec[1] - ref_vec[1] * hit_vec[0]
    dot   = np.dot(ref_vec, hit_vec)
    return float(np.arctan2(cross, dot))


# ── Numba JIT kernel ──────────────────────────────────────────────────────────
# Each asteroid is a test particle with an independent trajectory (analytic
# primaries make Earth/Luna positions a closed-form function of t).  We
# process all N asteroids in parallel — no shared mutable state.

if _NUMBA_OK:
    @njit(parallel=True, fastmath=True, cache=True)
    def _jit_simulate_all(
        init_pos,      # (N, 2) float64 — initial positions
        init_vel,      # (N, 2) float64 — initial velocities
        release_time,  # (N,)   float64 — start time for each asteroid
        duration,      # float64 — integration time per asteroid (s)
        dt,            # float64 — timestep (s)
        omega,         # float64 — Earth-Luna angular velocity (rad/s)
        r_E, r_L,      # float64 — primary orbital radii from CoM
        G_M_E, G_M_L,  # float64 — G * mass for each primary
        eps2,          # float64 — softening length squared (m²)
        R_E2, R_L2,    # float64 — impact radius squared for each body
        esc2,          # float64 — escape distance squared
        out_status,    # (N,) int8   — output: 0=active, 1=earth, 2=luna, 3=escaped
        out_ipos,      # (N, 2) float64 — impact position (inertial frame)
        out_itime,     # (N,)   float64 — impact time
        out_iangle,    # (N,)   float64 — impact angle (rad)
        out_ibody,     # (N,)   int8    — 0=no impact, 1=earth, 2=luna
    ):
        N = init_pos.shape[0]
        for i in _nb_prange(N):
            x  = init_pos[i, 0];  y  = init_pos[i, 1]
            vx = init_vel[i, 0];  vy = init_vel[i, 1]
            t     = release_time[i]
            t_end = t + duration

            # Primary positions at release time
            c = math.cos(omega * t);  s = math.sin(omega * t)
            ex = -r_E * c;  ey = -r_E * s
            lx =  r_L * c;  ly =  r_L * s

            # Initial asteroid acceleration (softened)
            drex = ex - x;  drey = ey - y
            drlx = lx - x;  drly = ly - y
            d2e = drex*drex + drey*drey + eps2
            d2l = drlx*drlx + drly*drly + eps2
            inv3e = 1.0 / (d2e * math.sqrt(d2e))
            inv3l = 1.0 / (d2l * math.sqrt(d2l))
            ax = G_M_E * drex * inv3e + G_M_L * drlx * inv3l
            ay = G_M_E * drey * inv3e + G_M_L * drly * inv3l

            hit_status = 0
            while t < t_end:
                # Velocity Verlet position update
                x_new = x + vx * dt + 0.5 * ax * dt * dt
                y_new = y + vy * dt + 0.5 * ay * dt * dt
                t_new = t + dt

                # Analytic primary positions at t_new
                c2 = math.cos(omega * t_new);  s2 = math.sin(omega * t_new)
                ex2 = -r_E * c2;  ey2 = -r_E * s2
                lx2 =  r_L * c2;  ly2 =  r_L * s2

                # New acceleration (softened, for Verlet velocity step)
                drex2 = ex2 - x_new;  drey2 = ey2 - y_new
                drlx2 = lx2 - x_new;  drly2 = ly2 - y_new
                d2e2 = drex2*drex2 + drey2*drey2 + eps2
                d2l2 = drlx2*drlx2 + drly2*drly2 + eps2
                inv3e2 = 1.0 / (d2e2 * math.sqrt(d2e2))
                inv3l2 = 1.0 / (d2l2 * math.sqrt(d2l2))
                ax2 = G_M_E * drex2 * inv3e2 + G_M_L * drlx2 * inv3l2
                ay2 = G_M_E * drey2 * inv3e2 + G_M_L * drly2 * inv3l2

                # Velocity full-step
                vx_new = vx + 0.5 * (ax + ax2) * dt
                vy_new = vy + 0.5 * (ay + ay2) * dt

                # Impact / escape detection (unsoftened distances)
                d2e_hit = drex2*drex2 + drey2*drey2
                d2l_hit = drlx2*drlx2 + drly2*drly2

                if d2e_hit < R_E2:
                    hit_status = 1
                    out_ipos[i, 0] = x_new;  out_ipos[i, 1] = y_new
                    out_itime[i]   = t_new
                    hx = x_new - ex2;  hy = y_new - ey2   # vec: Earth center → impact
                    rx = lx2   - ex2;  ry = ly2   - ey2   # ref: Earth → Luna
                    out_iangle[i]  = math.atan2(rx*hy - ry*hx, rx*hx + ry*hy)
                    out_ibody[i]   = 1
                    break
                elif d2l_hit < R_L2:
                    hit_status = 2
                    out_ipos[i, 0] = x_new;  out_ipos[i, 1] = y_new
                    out_itime[i]   = t_new
                    hx = x_new - lx2;  hy = y_new - ly2   # vec: Luna center → impact
                    rx = ex2   - lx2;  ry = ey2   - ly2   # ref: Luna → Earth
                    out_iangle[i]  = math.atan2(rx*hy - ry*hx, rx*hx + ry*hy)
                    out_ibody[i]   = 2
                    break
                elif x_new*x_new + y_new*y_new > esc2:
                    hit_status = 3
                    break

                x = x_new;  y = y_new
                vx = vx_new;  vy = vy_new
                ax = ax2;  ay = ay2
                t = t_new

            out_status[i] = hit_status

    @njit(parallel=True, fastmath=True, cache=True)
    def _jit_replay_trajectories(
        init_pos, init_vel, release_time, impact_time,
        dt, save_stride,
        omega, r_E, r_L, G_M_E, G_M_L, eps2,
        out_traj,    # (N, max_pts, 2) float64  — NaN-initialised by caller
        out_t_pts,   # (N, max_pts)    float64  — sim-time at each saved point
        out_n_pts,   # (N,)            int32    — actual saved-point count
    ):
        """Re-integrate impacting asteroid trajectories, saving at intervals."""
        N       = init_pos.shape[0]
        max_pts = out_traj.shape[1]
        for i in _nb_prange(N):
            x  = init_pos[i, 0];  y  = init_pos[i, 1]
            vx = init_vel[i, 0];  vy = init_vel[i, 1]
            t     = release_time[i]
            t_end = impact_time[i]

            c = math.cos(omega * t);  s = math.sin(omega * t)
            ex = -r_E * c;  ey = -r_E * s
            lx =  r_L * c;  ly =  r_L * s
            drex = ex - x;  drey = ey - y
            drlx = lx - x;  drly = ly - y
            d2e = drex*drex + drey*drey + eps2
            d2l = drlx*drlx + drly*drly + eps2
            ax = G_M_E * drex / (d2e * math.sqrt(d2e)) + G_M_L * drlx / (d2l * math.sqrt(d2l))
            ay = G_M_E * drey / (d2e * math.sqrt(d2e)) + G_M_L * drly / (d2l * math.sqrt(d2l))

            pt = 0
            if pt < max_pts:
                out_traj[i, pt, 0] = x;  out_traj[i, pt, 1] = y
                out_t_pts[i, pt]   = t;  pt += 1

            step_ct = 0
            while t + dt <= t_end:
                x_new = x + vx * dt + 0.5 * ax * dt * dt
                y_new = y + vy * dt + 0.5 * ay * dt * dt
                t_new = t + dt

                c2 = math.cos(omega * t_new);  s2 = math.sin(omega * t_new)
                ex2 = -r_E * c2;  ey2 = -r_E * s2
                lx2 =  r_L * c2;  ly2 =  r_L * s2
                drex2 = ex2 - x_new;  drey2 = ey2 - y_new
                drlx2 = lx2 - x_new;  drly2 = ly2 - y_new
                d2e2 = drex2*drex2 + drey2*drey2 + eps2
                d2l2 = drlx2*drlx2 + drly2*drly2 + eps2
                ax2 = G_M_E * drex2 / (d2e2 * math.sqrt(d2e2)) + G_M_L * drlx2 / (d2l2 * math.sqrt(d2l2))
                ay2 = G_M_E * drey2 / (d2e2 * math.sqrt(d2e2)) + G_M_L * drly2 / (d2l2 * math.sqrt(d2l2))

                vx_new = vx + 0.5 * (ax + ax2) * dt
                vy_new = vy + 0.5 * (ay + ay2) * dt

                step_ct += 1
                if step_ct % save_stride == 0 and pt < max_pts:
                    out_traj[i, pt, 0] = x_new;  out_traj[i, pt, 1] = y_new
                    out_t_pts[i, pt]   = t_new;  pt += 1

                x = x_new;  y = y_new
                vx = vx_new;  vy = vy_new
                ax = ax2;  ay = ay2
                t = t_new

            # Always capture the final state so the curve ends at impact
            if pt < max_pts:
                out_traj[i, pt, 0] = x;  out_traj[i, pt, 1] = y
                out_t_pts[i, pt]   = t_end;  pt += 1
            out_n_pts[i] = pt


# ── Trajectory replay (post-hoc, for analysis viewer) ────────────────────────

def reintegrate_impactors(results: SimResults,
                           stride: int = 200,
                           max_luna: int = 500,
                           max_earth: int = 300) -> dict:
    """Re-integrate impacting asteroid trajectories after the main sim.

    Only impacting asteroids are processed — memory is proportional to the
    number of hits, not to N.  Returns trajectories in the corotating frame.

    Returns dict with keys:
        'luna'  : list of (n_pts, 2) float64 corotating-frame arrays
        'earth' : list of (n_pts, 2) float64 corotating-frame arrays
        'luna_recs'  : corresponding ImpactRecord list
        'earth_recs' : corresponding ImpactRecord list
    """
    cfg  = results.config
    dt   = cfg.dt
    n_ast = results.ast_init_pos.shape[0]

    # Reconstruct release times with the same RNG used in run_simulation
    if cfg.release_window > 0:
        rng_rel  = np.random.default_rng(cfg.random_seed ^ 0xABCD1234)
        rel_all  = rng_rel.uniform(0.0, cfg.release_window, n_ast)
    else:
        rel_all = np.zeros(n_ast)

    luna_recs  = [r for r in results.impacts if r.body == "luna"][:max_luna]
    earth_recs = [r for r in results.impacts if r.body == "earth"][:max_earth]
    all_recs   = luna_recs + earth_recs

    if not all_recs:
        return {"luna": [], "earth": [], "luna_recs": [], "earth_recs": []}

    idx      = np.array([r.asteroid_idx for r in all_recs])
    init_pos = results.ast_init_pos[idx].astype(np.float64)
    init_vel = results.ast_init_vel[idx].astype(np.float64)
    rel_t    = rel_all[idx].astype(np.float64)
    imp_t    = np.array([r.time for r in all_recs], dtype=np.float64)

    max_dur = float((imp_t - rel_t).max()) if len(all_recs) > 0 else dt
    max_pts = int(max_dur / (dt * stride)) + 10

    M    = EARTH_MASS + LUNA_MASS
    r_E  = LUNA_MASS  * EARTH_LUNA_DIST / M
    r_L  = EARTH_MASS * EARTH_LUNA_DIST / M
    eps2 = float(1e6 ** 2)

    out_traj  = np.full((len(all_recs), max_pts, 2), np.nan, dtype=np.float64)
    out_t_pts = np.zeros((len(all_recs), max_pts),    dtype=np.float64)
    out_n_pts = np.zeros(len(all_recs),               dtype=np.int32)

    if _NUMBA_OK:
        print(f"Re-integrating {len(all_recs)} impactor trajectories "
              f"(stride={stride} steps, ≤{max_pts} pts each)…")
        _jit_replay_trajectories(
            init_pos, init_vel, rel_t, imp_t,
            float(dt), int(stride),
            float(EARTH_LUNA_OMEGA), float(r_E), float(r_L),
            float(G * EARTH_MASS), float(G * LUNA_MASS), eps2,
            out_traj, out_t_pts, out_n_pts,
        )
    else:
        # Pure-Python fallback: just store start + end so lines still appear
        print("Numba unavailable — trajectory replay using straight-line approx.")
        for i, rec in enumerate(all_recs):
            out_traj[i, 0, :] = init_pos[i];  out_t_pts[i, 0] = rel_t[i]
            out_traj[i, 1, :] = rec.pos;       out_t_pts[i, 1] = imp_t[i]
            out_n_pts[i] = 2

    # Transform each saved trajectory to the corotating frame
    omega = float(EARTH_LUNA_OMEGA)
    traj_corot: List[Optional[np.ndarray]] = []
    for i in range(len(all_recs)):
        n = int(out_n_pts[i])
        if n < 2:
            traj_corot.append(None)
            continue
        pts   = out_traj[i, :n, :]       # (n, 2) inertial metres
        t_pts = out_t_pts[i, :n]          # (n,)   sim-time seconds
        ang   = omega * t_pts
        c_a, s_a = np.cos(ang), np.sin(ang)
        xc = c_a * pts[:, 0] + s_a * pts[:, 1]
        yc = -s_a * pts[:, 0] + c_a * pts[:, 1]
        traj_corot.append(np.column_stack([xc, yc]))

    n_l = len(luna_recs)
    return {
        "luna":       traj_corot[:n_l],
        "earth":      traj_corot[n_l:],
        "luna_recs":  luna_recs,
        "earth_recs": earth_recs,
    }


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(config: SimConfig) -> SimResults:
    """Run the full Earth-Luna-asteroid simulation and return results."""
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(it, **kw): return it

    n_ast    = config.effective_n_asteroids()
    dt       = config.dt
    analytic = config.analytic_primaries

    _M    = EARTH_MASS + LUNA_MASS
    _r_E  = LUNA_MASS  * EARTH_LUNA_DIST / _M
    _r_L  = EARTH_MASS * EARTH_LUNA_DIST / _M
    _omega = EARTH_LUNA_OMEGA

    # JIT path: analytic primaries + Numba available + no trajectory saving needed
    use_jit = _NUMBA_OK and analytic

    # Auto-disable trajectory saving for large N regardless of path
    save_traj = config.save_trajectories
    if save_traj and n_ast > 10_000:
        print(f"Note: disabling trajectory saving for {n_ast:,} asteroids (memory).")
        save_traj = False
    if use_jit and save_traj:
        # JIT path doesn't record per-step positions
        save_traj = False

    ast_init_pos, ast_init_vel = _init_asteroids(config)

    # Release times — spread asteroids over the requested window
    if config.release_window > 0:
        rng_rel = np.random.default_rng(config.random_seed ^ 0xABCD1234)
        release_time = rng_rel.uniform(0.0, config.release_window, n_ast)
    else:
        release_time = np.zeros(n_ast)

    t_total = float(release_time.max()) + config.duration if n_ast > 0 else config.duration

    print(f"Sim: {n_ast:,} asteroids | dt={dt:.0f}s | "
          f"duration={config.duration/86400:.1f}d | "
          f"release_window={config.release_window/86400:.2f}d | "
          f"primaries={'analytic' if analytic else 'numeric'} | "
          f"jit={use_jit}")

    # ── JIT path ─────────────────────────────────────────────────────────────
    if use_jit:
        out_status = np.zeros(n_ast, dtype=np.int8)
        out_ipos   = np.zeros((n_ast, 2), dtype=np.float64)
        out_itime  = np.zeros(n_ast,      dtype=np.float64)
        out_iangle = np.zeros(n_ast,      dtype=np.float64)
        out_ibody  = np.zeros(n_ast,      dtype=np.int8)

        print("Running Numba JIT kernel (parallel, all cores)…"
              " (first run compiles — ~10 s, cached thereafter)")
        t0_wall = _time.time()
        _jit_simulate_all(
            ast_init_pos, ast_init_vel,
            release_time.astype(np.float64),
            float(config.duration), float(dt),
            float(_omega), float(_r_E), float(_r_L),
            float(G * EARTH_MASS), float(G * LUNA_MASS),
            float(1e6 ** 2),
            float(EARTH_RADIUS ** 2), float(LUNA_RADIUS ** 2),
            float(config.escape_distance ** 2),
            out_status, out_ipos, out_itime, out_iangle, out_ibody,
        )
        wall = _time.time() - t0_wall

        # Assemble ImpactRecord list from output arrays
        hit_idx = np.where(out_ibody > 0)[0]
        impacts: List[ImpactRecord] = [
            ImpactRecord(
                int(i),
                float(out_itime[i]),
                "earth" if out_ibody[i] == 1 else "luna",
                out_ipos[i].copy(),
                float(out_iangle[i]),
            )
            for i in hit_idx
        ]
        status = out_status

        # Earth/Luna trajectory history via analytic solution (for viewer)
        n_saved = max(2, int(t_total / (dt * config.save_every)) + 2)
        times = np.linspace(0.0, t_total, n_saved)
        _ca = np.cos(_omega * times);  _sa = np.sin(_omega * times)
        earth_pos_hist = np.column_stack([-_r_E * _ca, -_r_E * _sa])
        luna_pos_hist  = np.column_stack([ _r_L * _ca,  _r_L * _sa])
        earth_vel_hist = np.column_stack([ _r_E * _omega * _sa, -_r_E * _omega * _ca])
        luna_vel_hist  = np.column_stack([-_r_L * _omega * _sa,  _r_L * _omega * _ca])

        n_earth = sum(1 for r in impacts if r.body == "earth")
        n_luna  = sum(1 for r in impacts if r.body == "luna")
        n_esc   = int((status == 3).sum())
        rate    = n_ast / wall if wall > 0 else float("inf")
        print(f"Done in {wall:.1f}s ({rate:,.0f} ast/s) | "
              f"earth={n_earth} | luna={n_luna} | escaped={n_esc} | "
              f"active={(status==0).sum()}")

        return SimResults(
            config=config, omega=EARTH_LUNA_OMEGA,
            times=times,
            earth_pos=earth_pos_hist, luna_pos=luna_pos_hist,
            earth_vel=earth_vel_hist, luna_vel=luna_vel_hist,
            ast_init_pos=ast_init_pos, ast_init_vel=ast_init_vel,
            ast_pos=None, ast_vel=None,
            impacts=impacts, status=status,
        )

    # ── Python/NumPy path (fallback or numeric primaries) ────────────────────
    n_steps = int(t_total / dt)
    n_saved = n_steps // config.save_every + 1

    times          = np.empty(n_saved)
    earth_pos_hist = np.empty((n_saved, 2))
    luna_pos_hist  = np.empty((n_saved, 2))
    earth_vel_hist = np.empty((n_saved, 2))
    luna_vel_hist  = np.empty((n_saved, 2))

    if save_traj:
        ast_pos_hist = np.full((n_ast, n_saved, 2), np.nan)
        ast_vel_hist = np.full((n_ast, n_saved, 2), np.nan)
    else:
        ast_pos_hist = None
        ast_vel_hist = None

    impacts: List[ImpactRecord] = []
    status = np.zeros(n_ast, dtype=np.int8)

    el_pos, el_vel = _init_earth_luna()
    ast_pos = ast_init_pos.copy()
    ast_vel = ast_init_vel.copy()

    # In Python path, release_window is honoured by marking asteroids inactive
    # until their release_time and placing them at their initial position.
    # active[i] becomes True once t >= release_time[i].
    released = release_time <= 0.0
    active   = released.copy()

    step_fn  = _INTEGRATORS[config.integrator]
    el_acc   = np.zeros_like(el_pos) if analytic else _el_acc(el_pos)
    ast_acc  = np.zeros((n_ast, 2))
    if active.any():
        ast_acc[active] = _ast_acc(ast_pos[active], el_pos[0], el_pos[1])

    esc_dist2 = config.escape_distance ** 2

    save_idx = 0
    times[0] = 0.0
    earth_pos_hist[0] = el_pos[0]
    luna_pos_hist[0]  = el_pos[1]
    earth_vel_hist[0] = el_vel[0]
    luna_vel_hist[0]  = el_vel[1]
    if save_traj:
        ast_pos_hist[active, 0, :] = ast_pos[active]
        ast_vel_hist[active, 0, :] = ast_vel[active]

    t0_wall = _time.time()
    for step in _tqdm(range(1, n_steps + 1), desc="Simulating", unit="step"):
        t = step * dt

        # Activate newly-released asteroids
        newly = (~released) & (release_time <= t)
        if newly.any():
            released[newly] = True
            active[newly]   = True
            ast_acc[newly]  = _ast_acc(ast_pos[newly], el_pos[0], el_pos[1])

        if analytic:
            _c, _s = math.cos(_omega * t), math.sin(_omega * t)
            el_pos = np.array([[-_r_E * _c, -_r_E * _s],
                               [ _r_L * _c,  _r_L * _s]])
            el_vel = np.array([[ _r_E * _omega * _s, -_r_E * _omega * _c],
                               [-_r_L * _omega * _s,  _r_L * _omega * _c]])
            if active.any():
                ast_pos[active] += (ast_vel[active] * dt
                                    + 0.5 * ast_acc[active] * dt ** 2)
            ast_acc_new = np.zeros_like(ast_acc)
            if active.any():
                ast_acc_new[active] = _ast_acc(ast_pos[active], el_pos[0], el_pos[1])
            if active.any():
                ast_vel[active] += 0.5 * (ast_acc[active] + ast_acc_new[active]) * dt
            ast_acc = ast_acc_new
        else:
            el_pos, el_vel, el_acc, ast_acc = step_fn(
                el_pos, el_vel, el_acc, ast_pos, ast_vel, ast_acc, active, dt)

        # Impact & escape detection
        if active.any():
            active_idx = np.where(active)[0]
            ap = ast_pos[active_idx]
            d2_e = np.einsum("ij,ij->i", ap - el_pos[0], ap - el_pos[0])
            d2_l = np.einsum("ij,ij->i", ap - el_pos[1], ap - el_pos[1])
            d2_c = np.einsum("ij,ij->i", ap, ap)
            hit_e = d2_e < EARTH_RADIUS ** 2
            hit_l = (~hit_e) & (d2_l < LUNA_RADIUS ** 2)
            esc   = (~hit_e) & (~hit_l) & (d2_c > esc_dist2)

            for li in np.where(hit_e)[0]:
                gi = active_idx[li]
                angle = _compute_impact_angle(ast_pos[gi], el_pos[0], el_pos[1])
                impacts.append(ImpactRecord(int(gi), t, "earth", ast_pos[gi].copy(), angle))
                active[gi] = False; status[gi] = 1

            for li in np.where(hit_l)[0]:
                gi = active_idx[li]
                angle = _compute_impact_angle(ast_pos[gi], el_pos[1], el_pos[0])
                impacts.append(ImpactRecord(int(gi), t, "luna", ast_pos[gi].copy(), angle))
                active[gi] = False; status[gi] = 2

            active[active_idx[esc]] = False
            status[active_idx[esc]] = 3

        if step % config.save_every == 0:
            save_idx = step // config.save_every
            if save_idx < n_saved:
                times[save_idx]          = t
                earth_pos_hist[save_idx] = el_pos[0]
                luna_pos_hist[save_idx]  = el_pos[1]
                earth_vel_hist[save_idx] = el_vel[0]
                luna_vel_hist[save_idx]  = el_vel[1]
                if save_traj:
                    ast_pos_hist[active, save_idx, :] = ast_pos[active]
                    ast_vel_hist[active, save_idx, :] = ast_vel[active]

        # Stop early only when all asteroids have been released AND all resolved
        if released.all() and not active.any():
            print(f"\nAll asteroids resolved at step {step} / {n_steps}")
            last = save_idx + 1
            times          = times[:last]
            earth_pos_hist = earth_pos_hist[:last]
            luna_pos_hist  = luna_pos_hist[:last]
            earth_vel_hist = earth_vel_hist[:last]
            luna_vel_hist  = luna_vel_hist[:last]
            if save_traj:
                ast_pos_hist = ast_pos_hist[:, :last, :]
                ast_vel_hist = ast_vel_hist[:, :last, :]
            break

    wall    = _time.time() - t0_wall
    n_earth = sum(1 for r in impacts if r.body == "earth")
    n_luna  = sum(1 for r in impacts if r.body == "luna")
    n_esc   = int((status == 3).sum())
    print(f"\nDone in {wall:.1f}s | "
          f"earth={n_earth} | luna={n_luna} | escaped={n_esc} | "
          f"active={(status==0).sum()}")

    return SimResults(
        config=config, omega=EARTH_LUNA_OMEGA,
        times=times,
        earth_pos=earth_pos_hist, luna_pos=luna_pos_hist,
        earth_vel=earth_vel_hist, luna_vel=luna_vel_hist,
        ast_init_pos=ast_init_pos, ast_init_vel=ast_init_vel,
        ast_pos=ast_pos_hist, ast_vel=ast_vel_hist,
        impacts=impacts, status=status,
    )
