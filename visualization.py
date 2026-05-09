"""
Interactive matplotlib viewer + standalone impact analysis figures.

Public API:
    SimViewer(results).show()          # interactive 2D viewer
    plot_impact_analysis(results)      # static analysis figure
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Patch
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from matplotlib.gridspec import GridSpec

from constants import (G, EARTH_MASS, EARTH_RADIUS, LUNA_MASS, LUNA_RADIUS,
                       EARTH_LUNA_DIST)
from simulation import SimResults

D = EARTH_LUNA_DIST   # one display unit = mean Earth-Luna distance

# Impact-angle orientation notes (for reference):
#   angle=0       → Near Side  (Earth-facing hemisphere, tidal-lock face)
#   angle=±π      → Far Side   (anti-Earth hemisphere)
#   angle=−π/2    → Leading    (hemisphere facing the direction of Luna's orbital motion)
#   angle=+π/2    → Trailing   (hemisphere facing opposite to orbital motion)
# The angle is measured from the INSTANTANEOUS Earth–Luna direction, so tidal
# locking is implicit — "near-side" always means the same physical face.


def _label_polar_directions(ax, rmax: float) -> None:
    """Annotate a Luna impact polar-histogram with orbital reference labels."""
    r = max(rmax, 1) * 1.50
    ax.set_rmax(max(rmax, 1) * 1.65)
    kw = dict(ha="center", va="center", fontsize=6.5, color="#AAC",
               fontweight="bold")
    ax.text(0,          r, "Near\nSide",  **kw)
    ax.text(np.pi,      r, "Far\nSide",   **kw)
    ax.text(-np.pi / 2, r, "Leading",     **kw)   # direction of orbital motion
    ax.text( np.pi / 2, r, "Trailing",    **kw)


# ── Reference-frame transforms ──────────────────────────────────────────────

def _theta(earth_pos: np.ndarray, luna_pos: np.ndarray) -> float:
    """Angle of the Earth → Luna line in the inertial frame (radians)."""
    return float(np.arctan2(luna_pos[1] - earth_pos[1],
                            luna_pos[0] - earth_pos[0]))


def to_display(pos_m: np.ndarray, frame: str,
               earth_pos: np.ndarray, luna_pos: np.ndarray) -> np.ndarray:
    """Inertial-frame metres → display-frame coordinates (units of D)."""
    squeeze = pos_m.ndim == 1
    if squeeze:
        pos_m = pos_m[None, :]

    if frame in ("inertial", "com"):
        out = pos_m / D
    elif frame == "earth":
        out = (pos_m - earth_pos) / D
    elif frame == "luna":
        out = (pos_m - luna_pos) / D
    elif frame == "corotating":
        th = _theta(earth_pos, luna_pos)
        c, s = np.cos(th), np.sin(th)
        # Rotate by -theta:  R(-θ) @ [x,y]^T = [c·x + s·y, -s·x + c·y]
        out = np.column_stack([
            c * pos_m[:, 0] + s * pos_m[:, 1],
            -s * pos_m[:, 0] + c * pos_m[:, 1],
        ]) / D
    else:
        raise ValueError(f"Unknown frame: {frame!r}")

    return out[0] if squeeze else out


def to_inertial(pos_d: np.ndarray, frame: str,
                earth_pos: np.ndarray, luna_pos: np.ndarray) -> np.ndarray:
    """Inverse of to_display — display coordinates → inertial metres."""
    squeeze = pos_d.ndim == 1
    if squeeze:
        pos_d = pos_d[None, :]
    pos_m = pos_d * D

    if frame in ("inertial", "com"):
        out = pos_m
    elif frame == "earth":
        out = pos_m + earth_pos
    elif frame == "luna":
        out = pos_m + luna_pos
    elif frame == "corotating":
        th = _theta(earth_pos, luna_pos)
        c, s = np.cos(th), np.sin(th)
        # Rotate by +theta: R(+θ) @ [x,y]^T = [c·x - s·y, s·x + c·y]
        out = np.column_stack([
            c * pos_m[:, 0] - s * pos_m[:, 1],
            s * pos_m[:, 0] + c * pos_m[:, 1],
        ])
    else:
        raise ValueError(f"Unknown frame: {frame!r}")

    return out[0] if squeeze else out


def _potential_grid(earth_pos, luna_pos, frame, omega, n, xlim, ylim):
    """Compute log gravitational (+ centrifugal in corotating) potential.

    Grid spans the supplied axis limits so the field always covers the view.
    Returns (X_d, Y_d, log10(-phi)) suitable for contour().
    """
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X_d, Y_d = np.meshgrid(x, y)

    grid_d = np.column_stack([X_d.ravel(), Y_d.ravel()])
    grid_m = to_inertial(grid_d, frame, earth_pos, luna_pos)

    eps = 1.0e7   # 10,000 km softening so contours stay finite near bodies
    r_e = np.sqrt(np.sum((grid_m - earth_pos) ** 2, axis=1) + eps ** 2)
    r_l = np.sqrt(np.sum((grid_m - luna_pos)  ** 2, axis=1) + eps ** 2)
    phi = -G * EARTH_MASS / r_e - G * LUNA_MASS / r_l

    if frame == "corotating":
        # Centrifugal potential about CoM (origin in inertial frame)
        r_com = np.linalg.norm(grid_m, axis=1)
        phi -= 0.5 * omega ** 2 * r_com ** 2

    phi = phi.reshape(n, n)
    z = np.log10(np.maximum(-phi, 1e-12))
    finite = np.isfinite(z)
    if finite.any():
        lo, hi = np.percentile(z[finite], [3, 97])
        z = np.clip(z, lo, hi)
    return X_d, Y_d, z


# ── Interactive viewer ──────────────────────────────────────────────────────

class SimViewer:
    """Time-stepped 2D viewer with frame switching and field overlays."""

    FRAMES = ["inertial", "com", "earth", "luna", "corotating"]
    TOGGLES = ["Trails", "Field", "Lagrange", "Impacts"]

    def __init__(self, results: SimResults):
        self.r = results
        self.cfg = results.config
        self.n_saved = len(results.times)
        self.n_ast = results.ast_init_pos.shape[0]

        self.t_idx = 0
        self.frame = self.cfg.reference_frame
        self._show_trails = bool(self.cfg.show_trails)
        self._show_field = bool(self.cfg.show_field_lines)
        self._show_lagrange = True
        self._show_impacts = True
        self._playing = False
        self._play_step = 10   # saved frames advanced per timer tick

        # Pre-split impacts and pre-sort by time for binary-search filtering
        all_imp = sorted(results.impacts, key=lambda r: r.time)
        self._all_impacts = all_imp
        self._impact_times = np.array([r.time for r in all_imp]) if all_imp else np.array([])

        # Lagrange points (in corotating-frame metres, CoM at origin)
        M = EARTH_MASS + LUNA_MASS
        rE = LUNA_MASS  * EARTH_LUNA_DIST / M
        rL = EARTH_MASS * EARTH_LUNA_DIST / M
        rH = EARTH_LUNA_DIST * (LUNA_MASS / (3 * EARTH_MASS)) ** (1 / 3)
        self._lagrange = {
            "L1": np.array([rL - rH, 0.0]),
            "L2": np.array([rL + rH, 0.0]),
            "L4": np.array([(rL - rE) / 2, +(np.sqrt(3) / 2) * EARTH_LUNA_DIST]),
            "L5": np.array([(rL - rE) / 2, -(np.sqrt(3) / 2) * EARTH_LUNA_DIST]),
        }

    # ── Public ───────────────────────────────────────────────────────────
    def show(self):
        self._build_figure()
        self._init_artists()
        self._build_widgets()
        self._refresh(0)
        plt.show()

    # ── Figure layout ────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(22, 9), facecolor="#0D0D1A")
        self.fig.suptitle("Earth–Luna Asteroid Impact Simulator",
                          color="white", fontsize=12, y=0.98)

        # Left: live simulation view
        self.ax = self.fig.add_axes([0.03, 0.13, 0.40, 0.82])
        self.ax.set_facecolor("#070714")
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x  [Earth–Luna distances]", color="#888", fontsize=8)
        self.ax.set_ylabel("y  [Earth–Luna distances]", color="#888", fontsize=8)
        self.ax.tick_params(colors="#666", labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color("#333")

        # Centre: full impact-trajectory traces (corotating frame, always)
        self.ax_traces = self.fig.add_axes([0.46, 0.13, 0.22, 0.82])
        self.ax_traces.set_facecolor("#070714")
        self.ax_traces.set_aspect("equal")
        self.ax_traces.tick_params(colors="#666", labelsize=7)
        for sp in self.ax_traces.spines.values():
            sp.set_color("#333")

        # Right-top: Luna impact polar histogram
        self.ax_pol = self.fig.add_axes([0.71, 0.52, 0.27, 0.43],
                                        projection="polar", facecolor="#070714")
        self.ax_pol.tick_params(colors="#888", labelsize=7)
        self.ax_pol.grid(color="#333", alpha=0.4)

        # Right-bottom: stats text
        self.ax_stats = self.fig.add_axes([0.71, 0.13, 0.27, 0.35],
                                          facecolor="#070714")
        self.ax_stats.axis("off")

    def _init_artists(self):
        bs = self.cfg.body_scale
        self.r_e_disp = EARTH_RADIUS * bs / D
        self.r_l_disp = LUNA_RADIUS * bs / D

        self.earth_circle = Circle((0, 0), self.r_e_disp,
                                   color="#4B9CD3", zorder=5)
        self.luna_circle = Circle((0, 0), self.r_l_disp,
                                  color="#C8C8C8", zorder=5)
        self.ax.add_patch(self.earth_circle)
        self.ax.add_patch(self.luna_circle)

        self.earth_lbl = self.ax.text(0, 0, "Earth", color="white",
                                      fontsize=7, ha="center", va="bottom",
                                      zorder=10)
        self.luna_lbl = self.ax.text(0, 0, "Luna", color="white",
                                     fontsize=7, ha="center", va="bottom",
                                     zorder=10)

        self.trails = LineCollection([], linewidths=0.5, zorder=3)
        self.ax.add_collection(self.trails)

        self.scatter = self.ax.scatter([], [], s=4, color="#FFC34A",
                                       alpha=0.9, zorder=6)
        self.imp_luna = self.ax.scatter([], [], s=18, marker="x",
                                        color="#FF4D4D", linewidths=1.3,
                                        zorder=7)
        self.imp_earth = self.ax.scatter([], [], s=18, marker="x",
                                         color="#FF8C1A", linewidths=1.3,
                                         zorder=7)

        self._lag_artists = {}
        for name in self._lagrange:
            (pt,) = self.ax.plot([], [], "+", color="#5BD15B", ms=9, mew=1.5,
                                 zorder=4, alpha=0.85)
            txt = self.ax.text(0, 0, name, color="#5BD15B", fontsize=7,
                               ha="center", va="bottom", zorder=4, alpha=0.85)
            self._lag_artists[name] = (pt, txt)

        self._field_cs = None
        self.time_txt = self.ax.text(0.02, 0.97, "", transform=self.ax.transAxes,
                                     color="white", fontsize=10, va="top")

        # Initial axis limits: cover source distance with margin
        lim = self.cfg.source_distance / D * 1.15
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

        legend_handles = [
            Patch(color="#4B9CD3", label="Earth"),
            Patch(color="#C8C8C8", label="Luna"),
            Patch(color="#FFC34A", label="Active asteroid"),
            Patch(color="#FF4D4D", label="Luna impact"),
            Patch(color="#FF8C1A", label="Earth impact"),
        ]
        self.ax.legend(handles=legend_handles, loc="lower right",
                       facecolor="#0D0D1A", edgecolor="#333",
                       labelcolor="white", fontsize=7)

    def _build_widgets(self):
        # Time slider (in days) — sits under the live sim panel
        ax_sl = self.fig.add_axes([0.03, 0.05, 0.40, 0.025],
                                  facecolor="#1A1A2E")
        self.slider = Slider(ax_sl, "t (days)",
                             0.0, self.r.times[-1] / 86400.0,
                             valinit=0.0, color="#4B9CD3")
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        # Speed slider — sits under the live sim panel (row 2)
        ax_sp = self.fig.add_axes([0.03, 0.01, 0.32, 0.025],
                                  facecolor="#1A1A2E")
        self.speed_slider = Slider(ax_sp, "Speed",
                                   1, 50, valinit=self._play_step,
                                   valstep=1, color="#C8C8C8")
        self.speed_slider.label.set_color("white")
        self.speed_slider.valtext.set_color("white")
        self.speed_slider.on_changed(self._on_speed)

        # Frame radio — sits under the traces panel
        ax_fr = self.fig.add_axes([0.46, 0.01, 0.10, 0.10],
                                  facecolor="#1A1A2E")
        self.radio = RadioButtons(ax_fr, self.FRAMES,
                                  active=self.FRAMES.index(self.frame))
        for lbl in self.radio.labels:
            lbl.set_color("white"); lbl.set_fontsize(7)
        self.radio.on_clicked(self._on_frame)

        # Toggles — sits to the right of frame radio
        ax_ck = self.fig.add_axes([0.58, 0.01, 0.11, 0.10],
                                  facecolor="#1A1A2E")
        self.checks = CheckButtons(
            ax_ck, self.TOGGLES,
            [self._show_trails, self._show_field,
             self._show_lagrange, self._show_impacts],
        )
        for lbl in self.checks.labels:
            lbl.set_color("white"); lbl.set_fontsize(7)
        self.checks.on_clicked(self._on_toggle)

        # Play/pause
        ax_pl = self.fig.add_axes([0.91, 0.05, 0.06, 0.04])
        self.play_btn = Button(ax_pl, "Play", color="#1A1A2E",
                               hovercolor="#2A2A4E")
        self.play_btn.label.set_color("white")
        self.play_btn.on_clicked(self._on_play)

        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._anim_tick)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Refresh pipeline ─────────────────────────────────────────────────
    def _refresh(self, t_idx: int):
        t_idx = int(np.clip(t_idx, 0, self.n_saved - 1))
        self.t_idx = t_idx

        ep = self.r.earth_pos[t_idx]
        lp = self.r.luna_pos[t_idx]
        t = self.r.times[t_idx]

        # Bodies
        ep_d = to_display(ep, self.frame, ep, lp)
        lp_d = to_display(lp, self.frame, ep, lp)
        self.earth_circle.center = tuple(ep_d)
        self.luna_circle.center = tuple(lp_d)
        self.earth_lbl.set_position((ep_d[0], ep_d[1] + self.r_e_disp * 1.3))
        self.luna_lbl.set_position((lp_d[0], lp_d[1] + self.r_l_disp * 1.6))

        # Asteroids + trails
        if self.r.ast_pos is not None:
            self._refresh_asteroids(t_idx, ep, lp)

        # Field
        if self._show_field:
            self._refresh_field(ep, lp)
        elif self._field_cs is not None:
            self._remove_contour()

        # Impact markers (only those that have happened by now)
        self._refresh_impacts(t, ep, lp)

        # Lagrange (only meaningful in corotating frame)
        self._refresh_lagrange()

        # Polar histogram + stats + impact traces
        self._refresh_polar(t)
        self._refresh_stats(t)
        self._refresh_traces(t)

        self.time_txt.set_text(f"t = {t / 86400:.2f} days")
        self.fig.canvas.draw_idle()

    # ── Asteroids and trails ─────────────────────────────────────────────
    def _refresh_asteroids(self, t_idx, ep, lp):
        trail_len = self.cfg.trail_length
        t0 = max(0, t_idx - trail_len)

        segs: list = []
        cols: list = []
        live: list = []

        ast_pos = self.r.ast_pos                 # (N, T, 2), NaN for inactive
        epH = self.r.earth_pos                   # (T, 2)
        lpH = self.r.luna_pos                    # (T, 2)

        for i in range(self.n_ast):
            window = ast_pos[i, t0:t_idx + 1, :]            # (k, 2)
            valid = ~np.isnan(window[:, 0])
            if not valid.any():
                continue

            last = int(np.where(valid)[0][-1])
            run_start = last
            while run_start > 0 and valid[run_start - 1]:
                run_start -= 1
            run = window[run_start:last + 1]                 # (W, 2)

            # Current "live" point only if asteroid is still active at t_idx
            if last == (t_idx - t0):
                cur_d = self._xform_run(run[-1:], np.array([t_idx]), ep, lp)
                live.append(cur_d[0])

            if not self._show_trails or len(run) < 2:
                continue

            j_idx = np.arange(t0 + run_start, t0 + last + 1)
            j_idx = np.clip(j_idx, 0, self.n_saved - 1)
            run_d = self._xform_run(run, j_idx, ep, lp)

            pts = run_d.reshape(-1, 1, 2)
            seg_arr = np.concatenate([pts[:-1], pts[1:]], axis=1)  # (W-1, 2, 2)
            n_seg = len(seg_arr)
            alphas = np.linspace(0.05, 0.7, n_seg)
            segs.extend(seg_arr)
            cols.extend((0.35, 0.85, 1.0, a) for a in alphas)

        if segs:
            self.trails.set_segments(segs)
            self.trails.set_color(cols)
        else:
            self.trails.set_segments([])

        if live:
            self.scatter.set_offsets(np.array(live))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))

    def _xform_run(self, pos_m, j_idx, ep_now, lp_now):
        """Per-step transform of a (W,2) trajectory chunk into display frame.

        Only the corotating frame is per-step (rotation angle changes); other
        frames just use the current Earth/Luna positions.
        """
        if self.frame != "corotating":
            return to_display(pos_m, self.frame, ep_now, lp_now)

        ep_w = self.r.earth_pos[j_idx]                       # (W, 2)
        lp_w = self.r.luna_pos[j_idx]                        # (W, 2)
        thetas = np.arctan2(lp_w[:, 1] - ep_w[:, 1],
                            lp_w[:, 0] - ep_w[:, 0])         # (W,)
        c = np.cos(thetas)
        s = np.sin(thetas)
        x = c * pos_m[:, 0] + s * pos_m[:, 1]
        y = -s * pos_m[:, 0] + c * pos_m[:, 1]
        return np.column_stack([x, y]) / D

    # ── Field overlay ────────────────────────────────────────────────────
    def _remove_contour(self):
        if self._field_cs is None:
            return
        try:
            for c in self._field_cs.collections:
                c.remove()
        except Exception:
            pass
        self._field_cs = None

    def _refresh_field(self, ep, lp):
        self._remove_contour()
        try:
            X, Y, z = _potential_grid(ep, lp, self.frame, self.r.omega,
                                      self.cfg.field_grid_n,
                                      self.ax.get_xlim(), self.ax.get_ylim())
            self._field_cs = self.ax.contour(
                X, Y, z, levels=18, colors="white",
                alpha=0.18, linewidths=0.4, zorder=2,
            )
        except Exception:
            self._field_cs = None

    # ── Impact markers ───────────────────────────────────────────────────
    def _refresh_impacts(self, t_now, ep, lp):
        if not self._show_impacts or len(self._impact_times) == 0:
            self.imp_luna.set_offsets(np.empty((0, 2)))
            self.imp_earth.set_offsets(np.empty((0, 2)))
            return
        cutoff = np.searchsorted(self._impact_times, t_now, side="right")
        seen = self._all_impacts[:cutoff]
        luna_pts = np.array([r.pos for r in seen if r.body == "luna"]) \
            if any(r.body == "luna" for r in seen) else np.empty((0, 2))
        earth_pts = np.array([r.pos for r in seen if r.body == "earth"]) \
            if any(r.body == "earth" for r in seen) else np.empty((0, 2))

        self.imp_luna.set_offsets(
            to_display(luna_pts, self.frame, ep, lp) if len(luna_pts) else np.empty((0, 2))
        )
        self.imp_earth.set_offsets(
            to_display(earth_pts, self.frame, ep, lp) if len(earth_pts) else np.empty((0, 2))
        )

    # ── Lagrange points ──────────────────────────────────────────────────
    def _refresh_lagrange(self):
        show = self._show_lagrange and self.frame == "corotating"
        for name, (pt, txt) in self._lag_artists.items():
            if not show:
                pt.set_data([], [])
                txt.set_visible(False)
                continue
            xy = self._lagrange[name] / D
            pt.set_data([xy[0]], [xy[1]])
            txt.set_position((xy[0], xy[1] + 0.04))
            txt.set_visible(True)

    # ── Polar histogram of Luna impacts ──────────────────────────────────
    def _refresh_polar(self, t_now):
        ax = self.ax_pol
        ax.clear()
        ax.set_facecolor("#070714")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.tick_params(colors="#888", labelsize=7)
        ax.grid(color="#333", alpha=0.4)
        ax.set_title("Luna impact angles\n(blue=near · red=far · -90°=leading)",
                     color="white", fontsize=7.5, pad=10)

        if len(self._impact_times) == 0:
            ax.text(0, 0, "no impacts yet", color="#888", ha="center")
            return

        cutoff = np.searchsorted(self._impact_times, t_now, side="right")
        angles = [r.angle for r in self._all_impacts[:cutoff] if r.body == "luna"]
        if not angles:
            ax.text(0, 0, "no Luna impacts yet", color="#888", ha="center")
            return

        bins = np.linspace(-np.pi, np.pi, 37)
        counts, _ = np.histogram(angles, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = bins[1] - bins[0]
        colors = ["#4B9CD3" if abs(c) < np.pi / 2 else "#FF4D4D" for c in centers]
        ax.bar(centers, counts, width=width, color=colors, alpha=0.8,
               edgecolor="#333", linewidth=0.3)
        _label_polar_directions(ax, int(counts.max()) if counts.size else 1)

    # ── Stats panel ──────────────────────────────────────────────────────
    def _refresh_stats(self, t_now):
        ax = self.ax_stats
        ax.clear(); ax.axis("off"); ax.set_facecolor("#070714")

        cutoff = np.searchsorted(self._impact_times, t_now, side="right")
        seen = self._all_impacts[:cutoff]
        n_l = sum(1 for r in seen if r.body == "luna")
        n_e = sum(1 for r in seen if r.body == "earth")
        luna_angs = [r.angle for r in seen if r.body == "luna"]
        n_near    = sum(1 for a in luna_angs if abs(a) < np.pi / 2)
        n_far     = n_l - n_near
        n_leading  = sum(1 for a in luna_angs if a < 0)   # toward orbital-motion side
        n_trailing = n_l - n_leading
        n_esc_total = int((self.r.status == 3).sum())
        n_act = self.n_ast - len(seen) - n_esc_total

        rows = [
            ("Total asteroids:", f"{self.n_ast}"),
            ("Luna impacts:",    f"{n_l}  ({100*n_l/max(self.n_ast,1):.1f}%)"),
            ("  near-side:",     f"{n_near}  ({100*n_near/max(n_l,1):.1f}%)"),
            ("  far-side:",      f"{n_far}  ({100*n_far/max(n_l,1):.1f}%)"),
            ("  leading (-90°):",f"{n_leading}  ({100*n_leading/max(n_l,1):.1f}%)"),
            ("  trailing(+90°):",f"{n_trailing}  ({100*n_trailing/max(n_l,1):.1f}%)"),
            ("Earth impacts:",   f"{n_e}  ({100*n_e/max(self.n_ast,1):.1f}%)"),
            ("Active (now):",    f"{max(n_act, 0)}"),
            ("Escaped (final):", f"{n_esc_total}"),
        ]
        y = 0.97
        for k, v in rows:
            ax.text(0.04, y, k, color="#AAB", fontsize=8, transform=ax.transAxes,
                    va="top", family="monospace")
            ax.text(0.58, y, v, color="white", fontsize=8, transform=ax.transAxes,
                    va="top", family="monospace")
            y -= 0.097

    # ── Impact trajectory traces ─────────────────────────────────────────────
    def _refresh_traces(self, t_now: float):
        """Draw complete paths of every impactor that has struck by t_now."""
        ax = self.ax_traces
        ax.clear()
        ax.set_facecolor("#070714")
        ax.set_aspect("equal")
        ax.set_title("Impact Trajectories  (corotating)",
                     color="white", fontsize=8, pad=6)
        ax.set_xlabel("x  [Earth–Luna dist]", color="#888", fontsize=7)
        ax.set_ylabel("y  [Earth–Luna dist]", color="#888", fontsize=7)
        ax.tick_params(colors="#666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.grid(color="#222", alpha=0.5)

        # Earth and Luna at fixed corotating positions
        M = EARTH_MASS + LUNA_MASS
        r_e_c = -(LUNA_MASS * EARTH_LUNA_DIST / M) / D
        r_l_c =  (EARTH_MASS * EARTH_LUNA_DIST / M) / D
        bs = self.cfg.body_scale
        ax.add_patch(Circle((r_e_c, 0), EARTH_RADIUS * bs / D,
                            color="#4B9CD3", zorder=5))
        ax.add_patch(Circle((r_l_c, 0), LUNA_RADIUS * bs / D,
                            color="#C8C8C8", zorder=5))
        ax.text(r_e_c, EARTH_RADIUS * bs / D * 1.8, "Earth",
                color="white", fontsize=6, ha="center", va="bottom", zorder=6)
        ax.text(r_l_c, LUNA_RADIUS * bs / D * 3.0, "Luna",
                color="white", fontsize=6, ha="center", va="bottom", zorder=6)

        if self.r.ast_pos is None:
            ax.text(0.5, 0.5, "save_trajectories=False",
                    color="#888", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
            return

        cutoff = int(np.searchsorted(self._impact_times, t_now, side="right"))
        seen = self._all_impacts[:cutoff]
        n_l = n_e = 0

        for rec in seen:
            i = rec.asteroid_idx
            t_idx_imp = min(
                int(np.searchsorted(self.r.times, rec.time, side="right")),
                len(self.r.times) - 1,
            )
            traj = self.r.ast_pos[i, :t_idx_imp + 1, :]
            valid = ~np.isnan(traj[:, 0])
            if not valid.any():
                continue

            valid_idx = np.where(valid)[0]
            first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
            traj_v = traj[first_v:last_v + 1]
            j_idx = np.clip(np.arange(first_v, last_v + 1),
                            0, len(self.r.earth_pos) - 1)

            # Always corotate regardless of the main panel's frame setting
            ep_w = self.r.earth_pos[j_idx]
            lp_w = self.r.luna_pos[j_idx]
            th = np.arctan2(lp_w[:, 1] - ep_w[:, 1],
                            lp_w[:, 0] - ep_w[:, 0])
            c, s = np.cos(th), np.sin(th)
            x = c * traj_v[:, 0] + s * traj_v[:, 1]
            y = -s * traj_v[:, 0] + c * traj_v[:, 1]
            traj_d = np.column_stack([x, y]) / D

            if rec.body == "luna":
                color, alpha, n_l = "#FF6B6B", 0.80, n_l + 1
            else:
                color, alpha, n_e = "#FFA040", 0.65, n_e + 1

            ax.plot(traj_d[:, 0], traj_d[:, 1],
                    color=color, lw=0.7, alpha=alpha, zorder=3)
            ax.plot(traj_d[-1, 0], traj_d[-1, 1], "x",
                    color=color, ms=7, mew=1.5, zorder=7)

        handles = []
        if n_l:
            handles.append(Patch(color="#FF6B6B", label=f"Luna hits ({n_l})"))
        if n_e:
            handles.append(Patch(color="#FFA040", label=f"Earth hits ({n_e})"))
        if handles:
            ax.legend(handles=handles, loc="upper right",
                      facecolor="#0D0D1A", edgecolor="#333",
                      labelcolor="white", fontsize=7)
        else:
            ax.text(0.5, 0.5, "no impacts yet", color="#888",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)

    # ── Callbacks ────────────────────────────────────────────────────────
    def _on_slider(self, val):
        t_target = val * 86400.0
        idx = int(np.searchsorted(self.r.times, t_target))
        idx = min(idx, self.n_saved - 1)
        self._refresh(idx)

    def _on_frame(self, label):
        self.frame = label
        self._refresh(self.t_idx)

    def _on_toggle(self, label):
        if label == "Trails":   self._show_trails = not self._show_trails
        elif label == "Field":  self._show_field = not self._show_field
        elif label == "Lagrange": self._show_lagrange = not self._show_lagrange
        elif label == "Impacts": self._show_impacts = not self._show_impacts
        self._refresh(self.t_idx)

    def _on_play(self, event):
        self._playing = not self._playing
        self.play_btn.label.set_text("Pause" if self._playing else "Play")
        if self._playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _on_speed(self, val):
        self._play_step = int(val)

    def _anim_tick(self):
        if not self._playing:
            return
        nxt = self.t_idx + self._play_step
        if nxt >= self.n_saved:
            nxt = 0
        self.slider.eventson = False
        self.slider.set_val(self.r.times[nxt] / 86400.0)
        self.slider.eventson = True
        self._refresh(nxt)

    def _on_key(self, event):
        if event.key == "right":
            self._step(+self._play_step)
        elif event.key == "left":
            self._step(-self._play_step)
        elif event.key == " ":
            self._on_play(event)
        elif event.key in ("]", "=", "+"):
            self._change_speed(+5)
        elif event.key in ("[", "-"):
            self._change_speed(-5)

    def _change_speed(self, delta):
        new_val = int(np.clip(self._play_step + delta, 1, 50))
        self._play_step = new_val
        self.speed_slider.eventson = False
        self.speed_slider.set_val(new_val)
        self.speed_slider.eventson = True

    def _step(self, delta):
        nxt = int(np.clip(self.t_idx + delta, 0, self.n_saved - 1))
        self.slider.eventson = False
        self.slider.set_val(self.r.times[nxt] / 86400.0)
        self.slider.eventson = True
        self._refresh(nxt)


# ── Standalone analysis figure ──────────────────────────────────────────────

def plot_impact_analysis(results: SimResults):
    """One static figure summarising impact statistics (call after the sim)."""
    luna = [r for r in results.impacts if r.body == "luna"]
    earth = [r for r in results.impacts if r.body == "earth"]
    n_total = results.ast_init_pos.shape[0]
    n_luna, n_earth = len(luna), len(earth)
    n_esc = int((results.status == 3).sum())
    n_act = int((results.status == 0).sum())

    fig = plt.figure(figsize=(13, 8), facecolor="#0D0D1A")
    fig.suptitle("Impact Analysis", color="white", fontsize=12)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.06, right=0.97, top=0.92, bottom=0.08)

    # 1) Polar histogram of Luna impact angles
    ax1 = fig.add_subplot(gs[0, 0], projection="polar", facecolor="#070714")
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(1)
    if luna:
        a = np.array([r.angle for r in luna])
        bins = np.linspace(-np.pi, np.pi, 37)
        counts, _ = np.histogram(a, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        w = bins[1] - bins[0]
        colors = ["#4B9CD3" if abs(c) < np.pi / 2 else "#FF4D4D" for c in centers]
        ax1.bar(centers, counts, width=w, color=colors, alpha=0.85,
                edgecolor="#222", linewidth=0.3)
        _label_polar_directions(ax1, int(counts.max()))
    ax1.set_title("Luna impact angles\n(blue=near · red=far · −90°=leading)",
                  color="white", fontsize=9, pad=12)
    ax1.tick_params(colors="#888", labelsize=7)
    ax1.grid(color="#333", alpha=0.4)

    # 2) Impact time histogram
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#070714")
    if luna:
        ax2.hist([r.time / 86400 for r in luna], bins=30,
                 color="#C8C8C8", alpha=0.85, edgecolor="#333")
    ax2.set_title("Luna impact times", color="white", fontsize=9)
    ax2.set_xlabel("days", color="#888", fontsize=8)
    ax2.set_ylabel("count", color="#888", fontsize=8)
    _style_axes(ax2)

    # 3) Outcome pie
    ax3 = fig.add_subplot(gs[0, 2], facecolor="#070714")
    sizes = [n_luna, n_earth, n_esc, n_act]
    labels = ["Luna", "Earth", "Escaped", "Active"]
    cols = ["#C8C8C8", "#4B9CD3", "#666", "#FFC34A"]
    nz = [(s, l, c) for s, l, c in zip(sizes, labels, cols) if s > 0]
    if nz:
        s, l, c = zip(*nz)
        wedges, texts, autotexts = ax3.pie(
            s, labels=l, colors=c, autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 8},
            wedgeprops={"edgecolor": "#222", "linewidth": 0.5},
        )
        for at in autotexts:
            at.set_color("#222")
    ax3.set_title("Outcomes", color="white", fontsize=9)

    # 4) Approach speed vs impact angle (scatter, coloured by speed)
    ax4 = fig.add_subplot(gs[1, :2], facecolor="#070714")
    if luna:
        speeds_init = np.linalg.norm(results.ast_init_vel, axis=1)
        ang_deg = [np.degrees(r.angle) for r in luna]
        spd = [speeds_init[r.asteroid_idx] for r in luna]
        sc = ax4.scatter(ang_deg, spd, c=spd, cmap="plasma", s=14,
                         alpha=0.8, zorder=3)
        cb = plt.colorbar(sc, ax=ax4)
        cb.set_label("initial speed (m/s)", color="#888", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="#888")
        plt.setp(cb.ax.get_yticklabels(), color="#888")
        ax4.axvline(0,   color="#AAB", ls="--", lw=0.8, alpha=0.7,
                    label="Near side (0°)")
        ax4.axvline(-90, color="#6AF", ls=":",  lw=0.9, alpha=0.8,
                    label="Leading (−90°)")
        ax4.axvline(+90, color="#FA6", ls=":",  lw=0.9, alpha=0.8,
                    label="Trailing (+90°)")
        ax4.legend(fontsize=7, facecolor="#0D0D1A", edgecolor="#333",
                   labelcolor="white", loc="upper right")
    ax4.set_xlim(-180, 180)
    ax4.set_xticks([-180, -90, 0, 90, 180])
    ax4.set_xticklabels(["Far\n−180°", "Leading\n−90°", "Near\n0°",
                          "Trailing\n+90°", "Far\n+180°"], color="#888", fontsize=7)
    ax4.set_xlabel("impact angle — 0°=near-side  ±180°=far-side  −90°=leading",
                   color="#888", fontsize=8)
    ax4.set_ylabel("initial speed (m/s)", color="#888", fontsize=8)
    ax4.set_title("Approach speed vs Luna impact angle",
                  color="white", fontsize=9)
    _style_axes(ax4)

    # 5) Summary text
    ax5 = fig.add_subplot(gs[1, 2], facecolor="#070714")
    ax5.axis("off")
    ang_arr   = np.array([r.angle for r in luna]) if luna else np.array([])
    n_near    = int(np.sum(np.abs(ang_arr) < np.pi / 2)) if ang_arr.size else 0
    n_far     = n_luna - n_near
    n_leading  = int(np.sum(ang_arr < 0)) if ang_arr.size else 0
    n_trailing = n_luna - n_leading
    near_excess = (n_near    - n_far)     / max(n_luna, 1) * 100
    lead_excess = (n_leading - n_trailing) / max(n_luna, 1) * 100
    lines = [
        f"Total asteroids:  {n_total}",
        f"Luna hits:        {n_luna} ({100*n_luna/max(n_total,1):.1f}%)",
        f"  near-side:      {n_near} ({100*n_near/max(n_luna,1):.1f}%)",
        f"  far-side:       {n_far} ({100*n_far/max(n_luna,1):.1f}%)",
        f"  near excess:    {near_excess:+.1f}%",
        f"  leading (-90°): {n_leading} ({100*n_leading/max(n_luna,1):.1f}%)",
        f"  trailing(+90°): {n_trailing} ({100*n_trailing/max(n_luna,1):.1f}%)",
        f"  lead excess:    {lead_excess:+.1f}%",
        f"Earth hits:       {n_earth} ({100*n_earth/max(n_total,1):.1f}%)",
        f"Escaped:          {n_esc} ({100*n_esc/max(n_total,1):.1f}%)",
        f"Active at end:    {n_act}",
    ]
    for i, ln in enumerate(lines):
        ax5.text(0.04, 0.97 - i * 0.083, ln, transform=ax5.transAxes,
                 color="white", fontsize=8, family="monospace", va="top")
    ax5.set_title("Summary", color="white", fontsize=9)

    plt.show()


def _style_axes(ax):
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.grid(color="#333", alpha=0.3)
