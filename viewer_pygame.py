"""
Fast interactive Pygame viewer for the Earth-Luna asteroid impact simulation.

Renders 500k+ asteroid positions per frame by writing directly into a NumPy
surface array — no per-asteroid Python loops.  Impact trajectories are drawn
as anti-aliased polylines.

Controls
--------
  Space / P            Play / Pause
  ← / →               Step one tick backward / forward
  Shift + ← / →       Step 10× faster
  Mouse wheel          Zoom in / out, centred on cursor
  Left-click + drag    Pan
  L                    Snap view to Luna
  E                    Snap view to Earth
  R                    Reset view (whole system)
  T                    Toggle asteroid dot rendering
  I                    Toggle: show ALL asteroids vs impactors only
  Q / Escape           Quit
"""

from __future__ import annotations
import sys
import numpy as np
import pygame

from constants import (EARTH_RADIUS, LUNA_RADIUS, EARTH_MASS, LUNA_MASS,
                       EARTH_LUNA_DIST)
from simulation import SimResults

D = EARTH_LUNA_DIST   # one display unit

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = (7,   7,  20)
COL_EARTH   = (75, 156, 211)
COL_LUNA    = (200, 200, 200)
COL_ACTIVE  = (255, 195,  74)   # yellow — active asteroid
COL_HIT_L   = (255,  80,  80)   # red    — Luna impactor
COL_HIT_E   = (255, 140,  50)   # orange — Earth impactor
COL_HUD     = (200, 200, 200)
COL_GRID    = ( 25,  25,  50)


class PygameViewer:
    """Interactive Pygame viewer.  Call .run() to open the window."""

    W, H = 1440, 900

    def __init__(self, results: SimResults) -> None:
        self.r = results
        self.cfg = results.config
        self.n_saved = len(results.times)

        # Pre-sort impacts for binary-search time filtering
        self._impacts = sorted(results.impacts, key=lambda x: x.time)
        self._imp_times = np.array([x.time for x in self._impacts]) \
            if self._impacts else np.array([])

        # Pre-compute corotated impact trajectories (done once at startup)
        self._imp_traj_d: dict[int, np.ndarray] = {}  # asteroid_idx → (T,2) display
        self._imp_traj_col: dict[int, tuple] = {}

        # View state
        self.zoom   = 0.18          # display-units → fraction of half-screen
        self.pan    = np.zeros(2)   # world centre of view (display units)
        self._drag_start: tuple | None = None  # mouse drag anchor (px)
        self._pan_at_drag: np.ndarray | None = None

        # Playback state
        self.t_idx    = 0
        self.playing  = False
        self.play_step = 5

        # Render toggles
        self._show_dots   = True
        self._show_traces = True
        self._only_impactors = False   # when True, only draw impactor dots

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        pygame.init()
        flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((self.W, self.H), flags)
        pygame.display.set_caption("Earth-Luna Asteroid Viewer")
        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_lg = pygame.font.SysFont("monospace", 17)
        self.clock  = pygame.time.Clock()

        self._precompute_traces()

        while True:
            for event in pygame.event.get():
                if not self._handle_event(event):
                    pygame.quit()
                    return

            if self.playing:
                nxt = self.t_idx + self.play_step
                if nxt >= self.n_saved:
                    nxt = self.n_saved - 1
                    self.playing = False
                self.t_idx = nxt

            self.W, self.H = self.screen.get_size()
            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(60)

    # ── Pre-computation ──────────────────────────────────────────────────────

    def _precompute_traces(self) -> None:
        """Build corotated display-frame trajectory for every impactor once."""
        if self.r.ast_pos is None:
            return
        for rec in self.r.impacts:
            i = rec.asteroid_idx
            t_idx_imp = min(
                int(np.searchsorted(self.r.times, rec.time, side="right")),
                self.n_saved - 1,
            )
            traj = self.r.ast_pos[i, :t_idx_imp + 1, :]   # (k, 2)
            valid = ~np.isnan(traj[:, 0])
            if not valid.any():
                continue
            vi = np.where(valid)[0]
            seg = traj[vi[0]: vi[-1] + 1]
            j   = np.clip(np.arange(vi[0], vi[-1] + 1), 0, self.n_saved - 1)
            self._imp_traj_d[i] = self._corot_traj(seg, j)
            self._imp_traj_col[i] = COL_HIT_L if rec.body == "luna" else COL_HIT_E

    # ── Frame rendering ──────────────────────────────────────────────────────

    def _draw_frame(self) -> None:
        surf = self.screen
        surf.fill(BG)

        ep = self.r.earth_pos[self.t_idx]
        lp = self.r.luna_pos[self.t_idx]
        ep_d = self._corot_now(ep[None, :], self.t_idx)[0]
        lp_d = self._corot_now(lp[None, :], self.t_idx)[0]

        self._draw_grid(surf)

        if self._show_traces:
            self._draw_traces(surf)

        if self._show_dots and self.r.ast_pos is not None:
            self._draw_asteroids(surf, ep_d, lp_d)

        self._draw_body(surf, ep_d, EARTH_RADIUS * self.cfg.body_scale / D,
                        COL_EARTH, "Earth")
        self._draw_body(surf, lp_d, LUNA_RADIUS * self.cfg.body_scale / D,
                        COL_LUNA,  "Luna")

        self._draw_impact_markers(surf)
        self._draw_hud(surf)

    def _draw_grid(self, surf: pygame.Surface) -> None:
        W, H = self.W, self.H
        # light grid lines every 0.5 display units
        step = 0.5
        lo_x = self._pix_to_world(np.array([[0, 0]]))[0, 0]
        hi_x = self._pix_to_world(np.array([[W, 0]]))[0, 0]
        lo_y = self._pix_to_world(np.array([[0, H]]))[0, 1]
        hi_y = self._pix_to_world(np.array([[0, 0]]))[0, 1]
        gx = np.arange(np.floor(lo_x / step) * step,
                       np.ceil(hi_x / step) * step + step, step)
        gy = np.arange(np.floor(lo_y / step) * step,
                       np.ceil(hi_y / step) * step + step, step)
        for x in gx:
            px = self._world_to_pix(np.array([[x, 0]]))[0, 0]
            pygame.draw.line(surf, COL_GRID, (px, 0), (px, H))
        for y in gy:
            py = self._world_to_pix(np.array([[0, y]]))[0, 1]
            pygame.draw.line(surf, COL_GRID, (0, py), (W, py))

    def _draw_asteroids(self, surf: pygame.Surface,
                        ep_d: np.ndarray, lp_d: np.ndarray) -> None:
        t = self.t_idx
        pos_m = self.r.ast_pos[:, t, :]       # (N, 2), NaN if inactive
        valid = ~np.isnan(pos_m[:, 0])

        if self._only_impactors:
            # Keep only asteroids that eventually impacted
            imp_idx = {rec.asteroid_idx for rec in self.r.impacts}
            imp_mask = np.zeros(len(valid), dtype=bool)
            for i in imp_idx:
                if i < len(imp_mask):
                    imp_mask[i] = True
            valid = valid & imp_mask

        pos_d = self._corot_now(pos_m[valid], t)     # (K, 2)

        if len(pos_d) == 0:
            return

        pix = self._world_to_pix(pos_d)               # (K, 2) int
        in_bounds = ((pix[:, 0] >= 0) & (pix[:, 0] < self.W) &
                     (pix[:, 1] >= 0) & (pix[:, 1] < self.H))
        pix_v = pix[in_bounds]
        if len(pix_v) == 0:
            return

        # Write all active dots in one NumPy operation — fast for 500k points
        arr = pygame.surfarray.pixels3d(surf)
        arr[pix_v[:, 0], pix_v[:, 1]] = COL_ACTIVE
        del arr   # release surface lock

    def _draw_traces(self, surf: pygame.Surface) -> None:
        t_now = self.r.times[self.t_idx]
        cutoff = int(np.searchsorted(self._imp_times, t_now, side="right"))
        for rec in self._impacts[:cutoff]:
            i = rec.asteroid_idx
            traj_d = self._imp_traj_d.get(i)
            if traj_d is None or len(traj_d) < 2:
                continue
            pix = self._world_to_pix(traj_d)          # (T, 2)
            col = self._imp_traj_col[i]
            # Only render segments where at least one endpoint is on-screen
            visible = ((pix[:, 0] >= -20) & (pix[:, 0] < self.W + 20) &
                       (pix[:, 1] >= -20) & (pix[:, 1] < self.H + 20))
            if not visible.any():
                continue
            pts = [(int(p[0]), int(p[1])) for p in pix]
            if len(pts) >= 2:
                pygame.draw.lines(surf, col, False, pts, 1)
            # Impact × marker
            ex, ey = pts[-1]
            sz = 5
            pygame.draw.line(surf, col, (ex - sz, ey - sz), (ex + sz, ey + sz), 2)
            pygame.draw.line(surf, col, (ex + sz, ey - sz), (ex - sz, ey + sz), 2)

    def _draw_impact_markers(self, surf: pygame.Surface) -> None:
        """Always-visible × marks at impact positions (even with no traj data)."""
        if self.r.ast_pos is not None:
            return   # traces panel already marks them
        t_now = self.r.times[self.t_idx]
        cutoff = int(np.searchsorted(self._imp_times, t_now, side="right"))
        for rec in self._impacts[:cutoff]:
            pos_d = self._corot_single(rec.pos, self.t_idx)
            px, py = self._world_to_pix(pos_d[None, :])[0]
            col = COL_HIT_L if rec.body == "luna" else COL_HIT_E
            sz = 6
            pygame.draw.line(surf, col, (px-sz, py-sz), (px+sz, py+sz), 2)
            pygame.draw.line(surf, col, (px+sz, py-sz), (px-sz, py+sz), 2)

    def _draw_body(self, surf: pygame.Surface, pos_d: np.ndarray,
                   radius_d: float, color: tuple, label: str) -> None:
        scale = self._scale()
        px, py = self._world_to_pix(pos_d[None, :])[0]
        r_px = max(int(radius_d * scale), 3)
        pygame.draw.circle(surf, color, (int(px), int(py)), r_px)
        lbl = self.font_sm.render(label, True, (255, 255, 255))
        surf.blit(lbl, (int(px) - lbl.get_width() // 2, int(py) - r_px - 16))

    def _draw_hud(self, surf: pygame.Surface) -> None:
        t = self.r.times[self.t_idx]
        cutoff = int(np.searchsorted(self._imp_times, t, side="right"))
        n_l = sum(1 for r in self._impacts[:cutoff] if r.body == "luna")
        n_e = sum(1 for r in self._impacts[:cutoff] if r.body == "earth")
        n_ast = self.r.ast_init_pos.shape[0]
        fps = self.clock.get_fps()

        lines = [
            f"t = {t/86400:.3f} days  ({self.t_idx}/{self.n_saved-1})",
            f"zoom = {self.zoom:.3f}  fps = {fps:.0f}",
            f"speed = {self.play_step} steps/frame",
            f"Luna hits: {n_l}   Earth hits: {n_e}   total: {n_ast}",
            f"{'PLAYING' if self.playing else 'PAUSED'}",
        ]
        y = 8
        for ln in lines:
            txt = self.font_sm.render(ln, True, COL_HUD)
            surf.blit(txt, (8, y))
            y += 18

        # Key reference
        hints = [
            "Space=play/pause  ←/→=step  Shift+←/→=fast",
            "scroll=zoom  drag=pan  L=Luna  E=Earth  R=reset",
            "T=toggle dots  I=impactors-only  Q=quit",
        ]
        y = self.H - 18 * len(hints) - 6
        for h in hints:
            txt = self.font_sm.render(h, True, (100, 100, 130))
            surf.blit(txt, (8, y))
            y += 18

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def _scale(self) -> float:
        """Pixels per display unit."""
        return self.zoom * min(self.W, self.H) / 2.0

    def _world_to_pix(self, xy_d: np.ndarray) -> np.ndarray:
        """(N,2) display coords → (N,2) int pixel coords (y-flipped)."""
        scale = self._scale()
        cx, cy = self.W / 2, self.H / 2
        px = (xy_d[:, 0] - self.pan[0]) * scale + cx
        py = cy - (xy_d[:, 1] - self.pan[1]) * scale
        return np.column_stack([px, py]).astype(np.int32)

    def _pix_to_world(self, pix: np.ndarray) -> np.ndarray:
        """(N,2) pixel coords → (N,2) display coords."""
        scale = self._scale()
        cx, cy = self.W / 2, self.H / 2
        wx = (pix[:, 0] - cx) / scale + self.pan[0]
        wy = (cy - pix[:, 1]) / scale + self.pan[1]
        return np.column_stack([wx, wy])

    def _corot_now(self, pos_m: np.ndarray, t_idx: int) -> np.ndarray:
        """(N,2) inertial metres → (N,2) corotating display units at t_idx."""
        ep = self.r.earth_pos[t_idx]
        lp = self.r.luna_pos[t_idx]
        th = np.arctan2(lp[1] - ep[1], lp[0] - ep[0])
        c, s = np.cos(th), np.sin(th)
        x = c * pos_m[:, 0] + s * pos_m[:, 1]
        y = -s * pos_m[:, 0] + c * pos_m[:, 1]
        return np.column_stack([x, y]) / D

    def _corot_single(self, pos_m: np.ndarray, t_idx: int) -> np.ndarray:
        """(2,) inertial metres → (2,) corotating display units at t_idx."""
        return self._corot_now(pos_m[None, :], t_idx)[0]

    def _corot_traj(self, pos_m: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
        """(T,2) inertial trajectory → (T,2) corotating display units."""
        ep_w = self.r.earth_pos[j_idx]
        lp_w = self.r.luna_pos[j_idx]
        th = np.arctan2(lp_w[:, 1] - ep_w[:, 1], lp_w[:, 0] - ep_w[:, 0])
        c, s = np.cos(th), np.sin(th)
        x = c * pos_m[:, 0] + s * pos_m[:, 1]
        y = -s * pos_m[:, 0] + c * pos_m[:, 1]
        return np.column_stack([x, y]) / D

    # ── Zoom / pan helpers ────────────────────────────────────────────────────

    def _zoom_at(self, factor: float, mouse_xy: tuple) -> None:
        """Zoom by factor, keeping the point under the mouse fixed."""
        mx, my = mouse_xy
        world_before = self._pix_to_world(np.array([[mx, my]]))[0]
        self.zoom = float(np.clip(self.zoom * factor, 0.01, 2000.0))
        world_after  = self._pix_to_world(np.array([[mx, my]]))[0]
        self.pan += world_before - world_after

    def _snap_to(self, world_xy: np.ndarray) -> None:
        self.pan = world_xy.copy()

    # ── Event handling ────────────────────────────────────────────────────────

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            shift = bool(mods & pygame.KMOD_SHIFT)
            step = self.play_step * (10 if shift else 1)

            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            elif event.key in (pygame.K_SPACE, pygame.K_p):
                self.playing = not self.playing
            elif event.key == pygame.K_RIGHT:
                self.t_idx = min(self.t_idx + step, self.n_saved - 1)
                self.playing = False
            elif event.key == pygame.K_LEFT:
                self.t_idx = max(self.t_idx - step, 0)
                self.playing = False
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.play_step = min(self.play_step + 5, 200)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.play_step = max(self.play_step - 5, 1)
            elif event.key == pygame.K_r:
                self.zoom = 0.18
                self.pan  = np.zeros(2)
            elif event.key == pygame.K_l:
                # Snap to Luna (corotating position ≈ +r_L/D on x-axis)
                M = EARTH_MASS + LUNA_MASS
                r_l_d = EARTH_MASS * EARTH_LUNA_DIST / M / D
                self._snap_to(np.array([r_l_d, 0.0]))
                self.zoom = 5.0
            elif event.key == pygame.K_e:
                M = EARTH_MASS + LUNA_MASS
                r_e_d = -(LUNA_MASS * EARTH_LUNA_DIST / M) / D
                self._snap_to(np.array([r_e_d, 0.0]))
                self.zoom = 3.0
            elif event.key == pygame.K_t:
                self._show_dots = not self._show_dots
            elif event.key == pygame.K_i:
                self._only_impactors = not self._only_impactors

        elif event.type == pygame.MOUSEWHEEL:
            factor = 1.15 ** event.y
            self._zoom_at(factor, pygame.mouse.get_pos())

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._drag_start   = event.pos
            self._pan_at_drag  = self.pan.copy()

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_start  = None
            self._pan_at_drag = None

        elif event.type == pygame.MOUSEMOTION and self._drag_start is not None:
            dx = event.pos[0] - self._drag_start[0]
            dy = event.pos[1] - self._drag_start[1]
            scale = self._scale()
            self.pan = self._pan_at_drag + np.array([-dx / scale, dy / scale])

        return True
