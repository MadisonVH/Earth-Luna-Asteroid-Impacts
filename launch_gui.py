"""Tkinter launch dialog — configure simulation parameters before running."""

from __future__ import annotations
from typing import Optional

try:
    import tkinter as tk
except ImportError as _tk_err:
    raise ImportError("tkinter is required for the GUI launcher") from _tk_err

_BG     = "#0D0D1A"
_CARD   = "#111126"
_LINE   = "#252540"
_ENTRY  = "#1A1A2E"
_ACCENT = "#4B9CD3"
_FG     = "#CCCCDD"
_MONO   = ("Consolas", 9)
_MONO_B = ("Consolas", 9, "bold")


class _ConfigDialog:
    def __init__(self):
        self.result: Optional[dict] = None
        self._entries: dict[str, tk.Entry] = {}
        self._vars:    dict[str, tk.Variable] = {}
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        root = tk.Tk()
        self._root = root
        root.title("Earth–Luna Asteroid Impact Simulator — Configure Run")
        root.configure(bg=_BG)
        root.resizable(True, False)

        # Title bar
        tk.Label(
            root, text="EARTH–LUNA ASTEROID IMPACT SIMULATOR",
            bg=_BG, fg=_ACCENT, font=("Consolas", 12, "bold"),
            pady=10, padx=16, anchor="w",
        ).pack(fill="x")
        tk.Frame(root, bg=_LINE, height=1).pack(fill="x")

        # Scrollable body
        body = tk.Frame(root, bg=_BG)
        body.pack(fill="both", expand=True)

        canvas = tk.Canvas(body, bg=_BG, highlightthickness=0, width=560, height=420)
        vsb = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(canvas, bg=_BG)
        win_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")

        def _resize(e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())

        self._inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"),
        )

        self._add_simulation_section()
        self._add_physics_section()
        self._add_asteroids_section()
        self._add_display_section()

        # Button bar
        tk.Frame(root, bg=_LINE, height=1).pack(fill="x")
        btn_bar = tk.Frame(root, bg=_CARD, pady=8)
        btn_bar.pack(fill="x")

        tk.Button(
            btn_bar, text="Cancel", command=self._cancel,
            bg=_ENTRY, fg=_FG, relief="flat", padx=16, pady=6,
            font=_MONO_B, activebackground=_LINE, activeforeground=_FG,
            cursor="hand2", bd=0,
        ).pack(side="right", padx=8)

        tk.Button(
            btn_bar, text="▶  Run Simulation", command=self._run,
            bg=_ACCENT, fg="#040416", relief="flat", padx=16, pady=6,
            font=_MONO_B, activebackground="#6BC8E8", activeforeground="#040416",
            cursor="hand2", bd=0,
        ).pack(side="right", padx=4)

        # Center window on screen
        root.update_idletasks()
        w = root.winfo_reqwidth()
        h = min(root.winfo_reqheight(), root.winfo_screenheight() - 80)
        x = (root.winfo_screenwidth()  - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _section(self, title: str):
        f = tk.Frame(self._inner, bg=_BG)
        f.pack(fill="x", padx=8, pady=(12, 4))
        tk.Label(f, text=title, bg=_BG, fg=_ACCENT,
                 font=_MONO_B, padx=2).pack(side="left")
        tk.Frame(f, bg=_LINE, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=5)

    def _pair_row(self) -> tk.Frame:
        f = tk.Frame(self._inner, bg=_BG)
        f.pack(fill="x", padx=16, pady=2)
        return f

    def _entry_cell(self, parent: tk.Frame, col: int,
                    label: str, key: str, default, width: int = 9):
        """One label+entry pair placed at grid column col*2 / col*2+1."""
        tk.Label(parent, text=label, bg=_BG, fg=_FG, font=_MONO,
                 anchor="e").grid(row=0, column=col * 2, sticky="e",
                                  padx=(8, 2), pady=3)
        e = tk.Entry(
            parent, bg=_ENTRY, fg=_FG, insertbackground=_FG,
            relief="flat", width=width, font=_MONO,
            highlightthickness=1, highlightcolor=_ACCENT,
            highlightbackground=_LINE,
        )
        e.insert(0, str(default))
        e.grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 8), pady=3)
        self._entries[key] = e

    def _radio_row(self, label: str, key: str,
                   options: list[tuple[str, str]], default: str):
        f = self._pair_row()
        tk.Label(f, text=label, bg=_BG, fg=_FG, font=_MONO,
                 width=20, anchor="w").pack(side="left")
        v = tk.StringVar(value=default)
        self._vars[key] = v
        for text, val in options:
            tk.Radiobutton(
                f, text=text, variable=v, value=val,
                bg=_BG, fg=_FG, selectcolor=_ENTRY,
                activebackground=_BG, activeforeground=_ACCENT,
                font=_MONO, cursor="hand2",
            ).pack(side="left", padx=6)

    def _check_row(self, label: str, key: str, default: bool):
        f = self._pair_row()
        v = tk.BooleanVar(value=default)
        self._vars[key] = v
        tk.Checkbutton(
            f, text=label, variable=v,
            bg=_BG, fg=_FG, selectcolor=_ENTRY,
            activebackground=_BG, activeforeground=_ACCENT,
            font=_MONO, cursor="hand2",
        ).pack(side="left")

    def _option_row(self, label: str, key: str, options: list[str], default: str):
        f = self._pair_row()
        tk.Label(f, text=label, bg=_BG, fg=_FG, font=_MONO,
                 width=20, anchor="w").pack(side="left")
        v = tk.StringVar(value=default)
        self._vars[key] = v
        om = tk.OptionMenu(f, v, *options)
        om.configure(bg=_ENTRY, fg=_FG, relief="flat", font=_MONO,
                     activebackground=_LINE, activeforeground=_ACCENT,
                     highlightthickness=0, width=12)
        om["menu"].configure(bg=_ENTRY, fg=_FG, font=_MONO,
                             activebackground=_ACCENT, activeforeground="#040416")
        om.pack(side="left", padx=4)

    # ── Sections ──────────────────────────────────────────────────────────────

    def _add_simulation_section(self):
        self._section("Simulation")
        r1 = self._pair_row()
        self._entry_cell(r1, 0, "Asteroids:", "n_asteroids", 500)
        self._entry_cell(r1, 1, "Duration (days):", "days", 30.0)
        r2 = self._pair_row()
        self._entry_cell(r2, 0, "Timestep (s):", "dt", 60.0)
        self._entry_cell(r2, 1, "Random seed:", "seed", 42)
        r3 = self._pair_row()
        self._entry_cell(r3, 0, "Release window (days):", "release_window", 27.32)
        self._entry_cell(r3, 1, "Save every N steps:", "save_every", "auto", width=8)

    def _add_physics_section(self):
        self._section("Physics")
        self._radio_row("Init mode:", "init_mode",
                        [("Monte Carlo", "montecarlo"), ("Grid", "grid")],
                        "montecarlo")
        self._radio_row("Integrator:", "integrator",
                        [("Verlet  (recommended)", "verlet"),
                         ("Euler", "euler"), ("RK4", "rk4")],
                        "verlet")
        self._check_row(
            "Analytic primaries — exact Earth/Luna orbit, zero accumulated error",
            "analytic_primaries", True,
        )

    def _add_asteroids_section(self):
        self._section("Asteroid parameters")
        r = self._pair_row()
        self._entry_cell(r, 0, "Speed min (km/s):", "spd_min", 5.0,  width=8)
        self._entry_cell(r, 1, "Speed max (km/s):", "spd_max", 25.0, width=8)
        r2 = self._pair_row()
        self._entry_cell(r2, 0, "Luna target frac:", "luna_frac", 0.3, width=8)

    def _add_display_section(self):
        self._section("Display")
        self._option_row(
            "Reference frame:", "frame",
            ["corotating", "inertial", "com", "earth", "luna"],
            "corotating",
        )
        self._radio_row("Viewer:", "viewer",
                        [("Matplotlib", "matplotlib"),
                         ("Pygame", "pygame"),
                         ("None", "none")],
                        "matplotlib")
        self._check_row("Skip analysis figure", "no_analysis", False)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _cancel(self):
        self._root.destroy()

    def _run(self):
        def flt(key: str, default: float) -> float:
            try:
                return float(self._entries[key].get())
            except ValueError:
                return default

        def int_(key: str, default: int) -> int:
            try:
                return int(self._entries[key].get())
            except ValueError:
                return default

        d: dict = dict(
            n_asteroids          = int_("n_asteroids", 500),
            duration             = flt("days", 30.0) * 86400.0,
            dt                   = flt("dt", 60.0),
            random_seed          = int_("seed", 42),
            release_window       = flt("release_window", 27.32) * 86400.0,
            init_mode            = self._vars["init_mode"].get(),
            integrator           = self._vars["integrator"].get(),
            analytic_primaries   = bool(self._vars["analytic_primaries"].get()),
            speed_range          = (flt("spd_min", 5.0)  * 1000.0,
                                    flt("spd_max", 25.0) * 1000.0),
            luna_target_fraction = flt("luna_frac", 0.3),
            reference_frame      = self._vars["frame"].get(),
        )
        # Optional save_every
        save_ev = self._entries["save_every"].get().strip().lower()
        if save_ev not in ("", "auto"):
            try:
                d["save_every"] = int(save_ev)
            except ValueError:
                pass

        d["_viewer"]      = self._vars["viewer"].get()
        d["_no_analysis"] = bool(self._vars["no_analysis"].get())

        self.result = d
        self._root.destroy()

    def show(self) -> Optional[dict]:
        self._root.mainloop()
        return self.result


def show_config_gui() -> Optional[dict]:
    """Show startup config dialog. Returns settings dict or None if cancelled."""
    return _ConfigDialog().show()
