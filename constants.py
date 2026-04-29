"""Physical constants and body parameters — all SI units (m, kg, s)."""

G = 6.674e-11  # gravitational constant, m^3 kg^-1 s^-2

# Earth
EARTH_MASS = 5.972e24   # kg
EARTH_RADIUS = 6.371e6  # m
EARTH_COLOR = "#4B9CD3"

# Luna (Moon)
LUNA_MASS = 7.342e22   # kg
LUNA_RADIUS = 1.737e6  # m
LUNA_COLOR = "#C8C8C8"

# System geometry
EARTH_LUNA_DIST = 3.844e8  # m, mean Earth–Moon distance (semi-major axis)

# Derived: circular-orbit angular velocity
import math
EARTH_LUNA_OMEGA = math.sqrt(G * (EARTH_MASS + LUNA_MASS) / EARTH_LUNA_DIST**3)
# ≈ 2.665e-6 rad/s  →  period ≈ 27.32 days ✓
