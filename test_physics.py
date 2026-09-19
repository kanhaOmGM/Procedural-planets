"""
test_physics.py — STEP 4: verifiable physics self-test.

Two independent checks on the symplectic (kick-drift-kick leapfrog) Taichi
integrator in starsystem.py, using a 1 Msun star and a massless test planet
on a 1 AU circular orbit (units: AU, Msun, yr, so G = 4*pi^2 and the orbital
period is exactly 1.0 year):

  1. Energy conservation — a symplectic integrator's energy should NOT drift
     secularly the way forward-Euler's does. After 10,000 steps the relative
     change in specific orbital energy must be under 1e-4.

  2. Kepler's third law — the time to complete one full revolution (measured
     as the return to the starting phase, i.e. periapsis-to-periapsis for
     this circular case) must match T = a^1.5 = 1.0 year.

Run:  python test_physics.py
"""

import math
import unittest

import numpy as np

import starsystem as ss


class TestSymplecticIntegrator(unittest.TestCase):
    M_STAR = 1.0
    A = 1.0
    DT = 1e-4
    N_STEPS = 10000

    def setUp(self):
        ss.ensure_taichi()
        self.mu = ss.G * self.M_STAR
        v_circ = math.sqrt(self.mu / self.A)

        # Body 0: the star, fixed at the origin (no back-reaction, since the
        # test planet is given zero mass — a clean two-body Kepler problem).
        # Body 1: the test planet, on a 1 AU circular orbit.
        pos0 = np.array([[0.0, 0.0, 0.0], [self.A, 0.0, 0.0]])
        vel0 = np.array([[0.0, 0.0, 0.0], [0.0, v_circ, 0.0]])
        mass0 = np.array([self.M_STAR, 0.0])
        ss.load_bodies(pos0, vel0, mass0)

    def _specific_orbital_energy(self):
        """eps = v^2/2 - mu/r for the test planet relative to the star."""
        p = ss.pos.to_numpy()[:2]
        v = ss.vel.to_numpy()[:2]
        r = np.linalg.norm(p[1] - p[0])
        v2 = float(np.dot(v[1] - v[0], v[1] - v[0]))
        return 0.5 * v2 - self.mu / r

    def test_energy_conservation(self):
        e_before = self._specific_orbital_energy()
        ss.run_steps(self.N_STEPS, self.DT)
        e_after = self._specific_orbital_energy()

        drift = abs((e_after - e_before) / e_before)
        print("\n[energy] before=%.10f after=%.10f  relative drift=%.3e"
              % (e_before, e_after, drift))
        self.assertLess(
            drift, 1e-4,
            "Specific orbital energy drifted by %.3e over %d leapfrog steps — "
            "a symplectic integrator should hold this essentially flat. "
            "(Forward Euler fails this by orders of magnitude.)" % (drift, self.N_STEPS))

    def test_keplers_third_law(self):
        """Integrate step-by-step, watching for the planet to sweep back
        through its starting phase (y crosses zero from below, going
        counter-clockwise, with x > 0) — one full revolution. For this
        circular orbit periapsis = apoapsis = constant radius, so "return to
        perihelion" is exactly "return to the starting point on the orbit"."""
        prev_y = 0.0
        period = None
        # Run for the specified 10,000 steps, plus a short safety margin so a
        # measured period landing a hair past exactly 1.0 year is still caught.
        max_steps = self.N_STEPS + 200
        for step in range(1, max_steps + 1):
            ss.leapfrog_step(self.DT)
            p = ss.pos.to_numpy()[1]
            x, y = p[0], p[1]
            if step > 1 and prev_y < 0.0 <= y and x > 0.0:
                frac = -prev_y / (y - prev_y)          # linear interpolation
                period = (step - 1 + frac) * self.DT
                break
            prev_y = y

        expected = self.A ** 1.5
        print("\n[kepler] measured period=%.6f yr  expected a^1.5=%.6f yr"
              % (period if period else float("nan"), expected))
        self.assertIsNotNone(period, "planet never completed a full revolution "
                              "within %d steps" % max_steps)
        self.assertAlmostEqual(
            period, expected, delta=1e-3,
            msg="Measured orbital period %.6f yr does not match Kepler's third "
                "law prediction a^1.5 = %.6f yr" % (period, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
