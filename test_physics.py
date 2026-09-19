"""
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


class TestEccentricOrbit(unittest.TestCase):
    """A circular orbit is the easy case: |r| never changes, so the force never
    changes magnitude and almost any integrator looks good. An eccentric orbit
    is where a non-symplectic scheme actually falls apart, so it is the more
    honest test of the claim."""

    def test_energy_conservation_eccentric(self):
        ss.ensure_taichi()
        M, a, e = 1.0, 1.0, 0.6
        mu = ss.G * M
        r_peri = a * (1.0 - e)
        v_peri = math.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))   # vis-viva at periapsis

        pos0 = np.array([[0.0, 0.0, 0.0], [r_peri, 0.0, 0.0]])
        vel0 = np.array([[0.0, 0.0, 0.0], [0.0, v_peri, 0.0]])
        ss.load_bodies(pos0, vel0, np.array([M, 0.0]))

        def energy():
            p = ss.pos.to_numpy()[:2]
            v = ss.vel.to_numpy()[:2]
            return 0.5 * float(np.dot(v[1], v[1])) - mu / np.linalg.norm(p[1] - p[0])

        e0 = energy()
        ss.run_steps(20000, 1e-4)      # ~20 orbits, through 20 periapsis passages
        drift = abs((energy() - e0) / e0)
        print("\n[eccentric e=0.6] relative energy drift over 20 orbits = %.3e" % drift)
        self.assertLess(drift, 1e-4)

    def test_semi_major_axis_is_recovered(self):
        """Energy fixes the semi-major axis: a = -mu / (2*E). If the integrator
        secretly changed the orbit, a would move even if E looked stable."""
        ss.ensure_taichi()
        M, a_true, e = 1.0, 1.0, 0.6
        mu = ss.G * M
        r_peri = a_true * (1.0 - e)
        v_peri = math.sqrt(mu * (1.0 + e) / (a_true * (1.0 - e)))
        ss.load_bodies(np.array([[0.0, 0.0, 0.0], [r_peri, 0.0, 0.0]]),
                       np.array([[0.0, 0.0, 0.0], [0.0, v_peri, 0.0]]),
                       np.array([M, 0.0]))
        ss.run_steps(20000, 1e-4)
        p = ss.pos.to_numpy()[:2]
        v = ss.vel.to_numpy()[:2]
        energy = 0.5 * float(np.dot(v[1], v[1])) - mu / np.linalg.norm(p[1] - p[0])
        a_measured = -mu / (2.0 * energy)
        print("[eccentric e=0.6] semi-major axis: true=%.6f  measured=%.6f AU"
              % (a_true, a_measured))
        self.assertAlmostEqual(a_measured, a_true, delta=1e-4)


class TestFullNBodySystem(unittest.TestCase):
    """The two-body tests use a massless test particle, so they never exercise
    the mutual planet-planet term at all. This one integrates a whole generated
    system with every body pulling on every other."""

    def test_total_energy_and_angular_momentum(self):
        ss.ensure_taichi()
        system = ss.generate_star_system(1.0, seed=42, n_planets=20)
        pos0, vel0 = ss.initial_state(system, seed=42)
        m = system["mass"]
        n = len(m)
        ss.load_bodies(pos0, vel0, m)

        def conserved():
            p = ss.pos.to_numpy()[:n]
            v = ss.vel.to_numpy()[:n]
            kin = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
            pot = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d = math.sqrt(float(np.sum((p[j] - p[i]) ** 2)) + ss.EPS2)
                    pot -= ss.G * m[i] * m[j] / d
            ang = np.sum(m[:, None] * np.cross(p, v), axis=0)
            return kin + pot, ang

        e0, l0 = conserved()
        ss.run_steps(100000, 2e-4)      # 20 simulated years, all 21 bodies coupled
        e1, l1 = conserved()

        e_drift = abs((e1 - e0) / e0)
        l_drift = float(np.linalg.norm(l1 - l0) / np.linalg.norm(l0))
        print("\n[21-body, 20 yr] total energy drift = %.3e   "
              "angular momentum drift = %.3e" % (e_drift, l_drift))
        self.assertLess(e_drift, 1e-4)
        self.assertLess(l_drift, 1e-8)   # leapfrog conserves L to round-off


class TestIntegratorChoiceMatters(unittest.TestCase):
    """Shows the threshold is not trivially passable: the same orbit, same step
    size, integrated with forward Euler, misses it by orders of magnitude. This
    is what makes the symplectic claim verifiable rather than asserted."""

    def test_forward_euler_fails_the_same_threshold(self):
        mu = 4.0 * math.pi ** 2
        dt, n_steps = 1e-4, 10000
        r = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, math.sqrt(mu), 0.0])

        def energy(r, v):
            return 0.5 * float(v @ v) - mu / float(np.linalg.norm(r))

        e0 = energy(r, v)
        for _ in range(n_steps):
            a = -mu * r / float(np.linalg.norm(r)) ** 3
            r = r + dt * v
            v = v + dt * a                 # forward Euler: stale acceleration
        drift = abs((energy(r, v) - e0) / e0)
        print("\n[forward Euler control] relative energy drift = %.3e "
              "(leapfrog: ~1e-14)" % drift)
        self.assertGreater(drift, 1e-4, "Euler unexpectedly passed — the "
                           "threshold is too loose to mean anything.")


class TestHoverPicking(unittest.TestCase):
    """The hover tooltip depends on projecting bodies to screen space on the CPU,
    because GGUI exposes no pick buffer. That projection must agree with what the
    renderer draws or the tooltip labels the wrong planet — so it is checked here
    rather than by squinting at the window."""

    W, H = 1280, 720

    def test_projection_landmarks(self):
        cam, look, up = [0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
        half = math.tan(math.radians(ss.FOV_DEG) * 0.5)

        xy, depth, _ = ss.project_to_screen(cam, look, up, [[0, 0, 0]], [0.1],
                                            self.W, self.H)
        self.assertAlmostEqual(xy[0][0], self.W / 2, places=6)
        self.assertAlmostEqual(xy[0][1], self.H / 2, places=6)
        self.assertAlmostEqual(depth[0], 10.0, places=9)

        # A point at the top of the vertical FOV must land on the top edge, and
        # one at the horizontal edge (FOV scaled by aspect) on the right edge.
        xy, _, _ = ss.project_to_screen(cam, look, up, [[0, 0, 10 * half]], [0.1],
                                        self.W, self.H)
        self.assertAlmostEqual(xy[0][1], self.H, places=5)
        xy, _, _ = ss.project_to_screen(cam, look, up,
                                        [[10 * half * (self.W / self.H), 0, 0]],
                                        [0.1], self.W, self.H)
        self.assertAlmostEqual(xy[0][0], self.W, places=5)

        # Behind the camera must report negative depth so picking can reject it.
        _, depth, _ = ss.project_to_screen(cam, look, up, [[0, -20, 0]], [0.1],
                                           self.W, self.H)
        self.assertLess(depth[0], 0.0)

    def test_apparent_radius_falls_as_inverse_distance(self):
        look, up = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
        _, _, r_near = ss.project_to_screen([0.0, -10.0, 0.0], look, up,
                                            [[0, 0, 0]], [0.1], self.W, self.H)
        _, _, r_far = ss.project_to_screen([0.0, -20.0, 0.0], look, up,
                                           [[0, 0, 0]], [0.1], self.W, self.H)
        self.assertAlmostEqual(r_far[0] / r_near[0], 0.5, places=9)

    def test_nearer_body_wins_the_hover(self):
        """Two bodies on the same line of sight: the one in front must take the
        hover, otherwise a distant giant steals the tooltip from a planet
        visibly occluding it."""
        cam, look, up = [0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])   # index 1 is farther
        xy, depth, rad = ss.project_to_screen(cam, look, up, pts, [0.3, 0.3],
                                              self.W, self.H)
        self.assertTrue(np.allclose(xy[0], xy[1]))
        self.assertEqual(ss.pick_body([self.W / 2, self.H / 2], xy, depth, rad), 0)
        self.assertEqual(ss.pick_body([2.0, 2.0], xy, depth, rad), -1)

    def test_every_visible_body_is_pickable_in_a_real_system(self):
        ss.ensure_taichi()
        system = ss.generate_star_system(1.0, seed=42, n_planets=20)
        p0, v0 = ss.initial_state(system, seed=42)
        ss.load_bodies(p0, v0, system["mass"])
        n = len(system["mass"])
        live = ss.pos.to_numpy()[:n]

        radii = np.clip((system["radius_rsun"] * 0.045) ** 0.5, 0.022, 0.065)
        radii[0] = system["star_vis_r"]
        cam = np.array([2.5, -2.0, 1.6]) * max(system["snow_line_au"], 1.0)
        xy, depth, rad = ss.project_to_screen(cam, [0, 0, 0], [0, 0, 1], live,
                                              radii, self.W, self.H)

        checked = 0
        for i in range(n):
            if depth[i] <= 0:
                continue
            if not (0 <= xy[i][0] <= self.W and 0 <= xy[i][1] <= self.H):
                continue
            checked += 1
            got = ss.pick_body(xy[i], xy, depth, rad)
            occluded = (got >= 0 and depth[got] < depth[i]
                        and np.linalg.norm(xy[got] - xy[i]) <= max(rad[got], 10.0))
            self.assertTrue(got == i or occluded,
                            "hovering body %d returned %d" % (i, got))
        print("\n[picking] %d on-screen bodies, every one resolves to itself "
              "or a genuine occluder" % checked)
        self.assertGreater(checked, 10)


class TestStickySelection(unittest.TestCase):
    """A plain hover cannot work here: the bodies are moving, so a planet leaves
    a stationary cursor within a frame or two. The selection has to latch."""

    def test_latch_rule(self):
        self.assertEqual(ss.update_selection(-1, 3), 3, "a hit selects")
        self.assertEqual(ss.update_selection(3, -1), 3, "a miss keeps it")
        self.assertEqual(ss.update_selection(3, 7), 7, "a new hit replaces it")
        self.assertEqual(ss.update_selection(3, -1, clear_requested=True), -1,
                         "explicit clear drops it")
        self.assertEqual(ss.update_selection(3, 7, clear_requested=True), 7,
                         "clicking ON a body selects it rather than clearing")
        self.assertEqual(ss.update_selection(0, -1), 0,
                         "body 0 (the star) must not be mistaken for 'nothing'")
        # The cursor_moved gate: with the mouse untouched, nothing on screen may
        # change the selection -- not a miss, and not another planet drifting in.
        self.assertEqual(ss.update_selection(3, -1, cursor_moved=False), 3)
        self.assertEqual(ss.update_selection(3, 7, cursor_moved=False), 3,
                         "a planet drifting under a still cursor must not steal it")
        self.assertEqual(ss.update_selection(3, -1, cursor_moved=False,
                                             clear_requested=True), -1,
                         "an explicit click still works without mouse movement")

    def test_selection_survives_a_planet_orbiting_off_the_cursor(self):
        """The actual reported bug, reproduced: park the cursor on a planet, let
        the integrator run, and confirm the panel would stay up even after the
        planet has moved far away from the cursor."""
        ss.ensure_taichi()
        system = ss.generate_star_system(1.0, seed=42, n_planets=20)
        p0, v0 = ss.initial_state(system, seed=42)
        ss.load_bodies(p0, v0, system["mass"])
        n = len(system["mass"])
        radii = np.clip((system["radius_rsun"] * 0.045) ** 0.5, 0.022, 0.065)
        radii[0] = system["star_vis_r"]
        cam = np.array([2.5, -2.0, 1.6]) * max(system["snow_line_au"], 1.0)

        def project():
            return ss.project_to_screen(cam, [0, 0, 0], [0, 0, 1],
                                        ss.pos.to_numpy()[:n], radii, 1280, 720)

        # Park the cursor exactly on an inner planet (short period, moves fast).
        xy, depth, rad = project()
        target = 1
        cursor = xy[target].copy()
        sel = ss.update_selection(-1, ss.pick_body(cursor, xy, depth, rad))
        self.assertEqual(sel, target)

        # Now advance a quarter of its orbit with the cursor held still.
        misses = 0
        for _ in range(40):
            ss.run_steps(50, 2e-4)
            xy, depth, rad = project()
            hit = ss.pick_body(cursor, xy, depth, rad)
            if hit < 0:
                misses += 1
            # cursor_moved=False: the mouse is being held perfectly still, which
            # is exactly the case that used to drop (or hijack) the panel.
            sel = ss.update_selection(sel, hit, cursor_moved=False)

        drift_px = float(np.linalg.norm(xy[target] - cursor))
        print("\n[sticky] planet drifted %.0f px from the cursor over %d frames, "
              "%d of them outright misses; selection held = %s"
              % (drift_px, 40, misses, sel == target))
        self.assertGreater(misses, 0, "planet never left the cursor — this test "
                           "is not exercising the latch")
        self.assertEqual(sel, target, "selection was lost when the planet moved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
