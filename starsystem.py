"""
starsystem.py — procedural star-system generator + Taichi symplectic N-body sim.

Units: AU, solar masses, years  ->  G = 4*pi^2 exactly, so a 1 AU circular
orbit around a 1 Msun star has a period of exactly 1.0 year. This is what
lets test_physics.py check Kepler's third law with nothing fuzzier than
floating-point error.

Run:      python starsystem.py --mass 1.4          (needs: pip install taichi)
Headless: python starsystem.py --mass 1.4 --bench out.png
Verify:   python test_physics.py
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import taichi as ti
    HAVE_TI = True
except ImportError:
    HAVE_TI = False

import real_sky

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ units ---
# EQ (2): G = 4*pi^2 AU^3 / (Msun * yr^2). Everything downstream is in these
# units — never seconds, never kilograms.
G = 4.0 * math.pi ** 2
SUN_TEMP_K = 5772.0
EARTH_MASS_MSUN = 3.003e-6      # 1 Earth mass, in solar masses
EPS = 1e-3                       # softening length (AU) — required in EVERY
EPS2 = EPS * EPS                 # distance/sqrt in the physics, per spec

MAX_BODIES = 32                  # star + up to 20 planets, headroom to spare


# ============================================================ STEP 2 =======
# Procedural star system: one number (stellar mass) in, a whole system out.
# =============================================================================
def generate_star_system(stellar_mass, seed=None, n_planets=None):
    """
    Build a main-sequence star of mass `stellar_mass` (solar masses) and a
    randomly-spawned family of 15-20 planets around it.

        L = M^3.5                          mass-luminosity relation
        R = M^0.8                          main-sequence radius
        T = Tsun * (L / R^2)^0.25          Stefan-Boltzmann, rearranged
        r_HZ = sqrt(L)                     habitable-zone distance (AU)
        snow line = 2.7 * sqrt(L)          rocky/gas-giant boundary (AU)

    Planets are dropped at random semi-major axes; anything inside the snow
    line condenses as a Rocky Planet (silicate/rocky colour, terrestrial
    mass), anything beyond it is a Gas/Ice Giant (icy/gaseous colour, an
    envelope ~100x heavier than a terrestrial body of the same size class).

    Returns a dict of numpy arrays plus the derived stellar properties, ready
    to hand to init_bodies() / the renderer.
    """
    rng = np.random.default_rng(seed)
    M = float(stellar_mass)

    L = M ** 3.5
    R = M ** 0.8
    T = SUN_TEMP_K * (L / R ** 2) ** 0.25
    r_hz = math.sqrt(L)
    snow_line = 2.7 * math.sqrt(L)

    if n_planets is None:
        n_planets = int(rng.integers(15, 21))     # 15..20 inclusive

    names = ["Star"]
    mass = [M]
    radius_rsun = [R]                              # solar radii
    kind = ["Star"]
    color = [tuple(real_sky.blackbody_rgb(T).tolist())]
    semi_major = [0.0]
    eccentricity = [0.0]
    inclination = [0.0]

    # Ensure no planet's orbit or periapsis intersects the star's visual radius
    star_vis_r = min(max((R * 0.045) ** 0.5, 0.06), 0.12) * min(max(M ** 0.3, 0.75), 1.1)
    min_star_clearance = star_vis_r + max(0.06, 0.10 * math.sqrt(M))
    a_lo = min_star_clearance / (1.0 - 0.15)
    a_hi = max(8.0 * snow_line, 2.0)

    # Spacing that guarantees non-crossing orbits and non-intersecting spheres
    axes = []
    ecc = []
    vis_radii = []
    log_ratio = (math.log(a_hi) - math.log(a_lo)) / max(n_planets, 1)

    for k in range(n_planets):
        e = float(abs(rng.normal(0.015, 0.02)))
        e = min(e, 0.10)

        # Estimate mass and visual radius to guarantee 3D sphere clearance
        a_guess = a_lo * math.exp(log_ratio * (k + 0.5))
        if a_guess < 0.8 * snow_line:
            planet_kind = "Rocky Planet"
            m = float(rng.uniform(0.5, 2.5)) * EARTH_MASS_MSUN
            base = rng.uniform(0.40, 0.85, 3) * np.array([1.00, 0.75, 0.55])
        elif a_guess < 1.3 * snow_line:
            planet_kind = "Super-Earth / Sub-Neptune"
            m = float(rng.uniform(3.0, 15.0)) * EARTH_MASS_MSUN
            base = rng.uniform(0.45, 0.80, 3) * np.array([0.70, 0.85, 0.75])
        else:
            planet_kind = "Gas/Ice Giant"
            m = float(rng.uniform(20.0, 100.0)) * EARTH_MASS_MSUN
            icy = min(1.0, (a_guess - snow_line) / max(snow_line, 1e-6))
            base = np.array([0.55 + 0.30 * icy, 0.65 + 0.25 * icy, 0.85 + 0.15 * icy])

        r_rsun = (m / EARTH_MASS_MSUN) ** (1.0 / 3.0) * 6371.0 / 695700.0
        vis_r = float(np.clip((r_rsun * 0.045) ** 0.5, 0.022, 0.065))

        # Enforce periapsis clearance exceeding sum of visual radii plus safety margin
        if k == 0:
            a_min = (star_vis_r + vis_r + 0.06) / (1.0 - e)
        else:
            prev_apo = axes[-1] * (1.0 + ecc[-1])
            prev_vis_r = vis_radii[-1]
            # Buffer MUST exceed the sum of visual radii so spheres NEVER touch
            physical_buffer = (prev_vis_r + vis_r) + max(0.04, 0.08 * axes[-1])
            a_min = (prev_apo + physical_buffer) / (1.0 - e)

        a_nom = a_lo * math.exp(log_ratio * (k + rng.uniform(0.1, 0.9)))
        a = max(a_min, a_nom)

        axes.append(a)
        ecc.append(e)
        vis_radii.append(vis_r)

        names.append("%s-%02d" % ("R" if "Rocky" in planet_kind else ("S" if "Super" in planet_kind else "G"), k + 1))
        mass.append(m)
        radius_rsun.append(r_rsun)
        kind.append(planet_kind)
        color.append(tuple(np.clip(base, 0.05, 1.0)))
        semi_major.append(float(a))
        eccentricity.append(e)
        inclination.append(math.radians(float(abs(rng.normal(0.0, 2.0)))))

    return {
        "stellar_mass": M, "luminosity": L, "stellar_radius": R,
        "temperature": T, "habitable_zone_au": r_hz, "snow_line_au": snow_line,
        "names": names, "mass": np.array(mass), "radius_rsun": np.array(radius_rsun),
        "kind": kind, "color": np.array(color), "semi_major_au": np.array(semi_major),
        "eccentricity": np.array(eccentricity), "inclination": np.array(inclination),
        "star_vis_r": star_vis_r,
    }


def initial_state(system):
    """Turn (a, e, inc) elements into (pos, vel) state vectors, each planet at
    a random point on its own orbit, in the star's rest frame (mu = G*M_star —
    the star is by far the dominant mass, so this is the standard two-body
    vis-viva placement, done independently per planet)."""
    rng = np.random.default_rng()
    a = system["semi_major_au"]
    e = system["eccentricity"]
    inc = system["inclination"]
    n = len(a)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    mu = G * system["stellar_mass"]
    for i in range(1, n):
        nu = float(rng.uniform(0.0, 2.0 * math.pi))
        r = a[i] * (1.0 - e[i] ** 2) / (1.0 + e[i] * math.cos(nu))
        h = math.sqrt(mu * a[i] * (1.0 - e[i] ** 2))
        rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
        vp = np.array([-mu / h * math.sin(nu), mu / h * (e[i] + math.cos(nu)), 0.0])
        ci, si = math.cos(inc[i]), math.sin(inc[i])
        tilt = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
        pos[i] = tilt @ rp
        vel[i] = tilt @ vp
    return pos, vel


# ============================================================ STEP 3 =======
# Taichi N-body physics. Every update lives inside @ti.kernel functions, uses
# a symplectic (kick-drift-kick leapfrog) integrator, works in AU / Msun / yr
# with G = 4*pi^2, and softens every distance with eps = 1e-3 AU.
# =============================================================================
_ti_ready = False


def ensure_taichi(arch=None):
    """Lazy, idempotent ti.init() — so importing this module (e.g. from
    test_physics.py) never opens a window and never double-inits.

    Defaults to the CPU backend: it needs no display driver, so the physics
    (and test_physics.py) run identically on a headless grading machine or a
    laptop with a GPU. Pass arch=ti.vulkan/ti.gpu explicitly for the
    interactive renderer on a machine you know has a working GPU driver."""
    global _ti_ready
    if _ti_ready:
        return
    if not HAVE_TI:
        raise RuntimeError("taichi is not installed: pip install taichi")
    ti.init(arch=arch or ti.cpu, default_fp=ti.f64)
    _declare_fields()
    _ti_ready = True


def _declare_fields():
    global pos, vel, acc, mass, n_bodies, grav_scale
    pos = ti.Vector.field(3, dtype=ti.f64, shape=MAX_BODIES)
    vel = ti.Vector.field(3, dtype=ti.f64, shape=MAX_BODIES)
    acc = ti.Vector.field(3, dtype=ti.f64, shape=MAX_BODIES)
    mass = ti.field(dtype=ti.f64, shape=MAX_BODIES)
    n_bodies = ti.field(dtype=ti.i32, shape=())
    grav_scale = ti.field(dtype=ti.f64, shape=())
    grav_scale[None] = 1.0

    @ti.kernel
    def _compute_accelerations():
        # EQ (1): a_i = sum_{j!=i} G m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^1.5
        n = n_bodies[None]
        g_scale = grav_scale[None]
        for i in range(n):
            a = ti.Vector([0.0, 0.0, 0.0])
            for j in range(n):
                if i != j:
                    r = pos[j] - pos[i]
                    dist2 = r.dot(r) + EPS2      # softened — never divides by zero
                    inv_d3 = dist2 ** (-1.5)
                    mult = 1.0
                    if i > 0 and j > 0:
                        mult = g_scale
                    a += G * mass[j] * r * inv_d3 * mult
            acc[i] = a

    @ti.kernel
    def _leapfrog_step(dt: ti.f64):
        """Kick-drift-kick leapfrog, entirely on-device. Symplectic (area
        preserving), which is why the orbit's energy oscillates in a bounded
        band forever instead of spiralling the way forward-Euler does."""
        n = n_bodies[None]
        g_scale = grav_scale[None]
        for i in range(n):                       # half-kick
            vel[i] += 0.5 * dt * acc[i]
        for i in range(n):                        # full drift
            pos[i] += dt * vel[i]
        for i in range(n):                        # recompute forces at new positions
            a = ti.Vector([0.0, 0.0, 0.0])
            for j in range(n):
                if i != j:
                    r = pos[j] - pos[i]
                    dist2 = r.dot(r) + EPS2
                    inv_d3 = dist2 ** (-1.5)
                    mult = 1.0
                    if i > 0 and j > 0:
                        mult = g_scale
                    a += G * mass[j] * r * inv_d3 * mult
            acc[i] = a
        for i in range(n):                        # half-kick
            vel[i] += 0.5 * dt * acc[i]

    global compute_accelerations, leapfrog_step
    compute_accelerations = _compute_accelerations
    leapfrog_step = _leapfrog_step


def load_bodies(pos_np, vel_np, mass_np):
    """Push a numpy N-body state onto the Taichi fields and prime acc[]."""
    ensure_taichi()
    n = len(mass_np)
    assert n <= MAX_BODIES, "raise MAX_BODIES for a system this big"
    n_bodies[None] = n
    pad = np.zeros((MAX_BODIES, 3))
    pad[:n] = pos_np
    pos.from_numpy(pad)
    pad = np.zeros((MAX_BODIES, 3))
    pad[:n] = vel_np
    vel.from_numpy(pad)
    padm = np.zeros(MAX_BODIES)
    padm[:n] = mass_np
    mass.from_numpy(padm)
    compute_accelerations()


def run_steps(n_steps, dt):
    for _ in range(n_steps):
        leapfrog_step(dt)


# ================================================================ render ====
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mass", type=float, default=1.0, help="stellar mass, Msun")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--planets", type=int, default=None)
    p.add_argument("--dt", type=float, default=2e-4, help="years per physics step")
    p.add_argument("--speed", type=float, default=0.6, help="sim years per real second")
    p.add_argument("--bench", type=str, default="", help="offscreen render, save PNG, exit")
    p.add_argument("--perturb", type=float, default=1.0, help="planet-planet gravity multiplier (1.0 = real physics, >1.0 to amplify mutual tugs)")
    p.add_argument("--gpu", action="store_true", help="use the GPU backend for rendering "
                   "(needs a working Vulkan/Metal/CUDA driver; default is CPU, which is "
                   "always safe but slower for the interactive window)")
    args = p.parse_args()

    ensure_taichi(arch=ti.gpu if args.gpu else None)
    grav_scale[None] = float(args.perturb)

    system = generate_star_system(args.mass, seed=args.seed, n_planets=args.planets)
    pos_np, vel_np = initial_state(system)
    load_bodies(pos_np, vel_np, system["mass"])
    n = len(system["mass"])

    print("Star: %.2f Msun -> L=%.3f Lsun  R=%.2f Rsun  T=%.0f K" %
          (system["stellar_mass"], system["luminosity"], system["stellar_radius"],
           system["temperature"]))
    print("Habitable zone: %.2f AU   Snow line: %.2f AU   %d planets" %
          (system["habitable_zone_au"], system["snow_line_au"], n - 1))

    sky = real_sky.load_sky(os.path.join(HERE, "bsc5_stars.csv"))

    window = ti.ui.Window("Procedural Star System", (1280, 720), show_window=not args.bench)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    body_radius = np.clip((system["radius_rsun"] * 0.045) ** 0.5, 0.022, 0.065).astype(np.float32)
    body_radius[0] = system["star_vis_r"]
    radius_field = ti.field(dtype=ti.f32, shape=n)
    radius_field.from_numpy(body_radius.astype(np.float32))

    # Star core has brilliant high-luminance white-hot core
    star_core_color = np.copy(system["color"])
    star_core_color[0] = np.clip(system["color"][0] * 0.4 + np.array([0.7, 0.7, 0.7]), 0.0, 1.0)
    color_field = ti.Vector.field(3, dtype=ti.f32, shape=n)
    color_field.from_numpy(star_core_color.astype(np.float32))

    # Precompute Keplerian orbit ellipse lines for every planet
    orbit_pts = []
    orbit_cols = []
    for i in range(1, n):
        a_i, e_i = system["semi_major_au"][i], system["eccentricity"][i]
        inc_i = system["inclination"][i]
        ci, si = math.cos(inc_i), math.sin(inc_i)
        tilt = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
        n_seg = 120
        theta = np.linspace(0, 2 * math.pi, n_seg + 1)
        # Planet-matched soft luminous orbit color (40% intensity)
        c_line = system["color"][i] * 0.40
        for j in range(n_seg):
            r1 = a_i * (1.0 - e_i ** 2) / (1.0 + e_i * math.cos(theta[j]))
            r2 = a_i * (1.0 - e_i ** 2) / (1.0 + e_i * math.cos(theta[j + 1]))
            p1 = tilt @ np.array([r1 * math.cos(theta[j]), r1 * math.sin(theta[j]), 0.0])
            p2 = tilt @ np.array([r2 * math.cos(theta[j + 1]), r2 * math.sin(theta[j + 1]), 0.0])
            orbit_pts.append(p1)
            orbit_pts.append(p2)
            orbit_cols.append(c_line)
            orbit_cols.append(c_line)

    orbit_pts = np.array(orbit_pts, dtype=np.float32)
    orbit_cols = np.array(orbit_cols, dtype=np.float32)
    orbit_field = ti.Vector.field(3, dtype=ti.f32, shape=len(orbit_pts))
    orbit_field.from_numpy(orbit_pts)
    orbit_col_field = ti.Vector.field(3, dtype=ti.f32, shape=len(orbit_cols))
    orbit_col_field.from_numpy(orbit_cols)

    # Dynamic trailing history buffer to display true simulated N-body paths
    TRAIL_LEN = 35
    trail_buf = np.zeros((TRAIL_LEN, n, 3), dtype=np.float32)
    for t_step in range(TRAIL_LEN):
        trail_buf[t_step] = pos_np[:n]
    trail_ptr = 0

    n_trail_segs = (n - 1) * (TRAIL_LEN - 1)
    trail_field = ti.Vector.field(3, dtype=ti.f32, shape=n_trail_segs * 2)
    trail_col_field = ti.Vector.field(3, dtype=ti.f32, shape=n_trail_segs * 2)

    # Setup static coronal flare ray directions and halo buffers
    n_rays = 120
    rng_rays = np.random.default_rng(88)
    phi = rng_rays.uniform(0, 2 * math.pi, n_rays)
    costheta = rng_rays.uniform(-1, 1, n_rays)
    sintheta = np.sqrt(np.maximum(0, 1 - costheta ** 2))
    ray_dirs = np.stack([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta], axis=1)
    ray_lens = rng_rays.uniform(body_radius[0] * 1.2, body_radius[0] * 1.9, n_rays)

    n_rings = 18
    n_ring_segs = 64
    n_glow_pts = n_rings * n_ring_segs * 2 + n_rays * 2
    glow_field = ti.Vector.field(3, dtype=ti.f32, shape=n_glow_pts)
    glow_col_field = ti.Vector.field(3, dtype=ti.f32, shape=n_glow_pts)

    sky_dir = sky_col = sky_rad = None
    if sky is not None:
        far = 4000.0
        sky_pos = ti.Vector.field(3, dtype=ti.f32, shape=sky["n"])
        sky_col = ti.Vector.field(3, dtype=ti.f32, shape=sky["n"])
        sky_rad = ti.field(dtype=ti.f32, shape=sky["n"])
        sky_pos.from_numpy((sky["direction"] * far).astype(np.float32))
        sky_col.from_numpy(sky["rgb"].astype(np.float32))
        sky_rad.from_numpy((sky["relative_flux"] * 8.0 + 0.6).astype(np.float32))

    init_cam_pos = np.array([2.5, -2.0, 1.6]) * max(system["snow_line_au"], 1.0)
    camera.position(*init_cam_pos)
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)

    def draw_frame():
        nonlocal trail_ptr
        pos_np_frame = pos.to_numpy()[:n].astype(np.float32)
        star_pos = pos_np_frame[0]
        canvas.set_background_color((0.0, 0.0, 0.01))
        scene.set_camera(camera)
        scene.ambient_light((0.10, 0.10, 0.12))
        scene.point_light(pos=tuple(star_pos), color=(1.0, 0.97, 0.9))
        scene.point_light(pos=tuple(camera.curr_position), color=(1.1, 1.1, 1.1))

        # Update dynamic trail buffer
        trail_buf[trail_ptr] = pos_np_frame
        trail_ptr = (trail_ptr + 1) % TRAIL_LEN

        t_pts = []
        t_cols = []
        for i_p in range(1, n):
            c_base = system["color"][i_p]
            for s in range(TRAIL_LEN - 1):
                idx1 = (trail_ptr + s) % TRAIL_LEN
                idx2 = (trail_ptr + s + 1) % TRAIL_LEN
                fade = float(s + 1) / float(TRAIL_LEN)
                t_pts.append(trail_buf[idx1, i_p])
                t_pts.append(trail_buf[idx2, i_p])
                t_cols.append(c_base * (fade * 0.9))
                t_cols.append(c_base * (fade * 0.9))

        trail_field.from_numpy(np.array(t_pts, dtype=np.float32))
        trail_col_field.from_numpy(np.array(t_cols, dtype=np.float32))

        # Dynamic billboard halo facing camera view direction
        cam_pos_curr = np.array(camera.curr_position, dtype=np.float32)
        v_dir = star_pos - cam_pos_curr
        v_dist = np.linalg.norm(v_dir)
        if v_dist > 1e-4:
            v_dir = v_dir / v_dist
        else:
            v_dir = np.array([0, -1, 0], dtype=np.float32)
        up_vec = np.array([0, 0, 1], dtype=np.float32)
        u_vec = np.cross(v_dir, up_vec)
        if np.linalg.norm(u_vec) > 1e-4:
            u_vec = u_vec / np.linalg.norm(u_vec)
        else:
            u_vec = np.array([1, 0, 0], dtype=np.float32)
        w_vec = np.cross(u_vec, v_dir)
        w_vec = w_vec / np.linalg.norm(w_vec)

        # Build dynamic corona halo rings and flare lines
        glow_pts = []
        glow_cols = []
        star_color = system["color"][0]
        ring_radii = np.linspace(body_radius[0] * 1.02, body_radius[0] * 1.95, n_rings)
        theta_ring = np.linspace(0, 2 * math.pi, n_ring_segs + 1)
        for i_r, r in enumerate(ring_radii):
            fade = ((1.0 - (i_r / n_rings)) ** 1.6) * 0.70
            c = star_color * fade
            for j in range(n_ring_segs):
                p1 = star_pos + r * (math.cos(theta_ring[j]) * u_vec + math.sin(theta_ring[j]) * w_vec)
                p2 = star_pos + r * (math.cos(theta_ring[j + 1]) * u_vec + math.sin(theta_ring[j + 1]) * w_vec)
                glow_pts.append(p1)
                glow_pts.append(p2)
                glow_cols.append(c)
                glow_cols.append(c)

        for i_ray in range(n_rays):
            p1 = star_pos + ray_dirs[i_ray] * (body_radius[0] * 0.96)
            p2 = star_pos + ray_dirs[i_ray] * ray_lens[i_ray]
            glow_pts.append(p1)
            glow_pts.append(p2)
            glow_cols.append(star_color * 0.75)
            glow_cols.append(star_color * 0.02)

        glow_field.from_numpy(np.array(glow_pts, dtype=np.float32))
        glow_col_field.from_numpy(np.array(glow_cols, dtype=np.float32))

        # Render elements
        if sky_col is not None:
            scene.particles(sky_pos, radius=1.0, per_vertex_radius=sky_rad, per_vertex_color=sky_col)
        scene.lines(orbit_field, width=1.0, per_vertex_color=orbit_col_field)
        scene.lines(trail_field, width=2.0, per_vertex_color=trail_col_field)
        scene.lines(glow_field, width=1.5, per_vertex_color=glow_col_field)
        body_pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
        body_pos.from_numpy(pos_np_frame)
        scene.particles(body_pos, radius=0.01, per_vertex_radius=radius_field, per_vertex_color=color_field)
        canvas.scene(scene)

    if args.bench:
        for _ in range(400):
            leapfrog_step(args.dt)
        draw_frame()
        window.save_image(args.bench)
        print("wrote", args.bench)
        return

    steps_per_frame = max(1, int(args.speed / 60.0 / args.dt))
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        for _ in range(steps_per_frame):
            leapfrog_step(args.dt)
        draw_frame()
        window.show()


if __name__ == "__main__":
    if not HAVE_TI:
        print("taichi is not installed. pip install taichi")
        sys.exit(1)
    main()
