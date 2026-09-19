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

WIDTH, HEIGHT = 1280, 720        # render resolution; picking projects into this


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

   
    VIS_R_MAX = 0.065

    for k in range(n_planets):
        e = float(abs(rng.normal(0.015, 0.02)))
        e = min(e, 0.10)

        # --- 1. place the orbit, reserving worst-case sphere clearance ---
        if k == 0:
            a_min = (star_vis_r + VIS_R_MAX + 0.06) / (1.0 - e)
        else:
            prev_apo = axes[-1] * (1.0 + ecc[-1])
            physical_buffer = (vis_radii[-1] + VIS_R_MAX) + max(0.04, 0.08 * axes[-1])
            a_min = (prev_apo + physical_buffer) / (1.0 - e)

        a_nom = a_lo * math.exp(log_ratio * (k + rng.uniform(0.1, 0.9)))
        a = max(a_min, a_nom)

        # --- 2. classify from the axis the planet actually ended up on ---
        if a < 0.8 * snow_line:
            planet_kind = "Rocky Planet"
            m = float(rng.uniform(0.5, 2.5)) * EARTH_MASS_MSUN
            base = rng.uniform(0.40, 0.85, 3) * np.array([1.00, 0.75, 0.55])
        elif a < 1.3 * snow_line:
            planet_kind = "Super-Earth / Sub-Neptune"
            m = float(rng.uniform(3.0, 15.0)) * EARTH_MASS_MSUN
            base = rng.uniform(0.45, 0.80, 3) * np.array([0.70, 0.85, 0.75])
        else:
            planet_kind = "Gas/Ice Giant"
            m = float(rng.uniform(20.0, 100.0)) * EARTH_MASS_MSUN
            icy = min(1.0, (a - snow_line) / max(snow_line, 1e-6))
            base = np.array([0.55 + 0.30 * icy, 0.65 + 0.25 * icy, 0.85 + 0.15 * icy])

        r_rsun = (m / EARTH_MASS_MSUN) ** (1.0 / 3.0) * 6371.0 / 695700.0
        vis_r = float(np.clip((r_rsun * 0.045) ** 0.5, 0.022, VIS_R_MAX))

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


def initial_state(system, seed=None):
   
    rng = np.random.default_rng(seed)
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


FOV_DEG = 45.0          # must match camera.fov(FOV_DEG); GGUI treats it as VERTICAL


def project_to_screen(cam_pos, cam_lookat, cam_up, points, radii, width, height,
                      fov_deg=FOV_DEG):

    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    radii = np.atleast_1d(np.asarray(radii, dtype=np.float64))

    fwd = np.asarray(cam_lookat, dtype=np.float64) - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(cam_up, dtype=np.float64))
    nr = np.linalg.norm(right)
    # Degenerate only if the camera looks straight along its own up vector.
    right = right / nr if nr > 1e-12 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)

    rel = points - cam_pos
    depth = rel @ fwd
    half_t = math.tan(math.radians(fov_deg) * 0.5)
    aspect = float(width) / float(height)

    safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    ndc_x = (rel @ right) / (safe * half_t * aspect)
    ndc_y = (rel @ up) / (safe * half_t)

    xy = np.stack([(ndc_x * 0.5 + 0.5) * width,
                   (ndc_y * 0.5 + 0.5) * height], axis=1)
    pixel_radius = radii / (safe * half_t) * (height * 0.5)
    return xy, depth, pixel_radius


def update_selection(current, hit, cursor_moved=True, clear_requested=False):
   
    if not cursor_moved and not clear_requested:
        return current
    if hit >= 0:
        return hit
    if clear_requested:
        return -1
    return current


def pick_body(cursor_px, xy, depth, pixel_radius, grab_px=10.0):
    
    if len(xy) == 0:
        return -1
    cursor_px = np.asarray(cursor_px, dtype=np.float64)
    d = np.linalg.norm(xy - cursor_px[None, :], axis=1)
    hit = (depth > 1e-9) & (d <= np.maximum(pixel_radius, grab_px))
    if not np.any(hit):
        return -1
    candidates = np.where(hit)[0]
    return int(candidates[np.argmin(depth[candidates])])



# a symplectic (kick-drift-kick leapfrog) integrator, works in AU / Msun / yr
# with G = 4*pi^2, and softens every distance with eps = 1e-3 AU.
# =============================================================================
_ti_ready = False


def ensure_taichi(arch=None):
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



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mass", type=float, default=1.0, help="stellar mass, Msun")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--planets", type=int, default=None)
    p.add_argument("--dt", type=float, default=2e-4, help="years per physics step")
    p.add_argument("--speed", type=float, default=0.6, help="sim years per real second")
    p.add_argument("--bench", type=str, default="", help="offscreen render, save PNG, exit")
    p.add_argument("--perturb", type=float, default=1.0, help="planet-planet gravity multiplier (1.0 = real physics, >1.0 to amplify mutual tugs)")
    p.add_argument("--fill", type=float, default=1.1, help="camera fill-light strength; "
                   "use 0.2 to see true Lambert day/night terminators, 0 for starlight only")
    p.add_argument("--gpu", action="store_true", help="use the GPU backend for rendering "
                   "(needs a working Vulkan/Metal/CUDA driver; default is CPU, which is "
                   "always safe but slower for the interactive window)")
    args = p.parse_args()

    if args.planets is not None and args.planets > MAX_BODIES - 1:
        print("--planets %d exceeds the %d-planet field capacity; raise MAX_BODIES "
              "in starsystem.py to go higher." % (args.planets, MAX_BODIES - 1))
        sys.exit(1)

    ensure_taichi(arch=ti.gpu if args.gpu else None)
    grav_scale[None] = float(args.perturb)

    # If no seed was given, draw one and PRINT it, so any run a judge likes can
    # be reproduced exactly with --seed.
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 31))

    system = generate_star_system(args.mass, seed=seed, n_planets=args.planets)
    pos_np, vel_np = initial_state(system, seed=seed)
    load_bodies(pos_np, vel_np, system["mass"])
    n = len(system["mass"])

    print("Star: %.2f Msun -> L=%.3f Lsun  R=%.2f Rsun  T=%.0f K" %
          (system["stellar_mass"], system["luminosity"], system["stellar_radius"],
           system["temperature"]))
    print("Habitable zone: %.2f AU   Snow line: %.2f AU   %d planets   (seed %d)" %
          (system["habitable_zone_au"], system["snow_line_au"], n - 1, seed))
    for i in range(1, n):
        a_i = system["semi_major_au"][i]
        in_hz = abs(a_i - system["habitable_zone_au"]) < 0.25 * system["habitable_zone_au"]
        print("  %-6s %-26s a=%8.3f AU  P=%9.3f yr  m=%7.2f Me%s" %
              (system["names"][i], system["kind"][i], a_i,
               a_i ** 1.5 / math.sqrt(system["stellar_mass"]),
               system["mass"][i] / EARTH_MASS_MSUN,
               "   <- in habitable zone" if in_hz else ""))

    csv_path = os.path.join(HERE, "bsc5_stars.csv")
    sky = real_sky.load_sky(csv_path)
    if sky is None:
        print("\nWARNING: bsc5_stars.csv not found next to starsystem.py — the real\n"
              "         background sky is DISABLED. The simulation still runs; the\n"
              "         backdrop will just be empty.\n")
    else:
        print("Sky: %d real stars from the Yale Bright Star Catalogue" % sky["n"])

    try:
        window = ti.ui.Window("Snowline — procedural star system", (WIDTH, HEIGHT),
                              show_window=not args.bench)
    except Exception as exc:
        # Taichi's GGUI needs a Vulkan swapchain even when the physics runs on
        # the CPU backend. On a headless box or a machine without Vulkan drivers
        # say so plainly, and point at the part that still works.
        print("\nCould not open a render window: %s" % exc)
        print("Taichi's GGUI requires Vulkan, even with the CPU physics backend.")
        print("The physics itself does not: run  python test_physics.py  to verify")
        print("the integrator headlessly. For the visuals, install Vulkan drivers")
        print("(Mesa/lavapipe on Linux, or vendor GPU drivers) and retry.")
        sys.exit(1)
    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()
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

    # Allocated ONCE. Declaring a Taichi field inside the draw loop costs ~44 ms
    # per call and its SNode is never reclaimed, so it both caps the frame rate
    # around 20 fps and leaks memory for as long as the window is open.
    body_pos = ti.Vector.field(3, dtype=ti.f32, shape=n)

    # Hover highlight: a camera-facing ring drawn around whatever the cursor is
    # over. Fixed size, allocated once, collapsed to a point when nothing is hit.
    N_RING = 72
    ring_field = ti.Vector.field(3, dtype=ti.f32, shape=N_RING * 2)
    ring_col_field = ti.Vector.field(3, dtype=ti.f32, shape=N_RING * 2)
    ring_theta = np.linspace(0.0, 2.0 * math.pi, N_RING + 1)
    ring_cos, ring_sin = np.cos(ring_theta), np.sin(ring_theta)

    hover_idx = -1
    last_cursor_px = None

    init_cam_pos = np.array([2.5, -2.0, 1.6]) * max(system["snow_line_au"], 1.0)
    camera.position(*init_cam_pos)
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)
    # Pinned explicitly: the hover projection in project_to_screen() assumes this
    # exact vertical field of view. Leaving it at the library default would make
    # picking silently disagree with the image if that default ever changed.
    camera.fov(FOV_DEG)

    def draw_frame():
        nonlocal trail_ptr
        pos_np_frame = pos.to_numpy()[:n].astype(np.float32)
        star_pos = pos_np_frame[0]
        canvas.set_background_color((0.0, 0.0, 0.01))
        scene.set_camera(camera)
        scene.ambient_light((0.10, 0.10, 0.12))
        scene.point_light(pos=tuple(star_pos), color=(1.0, 0.97, 0.9))
        # A camera-mounted fill light keeps the outer giants readable, but at full
        # strength it erases the day/night terminator the star light produces.
        # --fill 0.2 gives real Lambert phases; --fill 0 is starlight only.
        if args.fill > 0.0:
            f = args.fill
            scene.point_light(pos=tuple(camera.curr_position), color=(f, f, f))

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

        # Hover highlight ring, billboarded to face the camera. Collapsed onto a
        # single point when nothing is hovered, so the field is always valid.
        ring_pts = np.zeros((N_RING * 2, 3), dtype=np.float32)
        ring_cols = np.zeros((N_RING * 2, 3), dtype=np.float32)
        if hover_idx >= 0:
            c = pos_np_frame[hover_idx]
            rr = float(body_radius[hover_idx]) * 1.9
            a_pts = c + rr * (np.outer(ring_cos[:-1], u_vec) + np.outer(ring_sin[:-1], w_vec))
            b_pts = c + rr * (np.outer(ring_cos[1:], u_vec) + np.outer(ring_sin[1:], w_vec))
            ring_pts[0::2] = a_pts
            ring_pts[1::2] = b_pts
            ring_cols[:] = np.array([1.0, 0.95, 0.5], dtype=np.float32)
        ring_field.from_numpy(ring_pts)
        ring_col_field.from_numpy(ring_cols)

        # Render elements
        if sky_col is not None:
            scene.particles(sky_pos, radius=1.0, per_vertex_radius=sky_rad, per_vertex_color=sky_col)
        scene.lines(orbit_field, width=1.0, per_vertex_color=orbit_col_field)
        scene.lines(trail_field, width=2.0, per_vertex_color=trail_col_field)
        scene.lines(glow_field, width=1.5, per_vertex_color=glow_col_field)
        if hover_idx >= 0:
            scene.lines(ring_field, width=2.5, per_vertex_color=ring_col_field)
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

   
    WHY = {
        "Rocky Planet": [
            "Inside the snow line: too hot for ice.",
            "Only silicates and metals condense, so the",
            "core stays small and cannot hold H/He.",
        ],
        "Super-Earth / Sub-Neptune": [
            "Straddling the snow line: some volatiles",
            "survive, so the core grew past terrestrial",
            "mass but captured only a thin envelope.",
        ],
        "Gas/Ice Giant": [
            "Beyond the snow line: water freezes to grains.",
            "The core snowballed fast enough to reach",
            "runaway accretion and sweep up H/He.",
        ],
    }

    steps_per_frame = max(1, int(args.speed / 60.0 / args.dt))
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        for _ in range(steps_per_frame):
            leapfrog_step(args.dt)

        # --- pick, before drawing so the ring lands on the right body ---
        live_pos = pos.to_numpy()[:n]
        mx, my = window.get_cursor_pos()
        cursor_px = np.array([mx * WIDTH, my * HEIGHT])
        sxy, sdepth, srad = project_to_screen(
            np.array(camera.curr_position, dtype=np.float64),
            np.array(camera.curr_lookat, dtype=np.float64),
            np.array(camera.curr_up, dtype=np.float64),
            live_pos, body_radius.astype(np.float64), WIDTH, HEIGHT)

        # STICKY selection — see update_selection() for the rule and why a plain
        # hover does not work on moving bodies. Suppressed entirely while the
        # right button is held so orbiting the camera cannot change the
        # selection or clear it.
        if not window.is_pressed(ti.ui.RMB):
            moved = (last_cursor_px is None
                     or float(np.linalg.norm(cursor_px - last_cursor_px)) > 1.5)
            hover_idx = update_selection(
                hover_idx,
                pick_body(cursor_px, sxy, sdepth, srad),
                cursor_moved=moved,
                clear_requested=window.is_pressed(ti.ui.LMB))
        last_cursor_px = cursor_px

        draw_frame()

        # Always-on system readout, top left. Independent of the selection so
        # the star's numbers never vanish just because a planet is selected.
        with gui.sub_window("Snowline", 0.02, 0.02, 0.31, 0.15):
            gui.text("%.2f Msun   %.0f K   %.3f Lsun"
                     % (system["stellar_mass"], system["temperature"],
                        system["luminosity"]))
            gui.text("snow line %.2f AU    HZ %.2f AU"
                     % (system["snow_line_au"], system["habitable_zone_au"]))
            if hover_idx < 0:
                gui.text("point at a body to inspect it")
            else:
                gui.text("left-click empty space to deselect")

        # Selection panel, pinned to a fixed spot on the right. It deliberately
        # does NOT follow the cursor: the selection outlives the hover, so a
        # panel chasing the mouse would describe a body the cursor has long left.
        if hover_idx == 0:
            with gui.sub_window("Star  (selected)", 0.66, 0.02, 0.32, 0.30):
                gui.text("%.2f solar masses" % system["stellar_mass"])
                gui.text("")
                gui.text("L = M^3.5   = %.3f Lsun" % system["luminosity"])
                gui.text("R = M^0.8   = %.3f Rsun" % system["stellar_radius"])
                gui.text("T = Tsun (L/R^2)^0.25")
                gui.text("            = %.0f K" % system["temperature"])
                gui.text("")
                gui.text("habitable zone  %.3f AU" % system["habitable_zone_au"])
                gui.text("snow line       %.3f AU" % system["snow_line_au"])
                gui.text("")
                gui.text("every number above follows")
                gui.text("from the mass alone")
        elif hover_idx > 0:
            i = hover_idx
            a_i = float(system["semi_major_au"][i])
            snow = float(system["snow_line_au"])
            r_now = float(np.linalg.norm(live_pos[i] - live_pos[0]))
            period = a_i ** 1.5 / math.sqrt(system["stellar_mass"])
            in_hz = abs(a_i - system["habitable_zone_au"]) < 0.25 * system["habitable_zone_au"]
            with gui.sub_window("%s  (selected)" % system["names"][i],
                                0.66, 0.02, 0.32, 0.40):
                gui.text(system["kind"][i])
                gui.text("")
                for line in WHY[system["kind"][i]]:
                    gui.text(line)
                gui.text("")
                gui.text("semi-major        %.3f AU" % a_i)
                gui.text("snow line         %.3f AU" % snow)
                gui.text("distance          %.3f AU" % r_now)
                gui.text("mass              %.2f Earth masses"
                         % (system["mass"][i] / EARTH_MASS_MSUN))
                gui.text("eccentricity      %.3f" % system["eccentricity"][i])
                gui.text("inclination       %.2f deg" % math.degrees(system["inclination"][i]))
                gui.text("period            %.3f yr" % period)
                gui.text("                 = a^1.5 / sqrt(M)")
                if in_hz:
                    gui.text("")
                    gui.text("within the habitable zone")

        window.show()


if __name__ == "__main__":
    if not HAVE_TI:
        print("taichi is not installed. pip install taichi")
        sys.exit(1)
    main()