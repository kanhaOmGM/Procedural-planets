import math
import os

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23
OBLIQUITY = math.radians(23.4392911)     # J2000 obliquity: equatorial -> ecliptic


LAM_RGB = np.array([600e-9, 550e-9, 450e-9])


def bv_to_temp(bv):
    """Ballesteros' relation: effective temperature straight from the B-V
    colour index the catalogue already lists. B-V = 0.65 gives ~5778 K (the
    Sun); more negative B-V (bluer) gives hotter, more positive (redder)
    gives cooler."""
    bv = np.asarray(bv, dtype=np.float64)
    return 4600.0 * (1.0 / (0.92 * bv + 1.70) + 1.0 / (0.92 * bv + 0.62))


def planck(lam, T):
    """Spectral radiance of a blackbody at wavelength `lam` (m), temperature
    `T` (K) — Planck's law."""
    x = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T)
    return (2.0 * H_PLANCK * C_LIGHT ** 2 / lam ** 5) / (np.exp(x) - 1.0)


def blackbody_rgb(T):
    """Sample the Planck curve at red/green/blue wavelengths and normalise so
    the brightest channel is 1.0 — an approximate but physically-grounded
    star colour from temperature alone."""
    c = np.array([planck(lam, T) for lam in LAM_RGB])
    return c / c.max()


def radec_to_cartesian(ra_deg, dec_deg):
    """Right Ascension / Declination -> unit vector, then rotated from the
    equatorial frame into the ecliptic frame (the same frame the orbital
    mechanics run in) by the 23.44-degree obliquity of the ecliptic."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    ce, se = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    return np.stack([x, y * ce + z * se, -y * se + z * ce], axis=1)


def load_sky(csv_path):
    """Parse the Bright Star Catalogue CSV and return everything the renderer
    needs: 3D direction, per-star colour, and a brightness/size term that
    scales inversely with apparent magnitude (fainter = smaller & dimmer)."""
    if not os.path.exists(csv_path):
        return None
    raw = np.loadtxt(csv_path, delimiter=",", comments="#")
    ra_deg, dec_deg, vmag, bv = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

    direction = radec_to_cartesian(ra_deg, dec_deg)
    temp_k = bv_to_temp(bv)
    rgb = np.stack([planck(lam, temp_k) for lam in LAM_RGB], axis=1)
    rgb = rgb / rgb.max(axis=1, keepdims=True)

    # STEP 1: brightness/size inversely proportional to apparent magnitude.
    # Pogson's ratio: five magnitudes fainter is exactly 1/100 the flux.
    relative_flux = 10.0 ** (-0.4 * vmag)
    relative_flux = relative_flux / relative_flux.max()

    return {
        "n": len(vmag), "ra_deg": ra_deg, "dec_deg": dec_deg, "vmag": vmag,
        "bv": bv, "temp_k": temp_k, "direction": direction, "rgb": rgb,
        "relative_flux": relative_flux,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sky = load_sky(os.path.join(here, "bsc5_stars.csv"))
    if sky is None:
        print("bsc5_stars.csv not found next to real_sky.py")
    else:
        print("%d stars loaded" % sky["n"])
        print("Sun-like check: B-V=0.65 -> %.0f K (expect ~5778 K)" % bv_to_temp(0.65))
        brightest = int(np.argmax(sky["relative_flux"]))
        print("Brightest star: vmag=%.2f  T=%.0f K  rgb=%s" %
              (sky["vmag"][brightest], sky["temp_k"][brightest],
               np.round(sky["rgb"][brightest], 2)))
