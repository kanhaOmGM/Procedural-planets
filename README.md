# Procedural Star System & Symplectic N-Body Simulation

A project to express my curiosity for physics built during my preparation for JEE, most of the formulas and methods used for this project are the ones I have come across during my end years of highschool such as Stefan-boltzman law, Kepler's orbital laws, Newton laws of gravity and calculus in one variable.

This project simulates a unique procedural stellar system as per user-input using N-body gravitational simulator built in Python and [Taichi](https://www.taichi-lang.org/).

Given a single input parameter—the stellar mass $M**$ in solar masses ($M_\odot$)—the engine creates a complete stellar system using stellar homology, blackbody radiation laws, and protoplanetary disk condensation zones, and integrates all mutual gravitational interactions in real time.

---

## Table of Contents

1.  [Physical Units & Dimensional Scaling](#1-physical-units--dimensional-scaling)
2.  [Homologous relations in Astrophysics](#2-stellar-astrophysics--homology-relations)
3.  [Planetary Condensation & Non-Crossing Spacing](#3-planetary-condensation--non-crossing-spacing)
4.  [Symplectic Leapfrog N-Body Physics Engine](#4-symplectic-leapfrog-n-body-physics-engine)
5.  [Colorimetry & Celestial Background (`real_sky.py`)](#5-colorimetry--celestial-background-realskypy)
6.  [Numerical Verification Suite (`test_physics.py`)](#6-numerical-verification-suite-testphysicspy)
7.  [Approximations & Scientific Assumptions](#7-approximations--scientific-assumptions)
8.  [CLI Usage & Interactive Controls](#8-cli-usage--interactive-controls)
9.  [References/Acknowledgments](#9-referencesacknowledgments)

---

## 1. Physical Units & Dimensional Scaling

Standard SI units (meters, kilograms, seconds) introduce catastrophic numerical underflow and catastrophic cancellation when integrating gravitational interactions over astronomical timescales.

This engine adopts a **natural astronomical coordinate system**:

| Dimension    | Unit              | Symbol      | Definition                                                      |
| :----------- | :---------------- | :---------- | :-------------------------------------------------------------- |
| **Distance** | Astronomical Unit | $\text{AU}$ | Mean Earth–Sun distance ($1.495978707 \times 10^{11}\text{ m}$) |
| **Mass**     | Solar Mass        | $M_\odot$   | Mass of the Sun ($1.98847 \times 10^{30}\text{ kg}$)            |
| **Time**     | Earth Year        | $\text{yr}$ | Orbital period of Earth ($3.15576 \times 10^7\text{ s}$)        |

### Exact Derivation of Newton's Constant $G$

From Kepler's Third Law for a orbit of $a = 1.0\text{ AU}$ around a star of $M = 1.0\ M_\odot$ with period $T = 1.0\text{ yr}$:
$$T^2 = \frac{4\pi^2}{G M} a^3$$
Substituting $T = 1.0$, $a = 1.0$, and $M = 1.0$:
$$G = 4\pi^2 \approx 39.47841760435743\ \text{AU}^3 \cdot M_\odot^{-1} \cdot \text{yr}^{-2}$$

Because $G = 4\pi^2$ is exact in these units:

- A $1.0\text{ AU}$ circular orbit around a $1.0\ M_\odot$ star has an orbital speed of $v_c = \sqrt{G M / a} = 2\pi\text{ AU/yr}$.
- Keplerian orbital periods equal $T = a^{3/2}\text{ yr}$ down to floating-point machine precision ($< 10^{-14}$ error).

---

## 2. Homologous relations in Astrophysics

Given only the stellar mass $M$ (in $M_\odot$), the star's fundamental physical properties are derived using standard main-sequence homology scaling relations:

### 2.1 Mass-Luminosity Relation

For main-sequence stars with radiative envelopes ($0.43\ M_\odot \lesssim M \lesssim 2.0\ M_\odot$):
$$L = M^{3.5}\ [L_\odot]$$

### 2.2 Mass-Radius Relation

Assuming homologous hydrostatic equilibrium with Kramer's opacity:
$$R = M^{0.8}\ [R_\odot]$$

### 2.3 Stefan-Boltzmann Law & Surface Temperature

From the Stefan-Boltzmann law $L = 4\pi R^2 \sigma T^4$, effective temperature $T$ scales relative to the solar effective temperature ($T_\odot = 5772\text{ K}$):
$$T = T_\odot \left(\frac{L}{R^2}\right)^{0.25} = 5772 \times \left(\frac{M^{3.5}}{(M^{0.8})^2}\right)^{0.25} = 5772 \times M^{0.475}\ \text{K}$$

### 2.4 Protoplanetary Condensation Boundaries

- **Habitable Zone ($r_{\text{HZ}}$)**: The equilibrium radiative flux distance where liquid water can remain stable on an Earth-like planetary surface:
  $$r_{\text{HZ}} = \sqrt{\frac{L}{L_\odot}}\ \text{AU} = \sqrt{L}\ \text{AU}$$
- **Volatile Snow Line ($r_{\text{snow}}$)**: The threshold where the ambient disk temperature drops below $\approx 150\text{–}170\text{ K}$, allowing water vapor and volatiles to freeze into solid ice grains:
  $$r_{\text{snow}} = 2.7 \sqrt{L}\ \text{AU}$$

---

## 3. Planetary Condensation & Non-Crossing Spacing

### 3.1 Planetary Mass Classes

Planet composition and envelope mass are determined by radial distance relative to the volatile snow line:

1.  **Rocky / Terrestrial Planets ($a < 0.8\ r_{\text{snow}}$)**:
    - Condenses inside the frost line; rocky silicates and iron cores without volatile envelopes.
    - Mass: $m \in [0.5, 2.5]\ M_\oplus$ ($1\ M_\oplus = 3.003 \times 10^{-6}\ M_\odot$).
    - Visual styling: Rust-red, iron, and slate-grey mineral tones.
2.  **Super-Earths / Sub-Neptunes ($0.8\ r_{\text{snow}} \le a < 1.3\ r_{\text{snow}}$)**:
    - Transition regime with volatile-rich mantles and modest volatile envelopes.
    - Mass: $m \in [3.0, 15.0]\ M_\oplus$.
    - Visual styling: Deep teal, olive, and grey-blue palettes.
3.  **Gas / Ice Giants ($a \ge 1.3\ r_{\text{snow}}$)**:
    - Formed beyond the snow line where abundant water ice permits runaway gas accretion.
    - Mass: $m \in [20.0, 120.0]\ M_\oplus$ (~100× heavier envelope).
    - Visual styling: Icy pale blue, cyan, and methane-white.

### 3.2 3D Non-Crossing Orbital Spacing (Hill Stability)

Random independent draws of $(a, e)$ in unconstrained systems cause intersecting orbits where an outer planet's periapsis dips inside an inner planet's apoapsis.

To guarantee physical non-intersection and long-term dynamical stability, the generator enforces:
$$\text{periapsis}(k) \ge \text{apoapsis}(k-1) + \left(R_{\text{vis}, k-1} + R_{\text{vis}, k}\right) + \text{margin}$$

where:

- $\text{periapsis}(k) = a_k (1 - e_k)$
- $\text{apoapsis}(k-1) = a_{k-1} (1 + e_{k-1})$
- $R_{\text{vis}}$ is the 3D rendered visual radius of each body.
- $\text{margin} = \max(0.04, 0.08 \cdot a_{k-1})\text{ AU}$.

This ensures that even during direct planetary conjunctions (planets aligning at the exact same angular longitude), their 3D physical spheres maintain at least $0.04\text{ AU}$ of open separation.

### 3.3 State Vector Conversion (Vis-Viva Equation)

Each planet is assigned a random true anomaly $\nu \in [0, 2\pi)$. In the 2-body central approximation ($\mu = G M_*$):
$$r(\nu) = \frac{a(1 - e^2)}{1 + e \cos\nu}$$
$$h = \sqrt{\mu a (1 - e^2)}$$
$$\vec{r}_{\text{plane}} = \begin{bmatrix} r\cos\nu \\ r\sin\nu \\ 0 \end{bmatrix}, \quad \vec{v}_{\text{plane}} = \begin{bmatrix} -\frac{\mu}{h}\sin\nu \\ \frac{\mu}{h}(e + \cos\nu) \\ 0 \end{bmatrix}$$
The 2D plane vectors are then rotated into 3D space by the inclination angle $i \sim \mathcal{N}(0, 2^\circ)$ using the rotation matrix:
$$\mathbf{R}_x(i) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos i & -\sin i \\ 0 & \sin i & \cos i \end{bmatrix}, \quad \vec{r} = \mathbf{R}_x(i) \vec{r}_{\text{plane}}, \quad \vec{v} = \mathbf{R}_x(i) \vec{v}_{\text{plane}}$$

---

## 4. Symplectic Leapfrog N-Body Physics Engine

### 4.1 Direct All-Pairs Gravitational Acceleration

Every body $i$ experiences the full Newtonian gravitational pull of every other body $j$ ($j \ne i$). To prevent division-by-zero singularities during close encounters, a **Plummer softening parameter** $\epsilon = 10^{-3}\text{ AU}$ ($\epsilon^2 = 10^{-6}\text{ AU}^2$) is applied:

$$\vec{a}_i = \sum_{j \ne i} \frac{G m_j (\vec{r}_j - \vec{r}_i)}{\left(|\vec{r}_j - \vec{r}_i|^2 + \epsilon^2\right)^{3/2}} \cdot \gamma_{ij}$$

where $\gamma_{ij}$ is the mutual gravitational coupling factor:

- $\gamma_{0, j} = \gamma_{j, 0} = 1.0$ (Star–Planet interaction: authentic Newtonian gravity).
- $\gamma_{i, j} = \alpha_{\text{perturb}}$ for $i, j > 0$ (Planet–Planet interaction multiplier, default: $1.0$).

### 4.2 Kick-Drift-Kick Symplectic Leapfrog Scheme

Standard numerical schemes (such as Forward Euler or 4th-order Runge-Kutta) are **non-symplectic**: they do not conserve the Poincaré 2-form $\mathrm{d}\vec{p} \wedge \mathrm{d}\vec{q}$ and systematically dissipate or inject artificial energy, causing planetary orbits to collapse into the sun or fly apart over thousands of orbits.

This engine executes a **2nd-order Kick-Drift-Kick (Leapfrog)** integration on-device in Taichi:

1.  **Half-Kick**: Update velocities by half a time step using current forces:
    $$\vec{v}_i\left(t + \frac{\Delta t}{2}\right) = \vec{v}_i(t) + \frac{\Delta t}{2} \vec{a}_i(t)$$
2.  **Full-Drift**: Update positions across the full time step:
    $$\vec{r}_i(t + \Delta t) = \vec{r}_i(t) + \Delta t\ \vec{v}_i\left(t + \frac{\Delta t}{2}\right)$$
3.  **Force Recalculation**: Recompute mutual accelerations $\vec{a}_i(t + \Delta t)$ at the updated positions $\vec{r}_i(t + \Delta t)$.
4.  **Half-Kick**: Complete the velocity update:
    $$\vec{v}_i(t + \Delta t) = \vec{v}_i\left(t + \frac{\Delta t}{2}\right) + \frac{\Delta t}{2} \vec{a}_i(t + \Delta t)$$

**Symplectic Property**: Phase space volume is preserved exactly (Liouville's theorem). The total specific orbital energy exhibits bounded, periodic micro-oscillations without secular drift over arbitrarily long integration times.

---

## 5. Colorimetry & Celestial Background (`real_sky.py`)

### 5.1 Coordinate Frame Alignment (J2000 Obliquity)

The Bright Star Catalogue (BSC5) catalog coordinates are specified in equatorial Right Ascension ($\alpha$) and Declination ($\delta$). To match the ecliptic orbital plane of the planetary simulation, coordinates are converted to Cartesian vectors and rotated by the **obliquity of the ecliptic** ($\varepsilon = 23.4392911^\circ$):

$$\begin{bmatrix} x \\ y \\ z \end{bmatrix}_{\text{ecliptic}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\varepsilon & \sin\varepsilon \\ 0 & -\sin\varepsilon & \cos\varepsilon \end{bmatrix} \begin{bmatrix} \cos\delta \cos\alpha \\ \cos\delta \sin\alpha \\ \sin\delta \end{bmatrix}$$

### 5.2 Ballesteros' Relation ($B-V \rightarrow T_{\text{eff}}$)

The color index $B-V$ from astronomical photometry is converted to effective temperature via Ballesteros' analytic formula:
$$T = 4600 \left(\frac{1}{0.92(B-V) + 1.70} + \frac{1}{0.92(B-V) + 0.62}\right)\ \text{K}$$

### 5.3 Planck's Law for Star Color Synthesis

The emission spectrum of each star is evaluated using Planck's blackbody spectral radiance formula:
$$B(\lambda, T) = \frac{2 h c^2}{\lambda^5 \left[\exp\left(\frac{h c}{\lambda k_B T}\right) - 1\right]}$$
Sampled at primary visible wavelengths:

- $\lambda_{\text{Red}} = 600\text{ nm}$
- $\lambda_{\text{Green}} = 550\text{ nm}$
- $\lambda_{\text{Blue}} = 450\text{ nm}$

The resulting RGB coordinates are normalized such that $\max(R, G, B) = 1.0$.

### 5.4 Pogson's Ratio (Visual Magnitude to Flux)

By Pogson's relation, an apparent magnitude increase of $\Delta V = 5$ corresponds to exactly $100\times$ fainter flux:
$$\text{Relative Flux} = 10^{-0.4 \cdot V}$$
Rendered background star particle size and brightness scale directly with this relative flux.

---

## 6. Numerical Verification Suite (`test_physics.py`)

The integrity of the symplectic integrator is validated against two rigorous analytical tests with a massless test particle orbiting a $1.0\ M_\odot$ star at $a = 1.0\text{ AU}$ ($\Delta t = 10^{-4}\text{ yr}$):

```powershell
python test_physics.py
```

### Test 1: Specific Orbital Energy Conservation

The specific orbital energy is:
$$\mathcal{E} = \frac{1}{2} v^2 - \frac{\mu}{r}$$
Across 10,000 continuous leapfrog integration steps:

- **Pass Threshold**: $\left|\frac{\mathcal{E}_{\text{after}} - \mathcal{E}_{\text{before}}}{\mathcal{E}_{\text{before}}}\right| < 10^{-4}$
- **Measured Result**: $\mathbf{6.119 \times 10^{-15}}$ (Matches 64-bit IEEE 754 machine precision limit).

### Test 2: Kepler's Third Law Period Verification

Measures the time required for the test particle to complete one full counter-clockwise revolution (phase angle zero-crossing return to perihelion):

- **Expected Period**: $T = a^{3/2} = 1.000000\text{ yr}$
- **Measured Period**: $\mathbf{1.000003\text{ yr}}$
- **Error**: $\Delta < 3 \times 10^{-6}\text{ yr}$ (Residual from discrete time step interpolation).

---

## 7. Approximations

To ensure high interactive frame rates (60+ FPS) while maintaining physical accuracy, the following engineering approximations are made:

1.  **Constant Mass-Luminosity Exponent ($L \propto M^{3.5}$)**:
    The mass-luminosity relation changes exponent for very low-mass stars ($M < 0.43\ M_\odot$, where $L \propto M^{2.3}$) and very high-mass stars ($M > 2\ M_\odot$, where radiation pressure dominates and $L \propto M$). The engine uses $3.5$ as a standard representative value across typical solar-mass stars.
2.  **Simplified 3-Wavelength Colorimetry**:
    Color synthesis samples Planck's law at 3 discrete visible wavelengths (600nm, 550nm, 450nm) rather than numerically integrating full CIE 1931 XYZ standard observer color-matching functions.
3.  **Newtonian Gravity without Relativistic Precession**:
    Simulates purely Newtonian gravity. General relativistic corrections (such as Mercury's perihelion shift $\Delta \phi \approx \frac{6\pi G M}{c^2 a (1-e^2)}$) are neglected.
4.  **Visual Sizing vs. Physical Sizing**:
    If true physical planetary radii ($R_{\text{Earth}} \approx 4.26 \times 10^{-5}\text{ AU}$) were rendered to scale at orbital distances of $10\text{–}30\text{ AU}$, planets would be sub-pixel specks. Planetary and stellar radii are magnified using square-root scaling ($R_{\text{vis}} \propto R_{\text{physical}}^{0.5}$) with minimum readable floors to ensure clear visual inspection while preserving relative hierarchy.

---

## 8. CLI Usage & Controls

### Basic Commands

- **Run with default Sun-like star ($1.0\ M_\odot$)**:

```powershell
python starsystem.py --mass 1.0
```

- **Simulate a cool Red Dwarf ($0.4\ M_\odot$)**:

```powershell
python starsystem.py --mass 0.4
```

- **Simulate a luminous Blue-White Star ($2.2\ M_\odot$)**:

```powershell
python starsystem.py --mass 2.2
```

- **Reproduce a specific configuration**:

```powershell
python starsystem.py --mass 1.0 --seed 42 --planets 12
```

- **Amplify planet-planet mutual gravity (exaggerate multi-body perturbation)**:

```powershell
python starsystem.py --mass 0.4 --seed 42 --planets 14 --perturb 15.0
```

- **Offscreen benchmark snapshot (headless render)**:

```powershell
python starsystem.py --mass 1.0 --bench snapshot.png
```

### 3D Camera Controls

- **Orbit / Rotate View**: Hold **Right Mouse Button (RMB)** and drag.
- **Pan / Dolly**: `W` / `S` keys or mouse scroll wheel to zoom in on planets or pull back to system view.

## 9. References/Acknowledgments

This simulation relies on the following scientific resources, astrodynamic formulas, and computational frameworks:

- **[Taichi Programming Language](https://www.taichi-lang.org/):** Used for the high-performance, GPU-accelerated `@ti.kernel` physics integration and rendering.
- **[The HYG Database / Yale Bright Star Catalog](http://www.astronexus.com/hyg):** The source catalog for the celestial background coordinates (RA/Dec) and apparent magnitudes.
- **[Ballesteros' Formula (2012)](https://arxiv.org/pdf/1201.1809.pdf):** "New insights into black bodies" — Used to derive the analytical conversion from the astronomical $B-V$ color index to effective surface temperature (Kelvin).
- **[Gaffer on Games: Integration Basics](https://gafferongames.com/post/integration_basics/):** Foundational reference for implementing the Symplectic Leapfrog (Kick-Drift-Kick) integrator to ensure energy conservation over Forward Euler methods.
- **[Mitchell Charity: What Color Are the Stars?](http://www.vendian.org/mncharity/dir3/starcolor/):** Reference for accurate Planck blackbody RGB rendering and colorimetry.
- **Carroll & Ostlie, _An Introduction to Modern Astrophysics_:** Used as the mathematical basis for the Main Sequence homology scaling relations ($L = M^{3.5}$) and protoplanetary snow-line boundaries.
