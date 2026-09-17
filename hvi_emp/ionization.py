"""Ionisation: multi-stage Saha equilibrium, energy partition, freeze-out.

Physics
-------
1.  **Saha ladder.**  For successive charge states j -> j+1 with ionisation
    potential chi_j and ground-state degeneracies g_j,

        n_{j+1} n_e / n_j = 2 (g_{j+1}/g_j) (2 pi m_e k T / h^2)^{3/2}
                            exp(-(chi_j - dchi_j)/kT)

    Charge conservation n_e = n_h * sum_j j f_j closes the system; it is
    solved by a bracketed root-find on n_e.

2.  **Continuum lowering.**  At the near-solid densities reached immediately
    behind the shock the isolated-atom potentials are strongly depressed.  We
    use the Debye-Hueckel form with an ion-sphere floor (Stewart-Pyatt-like
    interpolation),

        dchi_j = (j+1) e^2 / (4 pi eps0 * max(lambda_D, R_ion))

    This is the mechanism Fletcher (2021) calls "pressure ionisation" and is
    what pushes the plasma to full ionisation above ~15-20 km/s.

3.  **Energy partition.**  Given the residual (post-release) specific internal
    energy E_res, temperature follows from

        E_res * m_atom = E_cohesive + sum_j f_j sum_{k<j} chi_k
                         + (3/2) k T (1 + Zbar)

    i.e. sublimation + ionisation + translational energy of heavies and
    electrons.  Solved by brentq on T.

4.  **Freeze-out.**  As the plume expands, three-body recombination
    (alpha3 ~ 8.75e-39 Te[eV]^-4.5 m^6/s) and radiative recombination
    (alpha_rr ~ 2.7e-19 Z^2 Te[eV]^-0.75 m^3/s) compete with the expansion
    time.  When tau_rec > tau_exp the charge state freezes, which is why
    impact plasmas retain a high ionisation fraction far from the crater.

References
----------
Zel'dovich & Raizer (2002), Ch. III.
NRL Plasma Formulary (2019), pp. 54-55 (rate coefficients).
Stewart & Pyatt, ApJ 144, 1203 (1966) (continuum lowering).
Crawford, Procedia Eng. 103, 89 (2015) (Saha in a shock-physics code).
Fletcher, Close & Mathias, Phys. Plasmas 22, 093504 (2015) (ionisation
threshold 15-20 km/s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .constants import (E_CHARGE, EPS_0, EV, K_B, SAHA_PREFACTOR)
from .materials import Material
from .parallel import parallel_map


# ---------------------------------------------------------------------------
# Continuum lowering
# ---------------------------------------------------------------------------

def continuum_lowering(n_e: float, n_h: float, T: float, j: int) -> float:
    """Depression of the (j -> j+1) ionisation potential [J].

    Debye-Hueckel screening with an ion-sphere floor, so that the correction
    stays finite (and physical) at solid density where lambda_D << r_ion.
    """
    if n_e <= 0.0 or T <= 0.0:
        return 0.0
    lam_D = np.sqrt(EPS_0 * K_B * T / (n_e * E_CHARGE**2))
    R_ion = (3.0 / (4.0 * np.pi * max(n_h, 1.0))) ** (1.0 / 3.0)
    R = max(lam_D, R_ion)
    return (j + 1) * E_CHARGE**2 / (4.0 * np.pi * EPS_0 * R)


# ---------------------------------------------------------------------------
# Saha ladder
# ---------------------------------------------------------------------------

@dataclass
class IonisationState:
    """Equilibrium composition of a single-element plasma."""
    T: float                 # K
    n_h: float               # m^-3, heavy particles (atoms + ions)
    n_e: float               # m^-3, free electrons
    Zbar: float              # mean charge
    fractions: np.ndarray    # f_0 ... f_Jmax, sum = 1
    E_ionisation: float      # J per heavy particle stored in ionisation

    @property
    def T_eV(self) -> float:
        return self.T * K_B / EV

    @property
    def alpha(self) -> float:
        """Ionisation fraction 1 - f_0 (fraction of heavies that are ions)."""
        return float(1.0 - self.fractions[0])


def _saha_factors(mat: Material, T: float, n_e: float, n_h: float
                  ) -> np.ndarray:
    """S_j(T) = 2 (g_{j+1}/g_j) (2 pi m_e k T/h^2)^{3/2} exp(-chi_eff/kT)."""
    kT = K_B * T
    lam3 = SAHA_PREFACTOR * T**1.5          # (2 pi m_e k T / h^2)^{3/2}
    n_stage = len(mat.E_ion)
    S = np.zeros(n_stage)
    for j in range(n_stage):
        chi = mat.E_ion[j] - continuum_lowering(n_e, n_h, T, j)
        chi = max(chi, 0.0)                  # fully pressure-ionised
        g_ratio = mat.g_ion[j + 1] / mat.g_ion[j]
        expo = -chi / kT
        # guard against overflow; exp(700) is the double limit
        S[j] = 2.0 * g_ratio * lam3 * np.exp(np.clip(expo, -700.0, 700.0))
    return S


def _composition(mat: Material, T: float, n_e: float, n_h: float
                 ) -> tuple[np.ndarray, float]:
    """Charge-state fractions and Zbar for a *trial* n_e."""
    S = _saha_factors(mat, T, n_e, n_h)
    n_stage = len(S)
    ne_safe = max(n_e, 1e-30)
    # log-space ladder for numerical range
    log_r = np.log(np.maximum(S, 1e-300)) - np.log(ne_safe)
    log_f = np.concatenate(([0.0], np.cumsum(log_r)))
    log_f -= log_f.max()
    f = np.exp(np.clip(log_f, -700.0, 0.0))
    f /= f.sum()
    Zbar = float(np.dot(np.arange(n_stage + 1), f))
    return f, Zbar


def saha_solve(mat: Material, T: float, n_h: float) -> IonisationState:
    """Solve the coupled Saha + charge-neutrality system.

    Parameters
    ----------
    mat : Material
    T : float
        Temperature [K].
    n_h : float
        Heavy-particle number density [m^-3].
    """
    if T <= 0 or n_h <= 0:
        z = np.zeros(len(mat.E_ion) + 1)
        z[0] = 1.0
        return IonisationState(T, n_h, 0.0, 0.0, z, 0.0)

    Zmax = len(mat.E_ion)

    def residual(log_ne):
        ne = np.exp(log_ne)
        _, Zbar = _composition(mat, T, ne, n_h)
        return log_ne - np.log(max(Zbar * n_h, 1e-30))

    lo, hi = np.log(1e-12 * n_h), np.log(Zmax * n_h)
    try:
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0:
            log_ne = lo if abs(f_lo) < abs(f_hi) else hi
        else:
            log_ne = brentq(residual, lo, hi, xtol=1e-10, rtol=1e-12,
                            maxiter=300)
    except (ValueError, FloatingPointError):
        log_ne = lo

    n_e = float(np.exp(log_ne))
    f, Zbar = _composition(mat, T, n_e, n_h)
    n_e = Zbar * n_h

    cum_chi = np.concatenate(([0.0], np.cumsum(mat.E_ion)))
    E_ion_stored = float(np.dot(f, cum_chi))
    return IonisationState(T, n_h, n_e, Zbar, f, E_ion_stored)


# ---------------------------------------------------------------------------
# Energy partition: E_residual -> (T, Zbar)
# ---------------------------------------------------------------------------

def partition_energy(mat: Material, E_specific: float, rho: float,
                     T_lo: float = 300.0, T_hi: float = 5.0e6
                     ) -> IonisationState:
    """Find the equilibrium (T, Zbar) consistent with a given specific energy.

    Parameters
    ----------
    mat : Material
    E_specific : float
        Residual specific internal energy after release [J/kg].
    rho : float
        Mass density of the expanding vapour/plasma [kg/m^3].  Sets n_h.

    Notes
    -----
    Energy sinks, per heavy particle:
      * cohesive energy (solid -> free neutral atoms),
      * ionisation energy of the equilibrium composition,
      * translational energy 3/2 kT of heavies *and* electrons.
    Radiative losses and electronic excitation are neglected; both are small
    for the ~microsecond, optically-thin conditions of interest but see
    docs/THEORY.md Sec. 3.4 for the validity bound.
    """
    n_h = rho / mat.m_atom
    E_per_atom = E_specific * mat.m_atom

    def f(T):
        st = saha_solve(mat, T, n_h)
        used = (mat.E_cohesive + st.E_ionisation
                + 1.5 * K_B * T * (1.0 + st.Zbar))
        return used - E_per_atom

    if f(T_lo) > 0:
        # Not even enough energy to sublimate: no plasma.
        z = np.zeros(len(mat.E_ion) + 1)
        z[0] = 1.0
        return IonisationState(T_lo, n_h, 0.0, 0.0, z, 0.0)
    if f(T_hi) < 0:
        T_hi *= 20.0
        if f(T_hi) < 0:
            return saha_solve(mat, T_hi, n_h)

    T = float(brentq(f, T_lo, T_hi, xtol=1e-3, rtol=1e-10, maxiter=300))
    return saha_solve(mat, T, n_h)


# ---------------------------------------------------------------------------
# Recombination and freeze-out
# ---------------------------------------------------------------------------

def alpha_three_body(T_eV: float) -> float:
    """Three-body recombination rate coefficient [m^6/s].

    NRL Plasma Formulary: 8.75e-27 * Te[eV]^-4.5 cm^6/s.
    """
    return 8.75e-39 * max(T_eV, 1e-3) ** -4.5


def alpha_radiative(T_eV: float, Z: float = 1.0) -> float:
    """Radiative recombination rate coefficient [m^3/s].

    NRL Plasma Formulary: 2.7e-13 * Z^2 * Te[eV]^-0.75 cm^3/s.
    """
    return 2.7e-19 * Z**2 * max(T_eV, 1e-3) ** -0.75


def recombination_time(n_e: float, T_eV: float, Z: float = 1.0) -> float:
    """Combined recombination time [s]: (alpha3 n_e^2 + alpha_rr n_e)^-1."""
    if n_e <= 0:
        return np.inf
    rate = alpha_three_body(T_eV) * n_e**2 + alpha_radiative(T_eV, Z) * n_e
    return 1.0 / rate if rate > 0 else np.inf


def coulomb_log(n_e: float, T_eV: float, Z: float = 1.0) -> float:
    """Coulomb logarithm for electron-ion collisions (NRL, e-i, T_e > 10 Z^2 eV
    and T_e < 10 Z^2 eV branches), floored at 2 for strongly coupled plasma."""
    if n_e <= 0 or T_eV <= 0:
        return 2.0
    ne_cgs = n_e * 1e-6
    if T_eV < 10.0 * Z**2:
        ll = 23.0 - np.log(np.sqrt(ne_cgs) * Z * T_eV**-1.5)
    else:
        ll = 24.0 - np.log(np.sqrt(ne_cgs) / T_eV)
    return float(max(ll, 2.0))


def collision_frequency_ei(n_e: float, T_eV: float, Z: float = 1.0) -> float:
    """Electron-ion Coulomb collision frequency [1/s].

    NRL: nu_ei = 2.91e-6 n_e[cm^-3] lnLambda Te[eV]^-3/2  s^-1.
    Scales as n_e, i.e. *faster* than the plasma frequency (~ n_e^1/2), which
    is why an expanding impact plasma inevitably crosses from collisional to
    collisionless -- the trigger for the Close et al. EMP mechanism.
    """
    if n_e <= 0 or T_eV <= 0:
        return 0.0
    return (2.91e-12 * n_e * Z * coulomb_log(n_e, T_eV, Z)
            * max(T_eV, 1e-3) ** -1.5)


def plasma_frequency(n_e: float) -> float:
    """Electron plasma (angular) frequency [rad/s]."""
    from .constants import M_ELECTRON
    return np.sqrt(max(n_e, 0.0) * E_CHARGE**2 / (EPS_0 * M_ELECTRON))


def debye_length(n_e: float, T_eV: float) -> float:
    """Electron Debye length [m]."""
    if n_e <= 0:
        return np.inf
    return np.sqrt(EPS_0 * T_eV * EV / (n_e * E_CHARGE**2))


def coupling_parameter(n_e: float, T_eV: float) -> float:
    """Plasma coupling parameter Gamma = E_Coulomb / E_thermal.

    Gamma >> 1 is a strongly coupled plasma where the ideal Saha treatment and
    the Coulomb-collision formulary both break down (see THEORY.md Sec. 3.5).
    """
    if n_e <= 0 or T_eV <= 0:
        return np.inf
    a_ws = (3.0 / (4.0 * np.pi * n_e)) ** (1.0 / 3.0)
    return E_CHARGE**2 / (4.0 * np.pi * EPS_0 * a_ws * T_eV * EV)


__all__ = [
    "IonisationState", "continuum_lowering", "saha_solve", "partition_energy",
    "alpha_three_body", "alpha_radiative", "recombination_time",
    "coulomb_log", "collision_frequency_ei", "plasma_frequency",
    "debye_length", "coupling_parameter",
]


# ---------------------------------------------------------------------------
# Tabulated Saha solver (fast inner loop for the expansion ODE)
# ---------------------------------------------------------------------------

def _build_row(arg):
    """One density row of an ionisation table: Zbar and u at every T.

    Module-level and taking a single tuple because `ProcessPoolExecutor`
    dispatches by pickling the callable and its argument -- a closure over
    `self` would fail to pickle, and a method would drag the half-built
    table into the child.
    """
    mat, logn, logT = arg
    n = 10.0 ** logn
    nz = np.empty(len(logT))
    eu = np.empty(len(logT))
    for j, lt in enumerate(logT):
        T = 10.0 ** lt
        st = saha_solve(mat, T, n)
        nz[j] = st.Zbar
        eu[j] = 1.5 * K_B * T * (1.0 + st.Zbar) + st.E_ionisation
    return nz, eu


class IonisationTable:
    """Pre-tabulated Saha solution on a (log n_h, log T) grid.

    Solving the Saha ladder inside an ODE right-hand side costs a nested
    root-find per evaluation and dominates the runtime.  This class tabulates
    ``Zbar(n, T)`` and the internal energy per heavy particle

        u(n, T) = 3/2 k T (1 + Zbar) + E_ion(n, T)

    once, then serves bilinear interpolants.  ``u`` is monotone in T at fixed
    n, so the inverse ``T(n, u)`` is obtained by a bracketed search on the
    tabulated column -- roughly 3 orders of magnitude faster than the direct
    solve, at ~0.5% accuracy on the default grid.
    """

    def __init__(self, mat: Material, n_min: float = 1e14, n_max: float = 1e31,
                 T_min: float = 100.0, T_max: float = 5.0e6,
                 n_pts: int = 64, T_pts: int = 160,
                 workers: int | None = None):
        self.mat = mat
        self.logn = np.log10(np.geomspace(n_min, n_max, n_pts))
        self.logT = np.log10(np.geomspace(T_min, T_max, T_pts))

        # n_pts x T_pts fully independent nonlinear solves -- the framework's
        # dominant one-off cost, and embarrassingly parallel. Rows are the
        # unit of work: one row is ~160 solves, enough to amortise the cost
        # of shipping it to a worker, and there are enough rows to keep a
        # many-core machine busy. Serial below the threshold, because forking
        # a pool costs more than a small table.
        rows = [(mat, ln, self.logT) for ln in self.logn]
        if workers is None:
            workers = 1 if n_pts * T_pts < 4000 else None
        res = parallel_map(_build_row, rows, workers=workers, min_items=8)
        nz = np.array([r[0] for r in res])
        eu = np.array([r[1] for r in res])

        self.Zbar_grid = nz
        self.u_grid = eu
        self.log_u_grid = np.log(np.maximum(eu, 1e-300))
        # Precomputed bracket constants. Both axes are geometric in the
        # physical variable, hence uniform in log10, so locating a cell is
        # (x - x0) / dx rather than a binary search.
        self._n_floor = 10.0 ** self.logn[0]
        self._n_span = 1.0 / (self.logn[1] - self.logn[0])
        self._T_span = 1.0 / (self.logT[1] - self.logT[0])
        self._n_cell = n_pts - 2
        self._T_cell = T_pts - 2

    # -- interpolation helpers -------------------------------------------
    def _weights(self, logn: float):
        i = np.clip(np.searchsorted(self.logn, logn) - 1, 0,
                    len(self.logn) - 2)
        w = (logn - self.logn[i]) / (self.logn[i + 1] - self.logn[i])
        return i, float(np.clip(w, 0.0, 1.0))

    @staticmethod
    def _clamp(x, lo, hi):
        """`np.clip` without the overhead.

        `np.clip` goes through `fromnumeric._wrapfunc` and constructs a
        `numpy.finfo` on every call. At 445k calls per expansion that showed
        up as ~1.3 s of pure dispatch on arrays of 24 elements. Composing
        two ufuncs does the same arithmetic with none of the machinery.
        """
        return np.minimum(np.maximum(x, lo), hi)

    def _bracket(self, axis: np.ndarray, x, span: float, n_cell: int):
        """Index of the lower gridpoint and the fractional weight above it.

        Shape-preserving: `x` may be a scalar or any array. Values outside
        the grid clamp to the edge cell with weight 0 or 1 -- nearest-edge
        extrapolation, matching the `np.interp` behaviour this replaced.

        Both grids are geometric in the underlying variable, so they are
        *uniform* in the log coordinate. That makes the bracket index a
        division rather than a binary search, which removes the
        `searchsorted` call as well.
        """
        f = (x - axis[0]) * span                   # span = 1/spacing
        i = self._clamp(f.astype(np.intp) if np.ndim(f) else int(f),
                        0, n_cell)
        return i, self._clamp(f - i, 0.0, 1.0)

    def Zbar(self, n_h, T):
        """Interpolated mean charge state. Bilinear in (log n, log T).

        Accepts any broadcastable combination of scalars and arrays for
        ``n_h`` and ``T``, and returns a scalar only if both inputs are
        scalars.

        Performance
        -----------
        This is the hot path of the whole framework: profiling a single
        two-temperature expansion showed **89% of the runtime inside this
        method**, in a Python loop issuing 5.68 million scalar `np.interp`
        calls. Two things were wrong with it.

        First, it interpolated **every** density row of the grid (64 of
        them) when only the two rows bracketing the requested density are
        ever used -- 32x more work than needed, all of it discarded.

        Second, it did that with a list comprehension, so it paid NumPy's
        per-call dispatch overhead (~1.3 us) 64 times per invocation to do
        an amount of arithmetic that takes nanoseconds.

        The replacement gathers the four bracketing grid corners by fancy
        indexing and blends them. There is no Python-level loop and no
        `np.interp` call, so the cost is a handful of vector operations
        regardless of how many points are requested.

        Correctness note
        ----------------
        The previous vectorised branch collapsed the temperature to
        ``np.min(T)`` and applied that single value to every element. With
        the scalar ``T_e`` the 2T solver passes, that happened to be
        correct; with an array ``T`` it silently evaluated the entire
        result at the coldest temperature present. Temperature is now
        interpolated per element like density.
        """
        scalar = np.ndim(n_h) == 0 and np.ndim(T) == 0
        ln = np.log10(np.maximum(n_h, self._n_floor))
        lt = np.log10(np.maximum(T, 1.0))

        i, wn = self._bracket(self.logn, ln, self._n_span, self._n_cell)
        j, wt = self._bracket(self.logT, lt, self._T_span, self._T_cell)

        # Gathering the four corners broadcasts i against j automatically,
        # so no explicit broadcast of the inputs is needed.
        g = self.Zbar_grid
        z = ((1.0 - wn) * ((1.0 - wt) * g[i, j] + wt * g[i, j + 1])
             + wn * ((1.0 - wt) * g[i + 1, j] + wt * g[i + 1, j + 1]))
        return float(z) if scalar else z

    def u(self, n_h: float, T: float) -> float:
        """Interpolated internal energy per heavy particle [J]."""
        ln = np.log10(max(n_h, 10.0**self.logn[0]))
        lt = np.clip(np.log10(max(T, 1.0)), self.logT[0], self.logT[-1])
        i, w = self._weights(ln)
        u0 = np.interp(lt, self.logT, self.log_u_grid[i])
        u1 = np.interp(lt, self.logT, self.log_u_grid[i + 1])
        return float(np.exp((1 - w) * u0 + w * u1))

    def temperature(self, n_h: float, u_atom: float) -> float:
        """Invert u -> T at fixed density [K]."""
        if u_atom <= 0:
            return 10.0**self.logT[0]
        ln = np.log10(max(n_h, 10.0**self.logn[0]))
        i, w = self._weights(ln)
        col = (1 - w) * self.log_u_grid[i] + w * self.log_u_grid[i + 1]
        lu = np.log(u_atom)
        if lu <= col[0]:
            return 10.0**self.logT[0]
        if lu >= col[-1]:
            return 10.0**self.logT[-1]
        return float(10.0 ** np.interp(lu, col, self.logT))

    def state(self, n_h: float, u_atom: float) -> tuple[float, float]:
        """(T [K], Zbar) from density and internal energy per heavy particle."""
        T = self.temperature(n_h, u_atom)
        return T, self.Zbar(n_h, T)


_TABLE_CACHE: dict[str, "IonisationTable"] = {}

#: Composition spacing of the cached mixture tables. Tables are built lazily,
#: so a single scenario pays for the two that bracket its composition, and a
#: velocity sweep pays for at most 1/COMPOSITION_STEP + 1 of them.
COMPOSITION_STEP = 0.1


class MixtureTable:
    """Ionisation table for a two-component plume, continuous in composition.

    Building one `IonisationTable` costs ~1.5 s, so keying the cache on an
    exact composition would make a velocity sweep unusable -- every speed
    gives a slightly different projectile/target ratio and therefore a fresh
    table.

    The previous answer was to quantise the composition to 10% before building
    the mixture. That was wrong in a way that mattered: it put steps into the
    mean atomic mass, and through it into plume temperature and charge state,
    producing a **non-monotonic T_e as a function of impact speed** -- the
    plume got *colder* as the impact got faster, four times over Fletcher's
    velocity range. Physics should not be quantised to make a cache cheap.

    So: the mixture material is exact, and the *table* is interpolated. Tables
    live on a coarse composition grid and are built on demand; Zbar and the
    energy inversion are smooth in composition, so linear interpolation
    between neighbours 10% apart is accurate to well under a percent while the
    composition enters continuously.
    """

    def __init__(self, mat: Material, step: float = COMPOSITION_STEP, **kw):
        from .materials import get_material, mix_materials

        if mat.mix_of is None:
            raise ValueError("MixtureTable needs a material built by "
                             "mix_materials (mix_of is None)")
        name_a, name_b, w = mat.mix_of
        mat_a, mat_b = get_material(name_a), get_material(name_b)
        w = float(np.clip(w, 0.0, 1.0))

        lo = np.floor(w / step) * step
        hi = min(lo + step, 1.0)
        lo = min(lo, 1.0)
        self.w, self.w_lo, self.w_hi = w, lo, hi
        self.f = 0.0 if hi <= lo else (w - lo) / (hi - lo)

        def grid(wg):
            key = f"__mix__{name_a}|{name_b}|{wg:.6f}"
            if key not in _TABLE_CACHE:
                m = mix_materials(mat_a, wg, mat_b, 1.0 - wg)
                _TABLE_CACHE[key] = IonisationTable(m, **kw)
            return _TABLE_CACHE[key]

        self.lo_table = grid(lo)
        self.hi_table = grid(hi) if hi > lo else self.lo_table

    def _blend(self, a, b):
        return (1.0 - self.f) * a + self.f * b

    def Zbar(self, n_h, T):
        return self._blend(self.lo_table.Zbar(n_h, T),
                           self.hi_table.Zbar(n_h, T))

    def u(self, n_h: float, T: float) -> float:
        return self._blend(self.lo_table.u(n_h, T), self.hi_table.u(n_h, T))

    def temperature(self, n_h: float, u_atom: float) -> float:
        return self._blend(self.lo_table.temperature(n_h, u_atom),
                           self.hi_table.temperature(n_h, u_atom))

    def state(self, n_h: float, u_atom: float) -> tuple:
        T = self.temperature(n_h, u_atom)
        return T, self.Zbar(n_h, T)


def get_table(mat: Material, **kw):
    """Cached ionisation table for a material.

    Returns a `MixtureTable` for two-component plumes so that composition
    enters continuously, and a plain `IonisationTable` for pure materials.
    """
    if getattr(mat, "mix_of", None) is not None:
        return MixtureTable(mat, **kw)
    key = mat.name
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = IonisationTable(mat, **kw)
    return _TABLE_CACHE[key]


__all__ += ["IonisationTable", "MixtureTable", "get_table",
            "COMPOSITION_STEP"]
