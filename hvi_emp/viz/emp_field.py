"""Showing the EM radiation: what can honestly be drawn, and what cannot.

The EMP stage does not produce a field volume. It produces a **far-field
time series at one sensor** -- `E_t` at `r_sensor` = 0.3 m, `theta_deg` = 90
-- and its spectrum. There is no 3-D array of E(x,y,z) to volume-render, and
inventing one would be a picture of an assumption rather than a result.

What *is* recoverable, exactly, is the dipole far field. The model already
assumes a radiating dipole, and for that geometry the field everywhere
follows from the field at one point:

    E(r, theta, t) = E_t(t - (r - r0)/c) * (r0 / r) * sin(theta)

with `r0 = r_sensor`, because the sensor sits at theta = 90 where
sin(theta) = 1. The 1/r is the far-field falloff and sin(theta) is the
dipole pattern. This is a change of coordinates on a result the model has
already computed -- not a new physical claim.

Three things this module will not let you forget, because each of them
makes the picture mean less than it looks like it means:

1. **Scale.** For the reference case the spectral peak is ~77 MHz, so
   lambda ~ 3.9 m, against a plume of ~1 mm: the source is a POINT compared
   with the wave. That is exactly why a dipole approximation is legitimate,
   and it means the wavefront animation lives at METRE scale -- some three
   to four orders of magnitude larger than any plume image. It is a
   different picture of the same event, not a zoomed-out one. (The 916 MHz
   that appears elsewhere in this package is the resonance the WarpX bridge
   targets, not this scenario's spectral peak; do not conflate them.)

2. **Amplitude.** `docs/EMP_UNCERTAINTY.md` documents a 184x bracket on the
   radiated amplitude, dominated by one closure parameter. The shape of the
   pulse and its spectrum are far better constrained than its height. Colour
   scales here are therefore normalised and labelled as such by default.

3. **Near field.** Inside roughly one wavelength the far-field expression is
   simply wrong -- the static and induction terms dominate and fall off as
   1/r^3 and 1/r^2. `wavefront_series` refuses to write inside `r_min`
   rather than drawing a plausible-looking blob at the origin.
"""

from __future__ import annotations

import os

import numpy as np

from ..constants import C_LIGHT
from .vti import write_pvd, write_vti

__all__ = ["dipole_field_at", "wavefront_series", "pulse_figure",
           "EMP_VIZ_NOTICE"]

#: Burnt into every artefact this module writes.
EMP_VIZ_NOTICE = (
    "EM far field reconstructed from the dipole model, not simulated on a "
    "grid. Amplitude carries the 184x bracket in docs/EMP_UNCERTAINTY.md; "
    "shape and spectrum are better constrained than height."
)


def dipole_field_at(emp, r, theta, t):
    """E(r, theta, t) for the radiating dipole, in V/m.

    Exact re-expression of `emp.E_t`, which is the same field sampled at
    ``(emp.r_sensor, emp.theta_deg)``. Retardation is applied: a point at
    radius `r` sees what the sensor saw ``(r - r_sensor)/c`` earlier.

    Returns 0 where the retarded time falls outside the computed series,
    which is what makes the wavefront a shell rather than a filled ball.
    """
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)
    r0 = float(emp.r_sensor)
    sin0 = np.sin(np.radians(float(emp.theta_deg)))
    sin0 = sin0 if abs(sin0) > 1e-12 else 1.0

    t_ret = t - (r - r0) / C_LIGHT
    # np.interp clamps outside the range, which would smear the last sample
    # across the whole interior. Mask instead, so "no signal here yet" reads
    # as zero rather than as a DC level.
    inside = (t_ret >= emp.t[0]) & (t_ret <= emp.t[-1])
    e = np.zeros_like(r, dtype=float)
    if np.any(inside):
        e[inside] = np.interp(t_ret[inside], emp.t, emp.E_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = e * (r0 / np.maximum(r, 1e-30)) * (np.sin(theta) / sin0)
    return np.where(np.isfinite(e), e, 0.0)


def wavefront_series(scenario, directory: str, n_frames: int = 48,
                     shape=(96, 96, 96), r_max: float | None = None,
                     r_min: float | None = None, normalise: bool = True,
                     name: str = "emp", verbose: bool = True) -> dict:
    """Write the outgoing pulse as a `.vti` series. Returns the usual dict.

    Parameters
    ----------
    r_max : float or None
        Half-width of the box, metres. Defaults to four wavelengths at the
        spectral peak, which is enough to see the shell leave and not so
        much that it is one pixel thick.
    r_min : float or None
        Inside this radius nothing is written -- the far-field form does not
        hold there. Defaults to one wavelength.
    normalise : bool
        Write ``E_normalised`` in [-1, 1] alongside ``E_V_per_m``. On by
        default because the absolute height carries a 184x uncertainty and
        a colour bar in volts invites more confidence than the number
        supports.
    """
    emp = scenario.emp
    f_peak = _spectral_peak(emp)
    lam = C_LIGHT / f_peak if f_peak > 0 else 0.33
    r_max = float(r_max if r_max is not None else 4.0 * lam)
    r_min = float(r_min if r_min is not None else 1.0 * lam)

    nx, ny, nz = shape
    ax = np.linspace(-r_max, r_max, nx)
    ay = np.linspace(-r_max, r_max, ny)
    az = np.linspace(-r_max, r_max, nz)
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # theta from the dipole axis, which is the impact normal (+z).
    THETA = np.arccos(np.clip(Z / np.maximum(R, 1e-30), -1.0, 1.0))
    valid = R >= r_min

    # Follow the shell outwards: the first frame is the pulse still near the
    # source, the last has it crossing the far corner.
    t0 = float(emp.t[0]) + r_min / C_LIGHT
    t1 = float(emp.t[np.argmax(np.abs(emp.E_t))]) + (r_max * 1.7) / C_LIGHT
    times = np.linspace(t0, t1, n_frames)

    os.makedirs(directory, exist_ok=True)
    spacing = (ax[1] - ax[0], ay[1] - ay[0], az[1] - az[0])
    origin = (ax[0], ay[0], az[0])
    files, peak = [], 0.0

    for i, t in enumerate(times):
        e = dipole_field_at(emp, R, THETA, t)
        e = np.where(valid, e, 0.0)
        peak = max(peak, float(np.max(np.abs(e))))
        fields = {"E_V_per_m": e.astype(np.float32)}
        if normalise:
            scale = float(np.max(np.abs(e))) or 1.0
            fields["E_normalised"] = (e / scale).astype(np.float32)
        fn = f"{name}_{i:04d}.vti"
        write_vti(os.path.join(directory, fn), fields, spacing, origin)
        files.append(fn)

    comment = (
        f"{EMP_VIZ_NOTICE}\n"
        f"Dipole far field, axis = impact normal (+z). "
        f"Spectral peak {f_peak/1e6:.0f} MHz, lambda {lam*100:.1f} cm. "
        f"Box +/-{r_max*100:.1f} cm; nothing written inside r_min = "
        f"{r_min*100:.1f} cm because the far-field form does not hold there. "
        f"Sensor: r = {emp.r_sensor:.3g} m, theta = {emp.theta_deg:.0f} deg, "
        f"peak |E| = {emp.peak_field:.3e} V/m."
    )
    pvd = write_pvd(os.path.join(directory, f"{name}.pvd"), files,
                    times.tolist(), comment=comment)

    if verbose:
        print(f"wrote {len(files)} .vti + {name}.pvd to {directory}")
        print(f"  spectral peak {f_peak/1e6:.0f} MHz, lambda {lam*100:.1f} cm")
        print(f"  box +/-{r_max*100:.1f} cm, blanked inside {r_min*100:.1f} cm")
        print(f"  peak |E| in box {peak:.3e} V/m")
        print(f"  NOTE: amplitude carries the 184x bracket "
              f"(docs/EMP_UNCERTAINTY.md)")
    return {"pvd": pvd, "files": files, "times": times.tolist(),
            "r_max": r_max, "r_min": r_min, "lambda": lam,
            "f_peak": f_peak, "peak_field": peak}


def _spectral_peak(emp) -> float:
    """Frequency of the spectral maximum, Hz (0 if it cannot be found)."""
    mag = np.abs(np.asarray(emp.E_spec))
    f = np.asarray(emp.f, float)
    ok = f > 0
    if not np.any(ok) or not np.any(mag[ok]):
        return 0.0
    return float(f[ok][int(np.argmax(mag[ok]))])


def pulse_figure(scenario, path: str, dpi: int = 150) -> str | None:
    """The pulse and its spectrum: the honest primary view of the EMP.

    A wavefront animation is the memorable picture, but this is the one that
    can be read quantitatively. Returns the path, or None when matplotlib is
    absent (the package imports fine without it).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    emp = scenario.emp
    f_peak = _spectral_peak(emp)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    t_ns = np.asarray(emp.t) * 1e9
    ax1.plot(t_ns, emp.E_t, lw=1.0)
    ax1.set_xlabel("time [ns]")
    ax1.set_ylabel("E [V/m]")
    ax1.set_title(f"Radiated pulse at r = {emp.r_sensor:.3g} m, "
                  f"theta = {emp.theta_deg:.0f} deg")
    # The pulse is short compared with the series; show where it lives.
    i_pk = int(np.argmax(np.abs(emp.E_t)))
    span = max(t_ns[-1] * 0.02, 5.0)
    ax1.set_xlim(max(0, t_ns[i_pk] - span), t_ns[i_pk] + span)
    ax1.grid(alpha=0.3)

    mag = np.abs(np.asarray(emp.E_spec))
    f_mhz = np.asarray(emp.f) / 1e6
    ok = f_mhz > 0
    ax2.loglog(f_mhz[ok], mag[ok], lw=1.0)
    if f_peak > 0:
        ax2.axvline(f_peak / 1e6, color="tab:red", ls="--", lw=1.0,
                    label=f"peak {f_peak/1e6:.0f} MHz")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("frequency [MHz]")
    ax2.set_ylabel("|E(f)| [V/m/Hz]")
    ax2.set_title("Spectrum")
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"peak |E| = {emp.peak_field:.3e} V/m   "
        f"radiated energy = {emp.energy_radiated:.3e} J",
        fontsize=10)
    fig.text(0.5, 0.005,
             "Amplitude carries the 184x bracket in docs/EMP_UNCERTAINTY.md; "
             "pulse shape and spectrum are better constrained.",
             ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
