"""SYNTHETIC M2C output, in M2C's exact on-disk format. Not physics.

What this is for
----------------
Two jobs, both about plumbing:

1. Testing `m2c_output.read_m2c_vtr` against the real byte layout. The
   writer below follows PETSc's `DMDAVTKWriteAll_VTR` (grvtk.c) line for
   line -- one `<Piece>` per MPI rank, UInt64 byte-count prefixes, x-fastest
   Float64 -- plus M2C's own quirk of writing `electron_density` twice.
2. Driving the hydrocode -> PIC waterfall end to end on a machine without
   M2C, so the handoff and the PIC initialisation can be exercised before a
   real run exists.

What it is NOT
--------------
The plume is a prescribed self-similar Gaussian cloud with an assumed
temperature history and an assumed charge state. It is chosen to have
known integrals (mass, charge, momentum), so that the extractor can be
checked against them -- not to resemble what M2C will compute. Every file it
writes says SYNTHETIC in its name and in the .pvd, and `extract` refuses to
treat it as a real run unless asked.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from ..constants import AMU, K_B
from .m2c_stage1 import si_to_m2c

__all__ = ["write_petsc_vtr", "write_m2c_pvd", "synthetic_plume_run",
           "SyntheticPlume"]


def _block(arr: np.ndarray) -> bytes:
    """PetscViewerVTKFWrite: an 8-byte BYTE count, then the raw values."""
    a = np.ascontiguousarray(arr, dtype="<f8")
    return struct.pack("<Q", a.nbytes) + a.tobytes()


def write_petsc_vtr(path: str, x, y, z, fields: dict, ranks=(1, 1, 1),
                    duplicate: tuple = ("electron_density",)) -> str:
    """Write a `.vtr` exactly as PETSc does for a DMDA split over ranks.

    `fields` values are (nx, ny, nz) or (nx, ny, nz, nc) arrays in M2C
    units. `ranks` is the process grid; each rank's block becomes a Piece.
    `duplicate` repeats arrays by name, as M2C's Output.cpp does.
    """
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    nx, ny, nz = x.size, y.size, z.size

    def split(n, p):
        b = np.linspace(0, n, p + 1).round().astype(int)
        return [(b[i], b[i + 1] - b[i]) for i in range(p) if b[i + 1] > b[i]]

    pieces = [(xs, xm, ys, ym, zs, zm)
              for zs, zm in split(nz, ranks[2])
              for ys, ym in split(ny, ranks[1])
              for xs, xm in split(nx, ranks[0])]
    order = list(fields.items()) + [(n, fields[n]) for n in duplicate
                                    if n in fields]

    head = ['<?xml version="1.0"?>\n',
            '<VTKFile type="RectilinearGrid" version="0.1" '
            'byte_order="LittleEndian" header_type="UInt64">\n',
            f'  <RectilinearGrid WholeExtent="0 {nx - 1} 0 {ny - 1} 0 '
            f'{nz - 1}">\n']
    blobs = []
    off = 0
    for xs, xm, ys, ym, zs, zm in pieces:
        nn = xm * ym * zm
        head.append(f'    <Piece Extent="{xs} {xs + xm - 1} {ys} '
                    f'{ys + ym - 1} {zs} {zs + zm - 1}">\n')
        head.append('      <Coordinates>\n')
        for lab, arr in (("Xcoord", x[xs:xs + xm]), ("Ycoord", y[ys:ys + ym]),
                         ("Zcoord", z[zs:zs + zm])):
            head.append(f'        <DataArray type="Float64" Name="{lab}"  '
                        f'format="appended"  offset="{off}" />\n')
            blobs.append(_block(arr))
            off += arr.size * 8 + 8
        head.append('      </Coordinates>\n')
        head.append('      <PointData Scalars="ScalarPointData">\n')
        for name, a in order:
            a = np.asarray(a, float)
            nc = 1 if a.ndim == 3 else a.shape[3]
            sub = a[xs:xs + xm, ys:ys + ym, zs:zs + zm]
            # (i, j, k[, c]) -> memory with x fastest
            mem = (sub.transpose(2, 1, 0) if nc == 1
                   else sub.transpose(2, 1, 0, 3))
            head.append(f'        <DataArray type="Float64" Name="{name}" '
                        f'NumberOfComponents="{nc}" format="appended" '
                        f'offset="{off}" />\n')
            blobs.append(_block(mem.ravel()))
            off += nn * nc * 8 + 8
        head.append('      </PointData>\n')
        head.append('    </Piece>\n')
    head.append('  </RectilinearGrid>\n')
    head.append('  <AppendedData encoding="raw">\n')
    head.append('_')

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write("".join(head).encode())
        for b in blobs:
            fh.write(b)
        fh.write(b"\n </AppendedData>\n</VTKFile>\n")
    return path


def write_m2c_pvd(results_dir: str, entries, name: str = "solution.pvd",
                  closed: bool = True) -> str:
    """The collection file, in the form Output.cpp writes it."""
    lines = ['<?xml version="1.0"?>\n',
             '<VTKFile type="Collection" version="0.1"\n',
             'byte_order="LittleEndian">\n', '  <Collection>\n']
    for t, f in entries:
        lines.append(f'  <DataSet timestep="{t:e}" file="{f}"/>\n')
    if closed:
        lines += ['  </Collection>\n', '</VTKFile>\n']
    p = os.path.join(results_dir, name)
    with open(p, "w") as fh:
        fh.writelines(lines)
    return p


class SyntheticPlume:
    """A Gaussian vapour cloud leaving an Al surface at x = 0 toward -x.

    Known by construction, so the extractor can be checked against it:

        rho(x, r, t) = M / ((2 pi)^1.5 s_x s_r^2) exp(-(x - x_c)^2/2 s_x^2
                                                    - r^2 / 2 s_r^2)
        s_x = s_r = s0 + v_th t,  x_c = -(x0 + v_c t)
        v = (x - x_c)/s * v_th (self-similar) plus the drift -v_c along x

    Temperature follows `T_law`: "isothermal" (the assumption behind Close
    et al.'s mechanism, under which the collision frequency falls faster
    than the plasma frequency, as in Fletcher's Fig. 2) or "adiabatic"
    (monatomic, T ~ s^-2, under which nu_ei ~ n T^-3/2 stays CONSTANT while
    omega_pe falls -- the plume never becomes collisionless). The mean
    charge follows a prescribed decline. Only the half x < 0 is kept as
    plume; the target fills x >= 0. The x < 0 half carries mass M/2 only if
    x_c is far from the surface, so `mass_in_box()` integrates the exact
    profile on the mesh rather than quoting M.
    """

    def __init__(self, M=1.0e-8, v_c=5.0e3, v_th=4.0e3, s0=0.2e-3,
                 x0=0.5e-3, T0_K=4.0e4, Z0=1.2, A=26.98, t_Z=2.0e-6,
                 T_law="isothermal"):
        if T_law not in ("isothermal", "adiabatic"):
            raise ValueError("T_law must be 'isothermal' or 'adiabatic'")
        self.M, self.v_c, self.v_th, self.s0, self.x0 = M, v_c, v_th, s0, x0
        self.T0, self.Z0, self.A, self.t_Z = T0_K, Z0, A, t_Z
        self.T_law = T_law

    def temperature(self, t):
        if self.T_law == "isothermal":
            return self.T0
        return self.T0 * (self.s0 / self.sigma(t)) ** 2

    def sigma(self, t):
        return self.s0 + self.v_th * t

    def centre(self, t):
        return -(self.x0 + self.v_c * t)

    def state(self, X, R, t):
        """rho [kg/m^3], vx, vr [m/s], T [K], Zbar on the (X, R) grid."""
        s = self.sigma(t)
        xc = self.centre(t)
        g = np.exp(-((X - xc) ** 2 + R ** 2) / (2 * s * s))
        rho = self.M / ((2 * np.pi) ** 1.5 * s ** 3) * g
        vx = -self.v_c + (X - xc) / s * self.v_th
        vr = R / s * self.v_th
        T = self.temperature(t)
        Z = self.Z0 * np.exp(-t / self.t_Z)
        return rho, vx, vr, np.full_like(rho, T), np.full_like(rho, Z)


def synthetic_plume_run(results_dir: str, times=None, nx=160, nr=80,
                        L=6.0e-3, ranks=(4, 2, 1), plume=None) -> dict:
    """Write a SYNTHETIC M2C results/ directory: .pvd plus .vtr frames.

    Cylindrical layout as the decks use: X is the axis on [-L, L], Y the
    radius on (0, L], one cell in Z. Material IDs as the decks assign them:
    0 ambient, 1 target, 2 projectile (the plume here is labelled 1, target
    vapour). Returns the plume object and the per-frame exact integrals.
    """
    plume = plume or SyntheticPlume()
    if times is None:
        times = np.array([0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]) * 1e-6
    os.makedirs(results_dir, exist_ok=True)

    dx = 2 * L / nx
    dr = L / nr
    x = -L + dx * (np.arange(nx) + 0.5)
    r = dr * (np.arange(nr) + 0.5)
    z = np.array([0.0])
    X, R = np.meshgrid(x, r, indexing="ij")

    rho_al = 2700.0
    rho_amb = 1.0e-9
    entries, exact = [], []
    m_h = plume.A * AMU
    for k, t in enumerate(times):
        rho, vx, vr, T, Z = plume.state(X, R, t)
        target = X >= 0.0
        rho = np.where(target, rho_al, rho)
        is_plume = (~target) & (rho > 1e3 * rho_amb)
        mat = np.where(target, 1.0, np.where(is_plume, 1.0, 0.0))
        rho = np.where(mat == 0.0, rho_amb, rho)
        vx = np.where(is_plume, vx, 0.0)
        vr = np.where(is_plume, vr, 0.0)
        T = np.where(is_plume, T, 300.0)
        Z = np.where(is_plume, Z, 0.0)
        n_h = rho / m_h
        n_e = np.where(is_plume, Z * n_h, 0.0)

        # cylindrical cell volumes, as the reader rebuilds them
        ring = np.pi * ((r + dr / 2) ** 2 - (r - dr / 2) ** 2)
        vol = dx * ring[None, :]
        exact.append({"time": float(t),
                      "N_e": float((n_e * vol).sum()),
                      "mass_plume": float((np.where(is_plume, rho, 0) * vol)
                                          .sum()),
                      "T_K": float(plume.temperature(t)),
                      "Zbar": float(plume.Z0 * np.exp(-t / plume.t_Z)),
                      "v_axial_drift": -plume.v_c})

        f3 = lambda a: a[:, :, None]                    # noqa: E731
        vel = np.stack([vx, vr, np.zeros_like(vx)], axis=-1)[:, :, None, :]
        fields = {
            "density": f3(si_to_m2c(rho, "density")),
            "velocity": si_to_m2c(vel, "velocity"),
            "pressure": f3(n_h * (1 + Z) * K_B * T),            # Pa = M2C
            "materialid": f3(mat),
            "temperature": f3(T),
            "mean_charge_number": f3(Z),
            "heavy_particles_density": f3(si_to_m2c(n_h, "number_density")),
            "electron_density": f3(si_to_m2c(n_e, "number_density")),
        }
        fname = f"solution_SYNTHETIC_{k:04d}.vtr"
        write_petsc_vtr(os.path.join(results_dir, fname),
                        si_to_m2c(x, "length"), si_to_m2c(r, "length"),
                        si_to_m2c(z, "length"),
                        fields, ranks=ranks)
        entries.append((float(t), fname))
    write_m2c_pvd(results_dir, entries)
    with open(os.path.join(results_dir, "SYNTHETIC.txt"), "w") as fh:
        fh.write("This directory was written by hvi_emp.solvers.m2c_synthetic."
                 "\nIt is a prescribed Gaussian cloud in M2C's file format, "
                 "for testing the\nreader and the handoff. It is not an M2C "
                 "result and not physics.\n")
    return {"plume": plume, "exact": exact, "x": x, "r": r,
            "times": np.asarray(times)}
