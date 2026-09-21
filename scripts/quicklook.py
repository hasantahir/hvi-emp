#!/usr/bin/env python3
"""Render the frames without ParaView, and say why they were black.

    python scripts/quicklook.py --run-dir full_chain
    python scripts/quicklook.py --run-dir full_chain --stats-only
    python scripts/quicklook.py --pvd full_chain/reduced/plume/plume.pvd

Why this exists
---------------
A black frame out of pvbatch has three causes that look identical in the
log, and the first two had been chased repeatedly at the camera:

  1. the camera or clipping range is wrong
  2. the representation is wrong for the data type
  3. **the colour field is almost entirely floor**

The third is the one that was actually happening, and no camera change can
fix it. `log10_plasma_density` in the plume scene sits at its -12 floor over
99.99% of the volume; the first frame of every scene is *exactly* constant,
because nothing has happened yet. A volume renderer with the default opacity
ramp draws vacuum as transparent and the remaining 0.01% as a handful of
voxels. Correctly. The frame is black because the picture being asked for is
black.

So this renders from the arrays directly -- numpy and matplotlib, no VTK, no
ParaView, no camera. Whatever comes out is what is in the file. If quicklook
shows something and pvbatch does not, the problem is the scene; if quicklook
is empty too, the problem is upstream in the solver.

What it draws, and what that means
----------------------------------
Two panels per frame:

  * **max projection** -- the largest value along each ray. This finds
    faint, sparse structure that a volume render buries, which is exactly
    what is wanted here. It is *not* a volume rendering: it says where the
    peak is, and deliberately says nothing about how much material lies
    along the ray.
  * **mid-plane slice** -- one plane through the centre, honestly empty
    where the data is empty.

The colour scale is fixed across the whole series from the *signal*
percentiles, not from min/max, so the animation does not flicker and the map
is not spent on vacuum. Its limits are printed on every frame.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

#: Tried in this order, but only among fields that actually vary in the
#: frame. A constant field is a black picture whatever its name.
PREFER = ("log10_plasma_density", "temperature_eV", "ionisation",
          "plasma_density", "projectile_fraction", "shock_pressure",
          "E_normalised", "E_V_per_m", "material", "density")


def ok(m):
    print(f"  {GRN}[ ok ]{RST} {m}")


def warn(m):
    print(f"  {YLW}[warn]{RST} {m}")


def bad(m):
    print(f"  {RED}[FAIL]{RST} {m}")


def head(m):
    print(f"\n{m}\n" + "=" * 70)


def collections(root: Path):
    """Every `.pvd` under `root`, plus bare frame directories without one."""
    found = [Path(p) for p in sorted(glob.glob(str(root / "**" / "*.pvd"),
                                               recursive=True))]
    indexed = {p.parent for p in found}
    bare = []
    for pat in ("*.vti", "*.vtp"):
        for f in glob.glob(str(root / "**" / pat), recursive=True):
            d = Path(f).parent
            if d not in indexed and d not in bare:
                bare.append(d)
    return found, bare


def series(pvd: Path | None, directory: Path | None):
    """`[(time, path), ...]` from an index, or from the files themselves."""
    from hvi_emp.viz.vtkread import read_pvd

    if pvd is not None:
        return read_pvd(str(pvd))
    files = sorted([p for p in directory.iterdir()
                    if p.suffix in (".vti", ".vtp")])
    return [(float(i), str(p)) for i, p in enumerate(files)]


def scan(frames, field=None, verbose=True):
    """First pass: choose the field and fix the colour limits for the series.

    Both have to be decided over the whole series, not per frame. Per-frame
    limits make the animation pulse, and a field chosen from frame 0 is
    chosen from the one frame where nothing has happened.
    """
    import numpy as np

    from hvi_emp.viz.vtkread import field_stats, pick_field, read_vtk_xml

    votes, scanned, kind = {}, [], None
    for t, path in frames:
        try:
            d = read_vtk_xml(path)
        except Exception as exc:                            # noqa: BLE001
            scanned.append((t, path, {}, f"{type(exc).__name__}: {exc}"))
            continue
        kind = kind or d["type"]
        # Stats for EVERY field, not just this frame's winner. The series
        # picks one field; its limits must then come from that same field in
        # every frame. Taking each frame's own pick mixed 'material' (max 5)
        # into the colour limits of a log density that never exceeds -6, and
        # pushed the whole map off the top of the data.
        st_all = {n: field_stats(a) for n, a in d["arrays"].items()
                  if n != "points" and np.asarray(a).ndim <= 3}
        name, _ = pick_field(d["arrays"], prefer=PREFER)
        if name:
            votes[name] = votes.get(name, 0) + 1
        scanned.append((t, path, st_all, None))

    chosen = field or (max(votes, key=votes.get) if votes else None)

    rows = [(t, path, chosen, st_all.get(chosen), err)
            for t, path, st_all, err in scanned]

    lo_c, hi_c, n_const, n_live = [], [], 0, 0
    for t, path, name, st, err in rows:
        if st is None or st.get("constant"):
            n_const += 1
            continue
        n_live += 1
        lo_c.append(st["p50"])
        hi_c.append(st["max"])
    if lo_c:
        # p50 of the *signal* as the floor of the map: half the live voxels
        # are above it, so the map is spent on structure rather than vacuum.
        vmin, vmax = float(np.min(lo_c)), float(np.max(hi_c))
        if vmax <= vmin:
            vmax = vmin + 1e-12
    else:
        vmin = vmax = None

    if verbose:
        print(f"  frames {len(rows)}   type {kind}   field {chosen!r}")
        if votes and len(votes) > 1:
            warn(f"field varies by frame: {votes} -- using {chosen!r}")
        print(f"  colour limits {vmin!r} .. {vmax!r} "
              f"(signal p50 -> max over the series)")
        if n_const:
            warn(f"{n_const}/{len(rows)} frame(s) have NO variation in this "
                 f"field -- those render black correctly, because they are "
                 f"black")
    return chosen, vmin, vmax, rows, kind


def render(rows, chosen, vmin, vmax, out_dir: Path, title: str,
           dpi=110, cmap="inferno", caption=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from hvi_emp.viz.vtkread import read_vtk_xml

    out_dir.mkdir(parents=True, exist_ok=True)
    written, blank = [], 0

    for i, (t, path, name, st, err) in enumerate(rows):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
        fig.patch.set_facecolor("#111111")
        for ax in axes:
            ax.set_facecolor("#000000")
            ax.tick_params(colors="0.7", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("0.35")

        note = ""
        if err:
            note = err
        else:
            d = read_vtk_xml(path)
            if d["type"] == "ImageData" and chosen in d["arrays"]:
                a = np.asarray(d["arrays"][chosen], float)
                b = d["bounds"]
                # x-z plane, viewed along y. Project and slice the same way
                # so the two panels are directly comparable.
                ext = [b[0], b[1], b[4], b[5]]
                proj = np.nanmax(a, axis=1).T
                mid = a[:, a.shape[1] // 2, :].T
                for ax, img, lab in ((axes[0], proj, "max projection along y"),
                                     (axes[1], mid, "mid-plane slice (y = 0)")):
                    ax.imshow(img, origin="lower", extent=ext, cmap=cmap,
                              vmin=vmin, vmax=vmax, aspect="equal",
                              interpolation="nearest")
                    ax.set_title(lab, color="0.85", fontsize=9)
                    ax.set_xlabel("x [m]", color="0.7", fontsize=8)
                axes[0].set_ylabel("z [m]", color="0.7", fontsize=8)
                if st and not st.get("constant"):
                    note = (f"signal in {st['signal_fraction']*100:.3f}% of "
                            f"voxels   range {st['min']:.4g} .. "
                            f"{st['max']:.4g}")
                else:
                    note = "CONSTANT field -- nothing to draw in this frame"
                    blank += 1
            elif d["type"] == "PolyData":
                pts = d["arrays"].get("points")
                if pts is None or not len(pts):
                    note = "no points in this frame"
                    blank += 1
                else:
                    c = (np.asarray(d["arrays"][chosen], float)
                         if chosen in d["arrays"] else None)
                    for ax, (h, v, hl, vl) in (
                            (axes[0], (0, 2, "x [m]", "z [m]")),
                            (axes[1], (1, 2, "y [m]", "z [m]"))):
                        ax.scatter(pts[:, h], pts[:, v], s=1.0, c=c,
                                   cmap=cmap, vmin=vmin, vmax=vmax,
                                   linewidths=0)
                        ax.set_xlabel(hl, color="0.7", fontsize=8)
                        ax.set_ylabel(vl, color="0.7", fontsize=8)
                        ax.set_aspect("equal")
                    axes[0].set_title("x-z", color="0.85", fontsize=9)
                    axes[1].set_title("y-z", color="0.85", fontsize=9)
                    note = f"{len(pts)} points"
            else:
                note = f"{d['type']}: {chosen!r} not present"
                blank += 1

        fig.suptitle(f"{title}   |   {chosen}   |   t = {t:.4g} s   "
                     f"(frame {i + 1}/{len(rows)})",
                     color="0.95", fontsize=11)
        fig.text(0.5, 0.045, note, ha="center", color="0.6", fontsize=8)
        fig.text(0.5, 0.012,
                 "max projection is NOT a volume rendering: it reports the "
                 "peak along each ray, not how much lies along it."
                 + (f"  {caption}" if caption else ""),
                 ha="center", color="0.42", fontsize=7)
        fig.tight_layout(rect=[0, 0.075, 1, 0.94])
        p = out_dir / f"quicklook_{i:04d}.png"
        fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(p)

    return written, blank


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(_REPO / "full_chain"))
    ap.add_argument("--pvd", help="render one collection instead of scanning")
    ap.add_argument("--field", help="force a colour field")
    ap.add_argument("--out", help="where PNGs go (default: beside the data)")
    ap.add_argument("--stats-only", action="store_true",
                    help="report what is in the data, draw nothing")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args(argv)

    if args.pvd:
        targets = [(Path(args.pvd), None)]
    else:
        root = Path(args.run_dir)
        if not root.is_dir():
            bad(f"no such run directory: {root}")
            return 2
        pvds, bare = collections(root)
        targets = [(p, None) for p in pvds] + [(None, d) for d in bare]
        if not targets:
            bad(f"no .vti/.vtp/.pvd anywhere under {root}")
            print("  Nothing has been written yet. Run a stage first:")
            print("    python scripts/run_full_chain.py --md-scale talk --run")
            return 1

    total_written, total_blank, failures = 0, 0, 0
    for pvd, directory in targets:
        label = str(pvd or directory)
        head(label)
        frames = series(pvd, directory)
        if not frames:
            warn("no frames")
            continue
        if pvd is None:
            warn("no .pvd index -- times are frame numbers, not seconds "
                 "(python scripts/repair_pvd.py rebuilds the index)")

        chosen, vmin, vmax, rows, kind = scan(frames, field=args.field)
        if chosen is None:
            bad("no field in this collection varies at all -- every frame "
                "would be black, and correctly so")
            print("       This is a solver problem, not a rendering one.")
            failures += 1
            continue
        if args.stats_only:
            continue

        out_dir = (Path(args.out) if args.out
                   else (pvd.parent if pvd else directory) / "quicklook")
        written, blank = render(rows, chosen, vmin, vmax, out_dir,
                                title=Path(label).stem, dpi=args.dpi,
                                cmap=args.cmap)
        total_written += len(written)
        total_blank += blank
        ok(f"{len(written)} PNG -> {out_dir}")
        if blank:
            warn(f"{blank} of them are empty because the frame is empty")

    head("Summary")
    if args.stats_only:
        print("  stats only; nothing drawn")
        return 1 if failures else 0
    print(f"  {total_written} frame(s) drawn, {total_blank} empty, "
          f"{failures} collection(s) with no usable field")
    if total_written:
        print("\n  Assemble one with:")
        print("    ffmpeg -framerate 12 -i <dir>/quicklook_%04d.png \\")
        print("           -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4")
        print("\n  These bypass ParaView entirely. If they show structure and")
        print("  pvbatch does not, the scene is wrong, not the data.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
