#!/usr/bin/env python3
"""Is there anything to render? Answer before spending a night on pvbatch.

    python scripts/check_render_data.py full_chain

Black frames have two completely different causes, and the render log looks
identical for both:

  * the scene is wrong  -- camera, clipping, representation
  * the DATA is empty   -- the .pvd lists no timesteps, or the .vti/.vtp it
                           points at contain no cells or points

Only the first is fixable in the composite script. This reads the files
directly -- no ParaView, no VTK, just XML headers -- so it runs anywhere in
under a second and tells you which of the two you are looking at.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

GRN, RED, YLW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _n_cells(extent: str) -> int:
    """Cells implied by a VTK WholeExtent string, 'x0 x1 y0 y1 z0 z1'."""
    try:
        v = [int(t) for t in extent.split()]
    except ValueError:
        return 0
    if len(v) != 6:
        return 0
    n = 1
    for lo, hi in ((v[0], v[1]), (v[2], v[3]), (v[4], v[5])):
        n *= max(hi - lo, 1)
    return n


def inspect_piece(path: Path) -> dict:
    """Points/cells in one .vti or .vtp, from its XML header only.

    Reads a bounded prefix: these files carry megabytes of appended binary
    after the header, and parsing that to count points would defeat the
    purpose of a fast check.
    """
    out = {"path": path, "exists": path.is_file(), "points": 0, "cells": 0,
           "kind": "?", "bytes": 0}
    if not out["exists"]:
        return out
    out["bytes"] = path.stat().st_size
    try:
        head = path.open("rb").read(4096).decode("utf-8", "replace")
    except OSError:
        return out

    m = re.search(r'type="(\w+)"', head)
    if m:
        out["kind"] = m.group(1)
    m = re.search(r'WholeExtent="([-\d\s]+)"', head)
    if m:
        out["cells"] = _n_cells(m.group(1))
    m = re.search(r'NumberOfPoints="(\d+)"', head)
    if m:
        out["points"] = int(m.group(1))
    # ImageData has no NumberOfPoints; points follow from the extent.
    if out["kind"] == "ImageData" and not out["points"]:
        m = re.search(r'WholeExtent="([-\d\s]+)"', head)
        if m:
            v = [int(t) for t in m.group(1).split()]
            out["points"] = ((v[1] - v[0] + 1) * (v[3] - v[2] + 1)
                             * (v[5] - v[4] + 1))
    return out


def inspect_pvd(pvd: Path) -> dict:
    """Timesteps in a .pvd and whether the files it names are real."""
    res = {"pvd": pvd, "timesteps": 0, "missing": [], "empty": [],
           "pieces": [], "error": None}
    try:
        root = ET.parse(pvd).getroot()
    except Exception as exc:                                # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        return res
    datasets = root.findall(".//DataSet")
    res["timesteps"] = len(datasets)
    for ds in datasets:
        f = ds.get("file")
        if not f:
            continue
        p = (pvd.parent / f).resolve()
        info = inspect_piece(p)
        res["pieces"].append(info)
        if not info["exists"]:
            res["missing"].append(p.name)
        elif info["points"] == 0:
            res["empty"].append(p.name)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="full_chain",
                    help="the chain's output directory (default: full_chain)")
    args = ap.parse_args(argv)

    root = Path(args.run_dir)
    if not root.is_dir():
        print(f"{RED}no such directory: {root}{RST}")
        print("  pass the chain's --out directory, e.g. full_chain")
        return 2

    pvds = sorted(root.rglob("*.pvd"))
    print(f"scanning {root}")
    if not pvds:
        print(f"\n{RED}NO .pvd FILES AT ALL{RST}")
        print("  The composite reads .pvd collections. Without them every")
        print("  chapter is skipped and the render is black by construction.")
        vti = list(root.rglob("*.vti")) + list(root.rglob("*.vtp"))
        if vti:
            print(f"\n  {YLW}But {len(vti)} .vti/.vtp files ARE present.{RST}")
            print("  A stage was interrupted between writing its frames and")
            print("  writing the collection. Re-run that stage; the frames")
            print("  alone are not enough.")
        else:
            print("\n  No frame data either -- nothing has been run yet.")
        return 1

    total_ok = 0
    problems = []
    print()
    for pvd in pvds:
        r = inspect_pvd(pvd)
        rel = pvd.relative_to(root)
        if r["error"]:
            print(f"  {RED}[BAD ]{RST} {rel}  unreadable: {r['error']}")
            problems.append(str(rel))
            continue
        if r["timesteps"] == 0:
            print(f"  {RED}[BAD ]{RST} {rel}  lists NO timesteps")
            problems.append(str(rel))
            continue
        if r["missing"]:
            print(f"  {RED}[BAD ]{RST} {rel}  {len(r['missing'])} of "
                  f"{r['timesteps']} referenced files are missing "
                  f"(e.g. {r['missing'][0]})")
            problems.append(str(rel))
            continue
        if r["empty"]:
            print(f"  {RED}[BAD ]{RST} {rel}  {len(r['empty'])} of "
                  f"{r['timesteps']} frames contain zero points")
            problems.append(str(rel))
            continue

        pts = r["pieces"][0]["points"] if r["pieces"] else 0
        kind = r["pieces"][0]["kind"] if r["pieces"] else "?"
        mb = sum(p["bytes"] for p in r["pieces"]) / 1024 ** 2
        print(f"  {GRN}[ ok ]{RST} {rel}  {r['timesteps']} timesteps, "
              f"{kind}, {pts:,} points/frame, {mb:.1f} MB")
        total_ok += 1

    print()
    if problems:
        print(f"{RED}{len(problems)} collection(s) unusable.{RST}")
        print("  Black frames from these are a DATA problem, not a scene")
        print("  problem -- fixing the composite script will not help.")
        return 1

    print(f"{GRN}{total_ok} collection(s) have real data.{RST}")
    print("  If the render is still black, the cause is the scene:")
    print("  camera, clipping range, or representation. Those live in")
    print("  hvi_emp/viz/composite.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
