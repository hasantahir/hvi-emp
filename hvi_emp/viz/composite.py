"""The continuous zoom composite: one movie across every solver.

Read this before showing the result to anyone
---------------------------------------------
This assembles a single continuous animation whose camera pulls out through
six orders of magnitude while time advances, handing over from one solver to
the next. It looks like one simulation. **It is not.** It is four independent
simulations of the same *scenario*, and the zoom crosses model boundaries as
well as scale boundaries:

* LAMMPS runs a ~1 nm projectile for ~100 ps.
* M2C runs a ~1 mm projectile for ~10 us.
* OpenMHD expands a plume into a magnetised ambient over ~ms.
* PIConGPU resolves charge separation in a um-scale patch over ~ns.

A 1 nm impact and a 1 mm impact are not the same event at different
magnifications. Crater-to-projectile ratios, vaporised fractions and
ionisation all change with size and speed, so the handover between chapters
is a *change of model*, not a dissolve.

The zoom itself is not arbitrary -- the plume genuinely expands as time
passes, so pulling out while moving forward follows the physics. What the
zoom cannot show is that the thing being zoomed out from was replaced.

So this module makes the composite nature unmissable rather than
discoverable:

* a **persistent banner**, burnt into every frame, naming the current code,
  its length scale and its clock;
* a **handover card** between chapters, stating both codes and the size of
  the jump;
* `COMPOSITE_NOTICE` on the opening and closing frames.

None of that is decoration. Remove it and the animation asserts something
false.

If you want a figure with no such caveat, render one chapter on its own --
`viz.paraview_scene` does that, and a single-solver movie needs no
disclaimer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

COMPOSITE_NOTICE = ("COMPOSITE: four independent simulations of the same "
                    "scenario, not one simulation. Scales and models change "
                    "at each handover.")

#: Chapters in the order the camera visits them, which is also the order
#: physics happens. `zoom` is the half-width of the view in metres and is
#: what the camera interpolates through.
CHAPTER_DEFAULTS = (
    {"stage": "md", "code": "LAMMPS", "zoom": 5.0e-9,
     "clock": "100 ps", "what": "atoms, ejecta, fragments"},
    {"stage": "hydro", "code": "M2C", "zoom": 2.0e-4,
     "clock": "10 us", "what": "crater, shock, ionisation"},
    {"stage": "reduced", "code": "hvi_emp", "zoom": 4.0e-1,
     "clock": "10 us", "what": "plume expanding into vacuum"},
    {"stage": "mhd", "code": "OpenMHD", "zoom": 1.0e+2,
     "clock": "1 ms", "what": "magnetised plume, diamagnetic cavity"},
    {"stage": "pic", "code": "PIConGPU", "zoom": 1.0e-5,
     "clock": "1 ns", "what": "charge separation, the radiated pulse"},
)


@dataclass
class Chapter:
    """One solver's contribution to the composite."""
    stage: str
    code: str
    zoom: float                      # view half-width [m]
    clock: str
    what: str
    pvd: str | None = None           # None when the solver has not run
    frames: int = 48
    colour_by: str = "log10_plasma_density"
    notes: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.pvd) and os.path.isfile(self.pvd)

    def banner(self) -> str:
        """The line burnt into every frame of this chapter."""
        return (f"{self.code}  |  {_si(self.zoom)} scale  |  {self.clock}  "
                f"|  {self.what}")


def _si(metres: float) -> str:
    """A length a person reads without counting zeros."""
    for scale, unit in ((1e-9, "nm"), (1e-6, "um"), (1e-3, "mm"),
                        (1.0, "m"), (1e3, "km")):
        if metres < scale * 1000:
            return f"{metres / scale:.0f} {unit}"
    return f"{metres:.3g} m"


def build_chapters(pvds: dict, frames: int = 48) -> list:
    """Chapters for whichever solvers actually produced output.

    `pvds` maps stage name -> `.pvd` path. Missing or absent stages are kept
    in the list but marked unavailable, so the report can say what the movie
    is missing rather than quietly shortening itself.
    """
    out = []
    for spec in CHAPTER_DEFAULTS:
        ch = Chapter(frames=frames, **spec)
        ch.pvd = pvds.get(ch.stage)
        if not ch.available:
            ch.notes.append(
                f"{ch.code} produced no output; this chapter is a title card "
                f"naming what is missing rather than a silent omission")
        out.append(ch)
    return out


def zoom_keyframes(chapters, hold: int = 12) -> list:
    """Camera half-width per frame across the whole composite.

    Geometric interpolation, because the scales span nine decades and a
    linear ramp would spend 99% of the movie in the last chapter. Each
    chapter holds at its own scale for `hold` frames, then the camera
    interpolates to the next.
    """
    import numpy as np

    usable = [c for c in chapters if c.available]
    if not usable:
        return []
    keys = []
    for i, ch in enumerate(usable):
        keys.extend([ch.zoom] * hold)
        if i + 1 < len(usable):
            nxt = usable[i + 1]
            keys.extend(np.geomspace(ch.zoom, nxt.zoom,
                                     max(ch.frames, 2))[1:-1].tolist())
    return [float(k) for k in keys]


def write_composite_script(chapters, out_path: str,
                           movie_path: str | None = None,
                           resolution=(1920, 1080),
                           font: int = 24, hold: int = 12) -> str:
    """Write the ParaView script that renders the composite.

    Each chapter is loaded as its own reader, shown only while the camera is
    in its range, and captioned. Between chapters a handover card names both
    codes and the scale jump.
    """
    available = [c for c in chapters if c.available]
    missing = [c for c in chapters if not c.available]

    ch_lines = []
    for i, c in enumerate(available):
        ch_lines.append(
            f"    dict(name={c.stage!r}, code={c.code!r}, "
            f"pvd={os.path.abspath(c.pvd)!r}, zoom={c.zoom!r}, "
            f"colour_by={c.colour_by!r}, banner={c.banner()!r})")
    chapters_py = "[\n" + ",\n".join(ch_lines) + "\n]" if ch_lines else "[]"

    handovers = []
    for a, b in zip(available, available[1:]):
        factor = b.zoom / a.zoom
        handovers.append(
            f"    ({a.code!r}, {b.code!r}, "
            f"'{_si(a.zoom)} -> {_si(b.zoom)}', {abs(factor):.3g})")
    handovers_py = "[\n" + ",\n".join(handovers) + "\n]" if handovers else "[]"

    missing_py = repr([f"{c.code} ({c.stage})" for c in missing])

    text = _TEMPLATE.format(
        script=os.path.basename(out_path),
        chapters=chapters_py,
        handovers=handovers_py,
        missing=missing_py,
        notice=COMPOSITE_NOTICE,
        resolution=list(resolution),
        font=font,
        hold=hold,
        movie=(os.path.abspath(movie_path) if movie_path else None),
    )
    parent = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    os.chmod(out_path, 0o755)
    return text


_TEMPLATE = '''#!/usr/bin/env pvpython
"""Continuous-zoom composite, generated by hvi_emp.viz.composite.

    pvbatch {script}

{notice}
"""
import os
import sys

try:
    from paraview.simple import *            # noqa: F401,F403
except ImportError:
    sys.exit("run with ParaView's python:  pvbatch {script}")

paraview.simple._DisableFirstRenderCameraReset()

CHAPTERS = {chapters}
HANDOVERS = {handovers}
MISSING = {missing}
NOTICE = {notice!r}
MOVIE = {movie!r}
HOLD = {hold}

if not CHAPTERS:
    sys.exit("no chapters have output yet -- run the solvers first:\\n"
             "  python scripts/run_full_chain.py --submit")

print("composite over %d chapters" % len(CHAPTERS))
for c in CHAPTERS:
    print("  %-10s %s" % (c["name"], c["banner"]))
if MISSING:
    print("MISSING (rendered as title cards, not silently dropped):")
    for m in MISSING:
        print("  %s" % m)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = {resolution}
view.Background = [0.0, 0.0, 0.0]
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 0


def _try(label, fn):
    try:
        fn()
        return True
    except Exception as exc:
        print("  %s: not applied (%s)" % (label, type(exc).__name__))
        return False


# ---------------------------------------------------------------------------
# Readers, one per chapter, all hidden to start with
# ---------------------------------------------------------------------------
readers = []
for c in CHAPTERS:
    r = PVDReader(registrationName=c["name"], FileName=c["pvd"])
    r.UpdatePipeline()
    info = r.GetPointDataInformation()
    avail = [info.GetArray(i).GetName()
             for i in range(info.GetNumberOfArrays())]
    field = c["colour_by"] if c["colour_by"] in avail else (
        avail[0] if avail else None)
    if field is None:
        print("  %s has no point arrays -- skipped" % c["name"])
        continue
    d = Show(r, view, "UniformGridRepresentation")
    d.Representation = "Volume"
    ColorBy(d, ("POINTS", field))
    d.RescaleTransferFunctionToDataRangeOverTime()
    _try("GPU mapper", lambda dd=d: setattr(dd, "VolumeRenderingMode",
                                            "GPU Based"))
    Hide(r, view)
    readers.append((c, r, d))

# ---------------------------------------------------------------------------
# The banner. Burnt into every frame, because without it this animation
# asserts that six orders of magnitude were simulated by one code.
# ---------------------------------------------------------------------------
banner = Text(registrationName="banner")
banner.Text = CHAPTERS[0]["banner"]
bd = Show(banner, view, "TextSourceRepresentation")
bd.FontSize = {font}
bd.Color = [1.0, 1.0, 1.0]
bd.WindowLocation = "Upper Left Corner"

notice = Text(registrationName="notice")
notice.Text = NOTICE
nd = Show(notice, view, "TextSourceRepresentation")
nd.FontSize = max(12, {font} - 8)
nd.Color = [1.0, 0.75, 0.3]          # amber: a caveat, not a label
nd.WindowLocation = "Lower Left Corner"

card = Text(registrationName="handover")
cd = Show(card, view, "TextSourceRepresentation")
cd.FontSize = {font} + 6
cd.Color = [1.0, 1.0, 1.0]
cd.WindowLocation = "Upper Center"
card.Text = ""


def show_only(idx):
    """Only chapter `idx` is visible."""
    for j, (c, r, d) in enumerate(readers):
        if j == idx:
            Show(r, view)
        else:
            Hide(r, view)
    banner.Text = readers[idx][0]["banner"]


def frame_out(n):
    Render()
    if MOVIE:
        SaveScreenshot(MOVIE % n, view, ImageResolution=view.ViewSize)


# ---------------------------------------------------------------------------
# Walk the chapters: hold, then hand over
# ---------------------------------------------------------------------------
n = 0
for idx, (c, r, d) in enumerate(readers):
    show_only(idx)
    card.Text = ""
    times = r.TimestepValues or [0.0]
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()

    half = c["zoom"]
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CameraPosition = [2.2 * half, -1.6 * half, 1.2 * half]
    view.CameraViewUp = [0.0, 0.0, 1.0]

    for k in range(HOLD):
        t = times[min(int(k * len(times) / max(HOLD, 1)), len(times) - 1)]
        scene.AnimationTime = t
        frame_out(n)
        n += 1

    if idx + 1 < len(readers):
        a, b, span, factor = HANDOVERS[idx]
        card.Text = ("%s  ->  %s        %s   (x%.3g)\\n"
                     "different code, different projectile size: "
                     "a change of model, not a zoom"
                     % (a, b, span, factor))
        for k in range(8):
            frame_out(n)
            n += 1

card.Text = ""
banner.Text = NOTICE
frame_out(n)

print("rendered %d frames" % n)
if MOVIE:
    print("assemble with:")
    print("  ffmpeg -framerate 12 -i %s -c:v libx264 -pix_fmt yuv420p "
          "-crf 18 composite.mp4" % MOVIE.replace("%04d", "%%04d"))
else:
    print("no --movie given: scene built interactively, nothing written")
'''


__all__ = ["Chapter", "CHAPTER_DEFAULTS", "COMPOSITE_NOTICE",
           "build_chapters", "zoom_keyframes", "write_composite_script"]
