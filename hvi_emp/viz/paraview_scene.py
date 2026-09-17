"""Generate a ParaView scene that opens correctly the first time.

The problem this solves
-----------------------
A `.pvd` opened in ParaView with default settings is a grey outline. Getting
from there to a legible picture of an impact takes a dozen correct choices —
which array to colour by, log or linear, where to put the opacity knots, how
to make the target opaque without hiding the plume, camera, font sizes — and
getting any one of them wrong produces something that looks like a bug in the
physics. It is not; it is a transfer function.

So this module writes those choices down as a script.

Why a script rather than a `.pvsm`
----------------------------------
`.pvsm` is ParaView's own state format, and hand-writing one is a bad idea:
it is a large XML dump of proxy IDs that is tied to the writing version, and
a mismatch fails in ways that are tedious to debug. A `paraview.simple`
script is a few hundred readable lines, survives version changes, and can be
diffed.

The generated script's **last act is to call `SaveState()`**, so you get a
`.pvsm` produced by *your* ParaView for *your* version — valid by
construction. Run the script once, and from then on open the state file.

Usage
-----
    pvpython scene_impact.py            # batch; also writes the .pvsm
    # or, inside ParaView: Tools > Python Shell > Run Script

Nothing here imports `paraview`. This module only *writes* the script; it has
no ParaView dependency and is tested without one.

M2C
---
`FIELD_ALIASES` maps this framework's array names onto M2C's, so the same
scene works on a solved hydrocode result. Pass ``source="m2c"``. The mapping
is a guess at M2C's output names until a run confirms them — `--check` on the
generated script reports any array it cannot find rather than colouring by
nothing.
"""

from __future__ import annotations

import os

#: Colour-map choices per field. `preset` names are ParaView's built-ins.
#:
#: `log10_plasma_density` gets a *linear* map because the field is already
#: log10 — applying ParaView's log scaling on top would take the log of a
#: negative number and silently clamp.
FIELD_STYLE = {
    "log10_plasma_density": {
        "preset": "Inferno (matplotlib)",
        "title": "log10 plasma density [kg/m3]",
        "log": False,
    },
    "log10_electron_density": {
        "preset": "Viridis (matplotlib)",
        "title": "log10 electron density [m-3]",
        "log": False,
    },
    "temperature_eV": {
        "preset": "Black-Body Radiation",
        "title": "electron temperature [eV]",
        "log": False,
    },
    "ionisation": {
        "preset": "Cool to Warm",
        "title": "mean charge Zbar",
        "log": False,
    },
    "shock_pressure": {
        "preset": "Inferno (matplotlib)",
        "title": "shock pressure [Pa]  (PARAMETRIC)",
        "log": True,
    },
    "plasma_density": {
        "preset": "Inferno (matplotlib)",
        "title": "plasma density [kg/m3]",
        "log": True,
    },
    "density": {
        "preset": "Grayscale",
        "title": "density [kg/m3]",
        "log": False,
    },
    "projectile_fraction": {
        "preset": "Cool to Warm",
        "title": "projectile fraction  (UNIFORM - see docs)",
        "log": False,
    },
}

#: Our array name -> the equivalent in another code's output.
FIELD_ALIASES = {
    "hvi_emp": {},                       # identity
    "m2c": {
        "log10_plasma_density": "Density",
        "plasma_density": "Density",
        "temperature_eV": "Temperature",
        "ionisation": "MeanCharge",
        "log10_electron_density": "ElectronDensity",
        "shock_pressure": "Pressure",
        "density": "Density",
        "material": "MaterialID",
    },
}

#: Volume mappers. `VolumeRenderingMode` is the property that decides whether
#: a 256^3+ volume is interactive or a slideshow, and ParaView's default
#: ("Smart") does not guarantee the GPU path.
RENDERERS = ("gpu", "index", "smart")

#: Font sizes. A talk is viewed from the back of a room; the defaults are
#: sized for a desk.
PRESETS = {
    "talk": {"font": 22, "title_font": 26, "legend_len": 0.38,
             "background": [0.02, 0.02, 0.06], "resolution": [1920, 1080]},
    "paper": {"font": 14, "title_font": 16, "legend_len": 0.30,
              "background": [1.0, 1.0, 1.0], "resolution": [1600, 1200]},
    "diagnostic": {"font": 12, "title_font": 13, "legend_len": 0.25,
                   "background": [0.32, 0.34, 0.43],
                   "resolution": [1280, 960]},
}


def _fmt_list(x) -> str:
    return "[" + ", ".join(f"{v!r}" for v in x) + "]"


def write_scene_script(pvd_path: str, out_path: str,
                       colour_by: str = "log10_plasma_density",
                       preset: str = "talk",
                       source: str = "hvi_emp",
                       show_target: bool = True,
                       title: str | None = None,
                       state_path: str | None = None,
                       movie_path: str | None = None,
                       renderer: str = "gpu",
                       clip: bool = True) -> str:
    """Write a `paraview.simple` script that builds the scene.

    Parameters
    ----------
    pvd_path : str
        The `.pvd` to open. Relative paths are resolved against the script's
        own location at run time, so the pair can be moved together.
    colour_by : str
        Array to volume-render. Use `log10_plasma_density`, not
        `plasma_density` -- see the note in `viz.vti`.
    preset : {"talk", "paper", "diagnostic"}
    source : {"hvi_emp", "m2c"}
        Which array-name convention the `.pvd` uses.
    show_target : bool
        Draw the solid target and crater as an opaque surface underneath the
        plume. Turn off for a co-moving plume scene, where there is no target
        in the box.
    state_path : str or None
        Where the script should `SaveState()`. Defaults to the script path
        with a `.pvsm` suffix.
    movie_path : str or None
        If given, the script also writes an animation there (`.avi` or a
        `.png` series).
    renderer : {"gpu", "index", "smart"}
        `gpu` pins ParaView's GPU volume mapper -- the right default on any
        machine with a discrete card, and the setting whose absence makes a
        large volume crawl. `index` additionally tries to load the NVIDIA
        IndeX plugin, which is the better mapper above ~512^3 but is a
        separate plugin and not present in every build. `smart` leaves
        ParaView to choose, which is what happened before this existed.
    clip : bool
        Cut the volume in half at y = 0 and look into it. On by default,
        because a solid exterior hides the crater interior, the shock inside
        the target and where the plume sits relative to the cavity -- which
        is everything the picture is for. Both reference animations are cut
        this way.

    Returns
    -------
    The script text, which is also written to `out_path`.
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    if source not in FIELD_ALIASES:
        raise KeyError(f"unknown source {source!r}; "
                       f"have {sorted(FIELD_ALIASES)}")
    if renderer not in RENDERERS:
        raise KeyError(f"unknown renderer {renderer!r}; "
                       f"have {sorted(RENDERERS)}")
    p = PRESETS[preset]
    alias = FIELD_ALIASES[source]
    array = alias.get(colour_by, colour_by)
    style = FIELD_STYLE.get(colour_by, {
        "preset": "Inferno (matplotlib)", "title": array, "log": False})
    mat_array = alias.get("material", "material")

    if state_path is None:
        state_path = os.path.splitext(out_path)[0] + ".pvsm"

    title = title or f"HVI-EMP - {os.path.basename(pvd_path)}"

    text = '''#!/usr/bin/env pvpython
"""ParaView scene, generated by hvi_emp.viz.paraview_scene.

    pvpython {script}

or inside ParaView: Tools > Python Shell > Run Script.

It builds the whole scene, reports any array it could not find, and then
calls SaveState() so you get a .pvsm written by YOUR ParaView version. After
the first run you can just open that state file.

Colouring by: {array}
Preset:       {preset}
Source:       {source}
"""
import os
import sys

try:
    from paraview.simple import *            # noqa: F401,F403
except ImportError:
    sys.exit(
        "This script must be run by ParaView's Python, not the system one.\\n"
        "  pvpython {script}\\n"
        "or open ParaView and use Tools > Python Shell > Run Script.\\n"
        "Install ParaView with:  mamba install -c conda-forge paraview")

HERE = os.path.dirname(os.path.abspath(__file__))
PVD = os.path.join(HERE, {pvd_rel!r}) if not os.path.isabs({pvd_rel!r}) \\
    else {pvd_rel!r}
COLOUR_BY = {array!r}
MATERIAL = {mat_array!r}

if not os.path.isfile(PVD):
    sys.exit("cannot find %s -- generate it first with "
             "examples/14_impact_scenes.py" % PVD)

paraview.simple._DisableFirstRenderCameraReset()

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
src = PVDReader(registrationName={base!r}, FileName=PVD)
src.UpdatePipeline()

info = src.GetPointDataInformation()
available = [info.GetArray(i).GetName()
             for i in range(info.GetNumberOfArrays())]
print("arrays in this dataset: %s" % ", ".join(sorted(available)))

if COLOUR_BY not in available:
    sys.exit(
        "array %r is not in this dataset.\\n"
        "  available: %s\\n"
        "  Either regenerate with that field included, or edit COLOUR_BY "
        "above." % (COLOUR_BY, ", ".join(sorted(available))))

times = src.TimestepValues or [0.0]
print("%d timesteps, t = %.3e to %.3e s" % (len(times), times[0], times[-1]))

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = {resolution}
view.Background = {background}
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 1
view.OrientationAxesLabelColor = [0.9, 0.9, 0.9]

# ---------------------------------------------------------------------------
# Volume rendering of the plasma
# ---------------------------------------------------------------------------
disp = Show(src, view, "UniformGridRepresentation")
disp.Representation = "Volume"
ColorBy(disp, ("POINTS", COLOUR_BY))

# ---------------------------------------------------------------------------
# GPU volume rendering
#
# Without this ParaView uses the "Smart" mapper, which chooses for itself and
# on a large volume can fall back to CPU ray casting -- correct, and slow
# enough that people conclude the data is broken. Setting it explicitly is the
# difference between interactive and not.
#
# Every assignment here is wrapped: property names and the set of accepted
# values move between ParaView versions, and a hard failure would take the
# whole scene down for the sake of a rendering hint. Whatever does not apply
# is reported, not swallowed.
# ---------------------------------------------------------------------------
def _try(label, fn):
    try:
        fn()
        print("  %s: on" % label)
        return True
    except Exception as exc:
        print("  %s: not applied (%s)" % (label, type(exc).__name__))
        return False


print("renderer setup:")
{renderer_block}
# Shading makes a volume read as three-dimensional rather than as fog. It is
# nearly free on a GPU and expensive on a CPU mapper, which is the other
# reason to pin the mapper above.
_try("shading", lambda: setattr(disp, "Shade", 1))
_try("ambient", lambda: setattr(disp, "Ambient", 0.25))
_try("diffuse", lambda: setattr(disp, "Diffuse", 0.85))

# Rescale over the WHOLE time series, not the first frame. Rescaling per
# frame is the single most common way to make an expanding, cooling plume
# look static: the colour map follows the peak down and nothing appears to
# change.
disp.RescaleTransferFunctionToDataRangeOverTime()

lut = GetColorTransferFunction(COLOUR_BY)
pwf = GetOpacityTransferFunction(COLOUR_BY)
lut.ApplyPreset({style_preset!r}, True)
{log_line}
rng = lut.RGBPoints[0], lut.RGBPoints[-4]
lo, hi = float(rng[0]), float(rng[1])
print("colour range %.4g .. %.4g" % (lo, hi))

# Opacity: transparent at the bottom of the range, opaque at the top. A flat
# opacity ramp renders the Gaussian tail as fog and hides the core, which is
# the other half of the "featureless blob" problem.
span = hi - lo if hi > lo else 1.0
pwf.Points = [
    lo,               0.0,  0.5, 0.0,
    lo + 0.45 * span, 0.02, 0.5, 0.0,
    lo + 0.75 * span, 0.25, 0.5, 0.0,
    hi,               0.85, 0.5, 0.0,
]
disp.ScalarOpacityFunction = pwf
disp.ScalarOpacityUnitDistance = 1.0e-5
disp.SetScalarBarVisibility(view, 1)

bar = GetScalarBar(lut, view)
bar.Title = {style_title!r}
bar.ComponentTitle = ""
bar.TitleFontSize = {title_font}
bar.LabelFontSize = {font}
bar.ScalarBarLength = {legend_len}
bar.TitleColor = [1.0, 1.0, 1.0]
bar.LabelColor = [1.0, 1.0, 1.0]

{target_block}
# ---------------------------------------------------------------------------
# Annotation: time in units a person can read
# ---------------------------------------------------------------------------
ann = AnnotateTimeFilter(registrationName="clock", Input=src)
ann.Format = "t = {{time:.3g}} s"
annd = Show(ann, view, "TextSourceRepresentation")
annd.FontSize = {title_font}
annd.Color = [1.0, 1.0, 1.0]
annd.WindowLocation = "Upper Left Corner"

label = Text(registrationName="caption")
label.Text = {caption!r}
labeld = Show(label, view, "TextSourceRepresentation")
labeld.FontSize = {font}
labeld.Color = [0.75, 0.78, 0.85]
labeld.WindowLocation = "Lower Left Corner"

# ---------------------------------------------------------------------------
# Camera: slightly above the surface, looking down the impact axis
# ---------------------------------------------------------------------------
src.UpdatePipeline()
b = src.GetDataInformation().GetBounds()
cx = 0.5 * (b[0] + b[1])
cy = 0.5 * (b[2] + b[3])
cz = 0.5 * (b[4] + b[5])
reach = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])

view.CameraFocalPoint = [cx, cy, cz]
view.CameraPosition = [cx + 1.6 * reach, cy - 1.9 * reach, cz + 1.1 * reach]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelProjection = 0
ResetCamera()
view.CameraPosition = [cx + 1.6 * reach, cy - 1.9 * reach, cz + 1.1 * reach]

{clip_block}
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = times[len(times) // 2]
Render()

# ---------------------------------------------------------------------------
# Save a state file written by THIS ParaView, so it is valid for this version
# ---------------------------------------------------------------------------
STATE = os.path.join(HERE, {state_rel!r})
SaveState(STATE)
print("wrote %s -- open this instead of re-running the script" % STATE)
{movie_block}
print("done. Press play, or step the timeline.")
'''.format(
        script=os.path.basename(out_path),
        preset=preset,
        source=source,
        pvd_rel=os.path.relpath(pvd_path, os.path.dirname(
            os.path.abspath(out_path)) or "."),
        array=array,
        mat_array=mat_array,
        base=os.path.splitext(os.path.basename(pvd_path))[0],
        resolution=_fmt_list(p["resolution"]),
        background=_fmt_list(p["background"]),
        style_preset=style["preset"],
        style_title=style["title"],
        log_line=("lut.MapControlPointsToLogSpace()\nlut.UseLogScale = 1"
                  if style["log"] else "lut.UseLogScale = 0"),
        title_font=p["title_font"],
        font=p["font"],
        legend_len=p["legend_len"],
        caption=title,
        renderer_block=_renderer_block(renderer),
        clip_block=_CLIP_BLOCK if clip else
        "# clip=False: showing the exterior surface\n",
        target_block=_target_block() if show_target else
        "# target not drawn: no solid in view for this scene\n",
        state_rel=os.path.basename(state_path),
        movie_block=_movie_block(movie_path, out_path) if movie_path else "",
    )

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    os.chmod(out_path, 0o755)
    return text


def _renderer_block(renderer: str) -> str:
    """The mapper-selection lines for the generated script."""
    if renderer == "smart":
        return ('print("  mapper: left as ParaView\'s default (Smart)")\n'
                'print("  NOTE: Smart may pick a CPU mapper on a large "\n'
                '      "volume; pass renderer=\'gpu\' to pin the GPU path")\n')

    gpu = (
        '# "GPU Based" is the vtkGPUVolumeRayCastMapper. On an RTX card this\n'
        '# is the difference between rotating a 256^3 volume smoothly and\n'
        '# waiting seconds per frame.\n'
        '_gpu = _try("GPU volume mapper",\n'
        '            lambda: setattr(disp, "VolumeRenderingMode", '
        '"GPU Based"))\n'
        'if not _gpu:\n'
        '    print("    the property name or its accepted values differ in "\n'
        '          "this ParaView build; check the Properties panel, "\n'
        '          "Representation > Volume Rendering Mode")\n'
    )
    if renderer != "index":
        return gpu

    return gpu + (
        '\n'
        '# NVIDIA IndeX: a separate plugin, better than the built-in mapper\n'
        '# above roughly 512^3, and absent from some builds. Failure here is\n'
        '# not a problem -- the GPU mapper above is already active.\n'
        'try:\n'
        '    LoadDistributedPlugin("pvNVIDIAIndeX", True, globals())\n'
        '    _try("NVIDIA IndeX",\n'
        '         lambda: setattr(disp, "VolumeRenderingMode", '
        '"NVIDIA IndeX"))\n'
        'except Exception as exc:\n'
        '    print("  NVIDIA IndeX: not available (%s) -- staying on the "\n'
        '          "built-in GPU mapper" % type(exc).__name__)\n'
    )


_CLIP_BLOCK = '''# --------------------------------------------------------------
# Cutaway
#
# Both published impact animations are cut through the axis, and that is
# why they read as informative rather than as a rendered lump: a solid
# outer surface hides everything that matters -- the crater interior, the
# shock inside the target, where the plume sits relative to the cavity.
#
# This clips at y = 0 so the near half is removed and you look straight
# into the impact. Turn it off with `clip=False` if you want the exterior.
# --------------------------------------------------------------
_clip = Clip(registrationName="cutaway", Input=src)
_clip.ClipType = "Plane"
_clip.ClipType.Origin = [cx, cy, cz]
_clip.ClipType.Normal = [0.0, 1.0, 0.0]
try:
    _clip.Invert = 1
except Exception:
    pass
Hide(src, view)
disp = Show(_clip, view, "UnstructuredGridRepresentation")
disp.Representation = "Volume"
ColorBy(disp, ("POINTS", COLOUR_BY))
disp.RescaleTransferFunctionToDataRangeOverTime()
disp.SetScalarBarVisibility(view, 1)
print("cutaway: clipped at y = 0")

'''


def _target_block() -> str:
    """Opaque target and crater, drawn from the `material` labels."""
    return ('''# ------------------------------------------------------------
# Target and crater, as an opaque surface under the plume
#
# ILLUSTRATIVE. The reduced chain does not solve the solid target: the crater
# grows as sqrt(t) to a Cour-Palais radius and the shock is the model's own
# decay law on a hemisphere at Us*t. See docs/VISUALISATION.md. For a solved
# crater, run M2C and point this script at its output.
# ---------------------------------------------------------------------------
if MATERIAL in available:
    solid = Contour(registrationName="target", Input=src)
    solid.ContourBy = ["POINTS", MATERIAL]
    solid.Isosurfaces = [0.5]          # boundary of material 1 (target)
    solid.ComputeNormals = 1
    sd = Show(solid, view, "GeometryRepresentation")
    sd.Representation = "Surface"
    sd.AmbientColor = [0.45, 0.47, 0.52]
    sd.DiffuseColor = [0.45, 0.47, 0.52]
    sd.Opacity = 1.0
    ColorBy(sd, None)
else:
    print("no %r array -- target surface not drawn" % MATERIAL)

''')


def _movie_block(movie_path: str, script_path: str) -> str:
    """Batch-render the animation.

    Written to be run under `pvbatch`, which needs no display. On a build
    with EGL support this renders on the GPU headlessly -- the case that
    matters on a workstation you ssh into.
    """
    rel = os.path.relpath(movie_path, os.path.dirname(
        os.path.abspath(script_path)) or ".")
    stem = os.path.splitext(os.path.basename(rel))[0]
    return '''
# -------------------------------------------------------------------------
# Animation
#
# Run this under pvbatch for headless rendering:
#     pvbatch {script}
# A ParaView built with EGL renders offscreen on the GPU, so this works over
# ssh with no X display. If it complains about a display, the build is an
# OSMesa (software) one -- correct, but CPU-bound.
#
# A PNG series rather than a container format, because encoders vary between
# builds and a series can always be assembled afterwards:
#     ffmpeg -framerate 12 -i {stem}.%04d.png -c:v libx264 \\
#            -pix_fmt yuv420p -crf 18 {stem}.mp4
# -------------------------------------------------------------------------
MOVIE = os.path.join(HERE, {rel!r})
os.makedirs(os.path.dirname(MOVIE) or ".", exist_ok=True)
try:
    SaveAnimation(MOVIE, view, ImageResolution=view.ViewSize,
                  FrameRate=12, FrameWindow=[0, len(times) - 1])
    print("wrote %s (%d frames)" % (MOVIE, len(times)))
    print("  ffmpeg -framerate 12 -i {stem}.%04d.png -c:v libx264 "
          "-pix_fmt yuv420p -crf 18 {stem}.mp4")
except Exception as exc:
    print("animation failed (%s: %s)" % (type(exc).__name__, exc))
    print("  the scene and the .pvsm are still written; open them and use")
    print("  File > Save Animation by hand")
'''.format(rel=rel, stem=stem,
           script=os.path.basename(script_path))


def write_scene_for(result: dict, out_path: str, preset: str = "talk",
                    **kw) -> str:
    """Write a scene script straight from `viz.vti.write_scene`'s return.

    Picks `colour_by` from the scene, and turns the target surface off for a
    co-moving plume scene where there is no target in the box.
    """
    scene = result.get("scene", "")
    return write_scene_script(
        result["pvd"], out_path,
        colour_by=kw.pop("colour_by", result.get("colour_by",
                                                 "log10_plasma_density")),
        preset=preset,
        show_target=kw.pop("show_target", scene != "plume_comoving"),
        title=kw.pop("title", None),
        **kw)


#: Colour styling for per-atom arrays.
#:
#: `T_eV` gets a divergent map because the interesting thing in an MD frame is
#: the *contrast* between cold lattice and hot shocked material, and a
#: sequential map buries that in the middle of the range.
PARTICLE_STYLE = {
    "T_eV": {"preset": "Blue Orange (divergent)",
             "title": "temperature [eV]  (MD)", "log": False},
    "Zbar": {"preset": "Inferno (matplotlib)",
             "title": "mean charge Zbar  (INFERRED)", "log": False},
    "log10_electron_density": {"preset": "Viridis (matplotlib)",
                               "title": "log10 n_e [m-3]  (INFERRED)",
                               "log": False},
    "speed": {"preset": "Cool to Warm", "title": "speed [m/s]", "log": False},
    "fragment_id": {"preset": "Cool to Warm (Extended)",
                    "title": "fragment", "log": False},
    "fragment_atoms": {"preset": "Viridis (matplotlib)",
                       "title": "fragment size [atoms]", "log": True},
    "is_ejecta": {"preset": "Cool to Warm", "title": "ejecta", "log": False},
    "ke": {"preset": "Black-Body Radiation", "title": "kinetic energy [eV]",
           "log": False},
}

_PARTICLE_TEMPLATE = '''#!/usr/bin/env pvpython
"""MD particle scene, generated by hvi_emp.viz.paraview_scene.

    pvpython {script}

Particles coloured by {array}{withgrid}.
"""
import os
import sys

try:
    from paraview.simple import *            # noqa: F401,F403
except ImportError:
    sys.exit("Run this with ParaView's python:  pvpython {script}")

HERE = os.path.dirname(os.path.abspath(__file__))
ATOMS = os.path.join(HERE, {atoms_rel!r})
GRID = {grid_expr}
COLOUR_BY = {array!r}

if not os.path.isfile(ATOMS):
    sys.exit("cannot find %s" % ATOMS)

paraview.simple._DisableFirstRenderCameraReset()


def _try(label, fn):
    try:
        fn()
        print("  %s: on" % label)
        return True
    except Exception as exc:
        print("  %s: not applied (%s)" % (label, type(exc).__name__))
        return False


atoms = PVDReader(registrationName="atoms", FileName=ATOMS)
atoms.UpdatePipeline()
info = atoms.GetPointDataInformation()
available = [info.GetArray(i).GetName()
             for i in range(info.GetNumberOfArrays())]
print("per-atom arrays: %s" % ", ".join(sorted(available)))
if COLOUR_BY not in available:
    sys.exit("array %r not present; available: %s"
             % (COLOUR_BY, ", ".join(sorted(available))))

times = atoms.TimestepValues or [0.0]
print("%d timesteps" % len(times))

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = {resolution}
view.Background = [0.0, 0.0, 0.0]          # black: debris reads best on it
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 1
view.OrientationAxesLabelColor = [0.8, 0.8, 0.8]

# -------------------------------------------------------------------------
# Particles
#
# Point Gaussian, not Points. Plain square points alias badly on a regular
# lattice -- a crystal renders as a moire pattern that looks like structure
# and is not. Point Gaussian is GPU-rendered and scales with zoom.
# -------------------------------------------------------------------------
ad = Show(atoms, view, "GeometryRepresentation")
_try("point gaussian",
     lambda: setattr(ad, "Representation", "Point Gaussian"))
_try("point radius", lambda: setattr(ad, "GaussianRadius", {radius!r}))
_try("sphere shader", lambda: setattr(ad, "ShaderPreset", "Sphere"))
ColorBy(ad, ("POINTS", COLOUR_BY))
ad.RescaleTransferFunctionToDataRangeOverTime()

lut = GetColorTransferFunction(COLOUR_BY)
lut.ApplyPreset({style_preset!r}, True)
lut.UseLogScale = {log_flag}
ad.SetScalarBarVisibility(view, 1)
bar = GetScalarBar(lut, view)
bar.Title = {style_title!r}
bar.ComponentTitle = ""
bar.TitleFontSize = {title_font}
bar.LabelFontSize = {font}
bar.ScalarBarLength = {legend_len}
bar.TitleColor = [1.0, 1.0, 1.0]
bar.LabelColor = [1.0, 1.0, 1.0]

{grid_block}
ann = AnnotateTimeFilter(registrationName="clock", Input=atoms)
ann.Format = "t = {{time:.3g}} s"
annd = Show(ann, view, "TextSourceRepresentation")
annd.FontSize = {title_font}
annd.Color = [1.0, 1.0, 1.0]
annd.WindowLocation = "Upper Left Corner"

note = Text(registrationName="caption")
note.Text = ("MD: positions and temperature are simulated. "
             "IONISATION IS INFERRED -- classical MD has no electrons.")
noted = Show(note, view, "TextSourceRepresentation")
noted.FontSize = {font}
noted.Color = [0.7, 0.72, 0.8]
noted.WindowLocation = "Lower Left Corner"

atoms.UpdatePipeline()
b = atoms.GetDataInformation().GetBounds()
cx, cy, cz = 0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5])
reach = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
view.CameraFocalPoint = [cx, cy, cz]
view.CameraPosition = [cx + 2.2*reach, cy - 1.2*reach, cz + 0.6*reach]
view.CameraViewUp = [0.0, 0.0, 1.0]
ResetCamera()
view.CameraPosition = [cx + 2.2*reach, cy - 1.2*reach, cz + 0.6*reach]

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
Render()

STATE = os.path.join(HERE, {state_rel!r})
SaveState(STATE)
print("wrote %s" % STATE)
{movie_block}
print("done.")
'''

_PLASMA_VOLUME_BLOCK = '''# --------------------------------------------------
# Plasma, as an emissive volume behind the particles
#
# Additive blending, so the plasma glows THROUGH the debris rather than
# occluding it. Coloured by electron density, because the question this
# picture answers is "which of this debris is plasma", and n_e is what
# Stage 3 consumes.
#
# The opacity ramp starts at 60% of the range on purpose: the cold
# majority must contribute nothing, or the frame fills with haze and the
# particles vanish behind it.
# ----------------------------------------------------------------------
if GRID and os.path.isfile(GRID):
    grid = PVDReader(registrationName="plasma", FileName=GRID)
    grid.UpdatePipeline()
    ginfo = grid.GetPointDataInformation()
    gavail = [ginfo.GetArray(i).GetName()
              for i in range(ginfo.GetNumberOfArrays())]
    gfield = ("electron_density" if "electron_density" in gavail
              else (gavail[0] if gavail else None))
    if gfield is None:
        print("plasma grid has no arrays -- skipping the volume")
    else:
        gd = Show(grid, view, "UniformGridRepresentation")
        gd.Representation = "Volume"
        ColorBy(gd, ("POINTS", gfield))
        gd.RescaleTransferFunctionToDataRangeOverTime()
        _try("GPU volume mapper",
             lambda: setattr(gd, "VolumeRenderingMode", "GPU Based"))
        glut = GetColorTransferFunction(gfield)
        gpwf = GetOpacityTransferFunction(gfield)
        glut.ApplyPreset("Inferno (matplotlib)", True)
        lo, hi = float(glut.RGBPoints[0]), float(glut.RGBPoints[-4])
        span = (hi - lo) or 1.0
        gpwf.Points = [lo, 0.0, 0.5, 0.0,
                       lo + 0.60*span, 0.0, 0.5, 0.0,
                       lo + 0.85*span, 0.30, 0.5, 0.0,
                       hi, 0.9, 0.5, 0.0]
        gd.ScalarOpacityFunction = gpwf
        _try("additive blending",
             lambda: setattr(gd, "BlendMode", "Additive"))
        print("plasma volume on %r" % gfield)
elif GRID:
    print("plasma grid %s not found -- particles only" % GRID)

'''


def write_particle_scene_script(atoms_pvd: str, out_path: str,
                                grid_pvd: str | None = None,
                                colour_by: str = "T_eV",
                                preset: str = "talk",
                                point_size: float = 2.0,
                                movie_path: str | None = None,
                                state_path: str | None = None) -> str:
    """A scene for per-atom MD data: the dust cloud, optionally over plasma.

    The other half of the pair. `write_scene_script` volume-renders a
    continuum; this renders **discrete particles** -- the look of an MD impact
    render, where the target is a dense cloud of points and the ejecta a spray
    of individual atoms and fragments.

    Parameters
    ----------
    atoms_pvd : str
        The per-atom `.pvd` from `lammps_postprocess.convert_dump`.
    grid_pvd : str or None
        The binned `.pvd` from the same call. When given, the ionised material
        is volume-rendered *behind* the particles with additive blending, so
        you can see which part of the debris is actually plasma. Omitted, you
        get particles alone.
    colour_by : str
        A per-atom array. `T_eV` gives the classic thermal render; `Zbar` or
        `log10_electron_density` show the inferred plasma; `fragment_id`
        gives each fragment its own colour, which is the debris-cloud look
        and needs `convert_dump(..., fragments=True)`.
    point_size : float
        Gaussian radius. Larger reads better from the back of a room.
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    p = PRESETS[preset]
    if state_path is None:
        state_path = os.path.splitext(out_path)[0] + ".pvsm"
    here = os.path.dirname(os.path.abspath(out_path)) or "."
    style = PARTICLE_STYLE.get(colour_by, {
        "preset": "Inferno (matplotlib)", "title": colour_by, "log": False})

    text = _PARTICLE_TEMPLATE.format(
        script=os.path.basename(out_path),
        atoms_rel=os.path.relpath(atoms_pvd, here),
        grid_expr=(f"os.path.join(HERE, {os.path.relpath(grid_pvd, here)!r})"
                   if grid_pvd else "None"),
        array=colour_by,
        withgrid=(", plasma volume behind" if grid_pvd else ""),
        resolution=_fmt_list(p["resolution"]),
        radius=float(point_size),
        style_preset=style["preset"],
        style_title=style["title"],
        log_flag=1 if style["log"] else 0,
        title_font=p["title_font"],
        font=p["font"],
        legend_len=p["legend_len"],
        grid_block=(_PLASMA_VOLUME_BLOCK if grid_pvd
                    else "# no grid .pvd given: particles only\n"),
        state_rel=os.path.basename(state_path),
        movie_block=_movie_block(movie_path, out_path) if movie_path else "",
    )

    os.makedirs(here, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    os.chmod(out_path, 0o755)
    return text


__all__ = ["write_scene_script", "write_scene_for",
           "write_particle_scene_script", "FIELD_STYLE", "FIELD_ALIASES",
           "PARTICLE_STYLE", "PRESETS", "RENDERERS"]
