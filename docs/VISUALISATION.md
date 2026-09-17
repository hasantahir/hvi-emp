# Volumetric visualisation (`.vti` for ParaView / VisIt)

Time-resolved 3D volumes of the impact: projectile approach, shock into the
target, crater, plasma plume, and the slower condensate ("smoke").

```bash
python examples/14_impact_scenes.py            # two scenes + ParaView scripts
pvpython scenes/impact/scene_impact.py         # builds the view, saves a .pvsm
```

```python
from hvi_emp import run_scenario
from hvi_emp.viz.vti import write_scene
from hvi_emp.viz.paraview_scene import write_scene_for

sc = run_scenario("Fe", "Al", mass=1e-9, velocity=30e3)
res = write_scene(sc, "scenes/impact", scene="impact", n_frames=48)
write_scene_for(res, "scenes/impact/scene_impact.py", preset="talk")
```

Output is a `.vti` per frame plus a `.pvd` collection carrying the **real
physical times**, so ParaView's timeline is in seconds. No VTK dependency —
the XML/binary format is written directly.

---

## 0. One domain cannot show both the crater and the plume

**This is the first thing to understand, and it is why earlier animations
looked like a featureless ball drifting through empty space.**

The default domain is sized for the plume, which travels ~0.25 m and grows to
~0.035 m. On that box, for the reference case (Fe, 1 pg, 50 km/s → Al, 128³):

| | size | in cells |
|---|---|---|
| cell | 5.7 mm | 1 |
| crater radius | 0.074 mm | **0.013** |
| projectile radius | 0.0031 mm | **0.00054** |

The crater is a *seventy-eighth of one voxel*. The projectile is 1/1800th.
They are not small on screen — they are absent, and no colour map or opacity
setting will recover them. The shock front does not reach even one cell until
≈ 1 µs, by which time the crater stopped growing 50× earlier.

Four orders of magnitude separate the two events, so there are two scenes:

| scene | box | cell | crater | time window | shows |
|---|---|---|---|---|---|
| `impact` | ±0.22 mm | 3.5 µm | **21 cells** | 0 → ~40 ns | projectile, shock sweeping out, crater growing, plasma born |
| `plume` | ±0.36 m | 5.7 mm | 0.013 cells | ~1 → 100 µs | the plasma expanding and going collisionless |
| `plume_comoving` | follows the plume | — | — | ~1 → 100 µs | plume structure only, camera riding its centre of mass |

Render both and cut between them. `write_scene` reports
`cells_across_crater` so you always know which of the two you are looking at,
and the number is written into the `.pvd` header.

```python
from hvi_emp.viz.vti import SCENES, write_scene
write_scene(sc, "scenes/impact", scene="impact")
write_scene(sc, "scenes/plume",  scene="plume")
```

---

## 0b. Let ParaView build the view for you

Getting from a `.pvd` to a legible image takes a dozen correct choices, and
any one of them wrong looks like a bug in the physics. `paraview_scene`
writes them down:

```bash
python -c "
from hvi_emp.viz.paraview_scene import write_scene_script
write_scene_script('scenes/impact/impact.pvd', 'scenes/impact/scene.py',
                   preset='talk')"
pvpython scenes/impact/scene.py
```

The script sets the colour map, the opacity transfer function, the camera,
the time annotation and talk-sized fonts; it checks that the array it is
about to colour by actually exists and says so if not; and its **last act is
`SaveState()`**, so you end up with a `.pvsm` written by your own ParaView
version rather than a hand-rolled one that may not load. After the first run,
just open the `.pvsm`.

Three choices in there are worth knowing about, because each one on its own
produces the flat-blob failure:

* **Colour by `log10_plasma_density`, not `plasma_density`.** A Gaussian
  plume spans ~36 decades inside a single frame.
* **Rescale over the whole time series, not per frame.** Per-frame rescaling
  makes the colour map chase the decaying peak, so a plume that drops two
  decades in density looks identical in every frame.
* **Ramp the opacity.** Flat opacity renders the Gaussian tail as fog and
  hides the core.

`preset="talk"` gives large fonts and a dark background; `"paper"` is
white-background and smaller; `"diagnostic"` is dense.

For a solved crater rather than a drawn one, run M2C and pass
`source="m2c"` — the same scene, with the array names remapped.

---

## 0c. On a workstation with an NVIDIA card

Two separate things could use the GPU here, and only one of them is worth it.

**Rendering: yes, and it needs one line.** ParaView's default volume mapper is
`Smart`, which chooses for itself and on a large volume can fall back to CPU
ray casting — correct, and slow enough that people conclude the data is
broken. The generated script pins the GPU mapper explicitly:

```python
disp.VolumeRenderingMode = "GPU Based"
```

`renderer="index"` additionally tries the **NVIDIA IndeX** plugin, which is
the better mapper above roughly 512³ but ships only with some builds; if it is
absent the script stays on the built-in GPU mapper and says so. Every one of
these assignments is wrapped in a `try`, because the property names move
between ParaView versions and a rendering hint should never take the whole
scene down.

**Generating the `.vti`: no.** A 256³ frame samples in 1.7 s and compresses to
~10 MB. Forty-eight frames is a couple of minutes and well under a gigabyte.
There is nothing here for a GPU to do.

### What actually limits resolution

Host RAM during generation — *not* the card. Peak resident memory measures at
**~175 bytes per cell**, about 44× a single float32 field, because the sampler
works in float64 and holds a stack of intermediates:

| preset | grid | Mcells | peak RAM | one array in VRAM |
|---|---|---|---|---|
| `draft` | 96³ | 0.9 | 0.2 GB | 4 MB |
| `talk` | 256³ | 16.8 | 2.9 GB | 67 MB |
| `high` | 384³ | 56.6 | 9.9 GB | 226 MB |
| `workstation` | 512³ | 134.2 | 23.5 GB | 537 MB |
| `extreme` | 768³ | 453.0 | 79.3 GB | 1.8 GB |

A volume renderer uploads only the array it is colouring by, so even 768³ is
1.8 GB of a 24 GB card. **The GPU has headroom you cannot use**, because the
generator would need 79 GB of host memory to produce the data in the first
place.

`write_scene` estimates this before it starts and refuses rather than letting
the process get OOM-killed halfway through a series:

```bash
python examples/14_impact_scenes.py --list-quality
python examples/14_impact_scenes.py --quality workstation --renderer gpu
```

`--minimal-fields` writes only the two arrays a scene actually renders, which
is worth knowing at high resolution — you can only volume-render one at a
time anyway.

### Headless rendering over ssh

```bash
python examples/14_impact_scenes.py --quality high --movie
pvbatch scenes/impact/scene_impact.py
ffmpeg -framerate 12 -i scenes/impact/frames/impact.%04d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 18 impact.mp4
```

`pvbatch` needs no display. A ParaView built with **EGL** renders offscreen on
the GPU, which is the case that matters on a machine you ssh into; an OSMesa
build will also work but is CPU-bound. The script writes a PNG series rather
than a container, because encoders vary between builds and frames can always
be assembled afterwards.

---

## 0d. Two things the rendered images will make you ask

Both of these came from looking at actual ParaView output, and both answers
are about the model, not the renderer.

### "The crater is too regular, and the projectile looks uniformly deposited"

Correct on both counts, and the second observation is a sharper reading of
the model than the first.

**The crater is a spherical cap by construction.** `crater = solid & (Rsph <
r_c)` — a sphere intersected with the half-space, grown as √t to a
Cour-Palais radius. Depth equals radius, so depth/diameter = 0.5, which is
about right for a ductile metal at normal incidence; but there is no lip, no
spall, no roughness and no flow, because **nothing about the solid target is
solved**.

**The plume is a Gaussian ellipsoid, also by construction.** The "dome" that
appears right after impact is an isosurface of the Stage-2 self-similar
Gaussian. It is not projectile material piling up — it is the analytic plume
shape.

**And the projectile/target mixing really is uniform.** Stage 1 computes how
much of the vapour came from each body — for Fe → Al at 50 km/s,
`w_projectile = 0.464` against `w_target = 0.536` — and then mixes them into
*one* effective material with one temperature and one charge state. There is
no "projectile material here, target material there" anywhere in the reduced
chain. `projectile_fraction` is now written as a field precisely so this is
visible rather than something you have to infer: it renders as a flat colour,
and its provenance tag says UNIFORM in capitals.

Two further things the model knows and does **not** draw:

* `m_melt_target` = 5.3 × 10⁻¹¹ kg — **53× the projectile mass** — melted
  target lining the crater. As a lining on a 74 µm crater that is 0.57 µm
  thick, which is sub-cell even at 512³, so rendering it would be a lie of
  resolution rather than an omission.
* Jetting (`hvi_emp.jetting`) puts material at ~6× the impact speed, and none
  of that geometry is in this picture either.

**If you need a crater with real morphology and genuine projectile/target
mixing, that is what M2C is for.** Its level-set interface tracking keeps the
two materials distinct and gives per-cell `MaterialID`, so you see actual
mixing, jetting and an unforced crater shape. `source="m2c"` in
`write_scene_script` renders it with the same scene.

### "Why is the plume so far from the impact site?"

Because it genuinely is, and this is the single most counter-intuitive number
in the whole model:

| | speed |
|---|---|
| impact | 50.0 km/s |
| plume **drift** downrange (centre of mass) | 25.0 km/s |
| plume **expansion** (its own radius growing) | 3.1 km/s |

The plume does not sit over the crater inflating. It *leaves*, at half the
impact speed, while growing eight times more slowly. After 10 µs it has
travelled 0.25 m downrange and grown to 0.031 m — it is **eight of its own
radii away from the crater it came from**.

That is why an impact-scale box empties so fast. On the reference case the
plume's centre of mass crosses the top of the ±0.22 mm box at **9.2 ns**. The
first version of the impact scene ran to 43.6 ns, so roughly 80% of its
frames had nothing in the domain at all — which looks exactly like a
simulation that has stopped. `write_scene` now ends the impact scene at
`plume_departure_time()` and says so in the run notes.

So: `scene="impact"` is the crater story and it is over in nanoseconds;
`scene="plume"` is where the plasma goes. **If you load both `.pvd` files
into one ParaView session you will see a tiny dot in a huge empty box**,
because the two domains differ by a factor of 1600. Open them separately.

---

## 1. Read this before presenting any image

**The plume is solved. The crater and the shock are drawn.** The reduced
chain integrates the plume; it does not solve the solid target. Rendering
both in the same volume makes them look equally computed, so every field
carries a provenance tag, reproduced in the `.pvd` header:

| field | provenance |
|---|---|
| `plasma_density` | **solved** — Stage-2 self-similar expansion |
| `temperature_eV` | **solved** — Stage-2 energy equation |
| `electron_density` | **solved** — with shell-resolved freeze-out |
| `ionisation` | **solved** — Saha with freeze-out |
| `velocity` | **solved** — self-similar + centre-of-mass drift |
| `density` | solved (plume) + parametric (condensate) |
| `pressure` | solved in the plume, parametric in the target |
| `smoke_density` | **parametric** — two-phase inventory expanded at a chosen fraction of the plasma speed |
| `shock_pressure` | **parametric** — the model's own `P_ic (r/r_ic)^-n` decay law (THEORY §2.6) |
| `material` | **illustrative** — region labels only |
| `projectile` | **illustrative** — rigid sphere before contact |

Concretely: the shock front is a hemisphere at `Us·t` carrying the decay-law
pressure, not a solved wave. The crater grows as `√t` to a Cour-Palais final
radius, not from a computed flow. The condensate expands at 0.35× the plasma
speed — a rendering choice (`smoke_velocity_fraction`), because the two-phase
material arrives at the boiling point with no superheat and so must be
slower and cooler, but *how much* slower is not solved.

**If you need a genuinely solved crater and shock, run M2C** — see
[`SOLVERS.md`](SOLVERS.md) §5. A hydrocode writes its
own output which pySALEPlot converts; that gives you a real continuum
solution for the target, at the cost of a licence application and hours of
compute instead of seconds.

---

## 2. The domain grows with the plume

The plume spans **five decades** in size — tens of microns at the
collisionless transition, a metre by the end of the run. A fixed box cannot
show both.

* **`domain_mode="adaptive"`** (default): the box tracks the plume, so every
  epoch is visible. Origin and Spacing vary per frame; ParaView handles this
  across a `.pvd` without complaint. The animation reads as a zoom-out.
* **`domain_mode="fixed"`**: a true fixed-frame movie sized to the final
  plume — but the first ~80% of frames then show a sub-pixel plume. Use it
  only when you specifically want the "expansion into a static frame" look
  and are willing to start the series late.

The adaptive box never shrinks below a few crater radii, so the impact site
stays resolved in the earliest frames.

Frame times default to **log spacing** (`time_mode="log"`) for the same
reason — linear frames spend 90% of the animation on the last decade. A few
pre-impact frames are prepended so the projectile is seen arriving.

---

## 3. File size

Fields are Float32 and zlib-compressed in VTK's blocked format. For a
128³ grid with the default six fields:

| setting | per frame | 60 frames |
|---|---|---|
| default (6 fields, compressed) | ~1.5 MB | ~90 MB |
| `--all-fields` (11 fields, incl. vectors) | ~4 MB | ~240 MB |
| `--no-compress --all-fields` | ~110 MB | ~6.6 GB |

Compression buys ~10–50× here because the fields are smooth and mostly zero.
Turn it off only if a reader chokes (none should — it is standard VTK).

Grid cost scales as `n³`: 256³ is 8× the size and time of 128³.

---

## 4. A ParaView recipe

1. Open `impact.pvd` → **Apply**.
2. Representation → **Volume**, colour by `plasma_density`.
3. Enable **log scale** on the colour map — the density spans eight decades
   across the series, and a linear map shows a single frame's worth.
4. Add a second copy of the source coloured by `smoke_density` to see the
   condensate trailing the plasma.
5. **Threshold** on `material`: `1` = target solid, `2` = crater cavity,
   `3` = plume, `4` = condensate, `5` = projectile.
6. Press play. The timeline is in seconds; the interesting physics is in the
   first microsecond, which is why the frames are log-spaced.

For the velocity field, use `--all-fields` and add a **Glyph** filter on
`velocity`.

---

## 5. API

```python
from hvi_emp.viz import write_impact_series, ImpactVolume, write_vti

write_impact_series(
    scenario,                 # from run_scenario
    directory="vti_out",
    n_frames=60,
    shape=(128, 128, 128),
    fields=DEFAULT_FIELDS,    # or "all", or any subset
    time_mode="log",          # or "linear"
    domain_mode="adaptive",   # or "fixed"
    compress=True,
    smoke_velocity_fraction=0.35,
)
```

`ImpactVolume` exposes the sampler if you want a single frame:

```python
vol = ImpactVolume(sc, shape=(64, 64, 64))
frame = vol.frame(t=1e-7)          # dict of arrays
write_vti("one.vti", frame, spacing=vol.spacing, origin=vol.origin)
```

`write_vti` is a general-purpose VTK ImageData writer — pass any dict of
`(nx, ny, nz)` scalars or `(nx, ny, nz, 3)` vectors. The test suite
round-trips it byte-for-byte, compressed and uncompressed, including uint8
and vector arrays, with an independent parser rather than a library.


## Why the old animation looked like a static ball

The first `.vti` series defaulted to `domain_mode="adaptive"`, where the box
grows with the plume. Measured from the written files, that put the plume at
**exactly 344 voxels in every frame** while its physical radius grew from
2.1 cm to 11.4 cm — the cell size grew in lockstep. ParaView rendered exactly
what it was given: a ball of constant size. The animation was a zoom-out, not
an expansion.

The default is now `domain_mode="window"`:

* the box is **fixed** (constant `Spacing`, so growth on screen is real), and
* frame times are sampled **linearly over a sub-range** of the history rather
  than logarithmically over all five decades.

Measured end-to-end, the plume's half-max voxel count now grows 60–150× across
a series instead of staying constant.

### The three modes

| mode | box | what you see | use for |
|---|---|---|---|
| `window` (default) | fixed, sized to the window's end | a plume that visibly expands | **animations** |
| `adaptive` | grows with the plume | constant-size ball; a zoom-out | inspecting any single epoch |
| `fixed` | sized to the final plume | sub-voxel plume for ~90% of frames | almost nothing |

### Levers when it still looks small

The plume drifts several radii downrange as it expands, so a box that keeps the
impact site in view is necessarily much bigger than the plume early on.

```python
write_impact_series(sc, "vti_out", shape=(192, 192, 192),   # more cells
                    window_start=0.25,   # open later, plume already larger
                    n_frames=80)
```

`ImpactVolume.notes` reports when the opening frame is under-resolved rather
than leaving you to notice it in ParaView.

### ParaView settings that matter

A Gaussian plume spans orders of magnitude in density, so a **linear** colour
map saturates and renders a solid, featureless ball:

* colour by `plasma_density`, then **Log Scale** the colour map (Edit Color
  Map → Enable Log Scale). This is the single biggest visual improvement.
* Rescale to Data Range Over All Timesteps, or the map rescales every frame
  and the expansion is hidden by the recolouring.
* Volume representation with an opacity ramp starting near zero; the default
  opacity makes the outer 90% of the cloud opaque.

## If the plume renders as a flat blob that barely changes

This is the single most common problem with the output, it has two causes,
and both are now fixed by defaults you can also set explicitly.

### 1. Colour by `log10_plasma_density`, not `plasma_density`

The plume is a Gaussian. Inside **one** frame its density runs from the peak
down through ~36 decades before it underflows float32, and across the series
the peak itself falls ~2 decades more. A linear colour bar over that puts
everything except a handful of central voxels in the bottom colour. What you
see is a uniform blob in the "zero" colour with a faint halo -- regardless of
how much structure the solution actually has.

So the series now writes `log10_plasma_density` and `log10_electron_density`,
floored at `log_floor` (1e-12 kg/m^3) and `log_floor_ne` (1e12 m^-3). A
*linear* colour map on those is a *log* map on the density, and the range is
stable across frames, so you set it once.

In ParaView: colour by `log10_plasma_density`, then **Rescale to Custom Data
Range** and enter the range printed in the run notes (for the reference case,
about `-12` to `-6.5`). Do not use "Rescale to Data Range over All
Timesteps" on the raw density -- that is what produces the flat blob.

### 2. Use `frame_of_reference="plume"` to watch it expand

The plume drifts downrange at roughly half the impact speed while expanding
at about a tenth of that. For the reference 50 km/s case it travels **0.25 m
in the time it grows to 0.035 m radius**. A box that keeps the impact site in
view must therefore be ~0.5 m across to contain the *trajectory*, and the
object inside it is 14x smaller.

Measured on that case at 96^3, in the lab frame the plume occupies **0.03% of
the voxels in the first sampled frame and 1.9% in the last**. That is a small
ball in a large empty box, and the expansion is real but nearly invisible.

```python
vol = ImpactVolume(sc, shape=(128, 128, 128), frame_of_reference="plume")
```

The box then travels with the plume's centre of mass and needs to hold only
its *size*. Same case, same grid: **0.7% to 41%** of the voxels, growing by a
factor of 58 in volume on screen.

| | lab | plume |
|---|---|---|
| box half-width | 0.50 m | 0.14 m |
| fill, first sampled frame | 0.03% | 0.7% |
| fill, last frame | 1.9% | 41% |

The trade-off is explicit: in the co-moving frame the target, crater and
shock are **not drawn**, because they are hundreds of plume radii behind and
would otherwise appear as a slab through the middle of the plume. Use the lab
frame to show the impact geometry, the plume frame to show the expansion.

The box is sized once, from the plume at the end of the sampled window, and
held fixed. A box that resized every frame would keep the plume at constant
apparent size and hide the very thing being animated.

### 3. Do not use `domain_mode="adaptive"` for an animation

This is the one that produces a render which is *literally identical* in
every frame.

`adaptive` resizes the box to the plume at each timestep. The plume then
occupies the same fraction of the image throughout — measured at
**R/box = 0.068 in the first frame and 0.068 in the last**, while the plume
itself grew 3.5×. The animation shows nothing happening because only the
camera is moving.

| `domain_mode` | frame | plume R / box, first → last | fill |
|---|---|---|---|
| `adaptive` | lab | 0.068 → 0.068 | 0.89% → 0.90% |
| `window` | lab | 0.021 → 0.084 | 0.04% → 1.65% |
| `window` | plume | 0.063 → **0.250** | 0.35% → **21.5%** |

`window` is the class default and is now also the example's default;
`adaptive` announces itself in `vol.notes` when selected. It remains
available because it is genuinely useful for inspecting the plume's internal
structure at every scale — just not for showing that it expands.

`examples/08_vti_visualisation.py` now measures the on-screen growth and
says so if the plume barely changes size, naming which of the two settings
is responsible.
