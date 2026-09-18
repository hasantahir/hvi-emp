## Fix black frames in the composite render

Reported from a real `pvbatch` run: the animation produced frames with the
text overlays drawn correctly and **no geometry at all**. The script ran,
the readers loaded, and every pixel was background.

Three causes, all in `hvi_emp/viz/composite.py`.

### 1. The clipping range was never reset

The camera was positioned from each chapter's nominal zoom in metres, with
the focal point assumed to be the origin, and the near/far planes left at
whatever the previous chapter had set — with `_DisableFirstRenderCameraReset()`
ensuring nothing corrected it.

The chapters span nine decades (5 nm → 400 mm). The stale clipping range
from one chapter cuts the next one away entirely, so the scene renders
perfectly and shows nothing.

`ResetCamera()` now runs first for each chapter. It frames whatever is
actually visible and recomputes the clipping range, so it is correct
regardless of the data's units or whether it is centred on the origin. The
nominal zoom is now only a stylistic pull-back and **cannot make the data
vanish**.

### 2. Every chapter was forced to volume rendering

```python
d = Show(r, view, "UniformGridRepresentation")
d.Representation = "Volume"
```

That is an ImageData mode. The MD chapter is `.vtp` PolyData — one point per
atom — so it could never volume-render. This is specifically why the LAMMPS
frames showed their labels and none of their atoms.

Representation is now chosen from `GetDataClassName()`: Volume for
ImageData, Point Gaussian for points, with the radius derived from the
bounds diagonal. A radius fixed in metres is invisible at 5 nm and fills the
screen at 400 mm.

### 3. The handover card overlapped the banner

Both were drawn in the top strip at 1920×1080 and became unreadable. Moved
to Lower Center.

---

## Testing

Six tests added in `tests/test_chain.py`, asserting:

- `ResetCamera()` is present **and precedes** any manual camera move in the
  chapter loop
- `Volume` sits inside the ImageData branch rather than being unconditional
- the point radius derives from the data bounds, not a constant
- the handover card and banner are in different screen regions
- the generated script parses as Python
- the "not one simulation" caveat still cannot be lost

Full suite: **687 passed, 13 skipped, 0 failed.**

## What is not verified

**This has not been run against ParaView.** There is none in the environment
these changes were made in, and there never has been — the generated scripts
have only ever been checked by construction. Whether it renders is unknown
until `pvbatch` runs it.

If frames are still black after this, the next thing to check is whether the
readers contain any points at all:

```python
r.GetDataInformation().GetNumberOfPoints()
```

Empty `.pvd` files would produce identical frames, and that failure would
live upstream in the converters, not here.
