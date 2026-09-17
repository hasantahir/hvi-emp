"""Ejecta identification and fragment clustering for particle data.

What this is for
----------------
Two published visualisations of hypervelocity impact look completely
different, and the difference is not cosmetic:

* an **atomistic dust cloud** -- millions of points, coloured by temperature,
  the target a dense continuum of particles and the ejecta a fine spray; and
* a **fragment cloud** -- a few hundred *discrete objects* with sizes and
  shapes, flying off a perforated plate.

They are the same data at two levels of description. The second is the first
after you have asked *which atoms are stuck to which*. This module does that
asking: it separates ejecta from the bulk, groups the ejecta into connected
fragments, and reports a fragment size distribution -- which is a physical
result, not just a rendering aid.

The method, and its one free parameter
--------------------------------------
Fragments are **connected components under a distance cutoff**: two atoms are
in the same fragment if they are closer than `cutoff`, and fragments are the
transitive closure of that. This is the standard cluster analysis used on MD
impact data and it has exactly one free parameter, which changes the answer.

Too small and a solid fragment shatters into singletons because of thermal
displacement; too large and the whole debris cloud merges into one object.
The defensible choice is "just beyond the first neighbour shell", so the
default is derived from the data itself -- `CUTOFF_FACTOR` times the median
nearest-neighbour distance in the frame -- rather than hard-coded for one
lattice. `FragmentSet.cutoff` records what was used, and
`cutoff_sensitivity()` exists so you can show that your conclusion does not
rest on the choice.

What this does not do
---------------------
It does not know about bonding, only distance. A hot, expanded, still-bound
liquid droplet and a loose cluster of vapour atoms at the same mean spacing
are indistinguishable to it. Above the melt point especially, read
"fragment" as "connected region", and if the distinction matters, check
`cutoff_sensitivity()` before quoting a size distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Cutoff as a multiple of the median nearest-neighbour distance.
#:
#: 1.4 sits between the first and second neighbour shells for both fcc
#: (second shell at 1.41x) and bcc (second at 1.15x, third at 1.63x). For bcc
#: it therefore includes the second shell, which is the right call: bcc's
#: first and second shells are only 15% apart and splitting them would break
#: intact bcc fragments apart on thermal motion alone.
CUTOFF_FACTOR = 1.4

#: Above this many atoms, `query_pairs` builds a pair list large enough to
#: matter (~45 pairs/atom in a dense solid, 8 bytes each). Clustering is
#: applied to *ejecta only* precisely to stay below it.
PAIR_WARN_ATOMS = 2_000_000


@dataclass
class FragmentSet:
    """Connected fragments found in a set of particles."""

    #: Index into the original atom array for each clustered particle.
    index: np.ndarray
    #: Fragment id per clustered particle, 0..n_fragments-1, ordered so that
    #: 0 is the largest.
    labels: np.ndarray
    sizes: np.ndarray                 # atoms per fragment
    masses: np.ndarray                # kg per fragment
    com: np.ndarray                   # (nf, 3) centre of mass [m]
    velocity: np.ndarray              # (nf, 3) centre-of-mass velocity [m/s]
    radius: np.ndarray                # equivalent-sphere radius [m]
    cutoff: float                     # the distance used [m]
    notes: list = field(default_factory=list)

    @property
    def n_fragments(self) -> int:
        return int(self.sizes.size)

    @property
    def speed(self) -> np.ndarray:
        """Centre-of-mass speed per fragment [m/s]."""
        return np.linalg.norm(self.velocity, axis=-1)

    def summary(self) -> str:
        if self.n_fragments == 0:
            return "no fragments"
        singles = int(np.sum(self.sizes == 1))
        return (f"{self.n_fragments} fragments from {int(self.sizes.sum())} "
                f"ejecta atoms; largest {int(self.sizes[0])} atoms "
                f"({self.masses[0]:.3e} kg, r_eq {self.radius[0]*1e9:.2f} nm), "
                f"{singles} single atoms "
                f"({100.0*singles/self.n_fragments:.0f}% by count, "
                f"{100.0*singles/max(self.sizes.sum(),1):.1f}% by mass); "
                f"cutoff {self.cutoff*1e10:.2f} A")


def nearest_neighbour_distance(pos: np.ndarray,
                               sample: int = 20000) -> float:
    """Median nearest-neighbour distance [m].

    Sampled rather than exhaustive: the median is stable long before every
    atom has been examined, and this is called on frames with millions of
    particles.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(pos, dtype=float)
    if pos.shape[0] < 2:
        return 0.0
    tree = cKDTree(pos)
    if pos.shape[0] > sample:
        rng = np.random.default_rng(0)     # deterministic: a report is a report
        probe = pos[rng.choice(pos.shape[0], sample, replace=False)]
    else:
        probe = pos
    d, _ = tree.query(probe, k=2)
    return float(np.median(d[:, 1]))


def identify_ejecta(pos: np.ndarray, vel: np.ndarray,
                    surface: float | None = None,
                    axis: int = 2, margin: float = 0.0,
                    min_speed: float = 0.0) -> np.ndarray:
    """Boolean mask of particles that have left the target.

    The criterion is deliberately simple and geometric: **above the original
    free surface, and moving away from it**. Anything more elaborate -- a
    binding-energy test, say -- needs a potential this module does not have.

    Parameters
    ----------
    pos, vel : (n, 3) arrays [m], [m/s]
    surface : float or None
        Position of the undisturbed free surface along `axis`. ``None``
        estimates it as the 99.5th percentile of the initial-looking
        material, which is right for a flat target filling the box and wrong
        for anything else -- pass it explicitly when you know it.
    axis : int
        Surface normal. 2 (z) by default.
    margin : float
        Require the particle to be this far above the surface as well, which
        suppresses surface atoms merely vibrating.
    min_speed : float
        Require at least this outward speed [m/s].

    Returns
    -------
    (n,) bool
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    if pos.shape != vel.shape:
        raise ValueError(f"pos {pos.shape} and vel {vel.shape} must match")
    if surface is None:
        surface = float(np.percentile(pos[:, axis], 99.5))
    out = (pos[:, axis] > surface + margin) & (vel[:, axis] > min_speed)
    return out


def cluster(pos: np.ndarray, cutoff: float | None = None,
            cutoff_factor: float = CUTOFF_FACTOR) -> tuple:
    """Connected-component clustering under a distance cutoff.

    Returns ``(labels, cutoff)``. Labels are 0-based and arbitrary in order;
    `analyse_fragments` re-orders them by size.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), float(cutoff or 0.0)
    if n == 1:
        return np.zeros(1, dtype=int), float(cutoff or 0.0)

    if cutoff is None:
        d_nn = nearest_neighbour_distance(pos)
        cutoff = cutoff_factor * d_nn if d_nn > 0 else 0.0
    if cutoff <= 0:
        return np.arange(n), 0.0

    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if pairs.size == 0:
        return np.arange(n), float(cutoff)
    g = coo_matrix((np.ones(len(pairs), dtype=np.int8),
                    (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels, float(cutoff)


def analyse_fragments(pos: np.ndarray, vel: np.ndarray, m_atom: float,
                      mask: np.ndarray | None = None,
                      cutoff: float | None = None,
                      cutoff_factor: float = CUTOFF_FACTOR,
                      density: float | None = None) -> FragmentSet:
    """Cluster the selected particles and summarise each fragment.

    Parameters
    ----------
    pos, vel : (n, 3) [m], [m/s]
    m_atom : float
        Mass per particle [kg].
    mask : (n,) bool or None
        Which particles to cluster -- normally the ejecta. ``None`` uses all,
        which on a full frame means the target is returned as one enormous
        fragment.
    cutoff : float or None
        Distance defining "connected" [m]. ``None`` derives it from the
        **whole frame**, not from the masked subset -- see the note below.
    density : float or None
        Bulk density [kg/m^3] for the equivalent-sphere radius. ``None``
        derives it from the cutoff, which is cruder but needs no extra input.

    Notes
    -----
    **The cutoff is derived from the full frame on purpose.** "Connected"
    means "within a lattice spacing", which is a property of the *material*,
    not of whichever particles you happened to select. Deriving it from the
    ejecta alone looks equivalent and is not: ejecta are already dispersed,
    so their median nearest-neighbour distance is larger than the lattice's
    and it *keeps growing* as the debris flies apart. On a real frame here
    that was 3.07 A against the bulk's 2.67 A, a 15% inflation at 0.2 ps and
    rising. A cutoff that grows with the cloud progressively merges fragments
    over a time series -- the opposite of what the analysis is for.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    idx = (np.flatnonzero(mask) if mask is not None
           else np.arange(pos.shape[0]))
    notes: list = []

    if idx.size == 0:
        return FragmentSet(index=idx, labels=np.zeros(0, dtype=int),
                           sizes=np.zeros(0, dtype=int),
                           masses=np.zeros(0), com=np.zeros((0, 3)),
                           velocity=np.zeros((0, 3)), radius=np.zeros(0),
                           cutoff=float(cutoff or 0.0),
                           notes=["no particles selected"])
    if idx.size > PAIR_WARN_ATOMS:
        notes.append(
            f"clustering {idx.size} particles; the pair list is the memory "
            f"cost here and grows with density, not just count. Cluster "
            f"ejecta only (pass `mask`) if this is slow.")

    if cutoff is None:
        d_nn = nearest_neighbour_distance(pos)      # FULL frame, see Notes
        cutoff = cutoff_factor * d_nn if d_nn > 0 else None
        if cutoff:
            notes.append(
                f"cutoff {cutoff*1e10:.2f} A = {cutoff_factor} x the "
                f"{d_nn*1e10:.2f} A median nearest-neighbour distance of the "
                f"whole frame (not of the selection -- see analyse_fragments "
                f"Notes)")

    p, v = pos[idx], vel[idx]
    raw, used = cluster(p, cutoff=cutoff, cutoff_factor=cutoff_factor)

    # Re-label by descending size so fragment 0 is always the largest.
    counts = np.bincount(raw)
    order = np.argsort(counts)[::-1]
    remap = np.empty(counts.size, dtype=int)
    remap[order] = np.arange(counts.size)
    labels = remap[raw]
    sizes = counts[order]

    nf = sizes.size
    masses = sizes * float(m_atom)
    com = np.stack([np.bincount(labels, weights=p[:, d], minlength=nf) / sizes
                    for d in range(3)], axis=-1)
    vcm = np.stack([np.bincount(labels, weights=v[:, d], minlength=nf) / sizes
                    for d in range(3)], axis=-1)

    if density is None and used > 0:
        # one atom per cutoff-sized cell is crude but needs nothing else
        density = float(m_atom) / (used / CUTOFF_FACTOR) ** 3
    radius = ((3.0 * masses / (4.0 * np.pi * max(density or 1.0, 1e-30)))
              ** (1.0 / 3.0))

    fs = FragmentSet(index=idx, labels=labels, sizes=sizes, masses=masses,
                     com=com, velocity=vcm, radius=radius, cutoff=used,
                     notes=notes)
    fs.notes.append(fs.summary())
    if nf and sizes[0] > 0.5 * sizes.sum():
        fs.notes.append(
            f"fragment 0 holds {100.0*sizes[0]/sizes.sum():.0f}% of the "
            f"selected atoms. If you clustered a whole frame that is just "
            f"the target; if you clustered ejecta it is a spall plate or a "
            f"still-attached lip, and the surface estimate may be too low.")
    return fs


def size_distribution(frags: FragmentSet) -> dict:
    """Cumulative fragment size distribution.

    Returns ``{"mass", "n_ge", "mass_fraction_ge"}``: for each distinct
    fragment mass, the number of fragments at least that large and the
    fraction of ejected mass they carry. This is the form that is compared
    with Grady-Kipp and Mott distributions, and the form in which a
    fragmentation result is normally reported.
    """
    if frags.n_fragments == 0:
        return {"mass": np.zeros(0), "n_ge": np.zeros(0),
                "mass_fraction_ge": np.zeros(0)}
    m = np.sort(frags.masses)[::-1]
    total = m.sum()
    return {"mass": m,
            "n_ge": np.arange(1, m.size + 1),
            "mass_fraction_ge": np.cumsum(m) / total}


def cutoff_sensitivity(pos: np.ndarray, vel: np.ndarray, m_atom: float,
                       mask: np.ndarray | None = None,
                       factors=(1.1, 1.2, 1.4, 1.6, 2.0)) -> list:
    """Re-cluster at several cutoffs and report how much the answer moves.

    The cutoff is this module's only free parameter, so any number derived
    from a single clustering should be shown alongside this. A fragment count
    that halves between 1.2 and 1.6 is telling you the fragments are not
    well separated, which is itself worth knowing.
    """
    out = []
    for f in factors:
        fs = analyse_fragments(pos, vel, m_atom, mask=mask, cutoff_factor=f)
        out.append({
            "factor": float(f),
            "cutoff": fs.cutoff,
            "n_fragments": fs.n_fragments,
            "largest_atoms": int(fs.sizes[0]) if fs.n_fragments else 0,
            "singletons": int(np.sum(fs.sizes == 1)) if fs.n_fragments else 0,
        })
    return out


__all__ = ["FragmentSet", "identify_ejecta", "cluster", "analyse_fragments",
           "size_distribution", "cutoff_sensitivity",
           "nearest_neighbour_distance", "CUTOFF_FACTOR"]
