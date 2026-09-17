"""Validated schemas for everything that arrives from a YAML file.

Why this exists
---------------
Configuration is the least-tested surface of a scientific code. A material
with a negative density, an ionisation ladder whose statistical weights are
one short, or a velocity given in km/s where the code wants m/s will not
raise -- it will produce a number. On a cluster that number appears three
hours later in a plot, and nothing distinguishes it from a real result.

So every field is bounds-checked at the moment it is read, with the physics
reason attached to the failure. The checks that matter are not "is this a
float" but the cross-field ones:

* ``len(g_ion) == len(E_ion) + 1`` -- the Saha ladder needs a weight for the
  neutral atom *and* for every ion stage. Off by one and the partition
  function silently drops the last stage.
* ionisation potentials strictly increasing -- removing a second electron
  costs more than the first; a decreasing ladder is a transcription error.
* ``T_melt < T_vap`` -- otherwise the phase-fraction lever rule inverts.
* ``E_cohesive`` within a factor of a few of ``L_vap * m_atom`` -- these are
  the same physics by two routes, so wide disagreement means one is wrong.

Optional dependency
-------------------
pydantic is used when installed, and a dataclass-based validator with the
same rules is used when it is not. The framework's core still needs only
NumPy and SciPy, which is what makes it installable on a locked-down cluster
login node. `USING_PYDANTIC` says which path is active, and both raise
`ConfigError` with the same message text, so callers and tests do not care.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .constants import AMU, EV

__all__ = ["ConfigError", "USING_PYDANTIC", "require_number", "MaterialSpec", "ScenarioSpec",
           "SweepSpec", "ClusterSpec", "RunSpec", "validate_material",
           "validate_scenario", "validate_run", "describe_schema"]


class ConfigError(ValueError):
    """A configuration value is missing, malformed or physically impossible.

    Always carries the offending field and, where there is one, the physical
    reason the value cannot be right.
    """


def require_number(value, where: str):
    """Reject a numeric field that YAML handed back as a string.

    This is not pedantry about types. YAML 1.1 -- which PyYAML implements --
    requires a decimal point *and* a signed exponent for a float, so

        velocity: 50.0e3     ->  the string '50.0e3'
        L_vap: 5e+06         ->  the string '5e+06'
        velocity: 50000.0    ->  the float 50000.0
        L_vap: 5.0e+06       ->  the float 5000000.0

    Every one of those looks like a number to a human reader. Three of the
    eight material files in this repository shipped with the string form and
    nothing complained, because pydantic's lax mode coerces numeric strings
    silently -- so the pydantic and no-pydantic paths disagreed about
    whether the same file was valid.

    Rejecting the string form here makes the two backends behave identically
    and turns a silent coercion into a message that names the fix.
    """
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            raise ConfigError(
                f"{where}: expected a number, got the string "
                f"{value!r}") from None
        raise ConfigError(
            f"{where}: {value!r} is a string, not a number. YAML 1.1 needs "
            f"a decimal point and a signed exponent, so '5e+06' and "
            f"'50.0e3' parse as text while '5.0e+06' and '50000.0' parse as "
            f"floats. Write it as {float(value):g}.")
    return value


try:                                                        # pragma: no cover
    from pydantic import BaseModel, ConfigDict, Field, field_validator, \
        model_validator
    USING_PYDANTIC = True
except ImportError:                                         # pragma: no cover
    USING_PYDANTIC = False


# ---------------------------------------------------------------------------
# The rules, written once and shared by both backends
# ---------------------------------------------------------------------------

#: field -> (minimum, maximum, unit, why the bound exists)
MATERIAL_BOUNDS = {
    "rho0": (100.0, 3.0e4, "kg/m^3",
             "the commonest material typo by far is g/cm^3 left "
             "unconverted, and every such value lands below 23 (osmium, the "
             "densest element, is 22.59 g/cm^3), so the floor is set at 100 "
             "to catch all of them. That does exclude aerogels (1-150 "
             "kg/m^3), which are outside this framework's validity anyway: "
             "the linear Us-Up EOS and the vaporisation model both assume a "
             "dense condensed solid"),
    "c0": (100.0, 2.0e4, "m/s",
           "bulk sound speed; below 100 m/s or above 20 km/s is not a "
           "condensed material"),
    "s": (0.5, 3.0, "-",
          "Us-Up slope; essentially all condensed matter falls in 1.0-2.0, "
          "and s < 1 implies the shock decelerates with pressure"),
    "gamma0": (0.1, 5.0, "-", "Grueneisen parameter"),
    "cv_solid": (50.0, 5.0e3, "J/(kg K)",
                 "Dulong-Petit gives 3R/M, which spans ~130 (W) to ~900 "
                 "(Al) for the elements"),
    "T_melt": (10.0, 1.0e4, "K", "melting point"),
    "T_vap": (20.0, 1.0e4, "K", "1 atm boiling point"),
    "L_fusion": (1.0e3, 1.0e7, "J/kg", "latent heat of fusion"),
    "L_vap": (1.0e4, 1.0e8, "J/kg", "latent heat of vaporisation"),
    "A": (1.0, 300.0, "amu", "mean atomic mass"),
    "Z": (1, 100, "-", "atomic number"),
    "us_up_valid_to": (1.0e3, 1.0e5, "m/s",
                       "upper particle velocity of the linear Us-Up fit"),
}


def _check_material_fields(d: dict) -> dict:
    """Apply every material rule to a plain dict. Raises `ConfigError`.

    Shared by the pydantic and fallback paths so the two cannot drift.
    """
    name = d.get("name", "<unnamed>")

    def bad(field_name, value, why):
        raise ConfigError(f"material {name!r}, field {field_name!r} = "
                          f"{value!r}: {why}")

    for key, (lo, hi, unit, why) in MATERIAL_BOUNDS.items():
        if key not in d:
            bad(key, None, f"missing; required, in {unit} ({why})")
        v = require_number(d[key], f"material {name!r}, field {key!r}")
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or not math.isfinite(v):
            bad(key, v, f"must be a finite number in {unit}")
        if not (lo <= v <= hi):
            bad(key, v, f"outside the physical range {lo}-{hi} {unit} -- "
                        f"{why}")

    E_ion = list(d.get("E_ion", ()))
    g_ion = list(d.get("g_ion", ()))
    if not E_ion:
        bad("E_ion", E_ion, "at least the first ionisation potential is "
                            "required; Saha has nothing to solve without it")
    if len(g_ion) != len(E_ion) + 1:
        bad("g_ion", g_ion,
            f"needs exactly len(E_ion)+1 = {len(E_ion) + 1} entries (a "
            f"statistical weight for the neutral atom and for each ion "
            f"stage), got {len(g_ion)}. An off-by-one here silently drops "
            f"the last stage from the partition function.")
    if any(g <= 0 for g in g_ion):
        bad("g_ion", g_ion, "statistical weights are degeneracies and must "
                            "be positive")
    for i in range(1, len(E_ion)):
        if E_ion[i] <= E_ion[i - 1]:
            bad("E_ion", E_ion,
                f"stage {i + 1} ({E_ion[i] / EV:.4g} eV) is not greater than "
                f"stage {i} ({E_ion[i - 1] / EV:.4g} eV). Removing a further "
                f"electron from a more positive ion always costs more, so a "
                f"non-increasing ladder is a transcription error.")
    if any(e <= 0 for e in E_ion):
        bad("E_ion", E_ion, "ionisation potentials must be positive")

    if d["T_melt"] >= d["T_vap"]:
        bad("T_melt", d["T_melt"],
            f"must be below T_vap ({d['T_vap']} K); the melt/vapour lever "
            f"rule inverts otherwise")
    if d.get("E_cohesive", 0) <= 0:
        bad("E_cohesive", d.get("E_cohesive"), "must be positive")
    if d.get("work_function", 0) <= 0:
        bad("work_function", d.get("work_function"), "must be positive")

    # Cross-check: cohesive energy per atom against latent heat of
    # vaporisation per atom. For a solid that vaporises to free atoms these
    # measure the same binding by two independent routes, so a large
    # disagreement means one of them is wrong -- almost always a unit error.
    #
    # It is a genuinely sharp test on the built-in library: Al 1.15, Cu 1.12,
    # Fe 1.21, W 1.04, SiO2 1.87, Dolomite 2.51, Olivine 3.00.
    #
    # It does *not* apply to molecular solids. A polymer does not vaporise
    # into atoms; it depolymerises into fragments, so the cohesive energy of
    # a covalent backbone bond (~4 eV) and the enthalpy to release a fragment
    # (~0.4 eV per mean atom for Kapton) are different physics, not
    # inconsistent numbers. Hence `bonding`, which must be declared rather
    # than guessed: a metal mislabelled "molecular" would silently lose the
    # check that catches its unit errors.
    bonding = d.get("bonding", "atomic")
    if bonding not in ("atomic", "molecular"):
        bad("bonding", bonding,
            "must be 'atomic' (vaporises to free atoms: metals, oxides, "
            "silicates) or 'molecular' (depolymerises to fragments: "
            "polymers, ices)")
    e_vap_atom = d["L_vap"] * d["A"] * AMU
    ratio = d["E_cohesive"] / e_vap_atom if e_vap_atom > 0 else float("inf")
    lo, hi = (0.2, 5.0) if bonding == "atomic" else (0.2, 50.0)
    if not (lo < ratio < hi):
        bad("E_cohesive", d["E_cohesive"] / EV,
            f"eV/atom is {ratio:.2f}x the latent heat of vaporisation per "
            f"atom ({e_vap_atom / EV:.3g} eV), outside the {lo}-{hi} range "
            f"expected for bonding='{bonding}'. These measure the same "
            f"binding by different routes; check the unit on E_cohesive or "
            f"L_vap, or declare bonding='molecular' if this material "
            f"depolymerises rather than vaporising to free atoms.")
    return d


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

if USING_PYDANTIC:                                          # pragma: no cover

    class MaterialSpec(BaseModel):
        """One material, as it appears in `conf/materials/<name>.yaml`.

        Energies are given in eV in the file (that is how the literature
        quotes them) and converted to joules on load, so the `_eV` aliases
        below are the *file* spelling and the SI names are the attributes.
        """
        model_config = ConfigDict(extra="forbid", frozen=True)

        name: str = Field(min_length=1)
        bonding: str = "atomic"
        rho0: float
        c0: float
        s: float
        gamma0: float
        cv_solid: float
        T_melt: float
        T_vap: float
        L_fusion: float
        L_vap: float
        E_cohesive: float
        A: float
        Z: int
        E_ion: tuple[float, ...]
        g_ion: tuple[float, ...]
        work_function: float
        us_up_valid_to: float = 1.5e4

        @model_validator(mode="after")
        def _physics(self):
            _check_material_fields(self.model_dump())
            return self

    class ScenarioSpec(BaseModel):
        """One impact to simulate."""
        model_config = ConfigDict(extra="forbid")

        projectile: str = "Fe"
        target: str = "Al"
        mass: float | None = 1e-12
        diameter: float | None = None
        velocity: float = 50e3
        angle_deg: float = 0.0
        expansion_model: str = "1T"
        aspect0: float = 0.5
        t_end: float | None = None
        n_decay: float = 2.0
        core_scale: float = 1.0
        plume_expansion_factor: float = 3.0
        r_sensor: float = 0.30
        theta_deg: float = 90.0
        closure: str = "debye"
        separation_factor: float = 1.0
        bands: tuple[float, ...] = (315e6, 916e6)

        @field_validator("expansion_model")
        @classmethod
        def _model(cls, v):
            if v not in ("1T", "2T"):
                raise ConfigError(
                    f"expansion_model must be '1T' or '2T', got {v!r}")
            return v

        @model_validator(mode="after")
        def _physics(self):
            _check_scenario_fields(self.model_dump())
            return self

else:

    @dataclass(frozen=True)
    class MaterialSpec:                                     # type: ignore
        """Fallback material schema; identical rules, no pydantic."""
        name: str
        rho0: float
        c0: float
        s: float
        gamma0: float
        cv_solid: float
        T_melt: float
        T_vap: float
        L_fusion: float
        L_vap: float
        E_cohesive: float
        A: float
        Z: int
        E_ion: tuple
        g_ion: tuple
        work_function: float
        us_up_valid_to: float = 1.5e4
        bonding: str = "atomic"

        def __post_init__(self):
            _check_material_fields({k: getattr(self, k)
                                    for k in self.__dataclass_fields__})

        def model_dump(self) -> dict:
            return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @dataclass
    class ScenarioSpec:                                     # type: ignore
        """Fallback scenario schema; identical rules, no pydantic."""
        projectile: str = "Fe"
        target: str = "Al"
        mass: float | None = 1e-12
        diameter: float | None = None
        velocity: float = 50e3
        angle_deg: float = 0.0
        expansion_model: str = "1T"
        aspect0: float = 0.5
        t_end: float | None = None
        n_decay: float = 2.0
        core_scale: float = 1.0
        plume_expansion_factor: float = 3.0
        r_sensor: float = 0.30
        theta_deg: float = 90.0
        closure: str = "debye"
        separation_factor: float = 1.0
        bands: tuple = (315e6, 916e6)

        def __post_init__(self):
            if self.expansion_model not in ("1T", "2T"):
                raise ConfigError("expansion_model must be '1T' or '2T', "
                                  f"got {self.expansion_model!r}")
            _check_scenario_fields(
                {k: getattr(self, k) for k in self.__dataclass_fields__})

        def model_dump(self) -> dict:
            return {k: getattr(self, k) for k in self.__dataclass_fields__}


def resolve_projectile_size(d: dict) -> dict:
    """Let a scenario be given by `diameter` OR `mass`, never both.

    Debris and micrometeoroid populations are quoted by *size* -- a "1 mm
    particle" -- while the physics needs mass. Requiring the user to do
    rho*pi*D^3/6 by hand is an invitation to get it wrong by 10^3 somewhere,
    and a mass that is wrong by 10^3 produces a perfectly plausible-looking
    result.

    Specifying both is rejected rather than silently preferring one: if they
    disagree, there is no way to know which the user meant.
    """
    mass, diam = d.get("mass"), d.get("diameter")
    if diam is None:
        if mass is None:
            raise ConfigError(
                "scenario needs either 'mass' [kg] or 'diameter' [m]")
        return d
    if mass is not None and d.get("_mass_explicit", True) and \
            "mass" in d and d["mass"] is not None and diam is not None \
            and d.get("__both__", False):
        raise ConfigError("give 'mass' or 'diameter', not both")
    require_number(diam, "scenario field 'diameter'")
    if not (1e-9 <= diam <= 1.0):
        raise ConfigError(
            f"diameter = {diam!r} m is outside 1 nm - 1 m. Note the unit is "
            f"METRES: a 1 mm particle is 1.0e-3, not 1.0.")
    from .materials import get_material
    try:
        rho = get_material(d.get("projectile", "Fe")).rho0
    except KeyError as exc:
        raise ConfigError(f"scenario: {exc}") from None
    d = dict(d)
    d["mass"] = rho * math.pi * diam ** 3 / 6.0
    return d


def _check_scenario_fields(d: dict) -> dict:
    """Physics bounds on a scenario. Raises `ConfigError`."""
    for key in ("velocity", "mass", "angle_deg", "n_decay", "core_scale",
                "plume_expansion_factor", "r_sensor", "theta_deg",
                "aspect0"):
        if d.get(key) is not None:
            require_number(d[key], f"scenario field {key!r}")
    v = d.get("velocity", 0.0)
    if not (10.0 <= v <= 3.0e5):
        raise ConfigError(
            f"velocity = {v!r} m/s is outside 10 m/s - 300 km/s. Note the "
            f"unit is metres per second: a value near 50 almost certainly "
            f"means 50 km/s and should be 50e3.")
    if v < 1.0e3:
        raise ConfigError(
            f"velocity = {v} m/s. Below ~1 km/s nothing vaporises and the "
            f"framework has no plasma to model; if you meant km/s, multiply "
            f"by 1000.")
    m = d.get("mass", 0.0)
    if not (1e-21 <= m <= 1.0):
        raise ConfigError(
            f"mass = {m!r} kg is outside 1e-21 - 1 kg. The framework is "
            f"calibrated for micrometeoroid and debris masses, roughly "
            f"1e-15 to 1e-6 kg.")
    a = d.get("angle_deg", 0.0)
    if not (0.0 <= a < 90.0):
        raise ConfigError(
            f"angle_deg = {a!r} must be in [0, 90) from the surface normal; "
            f"90 degrees is a grazing miss, not an impact.")
    a0 = d.get("aspect0")
    if a0 is not None and not (0.02 <= a0 <= 5.0):
        raise ConfigError(
            f"aspect0 = {a0!r} is outside 0.02-5. It is the initial plume "
            f"R_z/R_r; below ~0.02 the seed is a disc thinner than one "
            f"Gaussian sigma and the self-similar model stops being "
            f"meaningful. 1.0 gives an isotropic plume (a fixed point of "
            f"the equations); the default 0.5 matches the measured cosine "
            f"angular law.")
    if d.get("closure") not in ("debye", "calibrated", "fletcher", None):
        raise ConfigError(
            f"closure must be 'debye' (conservative), 'calibrated' (xi=15 "
            f"from Close 2013) or 'fletcher' (aggressive), got "
            f"{d['closure']!r}. These differ by ~184x in EMP amplitude -- "
            f"see docs/EMP_UNCERTAINTY.md before choosing.")
    for f in d.get("bands") or ():
        if not (1e3 <= f <= 1e15):
            raise ConfigError(f"band frequency {f!r} Hz is implausible")
    t_end = d.get("t_end")
    if t_end is not None and not (1e-12 <= t_end <= 1.0):
        raise ConfigError(
            f"t_end = {t_end!r} s is outside 1 ps - 1 s")
    return d


# ---------------------------------------------------------------------------
# Sweep / cluster / run -- plain dataclasses either way; these carry no
# physics, so pydantic buys nothing beyond what is checked here.
# ---------------------------------------------------------------------------

@dataclass
class SweepSpec:
    """A parameter sweep, expanded into independent scenarios."""
    parameter: str = "velocity"
    values: tuple = ()
    start: float | None = None
    stop: float | None = None
    num: int | None = None
    log: bool = False

    def resolve(self) -> list:
        """The explicit list of values this sweep covers."""
        if self.values:
            vals = list(self.values)
        elif None not in (self.start, self.stop, self.num):
            if self.num < 1:
                raise ConfigError(f"sweep num = {self.num} must be >= 1")
            if self.log:
                if self.start <= 0 or self.stop <= 0:
                    raise ConfigError(
                        "log sweep needs strictly positive start and stop")
                vals = [self.start * (self.stop / self.start)
                        ** (i / max(self.num - 1, 1))
                        for i in range(self.num)]
            else:
                step = ((self.stop - self.start) / max(self.num - 1, 1))
                vals = [self.start + i * step for i in range(self.num)]
        else:
            raise ConfigError(
                "sweep needs either 'values', or all of 'start', 'stop' and "
                "'num'")
        if not vals:
            raise ConfigError("sweep resolved to zero values")
        return vals


@dataclass
class ClusterSpec:
    """Scheduler resources for one job.

    `cpus_per_task` is the number the process pool will actually see. It is
    deliberately explicit rather than inferred: a job that requests 4 CPUs
    and then forks 48 workers is the standard way to be thrown off a shared
    cluster.
    """
    backend: str = "slurm"
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    time: str = "01:00:00"
    cpus_per_task: int = 8
    mem_per_cpu: str = "2G"
    gpus: int = 0
    gpu_type: str | None = None
    nodes: int = 1
    ntasks: int = 1
    array_throttle: int | None = None
    modules: tuple = ()
    conda_env: str | None = None
    venv: str | None = None
    extra_directives: tuple = ()

    def __post_init__(self):
        if self.backend not in ("slurm", "local"):
            raise ConfigError(
                f"backend {self.backend!r} not supported. 'slurm' and "
                f"'local' are implemented; PBS/SGE/LSF can be added by "
                f"subclassing hvi_emp.cluster.Scheduler.")
        if self.cpus_per_task < 1:
            raise ConfigError("cpus_per_task must be >= 1")
        if self.gpus < 0:
            raise ConfigError("gpus must be >= 0")
        if not _valid_walltime(self.time):
            raise ConfigError(
                f"time = {self.time!r} is not a Slurm walltime. Use "
                f"HH:MM:SS, D-HH:MM:SS, MM:SS or an integer number of "
                f"minutes.")


def _valid_walltime(t: str) -> bool:
    """Accept Slurm's walltime spellings, reject anything else."""
    if not isinstance(t, str) or not t:
        return False
    body = t.split("-", 1)[1] if "-" in t else t
    if "-" in t:
        days = t.split("-", 1)[0]
        if not days.isdigit():
            return False
    parts = body.split(":")
    if len(parts) > 3:
        return False
    return all(p.isdigit() for p in parts)


@dataclass
class RunSpec:
    """A complete job: what to simulate, how to sweep it, where to run it."""
    scenario: Any = field(default_factory=lambda: ScenarioSpec())
    sweep: SweepSpec | None = None
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    output_dir: str = "results"
    seed: int = 0
    tag: str = "run"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _unwrap(exc: BaseException) -> str:
    """Pull the useful sentence out of a pydantic ValidationError.

    pydantic wraps a raised exception in "1 validation error for X / Value
    error, <message> [type=value_error, input_value={...}]" plus a docs URL.
    The message we wrote is in there, but a user reading a failed cluster job
    log should not have to find it. This extracts it, and falls back to the
    full text when the shape is not what we expect.
    """
    text = str(exc)
    marker = "Value error, "
    if marker in text:
        text = text.split(marker, 1)[1]
    for cut in (" [type=", "\nFor further information"):
        if cut in text:
            text = text.split(cut, 1)[0]
    return text.strip()


def validate_material(d: dict) -> MaterialSpec:
    """Build a validated `MaterialSpec` from a raw dict.

    Accepts the file spelling, in which energies carry an ``_eV`` suffix, and
    converts to joules. Unknown keys are rejected rather than ignored --
    a typo'd key that is silently dropped leaves the default in place, which
    is the hardest kind of configuration bug to see.
    """
    d = dict(d)
    for src, dst in (("E_cohesive_eV", "E_cohesive"),
                     ("work_function_eV", "work_function")):
        if src in d:
            d[dst] = float(d.pop(src)) * EV
    if "E_ion_eV" in d:
        d["E_ion"] = tuple(float(x) * EV for x in d.pop("E_ion_eV"))
    if "E_ion" in d:
        d["E_ion"] = tuple(float(x) for x in d["E_ion"])
    if "g_ion" in d:
        d["g_ion"] = tuple(float(x) for x in d["g_ion"])

    known = set(MaterialSpec.__dataclass_fields__) \
        if not USING_PYDANTIC else set(MaterialSpec.model_fields)
    unknown = set(d) - known
    if unknown:
        raise ConfigError(
            f"material {d.get('name', '?')!r}: unknown key(s) "
            f"{sorted(unknown)}. Known keys are {sorted(known)}. (A typo'd "
            f"key would otherwise be dropped and its default used.)")
    #: Fields with a schema default; everything else must be stated.
    OPTIONAL = {"us_up_valid_to", "bonding"}
    missing = known - set(d) - OPTIONAL
    if missing:
        raise ConfigError(
            f"material {d.get('name', '?')!r}: missing required key(s) "
            f"{sorted(missing)}")
    try:
        return MaterialSpec(**d)
    except ConfigError:
        raise
    except Exception as exc:                                # noqa: BLE001
        raise ConfigError(
            f"material {d.get('name', '?')!r}: {_unwrap(exc)}") from exc


def validate_scenario(d: dict) -> ScenarioSpec:
    """Build a validated `ScenarioSpec` from a raw dict.

    Accepts the projectile size as either ``mass`` [kg] or ``diameter`` [m].
    Exactly one of them must be *stated*; `diameter` is converted to mass
    using the projectile material's density.
    """
    d = dict(d)
    # The schema default gives mass=1e-12, so "both present" cannot be
    # detected from the merged dict alone -- only the raw input knows what
    # the user actually wrote.
    if d.get("diameter") is not None and d.get("mass") is not None:
        raise ConfigError(
            "scenario: give 'mass' (kg) or 'diameter' (m), not both -- if "
            "they disagree there is no way to know which you meant. The "
            "default mass is 1e-12 kg, so set 'mass: null' when using "
            "'diameter'.")
    if d.get("diameter") is not None:
        d = resolve_projectile_size(d)
    elif "mass" in d and d["mass"] is None:
        raise ConfigError(
            "scenario: 'mass' is null and no 'diameter' was given. Set one "
            "of them -- 'mass' in kg, or 'diameter' in metres (converted "
            "with the projectile material's density).")
    if "bands" in d and d["bands"] is not None:
        d["bands"] = tuple(float(x) for x in d["bands"])
    try:
        return ScenarioSpec(**d)
    except ConfigError:
        raise
    except Exception as exc:                                # noqa: BLE001
        raise ConfigError(f"scenario: {_unwrap(exc)}") from exc


def validate_run(d: dict) -> RunSpec:
    """Build a validated `RunSpec` from a nested config dict."""
    d = dict(d or {})
    scen = validate_scenario(d.get("scenario") or {})
    sweep = d.get("sweep")
    sweep_spec = None
    if sweep:
        try:
            sweep_spec = SweepSpec(**sweep)
        except TypeError as exc:
            raise ConfigError(f"sweep: {exc}") from exc
        sweep_spec.resolve()                    # fail now, not on the cluster
    try:
        cluster = ClusterSpec(**(d.get("cluster") or {}))
    except TypeError as exc:
        raise ConfigError(f"cluster: {exc}") from exc
    return RunSpec(scenario=scen, sweep=sweep_spec, cluster=cluster,
                   output_dir=str(d.get("output_dir", "results")),
                   seed=int(d.get("seed", 0)),
                   tag=str(d.get("tag", "run")))


def describe_schema() -> str:
    """Human-readable summary of the material bounds, for `--help`-style use."""
    lines = ["Material fields and their accepted ranges:", ""]
    for k, (lo, hi, unit, why) in MATERIAL_BOUNDS.items():
        lines.append(f"  {k:<16} {lo:>10g} .. {hi:<10g} {unit}")
        lines.append(f"  {'':<16} {why}")
    lines += ["", "  E_ion / E_ion_eV  strictly increasing, positive",
              "  g_ion             len(E_ion) + 1 positive weights",
              f"  backend           validated by pydantic: {USING_PYDANTIC}"]
    return "\n".join(lines)
