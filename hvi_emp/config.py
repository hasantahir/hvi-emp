"""Configuration composition: Hydra when present, plain YAML when not.

What Hydra buys here
--------------------
Three things this framework actually needs:

* **Composition.** A run is a scenario plus a sweep plus a cluster target,
  each chosen independently. Without groups those combinations multiply into
  a directory of near-duplicate files that drift apart.
* **Overrides.** ``scenario.velocity=62e3`` from the command line, with no
  argparse plumbing per parameter, and the override recorded in the run
  directory so the run is reproducible.
* **Multirun.** ``--multirun scenario.velocity=40e3,50e3,60e3`` expands to
  independent jobs, which is exactly the shape a scheduler wants.

What it does not buy is validation: Hydra checks that YAML parses and that
types match a structured config, not that a density is physical. That is
`hvi_emp.schema`'s job, and it runs on every composed config regardless of
which loader produced it.

Optional dependency
-------------------
Hydra is not required. `load_config` falls back to a small YAML composer
that understands the same ``defaults:`` lists and the same
``group=value`` / ``a.b=value`` override syntax. The fallback is
deliberately not a reimplementation of Hydra -- it supports the subset this
project's own configs use, and says so plainly when asked for more.

The reason for the fallback is practical: cluster login nodes are often
locked down, and a framework that cannot be inspected without installing
five packages does not get used.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .schema import ConfigError, RunSpec, validate_run

__all__ = ["CONF_DIR", "USING_HYDRA", "load_config", "compose_config",
           "apply_overrides", "to_run_spec", "config_to_yaml"]

#: Root of the configuration tree. Overridable so a project can carry its
#: own conf/ without editing the package.
CONF_DIR = Path(os.environ.get(
    "HVI_EMP_CONF", Path(__file__).resolve().parent / "conf"))

try:                                                        # pragma: no cover
    import hydra                                            # noqa: F401
    from omegaconf import OmegaConf                         # noqa: F401
    USING_HYDRA = True
except ImportError:                                         # pragma: no cover
    USING_HYDRA = False


# ---------------------------------------------------------------------------
# Plain-YAML composition (the fallback, and what the tests exercise)
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    import yaml
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:                                # noqa: BLE001
        raise ConfigError(f"{path}: not readable as YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got "
                          f"{type(data).__name__}")
    return data


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins. Lists replace rather than append.

    Lists replacing is the right default for this project: ``bands`` and
    ``values`` are complete specifications, not accumulations, and a config
    that appended would silently give you both the default bands and yours.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_group(conf_dir: Path, group: str, name: str,
                _seen: tuple = ()) -> dict:
    """Load ``<group>/<name>.yaml``, resolving its own ``defaults:`` first."""
    path = conf_dir / group / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (conf_dir / group).glob("*.yaml")) \
            if (conf_dir / group).is_dir() else []
        raise ConfigError(
            f"no config {name!r} in group {group!r} ({path} does not exist)."
            + (f" Available: {available}" if available else ""))
    if (group, name) in _seen:
        raise ConfigError(
            f"circular defaults: {group}/{name} includes itself via "
            f"{' -> '.join(f'{g}/{n}' for g, n in _seen)}")

    raw = _read_yaml(path)
    merged: dict = {}
    for entry in raw.pop("defaults", []) or []:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            merged = _deep_merge(merged,
                                 _load_group(conf_dir, group, entry,
                                             _seen + ((group, name),)))
        elif isinstance(entry, dict):
            for g, n in entry.items():
                merged = _deep_merge(
                    merged, {g: _load_group(conf_dir, g, str(n),
                                            _seen + ((group, name),))})
        else:
            raise ConfigError(f"{path}: bad defaults entry {entry!r}")
    return _deep_merge(merged, raw)


def _coerce(text: str) -> Any:
    """Turn an override string into the value it obviously denotes."""
    low = text.lower()
    if low in ("null", "none", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_coerce(x.strip()) for x in inner.split(",")] if inner else []
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def apply_overrides(cfg: dict, overrides, conf_dir: Path | None = None
                    ) -> dict:
    """Apply ``key.path=value`` and ``group=name`` overrides in order.

    A dotted key that does not already exist is an error rather than a new
    key. Silently accepting ``scenario.veloctiy=62e3`` would leave the real
    velocity at its default and produce a plausible wrong answer -- the
    exact failure mode this whole layer exists to prevent.
    """
    conf_dir = conf_dir or CONF_DIR
    cfg = copy.deepcopy(cfg)
    for item in overrides or ():
        if "=" not in item:
            raise ConfigError(
                f"override {item!r} is not of the form key=value")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()

        # `group=name` selects a whole config group
        if "." not in key and (conf_dir / key).is_dir():
            cfg[key] = _load_group(conf_dir, key, raw)
            continue

        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                raise ConfigError(
                    f"override {item!r}: {'.'.join(parts[:parts.index(p)+1])}"
                    f" is not a section of the config")
            node = node[p]
        leaf = parts[-1]
        if leaf not in node:
            near = [k for k in node
                    if k.lower().startswith(leaf[:3].lower())]
            raise ConfigError(
                f"override {item!r}: no key {leaf!r} in "
                f"{'.'.join(parts[:-1]) or '<root>'}. "
                + (f"Did you mean {near}? " if near else "")
                + f"Known keys: {sorted(node)}. (Unknown keys are rejected "
                  f"rather than added, because a typo would otherwise leave "
                  f"the real value at its default.)")
        node[leaf] = _coerce(raw)
    return cfg


def compose_config(overrides=None, conf_dir: Path | None = None,
                   config_name: str = "config") -> dict:
    """Compose the configuration tree into a plain dict.

    Uses the same ``defaults:`` semantics as Hydra for the subset this
    project uses: a list of ``{group: name}`` entries plus ``_self_``.
    """
    conf_dir = Path(conf_dir) if conf_dir else CONF_DIR
    root_path = conf_dir / f"{config_name}.yaml"
    if not root_path.is_file():
        raise ConfigError(
            f"no root config at {root_path}. Set $HVI_EMP_CONF to the "
            f"directory containing {config_name}.yaml.")
    root = _read_yaml(root_path)
    root.pop("hydra", None)                 # runtime settings, not physics
    defaults = root.pop("defaults", []) or []

    cfg: dict = {}
    for entry in defaults:
        if entry == "_self_":
            cfg = _deep_merge(cfg, root)
            root = {}
            continue
        if not isinstance(entry, dict):
            raise ConfigError(
                f"{root_path}: defaults entries must be `group: name` "
                f"mappings or `_self_`, got {entry!r}")
        for group, name in entry.items():
            cfg = _deep_merge(cfg,
                              {group: _load_group(conf_dir, group, str(name))})
    cfg = _deep_merge(cfg, root)
    return apply_overrides(cfg, overrides, conf_dir)


def load_config(overrides=None, conf_dir: Path | None = None,
                config_name: str = "config", validate: bool = True):
    """Compose and (by default) validate. Returns a plain dict.

    Validation is on by default and separate from composition on purpose:
    composing tells you the YAML is well-formed, validating tells you the
    physics is possible.
    """
    cfg = compose_config(overrides, conf_dir, config_name)
    if validate:
        to_run_spec(cfg)
    return cfg


def to_run_spec(cfg) -> RunSpec:
    """Validate a composed config into a `RunSpec`.

    Accepts a plain dict or an OmegaConf object, so the Hydra and fallback
    paths converge here and the schema runs exactly once either way.
    """
    if USING_HYDRA:                                         # pragma: no cover
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict):
        raise ConfigError(f"expected a mapping, got {type(cfg).__name__}")
    cfg = dict(cfg)
    cfg.pop("hydra", None)
    if not cfg.get("sweep"):                # `sweep: none` composes to {}
        cfg["sweep"] = None
    return validate_run(cfg)


def config_to_yaml(cfg) -> str:
    """Serialise a composed config, for writing next to the results.

    A results directory without the exact configuration that produced it is
    an orphan; six months later nobody can say what was run.
    """
    import yaml
    if USING_HYDRA:                                         # pragma: no cover
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    return yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
