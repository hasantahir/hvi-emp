"""The shared-alias collision, which is a hard compile error.

PIConGPU moved `UsedParticleShape`, `UsedField2Particle` and
`UsedParticleCurrentSolver` out of `speciesDefinition.param` and into
`species.param`. Because `species.param` is included first, a generated
`speciesDefinition.param` that declares them again does not warn -- nvcc
stops:

    error: type "picongpu::UsedParticleShape" has already been defined
           (previous definition at line 48 of .../param/species.param)

Every test here is that failure, or the older layout it has to keep working
with.
"""

import os
import textwrap

import pytest

from hvi_emp.solvers.picongpu_stage3 import (SHARED_ALIASES, PIConGPUConfig,
                                             find_picongpu, upstream_aliases,
                                             write_input_set,
                                             write_species_definition_param)

CFG = PIConGPUConfig(n_e0=1.0e24, T_eV=2.0)

MODERN_SPECIES = textwrap.dedent("""\
    #pragma once
    namespace picongpu
    {
        using UsedParticleShape = particles::shapes::PCS;
        using UsedField2Particle = FieldToParticleInterpolation<
            UsedParticleShape, AssignedTrilinearInterpolation>;
        using UsedParticleCurrentSolver = currentSolver::Esirkepov<
            UsedParticleShape>;
    }
    """)


def _checkout(tmp_path, name, species_src=None):
    d = tmp_path / name / "include" / "picongpu" / "param"
    d.mkdir(parents=True)
    (d / "grid.param").write_text("#pragma once\nnamespace picongpu {}\n")
    if species_src is not None:
        (d / "species.param").write_text(species_src)
    return str(tmp_path / name)


def _declared_by_us(text):
    return {a for a in SHARED_ALIASES if f"using {a} =" in text}


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_finds_the_aliases_in_species_param(tmp_path, monkeypatch):
    root = _checkout(tmp_path, "modern", MODERN_SPECIES)
    monkeypatch.setenv("PICSRC", root)
    found = upstream_aliases()
    assert set(found) == {"UsedParticleShape", "UsedField2Particle",
                          "UsedParticleCurrentSolver"}
    # The location is reported so the message can be checked against the
    # compiler's own "previous definition at line N of ..." text.
    assert all("species.param:" in v for v in found.values())


def test_older_checkout_declares_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PICSRC", _checkout(tmp_path, "old"))
    assert upstream_aliases() == {}


def test_find_picongpu_returns_none_rather_than_guessing(tmp_path, monkeypatch):
    for v in ("PICSRC", "PICONGPU_ROOT", "PICONGPU"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("PICSRC", str(tmp_path / "nope"))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "nope"))
    assert find_picongpu() is None


def test_a_directory_without_param_is_not_a_checkout(tmp_path, monkeypatch):
    (tmp_path / "bare").mkdir()
    monkeypatch.setenv("PICSRC", str(tmp_path / "bare"))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "bare"))
    assert find_picongpu() is None


# --------------------------------------------------------------------------
# what gets written
# --------------------------------------------------------------------------

def test_modern_checkout_gets_no_redeclaration(tmp_path, monkeypatch):
    """THE regression. Redeclaring any of these three stops the build."""
    monkeypatch.setenv("PICSRC", _checkout(tmp_path, "modern", MODERN_SPECIES))
    text = write_species_definition_param(CFG)
    clash = _declared_by_us(text) & {"UsedParticleShape", "UsedField2Particle",
                                     "UsedParticleCurrentSolver"}
    assert not clash, f"would collide with species.param: {sorted(clash)}"


def test_older_checkout_still_gets_the_aliases(tmp_path, monkeypatch):
    """The fix must not break the layout it used to be written for."""
    monkeypatch.setenv("PICSRC", _checkout(tmp_path, "old"))
    text = write_species_definition_param(CFG)
    assert _declared_by_us(text) == set(SHARED_ALIASES)


def test_species_flags_never_use_the_upstream_names(tmp_path, monkeypatch):
    """Our own flags must not depend on which layout is installed.

    This is what makes the file version-proof: the prefixed aliases are
    always declared here, so the MakeSeq_t flags resolve either way.
    """
    for name, src in (("modern", MODERN_SPECIES), ("old", None)):
        monkeypatch.setenv("PICSRC", _checkout(tmp_path, name, src))
        text = write_species_definition_param(CFG)
        flags = text[text.index("ParticleFlagsElectrons"):]
        for alias in SHARED_ALIASES:
            assert f"<{alias}>" not in flags
        for alias in ("HviParticleShape", "HviField2Particle",
                      "HviParticleCurrentSolver", "HviParticlePusher"):
            assert alias in text


def test_unknown_checkout_assumes_the_modern_layout(tmp_path, monkeypatch):
    """Guessing wrong here is asymmetric.

    Omitting a declaration nothing references is harmless; emitting one that
    collides stops the build. So with no checkout to inspect, omit -- and say
    so in the file.
    """
    for v in ("PICSRC", "PICONGPU_ROOT", "PICONGPU"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "nope"))
    text = write_species_definition_param(CFG)
    assert not _declared_by_us(text)
    assert "not found" in text and "PICSRC" in text


def test_provenance_is_recorded_in_the_file(tmp_path, monkeypatch):
    """A generated file should say what it checked and what it found."""
    monkeypatch.setenv("PICSRC", _checkout(tmp_path, "modern", MODERN_SPECIES))
    text = write_species_definition_param(CFG)
    assert "species.param:" in text
    assert "UsedParticleShape" in text          # named in the explanation


def test_write_input_set_threads_the_root_through(tmp_path, monkeypatch):
    for v in ("PICSRC", "PICONGPU_ROOT", "PICONGPU"):
        monkeypatch.delenv(v, raising=False)
    root = _checkout(tmp_path, "modern", MODERN_SPECIES)
    out = tmp_path / "run"
    written = write_input_set(CFG, str(out), picongpu_root=root)
    text = open(written["speciesDefinition.param"]).read()
    assert not _declared_by_us(text) & {"UsedParticleShape",
                                        "UsedField2Particle",
                                        "UsedParticleCurrentSolver"}
    assert "species.param:" in text


def test_the_file_is_still_structurally_sane(tmp_path, monkeypatch):
    monkeypatch.setenv("PICSRC", _checkout(tmp_path, "modern", MODERN_SPECIES))
    text = write_species_definition_param(CFG)
    # One opening declaration; the other occurrence is the closing comment.
    assert text.count("namespace picongpu\n{") == 1
    assert text.count("} // namespace picongpu") == 1
    assert text.count("{") == text.count("}")
    for needed in ("PIC_Electrons", "PIC_Ions", "Probes", "VectorAllSpecies"):
        assert needed in text
