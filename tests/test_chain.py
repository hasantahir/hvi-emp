"""Tests for the full-chain orchestrator and the composite animation.

Three things here are worth more than the rest:

* a stage that raises must not take the other four down -- on a fresh machine
  three of five solvers are absent and one of them *will* have a bad day;
* resume must be real, because a campaign spans days and re-running must not
  resubmit a job that is already burning 64 cores;
* the composite must not be able to lose its "this is not one simulation"
  caption, because without it the animation asserts something false.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hvi_emp.solvers import have_lammps

from hvi_emp.chain import (STAGE_INFO, STAGE_ORDER, Manifest, StageResult,
                           outputs_exist, report, write_stage_script)
from hvi_emp.viz.composite import (CHAPTER_DEFAULTS, COMPOSITE_NOTICE,
                                   Chapter, build_chapters,
                                   write_composite_script, zoom_keyframes)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_chain.py"


class TestStageDefinitions:

    def test_every_stage_has_a_description(self):
        for name in STAGE_ORDER:
            assert name in STAGE_INFO
            for key in ("code", "scale", "what", "cost"):
                assert STAGE_INFO[name][key]

    def test_order_is_the_order_physics_happens(self):
        assert STAGE_ORDER.index("reduced") < STAGE_ORDER.index("md")
        assert STAGE_ORDER.index("md") < STAGE_ORDER.index("hydro")
        assert STAGE_ORDER.index("hydro") < STAGE_ORDER.index("mhd")

    def test_result_ok_covers_only_progress(self):
        for s in ("done", "ran", "submitted"):
            assert StageResult("x", s).ok
        for s in ("blocked", "skipped"):
            assert not StageResult("x", s).ok


class TestManifest:

    def test_round_trips(self, tmp_path):
        p = str(tmp_path / "m.json")
        m = Manifest(p)
        m.set("hydro", job_id="99", status="submitted")
        assert Manifest(p).get("hydro")["job_id"] == "99"

    def test_write_is_atomic(self, tmp_path):
        """A kill mid-write must not destroy a multi-day campaign's state."""
        p = str(tmp_path / "m.json")
        m = Manifest(p)
        m.set("md", status="ran")
        assert not os.path.exists(p + ".tmp")
        json.load(open(p))                      # valid JSON, not truncated

    def test_a_corrupt_manifest_is_kept_not_overwritten(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text("{not json")
        m = Manifest(str(p))
        assert os.path.exists(str(p) + ".corrupt")
        assert "unreadable" in m.data.get("note", "")

    def test_unknown_stage_returns_empty(self, tmp_path):
        assert Manifest(str(tmp_path / "m.json")).get("nope") == {}


class TestOutputsExist:

    def test_empty_list_is_not_done(self):
        """Otherwise a stage with no declared outputs is 'done' forever."""
        assert outputs_exist([]) is False

    def test_zero_length_file_is_not_done(self, tmp_path):
        """A job killed at the wall clock leaves empty files behind."""
        f = tmp_path / "out.dat"
        f.write_text("")
        assert outputs_exist([str(f)]) is False
        f.write_text("x")
        assert outputs_exist([str(f)]) is True

    def test_directories_count_as_present(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        assert outputs_exist([str(d)]) is True

    def test_all_must_exist(self, tmp_path):
        a = tmp_path / "a"
        a.write_text("x")
        assert outputs_exist([str(a), str(tmp_path / "missing")]) is False


class TestSbatchScript:

    def test_contains_the_resources_it_was_given(self, tmp_path):
        p = write_stage_script(str(tmp_path / "s.sbatch"), "echo hi",
                               job_name="hvi-hydro", cores=64, hours=48,
                               mem_gb=128)
        t = open(p).read()
        assert "--ntasks=64" in t
        assert "--time=48:00:00" in t
        assert "--mem=128G" in t
        assert "--job-name=hvi-hydro" in t

    def test_gpu_directive_only_when_asked(self, tmp_path):
        no = open(write_stage_script(str(tmp_path / "a.sbatch"), "x",
                                     "j", gpus=0)).read()
        yes = open(write_stage_script(str(tmp_path / "b.sbatch"), "x",
                                      "j", gpus=1)).read()
        assert "--gres" not in no
        assert "--gres=gpu:1" in yes

    def test_fails_fast_on_error(self, tmp_path):
        """Without this a broken solver silently 'succeeds' and the stage is
        marked done on the next run."""
        t = open(write_stage_script(str(tmp_path / "s.sbatch"), "x",
                                    "j")).read()
        assert "set -euo pipefail" in t

    def test_is_executable(self, tmp_path):
        p = write_stage_script(str(tmp_path / "s.sbatch"), "x", "j")
        assert os.access(p, os.X_OK)


class TestComposite:

    def _chapters(self, tmp_path, stages=("md", "reduced")):
        pvds = {}
        for s in stages:
            f = tmp_path / f"{s}.pvd"
            f.write_text("<VTKFile/>")
            pvds[s] = str(f)
        return build_chapters(pvds)

    def test_missing_chapters_are_kept_and_flagged(self, tmp_path):
        """Silently shortening the movie would hide which physics is absent."""
        chs = self._chapters(tmp_path, ("md",))
        assert len(chs) == len(CHAPTER_DEFAULTS)
        missing = [c for c in chs if not c.available]
        assert missing and all(c.notes for c in missing)

    def test_available_only_when_the_file_is_there(self, tmp_path):
        c = Chapter(stage="md", code="X", zoom=1e-9, clock="ps", what="w",
                    pvd=str(tmp_path / "nope.pvd"))
        assert not c.available

    def test_banner_names_code_scale_and_clock(self, tmp_path):
        ch = self._chapters(tmp_path)[0]
        b = ch.banner()
        assert ch.code in b and ch.clock in b and "scale" in b

    def test_zoom_interpolation_is_geometric(self, tmp_path):
        """Nine decades linearly ramped would spend the whole movie in the
        last chapter."""
        keys = zoom_keyframes(self._chapters(tmp_path), hold=3)
        assert keys
        assert min(keys) > 0
        assert max(keys) / min(keys) > 100.0

    def test_no_chapters_gives_no_keyframes(self, tmp_path):
        assert zoom_keyframes(build_chapters({})) == []

    def test_script_is_valid_python(self, tmp_path):
        chs = self._chapters(tmp_path)
        t = write_composite_script(chs, str(tmp_path / "c.py"))
        ast.parse(t)

    def test_the_composite_caveat_is_in_the_script(self, tmp_path):
        """The single most important thing in this module."""
        t = write_composite_script(self._chapters(tmp_path),
                                   str(tmp_path / "c.py"))
        assert COMPOSITE_NOTICE in t
        assert "not one simulation" in COMPOSITE_NOTICE.lower() or \
               "independent simulations" in COMPOSITE_NOTICE

    def test_the_caveat_is_rendered_not_just_commented(self, tmp_path):
        """It has to reach the frame, not sit in a docstring."""
        t = write_composite_script(self._chapters(tmp_path),
                                   str(tmp_path / "c.py"))
        assert "notice = Text(" in t
        assert "Show(notice, view" in t

    def test_handover_card_says_it_is_a_model_change(self, tmp_path):
        t = write_composite_script(self._chapters(tmp_path),
                                   str(tmp_path / "c.py"))
        assert "change of model, not a zoom" in t

    def test_missing_solvers_are_listed_in_the_script(self, tmp_path):
        t = write_composite_script(self._chapters(tmp_path, ("md",)),
                                   str(tmp_path / "c.py"))
        assert "MISSING" in t

    def test_it_refuses_to_render_nothing(self, tmp_path):
        t = write_composite_script(build_chapters({}), str(tmp_path / "c.py"))
        ast.parse(t)
        assert "no chapters have output yet" in t

    def test_gpu_mapper_is_pinned(self, tmp_path):
        t = write_composite_script(self._chapters(tmp_path),
                                   str(tmp_path / "c.py"))
        assert "GPU Based" in t


class TestDriverCLI:
    """The entry point, exercised as a subprocess -- how it is actually used."""

    def _run(self, *args, env=None):
        e = dict(os.environ)
        e.pop("M2C_HOME", None)
        e.pop("OPENMHD", None)
        e.pop("PICSRC", None)
        if env:
            e.update(env)
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, cwd=str(ROOT),
                              env=e, timeout=600)

    def test_dry_invocation_reports_every_stage(self, tmp_path):
        r = self._run("--out", str(tmp_path / "fc"))
        assert r.returncode == 0, r.stderr[-2000:]
        for name in STAGE_ORDER:
            assert name in r.stdout

    def test_it_says_nothing_will_run_without_a_flag(self, tmp_path):
        r = self._run("--out", str(tmp_path / "fc"))
        assert "nothing will run" in r.stdout

    def test_absent_solvers_block_only_themselves(self, tmp_path):
        """Three of five are missing here and the run must still succeed."""
        r = self._run("--out", str(tmp_path / "fc"))
        assert r.returncode == 0
        assert "blocked" in r.stdout
        assert "M2C not found" in r.stdout

    def test_decks_are_written_even_when_the_solver_is_absent(self, tmp_path):
        d = tmp_path / "fc"
        self._run("--out", str(d))
        assert (d / "hydro" / "input.st").is_file()

    def test_manifest_is_created(self, tmp_path):
        d = tmp_path / "fc"
        self._run("--out", str(d))
        assert (d / "manifest.json").is_file()

    def test_only_restricts_the_stages(self, tmp_path):
        r = self._run("--out", str(tmp_path / "fc"), "--only", "reduced")
        assert "hydro" not in r.stdout.split("full-chain status")[1].split(
            "complete")[0]

    def test_render_without_data_is_a_clean_failure(self, tmp_path):
        r = self._run("--out", str(tmp_path / "fc"), "--render")
        assert r.returncode == 1
        assert "nothing to render" in r.stdout

    def test_the_composite_warning_is_printed_on_render(self, tmp_path):
        """Whoever runs this must be told before they show it to anyone."""
        d = tmp_path / "fc"
        r = self._run("--out", str(d), "--run", "--render",
                      "--only", "reduced", "--quality", "draft",
                      "--frames", "3")
        assert r.returncode == 0, r.stderr[-2000:]
        assert "NOT ONE SIMULATION" in r.stdout

    def test_resume_marks_completed_stages_done(self, tmp_path):
        d = tmp_path / "fc"
        first = self._run("--out", str(d), "--run", "--only", "reduced",
                          "--quality", "draft", "--frames", "3")
        assert "[ran ]" in first.stdout, first.stderr[-2000:]
        second = self._run("--out", str(d), "--run", "--only", "reduced",
                           "--quality", "draft", "--frames", "3")
        assert "[done]" in second.stdout


class TestReport:

    def test_counts_are_right(self):
        rs = [StageResult("a", "done"), StageResult("b", "submitted"),
              StageResult("c", "blocked")]
        t = report(rs)
        assert "1 complete, 1 queued, 1 blocked" in t

    def test_queued_work_prompts_a_re_run(self):
        assert "re-run" in report([StageResult("a", "submitted")])

    def test_blocked_work_points_at_the_install_doc(self):
        assert "INSTALLING_SOLVERS" in report([StageResult("a", "blocked")])


# ---------------------------------------------------------------------------
# MD scale presets
# ---------------------------------------------------------------------------

class TestMDScales:
    """The default 25k-atom run looks like a toy; these are the sizes that
    do not. The cost model behind them is measured, and the test suite has to
    keep the "measured" and "assumed" parts distinguishable."""

    def test_every_scale_builds_a_config(self):
        from hvi_emp.solvers.lammps_stage1 import MD_SCALES, config_for_scale
        for name in MD_SCALES:
            cfg = config_for_scale(name)
            assert cfg.estimated_atom_count() > 0

    def test_scales_increase_monotonically_in_atoms(self):
        from hvi_emp.solvers.lammps_stage1 import scale_table
        atoms = [r["atoms"] for r in scale_table()]
        assert atoms == sorted(atoms)

    def test_fraile_scale_matches_the_published_run(self):
        """Fraile et al. (2022) ran 4e7 atoms. If our preset does not land
        near that, the preset is misnamed."""
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        n = config_for_scale("fraile").estimated_atom_count()
        assert 3.0e7 < n < 5.0e7

    def test_talk_scale_is_a_big_step_up_from_smoke(self):
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        smoke = config_for_scale("smoke").estimated_atom_count()
        talk = config_for_scale("talk").estimated_atom_count()
        assert talk / smoke > 20

    def test_cost_scales_linearly_in_atoms_times_steps(self):
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        a = config_for_scale("talk")
        b = config_for_scale("talk", t_production=2 * a.t_production)
        ca, cb = a.estimate_cost(), b.estimate_cost()
        assert cb["wall_seconds"] / ca["wall_seconds"] == pytest.approx(
            cb["steps"] / ca["steps"], rel=0.02)

    def test_more_cores_is_proportionally_faster(self):
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        cfg = config_for_scale("talk")
        assert (cfg.estimate_cost(cores=8)["wall_seconds"]
                == pytest.approx(
                    8 * cfg.estimate_cost(cores=64)["wall_seconds"], rel=1e-6))

    def test_measured_and_assumed_are_distinguished(self):
        """The rate was measured; the MPI efficiency was not. A cost estimate
        that blurs the two invites treating the second as data."""
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        est = config_for_scale("talk").estimate_cost()
        assert est["rate_is_measured"] is True
        assert est["efficiency_is_assumed"] is True

    def test_memory_is_plausible_for_the_published_run(self):
        """4e7 atoms should be ~10 GB, not ~10 MB or ~10 TB."""
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        gb = config_for_scale("fraile").estimate_cost()["memory_GB"]
        assert 5.0 < gb < 30.0

    def test_unknown_scale_is_rejected(self):
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        with pytest.raises(KeyError):
            config_for_scale("enormous")

    def test_overrides_beat_the_preset(self):
        from hvi_emp.solvers.lammps_stage1 import config_for_scale
        cfg = config_for_scale("talk", target_cells=(10, 10, 7))
        assert cfg.resolved_cells() == (10, 10, 7)


@pytest.mark.skipif(not have_lammps(),
                    reason="lammps not importable; the md stage reports "
                           "'blocked' before it reaches the size guard")
class TestMDSizeGuard:
    """A 10-hour MD run must not silently occupy the terminal, and a 2-minute
    one must not need a scheduler."""

    def _run(self, *args, path_prefix=None):
        env = dict(os.environ)
        for k in ("M2C_HOME", "OPENMHD", "PICSRC"):
            env.pop(k, None)
        if path_prefix:
            env["PATH"] = f"{path_prefix}:{env['PATH']}"
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, cwd=str(ROOT),
                              env=env, timeout=600)

    def test_list_scales_reports_the_provenance_of_its_numbers(self):
        r = self._run("--list-scales")
        assert r.returncode == 0
        assert "MEASURED" in r.stdout and "ASSUMED" in r.stdout

    def test_a_heavy_run_is_not_started_in_process(self, tmp_path):
        r = self._run("--out", str(tmp_path / "fc"), "--run", "--only", "md",
                      "--cores", "4", "--md-scale", "paper")
        assert "pass --submit" in r.stdout or "too big" in r.stdout
        assert "[ran ]" not in r.stdout

    def test_the_estimate_is_shown_before_committing(self, tmp_path):
        """So you can decide, rather than finding out in six hours."""
        r = self._run("--out", str(tmp_path / "fc"), "--only", "md",
                      "--cores", "64", "--md-scale", "detailed")
        assert "M atoms" in r.stdout and "h on 64 cores" in r.stdout

    def test_submit_implies_run(self, tmp_path):
        """Asking for everything must not answer 'pass --run'."""
        r = self._run("--out", str(tmp_path / "fc"), "--submit",
                      "--only", "reduced", "--quality", "draft",
                      "--frames", "3")
        assert "pass --run" not in r.stdout


# ===========================================================================
# --submit on a machine with no scheduler
# ===========================================================================

def _chain_module():
    """Import scripts/run_full_chain.py as a module."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "run_full_chain.py"
    spec = importlib.util.spec_from_file_location("run_full_chain", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, submit=True, md_local_minutes=20.0):
        self.submit = submit
        self.md_local_minutes = md_local_minutes


def test_submit_without_sbatch_warns(monkeypatch):
    """A standalone workstation must be told why nothing queued.

    Reported from a 64-core box with no Slurm: `--submit` was accepted,
    all four heavy stages reported "blocked", and the only mention of the
    cause was one clause inside a per-stage line. It read as "the solvers
    are missing" when the real problem was "there is nowhere to submit".
    """
    mod = _chain_module()
    monkeypatch.setattr(mod, "slurm_available", lambda: False)
    w = mod.scheduler_warning(_Args())
    assert w is not None
    assert "no `sbatch` on PATH" in w
    # It must name the lever, not just the problem ...
    assert "--md-local-minutes" in w
    # ... and must distinguish itself from missing solvers, because the
    # two look identical in the status report but need opposite fixes.
    assert "NOT the same as the solvers being missing" in w


def test_no_warning_when_sbatch_exists(monkeypatch):
    mod = _chain_module()
    monkeypatch.setattr(mod, "slurm_available", lambda: True)
    assert mod.scheduler_warning(_Args()) is None


def test_no_warning_without_submit(monkeypatch):
    mod = _chain_module()
    monkeypatch.setattr(mod, "slurm_available", lambda: False)
    assert mod.scheduler_warning(_Args(submit=False)) is None


def test_warning_reports_the_current_budget(monkeypatch):
    """The suggested flag is useless without the value it replaces."""
    mod = _chain_module()
    monkeypatch.setattr(mod, "slurm_available", lambda: False)
    w = mod.scheduler_warning(_Args(md_local_minutes=45.0))
    assert "45 min" in w and "0.8 h" in w


def test_warning_is_printed_before_the_stage_report():
    """Ordering matters: after the report it is just more noise."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "run_full_chain.py").read_text()
    assert src.index("scheduler_warning(args)") < src.index("report(")


def test_empty_results_dir_is_not_mistaken_for_output(tmp_path):
    """An existing-but-empty results/ must not mark the stage done.

    The deck writer creates results/ up front, because M2C does not mkdir
    it and dies at the first output step without it. The completion check
    tested for the *directory*, so writing the deck immediately reported
    the stage as finished -- and the chain would have skipped it forever.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run = tmp_path / "fc"
    env = dict(os.environ, PYTHONPATH=str(root))

    def chain():
        r = subprocess.run(
            [sys.executable, str(root / "scripts" / "run_full_chain.py"),
             "--out", str(run), "--only", "hydro", "--run"],
            capture_output=True, text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr
        return r.stdout

    out = chain()
    results = run / "hydro" / "results"
    assert results.is_dir(), "deck writer must create results/"
    assert not any(results.iterdir()), "results/ should start empty"
    assert "output present" not in out, out[-800:]

    # A real output file flips it to done.
    (results / "solution_0000.vtr").write_text("x\n")
    assert "output present" in chain()
