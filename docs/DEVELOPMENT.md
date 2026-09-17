# Working on this code

## The repository

Version control starts at build `2026.09.17-b14`. Everything before that was
shipped as dated zip archives and has no commits — `CHANGELOG.md` records
what changed in that series, reconstructed from `BUILD.txt`, so the reasoning
survives even though the diffs do not.

```bash
git log --oneline
git tag                       # v2026.09.17-b14 is the snapshot
```

## Moving code to the workstation

The zip workflow it replaces had a real failure mode: three different builds
were shipped as `hvi_emp_20260916.zip`, and reznor ended up running the
oldest of them for two days. A bundle cannot do that — it carries commits, so
`git log` on either machine tells you exactly what you have.

**A bundle is a single file containing the whole repository.** Copy it like a
zip; clone or pull from it like a remote.

### First time

```bash
# here
git bundle create hvi_emp.bundle --all
scp hvi_emp.bundle hasan@reznor:~/

# on reznor
git clone ~/hvi_emp.bundle ~/hvi_emp
cd ~/hvi_emp && git log --oneline      # you can SEE what you got
```

### Updates after that

```bash
# here
git bundle create update.bundle main
scp update.bundle hasan@reznor:~/

# on reznor
cd ~/hvi_emp
git pull ~/update.bundle main
```

If `git pull` refuses, the bundle is missing a commit the clone needs; make a
complete one with `--all` and clone fresh.

### Changes made *on* reznor

Fixes found while running belong in the history too — a patch that only exists
in a terminal scrollback is lost:

```bash
# on reznor
git add -A && git commit -m "..."
git bundle create back.bundle main
scp back.bundle <here>:~/

# here
git pull ~/back.bundle main
```

## Checking what you are running

```bash
git describe --tags --always --dirty
```

`-dirty` means uncommitted edits — worth knowing before you attribute a
result to a version. `BUILD.txt` and the `--version` flag on
`install_solvers.sh` stay for zips already in circulation, but `git describe`
supersedes both.

## Before committing

```bash
python -m pytest tests/ -q
```

The suite takes a few minutes. The LAMMPS live smoke test is the slow part
and is skipped when LAMMPS is not importable.

## What is deliberately not tracked

Generated solver decks (`fletcher2021/`, `full_chain/`, `m2c_test/`), solver
output (`results/`, `*.vtr`, `*.vti`, `AtomicData/`) and figures. All are
reproducible from the bridges, and committing them adds churn without adding
information. `.gitignore` has the full list.

`AtomicData/` is ignored on purpose even though the run needs it:
`write_problem_directory()` regenerates it from the material YAML every time,
so a committed copy would silently go stale against the ladder it came from.
