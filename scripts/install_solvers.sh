#!/usr/bin/env bash
#
# Install the optional production solvers for HVI-EMP.
#
#   scripts/install_solvers.sh --dry-run            # show what would happen
#   scripts/install_solvers.sh idefix openmhd       # just those two
#   scripts/install_solvers.sh all
#
# Deliberate design choices:
#
#   * NEVER uses sudo, and never touches apt. Fixing the NVIDIA driver needs
#     root and a reboot; that is your call, not a script's. It prints the
#     commands and stops.
#   * Idempotent. Anything already present is skipped, so re-running after a
#     failure costs only the step that failed.
#   * Verifies each install and says plainly what failed.
#   * Writes the environment variables it needs to a file you source, rather
#     than editing your shell rc behind your back.
#
# iSALE is not here: it needs a licence application a human approves, and in
# practice those requests went unanswered. Use m2c instead -- same stage, no
# gatekeeper, and it does oblique impacts and ionisation as well.
#   https://isale-code.github.io/access.html
#
set -uo pipefail

# Build stamp. Bumped whenever this script's interface changes, so that
# "unknown option: --bootstrap-mamba" is answerable in one command
# (`install_solvers.sh --version`) instead of by guessing which archive got
# transferred. Older builds have no --version flag at all, which is itself
# the answer.
BUILD="2026.09.17-b14"

SRC="${HVI_SRC_DIR:-$HOME/src}"
ENVFILE="${HVI_ENVFILE:-$HOME/.hvi_emp_solvers.env}"
DRY=0
JOBS="$( (nproc 2>/dev/null || echo 4) )"
JOBS=$(( JOBS > 16 ? 16 : JOBS ))

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
FAILED=()
DONE=()
SKIPPED=()
MISSING=()

say()  { printf '%s\n' "$*"; }
head1() { printf '\n%s%s%s\n%s\n' "$BLD" "$*" "$RST" "$(printf '=%.0s' {1..64})"; }
ok()   { printf '  %s[ ok ]%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s[warn]%s %s\n' "$YLW" "$RST" "$*"; }
bad()  { printf '  %s[FAIL]%s %s\n' "$RED" "$RST" "$*"; }

run() {
    if [[ $DRY -eq 1 ]]; then
        printf '  would run: %s\n' "$*"
        return 0
    fi
    printf '  + %s\n' "$*"
    "$@"
}

have() { command -v "$1" >/dev/null 2>&1; }

# Run a build, keep the log, and show the tail if it fails.
#
# Swallowing build output with >/dev/null was a mistake: when a build failed
# the only advice this script could give was "run it by hand to see why",
# which is exactly the work it was supposed to save.
build_log() {   # build_log <logfile> <cmd...>
    local log="$1"; shift
    if [[ $DRY -eq 1 ]]; then
        printf '  would run: %s  (log: %s)\n' "$*" "$log"
        return 0
    fi
    printf '  + %s\n' "$*"
    if "$@" >"$log" 2>&1; then
        return 0
    fi
    bad "failed; last 15 lines of $log:"
    sed -e 's/^/      /' "$log" | tail -15
    return 1
}

# A missing prerequisite is a hard failure in a real run, but during a dry run
# nothing was attempted, so reporting it as "failed" would be a lie. Record it
# as a missing prerequisite instead and keep going.
need() {   # need <command> <target> <how-to-install>
    local cmd="$1" target="$2" how="$3"
    have "$cmd" && return 0
    if [[ $DRY -eq 1 ]]; then
        warn "$cmd not found -- $how"
        MISSING+=("$cmd (for $target)")
        return 1
    fi
    bad "$target: $cmd not found -- $how"
    FAILED+=("$target")
    return 1
}

# ---------------------------------------------------------------------------
# Package manager: micromamba > mamba > conda.
#
# The ordering is about solve time. These solver stacks (PETSc, MPI, VTK,
# CUDA) are exactly the dependency graphs where the classic conda resolver
# spends minutes backtracking.
#
# But "conda slow, mamba fast" is only half true, and the half that is false
# wastes people's time: conda >= 23.10 *defaults to libmamba*, the same solver
# mamba uses. If you are on a current conda, you do not need to install
# anything -- and if you are on an old one, `conda config --set solver
# libmamba` fixes it in a second without a download. Both cases are reported
# below rather than being papered over with "just use mamba".
# ---------------------------------------------------------------------------
PKG_MGR=""          # resolved once by resolve_pkg_mgr
PKG_MGR_WHY=""

# Three thresholds, not one -- the advice differs in each band:
#   >= 23.10  libmamba is the DEFAULT; nothing to do
#   >= 22.11  the `solver` config key exists; opt in
#    < 22.11  the key does NOT exist. `conda config --set solver libmamba`
#             fails with "Key 'solver' is not a known primitive parameter".
#             Never suggest it in this band.
LIBMAMBA_DEFAULT_MAJOR=23; LIBMAMBA_DEFAULT_MINOR=10
LIBMAMBA_KEY_MAJOR=22;     LIBMAMBA_KEY_MINOR=11

CONDA_VER=""
conda_version() {   # echoes e.g. "4.10", cached
    [[ -n "$CONDA_VER" ]] && { echo "$CONDA_VER"; return; }
    CONDA_VER="$(conda --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    echo "$CONDA_VER"
}

# echoes: default | configurable | unreachable | unknown
conda_solver_band() {
    local ver major minor
    case "${CONDA_SOLVER:-}" in
        libmamba) echo default; return ;;
    esac
    ver="$(conda_version)"
    [[ -z "$ver" ]] && { echo unknown; return; }
    # 10# forces base 10: bash reads a leading-zero field as octal, so a
    # hypothetical "23.08" would abort the arithmetic rather than compare.
    major=$((10#${ver%%.*})); minor=$((10#${ver##*.}))

    if [[ "${CONDA_SOLVER:-}" != classic ]]; then
        (( major > LIBMAMBA_DEFAULT_MAJOR )) && { echo default; return; }
        (( major == LIBMAMBA_DEFAULT_MAJOR && minor >= LIBMAMBA_DEFAULT_MINOR )) \
            && { echo default; return; }
    fi
    (( major > LIBMAMBA_KEY_MAJOR )) && { echo configurable; return; }
    (( major == LIBMAMBA_KEY_MAJOR && minor >= LIBMAMBA_KEY_MINOR )) \
        && { echo configurable; return; }
    echo unreachable
}

conda_is_fast() { [[ "$(conda_solver_band)" == default ]]; }

resolve_pkg_mgr() {
    [[ -n "$PKG_MGR" ]] && return 0
    local mm="$SRC/bin/micromamba"
    [[ -x "$mm" ]] && ! have micromamba && export PATH="$SRC/bin:$PATH"

    if have micromamba; then
        PKG_MGR=micromamba
        PKG_MGR_WHY="micromamba (static binary, libmamba solver)"
    elif have mamba; then
        PKG_MGR=mamba
        PKG_MGR_WHY="mamba (libmamba solver)"
    elif have conda; then
        PKG_MGR=conda
        local v; v="$(conda --version 2>/dev/null | awk '{print $2}')"
        case "$(conda_solver_band)" in
            default)
                PKG_MGR_WHY="conda $v -- already uses the libmamba solver,"
                PKG_MGR_WHY+=" so this is as fast as mamba bar process startup" ;;
            configurable)
                PKG_MGR_WHY="conda $v -- CLASSIC resolver, but can opt in" ;;
            unreachable)
                PKG_MGR_WHY="conda $v -- CLASSIC resolver, and too old to"
                PKG_MGR_WHY+=" switch (no 'solver' key before 22.11)" ;;
            *)
                PKG_MGR_WHY="conda $v" ;;
        esac
    else
        PKG_MGR=""
        PKG_MGR_WHY="none found"
        return 1
    fi
    return 0
}

# Print the choice, and -- only when it is actually the slow case -- how to
# fix it. Printing "use mamba" at someone already on libmamba is noise.
report_pkg_mgr() {
    head1 "Package manager"
    if ! resolve_pkg_mgr; then
        bad "no micromamba, mamba or conda on PATH"
        say "  Either install Miniforge (ships conda-forge + mamba):"
        say "      https://github.com/conda-forge/miniforge#install"
        say "  or let this script drop a standalone micromamba into $SRC/bin:"
        say "      scripts/install_solvers.sh --bootstrap-mamba"
        return 1
    fi
    ok "using $PKG_MGR_WHY"
    if [[ "$PKG_MGR" == conda ]]; then
        case "$(conda_solver_band)" in
            configurable)
                warn "this is the slow case. Cheapest fix first:"
                say "      conda config --set solver libmamba"
                say "  If that errors, install the plugin first:"
                say "      conda install -n base -c conda-forge conda-libmamba-solver"
                ;;
            unreachable)
                warn "this is the slow case, and conda cannot be switched:"
                say "  'conda config --set solver libmamba' will fail here with"
                say "  \"Key 'solver' is not a known primitive parameter\" --"
                say "  that key arrived in conda 22.11. Upgrading conda in place"
                say "  is itself a slow classic solve, so the cheap route is a"
                say "  standalone micromamba that ignores conda entirely:"
                say "      scripts/install_solvers.sh --bootstrap-mamba"
                say "  It installs into your active env (-p \$CONDA_PREFIX), so"
                say "  the environment you have keeps working."
                ;;
        esac
    fi
    return 0
}

# Fetch the standalone micromamba binary. Opt-in only (--bootstrap-mamba):
# this downloads and runs a binary, so it does not happen because you asked
# to install LAMMPS.
bootstrap_mamba() {
    head1 "micromamba bootstrap"
    if have micromamba; then
        ok "already present: $(command -v micromamba)"; return 0
    fi
    local sys arch plat url dest
    sys="$(uname -s)"; arch="$(uname -m)"
    case "$sys/$arch" in
        Darwin/arm64)          plat=osx-arm64 ;;
        Darwin/*)              plat=osx-64 ;;
        Linux/aarch64|Linux/arm64) plat=linux-aarch64 ;;
        Linux/ppc64le)         plat=linux-ppc64le ;;
        Linux/*)               plat=linux-64 ;;
        *) bad "unsupported platform $sys/$arch"; FAILED+=(micromamba); return 1 ;;
    esac
    url="https://micro.mamba.pm/api/micromamba/$plat/latest"
    dest="$SRC/bin"
    say "  platform: $plat"
    say "  source:   $url"
    need curl micromamba "install curl, or use Miniforge instead" || return 1
    run mkdir -p "$dest"
    if [[ $DRY -eq 1 ]]; then
        printf '  would run: curl -Ls %s | tar -xj -C %s --strip-components=1 bin/micromamba\n' "$url" "$dest"
        return 0
    fi
    printf '  + curl -Ls %s | tar -xj -C %s\n' "$url" "$dest"
    if ! curl -Ls "$url" | tar -xj -C "$dest" --strip-components=1 bin/micromamba; then
        bad "micromamba download failed"; FAILED+=(micromamba); return 1
    fi
    chmod +x "$dest/micromamba"
    export PATH="$dest:$PATH"
    record_env "export PATH=\"$dest:\$PATH\""
    record_env "export MAMBA_ROOT_PREFIX=\"${MAMBA_ROOT_PREFIX:-$HOME/micromamba}\""
    PKG_MGR=""   # force re-resolution now that it exists
    resolve_pkg_mgr
    ok "micromamba installed: $("$dest/micromamba" --version 2>/dev/null)"
    say "  Added to $ENVFILE -- 'source $ENVFILE' in new shells."
    DONE+=(micromamba)
    return 0
}

record_env() {
    [[ $DRY -eq 1 ]] && { printf '  would add to %s: %s\n' "$ENVFILE" "$1"; return; }
    touch "$ENVFILE"
    grep -qxF "$1" "$ENVFILE" || printf '%s\n' "$1" >> "$ENVFILE"
}

# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
usage: install_solvers.sh [--dry-run] [--jobs N] [--bootstrap-mamba] [TARGET ...]

options:
  --dry-run, -n       show what would happen, install nothing
  --jobs N, -j N      parallel make jobs
  --bootstrap-mamba   fetch a standalone micromamba into $HVI_SRC_DIR/bin.
                      Use when you have no conda at all, or when conda is
                      old enough to still solve with the classic resolver.
                      On its own it does nothing else; with targets, it runs
                      first and the targets then use it.

targets:
  m2c        Stage 1 hydro + ionisation. The iSALE replacement. Needs PETSc.
  idefix     Stage 2 MHD, GPU via Kokkos. One clone + one cmake. Easiest win.
  openmhd    Stage 2 MHD, second independent answer. Needs gfortran + MPI.
  warpx      Stage 3 PIC via conda-forge (CPU-only build; see the docs for CUDA).
  picongpu   Stage 3 PIC, GPU-native. Heaviest: expect an afternoon.
  paraview   .vti viewer, via conda-forge.
  lammps     Stage 1 MD, via conda-forge.
  all        everything above

environment:
  HVI_SRC_DIR   where to clone (default ~/src)
  HVI_ENVFILE   where to write env vars (default ~/.hvi_emp_solvers.env)

Not handled here: the NVIDIA driver (needs root and a reboot) and iSALE
(needs a licence application that in practice went unanswered -- use m2c).
Both are reported, not attempted.
EOF
}

# ---------------------------------------------------------------------------
# GPU driver: report only.
# ---------------------------------------------------------------------------
check_driver() {
    head1 "NVIDIA driver (report only -- needs root, not attempted here)"
    local kernel; kernel="$(uname -r)"

    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
        ok "driver working: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        return 0
    fi

    if ! ls /sys/bus/pci/devices/*/vendor >/dev/null 2>&1 \
       || ! grep -lq 0x10de /sys/bus/pci/devices/*/vendor 2>/dev/null; then
        warn "no NVIDIA card on the PCI bus -- CPU-only machine, nothing to fix"
        return 0
    fi

    bad "NVIDIA card present but the driver is not usable"
    if [[ ! -d "/lib/modules/$kernel/build" ]]; then
        say "  Cause: kernel headers for $kernel are missing, so DKMS cannot build."
        say "  Fix (needs root):"
        say "      sudo apt install linux-headers-$kernel"
        say "      sudo dkms autoinstall && sudo reboot"
    else
        say "  Headers are present. Try:"
        say "      sudo dkms autoinstall && sudo reboot"
        say "  If the build errors, read /var/lib/dkms/nvidia/*/build/make.log"
    fi
    say ""
    say "  Everything below still installs and runs on CPU. Only the CUDA"
    say "  builds need the driver, so this is not blocking."
    return 0
}

# ---------------------------------------------------------------------------
# conda-forge packages
# ---------------------------------------------------------------------------
pkg_install() {   # $1 = package, $2 = import test (optional)
    local pkg="$1" test="${2:-}"
    if ! resolve_pkg_mgr; then
        if [[ $DRY -eq 1 ]]; then
            warn "no micromamba/mamba/conda on PATH -- --bootstrap-mamba, or Miniforge"
            MISSING+=("micromamba/mamba/conda (for $pkg)")
        else
            bad "$pkg: no micromamba, mamba or conda on PATH"
            say "      scripts/install_solvers.sh --bootstrap-mamba"
            FAILED+=("$pkg")
        fi
        return 1
    fi
    # micromamba keys off MAMBA_ROOT_PREFIX, not CONDA_PREFIX. Inside an
    # activated conda env a bare `micromamba install` targets the wrong
    # prefix and fails only later, at import. Name the prefix explicitly.
    local -a argv=("$PKG_MGR" install -y)
    if [[ "$PKG_MGR" == micromamba && -n "${CONDA_PREFIX:-}" ]]; then
        argv+=(-p "$CONDA_PREFIX")
    fi
    argv+=(-c conda-forge "$pkg")

    run "${argv[@]}" || {
        bad "$pkg: install failed"; FAILED+=("$pkg"); return 1; }
    if [[ -n "$test" && $DRY -eq 0 ]]; then
        if python -c "$test" >/dev/null 2>&1; then ok "$pkg importable"
        else warn "$pkg installed but the import test failed"; fi
    fi
    DONE+=("$pkg"); return 0
}

do_lammps() {
    head1 "LAMMPS -- Stage 1 (molecular dynamics)"
    if python -c "import lammps" >/dev/null 2>&1; then
        ok "already importable"; SKIPPED+=(lammps); return 0
    fi
    pkg_install lammps "import lammps"
}

do_paraview() {
    head1 "ParaView -- .vti viewer"
    if have paraview || have pvpython; then
        ok "already installed: $(command -v paraview || command -v pvpython)"
        SKIPPED+=(paraview); return 0
    fi
    pkg_install paraview ""
}

do_warpx() {
    head1 "WarpX -- Stage 3 (EM PIC), conda-forge CPU build"
    if python -c "import pywarpx" >/dev/null 2>&1; then
        ok "pywarpx already importable"; SKIPPED+=(warpx); return 0
    fi
    say "  Note: the conda-forge package is CPU-only (feedstock issue 89)."
    say "  For a CUDA build use Spack or CMake -- docs/INSTALLING_SOLVERS.md section 2."
    say "  There is NO pywarpx package on PyPI; pip install pywarpx returns 404."
    pkg_install warpx "import pywarpx"
}

# ---------------------------------------------------------------------------
# Idefix
# ---------------------------------------------------------------------------
do_m2c() {
    head1 "M2C -- Stage 1 (multi-material flow + non-ideal Saha ionisation)"
    local dir="$SRC/m2c"

    if [[ -f "$dir/CMakeLists.txt" ]]; then
        ok "already cloned at $dir"
    else
        need git m2c "apt install git" || return 1
        need cmake m2c "mamba install -c conda-forge cmake" || return 1
        run mkdir -p "$SRC"
        run git clone https://github.com/kevinwgy/m2c.git "$dir" || {
            bad "clone failed"; FAILED+=(m2c); return 1; }
    fi

    # PETSc is the dependency that actually goes wrong. Say so before the
    # build fails with a cmake error nobody can read.
    if [[ $DRY -eq 0 && -z "${PETSC_DIR:-}" ]]; then
        if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/include/petsc" ]]; then
            export PETSC_DIR="$CONDA_PREFIX"
            say "  using PETSC_DIR=$CONDA_PREFIX"
        else
            warn "PETSC_DIR is unset and no conda PETSc found."
            warn "M2C will not configure without it. Either:"
            say  "      mamba install -c conda-forge petsc openmpi"
            say  "      export PETSC_DIR=\$CONDA_PREFIX"
            say  "  or, with apt:  sudo apt install petsc-dev libopenmpi-dev"
            MISSING+=("PETSc (for m2c)")
            FAILED+=(m2c)
            return 1
        fi
    fi

    local build="$dir/build"
    run mkdir -p "$build"
    if [[ $DRY -eq 0 && -x "$build/m2c" ]]; then
        ok "already built: $build/m2c"
    elif [[ $DRY -eq 0 ]]; then
        ( cd "$build" && cmake .. ) >"$build/cmake.log" 2>&1 || {
            bad "cmake failed; last 15 lines of $build/cmake.log:"
            sed -e 's/^/      /' "$build/cmake.log" | tail -15
            FAILED+=(m2c); return 1; }
        build_log "$build/build.log" make -C "$build" -j "$JOBS" || {
            FAILED+=(m2c); return 1; }
        if [[ -x "$build/m2c" ]]; then
            ok "M2C builds"
        else
            bad "make succeeded but no m2c executable appeared"
            FAILED+=(m2c); return 1
        fi
    fi

    record_env "export M2C_HOME=$build"

    # Verify the generated input grammar against this checkout -- cheap, and
    # it catches an upstream rename before it becomes a silently-ignored
    # keyword in a multi-hour run.
    if [[ $DRY -eq 0 ]] && have python; then
        if python -m hvi_emp.solvers.m2c_stage1 --check "$dir" >/dev/null 2>&1
        then
            ok "generated input grammar matches this checkout"
        else
            warn "input-grammar check reported differences; run"
            say  "      python -m hvi_emp.solvers.m2c_stage1 --check $dir"
        fi
    fi

    DONE+=(m2c)
}

do_idefix() {
    head1 "Idefix -- Stage 2 (MHD, spherical, GPU via Kokkos)"
    local dir="$SRC/idefix"

    if [[ -f "$dir/CMakeLists.txt" ]]; then
        ok "already cloned at $dir"
    else
        need git idefix "apt install git" || return 1
        need cmake idefix "mamba install -c conda-forge cmake" || return 1
        run mkdir -p "$SRC"
        # --recurse-submodules is NOT optional: Kokkos is a submodule and
        # cmake fails confusingly without it.
        run git clone --recurse-submodules \
            https://github.com/idefix-code/idefix.git "$dir" || {
            bad "clone failed"; FAILED+=(idefix); return 1; }
    fi

    if [[ $DRY -eq 0 && ! -f "$dir/src/kokkos/CMakeLists.txt" ]]; then
        warn "Kokkos submodule is empty -- fetching it"
        run git -C "$dir" submodule update --init --recursive
    fi

    record_env "export IDEFIX_DIR=$dir"

    # Smoke test: build the Orszag-Tang MHD problem, which ships with Idefix.
    local test_dir="$dir/test/MHD/OrszagTang"
    if [[ $DRY -eq 0 && ! -d "$test_dir" ]]; then
        warn "no $test_dir to smoke-test against; the clone looks incomplete"
        warn "reporting Idefix as present but UNVERIFIED"
    fi
    if [[ $DRY -eq 0 && -d "$test_dir" ]]; then
        say "  building the Orszag-Tang test as a smoke test (CPU)"
        if [[ -x "$test_dir/idefix" ]]; then
            ok "already built: $test_dir/idefix"
        else
            ( cd "$test_dir" && IDEFIX_DIR="$dir" cmake "$dir" -DIdefix_MHD=ON ) \
                >"$test_dir/cmake.log" 2>&1 || {
                bad "cmake failed; last 15 lines of $test_dir/cmake.log:"
                sed -e 's/^/      /' "$test_dir/cmake.log" | tail -15
                FAILED+=(idefix); return 1; }
            build_log "$test_dir/build.log" make -C "$test_dir" -j "$JOBS" || {
                FAILED+=(idefix); return 1; }
            if [[ -x "$test_dir/idefix" ]]; then
                ok "Idefix builds and the MHD test compiled"
            else
                bad "make succeeded but no idefix executable appeared"
                FAILED+=(idefix); return 1
            fi
        fi
    fi
    DONE+=(idefix)
    say "  GPU build (after the driver works):"
    say "      cmake \$IDEFIX_DIR -DIdefix_MHD=ON -DKokkos_ENABLE_CUDA=ON \\"
    say "            -DKokkos_ARCH_<YOURARCH>=ON"
    say "      python -m hvi_emp doctor   # prints the right arch for your card"
}

# ---------------------------------------------------------------------------
# OpenMHD
# ---------------------------------------------------------------------------
do_openmhd() {
    head1 "OpenMHD -- Stage 2 (MHD, Cartesian, second opinion)"
    local dir="$SRC/OpenMHD"

    if [[ -f "$dir/param.h" ]]; then
        ok "already cloned at $dir"
    else
        need git openmhd "apt install git" || return 1
        run mkdir -p "$SRC"
        run git clone https://github.com/zenitani/OpenMHD.git "$dir" || {
            bad "clone failed"; FAILED+=(openmhd); return 1; }
    fi

    record_env "export OPENMHD=$dir"

    local prob="$dir/2D_basic"
    if [[ $DRY -eq 0 && -f "$prob/a.out" && -f "$prob/ap.out" ]]; then
        ok "already built: $(cd "$prob" && echo *.out | tr ' ' ',')"
        SKIPPED+=(openmhd)
        return 0
    fi

    if ! have mpif90 && ! have gfortran; then
        if [[ $DRY -eq 1 ]]; then
            warn "no Fortran compiler -- mamba install -c conda-forge gfortran openmpi"
            MISSING+=("gfortran/mpif90 (for openmhd)")
        else
            bad "openmhd: no Fortran compiler -- mamba install -c conda-forge gfortran openmpi"
            FAILED+=(openmhd)
        fi
        return 1
    fi

    if [[ $DRY -eq 0 ]]; then
        say "  building 2D_basic serially (produces a.out and ap.out)"
        # NOT parallel. OpenMHD's hand-written Makefile does not declare the
        # Fortran .mod dependencies (mainp.o needs parallel.mod), so `make -j`
        # races and fails intermittently on exactly this kind of tree. A
        # serial build of ~15 files costs seconds; a flaky one costs an hour
        # of confusion.
        build_log "$prob/build.log" make -C "$prob" || {
            FAILED+=(openmhd); return 1; }

        # Trust the artifacts, not make's exit status.
        if [[ -f "$prob/a.out" && -f "$prob/ap.out" ]]; then
            ok "built: a.out (serial), ap.out (MPI)"
        elif [[ -f "$prob/a.out" ]]; then
            warn "built a.out but not ap.out -- the MPI target failed."
            warn "Serial runs work. For MPI: check mpif90 in $prob/Makefile"
        else
            bad "make reported success but produced no executable"
            FAILED+=(openmhd); return 1
        fi
    fi
    DONE+=(openmhd)
    say "  note: OpenMHD installs no executable and none is named 'OpenMHD'."
    say "  Each problem directory builds a.out/ap.out in place, which is why"
    say "  \$OPENMHD must point at the source tree for doctor to find it."
}

# ---------------------------------------------------------------------------
# PIConGPU
# ---------------------------------------------------------------------------
do_picongpu() {
    head1 "PIConGPU -- Stage 3 (EM PIC, GPU-native)"
    local dir="$SRC/picongpu"

    if [[ -d "$dir/bin" ]]; then
        ok "already cloned at $dir"
    else
        need git picongpu "apt install git" || return 1
        run mkdir -p "$SRC"
        run git clone \
            https://github.com/ComputationalRadiationPhysics/picongpu.git \
            "$dir" || { bad "clone failed"; FAILED+=(picongpu); return 1; }
    fi

    record_env "export PICSRC=$dir"
    record_env "export PATH=\$PICSRC/bin:\$PATH"
    record_env "export PIC_EXAMPLES=\$PICSRC/share/picongpu/examples"
    record_env "export PIC_BACKEND=cuda"

    say ""
    say "  PIConGPU needs boost, CMake, an MPI and openPMD-api/ADIOS2 before"
    say "  pic-build will work. In this conda environment:"
    say ""
    say "      mamba install -c conda-forge boost cmake openmpi \\"
    say "                    openpmd-api adios2 zlib"
    say ""
    say "  Then, per simulation:"
    say "      source $ENVFILE"
    say "      pic-create \$PIC_EXAMPLES/KelvinHelmholtz ~/picInputs/hviEMP"
    say "      cp -r <deck>/include <deck>/etc ~/picInputs/hviEMP/"
    say "      cd ~/picInputs/hviEMP && pic-build -b \"cuda:<arch>\""
    say ""
    say "  This one genuinely takes an afternoon. Everything else works"
    say "  without it, and the bridge writes the decks regardless."
    DONE+=(picongpu)
}

# ---------------------------------------------------------------------------

main() {
    local targets=() bootstrap=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run|-n) DRY=1 ;;
            --jobs|-j) shift; JOBS="$1" ;;
            --bootstrap-mamba) bootstrap=1 ;;
            --version|-V) say "install_solvers.sh build $BUILD"; return 0 ;;
            -h|--help) usage; return 0 ;;
            -*)
                say "unknown option: $1"
                say "(this script is build $BUILD -- if you expected an"
                say " option that is missing, you are running an older copy)"
                usage; return 2 ;;
            *) targets+=("$1") ;;
        esac
        shift
    done

    # --bootstrap-mamba on its own is a complete request: get the fast
    # installer and stop. Only default to "all" when no target was named
    # and nothing else was asked for.
    local bootstrap_only=0
    [[ $bootstrap -eq 1 && ${#targets[@]} -eq 0 ]] && bootstrap_only=1
    [[ ${#targets[@]} -eq 0 && $bootstrap_only -eq 0 ]] && targets=(all)

    if [[ " ${targets[*]} " == *" all "* ]]; then
        targets=(m2c lammps paraview warpx idefix openmhd picongpu)
    fi

    [[ $DRY -eq 1 ]] && say "${YLW}dry run: nothing will be installed${RST}"
    say "build:      $BUILD"
    say "clone root: $SRC"
    say "env file:   $ENVFILE"
    say "make jobs:  $JOBS"

    [[ $bootstrap -eq 1 ]] && bootstrap_mamba
    report_pkg_mgr
    if [[ $bootstrap_only -eq 1 ]]; then
        say ""
        say "Now install solvers with, e.g.:"
        say "      scripts/install_solvers.sh lammps paraview"
        [[ ${#FAILED[@]} -gt 0 ]] && return 1
        return 0
    fi

    check_driver

    head1 "iSALE (report only -- optional, and not the recommended route)"
    if [[ -n "${ISALE:-}" ]] || have iSALE2D; then
        ok "iSALE found"
    else
        say  "not installed, and not needed. Access requests went unanswered;"
        say  "  m2c covers the same stage with no gatekeeper, plus oblique"
        say  "  impacts and coupled ionisation. If you want iSALE anyway:"
        say  "      https://isale-code.github.io/access.html"
    fi

    for t in "${targets[@]}"; do
        case "$t" in
            m2c)      do_m2c ;;
            lammps)   do_lammps ;;
            paraview) do_paraview ;;
            warpx)    do_warpx ;;
            idefix)   do_idefix ;;
            openmhd)  do_openmhd ;;
            picongpu) do_picongpu ;;
            *) bad "unknown target: $t" ;;
        esac
    done

    head1 "Summary"
    [[ ${#DONE[@]}    -gt 0 ]] && ok   "installed: ${DONE[*]}"
    [[ ${#SKIPPED[@]} -gt 0 ]] && say  "  already present: ${SKIPPED[*]}"
    [[ ${#FAILED[@]}  -gt 0 ]] && bad  "failed: ${FAILED[*]}"
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        warn "prerequisites you need first:"
        for m in "${MISSING[@]}"; do say "      $m"; done
    fi

    if [[ $DRY -eq 0 ]]; then
        say ""
        say "Add the environment variables to your shell, then re-check:"
        say "      source $ENVFILE"
        say "      echo 'source $ENVFILE' >> ~/.bashrc"
        say "      python -m hvi_emp doctor"
    fi

    [[ ${#FAILED[@]} -gt 0 ]] && return 1
    return 0
}

main "$@"
