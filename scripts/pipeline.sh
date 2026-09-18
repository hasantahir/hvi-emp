#!/usr/bin/env bash
#
# Run the chain and render the composite, end to end.
#
#   scripts/pipeline.sh                  # run what is cheap, then render
#   scripts/pipeline.sh --render-only    # re-render existing data, run nothing
#   scripts/pipeline.sh --dry-run
#
# Based on Hasan's workflow script, with four changes:
#
#   * --render-only. Re-rendering is pure post-processing: the solver data
#     is untouched and no stage re-runs. When the scene was wrong and the
#     physics was fine, this is all you need, and it takes seconds.
#   * A data check before pvbatch. Black frames have two causes that look
#     identical in the log -- a wrong scene, or empty data -- and only one
#     is worth re-rendering for. check_render_data.py tells them apart in
#     under a second.
#   * Paths resolved from the repository, not hardcoded. The original had
#     /home/hasan/hvi_emp_29-9-17/... baked in, which breaks on the next
#     checkout, on another machine, and for anyone else.
#   * ffmpeg at the end, so the result is a movie rather than 300 PNGs.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HVI_OUT:-$REPO/full_chain}"
FRAMES="${HVI_FRAMES:-$OUT/frames}"
FPS="${HVI_FPS:-12}"
RENDER_ONLY=0
DRY=0
EXTRA=()

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s[ ok ]%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s[warn]%s %s\n' "$YLW" "$RST" "$*"; }
bad()  { printf '  %s[FAIL]%s %s\n' "$RED" "$RST" "$*"; }
step() { printf '\n%s==> %s%s\n' "$BLD" "$*" "$RST"; }
have() { command -v "$1" >/dev/null 2>&1; }

run() {
    if [[ $DRY -eq 1 ]]; then printf '  would run: %s\n' "$*"; return 0; fi
    printf '  + %s\n' "$*"
    "$@"
}

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;$d'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --render-only) RENDER_ONLY=1 ;;
        --dry-run|-n)  DRY=1 ;;
        --out)         shift; OUT="$1"; FRAMES="$OUT/frames" ;;
        --fps)         shift; FPS="$1" ;;
        -h|--help)     usage; exit 0 ;;
        *)             EXTRA+=("$1") ;;      # passed through to run_full_chain
    esac
    shift
done

CHAIN="$REPO/scripts/run_full_chain.py"
CHECK="$REPO/scripts/check_render_data.py"
COMPOSITE="$OUT/composite.py"
PATTERN="$FRAMES/composite.%04d.png"

say "${BLD}HVI-EMP pipeline${RST}"
say "  repo    : $REPO"
say "  out     : $OUT"
say "  frames  : $FRAMES"
[[ ${#EXTRA[@]} -gt 0 ]] && say "  extra   : ${EXTRA[*]}"

# ---------------------------------------------------------------------------
if [[ $RENDER_ONLY -eq 0 ]]; then
    step "Solver stages"
    run python "$CHAIN" --out "$OUT" --run ${EXTRA[@]+"${EXTRA[@]}"}
else
    step "Solver stages -- SKIPPED (--render-only)"
    say "  Existing solver output is reused. Re-rendering does not"
    say "  recompute anything: no stage re-runs and no data is rewritten."
fi

# ---------------------------------------------------------------------------
step "Is there anything to render?"
if [[ -f "$CHECK" ]]; then
    if ! run python "$CHECK" "$OUT"; then
        bad "the data is unusable -- re-rendering cannot fix this"
        say "  Fix the stage that produced it, then run again."
        exit 1
    fi
else
    warn "check_render_data.py not found; skipping the pre-flight"
fi

# ---------------------------------------------------------------------------
step "Composite scene"
run mkdir -p "$FRAMES"
run python "$CHAIN" --out "$OUT" --render --movie "$PATTERN"

if [[ $DRY -eq 0 && ! -f "$COMPOSITE" ]]; then
    bad "no composite script at $COMPOSITE"
    say "  --render writes it only when at least one chapter has data."
    exit 1
fi

# ---------------------------------------------------------------------------
step "Render (pvbatch)"
if ! have pvbatch; then
    bad "pvbatch not on PATH"
    say "  micromamba install -p \$CONDA_PREFIX -c conda-forge paraview"
    say "  The scene is written; run it later with:"
    say "      pvbatch --force-offscreen-rendering $COMPOSITE"
    exit 1
fi
run pvbatch --force-offscreen-rendering "$COMPOSITE"

if [[ $DRY -eq 0 ]]; then
    n=$(find "$FRAMES" -name 'composite.*.png' 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        bad "pvbatch wrote no frames"
        exit 1
    fi
    ok "$n frames in $FRAMES"

    # A frame that is entirely one colour is the black-frame failure. Worth
    # saying so here rather than after you have watched the movie.
    if have python; then
        python - "$FRAMES" <<'PY' || true
import glob, os, sys
d = sys.argv[1]
fs = sorted(glob.glob(os.path.join(d, "composite.*.png")))
if fs:
    mid = fs[len(fs) // 2]
    sz = os.path.getsize(mid)
    # A 1920x1080 PNG of a single flat colour compresses to a few kB.
    if sz < 20000:
        print("  \033[33m[warn]\033[0m %s is only %d bytes -- that is what a "
              "blank\n         frame looks like. Check the scene before "
              "assembling." % (os.path.basename(mid), sz))
PY
    fi
fi

# ---------------------------------------------------------------------------
step "Assemble"
MP4="$OUT/composite.mp4"
if have ffmpeg; then
    run ffmpeg -y -loglevel error -framerate "$FPS" \
        -i "$PATTERN" -c:v libx264 -pix_fmt yuv420p -crf 18 "$MP4"
    [[ $DRY -eq 0 ]] && ok "$MP4"
else
    warn "ffmpeg not on PATH -- frames written, movie not assembled"
    say "      ffmpeg -framerate $FPS -i '$PATTERN' \\"
    say "             -c:v libx264 -pix_fmt yuv420p -crf 18 '$MP4'"
fi

step "Done"
