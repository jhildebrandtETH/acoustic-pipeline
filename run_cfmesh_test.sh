#!/usr/bin/env bash
# Invoke with bash from Ubuntu/WSL; every output stays in this repository copy.
set -eo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TEST="$REPO/cfmesh_test"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash run_cfmesh_test.sh [--prepare-only]"
    echo "Settings: cfmesh_test/settings.json. Every invocation makes a new case."
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--prepare-only" ) ]]; then
    echo "Unknown option. Use --help." >&2
    exit 2
fi
# Use one consistent installed runtime, without changing the user's shell.
CFTEST_FOAM_BASHRC="${CFTEST_FOAM_BASHRC:-/usr/lib/openfoam/openfoam2512/etc/bashrc}"
if [[ ! -r "$CFTEST_FOAM_BASHRC" ]]; then
    echo "OpenFOAM environment missing: $CFTEST_FOAM_BASHRC" >&2
    echo "Set CFTEST_FOAM_BASHRC to a compatible OpenFOAM installation's etc/bashrc." >&2
    exit 2
fi
# OpenFOAM bashrc is not nounset-safe.
set +e
source "$CFTEST_FOAM_BASHRC"
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
for cmd in python3 cartesianMesh improveMeshQuality createPatch checkMesh tee stdbuf; do
    command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 2; }
done
mkdir -p "$TEST/runs"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
CASE="$TEST/runs/$RUN_ID"
LOG="$TEST/runs/$RUN_ID.log"
# Keep original stdout so each line is printed once while also saving stage and full logs.
exec 3>&1
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
run() {
    local stage="$1"; shift
    printf -v rendered '%q ' "$@"
    say "+ $rendered"
    local codes
    set +e
    stdbuf -oL -eL "$@" 2>&1 | tee -a "$LOG" "$CASE/log.$stage" >&3
    codes=("${PIPESTATUS[@]}")
    set -e
    if (( codes[0] != 0 || codes[1] != 0 )); then
        say "FAILED: $stage (command=${codes[0]}, logging=${codes[1]}). Case retained: $CASE"
        exit 1
    fi
}
say "Standalone cfMesh, no layers. Threads: $OMP_NUM_THREADS"
say "Runtime: ${WM_PROJECT_VERSION:-unknown}"
say "Mesher: $(command -v cartesianMesh)"
say "Case: $CASE"
say "Complete live transcript: $LOG"
# Preparation creates CASE only after input validation.
python3 -u "$TEST/prepare_case.py" "$CASE" 2>&1 | tee -a "$LOG"
printf '%s\n' "$CASE" > "$TEST/latest-case.txt"
cd "$CASE"
if [[ "${1:-}" == "--prepare-only" ]]; then
    say "Prepared only: no volume meshing executed."
    exit 0
fi
run cartesianMesh cartesianMesh
# Confirm the required stop was honoured before proceeding.
if ! grep -q 'Stopping after step edgeExtraction' log.cartesianMesh; then
    say "ERROR: cfMesh did not confirm the no-layer stopping point."
    exit 1
fi
# Volume-only cfMesh smoothing. Zero surface iterations preserves the mapped STL boundary.
run improveMeshQuality improveMeshQuality -nLoops 2 -nIterations 20 -nSurfaceIterations 0
# Merge the six box sides, which preserved sharp box corners during mapping.
run createPatch createPatch -overwrite
run checkMesh checkMesh -allGeometry -allTopology
# Some checkMesh versions exit zero even when a check fails.
run verify python3 "$TEST/verify_mesh.py" "$CASE"
say "PASS: mesh and checks completed. Inspect the blade surface before judging resolution."
say "View: cd \"$CASE\" && paraFoam -builtin"
say "Alternative: open $CASE/cfmesh_test.foam in ParaView."
