#!/usr/bin/env bash
#
# Capacity benchmark runner for the Mario neuroevolution experiment.
#
# Run this ON the JupyterHub box (or any Linux host you plan to run driver.py on).
# It probes what the machine actually gives you, prints the plan, runs the sweep,
# and leaves a log plus JSON/CSV you can bring back.
#
#   ./run_benchmark.sh                 # probe + plan + full run
#   ./run_benchmark.sh --dry-run       # probe + plan only, evaluates nothing
#   ./run_benchmark.sh --evals-per-worker 8      # longer, steadier numbers
#   PYTHON=/opt/conda/bin/python ./run_benchmark.sh
#
# Anything you pass is forwarded to `benchmark.py all`.

set -euo pipefail

cd "$(dirname "$0")"

# --------------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------------
# The venv committed in this repo is a WINDOWS venv (venv/Scripts/python.exe) and is
# unusable on Linux. On the remote you want either a Linux venv at venv/bin/python or
# whatever interpreter JupyterHub gave you.
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "venv/bin/python" ]]; then
    PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

echo "Interpreter: $PY  ($("$PY" --version 2>&1))"

# --------------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------------
missing="$("$PY" - <<'EOF'
import importlib.util   # NOT just `import importlib` -- .util is a submodule
need = ["numpy", "torch", "yaml", "psutil", "dask", "distributed", "ribs", "pandas"]
out = []
for m in need:
    try:
        if importlib.util.find_spec(m) is None:
            out.append(m)
    except (ImportError, ValueError):
        out.append(m)
print(" ".join(out))
EOF
)"

if [[ -n "$missing" ]]; then
    echo
    echo "ERROR: missing packages: $missing"
    echo
    echo "Set one up on this machine first, e.g.:"
    echo "    python3 -m venv venv-linux && . venv-linux/bin/activate"
    echo "    pip install -e ."
    echo "    PYTHON=venv-linux/bin/python ./run_benchmark.sh"
    exit 1
fi

# The repo has to be importable (runner, config, measures, pcg_benchmark).
if ! "$PY" -c "import runner, config" >/dev/null 2>&1; then
    echo "ERROR: cannot import runner/config. Run this from the repo root, and make"
    echo "       sure the package is installed:  pip install -e ."
    exit 1
fi

# --------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------
OUTDIR="$("$PY" -c "from config import get_output_dir; import os; print(os.path.join(get_output_dir(), 'bench'))")"
mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUTDIR/run-$(hostname)-$STAMP.log"

echo "Output dir:  $OUTDIR"
echo "Log:         $LOG"
echo

# Keep the parent shell's own threads out of the measurement. Each sweep point sets
# the worker environment itself; this only affects the driver process here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------
# Strip --dry-run out of the forwarded args: step 2 always passes it, and passing it
# twice would make step 3 a no-op.
DRY=0
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--dry-run" ]]; then DRY=1; else ARGS+=("$a"); fi
done

{
    echo "### run_benchmark.sh  $STAMP"
    echo "### host: $(hostname)"
    echo "### args: $*"
    echo

    echo "############ STEP 1/3: environment probe ############"
    "$PY" benchmark.py probe

    echo "############ STEP 2/3: plan ############"
    "$PY" benchmark.py all --dry-run ${ARGS+"${ARGS[@]}"}

    if [[ "$DRY" -eq 1 ]]; then
        echo "--dry-run requested; stopping before the sweep."
    else
        echo "############ STEP 3/3: sweep ############"
        "$PY" benchmark.py all ${ARGS+"${ARGS[@]}"}
    fi
} 2>&1 | tee "$LOG"

echo
if [[ "$DRY" -eq 1 ]]; then
    echo "Dry run only. Nothing was evaluated; no JSON/CSV written."
    echo "Re-run without --dry-run to execute."
else
    echo "Done. Results:"
    ls -1t "$OUTDIR" | head -5 | sed "s|^|  $OUTDIR/|"
    echo
    echo "The RECOMMENDATION block at the end of the log is the answer."
    echo "Log: $LOG"
fi
