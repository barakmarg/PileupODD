#!/usr/bin/env bash
#
# One-shot setup for a fresh user on the Weizmann (WIPP) cluster.
#
# Activates the shared conda environment, then builds and installs CLUEstering,
# which is a C++/CUDA extension with no usable wheel. Ends by printing a
# one-event command that exercises the whole pipeline end to end.
#
#   ./setup_cluster.sh              # set up, skipping work already done
#   ./setup_cluster.sh --smoke      # ... and run the end-to-end check
#   ./setup_cluster.sh --force      # rebuild CLUEstering even if it looks fine
#
# Safe to re-run: every step checks whether it is already satisfied.
#
# The GPU backend is the only usable one -- CLUEstering's CPU backends emit
# infinite values and collapse most hits into a single cluster -- so this script
# treats a missing CUDA backend as a hard failure rather than falling back.

set -euo pipefail

CONDA_ENV="/usr/wipp/conda/24.5.0u/envs/common"
CONDA_ACTIVATE="/usr/wipp/conda/24.5.0u/bin/activate"
CLUE_UPSTREAM="https://gitlab.cern.ch/kalos/CLUEstering.git"
CLUE_TAG="2.9.0"
CLUE_LOCAL_COPY="/storage/agrp/barakma/CLUEstering"
CLUE_BUILD_DIR="${CLUE_BUILD_DIR:-$HOME/CLUEstering}"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda/bin}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SMOKE=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) RUN_SMOKE=1 ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

step()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()    { printf '    \033[32mok\033[0m   %s\n' "$*"; }
warn()  { printf '    \033[33mwarn\033[0m %s\n' "$*"; }
die()   { printf '\n\033[31merror\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
step "Shared conda environment"

[ -d "$CONDA_ENV" ] || die "environment not found at $CONDA_ENV
This script targets the WIPP cluster. Elsewhere, create your own environment
with 'pip install -r requirements.txt' and build CLUEstering as below."

# The env's activate.d hooks reference unbound variables (MKL_INTERFACE_LAYER),
# so nounset has to come off for the duration of the source.
set +u
# shellcheck disable=SC1090
source "$CONDA_ACTIVATE" "$CONDA_ENV"
set -u
ok "$(python -V 2>&1) from $CONDA_ENV"

# ---------------------------------------------------------------------------
step "Pipeline Python dependencies"

missing=$(python - <<'PY'
import importlib
need = {"polars": "polars", "numpy": "numpy", "pyarrow": "pyarrow", "yaml": "PyYAML",
        "tqdm": "tqdm", "datasketches": "datasketches", "psutil": "psutil"}
print(" ".join(dist for mod, dist in need.items()
                if not importlib.util.find_spec(mod)))
PY
)
if [ -n "$missing" ]; then
    warn "missing: $missing -- installing into your user site"
    python -m pip install --user --quiet $missing
    ok "installed"
else
    ok "all present (polars, numpy, pyarrow, PyYAML, tqdm, datasketches, psutil)"
fi

# ---------------------------------------------------------------------------
step "GPU and CUDA toolchain"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "no GPU visible here. You can still build, but the pipeline cannot run"
    warn "on this node -- submit to a GPU queue (see 'Running at scale' in the README)."
fi

# nvcc must be on PATH *before* CMake configures CLUEstering: it probes with
# check_language(CUDA) and silently omits the CUDA backend when absent.
if ! command -v nvcc >/dev/null 2>&1; then
    [ -x "$CUDA_BIN/nvcc" ] || die "nvcc not found on PATH nor at $CUDA_BIN/nvcc.
Without it CLUEstering builds CPU-only backends, which cannot produce a valid
dataset. Set CUDA_BIN=/path/to/cuda/bin and re-run."
    export PATH="$CUDA_BIN:$PATH"
fi
ok "nvcc: $(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------
step "CLUEstering"

# Report the compiled backends, or 'ABSENT'. CLUE_GPU_CUDA is the one that counts.
clue_backends() {
    python - <<'PY' 2>/dev/null || echo ABSENT
import pathlib
import CLUEstering
lib = pathlib.Path(CLUEstering.__file__).parent / "lib"
names = sorted(p.name.split(".")[0] for p in lib.glob("*.so"))
print(" ".join(names) if names else "NO_BACKENDS")
PY
}

backends="$(clue_backends)"
if [ "$FORCE" -eq 0 ] && [[ "$backends" == *CLUE_GPU_CUDA* ]]; then
    ok "already installed with the CUDA backend"
    ok "backends: $backends"
else
    if [ "$FORCE" -eq 1 ]; then
        warn "--force given: rebuilding"
    elif [ "$backends" = "ABSENT" ]; then
        warn "not installed -- building from source (several minutes)"
    else
        warn "installed but WITHOUT the CUDA backend ($backends) -- rebuilding"
    fi

    if [ ! -d "$CLUE_BUILD_DIR" ]; then
        if [ -r "$CLUE_LOCAL_COPY/setup.py" ]; then
            step "  copying source from $CLUE_LOCAL_COPY"
            cp -r "$CLUE_LOCAL_COPY" "$CLUE_BUILD_DIR"
        else
            step "  cloning $CLUE_UPSTREAM at $CLUE_TAG"
            git clone --quiet "$CLUE_UPSTREAM" "$CLUE_BUILD_DIR"
            git -C "$CLUE_BUILD_DIR" checkout --quiet "$CLUE_TAG"
        fi
    else
        ok "reusing existing source at $CLUE_BUILD_DIR"
    fi

    # extern/pybind11 is a git submodule; the build fails outright without it.
    step "  submodules (extern/pybind11)"
    git -C "$CLUE_BUILD_DIR" submodule update --init --recursive --quiet
    [ -f "$CLUE_BUILD_DIR/extern/pybind11/CMakeLists.txt" ] \
        || die "extern/pybind11 is still missing -- the build cannot proceed."
    ok "present"

    # setup.py runs cmake -B build -DBUILD_PYTHON=ON itself, then builds.
    # alpaka 2.1.0 is downloaded during configure via CMake FetchContent, so
    # this step needs working network access.
    step "  compiling and installing (needs network for the alpaka download)"
    ( cd "$CLUE_BUILD_DIR" && python -m pip install --user . ) \
        || die "CLUEstering build failed. The log above has the reason; a common
cause is a compiler too old for C++20."

    backends="$(clue_backends)"
    ok "backends: $backends"
fi

[[ "$backends" == *CLUE_GPU_CUDA* ]] || die "CLUE_GPU_CUDA was not built.
This means CMake did not find a CUDA compiler at configure time. Confirm 'nvcc
--version' works, then re-run with --force. A CPU-only install cannot produce a
valid dataset."

# ---------------------------------------------------------------------------
step "Import check"

( cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" python -c "
import colliderml_pflow, colliderml_pflow.cli, colliderml_pflow.pipeline
print(f'    colliderml_pflow {colliderml_pflow.__version__} imports cleanly')
" ) || die "the package does not import -- run this script from inside the repo."

# ---------------------------------------------------------------------------
QUICK_CMD="python -m colliderml_pflow preprocess --config configs/quick_check.yaml"

if [ "$RUN_SMOKE" -eq 1 ]; then
    step "End-to-end check (one event, under a minute)"
    ( cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" eval "$QUICK_CMD" ) \
        || die "the end-to-end run failed -- see the output above."
    written=$(ls "$REPO_ROOT"/data/quick_check/*.parquet 2>/dev/null | wc -l)
    [ "$written" -eq 4 ] || die "expected 4 output tables, found $written."
    ok "4 tables written to data/quick_check/"
    printf '\n\033[1;32mSetup complete and verified end to end.\033[0m\n'
else
    printf '\n\033[1;32mSetup complete.\033[0m\n'
    printf '\nVerify it end to end -- one event, under a minute:\n\n'
    printf '    source %s %s\n' "$CONDA_ACTIVATE" "$CONDA_ENV"
    printf '    cd %s\n' "$REPO_ROOT"
    printf '    %s\n' "$QUICK_CMD"
    printf '\nIt should finish with "[ALL SHARDS DONE]" and write 4 parquet files\n'
    printf 'to data/quick_check/. Then see the README for a real run.\n'
fi
