#!/usr/bin/env bash
# Set up a parallel beverin env for icon4py + gt4py PR 2578 under ROCm 7.2.
#
# Layered on top of amd_scripts/install_icon4py_venv.sh (which is the
# canonical 7.1 path documented in AMD_INTRODUCTION.md):
#   - uses uv sync --extra rocm7, Python 3.12, same as that script
#   - then overrides gt4py with PR 2578 (iomaganaris:extend_loopblocking)
#   - applies the hlb2noscan patch to model_options.py
#   - installs rocprofiler-compute requirements (7.2 path, not 7.1)
#   - patches CuPy hip_workaround.cuh same as the canonical script
#
# Run from inside icon4py-rocm72 with the ROCm 7.2 uenv started:
#   uenv start --view default b2550889de318ab5
#   cd /capstor/scratch/cscs/gandanie/git/icon/icon4py-rocm72
#   bash amd_scripts/setup_env_rocm72.sh
#
# Pass --clean to wipe and start over.

set -eu

# --- config ---
GT4PY_REPO="https://github.com/iomaganaris/gt4py.git"
GT4PY_BRANCH="extend_loopblocking"
GT4PY_DIR="/capstor/scratch/cscs/gandanie/git/icon/gt4py-pr2578"
VENV_DIR=".venv_rocm72"
PATCH_FILE="model/common/src/icon4py/model/common/model_options.py"

CLEAN=0
[ "${1:-}" = "--clean" ] && CLEAN=1

# --- preflight ---
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$ICON4PY_GIT_ROOT"

if [ "$(basename "$PWD")" != "icon4py-rocm72" ]; then
    echo "WARN: cwd is $(basename "$PWD"), expected icon4py-rocm72" >&2
fi

if [ -z "${UENV_MOUNT_LIST:-}" ] && [ -z "${UENV_VIEW:-}" ]; then
    echo "ERROR: no uenv detected. Start the ROCm 7.2 uenv first:" >&2
    echo "  uenv start --view default b2550889de318ab5" >&2
    exit 1
fi

echo "==> uenv:    ${UENV_VIEW:-${UENV_MOUNT_LIST:-unknown}}"
echo "==> python:  $(command -v python3.12 || true) ($(python3.12 --version 2>&1))"
echo "==> hipcc:   $(command -v hipcc || true)"

# --- install uv locally (same as canonical script) ---
export PATH="$PWD/bin:$PATH"
if [ ! -x "$PWD/bin/uv" ]; then
    echo "==> installing uv at $PWD/bin/uv"
    curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$PWD/bin" sh
else
    echo "==> uv already installed at $PWD/bin/uv"
fi

# --- clean if requested ---
if [ "$CLEAN" -eq 1 ]; then
    echo "==> --clean: removing $VENV_DIR and $GT4PY_DIR"
    rm -rf "$VENV_DIR" "$GT4PY_DIR"
fi

# --- clone gt4py PR 2578 first, so uv can install it directly ---
if [ ! -d "$GT4PY_DIR/.git" ]; then
    echo "==> cloning gt4py PR 2578 to $GT4PY_DIR"
    git clone -b "$GT4PY_BRANCH" "$GT4PY_REPO" "$GT4PY_DIR"
else
    echo "==> gt4py clone exists at $GT4PY_DIR; pulling latest on $GT4PY_BRANCH"
    git -C "$GT4PY_DIR" fetch origin "$GT4PY_BRANCH"
    git -C "$GT4PY_DIR" checkout "$GT4PY_BRANCH"
    git -C "$GT4PY_DIR" pull --ff-only origin "$GT4PY_BRANCH"
fi
echo "==> gt4py HEAD: $(git -C "$GT4PY_DIR" log --oneline -1)"

# --- create venv via uv sync ---
# uv sync targets .venv/ by default; override via UV_PROJECT_ENVIRONMENT.
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
echo "==> uv sync --extra rocm7 (venv: $VENV_DIR)"
uv sync --extra rocm7 --python "$(which python3.12)"

# --- override gt4py with the PR 2578 editable install ---
# uv sync pulled the workspace's pinned gt4py (dace-43!2026.4.20). Replace it
# with our PR 2578 clone in editable mode so we can patch transformations later.
source "$VENV_DIR/bin/activate"
echo "==> overriding gt4py with editable install of $GT4PY_DIR (with [dace] extra)"
uv pip install -e "$GT4PY_DIR[dace]" --reinstall

# --- rocprofiler-compute requirements (7.2 path) ---
ROCPROF_REQ=$(ls -d /user-environment/linux-zen3/rocprofiler-compute-7.2*/libexec/rocprofiler-compute/requirements.txt 2>/dev/null | head -1)
if [ -n "$ROCPROF_REQ" ]; then
    echo "==> installing rocprofiler-compute requirements: $ROCPROF_REQ"
    uv pip install -r "$ROCPROF_REQ"
else
    echo "==> WARN: rocprofiler-compute 7.2 requirements.txt not found; skipping"
fi

# --- patch CuPy hip_workaround.cuh (same as canonical script) ---
CUPY_HIP_WORKAROUND=$(python -c "import cupy, os; print(os.path.join(os.path.dirname(cupy.__file__), '_core', 'include', 'cupy', 'hip_workaround.cuh'))" 2>/dev/null || true)
if [ -n "$CUPY_HIP_WORKAROUND" ] && [ -f "$CUPY_HIP_WORKAROUND" ]; then
    if grep -q "Patched: force mask-stripping" "$CUPY_HIP_WORKAROUND"; then
        echo "==> CuPy hip_workaround.cuh already patched"
    else
        sed -i 's/#if (HIP_VERSION < 60200000) || defined(HIP_DISABLE_WARP_SYNC_BUILTINS)/#if 1  \/\/ Patched: force mask-stripping for all ROCm versions (CuPy 14.0.1 bug)/' "$CUPY_HIP_WORKAROUND"
        echo "==> patched CuPy hip_workaround.cuh"
    fi
else
    echo "==> CuPy not installed in venv; skipping hip_workaround.cuh patch"
fi

# --- apply hlb2noscan patch to model_options.py ---
if grep -q "MAIN_HORIZONTAL_DIMENSIONS" "$PATCH_FILE"; then
    echo "==> hlb2noscan patch already applied to $PATCH_FILE; skipping"
else
    echo "==> applying hlb2noscan patch to $PATCH_FILE"
    python - <<'PY'
import pathlib
p = pathlib.Path("model/common/src/icon4py/model/common/model_options.py")
s = p.read_text()
s = s.replace(
    "from icon4py.model.common import model_backends",
    "from icon4py.model.common import dimension as dim, model_backends",
)
needle = 'optimization_args.setdefault("gpu_block_size_1d", (256, 1, 1))'
add = (
    '\n        optimization_args["blocking_dims"] = list(dim.MAIN_HORIZONTAL_DIMENSIONS.values())'
    '\n        optimization_args["blocking_size"] = 2'
    '\n        optimization_args["blocking_only_if_independent_nodes"] = False'
)
if needle not in s:
    raise SystemExit(f"ERROR: needle not found: {needle!r}")
s = s.replace(needle, needle + add, 1)
p.write_text(s)
print("patched.")
PY
fi

echo
echo "==> setup complete."
echo
echo "Re-activate later with:"
echo "  uenv start --view default b2550889de318ab5"
echo "  cd $(pwd)"
echo "  source ${VENV_DIR}/bin/activate"
echo "  source amd_scripts/setup_env.sh   # NOTE: still 7.1-hardcoded; review"
echo
echo "Sanity check:"
echo "  python -c 'import gt4py.next, icon4py.model.common; print(gt4py.next.__file__)'"
