# Reproduce the two compiler optimizations together

Use `dganellari/icon4py:dycore-optimizations` with the companion
[`dganellari/gt4py:dycore-fusion-passes`](https://github.com/dganellari/gt4py/tree/dycore-fusion-passes).
Both passes are selected through ICON4Py's normal `model_options.py`. The solver
uses its original equations. No archived experiment directory, copied compiler
patch, generated model rewrite or private setup script is required.

The controlled runner is [fusion_benchmark/run.py](../fusion_benchmark/run.py).
It measures the full `solve_nonhydro` granule, excluding diffusion and other
parts of a full model timestep. Device time and synchronized host wall time are
reported separately.

## Prepare the PR pair

Start the appropriate GPU uenv and activate a Python 3.12 environment containing
its working CuPy build. The supplied job wrappers name the uenvs used for the
measurements: ROCm `prgenv-gnu/7.2.3:2804758683` on Beverin and
`icon/26.7:v1@santis` on Santis. Compiler, Ninja, CMake and grid-data access are
required. The usual ICON4Py grid fixture downloads missing benchmark grids.
Keep the vendor CuPy installation; do not install a CUDA wheel on Beverin.

From a directory for the two checkouts:

```bash
git clone --branch dycore-optimizations https://github.com/dganellari/icon4py.git
git clone --branch dycore-fusion-passes https://github.com/dganellari/gt4py.git
cd icon4py

# Install this checkout's workspace and test/I/O dependencies into the active
# GPU venv. --inexact retains its vendor packages.
uv sync --active --frozen --group test --extra io --inexact

GT4PY_REV=$(python -c 'import json; print(json.load(open("amd_scripts/fusion_benchmark/stack.json"))["gt4py_commit"])')
DACE_REV=$(python -c 'import json; print(json.load(open("amd_scripts/fusion_benchmark/stack.json"))["dace_commit"])')
git -C ../gt4py checkout --detach "$GT4PY_REV"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps \
  "dace @ git+https://github.com/GridTools/dace.git@$DACE_REV"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --editable ../gt4py
```

For existing checkouts, fetch and update the two PR branches first, then use the
same installation steps. Do not reset or overwrite local changes. The compiler
pin is `403f9d996b4ab7435b6989bd004fcba39e7a1bf4`; the existing measured DaCe base
is `5115128a73dc518071dbe9580b63d382540efe46`. These are also recorded in
[stack.json](../fusion_benchmark/stack.json). The GT4Py pin includes the CuPy
scalar-conversion warning fix that was previously an undocumented local edit.

A later ordinary `uv sync` can replace these compiler versions. Reapply the
pinned installation after syncing. The runner rejects a wrong compiler revision,
local compiler edits, an ICON4Py package imported from another checkout, or a
CuPy runtime for the wrong vendor. It also checks a CuPy reduction and a small
CMake build before compiling the granule. `--check` checks the GPU environment
without running the granule; use it inside your GPU allocation.

## Run the combined comparison on both meshes

Submit from the ICON4Py checkout, with the GPU venv activated. You submit and
manage the jobs. Each command runs original versus both passes on regional/120
and global/120, using a fresh process and build directory for each mesh.

```bash
# Beverin / MI300A
sbatch --export=ALL,VENV_PATH="$VIRTUAL_ENV" \
  amd_scripts/fusion_benchmark/run_amd.sh

# Santis / GH200
sbatch --export=ALL,VENV_PATH="$VIRTUAL_ENV" \
  amd_scripts/fusion_benchmark/run_nvidia.sh
```

The wrappers request eight hours and account `csstaff`; override account, time
or uenv with `sbatch` flags if your allocation differs. They include the measured
workspace/layout settings and AMD compiler environment setup. They do not fetch,
install or alter source code in the job.

If already inside a GPU allocation with the working compiler environment, use:

```bash
python amd_scripts/fusion_benchmark/run.py --platform amd
# On GH200 use --platform nvidia.
```

Logs are `dycore_fusion_<job>_<node>.out` and
`fusion-results/<vendor>_<job>/<mesh>/<comparison>/run.log`. The latter prints
quartet progress. First compilation can dominate runtime; the CMake wrapper
unblocks inherited SIGCHLD and terminates a stalled compiler process group after
20 minutes. Do not modify either source checkout while the job is running.

## Measure each contribution separately

Select one comparison per job, for example:

```bash
sbatch --export=ALL,VENV_PATH="$VIRTUAL_ENV" \
  amd_scripts/fusion_benchmark/run_amd.sh --comparisons theta
sbatch --export=ALL,VENV_PATH="$VIRTUAL_ENV" \
  amd_scripts/fusion_benchmark/run_amd.sh --comparisons solver
```

Use the NVIDIA wrapper for GH200. Add `--grids regional` or `--grids global` to
limit the run. Multiple comparisons can share one job, for example
`--comparisons combined theta solver`; allow more compilation time. Every
comparison has its own build directory and process. Inputs are checked to match
between comparisons of the same mesh within a run.

| Comparison           | Arm A: theta, solver | Arm B: theta, solver | Question                                     |
| -------------------- | -------------------- | -------------------- | -------------------------------------------- |
| `combined` (default) | 0, 0                 | 1, 1                 | Total benefit of both passes                 |
| `theta`              | 0, 0                 | 1, 0                 | Theta pass alone                             |
| `solver`             | 0, 0                 | 0, 1                 | Solver pass alone                            |
| `solver-increment`   | 1, 0                 | 1, 1                 | Additional solver benefit with theta enabled |

Outside this benchmark, set `ICON4PY_DACE_THETA_FUSION=1` and/or
`ICON4PY_DACE_SOLVER_FUSION=1` before constructing the model. Both default to off.
Theta selection is restricted to `compute_rho_theta_pgrad_and_update_vn`; solver
selection is restricted to the predictor and corrector solver programs, with
bounded `field_operator` scope.

## What the experiment checks and saves

The runner retains the published measurement method:

- Same allocated model fields for both arms; deterministic fixture inputs.
- Restore arrays and primitive model state, including the Rayleigh cached
  timestep, before validation and every block. Compare all 148 state fields
  between arms at `rtol=1e-11`, `atol=1e-12`, recording the actual maximum error.
  Historical runs matched exactly; the runner does not assume exact equality.
- Twelve balanced ABBA/BAAB quartets, with interleaved A/A control quartets.
  Each block has five warmups and ten timed calls. State evolves within a block,
  identically to the published method; restoration is outside the measured calls.
- Sum per-program device medians for each block; independently record
  synchronized granule wall time. Compute uncertainty across quartets, not
  individual calls. Retain controls and order diagnostics.
- Save the normal backend options, generated SDFGs and code for target programs,
  initial-state fingerprints, package versions and source hashes. Verify sources
  stayed unchanged. A failed validation or incomplete run produces no `COMPLETE`.

Read `RESULTS.md` first, then `RESULTS.json`. A performance effect is labelled
resolved only if its interval excludes zero, it exceeds the conservative A/A
threshold, the matched-control adjustment agrees, and no order sensitivity is
detected. `COMPLETE` means the experiment completed its checks; it does **not**
mean the optimization was faster.

Raw per-call/per-program samples are in each `timing.json`; optimized SDFGs and
code are in `generated/A/` and `generated/B/`. A structurally unchanged program
may reuse its compiled artifact. The two arms never rename the program or
rewrite generated code to force a difference.

## Published reference and new replay status

| Mesh / levels  | MI300A device reduction | MI300A wall reduction | GH200 device reduction | GH200 wall reduction |
| -------------- | ----------------------: | --------------------: | ---------------------: | -------------------: |
| Regional / 120 |                   5.99% |                 4.55% |                  2.06% |    0.88%, unresolved |
| Global / 120   |                   4.81% |                 4.60% |                  4.45% |                4.23% |

These are the earlier controlled compiler-only runs: AMD 641726/644950 and
NVIDIA 873329/875596. See [REVIEW.md](REVIEW.md), [GLOBAL_REVIEW.md](GLOBAL_REVIEW.md)
and [COMPILER_FUSION_RESULTS.json](COMPILER_FUSION_RESULTS.json). Full historical
captures remain archived separately; they are not needed for a new run.

The new PR-only runner has local method and CPU compilation checks. **Its full
GPU replay is pending.** It makes the experiment reproducible from the published
source pair; it does not promise identical timing percentages on another node.
No new measurement has replaced the reference table above.

## Local checks

With the two compiler dependencies installed in the test environment:

```bash
python -m pytest -q -p no:tach --benchmark-disable \
  amd_scripts/fusion_benchmark/test_fusion_benchmark.py
python -m pytest -q -p no:tach \
  model/testing/tests/testing/unit_tests/test_grid_preset_levels.py
```

The harness tests cover restored state, validation failures, timer coverage,
controls, option selection and compilation of two configurations of the same
scan program. They use a NumPy stand-in for GPU storage only in method tests;
production runs require CuPy and the selected GPU vendor. GT4Py's own pass tests
and ICON4Py's model-option/solver-reference tests remain in their usual locations.
