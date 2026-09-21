# Use and benchmark the GT4Py fusion passes from ICON4Py

Run from `dganellari/icon4py:dycore-optimizations`, based on C2SM `mi300_opt`.
The original solver equations are retained. Both optimizations are independently
opt-in compiler transformations in [dganellari/gt4py:dycore-fusion-passes](https://github.com/dganellari/gt4py/tree/dycore-fusion-passes).
There are no compiler patch files to apply in ICON4Py.

## Install the companion compiler in your existing GPU environment

Use a working ICON4Py GPU environment with the vendor's CuPy package and compilers.
On your local setup or GPU machine, install the reviewed compiler revision into
that environment (after the normal ICON4Py dependency setup):

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps \
  'gt4py @ git+https://github.com/dganellari/gt4py.git@24ad90d2d0065c0a270f924aed6367b346aeb5db'
```

The measured DaCe base is `5115128a73dc518071dbe9580b63d382540efe46` from
`GridTools/dace`. Keep the working vendor environment, workspace allocation and
launch settings fixed when comparing variants. The repository's standard lock
still selects released GT4Py; a subsequent `uv sync` can replace the custom
compiler, so verify the installation after environment changes. `--no-deps`
above preserves an already configured stack; it does not create one from scratch.

Verify the two compiler capabilities before submitting a job:

```bash
python - <<'PY'
import gt4py
from gt4py.next.program_processors.runners.dace import scan_fusion
from gt4py.next.program_processors.runners.dace.transformations import map_fusion_extended
print(gt4py.__file__)
assert callable(scan_fusion.normalize_scan_producers)
assert callable(scan_fusion.fuse_scan_inputs)
assert 'allow_shared_data' in map_fusion_extended.VerticalSplitMapRange.__properties__
PY
```

The archived GPU stack also retained an unrelated `domain_utils.py` warning fix:
convert the CuPy scalar to `float` before `round` when formatting the
non-contiguous-domain warning. That fix is outside the fusion branch. If your
compiler reaches this warning and fails with `round(cupy.ndarray)`, the stack
needs that fix; do not interpret the failure as a fusion result. The measured
environment must retain it when replaying the historical experiments.

## Enable either pass or both

Set the variables before constructing the model/backend, in a fresh process.

| Variant     | ICON4PY_DACE_THETA_FUSION | ICON4PY_DACE_SOLVER_FUSION |
| ----------- | ------------------------: | -------------------------: |
| Original    |                         0 |                          0 |
| Theta only  |                         1 |                          0 |
| Solver only |                         0 |                          1 |
| Combined    |                         1 |                          1 |

Unset means `0`. Theta selection is restricted to
`compute_rho_theta_pgrad_and_update_vn`. Solver selection is restricted to the
predictor and corrector solver programs, using the bounded `field_operator`
scope. Unrelated programs keep their existing options. Missing compiler support
fails clearly when an option is enabled.

## Current-branch benchmark checks

Run the following **inside a GPU allocation you start yourself**, with the
working vendor environment activated and ICON4Py packages installed editable
from this checkout. This benchmarks the current branch and its normal model
configuration; it does not check out or patch an older implementation.

```bash
# Choose regional or global; repeat for each variant from the table.
GRID=regional
VARIANT=combined
export ICON4PY_DACE_THETA_FUSION=1
export ICON4PY_DACE_SOLVER_FUSION=1

OUT="$PWD/benchmark-results/$GRID/$VARIANT"
mkdir -p "$OUT"
export GT4PY_BUILD_CACHE_DIR="$OUT/build"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode=development
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_METRICS_OUTPUT_PATH="$OUT/program-metrics.json"
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592

python -m pytest -sv -p no:tach -m continuous_benchmarking \
  --backend=dace_gpu --grid="icon_benchmark_${GRID}:120" \
  --benchmark-warmup=on --benchmark-min-rounds=50 \
  --benchmark-json="$OUT/benchmark.json" \
  'model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]' \
  > "$OUT/run.log" 2>&1
```

Use a fresh output directory for each repeat. The explicit depth is now respected
by the fixture. The benchmark JSON records host wall time; the metrics JSON
contains instrumented program timings. Do not substitute one for the other or
divide raw timer sums from different round counts: `--benchmark-min-rounds` is a
floor. Generated code and caches are isolated by mesh and variant.

These commands let colleagues exercise either pass separately or together.
They are ordinary benchmark checks, **not a replay of the published paired
experiment**: they do not supply the frozen state restoration, 148-field A/B
comparison or interleaved identical-arm controls. Avoid attributing a small
difference from separate invocations to the optimization without those controls.

## What produced the published results

The controlled measurements used original→both compiler passes and
frontend-solver→compiler-solver comparisons, each with 12 balanced ABBA/BAAB
quartets, interleaved identical-arm controls, restored state, 148-field exact
checks, code-generation audits and unchanged source hashes.

| Mesh / levels  | MI300A job / node  | GH200 job / node   |
| -------------- | ------------------ | ------------------ |
| Regional / 120 | 641726 / nid002706 | 873329 / nid005017 |
| Global / 120   | 644950 / nid002934 | 875596 / nid005231 |

See [GLOBAL_REVIEW.md](GLOBAL_REVIEW.md), [REVIEW.md](REVIEW.md) and
[COMPILER_FUSION_RESULTS.json](COMPILER_FUSION_RESULTS.json). The complete frozen
run bundles and independent `review_results.py` checker remain in the author's
`amd_scripts/compiler_stage_fusion_runs/` experiment archive. They are needed
for exact historical replay and are not included in this PR. The previous
launchers replayed the older frontend experiments at `cdc034acb`, so they have
been removed instead of being presented as reproduction of the current passes.

## Local checks

With the companion compiler and normal test dependencies installed:

```bash
python -m pytest -q model/common/tests/common/test_model_options.py
python -m pytest -q model/testing/tests/testing/unit_tests/test_grid_preset_levels.py
python -m pytest -q --backend=embedded \
  model/atmosphere/dycore/tests/dycore/stencil_tests/test_solve_tridiagonal_matrix_for_w_forward_sweep.py
```

The model-options tests cover default-off behavior, targeted selection, invalid
settings, missing support, callback conflicts and the connection to the normal
auto-optimizer. The numerical solver test retains its independent NumPy
reference at 2, 40 and 120 levels. GPU timing claims refer to the recorded jobs;
this cleanup does not create a new GPU validation result.
