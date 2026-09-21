# Use and benchmark the dycore compiler optimizations

Both optimizations use the normal ICON4Py model configuration. The model keeps
its original equations; GT4Py performs the transformations during compilation.
Use the usual model driver or dycore benchmark with a DaCe backend.

## Enable either optimization or both

Set these before constructing the model/backend:

```bash
export ICON4PY_DACE_THETA_FUSION=1
export ICON4PY_DACE_SOLVER_FUSION=1
```

| Configuration | Theta switch | Solver switch |
| ------------- | -----------: | ------------: |
| Original      |            0 |             0 |
| Theta only    |            1 |             0 |
| Solver only   |            0 |             1 |
| Combined      |            1 |             1 |

Both default to `0`. Normal model construction follows
`SolveNonhydro → setup_program → model_options → GT4Py`:

- Theta fusion registers the restricted shared-output callback for
  `compute_rho_theta_pgrad_and_update_vn` in the existing DaCe auto-optimizer.
- Solver fusion enables `fuse_scan_inputs` with `scan_fusion_scope="field_operator"`
  for the predictor and corrector solver programs. GT4Py applies it to iterator
  IR before lowering to SDFG.

These settings use ICON4Py's configurable DaCe backend, selected as `dace_gpu`
by the benchmark. Passing an already constructed GT4Py backend directly bypasses
ICON4Py's backend customization, as it does for other model-specific options.

The installed GT4Py must contain shared-output and scan-input fusion support.
While those changes are under review, install the companion GT4Py branch into
the existing GPU environment. Once available in a release, use the normal
ICON4Py dependency update to select that release. The current dependency pin is
GT4Py 1.2.1, which predates these additions. Enabling an unsupported option fails
with a clear error; model execution does not check PR numbers or commit hashes.

## Run the existing dycore benchmark

Keep the usual AMD or NVIDIA environment and launch settings. Existing cluster
wrappers can continue calling the same benchmark with the switches above in
the environment. Inside a GPU allocation, the underlying command is:

```bash
# Select global or regional and one configuration from the table above.
GRID=regional
VARIANT=combined
OUT="$PWD/dycore-results/${GRID}-${VARIANT}"
mkdir -p "$OUT"

export GT4PY_BUILD_CACHE_DIR="$OUT/build"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode=development
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_METRICS_OUTPUT_PATH="$OUT/program-metrics.json"

python -m pytest -sv -p no:tach -m continuous_benchmarking \
  --backend=dace_gpu --grid="icon_benchmark_${GRID}:120" \
  --benchmark-warmup=on --benchmark-min-rounds=50 \
  --benchmark-json="$OUT/benchmark.json" \
  'model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]'
```

Repeat in a fresh process for each configuration and mesh, with a separate output
and build directory. `VARIANT` labels the output; the two environment switches
select the transformations. Keep the workspace, layout and compiler settings
unchanged between comparisons. The fixture respects the explicit 120 levels.

`benchmark.json` records synchronized host wall time; `program-metrics.json`
contains instrumented program timings. Compare per-call values, not raw timer
sums across runs: benchmark round counts can differ. Ordinary benchmark runs
exercise the same production code path but do not provide the paired noise
controls used to establish the small published improvements.

## Published results and validation

| Mesh / levels  | MI300A device reduction | MI300A wall reduction | GH200 device reduction | GH200 wall reduction |
| -------------- | ----------------------: | --------------------: | ---------------------: | -------------------: |
| Regional / 120 |                   5.99% |                 4.55% |                  2.06% |    0.88%, unresolved |
| Global / 120   |                   4.81% |                 4.60% |                  4.45% |                4.23% |

These are the earlier controlled compiler-only experiments, using balanced
A/B timings, interleaved A/A controls and restored model state. All 148 checked
fields matched exactly. They remain the reference evidence, not a guarantee of
identical timing percentages on another node. See [REVIEW.md](REVIEW.md),
[GLOBAL_REVIEW.md](GLOBAL_REVIEW.md) and
[COMPILER_FUSION_RESULTS.json](COMPILER_FUSION_RESULTS.json) for the original
run identities, measurement definitions and limitations.

The reviewed GT4Py implementation is `403f9d99` on `dycore-fusion-passes`, based
on `amd_chiplet_setting`; the recorded DaCe base is
`5115128a73dc518071dbe9580b63d382540efe46`. These identify the compiler stack for
review and historical comparison; they are not runtime restrictions. GT4Py also
contains the array-scalar warning fix required by the GPU compilation path.

Model-option tests live in
[the existing common test module](../../model/common/tests/common/test_model_options.py),
and the numerical/structural pass tests live in GT4Py. The new integration has
not yet had a full GPU run through the ordinary benchmark; the published GPU
results above came from the archived experiment harness.
