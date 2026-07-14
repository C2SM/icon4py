# Goal: Bitwise Identical GPU Results for Standalone Driver MPI Tests

Achieve bitwise identical results between single-rank and multi-rank
standalone driver runs on GPU backends (`gtfn_gpu`, `dace_gpu`), mirroring
the reproducibility already established for CPU backends.

## Scope

- **Tests**: `model/standalone_driver/tests/standalone_driver/mpi_tests/test_parallel_standalone_driver.py`
  - `integration` level (1 timestep, JW experiment)
  - `validation` level (7 days, JW experiment)
- **Backends**: `gtfn_gpu`, `dace_gpu`
- **Comparison**: single-rank vs multi-rank field comparison (`vn`, `w`, `exner`, `theta_v`, `rho`)

## Approach

1. **GPU compiler flags** (`ci/base.yml`): Add `NVCC_APPEND_FLAGS` (disable FMA
   contraction, pin IEEE division/sqrt), `CUPY_ACCELERATORS` (disable CUB/cuTENSOR),
   `CUBLAS_WORKSPACE_CONFIG` (deterministic cuBLAS) to the validation CI block,
   alongside the existing `CXXFLAGS=-ffp-contract=off`.

2. **Print instead of fail** (`ci/base.yml`): Set
   `ICON4PY_DALLCLOSE_PRINT_INSTEAD_OF_FAIL=1` in the validation block so
   remaining non-zero diffs are observable rather than hard-failing on the
   first mismatch.

3. **Zero tolerance for GPU** (`model/testing/.../test_utils.py`): Remove the
   `is_cpu_backend` restriction from `get_mpi_comparison_tolerance` so
   `gtfn`/`dace` GPU backends also get `(0.0, 0.0)` tolerance when
   `ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE=1`.

## Stop conditions

- If CI runner infrastructure issues block execution, stop and report.
- If non-zero diffs persist after analysis and up to 3 CI attempts, stop and
  report on remaining differences and likely causes.
- If root cause is unclear (speculation only), stop and report rather than
  making blind changes.
