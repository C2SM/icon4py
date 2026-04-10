# Branch Summary: `scientific_validation_distributed_driver_hannes`

Branched from `scientific_validation_distributed_driver` at commit `0ae91d930` ("adjust for r2b9").

## Goal

Run the Jablonowski-Williamson (JW) test case at scale with the icon4py standalone driver. Changes fall into two categories: **correctness** (fixing bugs that caused wrong or rank-count-dependent results in multi-rank runs) and **scalability** (reducing init time and enabling larger grids/rank counts).

## Correctness fixes

### 1. init_w bug fix

- `testcases/utils.py`: Changed edge upper bound from `Zone.INTERIOR` to `Zone.END` in `init_w`, ensuring all local edges (including halo) are included in the divergence computation.

### 2. wgtfac_e halo exchange fix (from upstream `edd455f16`)

- `metrics_factory.py`: Changed `do_exchange=False` to `do_exchange=True` for wgtfac_e, ensuring halo values are correct after computation.
- `mpi_decomposition.py`: Fixed `recv_buffer` allocation in `GlobalReductions._reduce` to use `array_ns.empty(1, dtype=buffer.dtype)` instead of `array_ns.empty_like(local_red_val)`.
- `test_parallel_grid_manager.py`: Re-enabled halo checking for WGTFAC_E (previously skipped due to the missing exchange).

### 3. w exchange after diffusion (from upstream `e5c9bb867`)

- `standalone_driver.py`: Added halo exchange for `prognostic_states.next.w` after diffusion run, ensuring w halo values are up-to-date for subsequent stencils.

## Decomposition reproducibility fixes

Changes to make results independent of MPI rank count (same answer regardless of decomposition).

### 4. Reproducible global reductions

- **`ReproducibleGlobalReductions`** class in `mpi_decomposition.py`: gathers full local buffers to rank 0, sorts values, sums in deterministic order, broadcasts result. MIN/MAX delegate to standard `Allreduce` (already order-independent).
- **`--reproducible-reductions`** CLI flag in `main.py` to enable this mode.
- **Owner-mask filtering** in `geometry.py`: mean providers (`mean_edge_length`, `mean_cell_area`, etc.) now exclude halo elements before reduction, fixing double-counting that caused rank-count-dependent means.

### 5. NumPy replacements for init-time GPU stencils

Replaced two GT4Py stencils in `testcases/initial_condition.py` with NumPy equivalents to eliminate GPU FMA non-determinism during initialization:

- **`cell_2_edge_interpolation`** (eta_v cell-to-edge): replaced with explicit NumPy weighted sum using `e2c` connectivity and `c_lin_e` coefficients.
- **`edge_2_cell_vector_rbf_interpolation`** (RBF u/v reconstruction): replaced with NumPy dot product using `C2E2C2E` connectivity and RBF coefficients.

**Result**: All init fields (rho, theta_v, exner, w, vn, u, v) are now **bitwise identical** across rank counts.

### 6. pack_data improvements (from upstream `e5c9bb867`)

- `standalone_driver.py`: Added `_compute_global_index_mapping()` to precompute global index mappings for CellDim and EdgeDim once at init, avoiding redundant MPI gathers per field per output call.
- `pack_data()` now accepts `prognostic_state`, includes `vn` and `w` fields, and uses dimension-aware local/global indexing via the precomputed mappings.

## Scalability improvements

### 7. StructuredDecomposer and decomposition options

- **`StructuredDecomposer`** added in `decomposer.py`: exploits ICON's structured cell ordering to assign contiguous block groups to ranks. Sub-millisecond execution vs ~215s for METIS at r2b8. Supported rank counts: any divisor of `20*R^2 * 4^k` for k=0,...,B.
- Default decomposer is METIS (more flexible rank counts); **`--structured-decomposition`** CLI flag enables the structured variant when rank count is compatible.
- TODO: verify that StructuredDecomposer produces correct results — not yet validated end-to-end.

- Note: if jitted backends supported symbolic domain ranges, all init-related computations could be pre-compiled once and reused across scales/decompositions, which would drastically reduce init time.

### 8. Configurable init backend

- **`--init-backend`** CLI flag: allows specifying a separate GT4Py backend for factory initialization (geometry, interpolation, metrics). Defaults to the main `--icon4py-backend` if not provided. Useful for using `gtfn` during init to avoid slow DaCe JIT while running timestepping on GPU.

### 9. Vectorized init computations

Replaced Python-level element-wise loops with vectorized NumPy operations to reduce init time at large grid sizes:

- **`grid_manager.py`**: E2C2V (diamond vertex) construction vectorized.
- **`interpolation_fields.py`**: Vectorized `_create_inverse_neighbor_index`, `compute_cells_aw_verts`, `compute_lsq_pseudoinv`, `compute_lsq_weights_c`, `compute_lsq_coeffs` (eliminated triple-nested loops).
- **`rbf_interpolation.py`**: Vectorized RBF coefficient computation.
- **`compute_diffusion_metrics.py`**: Vectorized diffusion metrics computation.

## Driver usability improvements

### 10. Driver output improvements

- `standalone_driver.py`: Added `_dump_output` and `_should_dump` methods with `OutputFrequency` enum (none/hourly/daily/final).
- `driver_states.py`: Added `OutputFrequency` enum and `DriverTimers` for structured timer reporting.
- `main.py`: Added `--output-frequency`, `--dtime` CLI flags.

### 11. Script generation

- `scripts/generate_driver_script.py`: Generates SLURM submission scripts with validation that rank counts are compatible with `StructuredDecomposer`.

## Debugging instrumentation

### 12. Profiling and monitoring

- **Fine-grained init timers** (WARNING-level log lines) in `grid_manager.py` for profiling grid construction steps.
- **GPU memory logging** in `standalone_driver.py`.

## Verification results

| Field   | Init comparison (1 vs 4 ranks) | Notes |
|---------|-------------------------------|-------|
| rho     | IDENTICAL | Bitwise match |
| theta_v | IDENTICAL | Bitwise match |
| exner   | IDENTICAL | Bitwise match |
| w       | IDENTICAL | After NumPy replacement |
| vn      | IDENTICAL | After NumPy replacement |
| u       | IDENTICAL | After NumPy replacement |
| v       | IDENTICAL | After NumPy replacement |

**After 10 timesteps** (with `--reproducible-reductions`): ~1e-8 to 1e-7 relative differences remain. TODO: need to investigate whether the wgtfac_e exchange fix resolves these.

## Upstream commits incorporated

From `upstream/scientific_validation_distributed_driver`:
- `e5c9bb867` — w exchange, pack_data improvements, init_w bug fix
- `edd455f16` — wgtfac_e do_exchange fix, recv_buffer fix

## Upstream commits NOT incorporated

- `578543fb6`, `12296e31a`, `d38c1ece1` — script fixes (our branch uses `generate_driver_script.py` instead)
- `147b62794`, `200e24500` — output path cleanup (minor, partially covered by our `_dump_output`)

## Proposed tasks for upstreaming

- Vectorization of slow NumPy computations in interpolation, RBF, diffusion metrics, and grid construction
- StructuredDecomposer (needs end-to-end testing before merging)
- Symbolic domain sizes for compiled backends to avoid recompilation when domain size changes — would eliminate init-time JIT cost
- Timing utilities for init phase profiling