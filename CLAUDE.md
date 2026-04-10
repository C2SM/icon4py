# CLAUDE.md - icon4py

## Project overview

icon4py is a Python/GT4Py reimplementation of the ICON atmospheric model. It uses GT4Py's `field_operator` and `scan_operator` abstractions for stencil computations on unstructured grids (icosahedral), with MPI parallelism via GHEX for halo exchange and mpi4py for global reductions.

## Key architecture

- **Grid**: Unstructured icosahedral grid with cell/edge/vertex dimensions and neighbor connectivities (C2E, E2C, V2E, C2E2C, etc.)
- **Domain decomposition**: Each MPI rank owns a subset of cells/edges/vertices plus halo regions. Owner masks (`decomposition_info.owner_mask(dim)`) distinguish owned from halo entries.
- **Halo exchange**: Via GHEX (`GHexMultiNodeExchange`). After stencil computations, halo values must be exchanged before neighbor access.
- **Global reductions**: `GlobalReductions` class in `mpi_decomposition.py` wraps `MPI.Allreduce` for min/max/sum/mean. A `ReproducibleGlobalReductions` variant (added 2026-04) gathers full arrays to rank 0 for bitwise-reproducible sums.
- **Zone indexing**: `h_grid.Zone` defines domain regions ordered from most inclusive (`END` = all local including halos) to most restrictive (`NUDGING`). Key zones: `END` > `HALO` > `LOCAL` > `INTERIOR` > `LATERAL_BOUNDARY_*`.

## Standalone driver

- Entry point: `model/standalone_driver/src/icon4py/model/standalone_driver/main.py`
- Driver init: `standalone_driver.py::initialize_driver()` sets up grid, geometry, interpolation, metrics, and dycore components
- Test case: Jablonowski-Williamson (JW) baroclinic wave, initial condition in `testcases/initial_condition.py`
- CLI uses typer; flags include `--reproducible-reductions` for bitwise-reproducible global sums

## Known issues and investigations

### Distributed reproducibility (2026-04)

When running the same setup with different MPI rank counts (1, 2, 4), dynamic fields (pressure, temperature, u, v) show small differences while static fields (cell_area, dz, z_mc) match exactly.

**Root causes identified:**

1. **Global reduction order dependence** (fixed): `GlobalReductions.mean()` uses `MPI.Allreduce(SUM)` where floating-point summation order depends on rank count. Fix: `ReproducibleGlobalReductions` gathers full arrays to rank 0 and sums deterministically. Enabled via `--reproducible-reductions`.

2. **Halo double-counting in mean computations** (fixed): Geometry mean providers (`mean_edge_length`, `mean_cell_area`, etc.) were passing full local buffers (owned + halo) to the mean reduction, double-counting halo values. Fix: owner masks are now applied before reduction in `geometry.py`.

**Verified (2026-04):**

- Connectivity neighbor ordering is **preserved exactly** across different rank decompositions (verified empirically on R02B04 for C2E2C, C2E, E2C, E2V, E2C2V, E2C2E across 1/2/4 ranks).
- No hidden reductions during timestepping. The only local reduction is `max_vertical_cfl` in `velocity_advection.py`, used for adaptive substep control (has TODO to make it global). Does not trigger for JW at typical resolutions (would show as log.warning).
- Init-phase reductions fully accounted for: 4 geometry means (geometry.py, via `_owned_mean`) + 1 metrics min (metric_fields.py `compute_nflat_gradp`).

**Init field comparison (verified 2026-04, r2b5, 1 rank vs 4 ranks, --dump-init-fields):**

Fields dumped immediately after initialization (before timestepping) using global indices for cross-rank mapping:

| Field   | Result     | Max abs diff | Max rel diff | Notes                                                                 |
| ------- | ---------- | ------------ | ------------ | --------------------------------------------------------------------- |
| rho     | IDENTICAL  | 0            | 0            | Exact bitwise match                                                   |
| theta_v | IDENTICAL  | 0            | 0            | Exact bitwise match                                                   |
| exner   | IDENTICAL  | 0            | 0            | Exact bitwise match                                                   |
| w       | ~identical | 5.15e-19     | —            | 11,560/21,120 cells affected; w values ≈ 1e-6, diffs at sub-ULP level |
| vn      | ~identical | 1.42e-14     | 1.47e-15     | ~1 ULP diffs on ~5–25% of edges depending on level                    |

**Root cause of vn/w init diffs:** GPU JIT compilation artifact. `vn` depends on `eta_v_at_edge`, computed by the GT4Py stencil `cell_2_edge_interpolation`. DaCe compiles separate CUDA kernels for different domain sizes (122,880 edges at 1 rank vs ~31,840 at 4 ranks). The CUDA compiler may make different FMA (fused multiply-add) contraction decisions per kernel, giving results that differ by up to 1 ULP. These propagate through `cos`/`sin`/power operations in the `vn` formula, amplifying to ~1.42e-14. The `w` diffs (~5e-19) are downstream propagation from `vn` through `init_w()`.

**Init conclusion:** Init is effectively identical (diffs at machine epsilon) after NumPy replacements for `cell_2_edge_interpolation` and `edge_2_cell_vector_rbf_interpolation`. With these replacements, all init fields are bitwise identical across rank counts.

**Timestepping diffs (under investigation):** The ~1e-7 relative differences after 10 timesteps were initially attributed to chaotic amplification, but a missing halo exchange for `wgtfac_e` (`do_exchange=False` in `metrics_factory.py`, fixed in `edd455f16`) is a more likely cause. TODO: re-run comparison with the wgtfac_e exchange fix to verify.

**Error magnitudes (JW r2b5, 10 timesteps, --reproducible-reductions enabled):**

- ranks2 vs ranks4: ~1e-9 relative (pressure), ~1e-8 relative (temperature)
- ranks1 vs ranks2: ~1e-8 relative (pressure), ~1e-7 relative (temperature)

## Grid decomposition and performance

### ICON grid structure

- ICON grids with root R and bisection level B have `20*R²` base diamond blocks, each containing `4^B` cells in contiguous index order.
- Cells within each block follow a recursive quad-tree subdivision: groups of `4^k` contiguous cells are always spatially compact.
- Cross-block boundary connectivity is ~6% of all neighbor references at block level.

### Decomposers (`model/common/src/icon4py/model/common/decomposition/decomposer.py`)

- **StructuredDecomposer** (added 2026-04): Exploits the structured cell ordering. Assigns contiguous groups of (sub-)blocks to ranks. Sub-millisecond execution. Supported rank counts: any divisor of `20*R² * 4^k` for k=0,...,B. For R02 grids (80 base blocks): 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 256, 320, 512, ...
- **MetisDecomposer**: Uses pymetis (CSR interface) for graph partitioning. Slow for large grids (~215s at r2b8, expected worse at r2b9). Used as fallback when rank count is not compatible with StructuredDecomposer.
- **SingleNodeDecomposer**: Assigns all cells to rank 0.
- Selection logic in `driver_utils.py::create_grid_manager()`: uses MetisDecomposer by default; `--structured-decomposition` CLI flag opts into StructuredDecomposer when rank count is compatible.

### Grid init performance timers

- `grid_manager.py` has fine-grained TIMER log lines (WARNING level) for all steps in `__call__` and `_construct_decomposed_grid`.
- At r2b8/32 ranks: decomposer was the dominant cost (215s with METIS, now \<1ms with StructuredDecomposer). Derived connectivities is second (~28s).

### Script generation

- `scripts/generate_driver_script.py`: generates SLURM submission scripts. Validates that `--ranks` is compatible with StructuredDecomposer, suggests nearby valid counts on error.
- `DEFAULT_RANKS` table maps (root, bisection) to recommended rank counts, all verified compatible with StructuredDecomposer.

## Development notes

- Comparison script: `scripts/compare_rank_outputs.py` compares pickle dumps field-by-field
- Test data: `ranks{1,2,4}.pkl` (original), `reprod_ranks{1,2,4}.pkl` (with reproducible reductions), `wfix_ranks{1,2,4}.pkl` (with w-init fix)
