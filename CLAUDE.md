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

3. **Residual stencil-level differences** (inherent): `neighbor_sum` operations in GT4Py stencils accumulate neighbor contributions in a fixed local order, but the *values* at halo boundaries carry accumulated rounding from prior timesteps. This is inherent to domain decomposition and cannot be fixed without deterministic global evaluation.

**Error magnitudes after fixes (JW r2b5, 10 timesteps):**

- ranks2 vs ranks4: ~1e-9 relative (pressure), ~1e-8 relative (temperature) -- excellent
- ranks1 vs ranks2: ~1e-8 relative (pressure), ~1e-7 relative (temperature) -- good, limited by stencil-level differences

## Development notes

- Comparison script: `scripts/compare_rank_outputs.py` compares pickle dumps field-by-field
- Test data: `ranks{1,2,4}.pkl` (original), `reprod_ranks{1,2,4}.pkl` (with reproducible reductions), `wfix_ranks{1,2,4}.pkl` (with w-init fix)
