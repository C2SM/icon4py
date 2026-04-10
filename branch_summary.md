# Branch Summary: `scientific_validation_distributed_driver_hannes`

Branched from `scientific_validation_distributed_driver` at commit `0ae91d930` ("adjust for r2b9").

## Goal

Run the Jablonowski-Williamson (JW) test case at scale with the icon4py standalone driver. Changes cover **correctness** (fixing bugs in multi-rank runs), **decomposition reproducibility** (making results independent of rank count), **scalability** (reducing init time and enabling larger grids/rank counts), and **driver usability** (output, scripting, debugging).

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

## Example of startup (with Metis decomposition), but vectorization applied

```
Backend name used for the model: dace_gpu
BackendLike derived from the backend name: {'backend_factory': <function make_custom_dace_backend at 0x400447310af0>, 'device': <DeviceType.CUDA: 2>}
Initializing the driver
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 11:43:17,874 - standalone_driver.py: initialize_driver   : WARNING TIMER: reading config completed in 0.000s
initializing the grid manager from '/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0017_R02B10_G.nc'
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 11:43:25,110 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: read full grid properties completed in 3.182s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 11:43:44,679 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: read global neighbor tables completed in 19.568s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:42:09,528 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: decomposer (rank mapping) completed in 3504.849s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:42:13,869 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: halo construction completed in 4.341s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:42:29,698 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: local connectivities completed in 15.828s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:42:32,900 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: derived connectivities completed in 3.202s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:43:02,239 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: refinement + domain bounds completed in 29.339s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:43:02,313 - grid_manager.py: _construct_decomposed_grid: WARNING TIMER: icon_grid construction completed in 0.073s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:43:02,313 - grid_manager.py: __call__            : WARNING TIMER: _construct_decomposed_grid completed in 3580.384s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:43:56,111 - grid_manager.py: __call__            : WARNING TIMER: _read_coordinates completed in 53.798s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:47:15,038 - grid_manager.py: __call__            : WARNING TIMER: _read_geometry_fields completed in 198.928s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:47:15,040 - standalone_driver.py: initialize_driver   : WARNING TIMER: initializing grid manager completed in 3837.166s
creating the decomposition info
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:47,232 - standalone_driver.py: initialize_driver   : WARNING TIMER: creating decomposition info completed in 92.192s
initializing the vertical grid
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:47,234 - standalone_driver.py: initialize_driver   : WARNING TIMER: initializing vertical grid completed in 0.001s
initializing the JW topography
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:48,219 - standalone_driver.py: initialize_driver   : WARNING TIMER: initializing JW topography completed in 0.985s
Backend name used for the model: dace_gpu
BackendLike derived from the backend name: {'backend_factory': <function make_custom_dace_backend at 0x400447310af0>, 'device': <DeviceType.CUDA: 2>}
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:48,419 - standalone_driver.py: initialize_driver   : WARNING Using 'dace_gpu' backend for factory initialization
initializing the static-field factories
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:49,998 - standalone_driver.py: initialize_driver   : WARNING TIMER: initializing static-field factories completed in 1.579s
initializing granules
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:51,583 - factory.py: get                 : WARNING TIMING: mean_cell_area took 1.585s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:48:51,583 - driver_utils.py: initialize_granules : WARNING TIMER: creating cell geometry completed in 1.585s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:52:19,084 - factory.py: get                 : WARNING TIMING: inverse_of_edge_length took 207.501s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:52:19,123 - factory.py: get                 : WARNING TIMING: inverse_of_length_of_dual_edge took 0.039s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:54:54,864 - factory.py: get                 : WARNING TIMING: vertex_vertex_length took 155.741s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:54:54,877 - factory.py: get                 : WARNING TIMING: inverse_of_vertex_vertex_length took 155.753s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 12:56:21,121 - factory.py: get                 : WARNING TIMING: x_component_of_edge_normal_unit_vector took 86.243s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:00:02,906 - factory.py: get                 : WARNING TIMING: eastward_component_of_edge_normal_on_vertex took 308.029s (provider: SparseFieldProviderWrapper)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:00:02,940 - factory.py: get                 : WARNING TIMING: eastward_component_of_edge_tangent_on_vertex took 0.034s (provider: SparseFieldProviderWrapper)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:00:37,667 - factory.py: get                 : WARNING TIMING: eastward_component_of_edge_normal_on_cell took 34.726s (provider: SparseFieldProviderWrapper)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:00:37,697 - factory.py: get                 : WARNING TIMING: eastward_component_of_edge_tangent_on_cell took 0.030s (provider: SparseFieldProviderWrapper)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:00:49,864 - factory.py: get                 : WARNING TIMING: edge_area took 12.167s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:00,585 - factory.py: get                 : WARNING TIMING: coriolis_parameter took 10.721s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:13,120 - factory.py: get                 : WARNING TIMING: eastward_component_of_edge_normal took 12.535s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:13,120 - driver_utils.py: initialize_granules : WARNING TIMER: creating edge geometry completed in 741.537s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:14,081 - factory.py: get                 : WARNING TIMING: bilinear_edge_cell_weight took 0.961s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:25,289 - factory.py: get                 : WARNING TIMING: x_component_of_vertex took 11.207s (provider: EmbeddedFieldOperatorProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:36,339 - factory.py: get                 : WARNING TIMING: x_coordinate_of_edge_center took 11.049s (provider: EmbeddedFieldOperatorProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:43,027 - factory.py: get                 : WARNING TIMING: mean_dual_area took 6.688s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:43,027 - factory.py: get                 : WARNING TIMING: characteristic_length took 6.688s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:43,028 - factory.py: get                 : WARNING TIMING: rbf_scale_vertex took 6.690s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:45,189 - factory.py: get                 : WARNING TIMING: rbf_interpolation_coefficient_vertex_1 took 31.107s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:46,367 - factory.py: get                 : WARNING TIMING: geometrical_factor_for_divergence took 1.178s (provider: EmbeddedFieldOperatorProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:46,947 - factory.py: get                 : WARNING TIMING: geometrical_factor_for_nabla_2_scalar took 0.580s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:47,169 - factory.py: get                 : WARNING TIMING: interpolation_coefficient_from_cell_to_edge took 0.222s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:01:48,524 - factory.py: get                 : WARNING TIMING: geometrical_factor_for_green_gauss_gradient_x took 1.578s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:01,976 - factory.py: get                 : WARNING TIMING: nudging_coefficients_for_edges took 13.451s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:01,976 - driver_utils.py: initialize_granules : WARNING TIMER: creating diffusion interpolation state completed in 48.855s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:04,713 - factory.py: get                 : WARNING TIMING: vertical_coordinates_on_half_levels took 2.663s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:16,024 - factory.py: get                 : WARNING TIMING: height took 13.975s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:30,636 - factory.py: get                 : WARNING TIMING: theta_ref_mc took 28.660s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:52,992 - factory.py: get                 : WARNING TIMING: wgtfac_c took 22.356s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:02:55,292 - factory.py: get                 : WARNING TIMING: max_nbhgt took 2.299s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:03:08,864 - factory.py: get                 : WARNING TIMING: ddxn_z_half_e took 13.572s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:03:19,744 - factory.py: get                 : WARNING TIMING: ddxn_z_full took 24.452s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:03:33,938 - factory.py: get                 : WARNING TIMING: maxslp took 38.646s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:03:38,493 - factory.py: get                 : WARNING TIMING: bilinear_cell_average_weight took 4.555s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:03:52,656 - factory.py: get                 : WARNING TIMING: maxslp_avg took 57.364s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:03:54,187 - factory.py: get                 : WARNING TIMING: zd_intcoef took 61.195s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:11,501 - factory.py: get                 : WARNING TIMING: zd_diffcoef took 17.314s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:11,501 - driver_utils.py: initialize_granules : WARNING TIMER: creating diffusion metric state completed in 129.525s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:12,980 - factory.py: get                 : WARNING TIMING: cell_to_vertex_interpolation_factor_by_area_weighting took 1.478s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:13,679 - factory.py: get                 : WARNING TIMING: e_flux_average took 0.699s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:13,890 - factory.py: get                 : WARNING TIMING: geometrical_factor_for_curl took 0.209s (provider: EmbeddedFieldOperatorProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:13,902 - factory.py: get                 : WARNING TIMING: eastward_component_of_edge_tangent took 0.012s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:13,915 - factory.py: get                 : WARNING TIMING: pos_on_tplane_e_x took 0.025s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:14,620 - factory.py: get                 : WARNING TIMING: rbf_interpolation_coefficient_edge took 0.704s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:14,620 - driver_utils.py: initialize_granules : WARNING TIMER: creating solve nonhydro interpolation state completed in 3.119s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:25,831 - factory.py: get                 : WARNING TIMING: mask_prog_halo_c took 11.211s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:26,119 - factory.py: get                 : WARNING TIMING: rayleigh_w took 0.287s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:39,328 - factory.py: get                 : WARNING TIMING: exner_exfac took 13.210s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:39,393 - factory.py: get                 : WARNING TIMING: weighting_factor_for_quadratic_interpolation_to_cell_surface took 0.065s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:04:50,173 - factory.py: get                 : WARNING TIMING: inverse_of_functional_determinant_of_metrics_on_full_levels took 10.780s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:05:07,043 - factory.py: get                 : WARNING TIMING: ddxt_z_half_e took 16.868s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:05:07,359 - factory.py: get                 : WARNING TIMING: implicitness_weight_for_exner_and_w_in_vertical_dycore_solver took 17.184s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:05:18,166 - factory.py: get                 : WARNING TIMING: explicitness_weight_for_exner_and_w_in_vertical_dycore_solver took 27.993s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:05:30,850 - factory.py: get                 : WARNING TIMING: d_exner_dz_ref_ic took 12.684s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:05:51,611 - factory.py: get                 : WARNING TIMING: functional_determinant_of_metrics_on_interface_levels took 20.760s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:06:04,054 - factory.py: get                 : WARNING TIMING: d2dexdz2_fac1_mc took 12.443s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:06:25,461 - factory.py: get                 : WARNING TIMING: rho_ref_me took 21.407s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:06:25,884 - factory.py: get                 : WARNING TIMING: flat_idx_max took 0.423s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:05,255 - factory.py: get                 : WARNING TIMING: zdiff_gradp took 99.795s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:15,327 - factory.py: get                 : WARNING TIMING: nflat_gradp took 10.071s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:32,878 - factory.py: get                 : WARNING TIMING: distance_for_pressure_gradient_extrapolation took 17.551s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:44,932 - factory.py: get                 : WARNING TIMING: functional_determinant_of_metrics_on_full_levels_on_edges took 12.054s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:46,724 - factory.py: get                 : WARNING TIMING: ddxt_z_full took 1.792s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:58,299 - factory.py: get                 : WARNING TIMING: wgtfac_e took 11.575s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:08:58,961 - factory.py: get                 : WARNING TIMING: weighting_factor_for_quadratic_interpolation_to_edge_center took 0.661s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:09:11,865 - factory.py: get                 : WARNING TIMING: horizontal_mask_for_3d_divdamp took 12.904s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:09:12,003 - factory.py: get                 : WARNING TIMING: scaling_factor_for_3d_divergence_damping took 0.138s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:09:23,320 - factory.py: get                 : WARNING TIMING: coeff1_dwdz took 11.317s (provider: ProgramFieldProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:09:26,446 - factory.py: get                 : WARNING TIMING: coeff_gradekin took 3.126s (provider: NumpyDataProvider)
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:09:26,446 - driver_utils.py: initialize_granules : WARNING TIMER: creating solve nonhydro metric state completed in 311.827s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:09:40,866 - driver_utils.py: initialize_granules : WARNING TIMER: creating diffusion granule completed in 14.420s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:12:51,351 - driver_utils.py: initialize_granules : WARNING TIMER: creating solve nonhydro granule completed in 190.485s
rank=0/128 [MPI_COMM_WORLD]  2026-04-08 13:13:55,058 - factory.py: get                 : WARNING TIMING: x_component_of_cell_center took 63.690s (provider: EmbeddedFieldOperatorProvider)

```

## Proposed tasks for upstreaming

- Vectorization of slow NumPy computations in interpolation, RBF, diffusion metrics, and grid construction
- StructuredDecomposer (needs end-to-end testing before merging)
- Symbolic domain sizes for compiled backends to avoid recompilation when domain size changes — would eliminate init-time JIT cost
- Timing utilities for init phase profiling