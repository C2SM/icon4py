# Icon4py performance on MI300

## Quickstart

```
# Connect to Beverin (CSCS system with MI300A)
ssh beverin.cscs.ch
```

In Beverin:
```
# Enter scratch directory
cd $SCRATCH

# Clone icon4py and checkout the correct branch
git clone git@github.com:C2SM/icon4py.git
cd icon4py
git checkout amd_profiling

# Pull the correct `uenv` image. *!* NECESSARY ONLY ONCE *!*
uenv image pull build::prgenv-gnu/25.12:2288359995

# Start the uenv and mount the ROCm 7.1.0 environment. *!* This needs to be executed before running anything everytime *!*
uenv start --view default prgenv-gnu/25.12:2288359995

# Install the necessary venv
bash amd_scripts/install_icon4py_venv.sh

# Source venv
source .venv/bin/activate

# Source other necessary environment variables
source amd_scripts/setup_env.sh

# Set GT4Py related environment variables
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_granule
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_DYCORE_ENABLE_METRICS="1"
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

# Benchmark dycore
pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=30 \
    --benchmark-json=pytest_benchmark_results.json \
    --backend=dace_gpu \
    --grid=icon_benchmark_regional \
    --benchmark-time-unit=ms \
    --benchmark-min-rounds 100 \
    model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[True-False]

# Print GT4Py timers
python print_gt4py_timers.py dycore_gt4py_program_metrics.json
```

For more information regarding benchmarking read the [Benchmarking](#benchmarking) chapter

## Intro to icon4py and GT4Py

In the following text we will give an overview of [icon4py](https://github.com/C2SM/icon4py), [GT4Py](https://github.com/GridTools/gt4py) and [DaCe](https://github.com/spcl/dace) and how they interact to compile our Python ICON implementation.

### icon4py

`icon4py` is a Python port of `ICON` implemented using the `GT4Py DSL`. Currently in `icon4py` there are only certain parts of `ICON` implemented. The most important being the `dycore`, which is the `ICON` component that takes most of the time to execute.
For this purpose we think it makes more sense to focus in this component.
The `icon4py` dycore implementation consists of ~20 `GT4Py Programs` or stencils. Each one of these programs consists of multiple GPU (CUDA or HIP) kernels and memory allocations/deallocations while in the full `icon4py` code there are also MPI/nccl communications. For now we will focus in the single node execution, so no communication is conducted.

### GT4Py

`GT4Py` is a compilation framework that provides a DSL which is used as frontend to write the stencil computations. This is done using a DSL embedded into Python code in `icon4py` as stated above.
Here is an example of a `GT4Py Program` from `icon4py`: [vertically_implicit_solver_at_predictor_step](https://github.com/C2SM/icon4py/blob/e88b14d8be6eed814faf14c5e8a96aca6dfa991e/model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/vertically_implicit_dycore_solver.py#L219).
`GT4Py` supports multiple backends. These are `embedded` (with numpy/JAX execution), `GTFN` (GridTools C++ implementation) and `DaCe`. For the moment the most efficient is `DaCe` so we'll focus on this one only. The code from the frontend is lowered from the `GT4Py DSL` to CUDA/HIP code after numerous transformations in `GT4Py IR (GTIR)` and then `DaCe Stateful Dataflow Graphs (SDFG)`. The lowering from `GTIR` to `DaCe SDFG` is done using the low level `DaCe` API.

### DaCe

`DaCe` is a programming framework that can take Python code and transform it to an SDFG, which is a representation that is easy to apply dataflow optimizations and achieve good performance in modern CPUs and GPUs. To see more information regarding how the SDFGs look like see the following [link](https://spcldace.readthedocs.io/en/latest/sdfg/ir.html).
`DaCe` includes also a code generator from SDFG to C++, HIP and CUDA code. The HIP generated code is CUDA code hipified basically so there are no big differences between the generated code for CUDA and HIP.


## Benchmarking

For the benchmarking we have focused on the `dycore` component of `icon4py` . We have measured the runtimes for the different `GT4Py Programs` executed in it between an `MI300A` and a `GH200 GPU` below:

```
+--------------------------------------------------------+-----------------+----------------+--------------------------------------------------------------+
| GT4Py Programs                                         | MI300A Time (s) | GH200 Time (s) | Acceleration of GH200 over MI300A (MI300A time / GH200 time) |
+--------------------------------------------------------+-----------------+----------------+--------------------------------------------------------------+
| compute_diagnostics_from_normal_wind                   | 0.000268        | 0.000150       | 1.79                                                         |
| compute_advection_in_predictor_vertical_momentum       | 0.000195        | 0.000129       | 1.51                                                         |
| compute_advection_in_horizontal_momentum               | 0.004871        | 0.000174       | 27.98                                                        |
| compute_perturbed_quantities_and_interpolation         | 0.000433        | 0.000255       | 1.70                                                         |
| compute_hydrostatic_correction_term                    | 0.000034        | 0.000026       | 1.30                                                         |
| compute_rho_theta_pgrad_and_update_vn                  | 0.105237        | 0.000404       | 260.40                                                       |
| compute_horizontal_velocity_quantities_and_fluxes      | 0.000562        | 0.000324       | 1.73                                                         |
| vertically_implicit_solver_at_predictor_step           | 0.011691        | 0.000601       | 19.46                                                        |
| compute_advection_in_corrector_vertical_momentum       | 0.010325        | 0.000209       | 49.51                                                        |
| compute_interpolation_and_nonhydro_buoy                | 0.000253        | 0.000135       | 1.87                                                         |
| apply_divergence_damping_and_update_vn                 | 0.000208        | 0.000114       | 1.83                                                         |
| vertically_implicit_solver_at_corrector_step           | 0.002938        | 0.000592       | 4.96                                                         |
+--------------------------------------------------------+-----------------+----------------+--------------------------------------------------------------+
```

Some of them show a dramatic slowdown in `MI300A` meanwhile in all of them the standard deviation in `MI300A` is much higher than `GH200`. The above are the median runtimes that are reported over 100 iterations (excluding the first slow one) using a C++ timer as close as possible to the kernel launches.

Since we only recently started looking at the results from the MI300A there are some issues to iron out, like the very slow executions compared to GH200 which most likely include some non-GPU kernel overhead.
What would be more interesting to look at is starting from a specific `GT4Py Program` and looking at the performance of each kernel launched from it.
To that end, we selected one of the `GT4Py Programs` that takes most of the time in a production simulation and has kernels with different representative patterns like: neighbor reductions, 2D maps and scans.
This is the `vertically_implicit_solver_at_predictor_step` `GT4Py program` and here is the comparison of its kernels:

```
+-----------------------------+-----------------------+------------------------+-----------------------------------------------------------+
| Name                        | MI300A Avg Time (μs)  | GH200 Mean Time (μs)   | Acceleration GH200 over MI300A (MI300A time / GH200 time) |
+-----------------------------+-----------------------+------------------------+-----------------------------------------------------------+
| map_100_fieldop_1_0_0_514   |                225.20 |                 123.20 |                                                     1.83  |
| map_115_fieldop_1_0_0_518   |                197.40 |                 113.04 |                                                     1.75  |
| map_60_fieldop_0_0_504      |                142.10 |                  86.66 |                                                     1.64  |
| map_85_fieldop_0_0_506      |                 80.45 |                  81.28 |                                                     0.99  |
| map_0_fieldop_0_0_500       |                 63.02 |                  31.68 |                                                     1.99  |
| map_31_fieldop_0_0_0_512    |                 54.46 |                  28.56 |                                                     1.91  |
| map_90_fieldop_0_0_508      |                 25.57 |                  18.62 |                                                     1.37  |
| map_91_fieldop_0_0_510      |                  7.99 |                   3.49 |                                                     2.29  |
| map_100_fieldop_0_0_0_0_520 |                  5.59 |                   5.07 |                                                     1.10  |
| map_13_fieldop_0_0_498      |                  5.32 |                   3.70 |                                                     1.44  |
| map_115_fieldop_0_0_0_516   |                  4.99 |                   5.28 |                                                     0.95  |
| map_35_fieldop_0_0_503      |                  3.62 |                   1.87 |                                                     1.93  |
+-----------------------------+-----------------------+------------------------+-----------------------------------------------------------+
```

The runtimes of the individual kernels are collected using `nsys` and `rocprofv3`.

The benchmarks were run on `Santis` (`GH200 GPU`) and `Beverin` (`MI300A GPU`) using the following uenv images:
- GH200: `icon/25.2:v3` (CUDA 12.6)
- MI300A: `build::prgenv-gnu/25.12:2288359995` (ROCM 7.1.0)

To reproduce the benchmark results on `Beverin` you can follow the instructions below:

```
# Pull the correct `uenv` image. *!* NECESSARY ONLY ONCE *!*
uenv image pull build::prgenv-gnu/25.12:2288359995

# Start the uenv and mount the ROCm 7.1.0 environment. *!* This needs to be executed before running anything everytime *!*
uenv start --view default prgenv-gnu/25.12:2288359995

# Run the whole `dycore` granule and gather the runtimes of the `GT4PY Programs`
sbatch benchmark_dycore.sh
# The script above will generate a json file with the names of the `GT4Py Programs` and their runtimes. The first one is always slow so we skip accounting it in our analysis
# With the following python script you can parse the json file and print the runtimes in a nice form
# python print_gt4py_timers.py dycore_gt4py_program_metrics.json # passing --csv will save them in a csv file

# Run the `vertically_implicit_solver_at_predictor_step` GT4Py program standalone. Notice the `GT4Py Timer Report` table printed from the first `pytest` invocation. The reported timers on this table are as close as possible to the kernel launches of the GT4Py program.
# The following script will benchmark the solver, run `rocprofv3` and collect a trace of it as well as run the `rocprof-compute` tool for all its kernels
sbatch benchmark_solver.sh
```

## Hackathon goals

- Understand what is the bottleneck in our currently generated kernel code
- Discuss what changes we can do either in the code generation, kernel configuration or memory layout to address these bottlenecks and make sure we have reached performance better comparable with GH200
- What further code changes do we have to do to take advantage of the full MI300A performance (shared memory, warp shuffling, etc)
- Fix any issues with ROCm profilers and learn how to effectively use them

## Notes

- To understand the code apart from the analysis the profilers there are the following sources:
  1. Look at the generated HIP code for the `GT4Py program` `vertically_implicit_solver_at_predictor_step` in `<icon4py_root_dir>/amd_profiling_solver/.gt4py_cache/vertically_implicit_solver_at_predictor_step_<HASH>/src/cuda/vertically_implicit_solver_at_predictor_step.cpp`. The code is generated from DaCe automatically and it's a bit too verbose. It would be good to have some feedback on whether the generated code is in a good form for the HIP compiler to optimize.
  2. Look at the generated assembly and HIP kernel characteristics (outputs of `-save-temps -Rpass-analysis=kernel-resource-usage`) in `<icon4py_root_dir>/amd_profiling_solver/.gt4py_cache/vertically_implicit_solver_at_predictor_step_<HASH>/build/vertically_implicit_solver_at_predictor_step_cuda-hip-amdgcn-amd-amdhsa-gfx942.s`.
  3. Look at the `icon4py` frontend code for the `vertically_implicit_solver_at_predictor_step` [here](https://github.com/C2SM/icon4py/blob/e88b14d8be6eed814faf14c5e8a96aca6dfa991e/model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/vertically_implicit_dycore_solver.py#L219)
  4. Look at the generated SDFG by DaCe. This can give a nice overview of the computations and kernels generated. Using [the DaCe documentation](https://spcldace.readthedocs.io/en/latest/sdfg/ir.html) can help you understand what is expressed in the SDFG. The generated SDFG is saved in `<icon4py_root_dir>/amd_profiling_solver/.gt4py_cache/vertically_implicit_solver_at_predictor_step_<HASH>/program.sdfg`. To view the SDFG there is a VSCode plugin (`DaCe IOE`) or you can download it locally and open it in https://spcl.github.io/dace-webclient/.

- Installing the AMD HIP/ROCm packages for our UENV with Spack required various changes which are done [here](https://github.com/eth-cscs/alps-uenv/pull/273)
