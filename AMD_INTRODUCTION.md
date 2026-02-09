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
+-------------------------------------------------------+---------------------+-------------+-----------------+
| GT4Py Program                                         | MI300A [persistent] | GH200 (ns)  | Acceleration    |
|                                                       |       (ns)          |             |                 |
+=======================================================+=====================+=============+=================+
| compute_advection_in_horizontal_momentum              | 0.00440             | 0.000176    | 25.01           |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| vertically_implicit_solver_at_corrector_step          | 0.00088             | 0.000578    | 1.53            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| vertically_implicit_solver_at_predictor_step          | 0.00080             | 0.000555    | 1.44            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| compute_rho_theta_pgrad_and_update_vn                 | 0.00074             | 0.000410    | 1.81            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| compute_horizontal_velocity_quantities_and_fluxes     | 0.00049             | 0.000329    | 1.51            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| compute_perturbed_quantities_and_interpolation        | 0.00039             | 0.000265    | 1.47            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| compute_advection_in_corrector_vertical_momentum      | 0.00030             | 0.000212    | 1.43            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| compute_interpolation_and_nonhydro_buoy               | 0.00023             | 0.000137    | 1.71            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
| compute_hydrostatic_correction_term                   | 0.00003             | 0.000029    | 1.03            |
+-------------------------------------------------------+---------------------+-------------+-----------------+
```

**Warning** By default some of `GT4Py Programs` executed on the `MI300A` show a dramatic slowdown meanwhile in all of them the standard deviation in `MI300A` is much higher than `GH200`. We figured out that the souruce of the issue is a call to `hipMallocAsync` which allocates temporaries necessary for each program. The call to this HIP API has a very high variability and some times it takes much longer (100x times) to execute. For the `MI300A` results above we have disabled these allocations/deallocations taking place for each `GT4Py Programs` to see more clear the runtimes however this is not feasible for real simulations, where the memory footprint of the temporary data of all `GT4Py Programs` cannot be preallocated. Since in `GH200` the memory allocations and deallocations are taken into account in the timings, the above results should be taken with a grain of salt.

We didn't have time yet to look into the `compute_advection_in_horizontal_momentum` regression.

In both cases in the table above we present the median runtimes that are reported over 100 iterations (excluding the first slow one) using a C++ timer as close as possible to the kernel launches.

What is interesting for us to look into is analyzing the performance of the kernels of a specific `GT4Py Program`.
To that end, we selected one of the `GT4Py Programs` that takes most of the time in a production simulation and has kernels with different representative patterns like: neighbor reductions, 2D maps and scans.
This is the `vertically_implicit_solver_at_predictor_step` `GT4Py program` and here is the comparison of its kernels:

```
+------------------------------------------+--------------------------+--------------------------+---------------------------------------------------------------+
| Kernel Name                              | MI300A Median Time (ns)  | GH200 Median Time (ns)   | Acceleration of GH200 over MI300A (MI300A time / GH200 time)  |
+------------------------------------------+--------------------------+--------------------------+---------------------------------------------------------------+
| map_100_fieldop_1_0_0_514                | 240026                   | 125024                   | 1.92                                                          |
| map_115_fieldop_1_0_0_518                | 208261                   | 113056                   | 1.84                                                          |
| map_60_fieldop_0_0_504                   | 148626                   | 86624                    | 1.72                                                          |
| map_0_fieldop_0_0_500                    | 66400                    | 31456                    | 2.11                                                          |
| map_85_fieldop_0_0_506                   | 59680                    | 66336                    | 0.90                                                          |
| map_31_fieldop_0_0_0_512                 | 46960                    | 28768                    | 1.63                                                          |
| map_90_fieldop_0_0_508                   | 26960                    | 20832                    | 1.29                                                          |
| map_91_fieldop_0_0_510                   | 8560                     | 3552                     | 2.41                                                          |
| map_13_fieldop_0_0_498                   | 5600                     | 3744                     | 1.50                                                          |
| map_100_fieldop_0_0_0_0_520              | 5560                     | 5472                     | 1.02                                                          |
| map_115_fieldop_0_0_0_516                | 4560                     | 5184                     | 0.88                                                          |
| map_35_fieldop_0_0_503                   | 3360                     | 1856                     | 1.81                                                          |
+------------------------------------------+--------------------------+--------------------------+---------------------------------------------------------------+
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

The generated code for the results above can be found in `Beverin` in:

`dycore`
```
/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/amd_profiling_solver_persistent_mem # GT4Py cache folder
/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/slurm-247696.out # Slurm output
```

`solver`
```
/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/amd_profiling_solver_regional # GT4Py cache folder
/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/slurm-247518.out # Slurm output
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

- Installing the AMD HIP/ROCm packages for our UENV with Spack required various changes which are done [here](https://github.com/eth-cscs/alps-uenv/pull/273). Maybe it would be worth to discuss with the packaging team how to streamline the spack package installation of some of the packages

- There are some TODOs in the scripts that mention some issues with the profilers. It would be great if you could help us fix them

- The kernel names may vary from execution to execution so in some cases differences in the kernel names can be expected

- The provided scripts are for guidance and should be handled with care
