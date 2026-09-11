# Running the Jocksch reference tests

How to run the tests of the FFSL-WENO port against A. Jocksch's Fortran capture and his
paper, per backend, and what the numbers were when this was written (2026-09-11, branch
`weno_idealized`, santis, uenv `icon/26.7:v1`, GCC 14.3, numpy 2.4 with OpenBLAS). Companion
to `weno_idealized_scope.md` (what is ported, Fortran <-> Python map) and
`weno_idealized_status.md`. GPU numbers: same day, GH200, CUDA 13.1 of the uenv, cupy 14.0.1
(`cuda13` extra).

## The two test modules

| module | what it compares | data |
|---|---|---|
| `model/atmosphere/tracer_advection/tests/tracer_advection/integration_tests/test_jocksch_reference.py` | **L1** init-time least-squares coefficients, **L2** one `Advection.run` per savepoint (steps 1, 2, 50, 100, tracer 0; tracers 1-3 of the capture are asserted bit-equal to tracer 0, they are the same cylinder advected with the same scheme; tracers are numbered from 0 as in icon4py, so the Fortran's tracers 1-5 are icon4py's 0-4, the last one the initial cylinder), **L3** the 100-step trajectory with the new tracer fed back | the serialbox capture, `datatest` mark |
| `model/driver/tests/driver/integration_tests/test_jocksch_cylinder.py` | the full Python experiment (driver, 100 steps) against the paper's Table 2 (truncation interval) and the Fortran's printed pair-sum error (relative tolerance) | the grid file only |

The reference test compares fields to the Fortran to round-off; the cylinder test compares
one scalar per case to printed digits. Both print their numbers with `-s`.

## Data (not downloadable)

Everything lives next to the Fortran runs under `<workspace>/weno_data/` and is linked into
the icon4py test-data layout by hand (recipe in `weno_idealized_scope.md`, section "Testing
against the Fortran capture"):

- `weno_data/reference/<case>/ser/` -- the capture of one ICON run per
  `(ihadv_tracer, itype_hlimit)` (branch `transport_ajocksch_capture`, built with
  `-Kieee -Mnofma -gpu=nofma`, `nproma = 1`, 10 levels), linked as
  `testdata/ser_icondata/mpitask1_jocksch_cylinder_<case>_v01/ser_data`; `error.txt` next to
  it holds the printed `#` error (the pair sum). `reference/jocksch_grid/<case>/` is the
  same on Andreas' own grid (not used by these tests yet).
- `weno_data/grids/torus_20x22_res5000m[_centred].nc` -- the grid files, linked as
  `testdata/grids/<name>/<name>.nc`; the cylinder test reads the original one directly.
- `weno_data/gt4py_cache/<backend>[_nofma]/` -- one GT4Py build cache per backend and per
  contraction mode (the cache key ignores both).
- `weno_data/slurm/` -- logs (`w5b_<backend>_<what>.log`, `w5c_gtfn_cpu_reference.log`
  `w5d_gtfn_cpu_reference.log` and `w5d_dace_cpu_ihadv132.log` from the runs below, `<jobid>.out` from the GPU jobs;
  their stderr is `<workspace>/slurm-<jobid>.err`).
- `weno_data/bin/uv` -- a copy of the `uv` binary for the GPU jobs (the compute cage hides
  the home directory, see below).

`ICON4PY_TEST_DATA_PATH=<workspace>/weno_data/testdata` selects the layout; the datatest
fixtures see the `.extraction_complete` markers and do not try to download.

## Running on a CPU backend

From `icon4py/` (the `uv` and compiler variables are the workspace's, see
`notes/sandbox.md`; the essential ones are `GT4PY_BUILD_CACHE_DIR`, `CXX` and
`ICON4PY_TEST_DATA_PATH`):

```bash
W=<workspace>/weno_data
ICON4PY_TEST_DATA_PATH=$W/testdata GT4PY_BUILD_CACHE_LIFETIME=persistent GT4PY_BUILD_JOBS=8 \
CXX=/user-environment/env/default/bin/g++ CC=/user-environment/env/default/bin/gcc \
GT4PY_BUILD_CACHE_DIR=$W/gt4py_cache/gtfn_cpu \
uv run --group test --frozen pytest -n0 -v -s --backend=gtfn_cpu --benchmark-disable \
  model/atmosphere/tracer_advection/tests/tracer_advection/integration_tests/test_jocksch_reference.py
# dace_cpu: --backend=dace_cpu and GT4PY_BUILD_CACHE_DIR=$W/gt4py_cache/dace_cpu
# the cylinder gates (slow: 4-6 min per case, the driver writes every step):
... pytest -n0 -v -s --backend=gtfn_cpu --benchmark-disable \
  model/driver/tests/driver/integration_tests/test_jocksch_cylinder.py
```

Wall times with a warm cache: reference test 3:44 (gtfn_cpu), 1:31 (dace_cpu); a cold
gtfn_cpu build of the reference test 5:47; the cylinder test about 40 min for its eight
cases on gtfn_cpu. GPU jobs (debug partition): gtfn_gpu 10:43 of pytest from a cold
stencil cache (job 858221, 14:43 from start to leaving the queue; the cache held only the
four geometry stencils and the compile-commands entry of the aborted job 858037, which
had died on the venv's `cupy-cuda12x`, after job 858036 had died on `uv: command not
found`), 6:57 warm (job 858330, 10:45); dace_gpu 13:27 cold (job 17:49), 3:54 warm (job
8:53); a first-time build fits the 30 min.

## Running on a GPU backend

`docs/run_jocksch_reference_gpu.sbatch` (one backend per job, debug partition, 30 min):

```bash
cd <workspace>   # not icon4py/: husk confines --output to the submitting directory and below
sbatch --partition=debug icon4py/model/atmosphere/tracer_advection/docs/run_jocksch_reference_gpu.sbatch gtfn_gpu
sbatch --partition=debug icon4py/model/atmosphere/tracer_advection/docs/run_jocksch_reference_gpu.sbatch dace_gpu
# with FMA contraction off (own cache directory <backend>_nofma):
sbatch --partition=debug ... run_jocksch_reference_gpu.sbatch dace_gpu nofma
```

Every variable is set inside the script (the sandbox's SLURM broker does not forward the
submitting shell's environment); the output is `weno_data/slurm/<jobid>.out`, and husk
forces stderr (its banner, tracebacks, pytest warnings) to `<workspace>/slurm-<jobid>.err`.
The compute cage hides the home directory, so the script runs a copy of `uv` staged in
`weno_data/bin/` (`cp -L ~/.local/bin/uv weno_data/bin/`; without it the job exits 127,
`uv: command not found`). The job is
a pytest run, so the submitter holds the workspace's pytest lock (`notes/workflow.md`)
for the job's whole lifetime, polling instead of `--wait` (which the broker refuses):

```bash
WS=<workspace>
until mkdir $WS/weno_data/pytest.lock 2>/dev/null; do sleep 30; done
echo "<agent> sbatch run_jocksch_reference_gpu.sbatch <backend>" > $WS/weno_data/pytest.lock/owner
cd $WS
id=$(sbatch --partition=debug --parsable icon4py/model/atmosphere/tracer_advection/docs/run_jocksch_reference_gpu.sbatch <backend>)
until ! squeue -j $id -h | grep -q .; do sleep 30; done
rm -rf $WS/weno_data/pytest.lock
```

If the job times out (30 min), resubmit: the persistent build cache resumes the build
where it stopped.

## The FMA switch

The Fortran was built without FMA contraction. `ICON4PY_FP_CONTRACT_OFF=1`
(`tests/tracer_advection/conftest.py`) exports `CXXFLAGS=-ffp-contract=off` (gtfn CMake,
DaCe compiler args) and `NVCC_APPEND_FLAGS=--fmad=false`, the values `ci/base.yml` uses
for its bit-reproducibility jobs. The flags do not enter the build-cache key, so the
conftest refuses to run without an explicit `GT4PY_BUILD_CACHE_DIR`; use
`gt4py_cache/<backend>_nofma`. Verified to reach the build:
`CMAKE_CXX_FLAGS:STRING=-ffp-contract=off` in every `CMakeCache.txt` of the `_nofma` cache,
empty in the default one.

**Measured effect (gtfn_cpu):** none at the level of the gates. Every L2 and trajectory
number changes in the last printed digit only (e.g. scheme 3 step-1 flux 4.717e-14 ->
4.728e-14, 102 trajectory worst 5.88e-15 -> 6.22e-15), with one telling exception: the
step-1 tracer of scheme 102 becomes bit-identical to the Fortran (max |dq| = 7e-42, a single
subnormal), where the contracted build shows 1.1e-16. So the residual 1e-16 of the linear
WENO step is the port's own FMA, and everything above that (the 1e-14 of scheme 3, the
1e-9 of scheme 103) has other causes (SVD round-off of the pseudoinverse; the Fortran's
single-precision smoothness indicator). The switch is kept as a porting tool, not as a
default: it costs a second build cache and does not change any verdict.

## Measured agreement, reference test

Notation: L2 = (max |q_py - q_f90|, max |F_py - F_f90| / max |F_f90|) of tracer 0 (the
four advected tracers 0-3 of the capture are identical, the test checks that);
trajectory = max |q_py - q_f90| over the 100 steps (worst step in brackets). The three CPU
columns agree to the printed digits except where shown; the gates in the module are twice
the worst value over the five backends (CPU below, GPU in the section after).

### L1

The stencil, moments, row weights, candidate weights, weight set and the
`weno_least_squares` pseudoinverses are pure numpy, hence the same on every backend. The
linear full pseudoinverse is the interpolation factory's and its SVD runs on the backend's
array namespace (`interpolation_fields.py`, `array_ns.linalg.svd`: numpy on CPU, cupy on
GPU), so its number is per backend, and so is its gate (selected by whether the backend
allocates on the GPU; one gate for both would blind the CPU path by 40x). The test also
recomputes this pseudoinverse with numpy from the same inputs brought to the host
(`compute_lsq_coeffs` on the `.asnumpy()` of the geometry fields and the owner mask):
that value is asserted at the CPU gate on every backend, so the GPU number comes from
cusolver's SVD, not from the cupy inputs; on CPU it is asserted bit-identical to the
factory's.

| array | max rel. diff. | gate |
|---|---|---|
| stencil_c9, moments, moments_hat, row weights, candidate weights, l_weights_s | 0 (bit-identical) | equality |
| quadratic pseudoinverse (SVD) | 7.226e-13 | 8e-13 |
| 27 quadratic candidate pseudoinverses (worst) | 2.5e-12 | 3e-12 |
| linear pseudoinverse (interpolation factory) | 3.521e-16 (CPU), 7.394e-15 (gtfn_gpu and dace_gpu: cusolver's SVD against LAPACK's) | 8e-16 (CPU), 2e-14 (GPU) |
| 3 linear candidate pseudoinverses | 2.711e-16 | 3e-16 |

### L2 per step, gtfn_cpu = dace_cpu = gtfn_cpu nofma (last digit apart)

| case | step 1 | step 2 | step 50 | step 100 | gate |
|---|---|---|---|---|---|
| ihadv2_hlim0 | 8.9e-16, 1.3e-15 | 8.9e-16, 9.8e-16 .. 1.1e-15 | 6.0e-16, 1.0e-15 | 6.5e-16, 1.1e-15 | 2e-15, 3e-15 |
| ihadv3_hlim0 | 2.1e-14, 4.7e-14 | 1.8e-14, 4.0e-14 | 6.2e-15, 1.4e-14 | 6.0e-15, 1.4e-14 | 5e-14, 1e-13 |
| ihadv102_hlim0 | 1.1e-16 (nofma: 7e-42), 1.3e-16 | 4.4e-16, 1.3e-15 | 2.2e-16 .. 3.3e-16, 6.6e-16 .. 7.2e-16 | 2.2e-16, 6.3e-16 | 2e-15, 3e-15 |
| ihadv103_hlim0 | 2.5e-09, 5.5e-09 | 3.5e-09, 7.8e-09 | 2.7e-09, 4.5e-09 | 1.3e-09, 2.0e-09 | 7e-9, 2e-8 |
| ihadv3_hlim3 | 2.2e-16, 1.9e-16 .. 2.4e-16 | 2.2e-16 .. 4.4e-16, 5.0e-16 .. 7.5e-16 | 3.7e-15, 8.2e-15 | 3.3e-15, 7.5e-15 | 8e-15, 2e-14 |
| ihadv3_hlim4 | 2.1e-14, 4.7e-14 | 1.8e-14, 4.0e-14 | 3.7e-15, 8.1e-15 | 3.3e-15, 7.3e-15 | 5e-14, 1e-13 |
| ihadv2_hlim4 | 8.9e-16, 1.3e-15 | 8.9e-16, 1.0e-15 | 4.4e-16, 8.7e-16 | 6.7e-16, 7.0e-16 | 2e-15, 3e-15 |
| ihadv132_hlim0 (gtfn_cpu = dace_cpu; no FMA-off run) | 2.9e-10, 6.4e-10 | 2.4e-09, 4.2e-09 | 8.1e-10, 1.3e-09 | 9.2e-10, 2.0e-09 | 5e-9, 9e-9 |

### Trajectory (max over 100 steps, worst step)

| case | gtfn_cpu | dace_cpu | gtfn_cpu nofma | gate |
|---|---|---|---|---|
| ihadv2_hlim0 | 8.66e-15 (96) | 8.80e-15 (96) | 8.63e-15 (96) | 2e-14 |
| ihadv3_hlim0 | 5.84e-14 (98) | 5.84e-14 (98) | 5.83e-14 (98) | 1.2e-13 |
| ihadv102_hlim0 | 5.88e-15 (97) | 6.00e-15 (97) | 6.22e-15 (97) | 1.3e-14 |
| ihadv103_hlim0 | 1.313e-08 (12) | 1.313e-08 (12) | 1.313e-08 (12) | 3e-8 |
| ihadv3_hlim3 | 4.89e-14 (97) | 4.90e-14 (97) | 4.92e-14 (97) | 1e-13 |
| ihadv3_hlim4 | 5.94e-14 (97) | 5.93e-14 (97) | 5.95e-14 (97) | 1.2e-13 |
| ihadv2_hlim4 | 1.40e-14 (61) | 1.39e-14 (61) | 1.44e-14 (61) | 3e-14 |
| ihadv132_hlim0 | 6.96e-09 (3) | 6.96e-09 (3) | -- | 1.4e-8 |

The per-step difference of the trajectory grows over the first tens of steps and then
saturates: from the step-1 level it reaches 10x for (2,0), (2,4), (3,0) and (3,4), 56x
for (102,0) and 220x for (3,3) (the two cases whose step-1 difference is a single ulp,
1.1e-16 / 2.2e-16), but the maximum over the 100 steps is within 2x of the trajectory's
own value at step 100 in every case (1.0x .. 1.4x) except 103 (2.2x, worst step 12) and
132 (worst 6.96e-9 at step 3: 2.4x its step-100 value, 3.8x its step-50 value, 24x its
step-1 value 2.9e-10), which do not grow after their early worst step. The one-step
differences (L2) stay at their level, so this is round-off accumulation, not a drifting
scheme.

What the numbers mean: 2 and 102 are bit-level agreement (a few ulp of q ~ 1); 3 (and 3 with
the PD limiter, which passes the unlimited flux through here) carries the 7e-13 SVD
round-off of the full quadratic pseudoinverse; 3 with the monotone limiter is at 1e-16 per
step because the limiter clips the reconstructed values; 103 is limited by the Fortran's
`REAL(sp)` smoothness indicator (`mo_advection_hflux.f90:2643,3007`), see the module
docstring, and is the one case where a double-precision port cannot get closer.

The hybrid scheme 132 (added to the test after the CPU sweep: gtfn_cpu, the two GPU
backends, then dace_cpu for its case alone, `w5d_dace_cpu_ihadv132.log`, 5 passed in
1:49 with the same digits as gtfn_cpu; not run with FMA contraction off) sits at the 103
level for the same reason: its WENO branch is the 103 blend with
the Fortran's `REAL(sp)` smoothness indicator, and its quadratic branch is scheme 3.

### GPU backends (jobs 858221 and 858330 gtfn_gpu, 858312 and 858316 dace_gpu)

The two GPU backends give the same numbers as each other and as the CPU backends to the
printed digits, with two exceptions, both round-off: the linear pseudoinverse of the
interpolation factory comes from cupy's (cusolver's) SVD instead of LAPACK's, 7.394e-15
against 3.5e-16 on CPU (its own gate, 2e-14, 2x that, the CPU gate staying 8e-16; the
scheme-2 flux built from it still agrees to 1.3e-15, so the pseudoinverse difference does
not propagate); and nvcc's
contraction makes the step-1 tracer of scheme 102 bit-identical to the Fortran (7.5e-42,
as the CPU build with `-ffp-contract=off`), where GCC's contracted build shows 1.1e-16.

L2 per step, gtfn_gpu = dace_gpu except where shown (the second value of a pair where
they differ):

| case | step 1 | step 2 | step 50 | step 100 | gate |
|---|---|---|---|---|---|
| ihadv2_hlim0 | 8.9e-16, 1.3e-15 | 8.9e-16, 9.8e-16 | 5.3e-16, 1.0e-15 | 5.3e-16, 1.1e-15 | 2e-15, 3e-15 |
| ihadv3_hlim0 | 2.1e-14, 4.7e-14 | 1.8e-14, 4.0e-14 | 6.2e-15, 1.4e-14 | 6.0e-15, 1.4e-14 | 5e-14, 1e-13 |
| ihadv102_hlim0 | 7.5e-42, 1.3e-16 | 4.4e-16, 1.3e-15 | 2.2e-16, 7.2e-16 / 6.6e-16 | 2.2e-16, 6.3e-16 | 2e-15, 3e-15 |
| ihadv103_hlim0 | 2.5e-09, 5.5e-09 | 3.5e-09, 7.8e-09 | 2.7e-09, 4.5e-09 | 1.3e-09, 2.0e-09 | 7e-9, 2e-8 |
| ihadv3_hlim3 | 2.2e-16, 2.0e-16 / 2.1e-16 | 4.4e-16, 7.5e-16 | 3.7e-15, 8.2e-15 | 3.3e-15, 7.5e-15 | 8e-15, 2e-14 |
| ihadv3_hlim4 | 2.1e-14, 4.7e-14 | 1.8e-14, 4.0e-14 | 3.7e-15, 8.1e-15 | 3.3e-15, 7.3e-15 | 5e-14, 1e-13 |
| ihadv2_hlim4 | 8.9e-16, 1.3e-15 | 8.9e-16, 1.1e-15 | 4.4e-16, 8.7e-16 | 4.4e-16, 6.5e-16 | 2e-15, 3e-15 |
| ihadv132_hlim0 | 2.9e-10, 6.4e-10 | 2.4e-09, 4.2e-09 | 8.1e-10, 1.3e-09 | 9.2e-10, 2.0e-09 | 5e-9, 9e-9 |

Trajectory (max over 100 steps, worst step):

| case | gtfn_gpu | dace_gpu | gate |
|---|---|---|---|
| ihadv2_hlim0 | 8.74e-15 (96) | 8.80e-15 (96) | 2e-14 |
| ihadv3_hlim0 | 5.83e-14 (98) | 5.83e-14 (98) | 1.2e-13 |
| ihadv102_hlim0 | 6.00e-15 (97) | 6.11e-15 (97) | 1.3e-14 |
| ihadv103_hlim0 | 1.313e-08 (12) | 1.313e-08 (12) | 3e-8 |
| ihadv3_hlim3 | 4.90e-14 (97) | 4.89e-14 (97) | 1e-13 |
| ihadv3_hlim4 | 5.95e-14 (97) | 5.94e-14 (97) | 1.2e-13 |
| ihadv2_hlim4 | 1.44e-14 (61) | 1.37e-14 (61) | 3e-14 |
| ihadv132_hlim0 | 6.96e-09 (3) | 6.96e-09 (3) | 1.4e-8 |

Both jobs: 40 passed and the L1 test failed at the then single 8e-16 linear-pseudoinverse
gate (the only gate any GPU number exceeded); with the per-device gate the module passes
on dace_gpu (job 858316, 41 passed, 3:54 of pytest with the warm cache, job 8:53) and on
gtfn_gpu (job 858330, 41 passed, 6:57 of pytest with the warm cache, job 10:45; its L1 line
prints cusolver's 7.394e-15 next to 3.521e-16 for the numpy recompute on the same inputs). Before 858221, two gtfn_gpu submissions aborted:
858036 with `uv: command not found` (the compute cage hides the home directory; hence the
copy in `weno_data/bin/`) and 858037 with `ImportError: libcublas.so.12` (the venv had
been synced with the `cuda12` extra, `cupy-cuda12x`, while the uenv `icon/26.7:v1` ships
CUDA 13.1); 858037 left five entries in the gtfn_gpu build cache, the rest of 858221's
build was cold. The venv was re-synced with `uv sync --frozen --group test --group dev
--extra all --extra cuda13` (the one-package delta `cupy-cuda12x -> cupy-cuda13x`,
nothing else changed); this changed the shared venv, `install_dependencies.sh` would put
`cuda12` back, and `--extra cuda12` reverts it: see `notes/sandbox.md`, section "GPU
jobs", for these three points and the submission from the workspace root. The FMA-off
GPU variant (`nofma`) was not run: the contracted GPU build already reproduces the CPU
nofma result where it differs from the contracted CPU one.

## Measured agreement, cylinder gates (gtfn_cpu)

All eight cases pass both gates. `pair sum` is Jocksch's measure (his printed `#` error,
neighbour pairs within an edge length, 3 sum e^2 without the seam pairs); the paper's value is
sqrt(pair sum / 3) truncated to three decimals; the relative difference is to the Fortran run
on the `_centred` grid copy (`weno_data/reference/<case>_centred/error.txt`), whose dt is
999.99999995 s against 1000 s here.

| case | pair sum | sqrt(pair sum / 3) | Table 2 | rel. diff. to Fortran | gate |
|---|---|---|---|---|---|
| miura (2, 0) | 48.561832 | 4.0234 | 4.023 | 2.945e-11 | 1e-10 |
| miura3 (3, 0) | 34.738525 | 3.4029 | 3.402 | 7.742e-11 | 3e-10 |
| miura_weno (102, 0) | 44.683764 | 3.8594 | 3.859 | 3.326e-10 | 1e-9 |
| miura3_weno (103, 0) | 28.064229 | 3.0587 | 3.058 | 3.496e-09 | 1e-8 |
| miura-monotonic (2, 3) | 42.826970 | 3.7783 | 3.778 | 1.733e-10 | 6e-10 |
| miura3-monotonic (3, 3) | 34.095277 | 3.3712 | 3.371 | 3.237e-11 | 1e-10 |
| miura-positive_definite (2, 4) | 43.144220 | 3.7923 | -- | 4.634e-11 | 2e-10 |
| miura3-positive_definite (3, 4) | 33.900494 | 3.3615 | 3.361 | 4.780e-11 | 2e-10 |

The relative differences are 1e-11 .. 3e-10 for the double-precision schemes (the dt
difference alone is 5e-11) and 3.5e-9 for 103 (the Fortran's single-precision smoothness
indicator, as in the reference test). The gates are about three times (2.9-4.3x) the gtfn_cpu value.
The mass is conserved to < 9e-15 relative in every case. The eight cases were run in one
session (`weno_data/slurm/w5b_gtfn_cpu_cylinder.log`); the miura3_weno row there failed
with a `SyntaxError` from a stencil module a parallel package was editing at that moment,
and was rerun alone after that edit was committed
(`w5b_gtfn_cpu_cylinder_miura3_weno.log`, 8:28), which is the value in the table.
