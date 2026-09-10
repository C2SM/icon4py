# Running the Jocksch reference tests

How to run the tests of the FFSL-WENO port against A. Jocksch's Fortran capture and his
paper, per backend, and what the numbers were when this was written (2026-09-11, branch
`weno_idealized`, santis, uenv `icon/26.7:v1`, GCC 14.3, numpy 2.4 with OpenBLAS). Companion
to `weno_idealized_scope.md` (what is ported, Fortran <-> Python map) and
`weno_idealized_status.md`.

## The two test modules

| module | what it compares | data |
|---|---|---|
| `model/atmosphere/tracer_advection/tests/tracer_advection/integration_tests/test_jocksch_reference.py` | **L1** init-time least-squares coefficients, **L2** one `Advection.run` per savepoint (steps 1, 2, 50, 100, tracers 1-4), **L3** the 100-step trajectory with the new tracer fed back | the serialbox capture, `datatest` mark |
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
- `weno_data/slurm/` -- logs (`w5b_<backend>_<what>.log` from the runs below, `<jobid>.out`
  from the GPU jobs).

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
cases on gtfn_cpu.

## Running on a GPU backend

`docs/run_jocksch_reference_gpu.sbatch` (one backend per job, debug partition, 30 min):

```bash
sbatch --partition=debug model/atmosphere/tracer_advection/docs/run_jocksch_reference_gpu.sbatch gtfn_gpu
sbatch --partition=debug model/atmosphere/tracer_advection/docs/run_jocksch_reference_gpu.sbatch dace_gpu
# with FMA contraction off (own cache directory <backend>_nofma):
sbatch --partition=debug ... run_jocksch_reference_gpu.sbatch dace_gpu nofma
```

Every variable is set inside the script (the sandbox's SLURM broker does not forward the
submitting shell's environment); poll with `squeue -j <id>`; the output is
`weno_data/slurm/<jobid>.out`.

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

Notation: L2 = (max |q_py - q_f90|, max |F_py - F_f90| / max |F_f90|) over the four tracers;
trajectory = max |q_py - q_f90| over the 100 steps (worst step in brackets). The three CPU
columns agree to the printed digits except where shown; the gates in the module are twice
the worst CPU value.

### L1 (numpy, identical on all backends)

| array | max rel. diff. | gate |
|---|---|---|
| stencil_c9, moments, moments_hat, row weights, candidate weights, l_weights_s | 0 (bit-identical) | equality |
| quadratic pseudoinverse (SVD) | 7.226e-13 | 8e-13 |
| 27 quadratic candidate pseudoinverses (worst) | 2.5e-12 | 3e-12 |
| linear pseudoinverse (interpolation factory) | 3.521e-16 | 4e-16 |
| 3 linear candidate pseudoinverses | 2.711e-16 | 3e-16 |

### L2 per step, gtfn_cpu = dace_cpu = gtfn_cpu nofma (last digit apart)

| case | step 1 | step 2 | step 50 | step 100 | gate |
|---|---|---|---|---|---|
| ihadv2_hlim0 | 8.9e-16, 1.3e-15 | 8.9e-16, 9.8e-16 | 6.0e-16, 1.0e-15 | 6.5e-16, 1.1e-15 | 2e-15, 3e-15 |
| ihadv3_hlim0 | 2.1e-14, 4.7e-14 | 1.8e-14, 4.0e-14 | 6.2e-15, 1.4e-14 | 6.0e-15, 1.4e-14 | 5e-14, 1e-13 |
| ihadv102_hlim0 | 1.1e-16 (nofma: 7e-42), 1.3e-16 | 4.4e-16, 1.3e-15 | 2.2e-16 .. 3.3e-16, 6.6e-16 .. 7.2e-16 | 2.2e-16, 6.3e-16 | 1e-15, 3e-15 |
| ihadv103_hlim0 | 2.5e-09, 5.5e-09 | 3.5e-09, 7.8e-09 | 2.7e-09, 4.5e-09 | 1.3e-09, 2.0e-09 | 7e-9, 2e-8 |
| ihadv3_hlim3 | 2.2e-16, 1.9e-16 .. 2.4e-16 | 2.2e-16 .. 4.4e-16, 5.0e-16 .. 7.5e-16 | 3.7e-15, 8.2e-15 | 3.3e-15, 7.5e-15 | 8e-15, 2e-14 |
| ihadv3_hlim4 | 2.1e-14, 4.7e-14 | 1.8e-14, 4.0e-14 | 3.7e-15, 8.1e-15 | 3.3e-15, 7.3e-15 | 5e-14, 1e-13 |
| ihadv2_hlim4 | 8.9e-16, 1.3e-15 | 8.9e-16, 1.0e-15 | 4.4e-16, 8.7e-16 | 6.7e-16, 7.0e-16 | 2e-15, 3e-15 |

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

The 103 trajectory has its worst step early (12) and does not grow afterwards; the others
grow by about one order of magnitude over the 100 steps, which is round-off accumulation
(the per-step differences are constant, see L2).

What the numbers mean: 2 and 102 are bit-level agreement (a few ulp of q ~ 1); 3 (and 3 with
the PD limiter, which passes the unlimited flux through here) carries the 7e-13 SVD
round-off of the full quadratic pseudoinverse; 3 with the monotone limiter is at 1e-16 per
step because the limiter clips the reconstructed values; 103 is limited by the Fortran's
`REAL(sp)` smoothness indicator (`mo_advection_hflux.f90:2643,3007`), see the module
docstring, and is the one case where a double-precision port cannot get closer.

### GPU backends

Not yet measured when this note was written: the jobs go through
`run_jocksch_reference_gpu.sbatch` (above) and their tables belong here, in the same
format, together with the job wall time (the debug partition's 30 min bounds a cold build).

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
indicator, as in the reference test). The gates are three times the gtfn_cpu value.
The mass is conserved to 7e-15 relative in every case.
