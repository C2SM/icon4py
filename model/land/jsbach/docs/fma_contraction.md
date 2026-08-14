# FMA contraction makes "bit-exact against ICON" backend-dependent

**For @muellc** — this came out of validating the ported JSBACH soil-energy (SSE)
kernels against ICON savepoints. It is not a JSBACH problem; it applies to any
Fortran-validated port, so it probably wants a decision at the icon4py/GT4Py level
rather than a workaround per test.

## What we see

Driving the three ported soil kernels on ICON's own savepoint inputs and comparing
against ICON's outputs (global R02B04 terra-planet, 20480 columns, varied soil
types, 3 timesteps):

| backend | t_soil_sl | t_soil_acoef | t_soil_bcoef | grnd_hflx | hcap_grnd |
| --- | --- | --- | --- | --- | --- |
| `gtfn_cpu` | bit-exact | bit-exact | bit-exact | bit-exact | bit-exact |
| `embedded` | ~1 ulp | ~1 ulp | bit-exact | ~1e-10 rel | ~1 ulp |

Same kernels, same inputs, same reference. The backend decides whether the result
is bit-identical to ICON.

## Why

Every one of those kernels contains an `a + b*c`:

- back substitution — `t_soil_acoef + t_soil_bcoef * t_soil_above`
  (`mo_sse_process.f90:499`)
- ground heat flux — `zdz1 * (t_soil_acoef + (t_soil_bcoef - 1) * t_soil_sl)`
  (`mo_sse_process.f90:750`)

ICON is built with nvfortran and `-acc=gpu`, which contracts those into a single
fused multiply-add: one rounding instead of two. The C++ that gtfn emits is compiled
by g++, whose default is `-ffp-contract=fast`, so gtfn fuses them too and lands on
exactly the same bits. numpy, which the embedded backend evaluates through, has no
contraction: it rounds the product, then rounds the sum.

The size of the resulting difference is set by the expression, not by the operation.
For the temperatures it is ~1 ulp (2e-16 relative). For `grnd_hflx` the same single
fused operation shows up as ~1e-10 relative, because the expression is a difference
of two O(300 K) terms scaled down to O(1 W/m²) — the cancellation amplifies the last
bit by ~6 orders of magnitude. Reconstructing the reference with python 3.13's
`math.fma` reproduces ICON's `grnd_hflx` bit-for-bit, which is how we identified this.

## Why it matters beyond this port

1. **"Bit-exact against ICON" is not a backend-independent property.** Our handover
   document specified a bit-exact validation gate. That gate is achievable on
   `gtfn_cpu` and unachievable on `embedded`, for a correct port. Any Fortran-validated
   port that states a bit-exact gate inherits this.
2. **Two icon4py backends can disagree with each other**, without either being wrong.
   `embedded` is usually treated as the reference implementation for a stencil; here
   it is the one that does *not* match ICON.
3. **It is silent.** Nothing in the toolchain reports that a contraction happened.
   We only found it because one field cancelled hard enough to lift 1 ulp into a
   visible number.

## Questions

- Is FP contraction something GT4Py wants to pin per backend (e.g. compile gtfn with
  `-ffp-contract=off` for reproducibility, or the opposite — deliberately fuse so
  gtfn matches Fortran)? Right now it is inherited from whatever the host compiler
  defaults to, which also makes it a function of the compiler version and target.
- Do the DaCe backends contract? We have not measured; only `embedded` and `gtfn_cpu`
  were available in this environment.
- Should validation helpers grow an FMA-aware comparison, or is a documented few-ulp
  tolerance the right answer? We chose the latter (see below), but a port that has to
  prove bit-exactness for certification would need the former.

## What this port does meanwhile

`model/land/jsbach/tests/jsbach/integration_tests/test_sse_datatest.py` gates on
`rtol=1e-13` for the temperatures and coefficients, and on an absolute tolerance for
`grnd_hflx` (`atol=1e-9`, since the relative measure is meaningless for a cancelling
quantity). Both admit the contraction difference and nothing larger; a genuine port
error moves these fields by orders of magnitude more. The two contraction-sensitive
expressions carry `NOTE (FMA contraction)` comments in
`stencils/soil_temperature.py` pointing here.

Reproduce with:

```bash
# in the ICON build tree used for the validation
python synthland/validate_sse.py --ser-data <experiment>/ser_data --backend embedded
python synthland/validate_sse.py --ser-data <experiment>/ser_data --backend gtfn_cpu
```
