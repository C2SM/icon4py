# Numerical-agreement gates

The turbulence granule is validated against serialized ICON reference data. **Every stencil is
expected to agree bit-exactly with the Fortran it was translated from.** Where that expectation
cannot be met, the exception is declared — per stencil, with a measured error and a stated reason —
in [`tests/turbulence/gate_registry.py`](../tests/turbulence/gate_registry.py).

This document is the procedure for changing an entry. It is written for the person who arrives with
a failing stencil test and wants to relax a threshold.

## Why the registry exists

Bit-exactness is the default because it is achievable for most of this scheme and because it is a
binary answer: either the port reproduces the reference or it does not, and no one has to argue
about how much disagreement is acceptable.

But it is not achievable for all of it. Both sides are compiled IEEE-strict with FMA contraction
off (`-Kieee -Mnofma` for nvhpc; no-fast-math plus `-ffp-contract=off` on CPU / `--fmad=false` on
GPU for the GT4Py backend), and the Python mirrors the Fortran's operation order and
parenthesisation literally. Neither of those constrains what GT4Py does *above* the arithmetic: it
re-associates expressions while inlining and fusing, substitutes reciprocals for divisions,
restructures expression trees, and lowers scans differently than a sequential Fortran `k`-loop.
Transcendentals go through a different libm and will differ by one to a few ULP whatever anyone
does.

So a set of stencils will not be bit-exact, and **which ones is discovered by measurement, not
predicted.** The EXCLAIM dycore needed tolerance for 2–3 stencils out of 110; for this scheme it
could plausibly be 20% or more. A downgrade is therefore not a defeat. It is an ordinary outcome
that has to be recorded.

The thing being prevented is narrower and more specific: **a stencil quietly moving from `Exact` to
`Tol` in some commit, with a threshold somebody guessed.** That is invisible in a green test run and
almost invisible in review, and once it has happened nobody can tell later whether the tolerance
describes a known property of the lowering or covers up a translation error. Putting the gates in
one version-controlled dict makes such a change a line in a diff that a reviewer has to approve, and
the recorded `measured_max_rel_err` makes the next person's request to widen `rtol` argue against a
number instead of against nothing.

Hence the registry has no default. `gate_for()` on an unregistered stencil raises rather than
assuming `Exact()`, because a silent default is the same failure mode in a different costume: a
stencil that was never gated looks exactly like one that was.

## A per-stencil gate is necessary but not sufficient

Do not read a green stencil gate as "the port agrees with ICON".

`tke` is **prognostic**, and it is temporally smoothed (`tkesmot = 0.15`), so each timestep's value
carries a weighted memory of the previous ones. A tolerance granted per call is therefore not a
bound on anything a forecast cares about: a small per-call disagreement is fed back into the next
call's input, and over a forecast it can compound. A registry of per-call tolerances cannot detect
that, no matter how tight each entry is — the quantity it fails to constrain is not measured at that
level.

That is why a separate **trajectory-level drift check** exists, in
`tests/turbulence/integration_tests/`: it runs the granule over all serialized timesteps and asserts
that the error in `tke` does not grow monotonically. The two checks answer different questions and
neither substitutes for the other. When you widen a gate on any stencil that feeds `tke`, `tkvm`,
`tkvh` or `rcld`, the drift check is the one that tells you whether you got away with it.

## The admissible reasons

`Reason` in the registry is a **closed** set. Every member names a mechanism by which GT4Py
legitimately produces a different rounding than the Fortran:

| `Reason` member           | means                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| `TRANSCENDENTAL`          | `exp`, `log`, `pow`, `tanh`, `**` resolve to a different libm implementation.                 |
| `REASSOCIATION`           | Inlining or fusion regrouped an expression tree, changing the order of `+`/`*`.               |
| `RECIPROCAL_SUBSTITUTION` | A division became a multiplication by a reciprocal — two roundings where the Fortran has one. |
| `SCAN_LOWERING`           | A `scan_operator` carry is accumulated differently than the sequential `k`-loop it came from. |

The set is closed on purpose. **If none of these describes your disagreement, you have not found a
tolerance case — you have found a bug.** Candidates, roughly in order of how often they turn out to
be the real answer:

- a mistranslated expression, a wrong constant, or a sign;
- a domain or vertical-bound off-by-one, so the comparison includes a level the Fortran never wrote;
- an uninitialised or stale field in the test setup;
- a branch taken differently because a guard threshold was translated as `<` instead of `<=`;
- a `wpfloat`/`vpfloat` mix-up (`hdef2`, `hdiv`, `dwdx`, `dwdy` are `vp`; everything else is `wp`);
- the strict-IEEE compile flags above not actually being in effect on the side you are testing.

Widening the reason set is a **spec-level decision**, not a code change: it means a new mechanism
has been identified, and the port spec's section 5.3 should say so before the enum does.

## Downgrading `Exact` → `Tol`

Do all of it, in order. Steps 1 and 2 are where nearly all the value is.

1. **Rule out a bug.** Work the list above. A disagreement that is not attributable to one of the
   four mechanisms is not a candidate for a gate. In particular, look at *where* the disagreement
   is: a mechanism-level rounding difference is spread thinly over the whole field, while a
   translation error is usually concentrated in a few columns, at one vertical level, or on one
   side of a branch. If your error map has structure, it is a bug.

2. **Measure the error distribution.** Not the maximum you happened to see once — the distribution,
   over the full reference dataset (`exp.mch_icon-ch2_small`, all serialized timesteps), for every
   output field of the stencil. Record:

   - the maximum relative error, which is the number that goes into `measured_max_rel_err`;
   - the shape of the distribution (a percentile or two is enough) — an error concentrated in a
     thin tail behaves very differently from one spread evenly, and the tail is where the next
     regression will show up;
   - which field and which level the maximum occurred on;
   - the backend and grid you measured on.

   **A guessed threshold is not acceptable at any stage of this procedure.** A tolerance nobody
   measured has no relationship to the mechanism it claims to describe, and it will be inherited,
   copied to the next stencil, and widened again by someone who assumes it meant something.

3. **Choose `rtol` from the measurement.** Round the measured maximum up to the next power of ten
   — enough headroom that ordinary variation between backends does not turn the gate red, little
   enough that a real regression still does. The registry enforces only the coherence condition
   (`0 < measured_max_rel_err <= rtol`); the headroom convention is on you. If you find yourself
   wanting orders of magnitude of headroom, go back to step 1 — that is the signature of an
   unexplained disagreement, not a rounding one.

4. **Write the entry, with provenance.** `Tol` carries three fields deliberately; everything else
   goes in a comment above the entry, and the comment is not optional. It is what makes step 5
   reviewable and what the next person will read instead of re-deriving your work:

   ```python
   GATES: dict[str, Gate] = {
       # Measured 2026-09-14 on exp.mch_icon-ch2_small, all 6 timesteps, gtfn_cpu:
       # max rel err 3.1e-15 in 'tke' at k=42, p99 4e-16. The Fortran computes
       # `a / (b * c)`; GT4Py fuses this with the caller and substitutes a reciprocal.
       # Fortran provenance: turb_diffusion.f90:1119-1131.
       "compute_turbulent_length_scale": Tol(
           rtol=1.0e-14,
           reason=Reason.RECIPROCAL_SUBSTITUTION,
           measured_max_rel_err=3.1e-15,
       ),
   }
   ```

5. **Get it reviewed.** A gate downgrade is a change to what the port claims about itself, so it
   needs a second reader — one who checks the reasoning in step 1, not just that the tests pass. Say
   in the commit message that a gate was downgraded and why; a reviewer skimming a large stencil
   diff should not have to notice it on their own.

6. **Check the trajectory drift test** if the stencil feeds `tke`, `tkvm`, `tkvh` or `rcld`. See
   above for why the per-stencil result does not answer this.

## Widening an existing `Tol`

Same procedure, from step 1. It is not an edit to a number.

Specifically: the fact that a stencil already has a `Tol` entry tells you that *one* mechanism was
identified and measured on the data available at the time. It does not license a larger tolerance
for the same reason. If the error has grown, the honest possibilities are that the reference data
now covers a case the original measurement did not, that a change elsewhere in the granule altered
what GT4Py fuses this stencil with, or that something regressed. Re-measure, and update
`measured_max_rel_err` together with `rtol` — leaving a stale measured value beside a widened `rtol`
destroys the only evidence the entry carries.

Going the other way — `Tol` → `Exact` — needs no ceremony beyond a green test run. Tightening a gate
cannot hide anything.

## Removing a stencil

Delete its entry in the same commit that deletes the stencil. An orphaned gate is harmless to the
tests and misleading to read, since it looks like a documented exception for code that no longer
exists.

## References

- Port spec, section 5.3 (*Gates*) — the reasoning this document implements.
- Port spec, section 5.5 (*Floating-point hazards*) — the two guards in this scheme that look like
  numerical risks, one real and deferred by the double-precision decision, one retired.
- `tests/turbulence/gate_registry.py` — the registry itself.
