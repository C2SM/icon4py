# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Counter-based RNG for AMPS habit selection (quality bar: statistical,
NOT cryptographic), graduated verbatim from
`spikes/spike_e_counter_rng.py` (docs/superpowers/facts/m1/
icon4py-m1-conventions.md "F5" SS6) into the package. Construction: an
int64 counter mixed by an affine combine of (cell, level, bin, step),
ONE nonlinear (quadratic) mixing round, then 3 Park-Miller Lehmer rounds
-- all reduced modulo the Mersenne prime `M31 = 2**31 - 1`.

`counter_hash01` (the field_operator) and `counter_hash01_numpy` (its
numpy replica) implement the IDENTICAL sequence of integer operations,
bit-for-bit -- `test_rng.py` asserts this on int64 intermediate/float64
output equality, not merely closeness. The mixing constants
(`M31`/`A1`/`A2`/`A3`/`C_CELL`/`C_K`/`C_BIN`/`C_STEP`) are unchanged from
the spike: bit-exactness against the same numpy replica logic that spike
already validated (mean~0.5, var~1/12, lag-1 correlation <0.01 along
both the cell and level axes, 16-bin histogram flat within 1%; see
`spike_e_counter_rng.py::main()` for that original empirical run) is
exactly what graduating this module is meant to preserve -- changing any
constant here would silently invalidate that prior validation.

FINDING 2 carried forward (DSL API): every mixing constant is threaded
through as an explicit `gtx.int64` scalar PARAMETER of `counter_hash01`,
not referenced as a bare module-level Python int/np.int64 closure
global. Both closure-reference forms fail on at least one backend
(implicit int32 widen rejected on embedded; unresolved closure symbol on
gtfn_cpu -- see `spike_e_counter_rng.py`'s own FINDING 2 for the full
verified trail); scalar call arguments hit neither failure mode.

--------------------------------------------------------------------
int64 domain bounds (no overflow)
--------------------------------------------------------------------

Two integer expressions in the mixing chain can in principle overflow
int64 (max `2**63 - 1 = 9_223_372_036_854_775_807`); both are checked
here with the ACTUAL worst-case numbers, not just asymptotically:

1. The initial affine combine,
   `x = cell*c_cell + k*c_k + bin*c_bin + step*c_step + 1`.
   Each of the four axis values (`cell_id`, `k_id`, `bin_id`, `step`)
   must be `< 2**31` (~2.147e9) and each mixing constant
   (`c_cell`/`c_k`/`c_bin`/`c_step`) must be `< 2**20` (~1.049e6) -- the
   bound this module's constants are chosen to satisfy (all four are
   < 10**6, comfortably under 2**20). Under those two bounds, every
   individual product is `< 2**51` (~2.252e15) and the sum of all four
   plus 1 is `< 2**53` (~9.007e15) -- about 1024x of headroom below
   int64's max, i.e. `axes < 2**31` is a generous, convenient bound (it
   fits any icon4py grid size and any int32-range Fortran counter) far
   below the actual overflow threshold, not a tight one.
   STEP BOUND specifically: `step` is normally a monotonically
   increasing substep/timestep counter over a run's lifetime (unlike
   `cell_id`/`k_id`/`bin_id`, which are bounded once and for all by grid
   size), so it is the one axis a long-running simulation could
   plausibly grow past `2**31` (~2.1 billion steps) if left unbounded --
   callers driving very long integrations should fold `step` (e.g.
   `step % 2**30`) before calling, rather than let it grow without
   bound; the RNG's OWN quality bar does not require monotonic `step`
   values, only that (cell, k, bin, step) as a 4-tuple stays distinct
   per draw.
2. The quadratic mixing round, `x = x*x + x + 1` (mod `m31`, so `x`
   entering this line is already `<= M31 - 1 = 2_147_483_646`). Worst
   case `x*x = 4_611_686_009_837_453_316` (~4.612e18); adding `x + 1` is
   negligible against that. This is the TIGHTER of the two checks:
   `4.612e18` vs. int64 max `9.223e18` is only ~2.0x of headroom (vs.
   ~1024x for check 1 above) -- the binding overflow constraint in this
   whole construction, exactly as `spike_e_counter_rng.py`'s FINDING 1
   already noted ("x stays < M31 ~= 2.1e9 at every step, so x*x peaks at
   ~4.6e18 -- comfortably inside int64's ~9.22e18 max, no overflow").

The 3 subsequent Lehmer rounds (`x = (x*a) % m31`, `a` in
`{A1, A2, A3}`, each `< 2**17`) each multiply a value `< M31` (~2^31) by
a constant `< 2**17`, giving products `< 2**48` -- not a binding
constraint (looser than either check above).

--------------------------------------------------------------------
Quadratic-round 2-to-1 collision caveat
--------------------------------------------------------------------

`f(x) = (x*x + x + 1) mod p` (`p = M31`, prime) is NOT a bijection of
`Z_p` onto itself -- it is exactly 2-to-1 (except at one fixed point).
Proof: `f(x1) = f(x2) mod p` iff `x1**2 - x2**2 + x1 - x2 = 0 mod p` iff
`(x1 - x2)*(x1 + x2 + 1) = 0 mod p`; since `p` is prime, this holds iff
`x1 = x2` or `x1 + x2 = p - 1 mod p`, i.e. `x2 = (p - 1 - x1) mod p`. So
every output value has exactly two preimages, `{x, p-1-x}`, EXCEPT the
single self-paired fixed point `x = (p-1)*inverse(2) mod p =
1_073_741_823` (verified: `2*1_073_741_823 mod M31 == M31 - 1`, and a
brute-force check at a small prime modulus confirms the general
"exactly 2 preimages per output, one single-preimage fixed point"
structure holds exactly). The 3 following Lehmer rounds are each a
BIJECTION on `Z_p \\ {0}` (multiplication by a nonzero constant modulo a
prime is invertible) and fix `0 -> 0`, so they permute -- but do not
remove -- the 2-to-1 structure the quadratic round already imposed: the
overall `counter_hash01` construction is (at most) 2-to-1 onto its
image, not a true bijection across all of `[0, M31)`. This is
IRRELEVANT to the "habit selection, not crypto" quality bar this module
targets (mean/variance/lag-1 correlation/histogram flatness, all
statistical properties unaffected by a uniform 2-to-1 folding) but
would matter for any hypothetical reuse requiring a genuine
no-collision/bijective guarantee (e.g. unique-ID generation) -- do not
repurpose this construction for that.

--------------------------------------------------------------------
Affine-composition rationale for the nonlinear round
--------------------------------------------------------------------

Carried forward from `spike_e_counter_rng.py`'s FINDING 1 (that spike's
own empirical trail, not re-derived here): the brief's original
3-round-Lehmer-ONLY construction (no quadratic round) FAILS the lag-1
correlation quality bar along both the cell and level axes
(`lag1_cell=-0.337`, `lag1_k=-0.496` vs. the `<0.01` bar), and stacking
MORE Lehmer rounds does not fix it (verified through 7 rounds). Root
cause: the initial `combine` is affine in `(cell, k, bin, step)` mod
`p`, and each Lehmer round `x -> (x*a) mod p` is a LINEAR map over the
ring `Z_p`; composing any number of linear maps with one affine map
stays affine. Along a unit-step axis (consecutive `cell` or `k`
indices), an affine map mod `p` is exactly a Weyl/sawtooth sequence
`x_n = n*delta + c (mod p)`, whose lag-1 correlation is a deterministic
function of the single scalar slope `delta` -- adding more multiplicative
(linear) rounds only changes `delta`, never breaks the affine structure
responsible for that correlation. Inserting ONE genuinely NONLINEAR
round (`x = (x*x + x + 1) mod p`, quadratic -- not affine -- in the
original index) breaks the Weyl structure and is what actually fixes
the lag-1 metric (down from the failing values above to `<0.01`),
without needing a 4th Lehmer round.

Run the graduation source's own empirical validation via:
`uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_e_counter_rng.py`
"""

from __future__ import annotations

import gt4py.next as gtx
import numpy as np
import numpy.typing as npt
from gt4py.next import astype

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


M31 = 2147483647  # 2^31 - 1 (Mersenne prime, Park-Miller modulus)
A1 = 16807  # Park-Miller multipliers for the (linear) Lehmer rounds
A2 = 48271
A3 = 69621
# Mixing multipliers for the initial counter combine (distinct odd
# primes, < 2^20 to keep products of int32-range counters safely inside
# int64 -- see module docstring, "int64 domain bounds", check 1).
C_CELL = 999983
C_K = 424243
C_BIN = 786433
C_STEP = 611953


@gtx.field_operator
def counter_hash01(  # noqa: PLR0917 -- mixing constants are scalar params,
    # not closure globals, per the module docstring's FINDING 2.
    cell_id: fa.CellField[gtx.int64],
    k_id: fa.KField[gtx.int64],
    bin_id: gtx.int64,
    step: gtx.int64,
    c_cell: gtx.int64,
    c_k: gtx.int64,
    c_bin: gtx.int64,
    c_step: gtx.int64,
    m31: gtx.int64,
    a1: gtx.int64,
    a2: gtx.int64,
    a3: gtx.int64,
) -> fa.CellKField[ta.wpfloat]:
    """Counter-based hash to a float in `[0, 1)`. See module docstring
    for the full construction rationale and the required int64 domain
    bounds on `cell_id`/`k_id`/`bin_id`/`step`/the mixing constants.
    Call with all eight mixing constants wrapped as `gtx.int64(...)` at
    the call site (see `counter_hash01_numpy` for the bit-identical
    numpy replica used to gate this operator's tests)."""
    one = astype(1, gtx.int64)
    x = (cell_id * c_cell + k_id * c_k + bin_id * c_bin + step * c_step + one) % m31
    x = (x * x + x + one) % m31  # nonlinear (quadratic) mixing round --
    # see module docstring "affine-composition rationale".
    x = (x * a1) % m31
    x = (x * a2) % m31
    x = (x * a3) % m31
    return astype(x, ta.wpfloat) / 2147483647.0


def counter_hash01_numpy(
    cell_id: npt.ArrayLike, k_id: npt.ArrayLike, bin_id: int, step: int
) -> np.ndarray:
    """Numpy replica of `counter_hash01`, bit-identical integer
    intermediates by construction (same operations, same order, same
    modulus/constants). `cell_id`/`k_id` are broadcast outer-product
    style (`cell_id[:, None]`, `k_id[None, :]`) to produce a
    `(len(cell_id), len(k_id))` array, matching `counter_hash01`'s
    `(Cell, K)` output shape for the same inputs."""
    cell = np.asarray(cell_id, dtype=np.int64)[:, None]
    k = np.asarray(k_id, dtype=np.int64)[None, :]
    x = (cell * C_CELL + k * C_K + int(bin_id) * C_BIN + int(step) * C_STEP + 1) % M31
    x = (x * x + x + 1) % M31
    for a in (A1, A2, A3):
        x = (x * a) % M31
    return x.astype(np.float64) / float(M31)
