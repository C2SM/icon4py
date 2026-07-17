# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Spike E: counter-based RNG expressible in the GT4Py DSL, which allows only
+ - * // % on integers (no bitwise ops, no shifts -- verified against gt4py
1.1.11 func_to_foast). Construction: an int64 counter mixed from
(cell, k, bin, step), one nonlinear (quadratic) mixing round, then 3
Park-Miller Lehmer rounds modulo the Mersenne prime 2^31-1.

Quality bar (for habit selection, NOT crypto): mean ~ 0.5, var ~ 1/12,
lag-1 correlation across each axis < 0.01, coarse 16-bin histogram flat
within 1%. All checked vs numpy replica (bit-identical integers) -- that
match is the gating assert for everything downstream.

Two findings diverge from the brief; both driven by actually running things,
not assumed (full numeric trail in task-11-report.md):

FINDING 1 (statistical construction): the brief's 3-round-Lehmer-only
recipe -- x = combine(cell,k,bin,step) % M31; x = (x*A1) % M31;
x = (x*A2) % M31; x = (x*A3) % M31 -- FAILS the lag-1 correlation bar badly
(lag1_cell=-0.337, lag1_k=-0.496 vs the <0.01 bar). The brief's prescribed
remedy for a failing assert ("add one more LCG round, A4=40692") does NOT
fix it (lag1_cell=-0.387, lag1_k=-0.442 with 4 rounds); rounds 5-7 (tried
out of caution) also fail, with lag1 in [0.27, 0.82] and no trend toward
zero. Root cause: `combine` is affine in (cell, k, bin, step) mod M31, and
each Lehmer round x -> (x*A) % M31 is a linear map over the ring Z_M31;
composing linear maps with an affine one stays affine. Along a unit-step
axis (consecutive cell or k indices), an affine map mod M31 is exactly a
Weyl/sawtooth sequence x_n = n*delta + c (mod M31) -- its lag-1 correlation
is a deterministic function of the single scalar slope `delta` and is not
damped by stacking more multiplicative rounds, since composing more linear
maps only changes `delta`, never removes the affine structure that causes
the correlation. This is structural, not statistical noise curable by "one
more round" of the same kind. FIX: insert one genuinely nonlinear mixing
round between `combine` and the Lehmer rounds -- squaring the running
state, `x = (x*x + x + 1) % M31`. Squaring makes the state quadratic (not
affine) in the original index, breaking the Weyl structure; `x*x`, `+`, `%`
are all DSL-legal integer ops, no new primitives needed. `x` stays < M31 ~=
2.1e9 at every step, so `x*x` peaks at ~4.6e18 -- comfortably inside
int64's ~9.22e18 max, no overflow. This one change takes all four quality
metrics from a hard fail to a clean pass (see RESULT rng_quality line),
still with only the brief's original 3 Lehmer rounds -- final construction
is 1 quadratic round + 3 Lehmer rounds (4 total mixing rounds after the
initial affine combine); no A4 needed.

FINDING 2 (DSL API, the potential pitfall the task brief flagged): the
brief's `_hash01` references its mixing multipliers (C_CELL, M31, A1, ...)
as bare module-level Python int globals inside the field_operator body.
That fails two different ways, verified in isolation before touching this
file:
  (a) a bare Python int literal in an int64 expression defaults to int32
      and gt4py rejects the implicit widen: `DSLError: Could not promote
      'Field[[Cell], int64]' and 'int32' to common type in call to '*'`.
  (b) wrapping the module constant as np.int64(...) fixes (a) (embedded
      then runs), but gtfn_cpu's ITIR lowering fails to resolve the
      constant as a closure symbol at all: `EveValueError: Symbols
      {SymbolRef('M31'), SymbolRef('C_CELL'), ...} not found`. This
      reproduces regardless of whether the reference is bare or wrapped in
      `astype(NAME, gtx.int64)` -- the closure *variable name* itself is
      what gtfn_cpu fails to resolve, not the type promotion.
  A bare integer *literal* (not a named global) written directly in the
  operator body, e.g. `astype(999983, gtx.int64)`, does resolve correctly
  on both backends -- but that would mean hardcoding all eight mixing
  constants as unnamed magic numbers in the body, which is worse than
  either failure mode above. The adopted fix instead promotes every mixing
  constant (c_cell, c_k, c_bin, c_step, m31, a1, a2, a3) to an explicit
  `gtx.int64` scalar parameter of `_hash01`, passed at each call site as
  `gtx.int64(C_CELL)` etc. from the still-named module-level Python int
  constants -- scalar call arguments are threaded through the compiled
  program directly and hit neither failure mode; `bin_id`/`step` were
  already scalar params in the brief, so this generalizes that existing
  idiom rather than introducing a new one. `gtx.int64(7)` scalar
  construction at the call site (the brief's other flagged pitfall) worked
  with no issue on either backend.

Run: uv run --frozen python model/atmosphere/subgrid_scale_physics/amps/spikes/spike_e_counter_rng.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import common
import gt4py.next as gtx
import numpy as np
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


M31 = 2147483647  # 2^31 - 1 (Mersenne prime, Park-Miller modulus)
A1 = 16807  # Park-Miller multipliers for the (linear) Lehmer rounds
A2 = 48271
A3 = 69621
# mixing multipliers for the initial counter combine (distinct odd primes,
# < 2^20 to keep products of int32-range counters safely inside int64)
C_CELL = 999983
C_K = 424243
C_BIN = 786433
C_STEP = 611953


def gt4py_cache_dir() -> Path:
    """The on-disk gt4py build cache directory used by this spike's run
    command (`cd $WT && uv run ...`), i.e. `<worktree_root>/.gt4py_cache`.
    Copied from spike_b_collection_codegen.py / spike_c_wide_scan.py /
    spike_d_esat.py (spikes are standalone scripts).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent / ".gt4py_cache"
    return Path.cwd() / ".gt4py_cache"


def clear_gt4py_cache() -> None:
    """Remove the gt4py on-disk build cache so the next gtfn_cpu compile is a
    genuine cold compile, not served from PERSISTENT on-disk cache left by a
    previous run. Safe: this is a rebuildable compiler-artifact cache; nothing
    else is touched.
    """
    cache_dir = gt4py_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"  (cleared gt4py cache: {cache_dir})")
    else:
        print(f"  (gt4py cache dir did not exist, nothing to clear: {cache_dir})")


@gtx.field_operator
def _hash01(  # noqa: PLR0917 -- mixing constants are scalar params, not
    # closure globals, per FINDING 2 above; that legitimately needs 8 extra
    # positional args beyond the 4 "real" ones (cell_id, k_id, bin_id, step).
    cell_id: gtx.Field[gtx.Dims[dims.CellDim], gtx.int64],
    k_id: gtx.Field[gtx.Dims[dims.KDim], gtx.int64],
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
    # mixing constants are scalar params, not module-level closure globals --
    # see module docstring FINDING 2 for why (gtfn_cpu fails to resolve
    # closure-captured constants; scalar call args work on both backends).
    one = astype(1, gtx.int64)
    x = (cell_id * c_cell + k_id * c_k + bin_id * c_bin + step * c_step + one) % m31
    x = (x * x + x + one) % m31  # nonlinear (quadratic) mixing round -- see
    # module docstring FINDING 1: breaks the affine/Weyl structure that
    # makes a pure Lehmer chain fail the lag-1 correlation bar.
    x = (x * a1) % m31
    x = (x * a2) % m31
    x = (x * a3) % m31
    return astype(x, ta.wpfloat) / 2147483647.0


def numpy_replica(ncells: int, nlev: int, bin_id: int, step: int) -> np.ndarray:
    cell = np.arange(ncells, dtype=np.int64)[:, None]
    k = np.arange(nlev, dtype=np.int64)[None, :]
    x = (cell * C_CELL + k * C_K + bin_id * C_BIN + step * C_STEP + 1) % M31
    x = (x * x + x + 1) % M31
    for a in (A1, A2, A3):
        x = (x * a) % M31
    return x.astype(np.float64) / float(M31)


def run_backend(name: str, backend, cell_id: gtx.Field, k_id: gtx.Field) -> None:
    if name == "gtfn_cpu":
        clear_gt4py_cache()

    op = _hash01.with_backend(backend) if backend is not None else _hash01
    out = common.zeros_field()

    def call():
        op(
            cell_id,
            k_id,
            gtx.int64(7),
            gtx.int64(1234),
            gtx.int64(C_CELL),
            gtx.int64(C_K),
            gtx.int64(C_BIN),
            gtx.int64(C_STEP),
            gtx.int64(M31),
            gtx.int64(A1),
            gtx.int64(A2),
            gtx.int64(A3),
            out=out,
            offset_provider={},
        )

    first, steady = common.time_first_and_steady(call)
    got = out.asnumpy()
    expected = numpy_replica(common.NCELLS, common.NLEV, 7, 1234)
    assert np.array_equal(got, expected), f"hash01 {name} != numpy replica (bit-exact required)"
    print(f"RESULT counter_rng backend={name} first={first:.2f}s steady={steady * 1e3:.2f}ms")


def main() -> None:
    cell_id = gtx.as_field((dims.CellDim,), np.arange(common.NCELLS, dtype=np.int64))
    k_id = gtx.as_field((dims.KDim,), np.arange(common.NLEV, dtype=np.int64))

    # gtfn-first (with a cold-cache clear), then embedded -- repo convention
    # from spike_c/spike_d; the brief's `common.backends().items()` loop is
    # embedded-first, reordered here for a genuine cold gtfn_cpu number.
    backends = common.backends()
    for name in ("gtfn_cpu", "embedded"):
        run_backend(name, backends[name], cell_id, k_id)

    # statistical quality on a larger sample across steps (numpy replica,
    # bit-identical to the DSL version as just asserted)
    sample = np.concatenate(
        [
            numpy_replica(common.NCELLS, common.NLEV, b, s).ravel()
            for b in range(4)
            for s in range(4)
        ]
    )
    mean, var = sample.mean(), sample.var()
    grid = numpy_replica(common.NCELLS, common.NLEV, 7, 1234)
    lag1_cell = np.corrcoef(grid[:-1].ravel(), grid[1:].ravel())[0, 1]
    lag1_k = np.corrcoef(grid[:, :-1].ravel(), grid[:, 1:].ravel())[0, 1]
    hist = np.histogram(sample, bins=16, range=(0.0, 1.0))[0] / sample.size
    hist_dev = np.abs(hist - 1.0 / 16).max()
    print(
        f"RESULT rng_quality mean={mean:.4f} var={var:.4f} "
        f"lag1_cell={lag1_cell:.4f} lag1_k={lag1_k:.4f} hist_dev={hist_dev:.4f}"
    )
    assert abs(mean - 0.5) < 0.005, "mean off"
    assert abs(var - 1.0 / 12.0) < 0.005, "variance off"
    assert abs(lag1_cell) < 0.01 and abs(lag1_k) < 0.01, "lag-1 correlation too high"
    assert hist_dev < 0.01, "histogram not flat"
    print("SPIKE E: PASS (1 quadratic mixing round + 3 Lehmer rounds)")


if __name__ == "__main__":
    main()
