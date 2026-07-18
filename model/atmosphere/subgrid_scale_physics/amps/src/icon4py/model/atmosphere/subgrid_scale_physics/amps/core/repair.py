# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`repair`'s liquid (rain) mass/concentration budget closure, transcribed
from AMPS Fortran (scale_amps repo) per
docs/superpowers/facts/m2/sedimentation-terminalvel.md ("G5" below) §4
(`mod_amps_check.F90:22-1181`, `NITER`/`NITER_TOTAL` limits quoted at G5's
own lines 2691-2692, original Fortran lines 112-113) -- M2a Task 6.

**Architecture mismatch this module resolves, and how**: G5's repair driver
operates on `dmassdt(rmt,1:n_tendpros)` -- a PER-PROCESS tendency ledger
(one slot per of ~12 named microphysical processes) that is accumulated
across `col_loop1`/`vap_loop` and only applied to `g%MS%mass` later, by
`update_group_all`. `cal_mass_budget_col`'s representative `liqbin_loop1`
block (G5 §4b, `mod_amps_check.F90:1261-1466`, quoted in full there) computes,
per bin `i` and column `n`:

    total_rgain = mass(rmt)/dt + sum_j max(+dmassdt(rmt,j), 0)*iupdate_gr(j)
    total_rloss = sum_j max(-dmassdt(rmt,j), 0)*iupdate_gr(j)

then, whenever `total_rgain < 0.99999*total_rloss .and. total_rloss>total_limit`
(the bin's mass WOULD go negative once the tendencies are applied):

    modc_r = max(0, min(1, total_rgain/max(total_rloss, total_limit)))

and multiplies every LOSS-causing process's `dmassdt(rmt,j)` (and its
`acc_mod_r` accumulator) by `modc_r`, so that when `update_group_all` later
applies `mass_new = mass/dt*dt + sum_j dmassdt(rmt,j)*dt`, the result is
guaranteed non-negative.

This port's `WarmLoopState` process hooks (`_activation`,
`_vapor_deposition_liquid`, Tasks 4/5) do NOT maintain a per-process
`dmassdt` ledger -- each hook returns the ALREADY-ADVANCED `LiquidState`
directly (`implementations/warm_loop.py`'s own module docstring: "each
process returns advanced state", `update_group_all` "inlined-by-design").
There is therefore no `dmassdt(rmt,1:n_tendpros)` array for `repair_liquid`
to rescale. This module instead ports the algorithm's NET EFFECT on a
single bin's own final mass, derived algebraically from the formulas above:

* When `total_rgain >= total_rloss` (no rescale, `modc_r=1`): final mass
  `= total_rgain - total_rloss`, already whatever it naturally was
  (non-negative by the branch condition).
* When `total_rgain < total_rloss` (rescale triggers): after multiplying
  every loss term by `modc_r = total_rgain/total_rloss`, the new loss total
  is `total_rloss*modc_r = total_rgain` EXACTLY, so the final mass is
  `total_rgain - total_rgain = 0`.

Both cases collapse to one formula: `final = max(total_rgain - total_rloss,
0)`. Since this port only ever SEES the state value `v = total_rgain -
total_rloss` (the tendencies already having been applied by the process
hooks, unlike Fortran's pre-application ledger), `total_rgain = max(v, 0)`
and `total_rloss = max(-v, 0)` by construction, and the formula reduces to
`final = max(v, 0)` -- i.e. **a non-negativity clip**, applied with the EXACT
same `0.99999`/`total_limit` trigger condition and `modc`-style multiplier
G5 uses (so the `total_limit=1e-30` floor -- tiny negative residues below it
are deliberately left untouched, matching G5 exactly -- and the `NITER`/
`NITER_TOTAL` host-loop bounds/convergence structure are preserved,
per this task's binding constraints), NOT a from-scratch `np.maximum(v, 0)`
one-liner. See `_nonneg_rescale_leg`'s own docstring for the loop.

This IS a deliberate simplification (flagged, not silent): G5's OWN
cross-bin conservation (e.g. the `j in {2,5,6,9}` collision-coalescence
branch multiplying ALL bins' `dmassdt(rmt,j)` by ONE bin's own `modc_r`,
approximately preserving whatever OTHER bins would have received from that
same redistributive process) has no analogue here -- this module's clip
touches ONLY the bin(s)/point(s) that are actually negative, leaves every
other bin exactly (bit-for-bit) unchanged, and does NOT attempt any
cross-bin mass redistribution to "pay for" the clipped deficit. This is
LESS strict than what G5 achieves for its zero-sum processes, matching this
task's own binding guidance ("match its [Fortran's] behavior, don't impose
stricter conservation than Fortran") -- inventing a column-total-preserving
redistribution scheme would be a stricter invariant than G5 itself
guarantees for arbitrary process mixes.

**Which PPV legs are repaired**: G5's mass/concentration/volume triad
(`cal_mass_budget_col`, `cal_con_budget_col`, `cal_vol_budget_col` --
"both follow the identical gain/loss -> modc -> per-process multiply
pattern, keyed on concentration and Vcs respectively", G5 §4b) maps onto
`LiquidState`'s 4 `LiquidPPV` members as follows:

* mass (`rmt_q`) -- `cal_mass_budget_col`'s own quoted block, ported.
* concentration (`rcon_q`) -- `cal_con_budget_col`, "identical pattern,
  keyed on concentration" per G5, ported with the SAME `_nonneg_rescale_leg`
  helper.
* volume -- `cal_vol_budget_col` is keyed on `Vcs` (circumscribing volume),
  which is an ICE-ONLY state field (`IcePPV.ivcs_q`, `core/index_maps.py`);
  `LiquidPPV` has NO volume/`Vcs` member at all (only `rmt_q`, `rcon_q`,
  `rmat_q` "total aerosol mass", `rmas_q` "soluble aerosol mass" --
  `core/index_maps.py`'s own citation of F2 §2 lines 365-367). Liquid/rain
  drops are diagnostically spherical (volume = mass/density, no
  independently-predicted shape DOF the way ice crystals have via
  `iacr_q`/`iccr_q`/`idcr_q`/axis lengths), so there is no PPV field for
  this leg to operate on -- this is a documented NO-OP for
  `repair_liquid`, not a silently-dropped leg.
* `rmat_q`/`rmas_q` (aerosol mass carried by a liquid bin) are OUT OF
  SCOPE: G5 §4b's quoted block does not address them (they are handled
  elsewhere in the Fortran repair driver via `ratio_M`-consistency logic,
  not shown in G5's excerpt), so this module does not invent behavior for
  them.
"""

from __future__ import annotations

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import LiquidState


# G5 mod_amps_check.F90:112 (original Fortran; quoted at
# docs/.../sedimentation-terminalvel.md:2691): inner per-budget convergence
# loop cap.
NITER = 300
# G5 mod_amps_check.F90:113 (quoted at .../sedimentation-terminalvel.md:2692):
# outer mass/con/vol coupled-budget loop cap.
NITER_TOTAL = 4
# G5 cal_mass_budget_col: `real(PS),parameter :: total_limit=1.0e-30_PS`
# (.../sedimentation-terminalvel.md:2783). Losses at or below this floor are
# left untouched (matching the Fortran's own `total_rloss>total_limit` guard
# on the rescale trigger), not clipped.
TOTAL_LIMIT = 1.0e-30


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Matches `core/vapor_deposition.py`'s/`core/liquid_diag.py`'s own
    identical-purpose, independently-duplicated `_safe_div` precedent
    (`core/` modules do not cross-import each other's private helpers)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )


def _nonneg_rescale_leg(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One "leg" of G5's repair (`cal_mass_budget_col`'s `liqbin_loop1` /
    the analogous `cal_con_budget_col`, collapsed to state level -- see
    module docstring for the tendency-vs-state equivalence derivation): the
    host-side `do itr1=1,NITER` `any()`-guarded convergence loop (G5 §4a:
    "inner `do itr1=1,NITER(=300)` per budget, exiting early on
    `.not.any(mark_mod>0)`").

    Per element: `total_rgain=max(value,0)`, `total_rloss=max(-value,0)`
    (G5's own decomposition, given this port only has the POST-tendency
    state value, not the pre-application ledger); the exact G5 trigger
    (`total_rgain < 0.99999*total_rloss .and. total_rloss>total_limit`)
    and clamp (`modc_r=max(0,min(1,gain/max(loss,total_limit)))`) are
    applied verbatim; `value*modc_r` reproduces the algebraic
    `max(value,0)` result for triggered elements (module docstring),
    untouched elements pass through bit-for-bit.

    Returns `(rescaled_value, mark)`: `mark` is G5's own `mark_mod`
    equivalent (nonzero wherever this pass rescaled an element), consumed
    by the outer `NITER_TOTAL` driver's `mark_all`/`sum()==0` convergence
    check (`repair_liquid` below).
    """
    v = value.copy()
    mark_total = np.zeros(v.shape, dtype=np.int64)
    itr1 = 0
    while itr1 < NITER:
        total_rgain = np.maximum(v, 0.0)
        total_rloss = np.maximum(-v, 0.0)
        trigger = (total_rgain < 0.99999 * total_rloss) & (total_rloss > TOTAL_LIMIT)
        if not np.any(trigger):
            break
        modc_r = np.clip(_safe_div(total_rgain, np.maximum(total_rloss, TOTAL_LIMIT)), 0.0, 1.0)
        v = np.where(trigger, v * modc_r, v)
        mark_total = np.where(trigger, mark_total + 1, mark_total)
        itr1 += 1
    else:
        # Loop ran NITER times without a clean `any(trigger)==False` exit --
        # G5's own non-convergence branch calls `PRC_abort` here
        # (mod_amps_check.F90, both `if(any(mark_mod(1:ag%L)>0)) ... call
        # PRC_abort` sites, G5 §4a). This should be unreachable in practice
        # (the derivation above shows one pass always resolves every
        # trigger), but is kept as a defensive, Fortran-fidelity guard
        # rather than silently returning a still-negative result.
        total_rgain = np.maximum(v, 0.0)
        total_rloss = np.maximum(-v, 0.0)
        if np.any((total_rgain < 0.99999 * total_rloss) & (total_rloss > TOTAL_LIMIT)):
            raise RuntimeError(
                f"repair_liquid: mass/concentration budget did not converge within "
                f"NITER={NITER} iterations (G5 mod_amps_check.F90 non-convergence -> "
                f"PRC_abort)."
            )
    return v, mark_total


def repair_liquid(liquid: LiquidState, config: AmpsConfig) -> LiquidState:
    """`repair`'s liquid mass/concentration/volume non-negativity closure
    (G5 §4, M2a Task 6) -- see module docstring for the full derivation,
    scope (mass + concentration; volume is a documented no-op for
    `LiquidState`), and the state-level simplification this necessitates.

    Host-side outer loop, G5 §4a's `do itr2=1,NITER_TOTAL` structure
    ("outer `NITER_TOTAL` loop wrapping three sub-blocks (mass, con, vol)
    ... exiting when `sum(mark_all)==0`"): each pass runs the mass leg then
    the concentration leg (matching G5's own `ifix_mass_col` before
    `ifix_con_col` ordering), accumulates `mark_all` (G5's own
    `mark_mass_r + mark_con_r + ...` sum), and exits early once nothing was
    rescaled. `config` is accepted for interface parity with the other two
    process hooks (`_activation`, `_vapor_deposition_liquid` in
    `implementations/warm_loop.py`) but UNUSED -- this closure's trigger/
    clamp formulas (G5 §4b) have no `AmpsConfig`-dependent branching.

    Args:
        liquid: current `LiquidState` (any `nbins`/`ncat`/`npoints`).
        config: `AmpsConfig` (unused, see above).

    Returns:
        A NEW `LiquidState` (input `liquid` is never mutated -- `.values`
        is copied before any modification) with `rmt_q`/`rcon_q` clipped
        non-negative per G5's own trigger/floor semantics; every other
        element (including `rmat_q`/`rmas_q`, out of this module's scope,
        and any bin/point that was never negative) is bit-for-bit
        unchanged.
    """
    del config  # unused, see docstring.

    lp = LiquidPPV
    values = liquid.values.copy()
    mass = values[lp.rmt_q.py_idx]
    con = values[lp.rcon_q.py_idx]

    itr2 = 0
    while itr2 < NITER_TOTAL:
        mass, mark_mass = _nonneg_rescale_leg(mass)
        con, mark_con = _nonneg_rescale_leg(con)
        mark_all = int(mark_mass.sum()) + int(mark_con.sum())
        if mark_all == 0:
            break
        itr2 += 1

    values[lp.rmt_q.py_idx] = mass
    values[lp.rcon_q.py_idx] = con
    return LiquidState(values=values)
