# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`repair`'s liquid budget closure -- BOTH of its two independent phases
(`mod_amps_check.F90:22-1181`, `NITER`/`NITER_TOTAL` limits quoted at
docs/superpowers/facts/m2/sedimentation-terminalvel.md ("G5" below) §4,
lines 2691-2692, original Fortran lines 112-113) -- M2a Task 6:

* `repair_liquid` -- the af_col (`irep_col`) phase: PER-BIN mass/
  concentration/volume non-negativity closure, `cal_mass_budget_col` /
  `cal_con_budget_col` / `cal_vol_budget_col` (G5 §4b, `mod_amps_check.F90:
  1261-1466` for the mass leg, quoted in full there).
* `repair_vapor` -- the af_vap (`irep_vap`) phase: a COLUMN/point-level
  vapor-SUPPLY closure, `cal_mass_budget_vapor` (`mod_amps_check.F90:
  2034-3145`) -- NOT the same algorithm as af_col (see that function's own
  docstring). G5 itself only quotes this phase's OUTER driver loop (§4a);
  `cal_mass_budget_vapor`'s own body is NOT quoted by G5 and was read
  directly from `contrib/AMPS/mod_amps_check.F90` (dispatch-authorized),
  cited by exact line below.

These two phases are mutually exclusive per `repair(...)` call (G1 §1: af_col
is called with masks `(irep_vap,irep_col)=(.false.,.true.)`, af_vap with
`(.true.,.false.)`) -- `implementations/warm_loop.py`'s `_repair` dispatches
`phase="collision"` to `repair_liquid` and `phase="vapor"` to `repair_vapor`
accordingly; earlier drafts of this module INCORRECTLY called `repair_liquid`
for both phases (caught in code review) -- see `repair_vapor`'s own
docstring for exactly what that got wrong (it clipped `rcon_q` on every
vapor substep, which af_vap's real Fortran algorithm never touches at all).

## repair_liquid (af_col) -- architecture mismatch this module resolves, and how

G5's repair driver operates on `dmassdt(rmt,1:n_tendpros)` -- a PER-PROCESS
tendency ledger (one slot per of ~12 named microphysical processes) that is
accumulated across `col_loop1`/`vap_loop` and only applied to `g%MS%mass`
later, by `update_group_all`. `cal_mass_budget_col`'s representative
`liqbin_loop1` block (G5 §4b, `mod_amps_check.F90:1261-1466`, quoted in full
there) computes, per bin `i` and column `n`:

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
`final = max(v, 0)`. `_nonneg_rescale_leg` computes this using G5's own
`0.99999`/`total_limit`/`modc`-style variable names and floor (so the
`total_limit=1e-30` floor -- tiny negative residues below it are
deliberately left untouched, matching G5 exactly -- and the `NITER`/
`NITER_TOTAL` host-loop bounds/convergence structure are preserved, per
this task's binding constraints), NOT a from-scratch `np.maximum(v, 0)`
one-liner written without reference to G5's own formula shape.

**Correction (code review): the `0.99999` trigger is NOT "applied verbatim"
in the sense of reproducing G5's own tolerance behavior** -- an earlier
draft of this docstring overclaimed this. In G5, `total_rgain`/`total_rloss`
are two INDEPENDENTLY-computed, generally large, positive sums (of several
process tendencies each); the `0.99999` factor creates a genuine tolerance
band: whenever `0.99999*total_rloss <= total_rgain < total_rloss` (gain
undershoots loss by LESS than 0.001%), Fortran does NOT trigger a rescale,
so that bin's naively-summed final mass is left MARGINALLY NEGATIVE (by up
to ~0.001% of `total_rloss`) uncorrected. This port never sees that
undershoot band as a genuine two-term comparison: given only the collapsed
`v`, `total_rgain=max(v,0)` is EXACTLY `0` for any `v<0`, so
`0 < 0.99999*total_rloss` holds for essentially any negative `v` down to
`total_limit` -- i.e. the `0.99999` tolerance band is structurally
UNREACHABLE here; only the `total_limit` floor is load-bearing. Net effect:
this port clips SOME bins non-negative that Fortran's own tolerance would
have left marginally (`<~1e-5` relative) negative -- a small, one-directional,
documented divergence to expect in any per-call validation against Fortran
dumps, not a bug.

**Mass and concentration are NOT independent in G5, unlike in this port**
(second code-review correction): after `cal_mass_budget_col`'s inner
`NITER` loop converges, the driver calls `mod_other_tendency(gr, acc_mod_r,
1, iupdate_gr)` (`mod_amps_check.F90:4377-4460`, `switch=1` case,
`mod_amps_check.F90:4390-4408` specifically) -- for EVERY tendency process
`j`, this multiplies `g%MS(i,n)%dcondt(j)` (the CONCENTRATION tendency,
read directly, confirmed by exact line) by the SAME `acc_mod_r(i,j,n)`
accumulator the mass pass just built up. So by the time `cal_con_budget_col`
runs its own `NITER` loop, its input `dcondt` array has ALREADY been
pre-scaled by whatever the mass pass decided. This module's `mass` and
`con` legs, by contrast, run on two COMPLETELY SEPARATE arrays with zero
cross-talk (`repair_liquid`'s `while itr2<NITER_TOTAL` loop below calls
`_nonneg_rescale_leg(mass)` then `_nonneg_rescale_leg(con)` independently).
Per-call divergence from Fortran dumps is therefore EXPECTED, not an edge
case, for ANY bin where multiple processes interact across both mass and
concentration in the same substep -- i.e. the general/common case, not a
rare corner.

**Minor footnote**: `cal_con_budget_col` has one degenerate special case
this module does not need to replicate (`mod_amps_check.F90:3251-3258`):
when the concentration loss is caused SOLELY by process 1 (vapor
deposition/evaporation, `total_rloss==max(-dcondt(1),0)`), Fortran sets
`dcondt(1)=-total_rgain` directly (an ADDITIVE correction) instead of the
usual multiplicative `modc_r` -- a numerical-precision special case for
that one degenerate sub-branch. This port's `max(v,0)` closure reaches the
identical `v=0` (or `v=total_rgain`, non-negative) result regardless of
which of Fortran's two code paths produced it, so no separate branch is
needed here.

**Which PPV legs `repair_liquid` repairs**: G5's mass/concentration/volume
triad (`cal_mass_budget_col`, `cal_con_budget_col`, `cal_vol_budget_col` --
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

## repair_vapor (af_vap) -- see its own docstring below for the full account.
"""

from __future__ import annotations

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.packing import get_thermo_prop
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# G5 mod_amps_check.F90:112 (original Fortran; quoted at
# docs/.../sedimentation-terminalvel.md:2691): inner per-budget convergence
# loop cap -- shared by BOTH af_col's per-leg loop (`_nonneg_rescale_leg`)
# and af_vap's own single loop (`repair_vapor`).
NITER = 300
# G5 mod_amps_check.F90:113 (quoted at .../sedimentation-terminalvel.md:2692):
# outer mass/con/vol coupled-budget loop cap -- af_col (`repair_liquid`)
# ONLY; af_vap's driver loop has no such outer layering (G5 §4a: "Phase 1 --
# vapor ... then the inner convergence loop", a single `do itr1=1,NITER`,
# confirmed by direct read of `cal_mass_budget_vapor`'s caller).
NITER_TOTAL = 4
# G5 cal_mass_budget_col: `real(PS),parameter :: total_limit=1.0e-30_PS`
# (.../sedimentation-terminalvel.md:2783); `cal_mass_budget_vapor` uses the
# IDENTICAL literal (`mod_amps_check.F90:2103`, `real(PS),parameter ::
# total_limit=1.0e-30_PS`) -- shared by both phases here too.
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
    """One "leg" of G5's af_col repair (`cal_mass_budget_col`'s
    `liqbin_loop1` / the analogous `cal_con_budget_col`, collapsed to state
    level -- see module docstring for the tendency-vs-state equivalence
    derivation AND the two code-review corrections on how closely this
    matches G5's own `0.99999` tolerance / mass-con independence). The
    host-side `do itr1=1,NITER` `any()`-guarded convergence loop (G5 §4a:
    "inner `do itr1=1,NITER(=300)` per budget, exiting early on
    `.not.any(mark_mod>0)`").

    Per element: `total_rgain=max(value,0)`, `total_rloss=max(-value,0)`
    (G5's own decomposition, given this port only has the POST-tendency
    state value, not the pre-application ledger); G5's trigger
    (`total_rgain < 0.99999*total_rloss .and. total_rloss>total_limit`) and
    clamp (`modc_r=max(0,min(1,gain/max(loss,total_limit)))`) are applied
    using G5's own variable shapes (module docstring: the `0.99999` band is
    structurally unreachable here, only `total_limit` is load-bearing);
    `value*modc_r` reproduces the algebraic `max(value,0)` result for
    triggered elements, untouched elements pass through bit-for-bit.

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
    """`repair`'s af_col (`irep_col`) phase: liquid mass/concentration/
    volume non-negativity closure (G5 §4, M2a Task 6) -- see module
    docstring for the full derivation, scope (mass + concentration; volume
    is a documented no-op for `LiquidState`), and the state-level
    simplification this necessitates. This is NOT the same algorithm as
    `repair_vapor` (af_vap) -- see that function's own docstring for why
    the two must not be conflated.

    Host-side outer loop, G5 §4a's `do itr2=1,NITER_TOTAL` structure
    ("outer `NITER_TOTAL` loop wrapping three sub-blocks (mass, con, vol)
    ... exiting when `sum(mark_all)==0`"): each pass runs the mass leg then
    the concentration leg (matching G5's own `ifix_mass_col` before
    `ifix_con_col` ordering), accumulates `mark_all` (G5's own
    `mark_mass_r + mark_con_r + ...` sum), and exits early once nothing was
    rescaled. `config` is accepted for interface parity with the other
    process hooks (`_activation`, `_vapor_deposition_liquid`, `_repair` in
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


# ---------------------------------------------------------------------------
# repair_vapor: the af_vap (irep_vap) phase, ported from the Fortran
# subroutine cal_mass_budget_vapor at mod_amps_check.F90 lines 2034 through
# 3145. This is a DIFFERENT algorithm than repair_liquid (af_col) above --
# see repair_vapor's own docstring below. Read directly from the Fortran
# source (G5 does not quote this routine's own body, only its outer driver
# loop, G5 section 4a "Phase 1 -- vapor").
# ---------------------------------------------------------------------------


def repair_vapor(
    liquid: LiquidState, thermo_state: ThermoState, config: AmpsConfig
) -> tuple[LiquidState, ThermoState]:
    """`repair`'s af_vap (`irep_vap`) phase: the COLUMN/point-level
    VAPOR-SUPPLY closure, `cal_mass_budget_vapor` (`mod_amps_check.F90:
    2034-3145`, read directly -- not quoted in full by G5). This is a
    DIFFERENT algorithm from `repair_liquid` (af_col) -- an earlier draft of
    this module incorrectly called `repair_liquid` for BOTH `_repair`
    phases (caught in code review), which is wrong on two counts: (1) it
    clips `rcon_q` (concentration) on every vapor substep (10x per
    collision step, cloudlab's `n_step_vp=10`), but `cal_mass_budget_vapor`
    NEVER reads or writes `dcondt` anywhere in its 1111-line body (confirmed
    by direct read) -- concentration is entirely outside af_vap's scope;
    (2) af_col repairs each BIN's own mass/con budget independently, while
    af_vap repairs ONE shared point-level vapor-supply budget against
    EVERY bin's condensational consumption at once.

    **The real Fortran algorithm** (`mod_amps_check.F90:2034-3145`): per
    column/point `n`,

        mass_v(n) = ag%TV(n)%rv * ag%TV(n)%den        ! line 2114, available vapor "supply"

        total_vloss(n) = sum_i max(+dmassdt_water(i,n), 0)   ! condensation, all bins (lines 2221-2226)
        total_vgain(n) = sum_i max(-dmassdt_water(i,n), 0) + mass_v(n)/dt   ! evaporation + existing supply (2227-2231)

        if total_vgain(n) < 0.99999*total_vloss(n) and total_vloss(n) > total_limit:   ! line 2235
            modc_v(n) = max(0, min(1, total_vgain(n)/max(total_vloss(n), total_limit)))   ! line 2236

    then EVERY bin's condensation-consuming tendency (`dmassdt(rmt,1)`
    "vapor deposition", `dmassdt(rmt,3)` "post-activation growth", PLUS
    their `rmat` aerosol-mass-tendency counterparts) is multiplied by the
    SAME shared `modc_v(n)` (`mod_amps_check.F90:2260-2296`) -- one scalar
    per column, applied uniformly, unlike af_col's per-bin `modc_r`. This
    runs ONCE through the driver's `do itr1=1,NITER` loop (G5 §4a "Phase 1
    -- vapor"; NO `NITER_TOTAL` outer layering the way af_col's mass/con/vol
    triad has -- confirmed by direct read of the driver around G5's own
    quoted "Phase 1" excerpt).

    Because `total_vgain(n)` always includes the non-negative `mass_v(n)/dt`
    term and `modc_v(n)` is clamped into `[0,1]`, G5's OWN algorithm
    structurally CANNOT produce a negative final vapor mass: the worst case
    (`modc_v=0`) reduces ALL condensational consumption to zero, leaving
    vapor at (at least) its pre-substep level. This port's state-level
    adaptation below does not have that same structural guarantee (see
    "State-level adaptation" below) -- flagged there, not glossed over.

    **State-level adaptation** (the SAME "no per-process tendency ledger"
    limitation `repair_liquid`/module-docstring already accepts): this port
    only ever sees the POST-hook `thermo_state.qvv` (vapor mixing ratio --
    this port's `mass_v` analog, ALREADY per-unit-moist-air-mass and
    directly additive with `LiquidState`'s mass fields, matching
    `core.vapor_deposition.vapor_deposition_liquid`'s own established
    `qv_new = qv_old + water_to_vapor` convention -- reused here, not
    re-derived, and NOT `_update_mesrc_warm`'s `qvv*moist_denv` mass-density
    reduction, which is a different quantity used for a different purpose),
    not a decomposed per-bin condensational-gain tendency to rescale
    multiplicatively. Whenever `qvv < 0` (the exact failure mode af_vap
    exists to prevent), the deficit `-qvv` is given back: `qvv` is floored
    at 0, and an EQUAL water mass is removed from `liquid`'s `rmt_q`
    (total mass), taken PROPORTIONALLY from every bin by its current share
    of total column mass -- the closest state-level analogue of G5's own
    single-shared-`modc_v`-applied-to-every-bin structure, given this port
    has no per-bin condensational-GAIN-only tendency to target specifically
    (a proportional-by-mass-share give-back is used instead of a uniform
    fractional shrink so that empty bins never go negative).

    `rcon_q` (concentration) is NEVER touched -- matching G5's own array
    scope exactly (see "wrong on two counts" above). `rmat_q`/`rmas_q`
    (aerosol content) are also left untouched, matching `repair_liquid`'s
    own documented scope limit (this function only gives back WATER mass,
    which does not change a bin's dissolved-aerosol content).

    **Where this can honestly diverge from G5's guarantee**: if the total
    liquid mass currently available (summed over bins, per point) is LESS
    than the vapor deficit, this function can only give back what exists --
    `qvv` may remain negative after the call (a state-only-architecture
    edge case: `vapor_deposition_liquid`'s own conservation ties `qvv`'s
    decrease directly to a matching liquid increase in the SAME substep, so
    this should not arise in practice for a state produced by this port's
    own `run_warm_micro_tendency` wiring, but is not assumed away here).
    The convergence loop below detects "no further mass available to give
    back" and exits early rather than looping to `NITER` uselessly or
    raising a spurious non-convergence error for what is a genuine physical
    limit, not a bug.

    Args:
        liquid: current `LiquidState` (post-activation, post-vapor-
            deposition, per `implementations/warm_loop.py`'s vap-loop
            ordering -- the point at which G1 §1 calls af_vap).
        thermo_state: current `ThermoState` (same point).
        config: `AmpsConfig` (unused, interface parity -- see
            `repair_liquid`'s own identical note).

    Returns:
        `(liquid', thermo')`: `liquid'.values[rmt_q]` reduced (never below
        0 per bin) by exactly the vapor deficit taken back, distributed
        proportionally by each bin's mass share; `thermo'.qvv` increased by
        that same amount (floored at 0 when enough mass was available);
        every other field -- INCLUDING `rcon_q`, `rmat_q`, `rmas_q`, and
        every other `ThermoState` field -- is bit-for-bit unchanged.
    """
    del config  # unused, see docstring.

    qvv = get_thermo_prop(thermo_state, ThermoProp.qvv)

    lp = LiquidPPV
    liquid_values = liquid.values.copy()
    mass = liquid_values[lp.rmt_q.py_idx]  # (nbins, ncat, npoints)

    qvv_new = qvv.copy()
    itr1 = 0
    while itr1 < NITER:
        deficit = np.maximum(0.0, -qvv_new)
        trigger = deficit > TOTAL_LIMIT
        if not np.any(trigger):
            break

        total_mass = np.sum(mass, axis=(0, 1))  # (npoints,) -- per-point column total
        actual_giveback = np.minimum(deficit, total_mass)
        if not np.any(actual_giveback > TOTAL_LIMIT):
            # No liquid mass left to give back at any still-triggered point
            # -- a genuine physical limit (see docstring), not a
            # non-convergence bug: further iterations would not change
            # anything, so stop here rather than spinning to NITER.
            break

        safe_total = np.where(total_mass > 0.0, total_mass, 1.0)
        weight = np.where(total_mass[None, None, :] > 0.0, mass / safe_total[None, None, :], 0.0)
        take_back = weight * actual_giveback[None, None, :]
        mass = (
            mass - take_back
        )  # always >=0 per bin: take_back_i <= mass_i (weight<=1, giveback<=total)
        qvv_new = qvv_new + actual_giveback
        itr1 += 1

    liquid_values[lp.rmt_q.py_idx] = mass
    liquid_out = LiquidState(values=liquid_values)

    thermo_values = thermo_state.values.copy()
    qvv_idx = list(ThermoState.PROPS).index(ThermoProp.qvv)
    thermo_values[qvv_idx, 0, 0, :] = qvv_new
    thermo_out = ThermoState(values=thermo_values)

    return liquid_out, thermo_out
