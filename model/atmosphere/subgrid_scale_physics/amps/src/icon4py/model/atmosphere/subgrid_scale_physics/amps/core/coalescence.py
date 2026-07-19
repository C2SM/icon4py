# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`coalesce_rain` -- the rain-rain (`token 1-1`) bin-pair collection
engine, transcribed (with an explicitly-sanctioned simplification, see
below) from AMPS Fortran (scale_amps repo) per
docs/superpowers/facts/m2b/coalescence-engine.md ("H1" below, the gap-fill
supplement, SS0/SS1.6/SS5) and docs/superpowers/facts/m2/coalescence.md
("G4" below, the base extraction, SS1/SS4.1/SS6). M2b Task 3.

SCOPE AND SIMPLIFICATION (read this first): the real Fortran
`collector_loop1` (H1 SS1.6, G4 SS1.6, "the hardest M2 kernel") builds, for
EACH collector bin `i`, a ragged set of sub-bins categorized by collection
RATE ("high"/"fortunate"/"low-medium" -- `jbin_h`/`jbin_lm`), accumulates a
mass/number PDF per sub-bin, remaps it onto shifted bin boundaries via a
linear/cubic fit (`cal_lincubprms_vec`/`cal_linprms_vec_s`), and scatters
via `cal_transbin_vec`. This task's own dispatch explicitly authorizes a
SIMPLER model instead, quoting H1's own "M2b codegen notes": "The
add_simple_vec boundary-search (G4 SS4.1) is the reference for the simpler
mean-mass jbin search M2 may emit as a fallback" -- and the dispatch's own
Deliverable bullet list names exactly this fallback's ingredients directly
("coalesced mass m_i+m_j, destination-bin search + scatter
(add_simple_vec/add_samebin_vec)"), not the sub-bin PDF machinery.

The model this module implements, per ORDERED bin PAIR `(i, j)` (`i` the
collector, `j` the collectee, exactly as H1/G4's own axis convention) and
per column point:

    n_ij = N_col(i,j) * E_coal(i,j)   -- number of COALESCENCE EVENTS

Each event is a literal "2 parents in, 1 merged child out" transformation
(the standard discretized stochastic-collection-equation picture -- ANY
two colliding, coalescing drops of different bins cease to exist as
separate entities and become exactly ONE new drop; this is not a modeling
choice specific to this port, it is what "coalescence" means): bin `i`
loses `n_ij` drops (mass `mean_mass_i` each), bin `j` loses `n_ij` drops
(mass `mean_mass_j` each), and `n_ij` NEW drops of mass
`mean_mass_i+mean_mass_j` each are scattered into the destination bin found
by `add_simple_vec`'s own boundary search (G4 SS4.1 verbatim,
`_find_destination_bin` below). This exactly conserves mass BY
CONSTRUCTION (removed mass `n_ij*(mean_mass_i+mean_mass_j)` telescopes
identically against added mass, summed over every pair -- see
`_needgive_repair`'s own docstring and the task report for the full
derivation) and strictly DEPLETES number (2 removed, ~1 added, net -1 per
event) -- matching this task's own conservation requirements exactly.

This DIFFERS from the true Fortran's own bin-`i`-preserving picture (where
a single massive collector bin literally keeps its particle count and only
grows in mean mass, feeding many `ndrop_B>1` collectee absorptions per
collector particle in the "H" pass) -- but is EXACTLY the fallback H1's own
"M2b codegen notes" name, and is the standard formulation used by
textbook/other bin-microphysics stochastic-collection schemes (e.g. Bott
1998's own flux method is a different discretization of the SAME
underlying two-body physics). Flagged here, not silently substituted; see
the task report for the full account of what is and is not modeled by this
simplification (in particular: the H/F/LM rate categorization, `iter_loop1`
over-depletion PRE-normalization, and `used_marker`/`left_N`/`left_M`
leftover-repopulation machinery are ALL folded into this module's simpler
per-pair depletion + `_needgive_repair` post-hoc fix-up instead).

PER-VOLUME basis (H1 SS5, the M2a-lesson-driven convention `core/
collision_kernel.py` already documents at length): this module's INPUT
`liquid_pv` AND its OUTPUT are per-VOLUME quantities -- `con`/`mass` =
`q_state * den` (mixing ratio times air density), NOT per-mass mixing
ratios. Exactly like `collision_kernel.py`, this module itself contains NO
`den` factor anywhere ("no stray `/den` or `*den` appears anywhere inside
`coalescence`" -- H1 SS5); the `*den`/`/den` round-trip at the
mixing-ratio/per-volume boundary is the CALLER's job (M2b Task 7's wiring
into `implementations/warm_loop.py`), mirroring `class_Group.F90:756-758`/
`:3910-3921` exactly. Do not multiply/divide by density anywhere in this
module or its tests.

Loop bounds/order, H1 SS1.6 verbatim (`mod_amps_core.F90:1971`,
`icolbin_min` construction `:1518-1523`):

    icolbin_min=1
    do i=1,g_1%N_bin
      if(g_1%binb(i+1)<amin_mass) then
        icolbin_min=i
      endif
    enddo

    collector_loop1: do i=g_1%N_bin,icolbin_min,-1

`amin_mass=4.188790d-12` (H1's own quoted parameter, G4 SS1.1) is the
minimum COLLECTOR mass; bins below `icolbin_min` may still act as
COLLECTEES (only the collector role, axis 0 / `i`, is restricted -- see
`_icolbin_min`'s own docstring for the Fortran's own off-by-one framing,
preserved verbatim, not "fixed"). Since this module is fully vectorized
(no Python loop over collector bins is needed -- there is no sequential
dependency between different `i` in this simplified per-pair model, unlike
the true Fortran's `left_N`/`used_marker` threading), `icolbin_min` is
applied as a boolean mask over axis 0 rather than a literal `for` loop --
mathematically identical to the Fortran's own descending `do i=N_bin,
icolbin_min,-1` (order does not matter for a purely per-pair-elementwise
computation with no cross-`i` state).

Self-collection (`i==j`) + kernel (a)symmetry: the kernel is NOT symmetric
(`KC(i,j) != KC(j,i)` in general -- H1's own `KC=E_c*(vtm_i-vtm_j)*A_c*
con_j*dt` formula uses `con_j`, not `con_i`, and the `vtm_i-vtm_j` term
flips sign under `i<->j`), so this module computes the FULL `(nbins,
nbins)` pair grid, never assuming an upper-triangle shortcut. `i==j` is a
proven, automatic no-op: `KC(i,i)=E_c*(vtm_i-vtm_i)*A_c*con_i*dt=0`
identically (the `vtm_i-vtm_i` factor is exactly zero since `diag_i` and
`diag_j` are the SAME data at the SAME bin) -- so `N_col(i,i)=0` via the
`KC>0` branch's own `else` default, for ANY con/mass/dt magnitude; see
`TestActiveBinsAndSelfCollection.test_self_collection_is_a_no_op`.

ibreak (collisional breakup) hook: `ibreak=True` is M2b Task 5 scope
(`add_fragments_col_vec`, H1 SS1.6j/G4 SS2.3, gated by
`pro_type==2.and.ibreak==1` -- ITSELF gated OFF for the warm rain-rain
`pro_type` in this port's own scope) -- this engine only implements the
`ibreak=0` (no-breakup) path; passing `ibreak=True` raises
`NotImplementedError` immediately rather than silently ignoring the flag.
"""

from __future__ import annotations

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import (
    bin_grid,
    collision_kernel as ck,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import AmpsLuts
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import LiquidState, ThermoState


# G4 SS1.1 (`mod_amps_core.F90:1397-1400`): "minimum collector mass to be
# considered", `real(PS),parameter :: amin_mass=4.188790d-12`.
AMIN_MASS = 4.188790e-12

# G4 SS2 (`mod_amps_core.F90:15894-15907`): icond1 activation floor, shared
# by both `con` and `mass(1)` -- matches `core/collision_kernel.py`'s own
# module docstring, which names this exact mask ("icond1") as the caller's
# responsibility to apply.
_ACTIVE_FLOOR = 1.0e-30


def _pairwise(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast two `(nbins, npoints)` per-bin arrays to the bin-PAIR
    shape `(nbins_i, nbins_j, npoints)` -- `a` (group 1/"i"/collector)
    along axis 0, `b` (group 2/"j"/collectee) along axis 1. A local copy
    of `collision_kernel.py`'s own identical-purpose helper (`core/`
    modules do not cross-import each other's private helpers -- matches
    `core/repair.py`'s own documented precedent for `_safe_div`)."""
    a_b, b_b = np.broadcast_arrays(a[:, None, :], b[None, :, :])
    return a_b, b_b


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Matches `core/liquid_diag.py`'s/`core/repair.py`'s own
    identical-purpose, independently-duplicated `_safe_div` precedent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )


def _icolbin_min(binb: np.ndarray) -> int:
    """H1 SS1.6's `icolbin_min` construction, G4 verbatim
    (`mod_amps_core.F90:1518-1523`):

        icolbin_min=1
        do i=1,g_1%N_bin
          if(g_1%binb(i+1)<amin_mass) then
            icolbin_min=i
          endif
        enddo

    Returns the 0-based Python collector-bin index (Fortran's 1-based
    `icolbin_min` minus 1): the SMALLEST bin index still eligible to act
    as a COLLECTOR (`collector_loop1: do i=g_1%N_bin,icolbin_min,-1` is
    inclusive of `icolbin_min` itself). Every bin remains eligible as a
    COLLECTEE regardless -- only the collector role (axis 0 of the
    bin-pair grid) is restricted by this floor.

    Preserved verbatim INCLUDING the Fortran's own off-by-one framing: the
    LAST bin index `i` for which `binb(i+1)<amin_mass` (i.e. the bin whose
    entire mass range is still below the collector floor) is itself
    INCLUDED as a valid collector by the `do i=N_bin,icolbin_min,-1`
    lower bound, not excluded. This is not "fixed" here -- transcribed as
    written.
    """
    nbins = binb.shape[0] - 1
    icolbin_min = 1  # Fortran 1-based, matches the literal init value
    for i in range(1, nbins + 1):  # Fortran i=1..N_bin
        if binb[i] < AMIN_MASS:  # Fortran binb(i+1) == python 0-based binb[i]
            icolbin_min = i
    return icolbin_min - 1  # convert to 0-based Python collector-bin index


def _find_destination_bin(binb: np.ndarray, target_mass: np.ndarray) -> np.ndarray:
    """`add_simple_vec`'s own destination-bin search, G4 SS4.1 verbatim
    (`mod_amps_core.F90:15452-15499`):

        if binb(N_binb) < target: jbin = N_BIN        (above the top -> last bin)
        elif binb(1) > target:    jbin = 1             (at/below the bottom -> first bin)
        else: jbin = the j with binb(j) < target <= binb(j+1)

    `binb` is 1D (`nbins+1` bin-mass boundaries, shared by both the
    collector and collectee axes -- rain-rain is a single species/group);
    `target_mass` is any shape. Returns the 0-based bin index, same shape
    as `target_mass`, ALWAYS a valid index in `[0, nbins-1]` (this port's
    own defensible substitute for the Fortran's own undefined behavior at
    an exact `target==binb(1)` boundary match -- an unreachable
    floating-point coincidence in practice, see the task report).

    Implemented via `np.searchsorted` (binb is sorted/monotonic by
    construction, `core/bin_grid.py`): for `target` in
    `(binb[b], binb[b+1]]`, `searchsorted(binb, target, side="left")`
    returns `b+1` (the first boundary `>= target`), so `-1` gives the
    0-based bin index `b` directly -- exactly the Fortran's own strict-
    lower/non-strict-upper bin semantics.
    """
    nbins = binb.shape[0] - 1
    idx = np.searchsorted(binb, target_mass, side="left") - 1
    idx = np.where(target_mass > binb[-1], nbins - 1, idx)
    idx = np.where(target_mass <= binb[0], 0, idx)
    return np.clip(idx, 0, nbins - 1)


def _needgive_repair(
    number: np.ndarray, mass_tot: np.ndarray, *extra_mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """`cal_needgive`'s WITHIN-GROUP repair branch, G4 SS6 verbatim
    (`class_Group.F90:9625-9817`, the `total_neg>0 and total_pos>0 and
    m_t1>=min_mt` branch): the ONLY branch reachable when repairing a
    SINGLE group against itself, as rain-rain coalescence does (`g_1` and
    `g_2` are literally the same array in the Fortran call -- the
    cross-group "borrow from `g_2`"/"borrow from vapor" branches, which
    only fire when the group's OWN total mass tendency `m_t1` has gone
    negative, i.e. `total_neg>total_pos`, never apply here and are left
    unrepaired -- flagged, not silently invented, see this module's
    docstring/the task report).

    Per column (axis 1 here; the Fortran's own per-`ngrid` scoping):
    negative-`mass_tot` bins are ZEROED and the deficit is subtracted
    PROPORTIONALLY from every originally-POSITIVE bin
    (`modrg=(total_pos-total_neg)/total_pos`) -- exactly mass-conserving
    (the Fortran's own SEQUENTIAL per-negative-bin processing is
    algebraically equivalent to this single-pass formula using the
    ORIGINAL totals, derived in the task report). The SAME per-bin ratio
    (`modrg(i)=new_mass(i)/old_mass(i)`, i.e. `0` for a just-zeroed bin,
    `scale` for an original positive bin, `1` for an untouched
    point/column) is applied to `number` and every `extra_mass` leg,
    matching the Fortran's own `mass(1+k)=mass(1+k)*modrg(i)` treatment of
    the other mass components (`:9836-9838`) -- this port's OWN,
    documented substitute for `check_con`'s own un-quoted body (G4/H1 do
    not quote it) is to scale `number` by the identical ratio, the
    physically consistent choice (preserves each repaired bin's own
    mean-mass ratio) given no further ground truth is available; see the
    task report's fact-gap note.

    Args:
        number: `(nbins, npoints)`.
        mass_tot: `(nbins, npoints)`, the repair TRIGGER (matches the
            Fortran's own single `mass(1)`-driven trigger).
        *extra_mass: any number of additional `(nbins, npoints)` legs
            (e.g. aerosol mass components) to scale identically.

    Returns:
        `(new_number, new_mass_tot, new_extra_mass)` -- `new_extra_mass`
        is a tuple in the same order as `*extra_mass`. Columns/points with
        no negative bin are returned bit-for-bit unchanged (`modrg=1`).
    """
    total_pos = np.sum(np.where(mass_tot > 0.0, mass_tot, 0.0), axis=0, keepdims=True)
    total_neg = np.sum(np.where(mass_tot < 0.0, -mass_tot, 0.0), axis=0, keepdims=True)
    can_repair = (total_pos > 0.0) & (total_neg > 0.0) & (total_pos >= total_neg)

    safe_total_pos = np.where(total_pos > 0.0, total_pos, 1.0)
    scale = np.broadcast_to((total_pos - total_neg) / safe_total_pos, mass_tot.shape)
    can_repair_b = np.broadcast_to(can_repair, mass_tot.shape)

    modrg = np.where(
        can_repair_b,
        np.where(mass_tot < 0.0, 0.0, np.where(mass_tot > 0.0, scale, 1.0)),
        1.0,
    )

    new_mass_tot = mass_tot * modrg
    new_number = number * modrg
    new_extra = tuple(e * modrg for e in extra_mass)
    return new_number, new_mass_tot, new_extra


def coalesce_rain(  # noqa: PLR0915, PLR0917 -- single verbatim-derived engine, many H1/G4 sub-steps
    liquid_pv: LiquidState,
    diag: LiquidDiag,
    thermo: ThermoState,
    config: AmpsConfig,
    dt: float,
    luts: AmpsLuts,
    *,
    ibreak: bool = False,
) -> LiquidState:
    """The rain-rain (`token 1-1`) bin-pair collection engine --
    `collector_loop1`'s simplified fallback, per this module's own
    docstring (read it first: the "2 parents in, 1 merged child out"
    model, PER-VOLUME contract, `icolbin_min` loop bound, self-collection
    proof, and `ibreak` hook are all documented there, not repeated here).

    Signature deviation from the dispatch's literal
    `coalesce_rain(liquid_pv, diag, config, dt, luts)` text (same
    necessary-addition pattern M2b Task 2 already established for
    `collision_kernel`'s own `luts`/`col_level` additions): `thermo:
    ThermoState` was added -- `coalescence_efficiency` (Task 2,
    `core/collision_kernel.py`) needs the column temperature for its own
    `th_var%sig_wa` surface-tension term, and there is no way to compute
    `E_coal` (load-bearing for `n_ij`, the used_N_2/used_M_2 depletion
    accounting, G4 SS1.2) without it.

    Args:
        liquid_pv: PER-VOLUME `LiquidState` (`con`/`mass` = `q*den`,
            NOT mixing ratios -- see module docstring). `ncat` must be 1.
        diag: `LiquidDiag` computed from (a state consistent with)
            `liquid_pv` -- `mean_mass`/`length`/`terminal_velocity`/`nre`
            are RATIO-invariant to the per-volume/per-mass convention
            (both numerator and denominator carry the SAME `den` factor,
            which cancels), so `diag` needs no separate conversion.
        thermo: column `ThermoState` (temperature only, see above).
        config: `AmpsConfig` -- supplies `coll_level` (Task 2's
            `collision_kernel` branch selector) and `num_h_bins[0]`/
            `nbin_h` (this module's own necessary addition: the bin-mass
            boundary grid `binb`, needed for the destination-bin search,
            is built internally via `bin_grid.make_bin_grid("liquid",
            config.num_h_bins[0], nbin_h=config.nbin_h)` -- `bin_grid` is
            not in the dispatch's own Reuse list but IS already used by
            `core/vapor_deposition.py` for the identical purpose, same
            justification).
        dt: collision-substep timestep (s), scalar.
        luts: `AmpsLuts` (Task 2's `collision_efficiency`'s own `drpdrp`
            LUT requirement).
        ibreak: collisional-breakup flag -- ONLY `False` (the default) is
            implemented; `True` raises `NotImplementedError` (M2b Task 5
            scope, see module docstring).

    Returns:
        A NEW, PER-VOLUME `LiquidState` (same shape as `liquid_pv`;
        `liquid_pv` itself is never mutated).

    Raises:
        NotImplementedError: `ibreak=True`, or `liquid_pv.ncat != 1`.
    """
    if ibreak:
        raise NotImplementedError(
            "coalesce_rain: ibreak=True (collisional breakup, add_fragments_col_vec, "
            "H1 SS1.6j/G4 SS2.3) is M2b Task 5 scope -- this engine only implements the "
            "ibreak=0 (no-breakup) path. See core/coalescence.py's module docstring."
        )
    if liquid_pv.ncat != 1:
        raise NotImplementedError(
            f"coalesce_rain only supports ncat=1 (see state.py's own to_fields()/"
            f"from_fields() convention); got ncat={liquid_pv.ncat}"
        )

    lp = LiquidPPV
    nbins = liquid_pv.nbins
    npoints = liquid_pv.npoints

    con = liquid_pv.values[lp.rcon_q.py_idx, :, 0, :]
    mass_tot = liquid_pv.values[lp.rmt_q.py_idx, :, 0, :]
    mass_aero_tot = liquid_pv.values[lp.rmat_q.py_idx, :, 0, :]
    mass_aero_sol = liquid_pv.values[lp.rmas_q.py_idx, :, 0, :]

    # icond1 (H1 SS0 quoting G4 SS2, `mod_amps_core.F90:15894-15907`):
    # active where BOTH con and mass(1) clear the floor, for BOTH bins of
    # the pair. Applied around every Task 2 efficiency/kernel call output
    # below, per this task's own dispatch instruction.
    active = (con > _ACTIVE_FLOOR) & (mass_tot > _ACTIVE_FLOOR)
    active_i, active_j = _pairwise(active, active)
    icond1_active = active_i & active_j

    if not icond1_active.any():
        # No active pair anywhere -- identity, matching
        # `vapor_deposition_liquid`'s own no-active-bin early-return
        # precedent (also sidesteps any degenerate-input arithmetic
        # below for a genuinely inert state).
        return LiquidState(values=liquid_pv.values.copy())

    binb = bin_grid.make_bin_grid("liquid", config.num_h_bins[0], nbin_h=config.nbin_h).binb
    if binb.shape[0] - 1 != nbins:
        raise ValueError(
            f"coalesce_rain: liquid_pv.nbins={nbins} does not match the bin grid built from "
            f"config (config.num_h_bins[0]={config.num_h_bins[0]}) -- liquid_pv must be on "
            f"config's own liquid bin grid."
        )

    # --- Task 2 kernel / efficiency, icond1-masked (G4 SS2's own
    # init-then-conditionally-overwrite pattern: KC/E_c/E_coal default to
    # 0 for inactive pairs, matching `:15894-15907`'s own `KC(i,j)=0.0;
    # E_c(i,j)=0.0; E_coal(i,j)=0.0` pre-loop init). ---
    # `errstate` suppresses benign 0/0 or x/0 RuntimeWarnings from
    # `collision_efficiency`'s own `rrat=len_j/len_i` computation
    # (`core/collision_kernel.py:226-232`) across the MANY inactive
    # (zero-length) bin pairs a realistic 40-bin grid contains -- Task 2's
    # own tests never exercised the full bin-pair grid at once (single-bin
    # fixtures only), so this call pattern (a full nbins x nbins pass,
    # most pairs inactive) is new here; the results are masked to 0 via
    # `icond1_active` immediately below regardless, matching Task 2's own
    # documented degenerate-path handling.
    with np.errstate(divide="ignore", invalid="ignore"):
        kc = ck.collision_kernel(diag, diag, con, dt, luts, col_level=config.coll_level)
        e_coal, *_rest = ck.coalescence_efficiency(diag, diag, thermo)

    kc = np.where(icond1_active, kc, 0.0)
    e_coal = np.where(icond1_active, e_coal, 0.0)

    # --- N_col, G4 SS1.2 verbatim (`mod_amps_core.F90:1634-1664`): only
    # the KC>0 branch applies for rain-rain (token_i==token_j, so the
    # abs(KC) riming branch, `:1668` guarded by `token /= token`, never
    # fires); clamp N_col<=con_j. ---
    con_i, con_j = _pairwise(con, con)
    n_col = np.where(kc > 0.0, con_i * kc, 0.0)
    n_col = np.minimum(n_col, con_j)
    n_col = np.where(icond1_active, n_col, 0.0)

    # --- icolbin_min collector floor, H1 SS1.6 verbatim (loop bound
    # applied as a mask over axis 0/"i", the collector role only -- see
    # module docstring). ---
    icolbin_min = _icolbin_min(binb)
    collector_mask = np.zeros(nbins, dtype=bool)
    collector_mask[icolbin_min:] = True
    n_col = n_col * collector_mask[:, None, None]

    n_ij = n_col * e_coal  # coalescence EVENTS per (i, j, point)

    # This module's OWN necessary addition, beyond G4's own literal
    # N_col<=con_j clamp (H1 SS1.2 only clamps against the COLLECTEE's
    # population): the "2 parents in, 1 child out" per-event model (see
    # module docstring) requires ONE i-drop per event just as much as one
    # j-drop, so a SINGLE pair's event count cannot exceed EITHER parent
    # bin's own population -- unlike the true Fortran, which allows one
    # collector particle to absorb MANY collectee particles per particle
    # (`ndrop_B>1` in the "H" pass, G4 SS1.6), a case this simplified
    # model does not represent (see module docstring's own account of
    # what collector_loop1's rate categorization is NOT reproduced by
    # this fallback). Without this, a single (i,j) pair could deplete
    # bin i below zero even before any OTHER pair is considered -- caught
    # in code review via TestPropertyTransfer's own conservation check
    # (n_ij(120.4) > con_i(100) for a real drpdrp-LUT-driven pair).
    # Residual over-subscription from SUMMING multiple pairs against the
    # SAME bin (the con_j clamp above is also only a per-pair bound) is
    # still handled by `_needgive_repair` below, per the dispatch's own
    # "cal_needgive inter-bin borrowing" requirement.
    n_ij = np.minimum(n_ij, con_i)

    # --- coalesced mass m_i+m_j (H1's own Deliverable-bullet phrasing)
    # and destination-bin search (add_simple_vec, G4 SS4.1). ---
    mean_mass = diag.mean_mass
    mean_mass_i, mean_mass_j = _pairwise(mean_mass, mean_mass)
    merged_mass = mean_mass_i + mean_mass_j
    dest = _find_destination_bin(binb, merged_mass)

    mp_ij = n_ij * merged_mass  # mass carried by the merge events, RAW (unclamped)

    # add_simple_vec's own Np boundary-safety clamp (G4 SS4.1 verbatim,
    # `mod_amps_core.F90:15505-15528`): Np=max(Mp/0.99/binb(j+1),
    # min(Mp/1.01/binb(j),Np)) -- applied ONLY to the scattered NUMBER,
    # never to Mp itself (mass is always added raw, see module docstring
    # for why this makes mass conservation exact regardless).
    binb_lo = binb[dest]
    binb_hi = binb[dest + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        term_hi = np.where(
            binb_hi > 0.0, mp_ij / (0.99 * np.where(binb_hi > 0.0, binb_hi, 1.0)), 0.0
        )
        term_lo_cap = np.where(
            binb_lo > 0.0, mp_ij / (1.01 * np.where(binb_lo > 0.0, binb_lo, 1.0)), n_ij
        )
    n_ij_scattered = np.maximum(term_hi, np.minimum(term_lo_cap, n_ij))

    # --- aerosol per-drop shares (cal_ratio_mass_col_vec's simplified,
    # per-pair analogue, H1 SS1.6f/G4 SS5): each merge event's child
    # inherits BOTH parents' aerosol mass, additively. RAW n_ij (matching
    # mp_ij's own raw-value precedent, mass-like quantities), so aerosol
    # mass conservation is exact by the identical telescoping argument. ---
    aero_tot_per_drop = _safe_div(mass_aero_tot, con)
    aero_sol_per_drop = _safe_div(mass_aero_sol, con)
    aero_tot_i, aero_tot_j = _pairwise(aero_tot_per_drop, aero_tot_per_drop)
    aero_sol_i, aero_sol_j = _pairwise(aero_sol_per_drop, aero_sol_per_drop)
    aero_tot_gain_pair = n_ij * (aero_tot_i + aero_tot_j)
    aero_sol_gain_pair = n_ij * (aero_sol_i + aero_sol_j)

    # --- Scatter: accumulate every (i, j) pair's contribution into its
    # destination bin, per column point (np.add.at handles duplicate
    # (dest, point) targets correctly -- multiple pairs landing in the
    # SAME destination bin at the SAME point are summed, not overwritten).
    # ---
    point_idx = np.broadcast_to(np.arange(npoints)[None, None, :], dest.shape)
    gain_number = np.zeros((nbins, npoints))
    gain_mass_tot = np.zeros((nbins, npoints))
    gain_mass_aero_tot = np.zeros((nbins, npoints))
    gain_mass_aero_sol = np.zeros((nbins, npoints))
    np.add.at(gain_number, (dest, point_idx), n_ij_scattered)
    np.add.at(gain_mass_tot, (dest, point_idx), mp_ij)
    np.add.at(gain_mass_aero_tot, (dest, point_idx), aero_tot_gain_pair)
    np.add.at(gain_mass_aero_sol, (dest, point_idx), aero_sol_gain_pair)

    # --- Depletion: bin k loses n_ij as COLLECTOR (axis 0, sum over j)
    # PLUS as COLLECTEE (axis 1, sum over i) -- RAW n_ij, exactly
    # proportional to mean_mass[k]/aero_*_per_drop[k] (uniform per bin
    # regardless of role), which is what makes mass conservation exact
    # (see module docstring's telescoping-sum derivation). ---
    loss_number = n_ij.sum(axis=1) + n_ij.sum(axis=0)
    loss_mass_tot = loss_number * mean_mass
    loss_mass_aero_tot = loss_number * aero_tot_per_drop
    loss_mass_aero_sol = loss_number * aero_sol_per_drop

    new_con = con - loss_number + gain_number
    new_mass_tot = mass_tot - loss_mass_tot + gain_mass_tot
    new_mass_aero_tot = mass_aero_tot - loss_mass_aero_tot + gain_mass_aero_tot
    new_mass_aero_sol = mass_aero_sol - loss_mass_aero_sol + gain_mass_aero_sol

    # --- cal_needgive inter-bin borrowing (G4 SS6), within-group repair
    # for any bin over-depleted by summing multiple pairs (this module's
    # per-pair N_col<=con_j clamp bounds each INDIVIDUAL pair, not the
    # SUM over every collector `i` depleting the SAME bin `j` -- see
    # `_needgive_repair`'s own docstring). ---
    new_con, new_mass_tot, (new_mass_aero_tot, new_mass_aero_sol) = _needgive_repair(
        new_con, new_mass_tot, new_mass_aero_tot, new_mass_aero_sol
    )

    out_values = liquid_pv.values.copy()
    out_values[lp.rcon_q.py_idx, :, 0, :] = new_con
    out_values[lp.rmt_q.py_idx, :, 0, :] = new_mass_tot
    out_values[lp.rmat_q.py_idx, :, 0, :] = new_mass_aero_tot
    out_values[lp.rmas_q.py_idx, :, 0, :] = new_mass_aero_sol
    return LiquidState(values=out_values)
