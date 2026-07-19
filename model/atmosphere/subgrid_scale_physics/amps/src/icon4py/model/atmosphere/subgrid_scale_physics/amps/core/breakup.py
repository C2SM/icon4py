# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-List (1982) collisional-breakup RUNTIME (`ibreak==1`), transcribed
VERBATIM (structurally) from AMPS Fortran (scale_amps repo) per
docs/superpowers/facts/m2b/collisional-breakup.md ("H2" below). M2b Task 5.

H2's own architecture summary: `micexfg(18)` is the KID collisional-breakup
switch; when it is `1`, the fragment lookup tables `bu_fd`/`bu_tmass` are
precomputed ONCE by `cal_breakup_dis_LL` (M2b Task 6, `core/breakfragment.py`
-- a TABLE BUILDER, already committed, reused here unchanged) and at RUNTIME
the collision routine (`core/coalescence.py::coalesce_rain`, M2b Task 3) sets
the internal `ibreak` flag which gates the pieces this module ports:

* `add_fragments_col_vec` (H2 SS3a, `mod_amps_core.F90:15659-15774`) -- the
  fragment-table CONSUMER: for a collector bin `i` and every collectee `j<i`
  with a tabulated fragment entry, injects `bu_fd(2,kk)`/`bu_fd(1,kk)`
  (fragment number/mass) scaled by `N_bk=max(0,1-E_coal)*N_col` (the
  breakup-fraction of THIS pair's collisions) and `mod_rat=(mean_mass_i
  +mean_mass_j)/bu_tmass(i1d_pair)` (a runtime-vs-table-build-time
  mean-mass correction) into the collector's OWN destination-bin
  accumulators. `add_fragments_col_vec` below.
* The `i1d_pair`/`kk` packed-triangular index math (H2 SS4, appears
  IDENTICALLY at all 3 runtime/build sites) -- `dense_fragment_tables`
  below re-expands `BreakupFragmentTables`' packed `bu_tmass`/`bu_fd` into
  dense `(nbins,nbins[,nbins])` arrays ONCE (independent of any runtime
  liquid state), avoiding re-deriving the packed index on every
  `coalesce_rain` call.
* `used_N_b` (H2 SS1c, `mod_amps_core.F90:1668-1707`) -- the ADDITIONAL
  depletion `ibreak==1` applies to EVERY bin (both COLLECTOR and COLLECTEE
  roles) from the `(1-E_coal)` breakup fraction of `N_col`, on TOP of the
  ordinary coalescence-only collectee depletion (`N_col*E_coal`) `ibreak==0`
  already accounts for. `breakup_number_consumed` below.
* `P_breakup`/`Q_breakup2` (H2 SS3b/SS3c) -- Komabayashi (1964)/Srivastava
  (1971), the SPONTANEOUS/large-drop breakup mechanism. H2's own closing
  note: this is "a SEPARATE mechanism from the Low-List COLLISIONAL breakup
  tables" -- ported here as standalone, independently-tested functions per
  the task's deliverable list, but NOT wired into `coalesce_rain`'s `ibreak`
  hook (that hook is the Low-List COLLISIONAL mechanism only; no fact
  document or dispatch instruction ties `micexfg(18)`/`ibreak` to the
  spontaneous mechanism at all -- H2 itself flags them as unrelated).

===========================================================================
RECONCILING T3's unconditional release with REAL `ibreak==1` (dispatch's
own "CRITICAL context" instruction) -- read directly (dispatch-authorized)
from `mod_amps_core.F90:1712-1890` (the `iter_loop1`/"Fix over-depletion"
block), not merely H2's own already-quoted excerpt:
===========================================================================

`core/coalescence.py`'s own `coalesce_rain` (M2b Task 3) already releases a
FULLY-DEPLETED collector's own OUTGOING claims UNCONDITIONALLY (`n_col =
np.where(used_marker[:, None, :], 0.0, n_col)`, every fixed-point round),
documented there as "a deliberate, physically-motivated correction... Read
as literally as possible, the real Fortran's own `ibreak=0` path may
therefore leak mass in this same class of configuration" -- because the
LITERAL Fortran only performs an analogous release
(`N_col(i,j,n)=N_col(i,j,n)*min(1,mod_ratio(n))` for `j/=i`) INSIDE
`if(ibreak==1)` (`:1748-1762`/`:1836-1850`), gated on the SAME per-collectee
`i` currently being fixed for over-depletion in `iter_loop1`'s own
`do i=1,g_2%N_BIN` sweep.

Reading that gated block in full (not just H2's own excerpt) shows it does
TWO things, both `ibreak==1`-only:

1. Scale bin `i`'s (the collectee just found over-depleted) OWN OUTGOING
   claims `N_col(i,j,n)` for `j/=i` by the SAME `mod_ratio` that just
   rescaled its INCOMING claims -- i.e. "release a depleted bin's own
   outgoing claims" IS the literal Fortran's `ibreak==1` mechanism, GRADED
   (a proportional `*mod_ratio` scale), not the pure hard-zero-once-latched
   invention T3's own `ibreak=0` release uses.
2. RECOMPUTE `used_N_b` (the SAME formula as SS1c, `N_bk=N_col*max(0,
   1-E_coal)`) over the FULL bin grid using the JUST-RESCALED `N_col`, and
   fold bin `i`'s own updated total into `used_M_2(i)`/`used_N_2(i)` --
   i.e. `ibreak==1` makes `used_N_b` a LIVE quantity, recomputed as `N_col`
   itself evolves during over-depletion fixing, not a one-shot initial
   tally.

**A genuine implementation subtlety this reconciliation surfaced (found by
empirical testing, not by static reading alone)**: naively applying (1)'s
GRADED row-rescale to the SAME `n_col` array T3's OWN hard-zero release
already uses, in the SAME round, either (a) oscillates -- a round that
correctly grades a row down via (1) is followed by a round whose
(unconditional) hard-zero release, triggered because the graded reduction
landed EXACTLY at the `left_n<=1e-30` boundary, destroys that already-
correct value, `used_marker` then flips back `False` since the now-zeroed
row no longer over-claims, and the fixed point converges on a WRONG,
too-small `n_col` (verified: this manifested as `ibreak=1`'s own OUTPUT
being numerically IDENTICAL to the pre-collision input -- a silent,
complete loss of the breakup interaction); or (b), if the hard-zero release
is dropped entirely in favor of the graded rescale, ordinary-coalescence
GROWTH claims survive on a row whose collector is ALREADY fully spoken for
by breakup ALONE, so that "phantom" growth-claim's mass gets subtracted
from the collectee (reducing its `left_m`) but is NEVER scattered anywhere
(the collector doesn't run `_collector_scatter`, `used_marker=True`) --
verified: a measurable (~1.5%) mass leak in a realistic 2-bin fixture. THE
ROOT CAUSE: ordinary coalescence does NOT deplete a COLLECTOR's own
population (only the COLLECTEE's) -- only BREAKUP does, via BOTH roles
(SS1c's own two-loop `used_N_b` accumulation) -- so a SINGLE `n_col` view
cannot correctly serve both purposes once a collector's OWN breakup-role
consumption alone exhausts its population.

**Resolution, implemented in `core/coalescence.py`'s own fixed-point loop**:
maintain TWO SEPARATE views of the collision grid under `ibreak=1`, sharing
the SAME initial values but evolving independently:

* `n_col` ("growth" view): T3's own hard-zero release, UNCHANGED, both
  `ibreak` values -- once a collector `i` latches `used_marker` (for ANY
  reason, ordinary-coalescence-as-collectee OR breakup-as-either-role), its
  ENTIRE row is zeroed, every subsequent round, since `i` will never run
  `_collector_scatter` and any "growth" mass attributed to it would
  otherwise vanish. Passed to `_collector_scatter` (growth pass) --
  IDENTICAL role to before this task, `ibreak=0` behavior is completely
  unaffected.
* `n_col_bk` ("breakup" view, `ibreak=1` ONLY): the literal Fortran's own
  GRADED row+column rescale (both axes, same `mod_ratio`/`rescale` factor,
  every round) -- NEVER hard-zeroed, so a used_marker'd collector's own
  breakup interactions still fire at the CORRECT (population-capped)
  level. Passed to `breakup_number_consumed`/`add_fragments_col_vec`.

**Verdict: under `ibreak=1`, this port's mechanism now ALIGNS WITH (same
physical intent -- release/cap a depleted bin's own outgoing claims, track
breakup consumption to a fixed point) but is NOT BIT-IDENTICAL TO the
literal Fortran's own `ibreak==1` path** (a single shared array with a
graded rescale there, vs. two separate views here -- necessitated by this
port's own coarser, vectorized-per-outer-round approximation of the
Fortran's finer-grained nested `do ii=1,iter` per-collectee loop, see T3's
own module docstring). Given T3's own explicit priority ("GUARANTEED
conservation... over bit-for-bit fidelity to a control-flow gap") and this
task's own binding test requirement ("mass conserved... to 1e-12"), this
port keeps that priority for `ibreak=1` too, VERIFIED (not merely argued)
via `TestCoalesceRainIbreak` (`tests/.../test_breakup.py`): mass and
aerosol mass conserved to machine precision (~1e-16) across every tested
fixture/`dt`, both the failure modes above are covered by regression tests
(`test_dominant_breakup_pair_conserves_mass_and_increases_number`).

===========================================================================
Conservation-by-construction (why `used_N_b` is masked to
`dense_fragment_tables`' own `valid` domain, not the full `E_coal(i,j)=1.0`
gate H2 SS4 quotes for `cal_collision_kernel_func`):
===========================================================================

H2 SS4 quotes a SEPARATE `ibreak==1` gate inside `cal_collision_kernel_func`
(`mod_amps_core.F90:16341-16359`, read directly, NOT reproduced verbatim
here for a documented reason): for pairs `(i,j)` inside
`[imin_bk,imax_bk]x[jmin_bk,jmax_bk]` but with `bu_tmass(i1d_pair)<=1e-30`
(no tabulated fragment entry -- table-build skips, e.g. `D_coal<=D_0` or
`CKE<=1e-20`, `core/breakfragment.py`), the Fortran forces `E_coal(i,j)=1.0`
(pure coalescence, no breakup) so `used_N_b` never gets a nonzero
contribution from an untabled pair. Reading the loop bounds literally
(`do j=1,g_2%N_BIN; do i=1,g_1%N_BIN`, no `i>j` restriction in the loop
itself, gated only by `icond1(i,j)==0`, i.e. ACTIVE -- `cal_collision_kernel_
func`'s OWN icond1 convention is INVERTED from this port's `icond1_active`,
confirmed by reading `:15818-15907` directly, `icond1(i,j)=0` when BOTH
`con`/`mass(1)` clear the floor), the `i1d_pair` formula it evaluates is
only well-founded for `i>j` (the table's own triangular construction) --
for `j>=i` inside that same bin range the SAME formula would read an
AVAILABLE-but-WRONG (aliased to a different valid pair) `bu_tmass` slot,
not a "no entry" sentinel. Rather than replicate that literal indexing
(whose own correctness for `j>=i` this port's own re-reading could not
resolve with confidence, and which never fires for cloudlab's own 40-bin
table regardless -- M2b Task 6's report: "136/136 pairs have nonzero
bu_tmass", i.e. every in-domain pair IS tabulated), this port takes a
STRICTLY SAFER, conservation-guaranteed-by-construction route: both
`breakup_number_consumed` (this module) and `add_fragments_col_vec`'s own
consultation are masked to `dense_fragment_tables().valid` -- `True` ONLY
for `i>j` pairs inside `[imin_bk,imax_bk]x[jmin_bk,jmax_bk]` WITH
`bu_tmass>1e-30`. This means: (a) breakup consumption is NEVER tallied for
a pair lacking a matching fragment re-injection (mass cannot leak by
construction, matching this port's own established priority), and (b) for
cloudlab specifically (the only real config exercised), this is
OBSERVATIONALLY IDENTICAL to the literal Fortran's own gate (since every
in-domain pair is tabulated there, the "force E_coal=1" branch never fires
either way). The shared `E_coal(i,j)` array itself (used by
`_collector_scatter`'s own `col_ratio`, the ordinary growth pass) is
DELIBERATELY left UNMODIFIED by this port (unlike the literal Fortran,
which mutates `E_coal` in place) -- a further, documented simplification:
entangling this task's addition with T3's own already-hardened growth-pass
fixed point was judged higher regression risk than the (empirically
inconsequential, for cloudlab) fidelity gap it would close. Flagged, not
silently decided -- see the task report's fact-gaps section.

Units: CGS, float64, per-volume throughout (matches `core/coalescence.py`'s
own contract -- no `den` factor anywhere in this module).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    BreakupFragmentTables,
)


_PI = float(AmpsConst.PI)


# ---------------------------------------------------------------------------
# i1d_pair/kk dense re-expansion (H2 SS4) -- built ONCE per
# (BreakupFragmentTables, nbins) pair, independent of any runtime liquid
# state, reused across every `coalesce_rain` call/timestep.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DenseFragmentTables:
    """Dense (0-based) re-expansion of a `BreakupFragmentTables`' packed
    `bu_tmass`/`bu_fd` arrays -- see `dense_fragment_tables` below.

    `tmass`: `(nbins, nbins)`, `tmass[i, j]` = `bu_tmass` for the (0-based)
    COLLECTOR bin `i` / COLLECTEE bin `j` pair; nonzero ONLY where `valid`.
    `valid`: `(nbins, nbins)` bool -- `True` exactly where `i>j`, `(i,j)`
    falls inside `[imin_bk,imax_bk]x[jmin_bk,jmax_bk]` (0-based, inclusive),
    AND `bu_tmass>1e-30` (H2 SS1d's `bu_tmass(i1d_pair)<=1.0e-30_PS` skip,
    module docstring's "conservation-by-construction" section). `frag_mass`/
    `frag_con`: `(nbins, nbins, nbins)`, `[i, j, k]` = `bu_fd(1, kk)`/
    `bu_fd(2, kk)` (fragment MASS/NUMBER landing in bin `k` from the `i>j`
    pair's own breakup); zero where `~valid[i, j]`.
    """

    tmass: np.ndarray
    valid: np.ndarray
    frag_mass: np.ndarray
    frag_con: np.ndarray
    jmin_bk: int  # 0-based
    imin_bk: int  # 0-based
    imax_bk: int  # 0-based, inclusive
    jmax_bk: int  # 0-based, inclusive


def dense_fragment_tables(tables: BreakupFragmentTables, nbins: int) -> DenseFragmentTables:
    """Re-expand `tables.bu_tmass`/`tables.bu_fd` (packed 1D/2D, `i1d_pair`/
    `kk`-indexed, H2 SS2m/SS4) into dense `(nbins,nbins[,nbins])` arrays.
    Vectorized (no Python-level `(i,j)` loop): builds the FULL `i1d_pair`
    formula (H2 SS4, identical at every build/runtime site)

        i1d_pair = (j - jmin_bk + 1) + (i - imin_bk)*(1 + i - imin_bk)//2

    over the whole `(nbins, nbins)` 1-based-index grid at once, masked to
    the table's own triangular domain (`imin_bk<=i<=imax_bk`,
    `jmin_bk<=j<=i-1` -- the `j<i` condition alone already implies
    `j<=jmax_bk=nbins-1` whenever `i<=imax_bk=nbins`, so no separate
    `j<=jmax_bk` check is needed), then gathers `bu_fd[:, kk0:kk0+nbins]`
    per valid pair via fancy indexing.
    """
    jmin_bk, imin_bk, imax_bk, jmax_bk = (
        tables.jmin_bk,
        tables.imin_bk,
        tables.imax_bk,
        tables.jmax_bk,
    )
    idx = np.arange(1, nbins + 1)  # 1-based Fortran bin index
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")  # i_grid varies along axis 0

    in_range = (i_grid >= imin_bk) & (i_grid <= imax_bk) & (j_grid >= jmin_bk) & (j_grid < i_grid)
    i1d_pair = np.where(
        in_range,
        (j_grid - jmin_bk + 1) + (i_grid - imin_bk) * (1 + i_grid - imin_bk) // 2,
        1,  # safe placeholder for masked-out entries; overwritten to 0.0 below
    )
    pair_idx0 = i1d_pair - 1  # 0-based into bu_tmass, always in-bounds by construction

    tmass = np.where(in_range, tables.bu_tmass[pair_idx0], 0.0)
    valid = in_range & (tmass > 1.0e-30)
    tmass = np.where(valid, tmass, 0.0)

    kk0 = np.where(valid, pair_idx0 * nbins, 0)
    kk = kk0[:, :, None] + np.arange(nbins)[None, None, :]
    frag_mass = np.where(valid[:, :, None], tables.bu_fd[0, kk], 0.0)
    frag_con = np.where(valid[:, :, None], tables.bu_fd[1, kk], 0.0)

    return DenseFragmentTables(
        tmass=tmass,
        valid=valid,
        frag_mass=frag_mass,
        frag_con=frag_con,
        jmin_bk=jmin_bk - 1,
        imin_bk=imin_bk - 1,
        imax_bk=imax_bk - 1,
        jmax_bk=jmax_bk - 1,
    )


# ---------------------------------------------------------------------------
# used_N_b (H2 SS1c) -- the ADDITIONAL breakup depletion, both COLLECTOR and
# COLLECTEE roles. See module docstring's "conservation-by-construction"
# section for why this is masked to `dense.valid`.
# ---------------------------------------------------------------------------


def breakup_number_consumed(
    n_col: np.ndarray, e_coal: np.ndarray, dense: DenseFragmentTables
) -> np.ndarray:
    """`used_N_b`, H2 SS1c verbatim (`mod_amps_core.F90:1668-1707`), MASKED
    to `dense.valid` (module docstring's "conservation-by-construction"
    note): only `(i,j)` pairs with an ACTUAL fragment-table entry
    contribute breakup consumption, so every unit of mass removed here has
    a matching `add_fragments_col_vec` re-injection.

    `n_col`/`e_coal`: `(nbins_i, nbins_j, npoints)` (collector axis 0,
    collectee axis 1, matching `core.coalescence`'s own `_pairwise`
    convention). Returns `used_n_b`, `(nbins, npoints)`: for bin `k`,
    `sum_j n_bk[k,j]` (k acting as COLLECTOR, SS1c's second loop) `+
    sum_i n_bk[i,k]` (k acting as COLLECTEE, SS1c's first loop),
    `n_bk[i,j]=n_col[i,j]*max(0,1-e_coal[i,j])`.
    """
    n_bk = n_col * np.maximum(0.0, 1.0 - e_coal)
    n_bk = np.where(dense.valid[:, :, None], n_bk, 0.0)
    return n_bk.sum(axis=1) + n_bk.sum(axis=0)


# ---------------------------------------------------------------------------
# add_fragments_col_vec (H2 SS3a) -- the runtime fragment-table CONSUMER.
# ---------------------------------------------------------------------------


def add_fragments_col_vec(  # noqa: PLR0917 [too-many-positional-arguments]
    i: int,
    n_col_i: np.ndarray,
    e_coal_i: np.ndarray,
    mean_mass: np.ndarray,
    mass_tot: np.ndarray,
    mass_aero_tot: np.ndarray,
    mass_aero_sol: np.ndarray,
    gate_i: np.ndarray,
    dense: DenseFragmentTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`add_fragments_col_vec`, H2 SS3a verbatim (`mod_amps_core.F90:
    15659-15774`), for a FIXED collector bin `i` (0-based) -- one call per
    `coalesce_rain` `collector_loop1` iteration, matching that Fortran's own
    per-`i` call site (`:2884-2893`, read directly: called for EVERY `i` in
    `collector_loop1`'s own range, gated internally by `used_marker(i,n)==1
    .or.icond1(i,n)==1` -- `gate_i` here is that same disjunction, the
    caller's responsibility per `core/coalescence.py`'s own wiring).

    Args:
        i: 0-based collector-bin index.
        n_col_i: `(nbins_j, npoints)`, row `i` of the full `N_col` grid
            (collector `i`, every collectee `j`) -- the FINAL, post-fixed-
            point value (same array `_collector_scatter` itself consumes).
        e_coal_i: `(nbins_j, npoints)`, row `i` of `E_coal`, same timing.
        mean_mass/mass_tot/mass_aero_tot/mass_aero_sol: `(nbins, npoints)`,
            EVERY bin's own per-column fields (collectee legs are read at
            arbitrary `j`, not just row `i`).
        gate_i: `(npoints,)` bool -- `used_marker[i] | icond1_i` (H2 SS3a's
            own `used_marker(i,n)==1.or.icond1(i,n)==1`).
        dense: precomputed `DenseFragmentTables` (this module).

    Returns:
        `(add_n, add_rmt, add_rmat, add_rmas)`, each `(nbins_k, npoints)` --
        matching `core.coalescence._collector_scatter`'s own first-four
        return values, ready to add into the caller's running
        `new_N_1`/`new_M_1(rmt/rmat/rmas)` accumulators. All-zero (not
        raising) if this `i` has no valid `j<i` fragment-table entry or
        `gate_i` is all-`False` -- mirrors the Fortran's own early
        `if(i<imin_bk.or.i>imax_bk) return` / `bu_tmass(i1d_pair)<=1e-30 ->
        cycle` no-op branches.
    """
    nbins, npoints = mean_mass.shape
    zero = np.zeros((nbins, npoints))

    valid_j = dense.valid[i, :]  # (nbins_j,) -- which collectee j<i is tabulated
    if not valid_j.any() or not gate_i.any():
        return zero, zero.copy(), zero.copy(), zero.copy()

    tmass_j = dense.tmass[i, :]  # (nbins_j,)
    safe_tmass_j = np.where(valid_j, tmass_j, 1.0)[:, None]

    n_bk = n_col_i * np.maximum(0.0, 1.0 - e_coal_i)  # (nbins_j, npoints)
    mod_rat = (mean_mass[i][None, :] + mean_mass) / safe_tmass_j  # (nbins_j, npoints)
    scaled = np.where(valid_j[:, None] & gate_i[None, :], n_bk * mod_rat, 0.0)  # (nbins_j, npoints)

    frag_mass_jk = dense.frag_mass[i]  # (nbins_j, nbins_k)
    frag_con_jk = dense.frag_con[i]

    denom = mass_tot[i][None, :] + mass_tot  # (nbins_j, npoints)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    ratio_rmat = np.where(
        denom > 0.0, (mass_aero_tot[i][None, :] + mass_aero_tot) / safe_denom, 0.0
    )
    ratio_rmas = np.where(
        denom > 0.0, (mass_aero_sol[i][None, :] + mass_aero_sol) / safe_denom, 0.0
    )

    add_n = np.einsum("jk,jn->kn", frag_con_jk, scaled)
    add_rmt = np.einsum("jk,jn->kn", frag_mass_jk, scaled)
    add_rmat = np.einsum("jk,jn->kn", frag_mass_jk, scaled * ratio_rmat)
    add_rmas = np.einsum("jk,jn->kn", frag_mass_jk, scaled * ratio_rmas)
    return add_n, add_rmt, add_rmat, add_rmas


# ---------------------------------------------------------------------------
# P_breakup / Q_breakup2 (H2 SS3b/SS3c) -- spontaneous large-drop breakup
# (Komabayashi 1964 / Srivastava 1971). NOT wired into `coalesce_rain`'s
# `ibreak` hook -- see module docstring's opening section.
# ---------------------------------------------------------------------------


def _fgm(p: float, x_min: float, x_max: float) -> float:
    """`fGM`, VERBATIM (`mod_amps_utility.F90:12734-12766`) -- ONLY the
    `p==4.0` branch, the sole case `q_breakup2`'s own `switch==3` reaches
    (its literal call is `fGM(4.0_PS, x1, x2)`); the `x_min=x_max=0`/
    general-integer-`p` branch is unreachable from `q_breakup2` and is not
    ported (matches `core/breakfragment.py`'s own "only the actually
    reachable path" scoping precedent, e.g. its `zbrent` `iwhich==3`
    omission)."""
    return (x_min**3.0 + 3.0 * x_min**2.0 + 6.0 * x_min + 6.0) * math.exp(-x_min) - (
        x_max**3.0 + 3.0 * x_max**2.0 + 6.0 * x_max + 6.0
    ) * math.exp(-x_max)


def p_breakup(phase: int, a_star: float, max_dim: float) -> float:
    """`P_breakup`, VERBATIM (H2 SS3b, `mod_amps_core.F90:19873-19904`) --
    Komabayashi et al. (1964) probability that a parent drop of radius
    `a_star` [mm] breaks up spontaneously within `dt`, given the
    population's own maximum radius `max_dim` [cm]. `phase`: 1=water,
    2=ice. NOT the Low-List COLLISIONAL mechanism this module's
    `add_fragments_col_vec`/`coalesce_rain` wiring implements -- see module
    docstring.
    """
    if phase == 1:
        if a_star >= max_dim:
            return 1.0
        return min(2.94e-7 * math.exp(3.4 * a_star * 10.0), 1.0)
    if phase == 2:
        if a_star >= max_dim:
            return 1.0
        k = 15.03968607 / (max_dim * 10.0)
        return min(2.94e-7 * math.exp(k * a_star * 10.0), 1.0)
    raise ValueError(f"p_breakup: phase must be 1 (water) or 2 (ice); got {phase}")


def q_breakup2(  # noqa: PLR0917 [too-many-positional-arguments]
    phase: int, a_star: float, a: float, m: float, aa: float, bb: float, switch: int
) -> float:
    """`Q_breakup2`, VERBATIM (H2 SS3c, `mod_amps_core.F90:19953-19997`) --
    Srivastava (1971) drop number/mass distribution for the spontaneous
    large-drop breakup path. `switch`: 1=number density between `m` and
    `m+dm`, 2=total number between `a` and `m` (`#/cm^3`), 3=total mass
    between `a` and `m` (phase 1/water only; `fGM`-based, see `_fgm`). NOT
    wired into `coalesce_rain`'s `ibreak` hook -- see module docstring.
    """
    if phase == 1:
        if switch == 1:
            return (aa * bb / 3.0 / m) * (a / a_star) * math.exp(-bb * a / a_star)
        if switch == 2:
            return -aa * (math.exp(-bb * m / a_star) - math.exp(-bb * a / a_star))
        if switch == 3:
            x1 = bb * a / a_star
            x2 = bb * m / a_star
            return (4.0 * _PI / 3.0) * aa * (a_star / bb) ** 3.0 * _fgm(4.0, x1, x2)
        raise ValueError(f"q_breakup2: switch must be 1, 2, or 3 for phase=1; got {switch}")
    if phase == 2:
        if switch == 1:
            return (aa * bb / 3.0 / m) * (a / a_star) * math.exp(-bb * a / a_star)
        if switch == 2:
            return -aa * (math.exp(-bb * m / a_star) - math.exp(-bb * a / a_star))
        raise ValueError(f"q_breakup2: switch must be 1 or 2 for phase=2; got {switch}")
    raise ValueError(f"q_breakup2: phase must be 1 (water) or 2 (ice); got {phase}")
