# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`coalesce_rain` -- the rain-rain (`token 1-1`) bin-pair collection
engine, transcribed VERBATIM (structurally) from AMPS Fortran (scale_amps
repo) per docs/superpowers/facts/m2b/coalescence-engine.md ("H1" below,
the gap-fill supplement, SS0/SS1.6/SS5) and
docs/superpowers/facts/m2/coalescence.md ("G4" below, the base
extraction, SS1/SS4.1/SS6). M2b Task 3, REVISED per code review: this
module now ports the REAL `collector_loop1` (H/F/LM rate categorization
-> fortunate/unfortunate split -> Mc mass-component seed -> three
accumulation passes -> `cal_ratio_mass_col_vec` -> shifted-boundary
construction -> multi-bin PDF fit/scatter), NOT the single-target-bin
"2 parents in, 1 merged child out" fallback the first revision used
(that fallback is what H1's own "M2b codegen notes" section calls out as
`add_simple_vec` being merely "the reference for the simpler mean-mass
jbin search M2 MAY emit as a fallback" -- a citation the ORIGINAL
dispatch mis-cited as authorizing it as the PRIMARY engine; code review
corrected this, see the M2b Task 3 report's revision section).

`contrib/AMPS/mod_amps_core.F90` line ranges below were read DIRECTLY
(dispatch-authorized) beyond what H1/G4 quote, to resolve structural
ambiguities H1's own summary left underspecified (`n_f_min`, the exact
`used_marker`/`used_M_2` reset-and-reuse semantics, `pro_type` for
rain-rain, `assign_tendency_vec`'s `(nN-aNd)/dt` interpretation of
`new_N_1`) -- each cited at its own point of use below.

===========================================================================
THE ALGORITHM (per collector bin `i`, H1 SS1.6 loop order, descending
`i=g_1%N_bin` down to `icolbin_min` -- `_icolbin_min`'s own docstring):
===========================================================================

For collector bin `i` and EVERY collectee bin `j`, `col_ratio(j)
=E_coal(i,j)*N_col(i,j)/con_i` measures how many `j`-drops (on average)
EACH `i`-drop absorbs this timestep (G4 `:2028-2054`, verbatim):

  * `col_ratio>10`: "high" only.
  * `1<=col_ratio<=10`: BOTH "high" AND "low-medium" (dual membership --
    the deterministic INTEGER part `floor(col_ratio)` goes through the
    "H" pass, the stochastic FRACTIONAL remainder through "F" or "LM").
  * `0<col_ratio<1`: "low-medium" only.

**Fortunate/unfortunate split** (G4 `:2060-2096`, a SEQUENTIAL scan over
low-medium `j` in DESCENDING bin-index order, `_fortunate_scan` below):
each low-medium `j`'s fractional collision probability `p_F` is
accumulated into a running `sum_ndrop_B`; the FIRST `j` (processed in
descending order) whose acceptance would push `sum_ndrop_B>=1` is
EXCLUDED (latched -- every SMALLER `j` for that column is skipped
entirely, `icond2_n(n)==0` gate, `:2069`) instead of accepted as
"fortunate". Every ACCEPTED low-medium `j` gets its OWN DEDICATED
sub-population (`Np(jj,n)=p_F*left_N(i,n)`, i.e. that FRACTION of the
collector's OWN surviving population, `left_N`, is deemed to have
collided with EXACTLY one `j`-drop); the REMAINING
`(1-sum_ndrop_B)*left_N(i,n)` fraction is the "unfortunate" sub-population
(collided with nothing this timestep). **This is the reviewer's #1
requirement**: the collector bin's OWN `con` is NEVER destroyed by a
merge event the way the reverted single-bin model did -- `left_N(i,n)`
(the collector's surviving population) is exactly PARTITIONED across
these sub-populations (`sum_k Np_sub(k)==left_N(i,n)` by construction,
verified algebraically in `_collector_scatter`'s own docstring), each
KEEPING its own count while GROWING in mean mass, then scattered by mass
into (possibly several) NEW destination bins -- NOT destroyed 2-for-1.

**Three accumulation passes**, all adding to the sub-population `Mp`
seeded from `Np*mean_mass_i` (the collector's OWN pre-collision mass) --
G4 `:2213-2382`, `mod_amps_core.F90:2213-2382` read directly for the
LM-pass verbatim (H1 quotes H+F in full; the LM-pass differs from H1's
own summary only in confirming it independently, see `mod_amps_core.F90:
2335-2382`):

  * **H** ("high"): `ndrop_B=floor(col_ratio)` if `col_ratio` in `[1,10]`
    else `col_ratio` itself (`>10`); `dM_dum=ndrop_B*con_i*mean_mass_j`,
    distributed ACROSS EVERY sub-population `k` proportional to its own
    share `Np_sub(k)/left_N(i)` (`:2213-2260`) -- every collector
    particle, dedicated-fortunate or not, deterministically absorbs the
    SAME high-rate growth. **Sets `used_marker(j,n)=1`** when the
    RUNNING `used_M_2(j,n)` (see below) reaches `mass(j,1)` -- the ONLY
    one of the three passes that does (confirmed by direct comparison of
    all three pass bodies, `mod_amps_core.F90:2213-2382`: F `:2299-2302`
    and LM `:2358-2361` both clamp `used_M_2` but NEITHER sets
    `used_marker`).
  * **F** ("fortunate"): for each ACCEPTED low-medium `j`, `dM_dum
    =p_F*con_i*mean_mass_j` (H1's own two branches, `col_ratio<1` vs
    `col_ratio` in `[1,10]`, algebraically identical -- see
    `_collector_scatter`'s own derivation), added ONLY to `j`'s OWN
    DEDICATED sub-population (`:2275-2322`).
  * **LM** ("low-medium leftover"): the SAME formula as H (`ndrop_B=p_F`
    now), for low-medium `j`'s NOT accepted as fortunate, distributed
    across ALL sub-populations exactly like H (`:2335-2382`, verbatim
    read directly, see above).

`cal_ratio_mass_col_vec` (G4 SS5, `mod_amps_core.F90:17633-17687`): each
sub-population's aerosol-mass FRACTION after growth is `(Mc+dM_1)/Mp`,
`Mc` the SEED's own aerosol share (collector `i`'s OWN
`mass_aero_tot/mass_tot` ratio, applied to the SEED mass BEFORE any
pass runs -- Mc-seed code, `:2126-2154`, runs strictly before the H-pass)
and `dM_1` the collected aerosol mass, weighted by EACH CONTRIBUTING
COLLECTEE `j`'s OWN aerosol FRACTION (`ratio_M_2(j)=mass_aero_tot(j)/
mass_tot(j)`, a dimensionless ratio -- NOT a per-drop mass, unlike the
reverted model's own aerosol formula).

**Shifted-boundary construction + PDF reconstruction + multi-bin scatter**
(G4 `:2537-2758`) -- **the reviewer's #2/#3 requirement, REUSING the M2a
ports verbatim, not reimplementing them**: each sub-population's shifted
mass interval is `[binb(i)+dmass_min(k), binb(i+1)+dmass_max(k)]`
(`dmass_min`/`dmass_max` accumulate `ndrop*binb(j)`/`ndrop*binb(j+1)`
from every contributing collectee, mirroring the mass-growth
accounting exactly); `cal_lincubprms_vec` fits a number-density PDF over
that interval from `(Np_sub,Mp_sub)`, and `cal_transbin_vec` GATHERS the
overlap of that PDF against EVERY original destination bin -- genuinely
spreading one sub-population's mass across MULTIPLE bins when its
shifted interval crosses more than one boundary. This module imports
`core.vapor_deposition._linear_fit` (=`cal_lincubprms_vec`'s base +
truncated-support linear fit, M2a Task 5) and
`core.vapor_deposition._gather_remap` (=`cal_transbin_vec`'s multi-bin
gather, M2a Task 5) DIRECTLY -- an explicit, reviewer-directed exception
to the general "core/ modules do not cross-import each other's private
helpers" convention (`core/repair.py`'s own documented precedent): the
reviewer's own instruction was to REUSE these exact, already-tested
routines, not re-derive a second, subtly-different copy.

**One deliberate, documented reduction**: `_cubic_upgrade` (M2a's own
Dinh & Durran 2012 interior-cubic refinement) is NOT reused here --
that routine's own "neighbor" concept (`np.roll` on the SOURCE-bin axis)
assumes a sequential, size-ordered array of ADJACENT bins (true for
`vapor_deposition`'s own one-sub-interval-per-original-bin usage); this
module's sub-population axis (fortunate-per-`j` sub-bins, ordered by
COLLECTEE bin index, plus one "unfortunate" slot) has no such adjacency
relationship -- applying `_cubic_upgrade`'s neighbor logic here would
graft a physically ungrounded relationship onto bins that are not
actually adjacent size classes. Every sub-population's PDF is
reconstructed via `_linear_fit` alone (base 2-moment linear fit +
truncated-support re-fit on negative density -- STILL a genuine,
non-degenerate multi-bin-capable PDF, just not cubic-upgraded).
Sub-populations whose `_linear_fit` itself fails (`ok=False`, the
genuinely-rare tail `core/vapor_deposition.py`'s own docstring
documents) fall back to a SAME-BIN (nearest, via `_find_destination_bin`)
concentrated placement -- matching `vapor_deposition_liquid`'s own
documented precedent for that identical tail case, not a new invention.

===========================================================================
STRUCTURAL PIECES NOT PART OF H1's OWN QUOTED EXCERPT, RESOLVED BY READING
`mod_amps_core.F90`/`assign_tendency_vec` DIRECTLY:
===========================================================================

* **`pro_type` for rain-rain is `2`** (`mod_amps_core.F90:1495-1496`:
  `if(g_1%token==g_2%token) pro_type=2`) -- so `collector_loop1`'s OWN
  `icond1` redefinition (`:1955-1966`, distinct from
  `cal_collision_kernel_func`'s simpler `icond1`, see `_ACTIVE_FLOOR`
  below) genuinely applies its `used_marker(i,n)/=1 .or. pro_type/=2`
  clause as `used_marker(i,n)/=1` (the OR's second disjunct is always
  FALSE for rain-rain): a bin whose ENTIRE mass was already consumed as
  a COLLECTEE is EXCLUDED from ALSO acting as its own COLLECTOR.
* **T7 (M2b Task 7, warm-dump-validation fix): `collector_loop1`'s OWN
  `icond1` redefinition ALSO requires a COLUMN-LEVEL `counter(n)>0`**
  (`mod_amps_core.F90:1955-1958`, quoted in full at this function's own
  `counter_active` computation below) -- `counter(n)` (`:1899-1907`)
  counts, per column, how many bins the O(n^2) kernel pass actually
  claimed (`used_N_2(i,n)>0`); `counter(n)==0` means `collector_loop1`
  skips EVERY bin in that column, independent of any individual bin's own
  con/mass_tot/mean_mass. G4/H1's own quoted excerpts never mention this
  conjunct (an omission this task's own real-dump validation surfaced,
  not a pre-existing fact-gap flagged by an earlier task) -- see
  `counter_active`'s own inline comment, at its point of use, for the
  full derivation and the exact real-dump failure mode it fixes.
* **`used_M_2` (NOT `used_N_2`) is reset to zero immediately before
  `collector_loop1` starts** (`mod_amps_core.F90:1938`:
  `used_M_2(:,:) = 0.0_PS`) -- collector_loop1's OWN H/F/LM-pass
  accounting is a FRESH tally, independent of the earlier O(n^2)
  kernel-loop's own `used_M_2` accumulation (which only feeds `left_N`/
  `left_M` via `used_N_2`, itself NOT reset). `used_marker`'s STARTING
  state entering `collector_loop1` (before any fresh H/F/LM contribution)
  is seeded from the EARLIER `iter_loop1` over-depletion fix's own
  used_marker-setting (G4 SS1.4) -- this module's own simplified,
  non-iterative `left_N`/`left_M` (see PER-VOLUME/simplification note
  below) makes `left_N<=0 or left_M<=0` an exact proxy for that seed
  (iter_loop1 sets `used_marker=1` PRECISELY when its own clamp forces
  `used_N_2>=con` or `used_M_2>=mass(1)`, i.e. precisely when `left_N`/
  `left_M` would be zero).
* **`assign_tendency_vec` (`mod_amps_core.F90:20891-21486`, read
  directly) confirms `new_N_1`/`new_M_1` are the FULL new bin population,
  not a delta**: `dcondt=(nN(i,n)-g%MS(i,n)%con)/dt` (`:21005`). This
  module's own `coalesce_rain` therefore returns a REPLACEMENT
  `LiquidState`, matching the ORIGINAL (pre-review) module's own
  contract -- unchanged by this revision.
* **`icolbin_min`-excluded bins (too small to ever act as a collector)
  and the post-loop leftover re-add**: the post-loop re-add
  (`mod_amps_core.F90:2899-2925`, quoted FULL in H1 SS1.6k) only fires
  for `used_marker(i,n)==1` bins. A bin below `icolbin_min` NEVER runs
  its own `collector_loop1` iteration (so its OWN `left_N`/`left_M`
  never gets scattered via the "unfortunate" sub-population route
  either) -- if such a bin is ALSO never `used_marker`-flagged, H1/G4's
  own quoted excerpt does not show ANY mechanism re-adding its `left_N`/
  `left_M`. Rather than risk a silent mass-loss bug for an ambiguity
  H1/G4 do not resolve, this module treats EVERY bin below `icolbin_min`
  as ALSO eligible for the post-loop re-add (`used_marker(i,n) OR
  i<icolbin_min`) -- a documented, MASS-SAFE (never loses number/mass),
  physically reasonable choice (such bins are inert as collectors by
  construction; their surviving population landing back at their own
  bin is the physically expected outcome) for a genuinely underspecified
  corner, flagged here and in the task report rather than silently
  guessed.

===========================================================================
PER-VOLUME basis (H1 SS5) and SIMPLIFICATIONS RELATIVE TO THE FULL FORTRAN
(each flagged, not silent):
===========================================================================

* This module's INPUT `liquid_pv` and OUTPUT are BOTH per-VOLUME
  (`con`/`mass=q*den`), exactly as before this revision -- no `den`
  factor appears anywhere in this module.
* **`iter_loop1` (G4 SS1.4, the O(n^2)-phase over-depletion
  PRE-normalization) IS ported, PLUS an outer fixed-point iteration this
  port's own hardening found necessary for mass conservation with 4+
  active bins.** Full derivation at `coalesce_rain`'s own inline comment
  at the fixed-point loop; summary of the two distinct issues it closes:

  (1) The per-pair `N_col<=con_j` clamp bounds each INDIVIDUAL
      collector's own claim on a collectee, but not the SUM across every
      collector targeting the SAME collectee -- ported as a proportional
      rescale, provably an EXACT single-pass equivalent of the Fortran's
      own iterative `do ii=1,iter` alternation IN THIS FORMULATION
      (`mean_mass(j)=mass_tot(j)/con(j)` exactly, so the number-limit and
      mass-limit halves, `:1713-1799`/`:1801-1884`, are mathematically
      IDENTICAL conditions here -- unlike the general ice/mixed-token
      case the Fortran's own alternation is built for).
  (2) **A collector that itself ends up fully claimed by an even bigger
      collector (`used_marker=1`, excluded from ever running its own
      `collector_loop1` body) still has its ORIGINAL, un-diminished claim
      on smaller collectees baked into their `left_N`/`left_M` -- a
      "phantom" claim that never actually executes** (the excluded
      collector never scatters anything), silently vanishing mass with
      no negative bin for `_needgive_repair` to catch. Discovered during
      hardening: a 5-active-bin fixture (con 10-400 cm^-3, dt=2s)
      leaked 14.8% of total mass; a realistic-magnitude 3-bin fixture
      leaked nothing (needs >=4 active bins with >=2 collectors
      over-claiming a shared, smaller bin to trigger). Fixed by
      ITERATING: zero an excluded collector's own outgoing claims, redo
      (1), re-determine `used_marker`, repeat until it stops changing
      (converges in <= `nbins` rounds; the 5-bin repro converges in 2).
      Post-fix, that same 5-bin fixture conserves mass/aerosol mass to
      machine precision (~1e-16), `TestMultiCollectorConservation`.

  **Fidelity note on (2), stated plainly**: literal `ibreak=0`
  Fortran does NOT release an excluded collector's own outgoing claims
  -- that release exists in the source but is gated behind
  `if(ibreak==1)` (`mod_amps_core.F90:1748-1786`/`:1836-1874`). This port
  applies the SAME release UNCONDITIONALLY (not gated by `ibreak`) as a
  deliberate, physically-motivated correction: a fully-depleted collector
  cannot ALSO be independently collecting elsewhere, regardless of whether
  collisional breakup accounting happens to be active. Read as literally
  as possible, the real Fortran's own `ibreak=0` path may therefore leak
  mass in this same class of configuration -- this remains an open
  question for per-call dump validation (real dumps now available,
  M2b Task 7) to confirm one way or the other for `ibreak=0`; this port
  prioritizes GUARANTEED conservation (the task's own explicit, paramount
  bar) over bit-for-bit fidelity to a control-flow gap whose own
  conservation properties for `ibreak=0` are, as far as this task's own
  re-reading of `mod_amps_core.F90` can tell, genuinely unresolved without
  running the real code.

  **M2b Task 5 update -- `ibreak=1` IS now implemented** (`core/
  breakup.py`, wired in below). For `ibreak=1`, this SAME unconditional
  release is no longer merely "this port's own correction" -- reading the
  literal `if(ibreak==1)` block in full (not just the fidelity note above,
  which only quotes the LOCATION of the gate) shows it performs EXACTLY
  this release (a depleted collectee's own outgoing claims, scaled down)
  as its OWN mechanism, just with a numerically different form (graded
  `*mod_ratio` scaling per inner iteration vs. this port's hard
  zero-at-latch per outer fixed-point round). See `core/breakup.py`'s own
  module docstring ("RECONCILING T3's unconditional release with REAL
  `ibreak==1`") for the full derivation of this reconciliation, read
  directly from `mod_amps_core.F90:1712-1890` per this task's own dispatch
  instruction.
  `_needgive_repair` (`cal_needgive`, G4 SS6, KEPT per the reviewer's
  explicit instruction) remains as a residual safety net for any OTHER
  source of negative mass (e.g. floating-point edge cases in the PDF
  fit/scatter round trip), not as the primary over-depletion guard.
* **`add_fragments_col_vec` (collisional breakup, `ibreak=1`) IS now
  ported** -- M2b Task 5, `core/breakup.py`. `ibreak=True` requires
  `config.rain_collisional_breakup=True` (raises `ValueError` otherwise --
  the runtime physics below assumes `micexfg(18)==1`, `core/breakup.py`'s
  own module docstring) and a `breakup_tables: BreakupFragmentTables`
  argument (M2b Task 6's `lookup_tables.make_breakup_fragment_tables`
  output; also `ValueError` if omitted). Two pieces are added ONLY when
  `ibreak=True`, both delegated to `core/breakup.py` (see that module's own
  docstring for the full derivation, including the T3-vs-real-`ibreak==1`
  reconciliation the dispatch that added this explicitly asked for):

  1. `breakup.breakup_number_consumed` -- an ADDITIONAL depletion
     (`used_N_b`, H2 SS1c) applied to EVERY bin (both collector and
     collectee roles) inside this module's own `iter_loop1` fixed-point
     loop, from the `(1-E_coal)` breakup fraction of `N_col`, masked to
     `core.breakup.DenseFragmentTables.valid` (only pairs with an ACTUAL
     fragment-table entry contribute -- guarantees every unit of mass
     removed here has a matching fragment re-injection, see (2)).
  2. `breakup.add_fragments_col_vec` -- called once per `collector_loop1`
     iteration (same `i`), gated by `used_marker[i] | icond1_i` (H2 SS3a's
     own `used_marker(i,n)==1.or.icond1(i,n)==1`) -- NOT the same gate as
     the ordinary H/F/LM growth-pass call (`icond1_i` alone), since a bin
     already excluded from growing (`used_marker=1`) can still have
     (already-released-to-near-zero) outgoing breakup collisions to
     account for. Injects fragment number/mass (scaled by
     `N_bk*mod_rat`) into the SAME `new_N_1`/`new_M_1(rmt/rmat/rmas)`
     accumulators the growth pass writes.

  Mass balance (verified in `core/breakup.py`'s module docstring and
  empirically in `tests/.../test_breakup.py::TestCoalesceRainIbreak`): for
  every `(i,j)` pair with a valid fragment-table entry, fragment mass
  injected `= N_bk(i,j)*mod_rat(i,j)*bu_tmass(pair) = N_bk(i,j)*
  (mean_mass_i+mean_mass_j)` (since `mod_rat=(mean_mass_i+mean_mass_j)/
  bu_tmass` and `sum_k(bu_fd(1,kk))=bu_tmass` by the table's own `mrat`
  normalization, `core/breakfragment.py`) EXACTLY equals the mass removed
  from bins `i` and `j` via their own `used_N_b` contributions
  (`N_bk(i,j)*mean_mass_i` from `i`'s own "as collector" term +
  `N_bk(i,j)*mean_mass_j` from `j`'s own "as collectee" term) -- a
  transfer, not a leak, by construction.
* **The KiD autoconversion/accretion diagnostic second scatter**
  (H1 SS1.6i, `dM_auto`/`dM_accr`) is NOT computed -- not part of this
  task's Deliverable return signature (a `LiquidState`, not a
  diagnostic tuple); flagged, not silently included then discarded.
* **`checker`/`Qp`/`cal_Qp2_vec`** (H1 SS1.6e) are SKIPPED entirely --
  H1's own note: the `checker`-driven no-collision rollback this
  produces is guarded `g_1%token==2.and.g_2%token==1`, "NOT executed
  for warm rain" (H1 SS1.6e verbatim).

Self-collection (`i==j`) remains a proven, automatic no-op: `KC(i,i)
=E_c*(vtm_i-vtm_i)*A_c*con_i*dt=0` identically (same `diag` data at the
same bin) -> `N_col(i,i)=0` -> `col_ratio(i,i)=0` -> contributes to
neither "high" nor "low-medium" -- unaffected by this revision, still
covered by `TestActiveBinsAndSelfCollection.test_self_collection_is_a_
no_op`.
"""

from __future__ import annotations

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import (
    bin_grid,
    breakup,
    collision_kernel as ck,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    BreakupFragmentTables,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.vapor_deposition import (
    _gather_remap as _cal_transbin_gather_remap,
    _linear_fit as _cal_lincubprms_linear_fit,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import LiquidState, ThermoState


# G4 SS1.1 (`mod_amps_core.F90:1397-1400`): "minimum collector mass to be
# considered", `real(PS),parameter :: amin_mass=4.188790d-12`.
AMIN_MASS = 4.188790e-12

# `cal_collision_kernel_func`'s OWN icond1 floor (G4 SS2, `:15894-15907`),
# used for the O(n^2) kernel/efficiency computation -- matches
# `core/collision_kernel.py`'s own module docstring, which names this
# exact mask ("icond1") as the caller's responsibility to apply.
_ACTIVE_FLOOR = 1.0e-30

# `collector_loop1`'s OWN, STRICTER icond1 redefinition (`mod_amps_core.
# F90:1955-1966`) additionally requires `mean_mass>1e-15` -- distinct from
# `_ACTIVE_FLOOR` above; see module docstring's "STRUCTURAL PIECES" section.
_MEAN_MASS_FLOOR = 1.0e-15


def _pairwise(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast two `(nbins, npoints)` per-bin arrays to the bin-PAIR
    shape `(nbins_i, nbins_j, npoints)` -- `a` (group 1/"i"/collector)
    along axis 0, `b` (group 2/"j"/collectee) along axis 1. A local copy
    of `collision_kernel.py`'s own identical-purpose helper (`core/`
    modules do not cross-import each other's PRIVATE helpers as a
    general rule -- see module docstring for the one explicit exception
    this module makes, `_linear_fit`/`_gather_remap`)."""
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

    NOTE (post-review): this is now used ONLY as `_collector_scatter`'s
    OWN same-bin fallback for sub-populations whose `_linear_fit` itself
    fails (see module docstring) -- it is NOT the primary destination
    mechanism (that is the multi-bin `_cal_transbin_gather_remap`,
    reused from `core/vapor_deposition.py`).

    `binb` is 1D (`nbins+1` bin-mass boundaries); `target_mass` is any
    shape. Returns the 0-based bin index, same shape as `target_mass`,
    ALWAYS a valid index in `[0, nbins-1]` (this port's own defensible
    substitute for the Fortran's own undefined behavior at an exact
    `target==binb(1)` boundary match -- an unreachable floating-point
    coincidence in practice, see the task report).
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


# ---------------------------------------------------------------------------
# collector_loop1's categorization + fortunate/unfortunate split, G4
# `:2028-2096` verbatim -- see module docstring's algorithm section.
# ---------------------------------------------------------------------------


def _categorize_collectees(
    col_ratio: np.ndarray, active_pair: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The categorization loop, G4 `:2028-2054` verbatim (vectorized over
    the full `(nbins_j, npoints)` grid at once -- the Fortran's own
    `do j=jmax,1,-1` order does not affect this PART, only
    `_fortunate_scan`'s sequential exclusion below does).

    Returns `(is_high, is_lm, frac_part, ndrop_high)`, each
    `(nbins_j, npoints)`:

    * `is_high`: `col_ratio>=1` (covers both `[1,10]` and `>10`).
    * `is_lm`: `0<col_ratio<=10` (covers `(0,1)` and `[1,10]` --
      DELIBERATE dual membership with `is_high` for `[1,10]`, matching
      the Fortran's own `jbin_h`/`jbin_lm` dual-list append, `:2044-2049`).
    * `frac_part`: `col_ratio-floor(col_ratio)` for `[1,10]`, `col_ratio`
      itself for `(0,1)`, else `0` -- the "p_F"/"ndrop_B" (LM-pass alias)
      quantity, meaningful only where `is_lm`.
    * `ndrop_high`: `floor(col_ratio)` for `[1,10]`, `col_ratio` itself
      for `>10` (NOT floored -- G4's own H-pass verbatim,
      `ndrop_B=col_ratio` in the `else` branch), meaningful only where
      `is_high`.
    """
    in_1_10 = (col_ratio >= 1.0) & (col_ratio <= 10.0)
    in_0_1 = (col_ratio > 0.0) & (col_ratio < 1.0)
    is_high = active_pair & (col_ratio >= 1.0)
    is_lm = active_pair & (col_ratio > 0.0) & (col_ratio <= 10.0)
    frac_part = np.where(in_1_10, col_ratio - np.floor(col_ratio), np.where(in_0_1, col_ratio, 0.0))
    ndrop_high = np.where(in_1_10, np.floor(col_ratio), np.where(col_ratio > 10.0, col_ratio, 0.0))
    return is_high, is_lm, frac_part, ndrop_high


def _fortunate_scan(is_lm: np.ndarray, frac_part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The "non-overlapping fraction" sequential scan, G4 `:2060-2096`
    verbatim: low-medium collectee bins `j` are visited in DESCENDING
    bin-index order (matching `jbin_lm`'s own construction order, itself
    built by a descending `do j=jmax,1,-1` categorization loop, `:2028`),
    accumulating `sum_ndrop_B`; the FIRST `j` whose acceptance would push
    `sum_ndrop_B>=1` is EXCLUDED (and every SMALLER `j` for that column
    is skipped entirely from then on -- `icond2_n(n)==0` gate, `:2069`,
    a LATCH, not a per-`j` independent check).

    A genuinely SEQUENTIAL scan (unlike `_categorize_collectees`) -- an
    excluded bin's `p_F` is NOT added to the running total, so a later
    (smaller) bin's own acceptance test uses the SAME running total the
    excluded bin saw, not a monotonically-larger one. Implemented as an
    explicit Python loop over the (at most ~80) bin axis, vectorized over
    columns at each step -- not reducible to a plain `np.cumsum` (see the
    task report's derivation).

    Returns `(is_fortunate, sum_ndrop_b_final)`: `is_fortunate` is
    `(nbins_j, npoints)` bool (which `j`'s got their OWN dedicated
    sub-population); `sum_ndrop_b_final` is `(npoints,)`, the running
    total AFTER the full scan (feeds the "unfortunate" sub-population's
    own size, `(1-sum_ndrop_b_final)*left_N`).
    """
    nbins, npoints = is_lm.shape
    is_fortunate = np.zeros((nbins, npoints), dtype=bool)
    total = np.zeros(npoints)
    excluded_latch = np.zeros(npoints, dtype=bool)
    for j in range(nbins - 1, -1, -1):
        active_j = is_lm[j] & ~excluded_latch
        candidate = total + frac_part[j]
        would_exceed = active_j & (candidate >= 1.0)
        fortunate_j = active_j & ~would_exceed
        is_fortunate[j] = fortunate_j
        total = np.where(fortunate_j, candidate, total)
        excluded_latch = excluded_latch | (active_j & would_exceed)
    return is_fortunate, total


# ---------------------------------------------------------------------------
# One collector_loop1 iteration (fixed collector bin `i`): Mc seed, H/F/LM
# accumulation, cal_ratio_mass_col_vec, shifted boundaries, PDF fit
# (reusing core.vapor_deposition._linear_fit) + multi-bin scatter (reusing
# core.vapor_deposition._gather_remap). See module docstring's algorithm
# section for the full derivation.
# ---------------------------------------------------------------------------


def _collector_scatter(  # noqa: PLR0915, PLR0917 -- single verbatim-derived engine step, many H1/G4 sub-steps
    i: int,
    binb: np.ndarray,
    con_i: np.ndarray,
    mean_mass_i: np.ndarray,
    mass_tot_i: np.ndarray,
    mass_aero_tot_i: np.ndarray,
    mass_aero_sol_i: np.ndarray,
    left_n_i: np.ndarray,
    n_col_i: np.ndarray,
    e_coal_i: np.ndarray,
    mean_mass_all: np.ndarray,
    ratio_aero_tot: np.ndarray,
    ratio_aero_sol: np.ndarray,
    icond1_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One `collector_loop1` iteration for collector bin `i` (0-based).

    `con_i`/`mean_mass_i`/`mass_tot_i`/`mass_aero_tot_i`/
    `mass_aero_sol_i`/`left_n_i`: `(npoints,)`, collector bin `i`'s own
    per-column fields. `n_col_i`/`e_coal_i`: `(nbins_j, npoints)`, row
    `i` of the full bin-pair `N_col`/`E_coal` grids. `mean_mass_all`/
    `ratio_aero_tot`/`ratio_aero_sol`: `(nbins, npoints)`, EVERY bin's
    own mean mass / aerosol-mass FRACTION (`mass_aero_tot/mass_tot`,
    `cal_ratio_mass_col_vec`'s own `ratio_M_2` -- a dimensionless ratio,
    not a per-drop mass). `icond1_i`: `(npoints,)` bool, collector-role
    eligibility for THIS `i` (the STRICTER `collector_loop1`-own icond1,
    caller's responsibility -- see `_MEAN_MASS_FLOOR` and the
    `used_marker`-exclusion note in the module docstring).

    Population-preservation identity (verified here, not just asserted):
    `Np_sub` sums EXACTLY to `left_n_i` by construction --
    `sum_k(Np_fortunate(k)) + Np_unfortunate = (sum_ndrop_b_final)*
    left_n_i + (1-sum_ndrop_b_final)*left_n_i = left_n_i` -- which is
    exactly why the H/LM-pass "distributed proportional to
    `Np_sub(k)/left_n_i`" formula telescopes to `used_M_2(j)+=dM_dum_total(j)`
    EXACTLY when summed over `k` (`sum_k(Np_sub(k)/left_n_i)=1`), matching
    G4's own per-`k`-loop accumulation without needing an explicit
    `(n_sub, nbins_j)` intermediate array.

    Returns `(add_n, add_rmt, add_rmat, add_rmas, h_gain, f_gain,
    lm_gain)`: the first four are `(nbins, npoints)` scatter
    contributions (already summed over every sub-population, ready to
    add into the caller's running `new_N_1`/`new_M_1(rmt/rmat/rmas)`);
    the last three are `(nbins_j, npoints)`, THIS collector's own H-pass/
    F-pass/LM-pass mass contribution to EACH collectee `j`'s `used_M_2`
    -- returned SEPARATELY (not pre-summed) because only the H-pass
    contribution may trigger `used_marker` (module docstring's
    "STRUCTURAL PIECES" section) -- the caller sequences H before F
    before LM per column, matching the Fortran's own pass ORDER.
    """
    nbins = binb.shape[0] - 1
    npoints = con_i.shape[0]

    safe_con_i = np.where(con_i > 0.0, con_i, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        col_ratio = np.where(icond1_i[None, :], e_coal_i * n_col_i / safe_con_i[None, :], 0.0)
    active_pair = icond1_i[None, :] & (n_col_i > 0.0)
    is_high, is_lm, frac_part, ndrop_high = _categorize_collectees(col_ratio, active_pair)
    is_fortunate, sum_ndrop_b_final = _fortunate_scan(is_lm, frac_part)
    is_lm_leftover = is_lm & ~is_fortunate

    unit_gain = con_i[None, :] * mean_mass_all  # (nbins_j, npoints): con_i*mean_mass(j)

    h_gain = np.where(is_high, ndrop_high * unit_gain, 0.0)
    lm_gain = np.where(is_lm_leftover, frac_part * unit_gain, 0.0)
    dedicated_gain = np.where(is_fortunate, frac_part * unit_gain, 0.0)  # F-pass, own sub-bin only
    f_gain = dedicated_gain

    distributed_gain_per_j = h_gain + lm_gain
    distributed_ndrop = np.where(is_high, ndrop_high, 0.0) + np.where(
        is_lm_leftover, frac_part, 0.0
    )
    distributed_gain_total = distributed_gain_per_j.sum(axis=0)  # (npoints,)
    dmass_min_distributed = (distributed_ndrop * binb[:-1, None]).sum(axis=0)
    dmass_max_distributed = (distributed_ndrop * binb[1:, None]).sum(axis=0)
    dm1_distributed = (distributed_gain_per_j * ratio_aero_tot).sum(axis=0)
    dm2_distributed = (distributed_gain_per_j * ratio_aero_sol).sum(axis=0)

    # --- Sub-population axis: 0..nbins-1 = "fortunate" dedicated to
    # collectee j (zero where not fortunate); index nbins = "unfortunate".
    n_sub = nbins + 1
    np_sub = np.zeros((n_sub, npoints))
    np_sub[:nbins] = np.where(is_fortunate, frac_part * left_n_i[None, :], 0.0)
    valid_unfortunate = icond1_i & (left_n_i > 0.0)
    np_sub[nbins] = np.where(valid_unfortunate, (1.0 - sum_ndrop_b_final) * left_n_i, 0.0)

    seed_mass = np_sub * mean_mass_i[None, :]  # Mc-seed timing: BEFORE any pass, module docstring.
    safe_left_n_i = np.where(left_n_i > 0.0, left_n_i, 1.0)
    share = (
        np_sub / safe_left_n_i[None, :]
    )  # Np_sub(k)/left_N(i), the H/LM-pass distribution weight

    mp_sub = seed_mass.copy()
    mp_sub[:nbins] += dedicated_gain  # F-pass: own sub-bin only
    mp_sub += share * distributed_gain_total[None, :]  # H+LM-pass: every sub-bin, proportional

    dmass_min = np.zeros((n_sub, npoints))
    dmass_max = np.zeros((n_sub, npoints))
    dmass_min[:nbins] = np.where(is_fortunate, binb[:-1, None], 0.0)  # F-pass: ndrop_F=1.0
    dmass_max[:nbins] = np.where(is_fortunate, binb[1:, None], 0.0)
    dmass_min += dmass_min_distributed[None, :]
    dmass_max += dmass_max_distributed[None, :]

    ratio_aero_tot_i = _safe_div(mass_aero_tot_i, mass_tot_i)  # (npoints,), collector i's OWN ratio
    ratio_aero_sol_i = _safe_div(mass_aero_sol_i, mass_tot_i)
    mc_rmat = seed_mass * ratio_aero_tot_i[None, :]
    mc_rmas = seed_mass * ratio_aero_sol_i[None, :]

    dm_1 = np.zeros((n_sub, npoints))
    dm_1[:nbins] = np.where(is_fortunate, ratio_aero_tot * dedicated_gain, 0.0)
    dm_1 += share * dm1_distributed[None, :]
    dm_2 = np.zeros((n_sub, npoints))
    dm_2[:nbins] = np.where(is_fortunate, ratio_aero_sol * dedicated_gain, 0.0)
    dm_2 += share * dm2_distributed[None, :]

    ratio_mp_rmat = _safe_div(mc_rmat + dm_1, mp_sub)
    ratio_mp_rmas = _safe_div(mc_rmas + dm_2, mp_sub)

    binb3d_lo = binb[i] + dmass_min
    binb3d_hi = binb[i + 1] + dmass_max

    valid_sub = (np_sub > 1.0e-30) & (mp_sub > 1.0e-30)

    a1, a2, a3, ok = _cal_lincubprms_linear_fit(np_sub, mp_sub, binb3d_lo, binb3d_hi)
    ok = ok & valid_sub

    intercept = np.where(a1 >= 0.0, a1, 0.0)
    poly_a0 = np.where(ok, intercept - a3 * a2, 0.0)
    poly_a1 = np.where(ok, a3, 0.0)
    poly_a2 = np.zeros_like(poly_a0)
    poly_a3 = np.zeros_like(poly_a0)
    truncated_left = ok & (a1 == -1.0)
    truncated_right = ok & (a1 == -2.0)
    eff_bd1 = np.where(truncated_right, binb3d_lo, np.where(truncated_left, a2, binb3d_lo))
    eff_bd2 = np.where(truncated_left, binb3d_hi, np.where(truncated_right, a2, binb3d_hi))
    ok = ok & (eff_bd2 > eff_bd1)

    add_n, add_rmt, add_rmat, add_rmas = _cal_transbin_gather_remap(
        binb, poly_a0, poly_a1, poly_a2, poly_a3, eff_bd1, eff_bd2, ratio_mp_rmat, ratio_mp_rmas, ok
    )

    # Fallback for sub-populations whose linear fit itself failed (module
    # docstring's "one deliberate, documented reduction"): same-bin
    # (nearest, by mean mass) concentrated placement, still exactly
    # N/M-conserving.
    fallback = valid_sub & ~ok
    if fallback.any():
        mean_mass_sub = _safe_div(mp_sub, np_sub)
        dest_fb = _find_destination_bin(binb, mean_mass_sub)
        pts = np.broadcast_to(np.arange(npoints)[None, :], dest_fb.shape)
        np.add.at(add_n, (dest_fb, pts), np.where(fallback, np_sub, 0.0))
        np.add.at(add_rmt, (dest_fb, pts), np.where(fallback, mp_sub, 0.0))
        np.add.at(add_rmat, (dest_fb, pts), np.where(fallback, mp_sub * ratio_mp_rmat, 0.0))
        np.add.at(add_rmas, (dest_fb, pts), np.where(fallback, mp_sub * ratio_mp_rmas, 0.0))

    return add_n, add_rmt, add_rmat, add_rmas, h_gain, f_gain, lm_gain


def coalesce_rain(  # noqa: PLR0912, PLR0915, PLR0917 -- single verbatim-derived engine, many H1/G4 sub-steps
    liquid_pv: LiquidState,
    diag: LiquidDiag,
    thermo: ThermoState,
    config: AmpsConfig,
    dt: float,
    luts: AmpsLuts,
    *,
    ibreak: bool = False,
    breakup_tables: BreakupFragmentTables | None = None,
) -> LiquidState:
    """The rain-rain (`token 1-1`) bin-pair collection engine --
    `collector_loop1` ported in full (H/F/LM categorization, multi-bin
    PDF scatter), per this module's own docstring (read it first: the
    full algorithm derivation, the `pro_type`/`used_marker`/
    `assign_tendency_vec` structural resolutions, PER-VOLUME contract,
    and every documented simplification are all there, not repeated
    here).

    Signature deviation from the dispatch's literal
    `coalesce_rain(liquid_pv, diag, config, dt, luts)` text (same
    necessary-addition pattern M2b Task 2 already established for
    `collision_kernel`'s own `luts`/`col_level` additions): `thermo:
    ThermoState` was added -- `coalescence_efficiency` (Task 2,
    `core/collision_kernel.py`) needs the column temperature for its own
    `th_var%sig_wa` surface-tension term, and there is no way to compute
    `E_coal` (load-bearing for `col_ratio`, the H/F/LM categorization
    driver) without it.

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
        ibreak: collisional-breakup flag (`micexfg(18)`/H2's own `ibreak`).
            `True` requires `config.rain_collisional_breakup=True` AND a
            `breakup_tables` argument (both validated below, `ValueError`
            otherwise) -- M2b Task 5, `core/breakup.py`, see module
            docstring's `add_fragments_col_vec` section.
        breakup_tables: `BreakupFragmentTables` (M2b Task 6,
            `lookup_tables.make_breakup_fragment_tables(config.
            num_h_bins[0], config.nbin_h)`) -- required when `ibreak=True`,
            ignored otherwise. NOT rebuilt internally (unlike `binb`
            above): it depends only on static config, not the runtime
            `liquid_pv`, so the caller is expected to build it ONCE and
            reuse it across timesteps (building it costs ~0.3s for
            cloudlab's 40-bin grid, M2b Task 6's own report).

    Returns:
        A NEW, PER-VOLUME `LiquidState` (same shape as `liquid_pv`;
        `liquid_pv` itself is never mutated).

    Raises:
        NotImplementedError: `liquid_pv.ncat != 1`.
        ValueError: `ibreak=True` with `config.rain_collisional_breakup
            =False` or `breakup_tables=None`.
    """
    if ibreak and not config.rain_collisional_breakup:
        raise ValueError(
            "coalesce_rain: ibreak=True requires config.rain_collisional_breakup=True "
            "(micexfg(18)==1) -- the ibreak=1 runtime (E_coal/N_col accounting, fragment "
            "injection) assumes breakup is the active config, not just a caller override. "
            "See core/breakup.py's module docstring."
        )
    if ibreak and breakup_tables is None:
        raise ValueError(
            "coalesce_rain: ibreak=True requires breakup_tables (a BreakupFragmentTables, "
            "e.g. lookup_tables.make_breakup_fragment_tables(config.num_h_bins[0], "
            "config.nbin_h)) -- see core/breakup.py."
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
    mean_mass = diag.mean_mass

    # icond1 (cal_collision_kernel_func's OWN mask, H1 SS0 quoting G4 SS2,
    # `mod_amps_core.F90:15894-15907`): active where BOTH con and mass(1)
    # clear the floor, for BOTH bins of the pair -- used ONLY for the O(n^2)
    # kernel/efficiency computation below, per this task's own dispatch
    # instruction. `collector_loop1`'s OWN, stricter icond1 is separate,
    # see `active_collector_base` further below.
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
    # `icolbin_min` is a pure function of `binb` alone -- computed here
    # (moved up from just before `collector_loop1` below) so the `ibreak`
    # domain assertion right below can use it too.
    icolbin_min = _icolbin_min(binb)

    dense: breakup.DenseFragmentTables | None = None
    if ibreak:
        assert breakup_tables is not None  # validated above
        dense = breakup.dense_fragment_tables(breakup_tables, nbins)
        # `breakup_number_consumed`/`add_fragments_col_vec` are only ever
        # consulted for `i>=dense.imin_bk` (the fragment table's own
        # domain) -- `collector_loop1` below never visits `i<icolbin_min`,
        # so conservation (module docstring's "conservation-by-
        # construction" note in `core/breakup.py`) requires the table's
        # own collector range to lie ENTIRELY inside the loop's own range.
        # Holds for cloudlab (`icolbin_min` is a tiny haze-scale cutoff far
        # below `jmin_bk`'s `D_0=0.01cm` rain-scale cutoff, M2b Task 6's
        # own report) -- asserted explicitly rather than silently risking
        # a mass leak for a hypothetical config where it does not.
        if dense.imin_bk < icolbin_min:
            raise ValueError(
                f"coalesce_rain: ibreak=True requires breakup_tables.imin_bk "
                f"({dense.imin_bk + 1}, 1-based) >= icolbin_min ({icolbin_min + 1}, 1-based) "
                f"-- collector_loop1 only visits i>=icolbin_min, so a smaller imin_bk would "
                f"leave some breakup-consumption pairs without a matching fragment-injection "
                f"call (see core/breakup.py's module docstring)."
            )

    # --- Task 2 kernel / efficiency, icond1-masked. `errstate` suppresses
    # benign 0/0 or x/0 RuntimeWarnings from `collision_efficiency`'s own
    # `rrat=len_j/len_i` computation across the MANY inactive (zero-length)
    # bin pairs a realistic 40-bin grid contains -- masked to 0 via
    # `icond1_active` immediately below regardless. ---
    with np.errstate(divide="ignore", invalid="ignore"):
        kc = ck.collision_kernel(diag, diag, con, dt, luts, col_level=config.coll_level)
        e_coal, *_rest = ck.coalescence_efficiency(diag, diag, thermo)

    kc = np.where(icond1_active, kc, 0.0)
    e_coal = np.where(icond1_active, e_coal, 0.0)

    # --- N_col, G4 SS1.2 verbatim (`mod_amps_core.F90:1634-1664`): only
    # the KC>0 branch applies for rain-rain (token_i==token_j, so the
    # abs(KC) riming branch, `:1668` guarded by `token /= token`, never
    # fires); clamp N_col<=con_j. ---
    con_i_pair, con_j_pair = _pairwise(con, con)
    n_col = np.where(kc > 0.0, con_i_pair * kc, 0.0)
    n_col = np.minimum(n_col, con_j_pair)
    n_col = np.where(icond1_active, n_col, 0.0)

    # --- iter_loop1 (G4 SS1.4, `mod_amps_core.F90:1713-1890`) PLUS this
    # module's own hardening-driven fixed-point extension -- see module
    # docstring's "iter_loop1 fixed point" section for the full
    # derivation and the two DISTINCT bugs this closes. Summary:
    #
    # (1) The per-pair `N_col<=con_j` clamp above bounds each INDIVIDUAL
    #     collector's own claim, but NOT the SUM across every collector
    #     `i` targeting the SAME collectee `j` -- with >=3 active bins,
    #     MULTIPLE collectors independently claiming up to `con_j` each
    #     manufactures more growth mass downstream than bin `j` actually
    #     has (a mass CREATION bug -- `_needgive_repair` cannot catch
    #     this, it only fires on NEGATIVE bins). Fixed by a proportional
    #     rescale of every collector's claim on an over-subscribed `j`,
    #     driving `used_N_2(j)` to EXACTLY `con(j)`.
    # (2) A collector `i` that ITSELF ends up fully claimed by an even
    #     bigger collector (`used_marker(i)=1`, excluded from ever
    #     running its own `collector_loop1` body, `mod_amps_core.F90:
    #     1955-1966`) still has its ORIGINAL, un-diminished claim on
    #     smaller collectees baked into THEIR `left_N`/`left_M` -- a
    #     "phantom" claim that never actually executes (since `i` never
    #     runs to scatter anything into any destination bin), silently
    #     vanishing mass with no negative bin for `_needgive_repair` to
    #     catch either. Fixed by ITERATING to a fixed point: zero an
    #     excluded collector's own OUTGOING claims, re-run fix (1), and
    #     re-determine `used_marker`; repeat until `used_marker` stops
    #     changing.
    #
    # `mean_mass(j)=mass_tot(j)/con(j)` exactly (`core/liquid_diag.py`'s
    # own definition), so `used_M_2(j)=used_N_2(j)*mean_mass(j)
    # >=mass_tot(j)` iff `used_N_2(j)>=con(j)` -- the Fortran's OWN
    # number-limit and mass-limit halves (`:1713-1799`/`:1801-1884`) are
    # mathematically IDENTICAL conditions here, so EACH round of fix (1)
    # is a single pass, not the Fortran's own `do ii=1,iter` alternation
    # (needed in the general ice/mixed-token case where mean_mass need
    # not be so uniformly defined). The OUTER fixed-point loop (over (2))
    # converges in at most `nbins` rounds (each round either marks >=1
    # NEW bin `used_marker` or the state stops changing and the loop
    # exits) -- `TestMultiCollectorConservation`'s own 5-bin fixture
    # converges in 2 rounds.
    ratio_aero_tot = _safe_div(mass_aero_tot, mass_tot)  # (nbins, npoints), dimensionless FRACTION
    ratio_aero_sol = _safe_div(mass_aero_sol, mass_tot)

    # `collector_loop1`'s OWN, stricter icond1 base (`mod_amps_core.F90:
    # 1955-1966`): adds the mean_mass floor -- the `used_marker/=1` half
    # (for rain-rain, pro_type=2, see module docstring) is the fixed
    # point this loop iterates on.
    #
    # T7 (M2b Task 7, warm-dump-validation fix) ADDS `mean_mass<=binb[-1]`
    # -- NOT part of any literal Fortran `icond1` clause (G4/H1's own
    # `con>1e-30 .and. mass(1)>1e-30 .and. mean_mass>1e-15` is a FLOOR only,
    # no ceiling anywhere in the quoted text), but a real-dump validation
    # necessity `_MEAN_MASS_FLOOR` alone cannot catch: a bin with a
    # floating-point-noise-level `con` (e.g. ~1e-14 per-volume, far above
    # `_ACTIVE_FLOOR`/the Fortran's own `1e-30`) paired with an ordinary-
    # magnitude leftover `mass_tot` computes `mean_mass=mass_tot/con` many
    # ORDERS OF MAGNITUDE beyond `binb[-1]` (the grid's own largest
    # representable mean-drop mass, ~0.52g for cloudlab's 40-bin liquid
    # grid -- already a generous upper bound on real raindrop size) --
    # observed real-dump values up to ~1.8e5 g (a "raindrop" ~1000x more
    # massive than a car). `_collector_scatter`'s own degenerate-fallback
    # path (`_find_destination_bin`) then clips such an out-of-grid mean
    # mass to the TOP bin, corrupting it with that noise bin's real (but
    # tiny) mass and its raw, never-consulted-by-the-real-Fortran `rmat`/
    # `rmas` payload -- confirmed against a real dump (M2b Task 7 report):
    # this is NOT a rare tail case, it recurs across many replayed
    # records/points, and is NOT the T3 `ibreak=0`-conservation hypothesis
    # this task was dispatched expecting to confirm/refute. `binb[-1]` (not
    # a per-bin `binb[i+1]`) is the deliberately LOOSE choice -- catching
    # only genuinely pathological (order(s)-of-magnitude) violations, never
    # a legitimate bin whose own PDF-remap-approximated mean mass lands
    # slightly outside ITS OWN narrow bin range but still well inside the
    # grid overall (an ordinary, harmless bin-scheme approximation, not a
    # defect). See also `counter_active`, below -- a SEPARATE, ALSO-
    # necessary T7 fix for the companion case (a column with only ONE
    # occupied bin, whose own mean_mass may be perfectly ordinary but which
    # never generates any genuine collision claim either way).
    active_collector_base = active & (mean_mass > _MEAN_MASS_FLOOR) & (mean_mass <= binb[-1])

    # `ibreak=True` needs TWO DISTINCT views of the collision-count grid,
    # both derived from the SAME initial `n_col` but evolving DIFFERENTLY
    # across the fixed-point rounds below -- discovered empirically (a
    # single shared `n_col` oscillates/leaks, see the report):
    #
    # * `n_col` ("growth" claims, E_coal-weighted): ordinary coalescence
    #   NEVER depletes the COLLECTOR's own population (only the
    #   COLLECTEE's) -- a fully-used_marker'd collector `i` cannot ALSO
    #   independently grow via ordinary absorption (it never runs
    #   `_collector_scatter` at all, `icond1_i=False` below), so `i`'s row
    #   must be released to a HARD ZERO (T3's own original bug-(2) fix,
    #   UNCHANGED, both `ibreak` values) once `i` latches `used_marker` --
    #   any mass a used_marker'd `i` would have "claimed" via ordinary
    #   growth must NEVER be subtracted from a collectee either (nothing
    #   would ever receive it).
    # * `n_col_bk` ("breakup" claims, `(1-E_coal)`-weighted, `ibreak=True`
    #   ONLY): breakup DOES deplete BOTH participants regardless of role
    #   (H2 SS1c's own `used_N_b`, both loops) -- fragment injection
    #   (`add_fragments_col_vec`, below) STILL needs to fire for a
    #   used_marker'd collector's own breakup interactions (a fully
    #   consumed bin's ENTIRE population may have gone into fragmenting
    #   collisions, not ordinary growth), so this view is GRADED
    #   (proportionally rescaled, both row AND column, matching H2's own
    #   literal `ibreak==1` mechanism -- `core/breakup.py`'s module
    #   docstring "RECONCILING..." section), never hard-zeroed.
    n_col_bk = n_col.copy() if ibreak else n_col

    # T7 companion fix (found while adding this task's own regression
    # tests, see TestT7DegenerateBinGuards::
    # test_mutually_degenerate_column_bounded_by_mean_mass_ceiling's own
    # docstring): the "release excluded i's outgoing claims" step below
    # MUST also cover a bin `i` that will NEVER run `_collector_scatter`
    # because it fails `active_collector_base` from round 0 (T7's
    # `mean_mass<=binb[-1]` ceiling, or the pre-existing `mean_mass>
    # _MEAN_MASS_FLOOR` floor) -- NOT only `used_marker`-latched bins.
    # Pre-T7, `active_collector_base=False` implied `mean_mass<=1e-15`,
    # so any such bin's own outgoing claim (`unit_gain=con_i*mean_mass_j`)
    # was already negligible and this gap was invisible; T7's own new
    # ceiling exclusion can apply to a bin with a REAL, substantial `con`
    # (just a wrong/absurd `mean_mass`), whose un-released phantom claim
    # on a genuine collectee is a real, non-negligible mass LEAK (found
    # empirically: a 2-bin fixture with a mean_mass-ceiling-excluded
    # collector leaked ~17% of total mass without this release).
    never_collects = ~active_collector_base
    used_marker = np.zeros((nbins, npoints), dtype=bool)
    left_n = con.copy()
    left_m = mass_tot.copy()
    for _round in range(nbins):
        n_col = np.where(
            (used_marker | never_collects)[:, None, :], 0.0, n_col
        )  # release excluded i's outgoing claims

        used_n_2 = (n_col * e_coal).sum(axis=0)  # (nbins_j, npoints), summed over collector i
        if ibreak:
            # H2 SS1c's `used_N_b` -- ADDITIONAL depletion (both collector
            # and collectee roles) from the breakup fraction of N_col, on
            # TOP of the ordinary coalescence-only collectee depletion
            # above. See core/breakup.py's module docstring for the full
            # `ibreak==1` reconciliation.
            assert dense is not None
            used_n_2 = used_n_2 + breakup.breakup_number_consumed(n_col_bk, e_coal, dense)
        over_claimed = used_n_2 > con
        rescale = np.where(over_claimed, _safe_div(con, used_n_2), 1.0)
        n_col = n_col * rescale[None, :, :]  # column k (as collectee) rescale -- BOTH ibreak values
        if ibreak:
            # H2's own `ibreak==1`-gated block (`mod_amps_core.F90:1748-
            # 1762`/`:1836-1850`, read directly per this task's dispatch --
            # see core/breakup.py's module docstring for the full quote):
            # `n_col_bk` gets BOTH the column rescale (over-claimed as
            # collectee, same factor as `n_col` above) AND the row rescale
            # (over-claimed as collector-via-breakup, `k`'s OWN outgoing
            # claims `n_col_bk[k,j]` for `j/=k`) -- graded, not zeroed, so
            # a used_marker'd `k`'s own breakup consumption still fires at
            # the CORRECT (population-capped) level.
            n_col_bk = n_col_bk * rescale[None, :, :] * rescale[:, None, :]

        used_n_2 = (n_col * e_coal).sum(axis=0)
        if ibreak:
            assert dense is not None
            used_n_2 = used_n_2 + breakup.breakup_number_consumed(n_col_bk, e_coal, dense)
        used_m_2 = used_n_2 * mean_mass  # mean_mass(j) doesn't depend on i, factor out
        left_n = np.maximum(con - used_n_2, 0.0)
        left_m = np.maximum(mass_tot - used_m_2, 0.0)

        new_used_marker = active_collector_base & ((left_n <= 1.0e-30) | (left_m <= 1.0e-30))
        converged = np.array_equal(new_used_marker, used_marker)
        used_marker = new_used_marker
        if converged:
            break

    # M2b Task 7 warm-dump-validation fix (T7): `counter(n)>0`, G4/H1's OWN
    # excerpts never quote this -- found by reading `mod_amps_core.F90:
    # 1899-1907`/`:1955-1958` DIRECTLY (dispatch-authorized) while
    # diagnosing a real-dump divergence. `collector_loop1`'s OWN `icond1`
    # redefinition (`:1955-1966`) is `icycle_n(n)==0 .and. counter(n)>0
    # .and. (con>1e-30 .and. mass(1)>1e-30 .and. mean_mass>1e-15) .and.
    # (used_marker/=1 .or. pro_type/=2)` -- a COLUMN-LEVEL `counter(n)>0`
    # conjunct this module's own `active_collector_base` (the parenthesized
    # per-bin clause alone) was missing entirely. `counter(n)` (`:1899-1907`,
    # computed AFTER `iter_loop1` finishes, from the SAME final `used_N_2`
    # this loop's own `used_n_2` matches) counts, per column, how many bins
    # were ACTUALLY claimed as a collectee by the O(n^2) kernel pass --
    # `counter(n)==0` means NOTHING in that whole column ever had a
    # meaningful collision claim, so `collector_loop1` skips EVERY bin in
    # that column, REGARDLESS of any individual bin's own con/mass_tot/
    # mean_mass. Root cause of a real-dump-only failure mode this port's
    # synthetic-fixture tests never triggered: a bin with a floating-point-
    # noise-level residual `con` (e.g. ~1e-14 per-volume, comfortably above
    # the bare `>1e-30` floor but PHYSICALLY negligible) paired with a
    # normal-magnitude leftover `mass_tot` computes an absurd `mean_mass=
    # mass_tot/con` (millions of grams per "drop") that clears
    # `_MEAN_MASS_FLOOR` easily and gets treated as a legitimate ACTIVE
    # COLLECTOR -- `_collector_scatter`'s own degenerate-fallback path
    # (`_find_destination_bin`) then clips that absurd mean_mass to the
    # GRID'S OWN TOP BIN, corrupting it with the noise bin's real (but
    # tiny) mass AND its raw, never-consulted-by-the-real-Fortran `rmat`/
    # `rmas` payload. `counter(n)>0` is EXACTLY the guard that (in the real
    # Fortran) keeps such a bin inert: if it never generated/received a
    # real collision claim (which a ~1e-14 con bin never will, since every
    # `N_col` claim against/from it is clamped by that same tiny `con`),
    # `counter(n)` for its WHOLE column stays 0 and `collector_loop1` never
    # touches ANY bin there.
    counter_active = np.any(used_n_2 > 0.0, axis=0)  # (npoints,)

    # A bin that would have been an eligible collector under the PRE-T7
    # rule (`active & mean_mass>_MEAN_MASS_FLOOR`, the module's own
    # original `active_collector_base` formula) but is excluded by EITHER
    # new T7 conjunct (`mean_mass<=binb[-1]` or `counter_active`) still
    # holds its own genuine `left_n`/`left_m` (unchanged from `con`/
    # `mass_tot` minus whatever a VALID other collector legitimately
    # claimed of it as a collectee -- itself naturally small, since a bin
    # excluded here has either a degenerate mean_mass or sits in a column
    # with no real collision activity) -- re-added below, mirroring the
    # icolbin_min-excluded-bin precedent (module docstring). Bins that
    # were ALREADY excluded pre-T7 (`mean_mass<=_MEAN_MASS_FLOOR`) are
    # UNCHANGED by this -- that pre-existing, narrower gap (both con AND
    # mass_tot near-zero, contributing ~nothing either way) is out of this
    # fix's own scope.
    excluded_by_t7 = (
        active
        & (mean_mass > _MEAN_MASS_FLOOR)
        & ((mean_mass > binb[-1]) | ~counter_active[None, :])
    )

    used_m_2_total = np.zeros((nbins, npoints))
    new_n_1 = np.zeros((nbins, npoints))
    new_m_rmt = np.zeros((nbins, npoints))
    new_m_rmat = np.zeros((nbins, npoints))
    new_m_rmas = np.zeros((nbins, npoints))

    # collector_loop1: do i=g_1%N_bin,icolbin_min,-1 (H1 SS1.6 verbatim,
    # Fortran 1-based descending) -- Python 0-based descending equivalent:
    for i in range(nbins - 1, icolbin_min - 1, -1):
        icond1_i = active_collector_base[i] & ~used_marker[i] & counter_active
        if icond1_i.any():
            add_n, add_rmt, add_rmat, add_rmas, h_gain, f_gain, lm_gain = _collector_scatter(
                i,
                binb,
                con[i],
                mean_mass[i],
                mass_tot[i],
                mass_aero_tot[i],
                mass_aero_sol[i],
                left_n[i],
                n_col[i],
                e_coal[i],
                mean_mass,
                ratio_aero_tot,
                ratio_aero_sol,
                icond1_i,
            )
            new_n_1 = new_n_1 + add_n
            new_m_rmt = new_m_rmt + add_rmt
            new_m_rmat = new_m_rmat + add_rmat
            new_m_rmas = new_m_rmas + add_rmas

            # Sequenced used_M_2 update, H then F then LM (module
            # docstring's "STRUCTURAL PIECES" section) -- ONLY the H-pass
            # may set used_marker; F/LM clamp used_M_2 without setting it.
            used_m_2_after_h = used_m_2_total + h_gain
            newly_marked = used_m_2_after_h >= mass_tot
            used_m_2_total = np.where(newly_marked, mass_tot, used_m_2_after_h)
            used_marker = used_marker | newly_marked

            used_m_2_total = np.minimum(used_m_2_total + f_gain, mass_tot)
            used_m_2_total = np.minimum(used_m_2_total + lm_gain, mass_tot)

        if ibreak:
            # H2 SS3a's own `used_marker(i,n)==1.or.icond1(i,n)==1` gate --
            # BROADER than `icond1_i` alone (a bin already excluded from
            # growing can still have outgoing breakup collisions to
            # account for, module docstring's `add_fragments_col_vec`
            # section). Called for THIS SAME i, after the growth pass
            # above (H2's own call-site ordering, mod_amps_core.F90:
            # 2884-2893), using `n_col_bk[i]` (the GRADED breakup view, NOT
            # `n_col[i]` -- the growth view's row is hard-zeroed once `i`
            # is used_marker'd, which would incorrectly zero out `i`'s own
            # breakup interactions too; see this loop's own fixed-point
            # comment above).
            assert dense is not None
            frag_gate_i = used_marker[i] | icond1_i
            if frag_gate_i.any():
                add_n_f, add_rmt_f, add_rmat_f, add_rmas_f = breakup.add_fragments_col_vec(
                    i,
                    n_col_bk[i],
                    e_coal[i],
                    mean_mass,
                    mass_tot,
                    mass_aero_tot,
                    mass_aero_sol,
                    frag_gate_i,
                    dense,
                )
                new_n_1 = new_n_1 + add_n_f
                new_m_rmt = new_m_rmt + add_rmt_f
                new_m_rmat = new_m_rmat + add_rmat_f
                new_m_rmas = new_m_rmas + add_rmas_f

    # --- Post-loop leftover re-add, G4 SS1.6k verbatim
    # (`mod_amps_core.F90:2899-2925`) PLUS this module's own documented,
    # mass-safe extensions: icolbin_min-excluded bins (module docstring's
    # "STRUCTURAL PIECES" section) AND, as of the T7 fixes above,
    # `excluded_by_t7` bins -- a bin excluded from `collector_loop1` by
    # EITHER new T7 gate (`mean_mass>binb[-1]` or `counter_active==False`)
    # is, by the SAME construction as an icolbin_min-excluded bin, inert
    # for THIS role (`left_n`/`left_m` reflecting only whatever a VALID
    # other collector legitimately claimed of it as a collectee, if
    # anything): re-adding it is what reproduces the real Fortran's own
    # observed behavior for such a bin (bit-identical pre->post for the
    # isolated-degenerate-bin case, confirmed against a real dump -- see
    # the M2b Task 7 report) -- NOT the literal Fortran's own quoted
    # `used_marker(i,n)==1`-only condition (H1 SS1.6k), which this port's
    # own architecture (computing REPLACEMENT `new_N_1`/`new_M_1` values,
    # not the raw Fortran's `assign_tendency_vec`-mediated tendencies) has
    # no other mechanism to reproduce -- same "documented, MASS-SAFE choice
    # for a genuinely underspecified corner" precedent as icolbin_min. ---
    below_icolbin_min = np.zeros(nbins, dtype=bool)
    below_icolbin_min[:icolbin_min] = True
    re_add = used_marker | below_icolbin_min[:, None] | excluded_by_t7
    new_n_1 = new_n_1 + np.where(re_add, left_n, 0.0)
    new_m_rmt = new_m_rmt + np.where(re_add, left_m, 0.0)
    new_m_rmat = new_m_rmat + np.where(re_add, left_m * ratio_aero_tot, 0.0)
    new_m_rmas = new_m_rmas + np.where(re_add, left_m * ratio_aero_sol, 0.0)

    # --- cal_needgive inter-bin borrowing (G4 SS6), within-group repair
    # for any bin over-depleted by the non-iterative left_N/left_M
    # approximation (module docstring's own flagged reduction). ---
    new_n_1, new_m_rmt, (new_m_rmat, new_m_rmas) = _needgive_repair(
        new_n_1, new_m_rmt, new_m_rmat, new_m_rmas
    )

    out_values = liquid_pv.values.copy()
    out_values[lp.rcon_q.py_idx, :, 0, :] = new_n_1
    out_values[lp.rmt_q.py_idx, :, 0, :] = new_m_rmt
    out_values[lp.rmat_q.py_idx, :, 0, :] = new_m_rmat
    out_values[lp.rmas_q.py_idx, :, 0, :] = new_m_rmas
    return LiquidState(values=out_values)
