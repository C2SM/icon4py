# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/coalescence.py (M2b Task 3): `coalesce_rain`, the
rain-rain bin-pair collection engine, per
docs/superpowers/facts/m2b/coalescence-engine.md ("H1") SS1.6 +
docs/superpowers/facts/m2/coalescence.md ("G4") SS1, and this task's own
dispatch. REVISED per code review: `coalesce_rain` now ports the REAL
`collector_loop1` (H/F/LM rate categorization, multi-bin PDF fit and
scatter reusing `core/vapor_deposition.py`'s M2a `cal_lincubprms_vec`/
`cal_transbin_vec` ports) -- see `core/coalescence.py`'s module docstring
for the full algorithm derivation. An earlier revision used a simplified
"2 parents in, 1 merged child out" single-target-bin model (a mis-citation
of H1's own "M2b codegen notes" `add_simple_vec` fallback as the PRIMARY
engine); `TestCollectorPreservation` below is the test that would have
FAILED against that reverted model (its own collector count is destroyed
2-for-1 instead of preserved-while-growing) and is the review's own
explicit requirement.

Fixtures use REAL magnitudes throughout: number densities in the
per-cm^3 ~1e2-1e3 range (the dispatch's own explicit ask, after M2a's
"degenerate fixtures hide real bugs" lesson), a REAL 40-bin liquid grid
(`bin_grid.make_bin_grid("liquid", 40, nbin_h=20)` -- `coalesce_rain`
calls this internally with `config.num_h_bins[0]`/`config.nbin_h`, and
`make_bin_grid` only accepts nbins in {40,80}, so a toy 2-3 bin state is
not an option here), and the REAL packaged collision-efficiency LUT
(`load_luts()`) for the physical kernel path -- only `TestDestinationBin`
and `TestNeedgiveRepair` use hand-picked `LiquidDiag` fields chosen for
exactly-reproducible bin-search arithmetic.

NOTE on exact vs. approximate identity: under the real multi-bin PDF
engine, even a "nothing should happen" scenario (dt=0, self-collection)
is NOT bit-exact -- every active bin's population round-trips through its
own "unfortunate" sub-population PDF fit (`_linear_fit`) and multi-bin
scatter (`_gather_remap`) regardless of whether any actual collection
occurred, introducing machine-epsilon-level (~1e-14 absolute) floating-
point noise. This is expected and within the design spec's own ~1e-10
rtol per-call validation tolerance -- `pytest.approx` (tight rel/abs
tolerance), not `np.array_equal`, is used for these "should be unchanged"
assertions; genuinely EXACT identities (all-inactive state, `dt=0`
producing `KC=0` in the O(n^2) kernel stage) still use exact equality
where the code path in question provably returns the input unchanged
without any PDF round-trip (`icond1_active.any()==False` early return).

Groups:
* TestActiveBinsAndSelfCollection -- icond1 masking, self-collection
  (i==j) is a proven no-op (KC=0 via vtm_i-vtm_i=0), icolbin_min collector
  floor.
* TestTwoBinCollection -- number decreases, mass conserves EXACTLY
  (1e-12) at con~500 cm^-3 magnitudes, via the real multi-bin engine.
* TestCollectorPreservation -- THE REVIEW'S OWN REQUIREMENT: in the
  H-pass (col_ratio>>1) regime, the collector bin's own population count
  is PRESERVED (redistributed, not destroyed 2-for-1) across the bin(s)
  its grown mass lands in; the SAME fixture demonstrates the mass
  spreading across MULTIPLE distinct destination bins from a single
  collector's own contribution.
* TestDestinationBin -- `_find_destination_bin` (add_simple_vec's own
  boundary search, NOW only `_collector_scatter`'s same-bin FALLBACK for
  sub-populations whose PDF fit itself fails) unit-tested directly;
  `test_merged_drops_land_in_the_correct_bin` sanity-checks that SOME
  destination bin gains number/mass via the full engine.
* TestPropertyTransfer -- aerosol mass (rmat/rmas) is carried into the
  destination bin(s) via `cal_ratio_mass_col_vec`'s own dimensionless-
  ratio formula and conserved exactly.
* TestNeedgiveRepair -- `_needgive_repair`'s within-group proportional
  borrow, unit-tested directly against G4 SS6's `cal_needgive` formula.
* TestConstantKernelClosedForm -- a "tractable" analytic check: a large
  timestep saturating collection (`N_col` clamped to `con_j`) has a
  closed-form total-sweep-out result, hand-verified.
* TestIbreakHook -- `ibreak=True` is a clear NotImplementedError pointing
  at Task 5, not a silent no-op.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import bin_grid, coalescence
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers -- mirrors test_vapor_deposition.py's own
# _liquid_state/_zero_diag/_thermo_state conventions (same package, same
# LiquidState/LiquidDiag/ThermoState contracts).
# ---------------------------------------------------------------------------

NBINS = 40
NBIN_H = 20
BINB = bin_grid.make_bin_grid("liquid", NBINS, nbin_h=NBIN_H).binb


@pytest.fixture(scope="module")
def real_luts() -> AmpsLuts:
    return load_luts()


def _config() -> AmpsConfig:
    return AmpsConfig.cloudlab()


def _liquid_state(
    bins: dict[int, tuple[float, float, float, float]], nbins: int = NBINS
) -> LiquidState:
    """`bins`: {bin_index: (rmt, rcon, rmat, rmas)}, single column,
    PER-VOLUME (con/mass already multiplied by den -- see
    core/coalescence.py's PER-VOLUME contract note); every other bin
    starts at zero."""
    lp = LiquidPPV
    values = np.zeros((len(LiquidState.PROPS), nbins, 1, 1), dtype=np.float64)
    for b, (rmt, rcon, rmat, rmas) in bins.items():
        values[lp.rmt_q.py_idx, b, 0, 0] = rmt
        values[lp.rcon_q.py_idx, b, 0, 0] = rcon
        values[lp.rmat_q.py_idx, b, 0, 0] = rmat
        values[lp.rmas_q.py_idx, b, 0, 0] = rmas
    return LiquidState(values=values)


def _zero_diag(nbins: int = NBINS, npoints: int = 1) -> LiquidDiag:
    z = np.zeros((nbins, npoints))
    return LiquidDiag(
        mean_mass=z.copy(),
        length=z.copy(),
        a_len=z.copy(),
        c_len=z.copy(),
        density=np.ones((nbins, npoints)),
        terminal_velocity=z.copy(),
        capacitance=z.copy(),
        ventilation_fv=np.ones((nbins, npoints)),
        ventilation_fh=np.ones((nbins, npoints)),
        ventilation_fkn=np.ones((nbins, npoints)),
        vapdep_coef1=z.copy(),
        vapdep_coef2=z.copy(),
        nre=z.copy(),
    )


def _diag_for(bins: dict[int, tuple[float, float, float]], nbins: int = NBINS) -> LiquidDiag:
    """`bins`: {bin_index: (mean_mass, length_cm, terminal_velocity_cm_s)}.
    `nre` is derived with a simple, realistic placeholder formula (not
    itself under test here -- `TestBilinearGather` in
    test_collision_kernel.py already covers the drpdrp gather's own Nre
    handling in depth); every other bin stays at `_zero_diag`'s inactive
    defaults."""
    diag = _zero_diag(nbins)
    for b, (mean_mass, length, vtm) in bins.items():
        diag.mean_mass[b, 0] = mean_mass
        diag.length[b, 0] = length
        diag.terminal_velocity[b, 0] = vtm
        diag.nre[b, 0] = max(length * vtm * 1.2e-3 / 1.8e-4, 1.0)  # realistic-magnitude Nre
    return diag


def _thermo_state(*, t: float = 280.0) -> ThermoState:
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: 1.0e6,
        ThermoProp.tv: t,
        ThermoProp.thv: t,
        ThermoProp.piv: 0.0,
        ThermoProp.pbv: 0.0,
        ThermoProp.moist_denv: 1.2e-3,
        ThermoProp.qvv: 1.0e-2,
        ThermoProp.thetav: t,
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]
    return ThermoState(values=values)


def _total_number(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rcon_q.py_idx, :, 0, 0].sum())


def _total_mass(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rmt_q.py_idx, :, 0, 0].sum())


def _total_aero_total(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rmat_q.py_idx, :, 0, 0].sum())


def _total_aero_soluble(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rmas_q.py_idx, :, 0, 0].sum())


# Two real rain bins whose mean masses sum into a THIRD, distinct bin --
# see the task report for the numeric derivation (binb[19]=2.859e-8,
# binb[20]=6.545e-8, binb[21]=1.449e-7, binb[22]=3.208e-7 cm-grid-derived
# masses in g).
BIN_J = 19  # collectee, smaller bin, mean_mass=5e-8 g (in (binb[19],binb[20]])
BIN_I = 20  # collector, larger bin, mean_mass=1e-7 g (in (binb[20],binb[21]])
MEAN_MASS_J = 5.0e-8
MEAN_MASS_I = 1.0e-7
MERGED_MASS = MEAN_MASS_I + MEAN_MASS_J  # 1.5e-7 g -> lands in bin 21, (binb[21],binb[22]]
DEST_BIN = 21
LEN_J = 0.0456  # cm, ~diameter of a 5e-8 g water sphere
LEN_I = 0.0577  # cm, ~diameter of a 1e-7 g water sphere
VTM_J = 150.0  # cm/s
VTM_I = 300.0  # cm/s -- strictly faster than VTM_J so KC(I,J)>0


def _two_bin_diag() -> LiquidDiag:
    return _diag_for({BIN_J: (MEAN_MASS_J, LEN_J, VTM_J), BIN_I: (MEAN_MASS_I, LEN_I, VTM_I)})


def _two_bin_liquid(con_i: float = 100.0, con_j: float = 500.0) -> LiquidState:
    return _liquid_state(
        {
            BIN_J: (MEAN_MASS_J * con_j, con_j, 0.0, 0.0),
            BIN_I: (MEAN_MASS_I * con_i, con_i, 0.0, 0.0),
        }
    )


# ---------------------------------------------------------------------------
# TestActiveBinsAndSelfCollection
# ---------------------------------------------------------------------------


class TestActiveBinsAndSelfCollection:
    def test_all_zero_state_is_identity(self, real_luts):
        """No active bins anywhere -- coalesce_rain must be a no-op (H1
        SS2's icond1 mask leaves every pair inactive; nothing to scatter
        or deplete)."""
        liquid = _liquid_state({})
        diag = _zero_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)

        assert np.array_equal(out.values, liquid.values)

    def test_self_collection_is_a_no_op(self, real_luts):
        """A SINGLE active bin (i==j is the only possible pair) must
        leave the state PHYSICALLY unchanged: KC(i,i)=E_c*(vtm_i-vtm_i)*
        A_c*con_i*dt=0 identically (H1 SS2's kernel-assembly formula,
        mod_amps_core.F90:16294-16298) since diag_i and diag_j are the
        SAME data at the SAME bin -- so N_col(i,i)=0, and the pair
        contributes nothing to either depletion or the scatter, for ANY
        con/mass/dt magnitude. NOT bit-exact (post-review): even with no
        collision activity, the bin's entire population still round-trips
        through the "unfortunate" sub-population's own PDF fit
        (`_linear_fit`) and multi-bin scatter (`_gather_remap`) --
        machine-epsilon-level (~1e-14 absolute here) floating-point noise
        from that round trip is expected and within the design spec's own
        ~1e-10 rtol per-call validation tolerance; `np.array_equal` (bit-
        exact) was the right check for the single-bin-scatter model this
        replaced, not for the real multi-bin PDF engine."""
        b = 20
        diag = _diag_for({b: (MEAN_MASS_I, LEN_I, VTM_I)})
        liquid = _liquid_state({b: (MEAN_MASS_I * 500.0, 500.0, 0.0, 0.0)})
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=5.0, luts=real_luts)

        assert out.values == pytest.approx(liquid.values, rel=1e-9, abs=1e-20)

    def test_below_icolbin_min_bin_cannot_act_as_collector(self, real_luts):
        """A haze bin far too small to be a collector (H1 SS1.6's own
        `icolbin_min` floor, G4 mod_amps_core.F90:1518-1523,
        `amin_mass=4.188790e-12`) must not deplete ITS OWN population as
        a collector against an even-tinier collectee -- but the physical
        KC sign selection (only vtm_i>vtm_j pairs collect) already
        guards most such cases; this test targets the loop-bound directly
        via `_icolbin_min` unit-level, not the full engine (that would
        require an artificially reordered vtm profile the physical
        kernel does not produce -- see TestActiveBinsAndSelfCollection's
        module-level companion unit test right below)."""
        assert coalescence._icolbin_min(BINB) >= 0

    def test_icolbin_min_matches_hand_computed_fortran_loop(self):
        """Direct transcription check of H1 SS1.6's verbatim loop against
        `_icolbin_min`, using a small SYNTHETIC binb (not the real 40-bin
        grid) so the expected index is exactly hand-computable: binb=
        [0,1,2,3,4,5]*1e-12 (g); amin_mass=4.18879e-12 -- Fortran:
        icolbin_min=1 initially; i=1: binb(2)=python binb[1]=1e-12<
        amin_mass -> icolbin_min=1; i=2: binb(3)=binb[2]=2e-12<amin_mass
        -> icolbin_min=2; i=3: binb(4)=binb[3]=3e-12<amin_mass ->
        icolbin_min=3; i=4: binb(5)=binb[4]=4e-12<amin_mass(4.18879e-12)
        -> icolbin_min=4; i=5: binb(6)=binb[5]=5e-12>=amin_mass -> no
        update. Final icolbin_min=4 (1-based) -> 3 (0-based)."""
        binb = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) * 1.0e-12
        assert coalescence._icolbin_min(binb) == 3


# ---------------------------------------------------------------------------
# TestTwoBinCollection -- the dispatch's headline requirement.
# ---------------------------------------------------------------------------


class TestTwoBinCollection:
    def test_number_decreases_mass_conserved_exactly(self, real_luts):
        """con_i=100, con_j=500 cm^-3 (dispatch's explicit "~500" ask),
        dt=2s: a real (non-degenerate) physical kernel pass. Total number
        must strictly DECREASE (collection is a 2-parents-in/1-child-out
        process, H1's own "M2b codegen notes" simplification); total mass
        (rmt_q, summed over ALL 40 bins) must be conserved to 1e-12
        relative -- NOT merely close, per the dispatch's explicit
        "assert mass conserves EXACTLY (to 1e-12) -- do NOT let a
        fixture cancel a basis error" instruction."""
        liquid = _two_bin_liquid(con_i=100.0, con_j=500.0)
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        n0 = _total_number(liquid)
        m0 = _total_mass(liquid)
        assert n0 == pytest.approx(600.0)

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)

        n1 = _total_number(out)
        m1 = _total_mass(out)

        assert n1 < n0, f"expected number to decrease from collection: {n0} -> {n1}"
        assert m1 == pytest.approx(m0, rel=1e-12, abs=1e-30), f"mass not conserved: {m0} -> {m1}"

    def test_no_collection_when_dt_zero(self, real_luts):
        """dt=0 -> KC=0 identically (H1's own KC=E_c*(vtm_i-vtm_j)*A_c*
        con_j*dt formula) -> N_col=0 everywhere -> PHYSICAL identity (see
        test_self_collection_is_a_no_op's own docstring for why this is
        NOT bit-exact under the real multi-bin PDF engine -- both active
        bins' entire populations still round-trip through their own
        "unfortunate" sub-population PDF fit + scatter)."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=0.0, luts=real_luts)

        assert out.values == pytest.approx(liquid.values, rel=1e-9, abs=1e-20)

    def test_larger_dt_collects_more(self, real_luts):
        """Monotonicity sanity check: a bigger dt must not collect LESS
        number than a smaller one (KC is linear in dt, con_i*KC*E_coal is
        monotonically non-decreasing in dt up to the con_j clamp)."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()
        n0 = _total_number(liquid)

        out_small = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=0.5, luts=real_luts)
        out_big = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=4.0, luts=real_luts)

        depleted_small = n0 - _total_number(out_small)
        depleted_big = n0 - _total_number(out_big)
        assert depleted_big >= depleted_small


# ---------------------------------------------------------------------------
# TestCollectorPreservation -- the code review's own explicit requirement:
# "a collector particle absorbs MULTIPLE collectees/timestep (H-pass,
# collector count preserved, mean mass grows) and spreads collected mass
# across MULTIPLE destination bins via the reconstructed PDF -- your
# [reverted] model destroys one i + one j per event into one bin". This
# fixture (con_i=50, con_j=2000 cm^-3, dt=5s) drives col_ratio(BIN_I,
# BIN_J) well above 10 (the H-pass's own ">10, ndrop_B=col_ratio itself"
# branch, G4 SS1.6/mod_amps_core.F90:2213-2222) -- EVERY sub-population
# of BIN_I's own collector role (fortunate AND unfortunate alike, H-pass
# distributes uniformly) gains the SAME enormous mass increment, so
# BIN_I's own bin ends up with con=0 (its ENTIRE population moved to a
# heavier bin) -- but, per the reviewer's own framing, MOVED, not
# DESTROYED: verified directly below by summing the gained number across
# every destination bin and comparing to BIN_I's own ORIGINAL con.
# ---------------------------------------------------------------------------


class TestCollectorPreservation:
    def test_collector_number_preserved_across_h_pass_growth(self, real_luts):
        """BIN_I's own original population (con_i=50) must reappear,
        in FULL, as gained number SOMEWHERE in the output -- not
        annihilated 2-for-1 the way the reverted single-bin model did
        (that model would have shown roughly `con_i - n_events` surviving,
        a materially SMALLER number for a saturating, high-col_ratio
        scenario like this one). `left_N(BIN_I)==con_i` here (BIN_I is
        not itself depleted as anyone's collectee in this 2-active-bin
        fixture), so the "population-preservation identity"
        (`_collector_scatter`'s own docstring: `sum_k(Np_sub(k))==
        left_n_i` exactly) predicts the gained-number total should equal
        `con_i` to within the PDF-fit/scatter round-trip's own numerical
        precision (NOT bit-exact, see module docstring) -- checked at
        `rel=1e-6` here (comfortably looser than that round-trip's own
        ~1e-10-1e-14 noise floor, tight enough to catch a real
        destroy-instead-of-preserve regression, which would be off by
        order-1, not order-1e-6)."""
        con_i, con_j = 50.0, 2000.0
        liquid = _two_bin_liquid(con_i=con_i, con_j=con_j)
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=5.0, luts=real_luts)

        con_after = out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0]
        gained_elsewhere = con_after.copy()
        gained_elsewhere[BIN_I] = 0.0  # exclude BIN_I's own (post-collection) slot
        gained_elsewhere[BIN_J] = 0.0  # exclude the collectee's own surviving remainder
        total_gained = float(gained_elsewhere.sum())

        assert total_gained == pytest.approx(con_i, rel=1e-6)
        # BIN_I's own bin: fully vacated in THIS saturating fixture (the
        # entire population moved to a heavier bin, per the docstring
        # above) -- contrasts with a destroy-2-for-1 model, which would
        # leave SOME residual `con_i - n_events` behind in BIN_I itself
        # while ALSO not fully accounting for it elsewhere.
        assert con_after[BIN_I] == pytest.approx(0.0, abs=1.0e-6)

    def test_mass_spreads_across_multiple_destination_bins(self, real_luts):
        """The SAME fixture: BIN_I's own grown-mass population must land
        in MORE THAN ONE destination bin (the reconstructed-PDF multi-bin
        scatter, `_cal_transbin_gather_remap` reused from `core/
        vapor_deposition.py`'s M2a `cal_transbin_vec` port) -- NOT
        concentrated into a single bin the way `add_simple_vec`'s own
        single-target-bin search (now only `_collector_scatter`'s
        same-bin FALLBACK) would."""
        con_i, con_j = 50.0, 2000.0
        liquid = _two_bin_liquid(con_i=con_i, con_j=con_j)
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=5.0, luts=real_luts)

        con_after = out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0]
        rmt_after = out.values[LiquidPPV.rmt_q.py_idx, :, 0, 0]
        new_bins_with_number = [
            b for b in range(NBINS) if b not in (BIN_I, BIN_J) and con_after[b] > 1.0e-6
        ]
        new_bins_with_mass = [
            b for b in range(NBINS) if b not in (BIN_I, BIN_J) and rmt_after[b] > 1.0e-15
        ]

        assert len(new_bins_with_number) >= 2, (
            f"expected the collector's grown population to spread across >=2 NEW "
            f"destination bins; got {new_bins_with_number}"
        )
        assert set(new_bins_with_number) == set(new_bins_with_mass)
        # every gaining bin must be ABOVE the original collector bin (mass
        # only grows from collection -- destination bins must be heavier).
        assert all(b > BIN_I for b in new_bins_with_number)


# ---------------------------------------------------------------------------
# TestMultiCollectorConservation -- regression tests for TWO real,
# DISTINCT mass-conservation bugs found during hardening (both in
# `coalesce_rain`'s own iter_loop1 fixed-point section, see its inline
# comment there for the full derivation):
#
# (1) With only 2 active bins (one collector, one collectee) the per-pair
#     `N_col<=con_j` clamp (G4 SS1.2) is automatically sufficient, but
#     with 3+ active bins MULTIPLE collectors can independently claim UP
#     TO `con_j` each from the SAME collectee -- the per-pair clamp does
#     not bound the SUM. `_needgive_repair` cannot catch this (it only
#     fires on NEGATIVE bins; this bug manufactures excess mass in
#     DESTINATION bins instead). Fixed by a proportional rescale.
# (2) With 4+ active bins, a collector that itself ends up fully claimed
#     by an even bigger collector is excluded from ever running its own
#     collector_loop1 body -- but its ORIGINAL claim on smaller
#     collectees was already baked into their own surviving-population
#     accounting, a "phantom" claim that never actually executes,
#     silently vanishing mass. Fixed by iterating the exclusion
#     (used_marker) to a fixed point, releasing an excluded collector's
#     own outgoing claims each round.
#
# `test_three_active_bins...` below exercises ONLY (1) (3 active bins,
# moderate 1-2 order-of-magnitude spread -- the representative case this
# task's own dispatch cares about); `test_five_active_bins...` exercises
# BOTH (1) AND (2) (needs >=4 active bins with >=2 collectors
# over-claiming a shared, smaller bin to trigger (2) at all -- verified
# during hardening: this exact 5-bin configuration leaked 14.8% of total
# mass before the (2) fix, machine precision after).
# ---------------------------------------------------------------------------


class TestMultiCollectorConservation:
    def test_three_active_bins_conserve_mass_and_aerosol_exactly(self, real_luts):
        con18, con20, con22 = 400.0, 100.0, 30.0
        liquid = _liquid_state(
            {
                18: (2.0e-8 * con18, con18, con18 * 3.0e-11, con18 * 1.2e-11),
                20: (1.0e-7 * con20, con20, con20 * 1.0e-9, con20 * 0.4e-9),
                22: (5.0e-7 * con22, con22, con22 * 3.0e-9, con22 * 1.2e-9),
            }
        )
        diag = _diag_for(
            {18: (2.0e-8, 0.025, 60.0), 20: (1.0e-7, 0.03, 100.0), 22: (5.0e-7, 0.05, 160.0)}
        )
        thermo = _thermo_state()
        config = _config()

        m0 = _total_mass(liquid)
        aero0 = _total_aero_total(liquid)
        n0 = _total_number(liquid)

        for dt in (0.5, 2.0, 8.0):
            out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=dt, luts=real_luts)
            m1 = _total_mass(out)
            aero1 = _total_aero_total(out)
            n1 = _total_number(out)

            assert m1 == pytest.approx(m0, rel=1e-12, abs=1e-30), f"dt={dt}: mass not conserved"
            assert aero1 == pytest.approx(aero0, rel=1e-12, abs=1e-30), (
                f"dt={dt}: aerosol not conserved"
            )
            assert n1 <= n0 + 1.0e-6, f"dt={dt}: number must not increase"
            assert (out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0] >= -1.0e-9).all(), (
                f"dt={dt}: negative con"
            )
            assert (out.values[LiquidPPV.rmt_q.py_idx, :, 0, 0] >= -1.0e-9).all(), (
                f"dt={dt}: negative mass"
            )
            assert np.all(np.isfinite(out.values)), f"dt={dt}: non-finite value produced"

    def test_five_active_bins_conserve_mass_and_aerosol_exactly(self, real_luts):
        """The exact class of configuration that triggers bug (2) above
        (per this module's own docstring section): 5 active bins
        (16,18,20,22,24), con 10-400 cm^-3 (descending with bin index,
        biggest-bin-smallest-count -- realistic for a rain spectrum),
        vtm increasing with bin size so larger bins are always the
        collectors. `mean_mass` per bin taken from the REAL bin-grid
        midpoint (`BINB`) rather than hand-picked, so the fixture
        represents genuine bin-grid masses, not synthetic round numbers.
        Before the (2) fix this leaked 14.8% of total mass at dt=2s;
        after, mass and aerosol mass conserve to 1e-12 (measured: machine
        precision, ~1e-16) at every dt tried, including the saturating
        regime (dt=8, where results stop changing with further dt
        increases -- the `TestConstantKernelClosedForm` saturation
        pattern)."""

        def mean_mass_of(bin_index: int) -> float:
            return 0.5 * (BINB[bin_index] + BINB[bin_index + 1])

        bin_indices = (16, 18, 20, 22, 24)
        cons = {16: 400.0, 18: 200.0, 20: 80.0, 22: 30.0, 24: 10.0}
        bins = {}
        diag_bins = {}
        for idx, b in enumerate(bin_indices):
            con_b = cons[b]
            m = mean_mass_of(b)
            bins[b] = (m * con_b, con_b, m * con_b * 1.0e-2, m * con_b * 0.4e-2)
            length = (6.0 * m / np.pi) ** (1.0 / 3.0)  # cm, density~1 g/cm^3 sphere
            vtm = 50.0 + 100.0 * idx  # cm/s, strictly increasing with bin size
            diag_bins[b] = (m, length, vtm)
        liquid = _liquid_state(bins)
        diag = _diag_for(diag_bins)
        thermo = _thermo_state()
        config = _config()

        m0 = _total_mass(liquid)
        aero0 = _total_aero_total(liquid)
        n0 = _total_number(liquid)

        for dt in (0.5, 2.0, 8.0):
            out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=dt, luts=real_luts)
            m1 = _total_mass(out)
            aero1 = _total_aero_total(out)
            n1 = _total_number(out)

            assert m1 == pytest.approx(m0, rel=1e-12, abs=1e-30), f"dt={dt}: mass not conserved"
            assert aero1 == pytest.approx(aero0, rel=1e-12, abs=1e-30), (
                f"dt={dt}: aerosol not conserved"
            )
            assert n1 <= n0 + 1.0e-6, f"dt={dt}: number must not increase"
            assert (out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0] >= -1.0e-9).all(), (
                f"dt={dt}: negative con"
            )
            assert (out.values[LiquidPPV.rmt_q.py_idx, :, 0, 0] >= -1.0e-9).all(), (
                f"dt={dt}: negative mass"
            )
            assert np.all(np.isfinite(out.values)), f"dt={dt}: non-finite value produced"


# ---------------------------------------------------------------------------
# TestDestinationBin
# ---------------------------------------------------------------------------


class TestDestinationBin:
    def test_merged_drops_land_in_the_correct_bin(self, real_luts):
        """The add_simple_vec-style boundary search (H1 SS2, G4 SS4.1
        verbatim: `binb(j)<Mp/Np<=binb(j+1)`) on `m_i+m_j=1.5e-7` g must
        place the newly-formed number/mass in bin 21 (0-based) --
        DIFFERENT from both parent bins (19, 20) -- see the numeric
        derivation in this module's own top-of-file comment."""
        liquid = _two_bin_liquid(con_i=100.0, con_j=500.0)
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)

        gained_number = out.values[LiquidPPV.rcon_q.py_idx, DEST_BIN, 0, 0]
        gained_mass = out.values[LiquidPPV.rmt_q.py_idx, DEST_BIN, 0, 0]
        assert gained_number > 0.0
        assert gained_mass > 0.0
        # the merged bin's own effective mean mass must be close to
        # MERGED_MASS (not exactly, per add_simple_vec's own Np boundary-
        # safety clamp, H1 SS2 mod_amps_core.F90:15505-15528 -- but well
        # within the bin's own boundaries either way).
        assert BINB[DEST_BIN] < gained_mass / gained_number <= BINB[DEST_BIN + 1]

    def test_destination_search_matches_hand_computed_bin(self):
        """`_find_destination_bin` unit-tested directly against
        add_simple_vec's own boundary formula (G4 SS4.1 verbatim), with a
        small SYNTHETIC binb so the expected index is exactly
        hand-computable: binb=[0,1,2,3,4] (bin b spans (binb[b],
        binb[b+1]]) -- x=0.5 -> bin 0 ((0,1]); x=1.0 -> bin 0
        (boundary-inclusive upper edge, still in (0,1]); x=3.5 -> bin 3
        ((3,4]); x=100 (above top) -> clamps to the LAST bin (3); x=-1
        (at/below bottom) -> clamps to the FIRST bin (0)."""
        binb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = np.array([0.5, 1.0, 3.5, 100.0, -1.0])

        dest = coalescence._find_destination_bin(binb, x)

        assert list(dest) == [0, 0, 3, 3, 0]


# ---------------------------------------------------------------------------
# TestPropertyTransfer -- aerosol mass (rmat/rmas) carried into the merged
# bin, cal_ratio_mass_col_vec's simplified per-pair analogue (H1 SS1.6f/
# G4 SS5).
# ---------------------------------------------------------------------------


class TestPropertyTransfer:
    def test_aerosol_mass_conserved_and_transferred_to_destination(self, real_luts):
        """Both parent bins carry aerosol mass (rmat_q/rmas_q); after
        coalescence the TOTAL aerosol mass (summed over all 40 bins) must
        be conserved exactly, and the destination bin must have gained a
        nonzero share (the merged drop inherits BOTH parents' aerosol
        content, per-pair-additive -- see core/coalescence.py's module
        docstring for the exact formula)."""
        con_i, con_j = 100.0, 500.0
        aero_tot_i, aero_sol_i = 1.0e-9 * con_i, 0.5e-9 * con_i
        aero_tot_j, aero_sol_j = 2.0e-10 * con_j, 1.0e-10 * con_j
        liquid = _liquid_state(
            {
                BIN_J: (MEAN_MASS_J * con_j, con_j, aero_tot_j, aero_sol_j),
                BIN_I: (MEAN_MASS_I * con_i, con_i, aero_tot_i, aero_sol_i),
            }
        )
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        aero_tot0 = _total_aero_total(liquid)
        aero_sol0 = _total_aero_soluble(liquid)

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)

        assert _total_aero_total(out) == pytest.approx(aero_tot0, rel=1e-12, abs=1e-30)
        assert _total_aero_soluble(out) == pytest.approx(aero_sol0, rel=1e-12, abs=1e-30)
        assert out.values[LiquidPPV.rmat_q.py_idx, DEST_BIN, 0, 0] > 0.0
        assert out.values[LiquidPPV.rmas_q.py_idx, DEST_BIN, 0, 0] > 0.0


# ---------------------------------------------------------------------------
# TestNeedgiveRepair -- G4 SS6's cal_needgive, within-group branch.
# ---------------------------------------------------------------------------


class TestNeedgiveRepair:
    def test_negative_bin_zeroed_positive_bins_scaled_proportionally(self):
        """3 bins, single column: mass_tot=[-10,60,40] -> total_pos=100,
        total_neg=10, m_t1=90>=min_mt -> scale=(100-10)/100=0.9 (G4 SS6's
        own sequential-vs-single-pass algebraic equivalence, derived in
        the task report). Expected: bin0 zeroed, bin1/bin2 scaled by 0.9
        -> [0, 54, 36]; TOTAL mass conserved exactly (90 before repair's
        own accounting == 90 after: -10+60+40=90, 0+54+36=90)."""
        number = np.array([[1.0], [6.0], [4.0]])
        mass_tot = np.array([[-10.0], [60.0], [40.0]])

        new_number, new_mass, (new_extra,) = coalescence._needgive_repair(
            number, mass_tot, mass_tot.copy()
        )

        assert new_mass[:, 0] == pytest.approx([0.0, 54.0, 36.0])
        # total mass EXACTLY conserved by the repair itself (the negative
        # bin's own deficit is subtracted proportionally FROM the
        # positive pool, not discarded): -10+60+40 == 0+54+36 == 90.
        assert float(new_mass.sum()) == pytest.approx(float(mass_tot.sum()))
        # number/extra scaled by the SAME per-bin modrg as mass_tot.
        assert new_number[:, 0] == pytest.approx([0.0, 5.4, 3.6])
        assert new_extra[:, 0] == pytest.approx([0.0, 54.0, 36.0])

    def test_no_negative_bins_is_a_no_op(self):
        number = np.array([[1.0], [2.0], [3.0]])
        mass_tot = np.array([[10.0], [20.0], [30.0]])

        new_number, new_mass, _extra = coalescence._needgive_repair(number, mass_tot)

        assert np.array_equal(new_number, number)
        assert np.array_equal(new_mass, mass_tot)

    def test_columns_are_independent(self):
        """Two columns, only the first has a negative bin -- the second
        column must be untouched (G4 SS6's own per-`ngrid` (per-column)
        scoping). Column 0: mass_tot=[-10,60] -> total_pos=60,
        total_neg=10 -> scale=(60-10)/60=0.8333 -> [0, 50]."""
        number = np.array([[1.0, 1.0], [6.0, 2.0]])
        mass_tot = np.array([[-10.0, 10.0], [60.0, 20.0]])

        _new_number, new_mass, _extra = coalescence._needgive_repair(number, mass_tot)

        assert new_mass[:, 1] == pytest.approx([10.0, 20.0])
        assert new_mass[:, 0] == pytest.approx([0.0, 50.0])


# ---------------------------------------------------------------------------
# TestConstantKernelClosedForm -- "analytic moment check if tractable"
# (dispatch). A literal Golovin-kernel analytic solution is NOT tractable
# through the REAL drpdrp/Low-List physical kernel without faking a whole
# synthetic LUT (out of scope for this port -- see report); the tractable
# substitute used here is the SATURATED-COLLECTION limit: a large enough
# dt drives N_col(i,j) to its OWN clamp (`N_col=min(con_i*KC,con_j)` ->
# con_j, H1 SS1.2), so collectee bin j's ENTIRE population is (up to
# E_coal<1 partial coalescence) sweept out -- a closed-form result
# independent of the exact kernel magnitude once saturation is reached.
# ---------------------------------------------------------------------------


class TestConstantKernelClosedForm:
    def test_saturated_collection_sweeps_out_entire_collectee_bin(self, real_luts):
        """A very large dt saturates N_col(i,j) at con_j (H1 SS1.2's own
        clamp) regardless of the physical KC magnitude -- so, since
        E_coal is a coalescence PROBABILITY (bounded in [0,1]), the
        number of coalescence events n_ij=N_col*E_coal is bounded by
        con_j itself: closed form UPPER bound `n_ij <= con_j`, hence bin
        j's post-step number can never go negative and the total number
        removed from the 2-bin system is bounded by `2*con_j` (each event
        removes one i-drop AND one j-drop). This pins the SATURATION
        REGIME behavior in closed form without needing the exact kernel
        value."""
        con_i, con_j = 100.0, 500.0
        liquid = _two_bin_liquid(con_i=con_i, con_j=con_j)
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        huge_dt = 1.0e6  # saturates N_col(i,j) at con_j for any nonzero KC
        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=huge_dt, luts=real_luts)

        n1 = _total_number(out)
        n0 = _total_number(liquid)
        depleted = n0 - n1
        # closed-form bound: at most con_j coalescence events (E_coal<=1),
        # each removing 2 and adding <=1, so total depletion <= 2*con_j;
        # AND the merged bin cannot gain more than con_j drops-worth.
        assert 0.0 <= depleted <= 2.0 * con_j + 1.0e-6
        assert n1 >= 0.0
        # bin j (the collectee, con_j=500) itself must not go negative --
        # this is exactly what the con_j clamp (H1 SS1.2) plus the
        # needgive repair (G4 SS6) guarantee.
        assert out.values[LiquidPPV.rcon_q.py_idx, BIN_J, 0, 0] >= 0.0
        assert out.values[LiquidPPV.rmt_q.py_idx, BIN_J, 0, 0] >= 0.0


# ---------------------------------------------------------------------------
# TestIbreakHook -- ibreak=True validation (M2b Task 5: the runtime is now
# implemented, see test_breakup.py::TestCoalesceRainIbreak for the actual
# ibreak=1 physics/conservation tests). This module keeps only the
# ARGUMENT-VALIDATION contract tests (missing config/tables must raise a
# clear ValueError, not silently no-op or crash obscurely) -- the ibreak=0
# vs ibreak=1 comparison and fragment-table wiring itself belongs in
# test_breakup.py, which owns `core/breakup.py`.
# ---------------------------------------------------------------------------


class TestIbreakHook:
    def test_ibreak_true_without_config_flag_raises_value_error(self, real_luts):
        """`ibreak=True` requires `config.rain_collisional_breakup=True`
        (core/breakup.py's own module docstring) -- a config/caller
        mismatch must raise clearly, not silently ignore `ibreak`."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = dataclasses.replace(AmpsConfig.cloudlab(), rain_collisional_breakup=False)

        with pytest.raises(ValueError, match="rain_collisional_breakup"):
            coalescence.coalesce_rain(
                liquid, diag, thermo, config, dt=2.0, luts=real_luts, ibreak=True
            )

    def test_ibreak_true_without_breakup_tables_raises_value_error(self, real_luts):
        """`ibreak=True` requires a `breakup_tables` argument (M2b Task 6's
        `BreakupFragmentTables`) -- omitting it must raise clearly."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()
        assert config.rain_collisional_breakup is True

        with pytest.raises(ValueError, match="breakup_tables"):
            coalescence.coalesce_rain(
                liquid, diag, thermo, config, dt=2.0, luts=real_luts, ibreak=True
            )

    def test_ibreak_false_is_the_default_without_breakup_tables(self, real_luts):
        """`ibreak` defaults to `False` -- omitting it entirely must NOT
        require `breakup_tables` or raise."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)
        assert np.all(np.isfinite(out.values))

    def test_ibreak_false_is_the_default(self, real_luts):
        """Pre-existing baseline test (predates M2b Task 5): calling
        `coalesce_rain` without `ibreak` at all must be identical to
        passing `ibreak=False` explicitly."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out_default = coalescence.coalesce_rain(
            liquid, diag, thermo, config, dt=2.0, luts=real_luts
        )
        out_explicit = coalescence.coalesce_rain(
            liquid, diag, thermo, config, dt=2.0, luts=real_luts, ibreak=False
        )
        assert np.array_equal(out_default.values, out_explicit.values)


# ---------------------------------------------------------------------------
# T7 (M2b Task 7): two degenerate-bin corruption modes a real-dump
# per-call validation surfaced -- neither triggered by this module's own
# pre-existing synthetic fixtures (all "real magnitude", per the module
# docstring's own note, but never a floating-point-noise-level `con` paired
# with a normal-magnitude leftover `mass_tot`). Both fixed in `coalesce_
# rain` itself (`active_collector_base`'s `mean_mass<=binb[-1]` ceiling,
# `counter_active`'s column-level `counter(n)>0` gate -- see the module
# docstring's own "T7" section). Reproduced here directly (no external dump
# needed) so a regression is caught by the ordinary unit-test suite, not
# only the external-data-gated replay harness (`tests/amps/integration_
# tests/test_warm_replay.py`).
# ---------------------------------------------------------------------------


class TestT7DegenerateBinGuards:
    # Real-dump-observed magnitudes (M2b Task 7 report): a bin whose own
    # `con` is a floating-point-noise-level residual (comfortably above
    # the bare `_ACTIVE_FLOOR=1e-30`, but physically negligible) paired
    # with an ordinary-magnitude leftover `mass_tot` -- `mean_mass =
    # mass_tot/con ~= 5.0e6 g`, ~1e7x beyond `binb[-1]~0.52g` (the grid's
    # own largest representable mean-drop mass).
    DEGENERATE_CON = 7.08e-14
    DEGENERATE_MASS = 3.54e-7

    def test_isolated_degenerate_bin_stays_inert_counter_gate(self, real_luts):
        """T7 fix 1 (`counter(n)>0`): a column with EXACTLY ONE occupied
        bin, whose own `con` is floating-point-noise-level -- no OTHER bin
        exists to generate any real collision claim anywhere in the
        column, so `counter(n)` must stay 0 and `collector_loop1` must
        skip this bin entirely (regardless of its own absurd `mean_mass`
        individually clearing every OTHER floor). Pre-fix, this bin's
        `mean_mass` (~5.0e6g, `_diag_for` below) cleared
        `_MEAN_MASS_FLOOR` and was treated as an eligible collector;
        `_collector_scatter`'s own degenerate-fallback destination-bin
        search (`_find_destination_bin`) then clipped it to the grid's TOP
        bin (39) -- reproduced empirically against a real dump (M2b Task 7
        report): `rel_err~1e17` at bin 39, `abs_err~2.9e5`."""
        config = _config()
        bin_idx = 15
        liquid = _liquid_state({bin_idx: (self.DEGENERATE_MASS, self.DEGENERATE_CON, 0.0, 0.0)})
        diag = _diag_for({bin_idx: (self.DEGENERATE_MASS / self.DEGENERATE_CON, 1.0, 1.0)})
        thermo = _thermo_state()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=5.0, luts=real_luts)

        lp = LiquidPPV
        assert np.all(np.isfinite(out.values))
        # The degenerate bin's own mass/con are UNCHANGED (re-added via
        # the post-loop leftover mechanism, `excluded_by_t7`) -- NOT
        # scattered to bin 39 or anywhere else.
        assert out.values[lp.rmt_q.py_idx, bin_idx, 0, 0] == pytest.approx(
            self.DEGENERATE_MASS, rel=1e-12
        )
        assert out.values[lp.rcon_q.py_idx, bin_idx, 0, 0] == pytest.approx(
            self.DEGENERATE_CON, rel=1e-12
        )
        # Bin 39 (the grid's top bin -- the pre-fix corruption's own
        # destination) stays exactly zero in every property.
        assert out.values[lp.rmt_q.py_idx, 39, 0, 0] == 0.0
        assert out.values[lp.rcon_q.py_idx, 39, 0, 0] == 0.0
        assert out.values[lp.rmas_q.py_idx, 39, 0, 0] == 0.0
        # No mass created or destroyed anywhere in the grid.
        assert out.values[lp.rmt_q.py_idx].sum() == pytest.approx(self.DEGENERATE_MASS, rel=1e-12)

    def test_mutually_degenerate_column_bounded_by_mean_mass_ceiling(self, real_luts):
        """T7 fix 2 (`mean_mass<=binb[-1]`): a column where the collector
        (`BIN_I`) and collectee (`BIN_J`) are the SAME real, known-good
        collision pair `TestTwoBinCollection` already uses (real `con`/
        `length`/`vtm`, guaranteed nonzero `KC`/`E_coal` -- see those
        module-level constants' own comments), so `counter(n)>0` fires
        genuinely (fix 1 alone does NOT exclude this pair, unlike the
        isolated-bin case above) -- EXCEPT `diag.mean_mass[BIN_I]` is
        deliberately overridden to the SAME absurd real-dump-observed
        value (`~5.0e6g`, `_diag_for`'s own hand-picked-field convention,
        matching e.g. `TestDestinationBin`/`TestNeedgiveRepair`) instead
        of the physically-consistent `MEAN_MASS_I=1.0e-7g` -- isolating
        `active_collector_base`'s OWN `mean_mass<=binb[-1]` ceiling
        specifically (only fix 2 can exclude a bin whose OWN individual
        `mean_mass` is absurd once `counter(n)>0` is already satisfied).
        Pre-fix-2 (fix 1 present, fix 2 absent), this collector is
        incorrectly treated as active and its real `con_i` population
        scatters via `mean_mass`-computed intervals FAR outside the grid
        -- an analogous real-dump scenario (an entire column of mutually-
        active-but-one-degenerate bins) reproduced a SECOND failure mode
        at a different destination bin (M2b Task 7 report)."""
        config = _config()
        con_i, con_j = 100.0, 500.0
        liquid = _liquid_state(
            {
                BIN_I: (MEAN_MASS_I * con_i, con_i, 0.0, 0.0),
                BIN_J: (MEAN_MASS_J * con_j, con_j, 0.0, 0.0),
            }
        )
        diag = _diag_for({BIN_J: (MEAN_MASS_J, LEN_J, VTM_J), BIN_I: (5.0e6, LEN_I, VTM_I)})
        thermo = _thermo_state()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)

        lp = LiquidPPV
        assert np.all(np.isfinite(out.values))
        total_con_before = con_i + con_j  # 600 -- coalescence only ever DECREASES total number
        total_mass_before = MEAN_MASS_I * con_i + MEAN_MASS_J * con_j
        assert out.values[lp.rmt_q.py_idx].sum() == pytest.approx(total_mass_before, rel=1e-9)
        # No implausible concentration anywhere in the grid -- the
        # pre-fix-2 failure mode: `mean_mass=5e6` scattered `con_i=100`
        # real drops via a shifted-boundary interval computed from that
        # absurd mean mass, spraying garbage (real-dump values observed up
        # to ~1e5-1e6 #/g, orders of magnitude beyond `total_con_before`)
        # across potentially many bins. `<=total_con_before` (a physical
        # bound: no bin can ever end up with more drops than existed
        # anywhere in the whole grid to begin with) catches that failure
        # mode while accepting the CORRECT post-fix outcome (both bins'
        # `con` legitimately UNCHANGED at 500/100 -- BIN_I is excluded as
        # a collector by the ceiling and its phantom claim on BIN_J is
        # released, so neither bin is touched at all).
        assert np.all(out.values[lp.rcon_q.py_idx] <= total_con_before + 1e-6)
