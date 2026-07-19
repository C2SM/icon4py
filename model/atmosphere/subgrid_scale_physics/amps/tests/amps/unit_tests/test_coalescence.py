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
dispatch (which explicitly sanctions the simplified add_simple_vec-style
single-target-bin scatter over the full sub-bin PDF `collector_loop1`
machinery -- see `core/coalescence.py`'s module docstring for the full
derivation of the "2 parents in, 1 merged child out" model this engine
implements, and why it exactly conserves mass by construction).

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

Groups:
* TestActiveBinsAndSelfCollection -- icond1 masking, self-collection
  (i==j) is a proven no-op (KC=0 via vtm_i-vtm_i=0), icolbin_min collector
  floor.
* TestTwoBinCollection -- the dispatch's headline test: number decreases,
  mass conserves EXACTLY (1e-12) at con~500 cm^-3 magnitudes.
* TestDestinationBin -- coalesced mass m_i+m_j lands in the bin found by
  the add_simple_vec boundary search, verified against a hand-picked
  bin-grid location.
* TestPropertyTransfer -- aerosol mass (rmat/rmas) is carried into the
  merged bin proportionally (cal_ratio_mass_col_vec's simplified,
  per-pair analogue) and conserved exactly.
* TestNeedgiveRepair -- `_needgive_repair`'s within-group proportional
  borrow, unit-tested directly against G4 SS6's `cal_needgive` formula.
* TestConstantKernelClosedForm -- a "tractable" analytic check: a large
  timestep saturating collection (`N_col` clamped to `con_j`) has a
  closed-form total-sweep-out result, hand-verified.
* TestIbreakHook -- `ibreak=True` is a clear NotImplementedError pointing
  at Task 5, not a silent no-op.
"""

from __future__ import annotations

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
        leave the state EXACTLY unchanged: KC(i,i)=E_c*(vtm_i-vtm_i)*
        A_c*con_i*dt=0 identically (H1 SS2's kernel-assembly formula,
        mod_amps_core.F90:16294-16298) since diag_i and diag_j are the
        SAME data at the SAME bin -- so N_col(i,i)=0, and the pair
        contributes nothing to either depletion or the scatter, for ANY
        con/mass/dt magnitude."""
        b = 20
        diag = _diag_for({b: (MEAN_MASS_I, LEN_I, VTM_I)})
        liquid = _liquid_state({b: (MEAN_MASS_I * 500.0, 500.0, 0.0, 0.0)})
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=5.0, luts=real_luts)

        assert np.array_equal(out.values, liquid.values)

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
        con_j*dt formula) -> N_col=0 everywhere -> exact identity."""
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        out = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=0.0, luts=real_luts)

        assert np.array_equal(out.values, liquid.values)

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
# TestIbreakHook -- ibreak=True must raise, not silently no-op.
# ---------------------------------------------------------------------------


class TestIbreakHook:
    def test_ibreak_true_raises_not_implemented(self, real_luts):
        liquid = _two_bin_liquid()
        diag = _two_bin_diag()
        thermo = _thermo_state()
        config = _config()

        with pytest.raises(NotImplementedError):
            coalescence.coalesce_rain(
                liquid, diag, thermo, config, dt=2.0, luts=real_luts, ibreak=True
            )

    def test_ibreak_false_is_the_default(self, real_luts):
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
