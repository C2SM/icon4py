# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/vapor_deposition.py (M2a Task 5): `vapor_deposition`'s
LIQUID condensation/evaporation growth + mass-space bin remap, per
docs/superpowers/facts/m2/vapor-deposition.md ("G3") and this task's own
dispatch. See `core/vapor_deposition.py`'s module docstring for every
design decision (numbered items 1-8) the tests below are constructed
around.

Groups:
* TestExcessVaporDensity -- `excess_vapor_density` vs G3's own formula,
  hand-computed.
* TestLinearReconstructionRoundTrip -- `_linear_reconstruction` +
  `_moment_integral` reproduce the exact input moments when integrated
  over the FULL shifted interval (internal consistency of the
  un-quoted-`cal_lincubprms_vec` substitute, module docstring item 3).
* TestGrowthShiftsToAdjacentBin -- supersaturated single populated bin:
  engineered so the ENTIRE mean-mass point crosses fully into the
  adjacent (upper) bin, giving an exact expected post-remap state;
  mass+number conserved to 1e-12.
* TestEvaporationShiftsDown -- subsaturated (evaporating) single
  populated bin, partial (non-total) evaporation: some mass appears in
  the adjacent LOWER bin; mass+number conserved (mass changes by exactly
  the analytically predicted growth amount) to 1e-12.
* TestRemapMassConserving -- multi-bin mix of growth/evaporation: total
  number and mass (mass exactly by the predicted growth amount) conserved
  to 1e-12; aerosol mass (rmat/rmas) exactly conserved (redistributes,
  never created/destroyed).
* TestPassthroughInactiveBins -- bins `diag.mean_mass<=0` pass through
  byte-identical.
* TestTotalEvaporation -- a bin whose entire water content evaporates
  this substep loses both its number and mass entirely (goes to 0).
* test_vapor_deposition_replay_against_m0_dump (`pytest.mark.datatest`,
  skipped) -- per-call replay vs spin-up dumps, matching
  `test_activation.py`'s own precedent (no local dumps available).
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import (
    bin_grid,
    index_maps,
    thermo,
    vapor_deposition as vd,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

P_STD = float(AmpsConst.p00)
T_STD = 280.0
QV_STD = 1.0e-2
NBINS = 40
NBIN_H = 20


def _thermo_state(*, p: float = P_STD, t: float = T_STD, qv: float = QV_STD) -> ThermoState:
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: p,
        ThermoProp.tv: t,
        ThermoProp.thv: t,
        ThermoProp.piv: 0.0,
        ThermoProp.pbv: 0.0,
        ThermoProp.moist_denv: 1.2e-3,
        ThermoProp.qvv: qv,
        ThermoProp.thetav: t,
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]
    return ThermoState(values=values)


def _liquid_state(nbins: int, bins: dict[int, tuple[float, float, float, float]]) -> LiquidState:
    """`bins`: {bin_index: (rmt, rcon, rmat, rmas)}, single column."""
    lp = index_maps.LiquidPPV
    values = np.zeros((len(LiquidState.PROPS), nbins, 1, 1), dtype=np.float64)
    for b, (rmt, rcon, rmat, rmas) in bins.items():
        values[lp.rmt_q.py_idx, b, 0, 0] = rmt
        values[lp.rcon_q.py_idx, b, 0, 0] = rcon
        values[lp.rmat_q.py_idx, b, 0, 0] = rmat
        values[lp.rmas_q.py_idx, b, 0, 0] = rmas
    return LiquidState(values=values)


def _zero_diag(nbins: int, npoints: int = 1) -> LiquidDiag:
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
    )


def _diag_with(
    nbins: int, mean_mass: dict[int, float], coef1: dict[int, float], coef2: dict[int, float]
) -> LiquidDiag:
    diag = _zero_diag(nbins)
    for b, v in mean_mass.items():
        diag.mean_mass[b, 0] = v
    for b, v in coef1.items():
        diag.vapdep_coef1[b, 0] = v
    for b, v in coef2.items():
        diag.vapdep_coef2[b, 0] = v
    return diag


BINB = bin_grid.make_bin_grid("liquid", NBINS, nbin_h=NBIN_H).binb


# ---------------------------------------------------------------------------
# excess_vapor_density vs G3's own formula.
# ---------------------------------------------------------------------------


class TestExcessVaporDensity:
    def test_matches_hand_formula(self):
        estbar, esitbar = thermo.make_esat_tables()
        token = 1
        s_v_ice = np.array([0.05])
        e_sat_ice = np.array([260.0])
        t_ambient = np.array([270.0])
        tmp_particle = np.array([269.5])
        con = np.array([1.0])
        mass_total = np.array([1.0e-6])

        result = vd.excess_vapor_density(
            token, s_v_ice, e_sat_ice, t_ambient, tmp_particle, con, mass_total, estbar, esitbar
        )

        r_v = float(AmpsConst.R_v)
        den_v_inf = (s_v_ice + 1.0) * e_sat_ice / (r_v * t_ambient)
        e_s = thermo.esat_lk(token, tmp_particle, estbar, esitbar)
        den_sv_sfc = e_s / (r_v * tmp_particle)
        expected = den_v_inf - den_sv_sfc

        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_zero_for_empty_bin(self):
        estbar, esitbar = thermo.make_esat_tables()
        result = vd.excess_vapor_density(
            1,
            np.array([0.05]),
            np.array([260.0]),
            np.array([270.0]),
            np.array([269.5]),
            con=np.array([0.0]),
            mass_total=np.array([0.0]),
            estbar=estbar,
            esitbar=esitbar,
        )
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# _linear_reconstruction + _moment_integral round-trip.
# ---------------------------------------------------------------------------


class TestLinearReconstructionRoundTrip:
    def test_reproduces_input_moments_over_full_interval(self):
        bd1 = np.array([1.0e-7, 1.0e-10, 1.0e-5])
        bd2 = np.array([5.0e-6, 1.0e-8, 8.0e-5])
        number = np.array([2.5, 0.001, 100.0])
        # mean=mass/number centered in [bd1,bd2]: keeps the reconstructed
        # density non-negative throughout (see
        # test_subinterval_moments_sum_to_full_interval's own docstring).
        mass = number * 0.5 * (bd1 + bd2)

        a0, a1, well_posed = vd._linear_reconstruction(number, mass, bd1, bd2)
        assert np.all(well_posed)

        trans_n, trans_m = vd._moment_integral(a0, a1, bd1, bd2)
        np.testing.assert_allclose(trans_n, number, rtol=1e-11)
        np.testing.assert_allclose(trans_m, mass, rtol=1e-11)

    def test_degenerate_zero_width_flagged_not_well_posed(self):
        _a0, _a1, well_posed = vd._linear_reconstruction(
            np.array([1.0]), np.array([1.0e-6]), np.array([5.0e-6]), np.array([5.0e-6])
        )
        assert not well_posed[0]

    def test_negative_density_from_skewed_moments_flagged_not_well_posed(self):
        """A moment pair whose mean (mass/number) sits close to `bd1`
        forces a linear fit that goes NEGATIVE near `bd2` -- exactly the
        `cal_lincubprms_vec` "n(x'_2)<0" case this module's own
        `_linear_reconstruction` does not reproduce (module docstring
        item 3). Must be flagged `well_posed=False` (not silently return
        a locally-negative density that `_moment_integral`'s per-piece
        clamp would later un-conserve, per the TDD failure this check was
        added to fix)."""
        number, mass = 4.0, 6.0e-6  # mean=1.5e-6, close to bd1=1e-6
        bd1, bd2 = 1.0e-6, 9.0e-6
        a0, a1, well_posed = vd._linear_reconstruction(
            np.array([number]), np.array([mass]), np.array([bd1]), np.array([bd2])
        )
        assert not well_posed[0]
        assert a0[0] == 0.0
        assert a1[0] == 0.0

    def test_subinterval_moments_sum_to_full_interval(self):
        """Splitting [bd1,bd2] at an interior point and summing the two
        piece integrals must reproduce the full-interval moments -- the
        mathematical property the gather-remap's dense i x ibx overlap
        sum relies on. Mean (mass/number = 5e-6) is well-centered in
        [bd1,bd2] so the reconstructed linear density stays non-negative
        throughout (module docstring item 3's un-reproduced "negative
        density" fallback is a separate, documented limitation for
        strongly-skewed moments -- not exercised by this identity check)."""
        number, mass = 4.0, 4.0 * 5.0e-6
        bd1, bd2 = 1.0e-6, 9.0e-6
        a0, a1, well_posed = vd._linear_reconstruction(
            np.array([number]), np.array([mass]), np.array([bd1]), np.array([bd2])
        )
        assert well_posed[0]
        mid = np.array([4.5e-6])
        n1, m1 = vd._moment_integral(a0, a1, np.array([bd1]), mid)
        n2, m2 = vd._moment_integral(a0, a1, mid, np.array([bd2]))
        assert (n1 + n2)[0] == pytest.approx(number, rel=1e-11)
        assert (m1 + m2)[0] == pytest.approx(mass, rel=1e-11)


# ---------------------------------------------------------------------------
# End-to-end: a skewed mean_mass (close to its bin's own lower edge) still
# conserves exactly through vapor_deposition_liquid's same-bin-passthrough
# fallback, even though it would defeat a naive (un-guarded) gather -- see
# core/vapor_deposition.py's _linear_reconstruction docstring.
# ---------------------------------------------------------------------------


class TestSkewedMeanMassFallbackConserves:
    def test_skewed_growth_still_conserves_exactly(self):
        b = 25
        # 5% into the bin from its lower edge -- skewed enough to defeat
        # the naive linear reconstruction after growth (independently
        # confirmed via _linear_reconstruction directly during this
        # task's own TDD).
        mean_mass0 = BINB[b] + 0.05 * (BINB[b + 1] - BINB[b])
        n0 = 1.0
        m0 = n0 * mean_mass0
        liquid = _liquid_state(NBINS, {b: (m0, n0, 0.0, 0.0)})
        dt_vp = 1.0
        d_target = 0.5 * (BINB[b + 1] - BINB[b])
        diag = _diag_with(NBINS, mean_mass={b: mean_mass0}, coef1={}, coef2={b: d_target / dt_vp})
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        # Sanity: this scenario genuinely hits the negative-density
        # fallback (not a vacuous test).
        r_lo = (BINB[b] / mean_mass0) ** (1.0 / 3.0)
        r_hi = (BINB[b + 1] / mean_mass0) ** (1.0 / 3.0)
        shifted_lo = max(0.0, BINB[b] + d_target * r_lo)
        shifted_hi = max(0.0, BINB[b + 1] + d_target * r_hi)
        _, _, well_posed = vd._linear_reconstruction(
            np.array([n0]),
            np.array([m0 + n0 * d_target]),
            np.array([shifted_lo]),
            np.array([shifted_hi]),
        )
        assert not well_posed[0]

        result = vd.vapor_deposition_liquid(liquid, thermo_state, config, dt_vp, diag)

        lp = index_maps.LiquidPPV
        new_con = result.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = result.values[lp.rmt_q.py_idx, :, 0, 0]
        expected_dmcon = n0 * d_target
        assert new_con.sum() == pytest.approx(n0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(m0 + expected_dmcon, abs=1e-12, rel=1e-12)
        # Same-bin fallback: mass/number land back in the ORIGINAL bin b,
        # not redistributed to b+1 (documented, conserving simplification).
        assert new_con[b] == pytest.approx(n0, rel=1e-11)
        assert new_mass[b] == pytest.approx(m0 + expected_dmcon, rel=1e-11)


# ---------------------------------------------------------------------------
# Supersaturated single-bin growth -- engineered full clear-out into the
# adjacent (upper) bin.
# ---------------------------------------------------------------------------


class TestGrowthShiftsToAdjacentBin:
    B = 25
    N0 = 1.0

    def _mean_mass0(self) -> float:
        return float(np.sqrt(BINB[self.B] * BINB[self.B + 1]))

    def _d_mean_mass_full_shift(self) -> float:
        """Chosen so shifted_lo/hi (per vapor_deposition_liquid's own
        boundary-shift formula) land entirely within
        (binb[B+1], binb[B+2]) -- i.e. bin B's ENTIRE population moves to
        bin B+1, none left behind, none overflowing into B+2."""
        mean_mass0 = self._mean_mass0()
        r_lo = (BINB[self.B] / mean_mass0) ** (1.0 / 3.0)
        r_hi = (BINB[self.B + 1] / mean_mass0) ** (1.0 / 3.0)
        d_cross = (BINB[self.B + 1] - BINB[self.B]) / r_lo
        d_overflow = (BINB[self.B + 2] - BINB[self.B + 1]) / r_hi
        return 0.5 * (d_cross + d_overflow)

    def test_full_clear_conserves_mass_and_number_and_shifts(self):
        mean_mass0 = self._mean_mass0()
        m0 = self.N0 * mean_mass0
        liquid = _liquid_state(NBINS, {self.B: (m0, self.N0, 0.0, 0.0)})
        dt_vp = 1.0
        d_target = self._d_mean_mass_full_shift()
        # coef1=0 makes d_mean_mass = coef2*dt_vp EXACTLY, independent of
        # s_v_n's precise numeric value (avoids needing to hand-verify the
        # esat table lookup for this conservation-focused test).
        diag = _diag_with(
            NBINS, mean_mass={self.B: mean_mass0}, coef1={}, coef2={self.B: d_target / dt_vp}
        )
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        result = vd.vapor_deposition_liquid(liquid, thermo_state, config, dt_vp, diag)

        lp = index_maps.LiquidPPV
        new_con = result.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = result.values[lp.rmt_q.py_idx, :, 0, 0]

        expected_dmcon = self.N0 * d_target
        # Number exactly conserved; mass conserved up to the analytically
        # predicted growth amount.
        assert new_con.sum() == pytest.approx(self.N0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(m0 + expected_dmcon, abs=1e-12, rel=1e-12)

        # Bin B fully cleared; bin B+1 receives everything.
        assert new_con[self.B] == pytest.approx(0.0, abs=1e-12)
        assert new_mass[self.B] == pytest.approx(0.0, abs=1e-12)
        assert new_con[self.B + 1] == pytest.approx(self.N0, rel=1e-11)
        assert new_mass[self.B + 1] == pytest.approx(m0 + expected_dmcon, rel=1e-11)
        # No leakage into any other bin.
        other = np.delete(new_con, [self.B, self.B + 1])
        np.testing.assert_allclose(other, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Subsaturated (evaporating) single-bin, partial evaporation -- shifts
# down into the adjacent lower bin.
# ---------------------------------------------------------------------------


class TestEvaporationShiftsDown:
    B = 25
    N0 = 1.0

    def _mean_mass0(self) -> float:
        return float(np.sqrt(BINB[self.B] * BINB[self.B + 1]))

    def test_partial_evaporation_shifts_into_lower_bin_and_conserves(self):
        mean_mass0 = self._mean_mass0()
        m0 = self.N0 * mean_mass0
        # Pure water (no aerosol): temp_dM = mass_aero_total - mass_total =
        # -m0, comfortably below the modest evaporative loss chosen below,
        # so the total-evaporation check never triggers (module docstring
        # item 4's r_act sub-condition is moot here either way).
        liquid = _liquid_state(NBINS, {self.B: (m0, self.N0, 0.0, 0.0)})
        dt_vp = 1.0
        d_evap = -0.10 * mean_mass0  # modest 10% mean-mass shrink
        diag = _diag_with(
            NBINS, mean_mass={self.B: mean_mass0}, coef1={}, coef2={self.B: d_evap / dt_vp}
        )
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        # Sanity: this scenario is NOT total evaporation and DOES cross
        # down below binb[B] (module docstring's own reasoning).
        r_lo = (BINB[self.B] / mean_mass0) ** (1.0 / 3.0)
        shifted_lo = BINB[self.B] + d_evap * r_lo
        assert shifted_lo < BINB[self.B]
        assert shifted_lo > BINB[0]

        result = vd.vapor_deposition_liquid(liquid, thermo_state, config, dt_vp, diag)

        lp = index_maps.LiquidPPV
        new_con = result.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = result.values[lp.rmt_q.py_idx, :, 0, 0]

        expected_dmcon = self.N0 * d_evap
        assert new_con.sum() == pytest.approx(self.N0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(m0 + expected_dmcon, abs=1e-12, rel=1e-12)

        # Some (but not all) mass/number shifted down into bin B-1.
        assert new_con[self.B - 1] > 0.0
        assert new_con[self.B - 1] < self.N0
        assert new_mass[self.B - 1] > 0.0


# ---------------------------------------------------------------------------
# General multi-bin mass/number conservation (growth + evaporation mix,
# with aerosol content).
# ---------------------------------------------------------------------------


class TestRemapMassConserving:
    def test_mixed_growth_and_evaporation_conserves_totals(self):
        # mean_mass (rmt/rcon) for each bin is the geometric mean of that
        # bin's own boundaries -- i.e. genuinely WITHIN the bin, matching
        # the physical precondition d_mean_mass's boundary-shift formula
        # assumes (mirrors TestGrowthShiftsToAdjacentBin/
        # TestEvaporationShiftsDown's own bin construction).
        specs = {10: (1.0, 0.2, 0.05), 20: (3.0, 0.03, 0.005), 30: (0.5, 0.02, 0.01)}
        bins = {}
        mean_mass = {}
        for b, (rcon, eps_total_frac, eps_soluble_frac) in specs.items():
            mm = float(np.sqrt(BINB[b] * BINB[b + 1]))
            mean_mass[b] = mm
            rmt = rcon * mm
            rmat = eps_total_frac * rmt
            rmas = eps_soluble_frac * rmt
            bins[b] = (rmt, rcon, rmat, rmas)
        liquid = _liquid_state(NBINS, bins)
        dt_vp = 0.5
        # growth/evap amounts are small fractions of each bin's own
        # mean_mass, keeping the shifted interval's mean well inside it
        # (module docstring item 3's negative-density caveat).
        coef2 = {
            10: 0.05 * mean_mass[10] / dt_vp,
            20: -0.05 * mean_mass[20] / dt_vp,
            30: 0.03 * mean_mass[30] / dt_vp,
        }
        diag = _diag_with(NBINS, mean_mass=mean_mass, coef1={}, coef2=coef2)
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        result = vd.vapor_deposition_liquid(liquid, thermo_state, config, dt_vp, diag)

        lp = index_maps.LiquidPPV
        new_con = result.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = result.values[lp.rmt_q.py_idx, :, 0, 0]
        new_rmat = result.values[lp.rmat_q.py_idx, :, 0, 0]
        new_rmas = result.values[lp.rmas_q.py_idx, :, 0, 0]

        old_con = liquid.values[lp.rcon_q.py_idx, :, 0, 0]
        old_mass = liquid.values[lp.rmt_q.py_idx, :, 0, 0]
        old_rmat = liquid.values[lp.rmat_q.py_idx, :, 0, 0]
        old_rmas = liquid.values[lp.rmas_q.py_idx, :, 0, 0]

        expected_dmcon_total = sum(
            bins[b][1] * coef2[b] * dt_vp
            for b in bins  # con * d_mean_mass
        )

        assert new_con.sum() == pytest.approx(old_con.sum(), abs=1e-12)
        assert new_mass.sum() == pytest.approx(
            old_mass.sum() + expected_dmcon_total, abs=1e-12, rel=1e-11
        )
        # Aerosol mass (total + soluble) is never created/destroyed by
        # vapor deposition -- only redistributed.
        assert new_rmat.sum() == pytest.approx(old_rmat.sum(), rel=1e-11)
        assert new_rmas.sum() == pytest.approx(old_rmas.sum(), rel=1e-11)


# ---------------------------------------------------------------------------
# Inactive bins (diag.mean_mass<=0) pass through unchanged.
# ---------------------------------------------------------------------------


class TestPassthroughInactiveBins:
    def test_untouched_bin_is_byte_identical(self):
        liquid = _liquid_state(
            NBINS,
            {
                5: (1.0e-9, 2.0, 1.0e-11, 5.0e-12),  # inactive (diag.mean_mass left 0)
                15: (5.0e-8, 1.0, 0.0, 0.0),
            },
        )
        mean_mass15 = 5.0e-8 / 1.0
        diag = _diag_with(NBINS, mean_mass={15: mean_mass15}, coef1={}, coef2={15: 0.0})
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        result = vd.vapor_deposition_liquid(liquid, thermo_state, config, 1.0, diag)

        lp = index_maps.LiquidPPV
        np.testing.assert_array_equal(
            result.values[lp.rcon_q.py_idx, 5, 0, 0], liquid.values[lp.rcon_q.py_idx, 5, 0, 0]
        )
        np.testing.assert_array_equal(
            result.values[lp.rmt_q.py_idx, 5, 0, 0], liquid.values[lp.rmt_q.py_idx, 5, 0, 0]
        )
        np.testing.assert_array_equal(
            result.values[lp.rmat_q.py_idx, 5, 0, 0], liquid.values[lp.rmat_q.py_idx, 5, 0, 0]
        )


# ---------------------------------------------------------------------------
# Total evaporation removes a bin entirely.
# ---------------------------------------------------------------------------


class TestTotalEvaporation:
    def test_total_water_loss_zeroes_bin(self):
        b = 15
        mean_mass0 = float(np.sqrt(BINB[b] * BINB[b + 1]))
        n0 = 1.0
        m0 = n0 * mean_mass0
        liquid = _liquid_state(NBINS, {b: (m0, n0, 0.0, 0.0)})  # no aerosol -> temp_dM=-m0
        dt_vp = 1.0
        # Evaporate MORE than the entire mass budget in one substep.
        d_evap = -2.0 * mean_mass0
        diag = _diag_with(NBINS, mean_mass={b: mean_mass0}, coef1={}, coef2={b: d_evap / dt_vp})
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        result = vd.vapor_deposition_liquid(liquid, thermo_state, config, dt_vp, diag)

        lp = index_maps.LiquidPPV
        new_con = result.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = result.values[lp.rmt_q.py_idx, :, 0, 0]
        assert new_con.sum() == pytest.approx(0.0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Per-call replay against a real scale_amps M0 dump (marker-gated).
# ---------------------------------------------------------------------------


@pytest.mark.datatest
def test_vapor_deposition_replay_against_m0_dump() -> None:
    """Would spin up a pre-recorded liquid+thermo state (scale_amps M0
    per-call DEBUG dump, `vapor_deposition` LIQUID call site), run
    `vapor_deposition_liquid`, and compare the resulting liquid state
    against the recorded post-call state (rtol ~1e-8). SKIPPED: no local
    scale_amps M0 per-call vapor-deposition dumps exist in this checkout
    (`driver/ref_data.py` can load `amps_dump_r*.bin` if produced by a
    real scale_amps DEBUG run -- see that module's `read_dump_file`/
    `load_reference` -- none are committed here; matches
    `test_activation.py`'s own `test_activation_replay_against_m0_dump`
    precedent)."""
    pytest.skip(
        "No local scale_amps M0 per-call vapor-deposition dumps available in this checkout -- "
        "see driver/ref_data.py (read_dump_file/load_reference) for the loader once real "
        "amps_dump_r*.bin files (DEBUG-mode scale_amps run, vapor_deposition LIQUID call site) "
        "exist."
    )
