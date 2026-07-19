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
dispatch (including a code-review revision that added the full
`cal_lincubprms_vec` linear/cubic reconstruction and aerosol/vapor
diversion on evaporation -- see `core/vapor_deposition.py`'s module
docstring for every design decision the tests below are constructed
around).

Groups:
* TestExcessVaporDensity -- `excess_vapor_density` vs G3's own formula.
* TestMomentIntegral -- `_moment_integral`'s general cubic formula vs an
  exact closed-form antiderivative.
* TestLinearFitAndReconstruction -- `_linear_fit`/`_reconstruct_distribution`:
  plain fit, degenerate detection, the truncated-support negative-density
  fallback (module docstring item 3), sub-interval additivity.
* TestCubicReconstruction -- the Dinh & Durran (2012) interior cubic
  upgrade, INDEPENDENTLY hand-computed (closed-form Legendre-to-
  polynomial conversion, re-derived by hand, not copied from the
  module-under-test) and a boundary-crossing gather split verified
  against an exact closed-form antiderivative.
* TestSkewedMeanMassReconstruction -- a skewed shifted-interval mean now
  redistributes via the truncated-support mechanism (not a same-bin
  passthrough).
* TestGrowthShiftsToAdjacentBin / TestEvaporationShiftsDown /
  TestRemapMassConserving / TestPassthroughInactiveBins -- liquid-state
  behavior, updated for the `(liquid, aerosol, thermo)` in/out signature.
* TestTotalEvaporation -- a fully-evaporated bin now correctly diverts to
  `AerosolState`/vapor instead of discarding its mass/number (the
  critical fix).
* TestUnderflowDiversion -- a PARTIALLY evaporating bin whose shifted
  interval dips below `binb[0]` diverts that piece to aerosol/vapor.
* TestConservationAcrossEvaporation -- the explicit, paramount
  requirement: total water (vapor+liquid) AND total aerosol mass/number
  conserved to 1e-12 across an evaporation step.
* test_vapor_deposition_replay_against_m0_dump (`pytest.mark.datatest`,
  skipped) -- per-call replay vs spin-up dumps, matching
  `test_activation.py`'s own precedent.
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
    AerosolState,
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
    """`p` is the CGS pressure (dyn/cm^2, `AmpsConst.p00`-scale) this
    file's own hand-computed "golden" references use directly;
    `ThermoState.ptotv` itself is CGS (`state.py`'s own UNIT CONTRACT
    note on `ThermoProp.ptotv`, canonicalized at the two `ThermoState`
    producers) -- stored directly (no conversion) here, so
    `vapor_deposition_liquid` (`core/vapor_deposition.py`, which now
    reads it as-is) sees exactly `p`, keeping every existing golden-value
    comparison in this file unchanged."""
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: p,
        ThermoProp.tv: t,
        ThermoProp.thv: t,
        ThermoProp.piv: 0.0,
        ThermoProp.pbv: 0.0,
        # CGS g/cm^3 (state.py's own UNIT CONTRACT note on ThermoProp.
        # moist_denv) -- INERT for this module (vapor_deposition_liquid
        # never reads ThermoProp.moist_denv, confirmed by grep), stored at
        # a realistic CGS magnitude purely for contract consistency with
        # every other _thermo_state helper in this test suite, not because
        # its value affects any assertion in this file.
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


def _aerosol_state(
    ncat: int = 2, cat0: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> AerosolState:
    """Single-bin AerosolState, single column; category 0 gets `cat0`
    (amt, acon, ams), all other categories start at zero (available to
    receive evaporation-return mass)."""
    ap = index_maps.AerosolPPV
    values = np.zeros((len(AerosolState.PROPS), 1, ncat, 1), dtype=np.float64)
    values[ap.amt_q.py_idx, 0, 0, 0] = cat0[0]
    values[ap.acon_q.py_idx, 0, 0, 0] = cat0[1]
    values[ap.ams_q.py_idx, 0, 0, 0] = cat0[2]
    return AerosolState(values=values)


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


def _total_water(liquid: LiquidState, thermo_state: ThermoState) -> float:
    """`qv + sum(liquid water content)` -- liquid water content is
    `rmt_q - rmat_q` (total mass minus aerosol mass) per bin."""
    lp = index_maps.LiquidPPV
    qv = float(thermo_state.values[list(ThermoState.PROPS).index(ThermoProp.qvv), 0, 0, 0])
    rmt = liquid.values[lp.rmt_q.py_idx, :, 0, 0].sum()
    rmat = liquid.values[lp.rmat_q.py_idx, :, 0, 0].sum()
    return qv + float(rmt - rmat)


def _total_aerosol_mass(liquid: LiquidState, aerosol: AerosolState) -> float:
    lp = index_maps.LiquidPPV
    ap = index_maps.AerosolPPV
    liq_aero = liquid.values[lp.rmat_q.py_idx, :, 0, 0].sum()
    free_aero = aerosol.values[ap.amt_q.py_idx, 0, :, 0].sum()
    return float(liq_aero + free_aero)


def _total_number(liquid: LiquidState, aerosol: AerosolState) -> float:
    lp = index_maps.LiquidPPV
    ap = index_maps.AerosolPPV
    liq_n = liquid.values[lp.rcon_q.py_idx, :, 0, 0].sum()
    free_n = aerosol.values[ap.acon_q.py_idx, 0, :, 0].sum()
    return float(liq_n + free_n)


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
# _moment_integral vs an exact closed-form antiderivative.
# ---------------------------------------------------------------------------


class TestMomentIntegral:
    def test_general_cubic_matches_exact_antiderivative(self):
        # Coefficients/bounds chosen so BOTH the 0th and 1st moment are
        # genuinely non-negative over [bd1,bd2] -- _moment_integral clamps
        # negative results to 0 (G3's own literal clamp), so a scenario
        # that triggers the clamp would make this exact-antiderivative
        # comparison meaningless; this checks the unclamped formula path.
        a0, a1, a2, a3 = 1.3, 0.4, 0.02, 0.001
        bd1, bd2 = 2.0, 7.0

        def antideriv_n(x):
            return a0 * x + a1 * x**2 / 2.0 + a2 * x**3 / 3.0 + a3 * x**4 / 4.0

        def antideriv_m(x):
            return a0 * x**2 / 2.0 + a1 * x**3 / 3.0 + a2 * x**4 / 4.0 + a3 * x**5 / 5.0

        expected_n = antideriv_n(bd2) - antideriv_n(bd1)
        expected_m = antideriv_m(bd2) - antideriv_m(bd1)

        tdn, tdm = vd._moment_integral(
            np.array([a0]),
            np.array([a1]),
            np.array([a2]),
            np.array([a3]),
            np.array([bd1]),
            np.array([bd2]),
        )
        assert tdn[0] == pytest.approx(expected_n, rel=1e-12)
        assert tdm[0] == pytest.approx(expected_m, rel=1e-12)


# ---------------------------------------------------------------------------
# _linear_fit / _reconstruct_distribution.
# ---------------------------------------------------------------------------


class TestLinearFitAndReconstruction:
    def test_plain_fit_reproduces_moments_over_full_interval(self):
        bd1 = np.array([[1.0e-7, 1.0e-10, 1.0e-5]])
        bd2 = np.array([[5.0e-6, 1.0e-8, 8.0e-5]])
        number = np.array([[2.5, 0.001, 100.0]])
        mass = number * 0.5 * (bd1 + bd2)  # mean centered -> non-negative density

        a0, a1, a2, a3, eb1, eb2, ok = vd._reconstruct_distribution(number, mass, bd1, bd2)
        assert np.all(ok)
        np.testing.assert_array_equal(eb1, bd1)
        np.testing.assert_array_equal(eb2, bd2)

        tn, tm = vd._moment_integral(a0, a1, a2, a3, eb1, eb2)
        np.testing.assert_allclose(tn, number, rtol=1e-11)
        np.testing.assert_allclose(tm, mass, rtol=1e-11)

    def test_degenerate_zero_width_not_ok(self):
        _a1, _a2, _a3, ok = vd._linear_fit(
            np.array([[1.0]]), np.array([[1.0e-6]]), np.array([[5.0e-6]]), np.array([[5.0e-6]])
        )
        assert not ok[0, 0]

    def test_negative_density_triggers_truncated_support_not_discarded(self):
        """A moment pair whose mean sits close to `bd1` (far from `bd2`)
        forces a base linear fit that must be steeply DECREASING to keep
        that low mean over the wide interval -- density goes negative
        near `bd2` (the far edge). The TRUNCATED-SUPPORT re-fit (module
        docstring item 3) must succeed (`ok=True`), narrow the support
        from the RIGHT (`a1` sentinel `-2.0`), and still conserve N/M
        exactly."""
        number = np.array([[4.0]])
        mass = np.array([[6.0e-6]])  # mean=1.5e-6, close to bd1=1e-6
        bd1 = np.array([[1.0e-6]])
        bd2 = np.array([[9.0e-6]])

        a1, _a2, _a3, ok = vd._linear_fit(number, mass, bd1, bd2)
        assert ok[0, 0]
        assert a1[0, 0] == -2.0  # truncated RIGHT (density was negative at bd2)

        a0p, a1p, a2p, a3p, eb1, eb2, ok2 = vd._reconstruct_distribution(number, mass, bd1, bd2)
        assert ok2[0, 0]
        assert eb1[0, 0] == bd1[0, 0]
        assert eb2[0, 0] < bd2[0, 0]  # support narrowed from the right

        tn, tm = vd._moment_integral(a0p, a1p, a2p, a3p, eb1, eb2)
        assert tn[0, 0] == pytest.approx(4.0, rel=1e-11)
        assert tm[0, 0] == pytest.approx(6.0e-6, rel=1e-11)

    def test_subinterval_moments_sum_to_full_interval(self):
        """Splitting `[bd1,bd2]` at an interior point and summing the two
        piece integrals must reproduce the full-interval moments -- the
        mathematical property the gather-remap's dense i x ibx overlap
        sum relies on."""
        number = np.array([[4.0]])
        mass = np.array([[4.0 * 5.0e-6]])  # centered mean
        bd1 = np.array([[1.0e-6]])
        bd2 = np.array([[9.0e-6]])

        a0, a1, a2, a3, eb1, eb2, ok = vd._reconstruct_distribution(number, mass, bd1, bd2)
        assert ok[0, 0]
        mid = np.array([[4.5e-6]])
        n1, m1 = vd._moment_integral(a0, a1, a2, a3, eb1, mid)
        n2, m2 = vd._moment_integral(a0, a1, a2, a3, mid, eb2)
        assert (n1 + n2)[0, 0] == pytest.approx(4.0, rel=1e-11)
        assert (m1 + m2)[0, 0] == pytest.approx(4.0 * 5.0e-6, rel=1e-11)


# ---------------------------------------------------------------------------
# Interior cubic (Dinh & Durran 2012) upgrade -- hand-computed.
# ---------------------------------------------------------------------------


class TestCubicReconstruction:
    """Fixed (not RNG-generated at test time) 3-bin scenario, found by an
    offline random search over `_cubic_upgrade`'s own validity gates for
    a case where `i_cubic=True` (the literal, precedence-bugged `b3`
    formula -- see `core/vapor_deposition.py`'s module docstring item 3 --
    makes this the less-common outcome for arbitrary smooth data, so a
    fixed, hand-verified scenario is used rather than a live random
    search in the test itself)."""

    BINB3 = np.array([0.88765071, 1.78790881, 7.27210638, 8.24202062])
    NUMBER3 = np.array([[2.50043225], [4.99800762], [3.50136963]])
    MASS3 = np.array([[3.54257168], [21.47754218], [27.96784781]])

    def _bounds(self):
        bd1 = self.BINB3[:-1][:, None]
        bd2 = self.BINB3[1:][:, None]
        return bd1, bd2

    def test_middle_bin_upgrades_to_cubic(self):
        bd1, bd2 = self._bounds()
        _a0, _a1, a2, a3, eb1, eb2, ok = vd._reconstruct_distribution(
            self.NUMBER3, self.MASS3, bd1, bd2
        )
        assert ok[1, 0]
        # a2/a3 nonzero -> genuinely cubic (not the plain/truncated linear
        # fallback, which always has a2=a3=0).
        assert a2[1, 0] != 0.0
        assert a3[1, 0] != 0.0
        # Support is the FULL (non-truncated) bin -- cubic bins never use
        # the truncated-support mechanism (module docstring item 3).
        assert eb1[1, 0] == bd1[1, 0]
        assert eb2[1, 0] == bd2[1, 0]

    def test_cubic_coefficients_match_independent_hand_derivation(self):
        """Independently re-derives bin 1's cubic coefficients via the
        SAME Legendre-basis algorithm (Dinh & Durran 2012,
        `mod_amps_utility.F90:9908-10005`), written from scratch here
        (not calling `_cubic_upgrade`), including the LITERAL
        (precedence-bugged) `b3` formula -- see `core/vapor_deposition.
        py`'s module docstring item 3. Cross-checked against
        `_reconstruct_distribution`'s own output."""
        bd1_all, bd2_all = self._bounds()
        i = 1
        bd1, bd2 = bd1_all[i, 0], bd2_all[i, 0]
        delta_m = bd2 - bd1
        am0 = 0.5 * (bd1 + bd2)
        number = self.NUMBER3[i, 0]
        mass = self.MASS3[i, 0]

        b0 = number / delta_m
        b1 = 6.0 * (mass - am0 * number) / delta_m**2

        # Left/right neighbors' OWN plain 2-moment linear fit (bd1<bd2 for
        # both, no truncation needed for this fixed scenario -- verified
        # via _linear_fit directly below).
        a1_lin, a2_lin, a3_lin, ok_lin = vd._linear_fit(self.NUMBER3, self.MASS3, bd1_all, bd2_all)
        assert ok_lin.all()

        amb_l = self.MASS3[i - 1, 0] / self.NUMBER3[i - 1, 0]
        anb_l = max(0.0, a1_lin[i - 1, 0]) + a3_lin[i - 1, 0] * (amb_l - a2_lin[i - 1, 0])
        xib_l = 2.0 * (amb_l - am0) / delta_m

        amb_r = self.MASS3[i + 1, 0] / self.NUMBER3[i + 1, 0]
        anb_r = max(0.0, a1_lin[i + 1, 0]) + a3_lin[i + 1, 0] * (amb_r - a2_lin[i + 1, 0])
        xib_r = 2.0 * (amb_r - am0) / delta_m

        a_l = 0.5 * (3.0 * xib_l**2 - 1.0)
        b_l = 0.5 * (5.0 * xib_l**3 - 3.0 * xib_l)
        c_l = anb_l - b0 - b1 * xib_l
        a_r = 0.5 * (3.0 * xib_r**2 - 1.0)
        b_r = 0.5 * (5.0 * xib_r**3 - 3.0 * xib_r)
        c_r = anb_r - b0 - b1 * xib_r

        # Literal Fortran operator precedence (NOT Cramer's rule):
        b3 = (c_l * a_r - a_l * c_r) / b_l * a_r - a_l * b_r
        b2 = (c_l - b3 * b_l) / a_l

        q = am0 / delta_m
        a0_c = b0 - 2.0 * q * b1 + (6.0 * q * q - 0.5) * b2 - (20.0 * q**3 - 3.0 * q) * b3
        a1_c = (2.0 * b1 - 12.0 * q * b2 + (60.0 * q * q - 3.0) * b3) / delta_m
        a2_c = (6.0 * b2 - 60.0 * q * b3) / delta_m**2
        a3_c = 20.0 * b3 / delta_m**3

        a0p, a1p, a2p, a3p, _eb1, _eb2, ok = vd._reconstruct_distribution(
            self.NUMBER3, self.MASS3, bd1_all, bd2_all
        )
        assert ok[i, 0]
        assert a0p[i, 0] == pytest.approx(a0_c, rel=1e-10)
        assert a1p[i, 0] == pytest.approx(a1_c, rel=1e-10)
        assert a2p[i, 0] == pytest.approx(a2_c, rel=1e-10)
        assert a3p[i, 0] == pytest.approx(a3_c, rel=1e-10)

        # And this hand-derived cubic still reproduces bin 1's own moments
        # exactly over the full interval (Legendre P2/P3 orthogonality to
        # P0/P1 -- adding shape terms never perturbs the base 2 moments).
        tn, tm = vd._moment_integral(
            np.array([a0_c]),
            np.array([a1_c]),
            np.array([a2_c]),
            np.array([a3_c]),
            np.array([bd1]),
            np.array([bd2]),
        )
        assert tn[0] == pytest.approx(number, rel=1e-10)
        assert tm[0] == pytest.approx(mass, rel=1e-10)

    def test_boundary_crossing_gather_matches_exact_closed_form_split(self):
        """Redistribute bin 1's cubic-reconstructed distribution across a
        DESTINATION grid with a boundary INSIDE bin 1's own interval
        (`x=4.5`, strictly between `1.7879` and `7.2721`) -- the
        reviewer's own requested test. The gather's split is compared
        against an EXACT closed-form antiderivative evaluation (not
        `_moment_integral` again -- an independent hand computation)."""
        bd1_all, bd2_all = self._bounds()
        a0p, a1p, a2p, a3p, eb1, eb2, ok = vd._reconstruct_distribution(
            self.NUMBER3, self.MASS3, bd1_all, bd2_all
        )
        assert ok[1, 0]
        a0, a1, a2, a3 = a0p[1, 0], a1p[1, 0], a2p[1, 0], a3p[1, 0]

        dest_binb = np.array([0.0, 1.78790881, 4.5, 7.27210638, 10.0])
        poly_a0 = a0p[1:2, :]
        poly_a1 = a1p[1:2, :]
        poly_a2 = a2p[1:2, :]
        poly_a3 = a3p[1:2, :]
        eff_bd1 = eb1[1:2, :]
        eff_bd2 = eb2[1:2, :]
        ratio_rmat = np.zeros((1, 1))
        ratio_rmas = np.zeros((1, 1))
        source_valid = np.array([[True]])

        new_n, new_rmt, _new_rmat, _new_rmas = vd._gather_remap(
            dest_binb,
            poly_a0,
            poly_a1,
            poly_a2,
            poly_a3,
            eff_bd1,
            eff_bd2,
            ratio_rmat,
            ratio_rmas,
            source_valid,
        )

        def antideriv_n(x):
            return a0 * x + a1 * x**2 / 2.0 + a2 * x**3 / 3.0 + a3 * x**4 / 4.0

        def antideriv_m(x):
            return a0 * x**2 / 2.0 + a1 * x**3 / 3.0 + a2 * x**4 / 4.0 + a3 * x**5 / 5.0

        expected_n1 = antideriv_n(4.5) - antideriv_n(1.78790881)
        expected_m1 = antideriv_m(4.5) - antideriv_m(1.78790881)
        expected_n2 = antideriv_n(7.27210638) - antideriv_n(4.5)
        expected_m2 = antideriv_m(7.27210638) - antideriv_m(4.5)

        assert new_n[1, 0] == pytest.approx(expected_n1, rel=1e-10)
        assert new_rmt[1, 0] == pytest.approx(expected_m1, rel=1e-10)
        assert new_n[2, 0] == pytest.approx(expected_n2, rel=1e-10)
        assert new_rmt[2, 0] == pytest.approx(expected_m2, rel=1e-10)
        # Total conserved across the split.
        assert new_n[1, 0] + new_n[2, 0] == pytest.approx(self.NUMBER3[1, 0], rel=1e-10)
        assert new_rmt[1, 0] + new_rmt[2, 0] == pytest.approx(self.MASS3[1, 0], rel=1e-10)


# ---------------------------------------------------------------------------
# A skewed shifted-interval mean now redistributes via truncation, not a
# same-bin passthrough (module docstring item 3 -- superseded mechanism).
# ---------------------------------------------------------------------------


class TestSkewedMeanMassReconstruction:
    def test_skewed_mean_mass_now_reconstructs_via_truncation(self):
        b = 25
        mean_mass0 = BINB[b] + 0.05 * (BINB[b + 1] - BINB[b])  # skewed toward bin b's own edge
        n0 = 1.0
        r_lo = (BINB[b] / mean_mass0) ** (1.0 / 3.0)
        r_hi = (BINB[b + 1] / mean_mass0) ** (1.0 / 3.0)
        d_target = 0.5 * (BINB[b + 1] - BINB[b])
        shifted_lo = max(0.0, BINB[b] + d_target * r_lo)
        shifted_hi = max(0.0, BINB[b + 1] + d_target * r_hi)

        number = np.array([[n0]])
        mass = np.array([[n0 * (mean_mass0 + d_target)]])
        bd1 = np.array([[shifted_lo]])
        bd2 = np.array([[shifted_hi]])

        a0, a1, a2, a3, eb1, eb2, ok = vd._reconstruct_distribution(number, mass, bd1, bd2)
        assert ok[0, 0]  # no longer needs the same-bin fallback

        tn, tm = vd._moment_integral(a0, a1, a2, a3, eb1, eb2)
        assert tn[0, 0] == pytest.approx(n0, rel=1e-11)
        assert tm[0, 0] == pytest.approx(n0 * (mean_mass0 + d_target), rel=1e-11)


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
        aerosol = _aerosol_state()
        dt_vp = 1.0
        d_target = self._d_mean_mass_full_shift()
        diag = _diag_with(
            NBINS, mean_mass={self.B: mean_mass0}, coef1={}, coef2={self.B: d_target / dt_vp}
        )
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        liquid_out, aerosol_out, thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        lp = index_maps.LiquidPPV
        new_con = liquid_out.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = liquid_out.values[lp.rmt_q.py_idx, :, 0, 0]

        expected_dmcon = self.N0 * d_target
        assert new_con.sum() == pytest.approx(self.N0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(m0 + expected_dmcon, abs=1e-12, rel=1e-12)

        assert new_con[self.B] == pytest.approx(0.0, abs=1e-12)
        assert new_mass[self.B] == pytest.approx(0.0, abs=1e-12)
        assert new_con[self.B + 1] == pytest.approx(self.N0, rel=1e-11)
        assert new_mass[self.B + 1] == pytest.approx(m0 + expected_dmcon, rel=1e-11)
        other = np.delete(new_con, [self.B, self.B + 1])
        np.testing.assert_allclose(other, 0.0, atol=1e-12)

        # Growth (condensation) consumes vapor; no evaporation, so aerosol
        # is untouched.
        qv_before = QV_STD
        qv_after = float(thermo_out.values[list(ThermoState.PROPS).index(ThermoProp.qvv), 0, 0, 0])
        assert qv_after == pytest.approx(qv_before - expected_dmcon, rel=1e-10)
        np.testing.assert_array_equal(aerosol_out.values, aerosol.values)


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
        liquid = _liquid_state(NBINS, {self.B: (m0, self.N0, 0.0, 0.0)})
        aerosol = _aerosol_state()
        dt_vp = 1.0
        d_evap = -0.10 * mean_mass0  # modest 10% mean-mass shrink
        diag = _diag_with(
            NBINS, mean_mass={self.B: mean_mass0}, coef1={}, coef2={self.B: d_evap / dt_vp}
        )
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        r_lo = (BINB[self.B] / mean_mass0) ** (1.0 / 3.0)
        shifted_lo = BINB[self.B] + d_evap * r_lo
        assert shifted_lo < BINB[self.B]
        assert shifted_lo > BINB[0]

        liquid_out, _aerosol_out, _thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        lp = index_maps.LiquidPPV
        new_con = liquid_out.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = liquid_out.values[lp.rmt_q.py_idx, :, 0, 0]

        expected_dmcon = self.N0 * d_evap
        assert new_con.sum() == pytest.approx(self.N0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(m0 + expected_dmcon, abs=1e-12, rel=1e-12)

        assert new_con[self.B - 1] > 0.0
        assert new_con[self.B - 1] < self.N0
        assert new_mass[self.B - 1] > 0.0


# ---------------------------------------------------------------------------
# General multi-bin mass/number conservation (growth + evaporation mix,
# with aerosol content).
# ---------------------------------------------------------------------------


class TestRemapMassConserving:
    def test_mixed_growth_and_evaporation_conserves_totals(self):
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
        aerosol = _aerosol_state()
        dt_vp = 0.5
        coef2 = {
            10: 0.05 * mean_mass[10] / dt_vp,
            20: -0.05 * mean_mass[20] / dt_vp,
            30: 0.03 * mean_mass[30] / dt_vp,
        }
        diag = _diag_with(NBINS, mean_mass=mean_mass, coef1={}, coef2=coef2)
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        liquid_out, _aerosol_out, _thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        lp = index_maps.LiquidPPV
        new_con = liquid_out.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = liquid_out.values[lp.rmt_q.py_idx, :, 0, 0]
        new_rmat = liquid_out.values[lp.rmat_q.py_idx, :, 0, 0]
        new_rmas = liquid_out.values[lp.rmas_q.py_idx, :, 0, 0]

        old_con = liquid.values[lp.rcon_q.py_idx, :, 0, 0]
        old_mass = liquid.values[lp.rmt_q.py_idx, :, 0, 0]
        old_rmat = liquid.values[lp.rmat_q.py_idx, :, 0, 0]
        old_rmas = liquid.values[lp.rmas_q.py_idx, :, 0, 0]

        expected_dmcon_total = sum(bins[b][1] * coef2[b] * dt_vp for b in bins)

        assert new_con.sum() == pytest.approx(old_con.sum(), abs=1e-12)
        assert new_mass.sum() == pytest.approx(
            old_mass.sum() + expected_dmcon_total, abs=1e-12, rel=1e-11
        )
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
        aerosol = _aerosol_state()
        mean_mass15 = 5.0e-8 / 1.0
        diag = _diag_with(NBINS, mean_mass={15: mean_mass15}, coef1={}, coef2={15: 0.0})
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        liquid_out, _aerosol_out, _thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, 1.0, diag
        )

        lp = index_maps.LiquidPPV
        np.testing.assert_array_equal(
            liquid_out.values[lp.rcon_q.py_idx, 5, 0, 0], liquid.values[lp.rcon_q.py_idx, 5, 0, 0]
        )
        np.testing.assert_array_equal(
            liquid_out.values[lp.rmt_q.py_idx, 5, 0, 0], liquid.values[lp.rmt_q.py_idx, 5, 0, 0]
        )
        np.testing.assert_array_equal(
            liquid_out.values[lp.rmat_q.py_idx, 5, 0, 0], liquid.values[lp.rmat_q.py_idx, 5, 0, 0]
        )


# ---------------------------------------------------------------------------
# Total evaporation now correctly diverts to AerosolState/vapor (the
# critical fix) instead of discarding the bin's mass/number.
# ---------------------------------------------------------------------------


class TestTotalEvaporation:
    B = 15

    def _scenario(self, *, aero_frac: float, eps_soluble_frac: float):
        mean_mass0 = float(np.sqrt(BINB[self.B] * BINB[self.B + 1]))
        n0 = 1.0
        aero_total = aero_frac * mean_mass0 * n0
        aero_soluble = eps_soluble_frac * aero_total
        m0 = n0 * mean_mass0
        liquid = _liquid_state(NBINS, {self.B: (m0, n0, aero_total, aero_soluble)})
        aerosol = _aerosol_state()
        dt_vp = 1.0
        d_evap = -2.0 * mean_mass0  # evaporate MORE than the entire mass budget
        diag = _diag_with(
            NBINS, mean_mass={self.B: mean_mass0}, coef1={}, coef2={self.B: d_evap / dt_vp}
        )
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()
        return liquid, aerosol, thermo_state, config, dt_vp, diag, n0, m0, aero_total, aero_soluble

    def test_total_water_loss_zeroes_liquid_bin(self):
        liquid, aerosol, thermo_state, config, dt_vp, diag, *_ = self._scenario(
            aero_frac=0.0, eps_soluble_frac=0.0
        )
        liquid_out, _aerosol_out, _thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )
        lp = index_maps.LiquidPPV
        new_con = liquid_out.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = liquid_out.values[lp.rmt_q.py_idx, :, 0, 0]
        assert new_con.sum() == pytest.approx(0.0, abs=1e-12)
        assert new_mass.sum() == pytest.approx(0.0, abs=1e-12)

    def test_number_and_aerosol_mass_returned_to_aerosol_state(self):
        """THE CRITICAL FIX: a fully-evaporated bin's number and aerosol
        (total + soluble) mass must reappear in `AerosolState`, not
        vanish."""
        (
            liquid,
            aerosol,
            thermo_state,
            config,
            dt_vp,
            diag,
            n0,
            _m0,
            aero_total,
            aero_soluble,
        ) = self._scenario(aero_frac=0.4, eps_soluble_frac=0.6)
        _liquid_out, aerosol_out, _thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )
        ap = index_maps.AerosolPPV
        total_number_returned = aerosol_out.values[ap.acon_q.py_idx, 0, :, 0].sum()
        total_mass_returned = aerosol_out.values[ap.amt_q.py_idx, 0, :, 0].sum()
        total_soluble_returned = aerosol_out.values[ap.ams_q.py_idx, 0, :, 0].sum()

        assert total_number_returned == pytest.approx(n0, rel=1e-11)
        assert total_mass_returned == pytest.approx(aero_total, rel=1e-11)
        assert total_soluble_returned == pytest.approx(aero_soluble, rel=1e-11)

    def test_water_only_remainder_goes_to_vapor(self):
        liquid, aerosol, thermo_state, config, dt_vp, diag, _n0, m0, aero_total, _aero_soluble = (
            self._scenario(aero_frac=0.4, eps_soluble_frac=0.6)
        )
        _liquid_out, _aerosol_out, thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )
        qv_before = QV_STD
        qv_after = float(thermo_out.values[list(ThermoState.PROPS).index(ThermoProp.qvv), 0, 0, 0])
        expected_water_only = m0 - aero_total
        assert qv_after == pytest.approx(qv_before + expected_water_only, rel=1e-10)


# ---------------------------------------------------------------------------
# Partial (non-total) evaporation whose shifted interval underflows below
# binb[0] -- the underflow piece diverts to aerosol/vapor.
# ---------------------------------------------------------------------------


class TestUnderflowDiversion:
    B = 0  # smallest liquid bin

    def test_underflow_piece_diverts_and_conserves(self):
        mean_mass0 = float(np.sqrt(BINB[self.B] * BINB[self.B + 1]))
        n0 = 1.0
        aero_total = 0.3 * mean_mass0 * n0
        aero_soluble = 0.5 * aero_total
        m0 = n0 * mean_mass0
        liquid = _liquid_state(NBINS, {self.B: (m0, n0, aero_total, aero_soluble)})
        aerosol = _aerosol_state()
        dt_vp = 1.0
        d_evap = -0.5 * mean_mass0
        diag = _diag_with(
            NBINS, mean_mass={self.B: mean_mass0}, coef1={}, coef2={self.B: d_evap / dt_vp}
        )
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()

        # Sanity: this is genuinely a PARTIAL evaporation with underflow,
        # not total evaporation (independently verified during this
        # task's own TDD).
        r_lo = (BINB[self.B] / mean_mass0) ** (1.0 / 3.0)
        shifted_lo = BINB[self.B] + d_evap * r_lo
        assert shifted_lo < BINB[0]

        liquid_out, aerosol_out, _thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        lp = index_maps.LiquidPPV
        ap = index_maps.AerosolPPV
        new_con = liquid_out.values[lp.rcon_q.py_idx, :, 0, 0]
        new_mass = liquid_out.values[lp.rmt_q.py_idx, :, 0, 0]
        aero_number_returned = aerosol_out.values[ap.acon_q.py_idx, 0, :, 0].sum()
        aero_mass_returned = aerosol_out.values[ap.amt_q.py_idx, 0, :, 0].sum()

        # Some number/mass DID reach the aerosol group (this is the
        # distinguishing behavior of this test vs TestEvaporationShiftsDown).
        assert aero_number_returned > 0.0
        assert aero_mass_returned > 0.0

        expected_dmcon = n0 * d_evap
        assert new_con.sum() + aero_number_returned == pytest.approx(n0, rel=1e-11)
        assert new_mass.sum() + aero_mass_returned == pytest.approx(
            m0 + expected_dmcon + aero_mass_returned, rel=1e-10
        )


# ---------------------------------------------------------------------------
# The paramount conservation requirement: total water (vapor+liquid) AND
# total aerosol mass/number conserved to 1e-12 across an evaporation step.
# ---------------------------------------------------------------------------


class TestConservationAcrossEvaporation:
    def test_total_water_and_aerosol_conserved_total_evaporation(self):
        b = 15
        mean_mass0 = float(np.sqrt(BINB[b] * BINB[b + 1]))
        n0 = 1.0
        aero_total = 0.4 * mean_mass0 * n0
        aero_soluble = 0.6 * aero_total
        m0 = n0 * mean_mass0
        liquid = _liquid_state(NBINS, {b: (m0, n0, aero_total, aero_soluble)})
        aerosol = _aerosol_state()
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()
        dt_vp = 1.0
        d_evap = -2.0 * mean_mass0  # total evaporation
        diag = _diag_with(NBINS, mean_mass={b: mean_mass0}, coef1={}, coef2={b: d_evap / dt_vp})

        water_before = _total_water(liquid, thermo_state)
        aero_mass_before = _total_aerosol_mass(liquid, aerosol)
        number_before = _total_number(liquid, aerosol)

        liquid_out, aerosol_out, thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        water_after = _total_water(liquid_out, thermo_out)
        aero_mass_after = _total_aerosol_mass(liquid_out, aerosol_out)
        number_after = _total_number(liquid_out, aerosol_out)

        assert water_after == pytest.approx(water_before, abs=1e-12, rel=1e-12)
        assert aero_mass_after == pytest.approx(aero_mass_before, abs=1e-12, rel=1e-12)
        assert number_after == pytest.approx(number_before, abs=1e-12, rel=1e-12)

    def test_total_water_and_aerosol_conserved_underflow_evaporation(self):
        b = 0
        mean_mass0 = float(np.sqrt(BINB[b] * BINB[b + 1]))
        n0 = 1.0
        aero_total = 0.3 * mean_mass0 * n0
        aero_soluble = 0.5 * aero_total
        m0 = n0 * mean_mass0
        liquid = _liquid_state(NBINS, {b: (m0, n0, aero_total, aero_soluble)})
        aerosol = _aerosol_state()
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()
        dt_vp = 1.0
        d_evap = -0.5 * mean_mass0  # partial, underflowing
        diag = _diag_with(NBINS, mean_mass={b: mean_mass0}, coef1={}, coef2={b: d_evap / dt_vp})

        water_before = _total_water(liquid, thermo_state)
        aero_mass_before = _total_aerosol_mass(liquid, aerosol)
        number_before = _total_number(liquid, aerosol)

        liquid_out, aerosol_out, thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        water_after = _total_water(liquid_out, thermo_out)
        aero_mass_after = _total_aerosol_mass(liquid_out, aerosol_out)
        number_after = _total_number(liquid_out, aerosol_out)

        assert water_after == pytest.approx(water_before, abs=1e-12, rel=1e-12)
        assert aero_mass_after == pytest.approx(aero_mass_before, abs=1e-12, rel=1e-12)
        assert number_after == pytest.approx(number_before, abs=1e-12, rel=1e-12)

    def test_total_water_and_aerosol_conserved_multi_bin_mixed(self):
        specs = {5: (1.0, 0.2, 0.05, 0.4), 15: (2.0, 0.3, 0.1, -0.6), 25: (0.7, 0.1, 0.02, 0.15)}
        bins = {}
        mean_mass = {}
        coef2 = {}
        dt_vp = 0.5
        for b, (rcon, eps_total_frac, eps_soluble_frac, growth_frac) in specs.items():
            mm = float(np.sqrt(BINB[b] * BINB[b + 1]))
            mean_mass[b] = mm
            rmt = rcon * mm
            rmat = eps_total_frac * rmt
            rmas = eps_soluble_frac * rmt
            bins[b] = (rmt, rcon, rmat, rmas)
            coef2[b] = growth_frac * mm / dt_vp
        liquid = _liquid_state(NBINS, bins)
        aerosol = _aerosol_state(cat0=(1.0e-10, 50.0, 5.0e-11))
        thermo_state = _thermo_state()
        config = AmpsConfig.cloudlab()
        diag = _diag_with(NBINS, mean_mass=mean_mass, coef1={}, coef2=coef2)

        water_before = _total_water(liquid, thermo_state)
        aero_mass_before = _total_aerosol_mass(liquid, aerosol)
        number_before = _total_number(liquid, aerosol)

        liquid_out, aerosol_out, thermo_out = vd.vapor_deposition_liquid(
            liquid, aerosol, thermo_state, config, dt_vp, diag
        )

        water_after = _total_water(liquid_out, thermo_out)
        aero_mass_after = _total_aerosol_mass(liquid_out, aerosol_out)
        number_after = _total_number(liquid_out, aerosol_out)

        assert water_after == pytest.approx(water_before, abs=1e-12, rel=1e-11)
        assert aero_mass_after == pytest.approx(aero_mass_before, abs=1e-12, rel=1e-11)
        assert number_after == pytest.approx(number_before, abs=1e-12, rel=1e-11)


# ---------------------------------------------------------------------------
# Per-call replay against a real scale_amps M0 dump (marker-gated).
# ---------------------------------------------------------------------------


@pytest.mark.datatest
def test_vapor_deposition_replay_against_m0_dump() -> None:
    """Would spin up a pre-recorded liquid+aerosol+thermo state (scale_amps
    M0 per-call DEBUG dump, `vapor_deposition` LIQUID call site), run
    `vapor_deposition_liquid`, and compare the resulting liquid/aerosol/
    thermo state against the recorded post-call state (rtol ~1e-8).
    SKIPPED: no local scale_amps M0 per-call vapor-deposition dumps exist
    in this checkout (`driver/ref_data.py` can load `amps_dump_r*.bin` if
    produced by a real scale_amps DEBUG run -- see that module's
    `read_dump_file`/`load_reference` -- none are committed here; matches
    `test_activation.py`'s own `test_activation_replay_against_m0_dump`
    precedent)."""
    pytest.skip(
        "No local scale_amps M0 per-call vapor-deposition dumps available in this checkout -- "
        "see driver/ref_data.py (read_dump_file/load_reference) for the loader once real "
        "amps_dump_r*.bin files (DEBUG-mode scale_amps run, vapor_deposition LIQUID call site) "
        "exist."
    )
