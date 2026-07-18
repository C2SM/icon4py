# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/activation.py (M2a Task 3, Part 1): the supersaturation
Brent solver + vapor/temperature objective functions, per
docs/superpowers/facts/m2/activation.md ("G2").

`zbrent_act_vec` is tested against SYNTHETIC monotone functions (per this
task's own test spec) -- it takes a generic `residual_fn`, so this
directly exercises the ported Brent MECHANICS (bracket-expansion doubling/
additive steps, main iteration, tolerance, non-convergence bookkeeping)
independent of the (much larger, partially scope-limited) ice/liquid
vapor-objective physics. `func_liqvap_vec`/`func_icevap_vec` are checked
against independently hand-derived scalar reference values for a known
(T, S, coefficients) triple -- the reference calls `core.thermo.esat_lk`
directly (an already-independently-tested M1 primitive, same precedent as
`test_liquid_diag.py`'s own `_golden_bin`) but hand-codes the G2 formulas
themselves, not a copy of the module-under-test's numpy code.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import activation, thermo
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst


ESTBAR, ESITBAR = thermo.make_esat_tables()


def _residual_fn(shift: float) -> Callable[[np.ndarray], np.ndarray]:
    """A trivially monotone synthetic objective with a known root at
    `x=shift`: `f(x) = x - shift`."""

    def _fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) - shift

    return _fn


# ---------------------------------------------------------------------------
# zbrent_act_vec -- synthetic monotone functions.
# ---------------------------------------------------------------------------


class TestZbrentActVec:
    def test_converges_within_initial_bracket(self) -> None:
        """iphase==1 seeds b=-0.5, a=0.2; root at 0.05 is already inside
        that bracket -- no bracket-expansion needed."""
        npoints = 3
        result = activation.zbrent_act_vec(
            _residual_fn(0.05),
            iphase=np.full(npoints, 1),
            iswitch=1,
            sw_o=np.zeros(npoints),
            t_a_o=np.full(npoints, 260.0),
            estbar=ESTBAR,
            esitbar=ESITBAR,
        )
        np.testing.assert_allclose(result.sw_n, 0.05, atol=1.0e-6)
        np.testing.assert_array_equal(result.ierror1, 0)

    def test_bracket_expansion_additive_root_outside_initial_bracket(self) -> None:
        """Root at 5.0 is far outside the initial [-0.5, 0.2] bracket;
        with iswitch!=2 the bracket expands additively (a+=0.2 per
        iteration), needing ~24 of the ITMAX_ini=30 budget to reach it."""
        npoints = 1
        result = activation.zbrent_act_vec(
            _residual_fn(5.0),
            iphase=np.full(npoints, 1),
            iswitch=1,
            sw_o=np.zeros(npoints),
            t_a_o=np.full(npoints, 260.0),
            estbar=ESTBAR,
            esitbar=ESITBAR,
        )
        np.testing.assert_allclose(result.sw_n, 5.0, atol=1.0e-6)
        np.testing.assert_array_equal(result.ierror1, 0)

    def test_bracket_expansion_multiplicative_iswitch2(self) -> None:
        """With iswitch==2 the upper bracket expands multiplicatively
        (a*=2.0 per iteration): 0.2 -> 0.4 -> ... reaches 50 in 8 steps."""
        npoints = 1
        result = activation.zbrent_act_vec(
            _residual_fn(50.0),
            iphase=np.full(npoints, 1),
            iswitch=2,
            sw_o=np.zeros(npoints),
            t_a_o=np.full(npoints, 260.0),
            estbar=ESTBAR,
            esitbar=ESITBAR,
        )
        np.testing.assert_allclose(result.sw_n, 50.0, atol=1.0e-6)
        np.testing.assert_array_equal(result.ierror1, 0)

    def test_multiple_lanes_converge_independently(self) -> None:
        roots = np.array([-0.3, 0.0, 0.15, 3.0])
        result = activation.zbrent_act_vec(
            lambda x: np.asarray(x) - roots,
            iphase=np.full(4, 1),
            iswitch=1,
            sw_o=np.zeros(4),
            t_a_o=np.full(4, 260.0),
            estbar=ESTBAR,
            esitbar=ESITBAR,
        )
        np.testing.assert_allclose(result.sw_n, roots, atol=1.0e-6)

    def test_bracket_never_forms_flags_ierror1(self) -> None:
        """An always-positive residual can never bracket a root: `b`
        keeps decreasing (-0.2/iteration) until it crosses the -2.0
        failure bound -> ierror1==2."""
        npoints = 1
        result = activation.zbrent_act_vec(
            lambda x: np.asarray(x) + 10.0,
            iphase=np.full(npoints, 1),
            iswitch=1,
            sw_o=np.array([0.123]),
            t_a_o=np.full(npoints, 260.0),
            estbar=ESTBAR,
            esitbar=ESITBAR,
        )
        np.testing.assert_array_equal(result.ierror1, 2)

    def test_iphase2_bracket_uses_ice_reference_temperature(self) -> None:
        """iphase==2's `b` bracket is `(-0.5+1.0)/r_e - 1.0`,
        `r_e=e_satw(T_a_o)/e_sati(min(T_0,T_a_o))` -- verify the seeded
        bracket (hence the converged root, since here the root sits
        exactly on that seed for iswitch=2's a-side, and we probe with a
        residual whose root is at the iphase==2 b-seed itself) matches
        the closed-form formula."""
        t_a_o = np.array([250.0])
        e_satw = thermo.esat_lk(1, t_a_o, ESTBAR, ESITBAR)
        e_sati = thermo.esat_lk(2, np.minimum(float(AmpsConst.T_0), t_a_o), ESTBAR, ESITBAR)
        r_e = e_satw / e_sati
        expected_b = (-0.5 + 1.0) / r_e - 1.0

        result = activation.zbrent_act_vec(
            _residual_fn(float(expected_b[0])),
            iphase=np.array([2]),
            iswitch=1,
            sw_o=np.array([0.0]),
            t_a_o=t_a_o,
            estbar=ESTBAR,
            esitbar=ESITBAR,
        )
        np.testing.assert_allclose(result.sw_n, expected_b, atol=1.0e-6)


# ---------------------------------------------------------------------------
# cal_air_temp / cal_coef_svsteady_init.
# ---------------------------------------------------------------------------


class TestCalAirTemp:
    def test_matches_linear_branch(self) -> None:
        til = np.array([250.0])
        qr = np.array([1.0e-4])
        qi = np.array([0.0])
        heat = float(AmpsConst.L_e) * qr[0]
        t_lin = til[0] * (1.0 + heat / (float(AmpsConst.C_pa) * 253.0))
        assert t_lin <= 253.0  # sanity: exercises the linear branch
        result = activation.cal_air_temp(til, qr, qi)
        np.testing.assert_allclose(result, t_lin, rtol=1.0e-12)

    def test_matches_quadratic_branch(self) -> None:
        til = np.array([270.0])
        qr = np.array([5.0e-3])
        qi = np.array([1.0e-3])
        c_pa = float(AmpsConst.C_pa)
        heat = float(AmpsConst.L_e) * qr[0] + float(AmpsConst.L_s) * qi[0]
        t_lin = til[0] * (1.0 + heat / (c_pa * 253.0))
        assert t_lin > 253.0  # sanity: exercises the quadratic branch
        t_quad = 0.5 * (til[0] + np.sqrt(til[0] ** 2 + 4.0 * til[0] / c_pa * heat))
        result = activation.cal_air_temp(til, qr, qi)
        np.testing.assert_allclose(result, t_quad, rtol=1.0e-12)


class TestCalCoefSvsteadyInit:
    def test_scales_by_dt(self) -> None:
        coef1 = np.array([1.5e-9, -2.0e-10])
        result = activation.cal_coef_svsteady_init(coef1, dt=3.0)
        np.testing.assert_allclose(result, coef1 * 3.0)


# ---------------------------------------------------------------------------
# Koehler critical/haze radius vs an independent scalar reference.
# ---------------------------------------------------------------------------


def _golden_critrad(aa: float, sb: float, beta: float, r_n: float) -> float:
    zd = r_n**beta * np.sqrt(sb / (3.0 * aa))
    if zd > 1.0e-2:
        pp = (zd**3 + np.sqrt(zd**3 + 0.25) + 0.5) ** (1.0 / 3.0)
        pm_arg = zd**3 - np.sqrt(zd**3 + 0.25) + 0.5
        pm = np.sign(pm_arg) * abs(pm_arg) ** (1.0 / 3.0)  # real cube root, pm_arg may be < 0
        return r_n * (zd + pp + pm)
    return r_n * (1.0 + zd + 2.0 * zd**3 / 3.0)


class TestKohlerCriticalRadius:
    def test_matches_independent_scalar_reference_large_zd(self) -> None:
        # Ammonium-bisulfate-like aerosol: dry radius 0.05 micron (5e-6 cm),
        # T=280K Kelvin coefficient, eps_map=1 (fully soluble).
        t = 280.0
        sig_wa = float(thermo.sfc_tension(np.array([t]))[0])
        aa = 2.0 * sig_wa / (float(AmpsConst.den_w) * float(AmpsConst.R_v) * t)
        sb = 2.0 * 1.0 * 115.11 * 1.79 / (115.11 * 1.0) * 0.75  # nu*eps*M_w*den/(M_aps*den_w)*phi
        r_n = 5.0e-6
        beta = 0.5

        expected = _golden_critrad(aa, sb, beta, r_n)
        result = activation.kohler_critical_radius(
            np.array([aa]), np.array([sb]), np.array([beta]), np.array([r_n])
        )
        np.testing.assert_allclose(result, expected, rtol=1.0e-10)
        assert result[0] > r_n  # critical radius must exceed the dry radius

    def test_matches_independent_scalar_reference_small_zd(self) -> None:
        # Large AA (weak solute effect relative to Kelvin term) drives Zd<=1e-2.
        aa = 100.0
        sb = 1.0e-8
        r_n = 1.0e-6
        beta = 0.5
        expected = _golden_critrad(aa, sb, beta, r_n)
        zd = r_n**beta * np.sqrt(sb / (3.0 * aa))
        assert zd <= 1.0e-2  # sanity: exercises the small-Zd branch
        result = activation.kohler_critical_radius(
            np.array([aa]), np.array([sb]), np.array([beta]), np.array([r_n])
        )
        np.testing.assert_allclose(result, expected, rtol=1.0e-10)


class TestKohlerHazeRadius:
    def test_matches_independent_scalar_reference_subsaturated(self) -> None:
        # S_w < 0.97 branch, closed form.
        aa = 1.0e-5
        sb = 1.0e-3
        r_n = 1.0e-5
        beta = 0.5
        s_w = 0.9

        neg_log_sw = -np.log(s_w)
        c_w = aa / (3.0 * sb ** (1.0 / 3.0) * r_n ** (2.0 * (1.0 + beta) / 3.0))
        expected = r_n * (
            1.0
            + sb
            * r_n ** (2.0 * (1.0 + beta) - 3.0)
            / neg_log_sw
            * (1.0 + c_w * neg_log_sw ** (-2.0 / 3.0)) ** (-3.0)
        ) ** (1.0 / 3.0)

        result = activation.kohler_haze_radius(
            np.array([aa]), np.array([sb]), np.array([beta]), np.array([s_w]), np.array([r_n])
        )
        np.testing.assert_allclose(result, expected, rtol=1.0e-9)
        assert result[0] > r_n

    def test_matches_independent_scalar_reference_near_saturation_small_zd(self) -> None:
        # S_w>=0.97 but Zd<=0.1 -> series branch.
        aa = 1.0
        sb = 1.0e-3
        r_n = 1.0e-6
        beta = 0.5
        s_w = 0.99
        zd = r_n**beta * np.sqrt(sb / (3.0 * aa))
        assert zd <= 0.1
        zd2 = zd * zd
        expected = r_n * (1.0 + zd2 - (1.0 / 3.0) * zd2 * zd2 * zd2)
        result = activation.kohler_haze_radius(
            np.array([aa]), np.array([sb]), np.array([beta]), np.array([s_w]), np.array([r_n])
        )
        np.testing.assert_allclose(result, expected, rtol=1.0e-9)


# ---------------------------------------------------------------------------
# func_liqvap_vec -- golden scalar reference (single point, single bin).
# ---------------------------------------------------------------------------


def _box_liq(**overrides: np.ndarray) -> activation.ActivationBoxState:
    npoints = 1
    defaults = dict(
        mes_rc=np.array([2]),
        t=np.array([270.0]),
        til=np.array([270.0]),
        qr_0=np.zeros(npoints),
        qi_0=np.zeros(npoints),
        gain_mi_rim=np.zeros(npoints),
        gain_mi_frn=np.zeros(npoints),
        den=np.array([1.2e-3]),
        qtp=np.array([1.0e-2]),
        pressure=np.array([1.0e6]),
        qr_b=np.zeros(npoints),
        qi_b=np.zeros(npoints),
        rv=np.array([1.0e-2]),
        sw=np.array([0.05]),
        used_ma_act=np.array([1.0e-10]),
        akk_lmt=np.array([0.8]),
        sw_allact=np.array([0.1]),
        si_alldep=np.zeros(npoints),
        used_ma_dep=np.zeros(npoints),
        aerosol_dep_con=np.zeros(npoints),
        aerosol_dep_mass=np.zeros(npoints),
        s_c_dhfmin=np.zeros(npoints),
        ds_alldhf=np.zeros(npoints),
        akk_lmt_dhf=np.zeros(npoints),
        used_ma_dhf=np.zeros(npoints),
    )
    defaults.update(overrides)
    return activation.ActivationBoxState(**defaults)


class TestFuncLiqvapVec:
    def test_matches_golden_scalar_reference(self) -> None:
        x = np.array([0.05])
        con, mass_total, mass_aerosol = 10.0, 1.0e-9, 1.0e-12
        coef1, coef2 = 1.0e-10, -1.0e-12
        r_act, mean_mass, density = 1.0e-4, 1.0e-10, 1.0
        dt = 1.0

        liq = activation.LiquidBinState(
            con=np.array([[con]]),
            mass_total=np.array([[mass_total]]),
            mass_aerosol=np.array([[mass_aerosol]]),
            coef1=np.array([[coef1]]),
            coef2=np.array([[coef2]]),
            r_act=np.array([[r_act]]),
            mean_mass=np.array([[mean_mass]]),
            density=np.array([[density]]),
        )
        box = _box_liq()

        result = activation.func_liqvap_vec(x, box, liq, flagp_r=1, dt=dt)

        # --- golden scalar reference, independently hand-derived from G2 ---
        akk_expected = min(1.0, 0.8, 0.05 / 0.1)  # akk_lmt, x/sw_allact
        used_mr_act_expected = akk_expected * 1.0e-10

        d_mean_mass = (coef1 * x[0] + coef2) * dt
        water_mass = mass_total - mass_aerosol
        cond1 = d_mean_mass * con <= (mass_aerosol - mass_total)
        shrink_radius = float(AmpsConst.coef3i4p1i3) * np.cbrt((mean_mass + d_mean_mass) / density)
        cond2 = (d_mean_mass < 0.0) and (r_act > shrink_radius)
        if cond1 or cond2:
            used_mr_vap_expected = -water_mass
            liq_left_expected = 0.0
        else:
            used_mr_vap_expected = d_mean_mass * con
            liq_left_expected = water_mass + d_mean_mass * con

        np.testing.assert_allclose(result.used_mr_act, [used_mr_act_expected], rtol=1.0e-12)
        np.testing.assert_allclose(result.used_mr_vap, [used_mr_vap_expected], rtol=1.0e-10)
        np.testing.assert_allclose(result.liq_left, [liq_left_expected], rtol=1.0e-10)
        np.testing.assert_array_equal(result.noccnt, [0])
        np.testing.assert_allclose(result.akk, [akk_expected], rtol=1.0e-12)

    def test_flagp_r_off_returns_zeros(self) -> None:
        x = np.array([0.05])
        liq = activation.LiquidBinState(*(np.array([[1.0]]) for _ in range(8)))
        box = _box_liq()
        result = activation.func_liqvap_vec(x, box, liq, flagp_r=0, dt=1.0)
        np.testing.assert_array_equal(result.used_mr_vap, [0.0])
        np.testing.assert_array_equal(result.used_mr_act, [0.0])
        np.testing.assert_array_equal(result.liq_left, [0.0])
        np.testing.assert_array_equal(result.noccnt, [1])

    def test_bin_below_number_threshold_contributes_nothing(self) -> None:
        x = np.array([0.05])
        liq = activation.LiquidBinState(
            con=np.array([[1.0e-31]]),  # below NLMT
            mass_total=np.array([[1.0e-9]]),
            mass_aerosol=np.array([[1.0e-12]]),
            coef1=np.array([[1.0e-10]]),
            coef2=np.array([[-1.0e-12]]),
            r_act=np.array([[1.0e-4]]),
            mean_mass=np.array([[1.0e-10]]),
            density=np.array([[1.0]]),
        )
        box = _box_liq(
            sw=np.array([-0.01]), used_ma_act=np.zeros(1)
        )  # also suppress activation term
        result = activation.func_liqvap_vec(x, box, liq, flagp_r=1, dt=1.0)
        np.testing.assert_array_equal(result.used_mr_vap, [0.0])
        np.testing.assert_array_equal(result.liq_left, [0.0])


# ---------------------------------------------------------------------------
# func_icevap_vec -- golden scalar reference (single point, single bin).
# ---------------------------------------------------------------------------


def _box_ice(**overrides: np.ndarray) -> activation.ActivationBoxState:
    npoints = 1
    defaults = dict(
        mes_rc=np.array([3]),
        t=np.array([280.0]),
        til=np.array([280.0]),
        qr_0=np.zeros(npoints),
        qi_0=np.zeros(npoints),
        gain_mi_rim=np.zeros(npoints),
        gain_mi_frn=np.zeros(npoints),
        den=np.array([1.2e-3]),
        qtp=np.array([1.0e-2]),
        pressure=np.array([1.0e6]),
        qr_b=np.zeros(npoints),
        qi_b=np.zeros(npoints),
        rv=np.array([1.0e-2]),
        sw=np.zeros(npoints),
        used_ma_act=np.zeros(npoints),
        akk_lmt=np.zeros(npoints),
        sw_allact=np.zeros(npoints),
        si_alldep=np.zeros(npoints),
        used_ma_dep=np.zeros(npoints),
        aerosol_dep_con=np.zeros(npoints),
        aerosol_dep_mass=np.zeros(npoints),
        s_c_dhfmin=np.zeros(npoints),
        ds_alldhf=np.zeros(npoints),
        akk_lmt_dhf=np.zeros(npoints),
        used_ma_dhf=np.zeros(npoints),
    )
    defaults.update(overrides)
    return activation.ActivationBoxState(**defaults)


def _ice_bin(**overrides: np.ndarray) -> activation.IceBinState:
    defaults = dict(
        con=np.array([[5.0]]),
        mass_total=np.array([[1.0e-6]]),
        mass_aerosol=np.array([[1.0e-9]]),
        mass_meltwater=np.array([[0.0]]),
        mean_mass=np.array([[2.0e-7]]),
        coef1=np.array([[0.0]]),
        coef2=np.array([[-1.0e-11]]),
        phase2=np.array([[1]]),
        ts_a1=np.array([[1.0]]),
        ts_b11=np.array([[0.0]]),
        ts_b12=np.array([[0.0]]),
        ts_b13=np.array([[0.0]]),
        ts_d1=np.array([[1.0]]),
        tmax=np.array([[300.0]]),
        tmp_prev=np.array([[260.0]]),
        ldmassdt2=np.array([[0.0]]),
        e_l01=np.array([[0.0]]),
    )
    defaults.update(overrides)
    return activation.IceBinState(**defaults)


class TestFuncIcevapVec:
    def test_matches_golden_scalar_reference_growth_no_melt(self) -> None:
        """box.t=280K > T_0 => ts_gate is False => tmp=T_0 trivially (no
        need to replicate the 5-trial Lagrange solve here). coef1=0,
        coef2<0 small => growth term is negative-but-small (shrink, not
        full evaporation) AND the melting bracket (same coef1/coef2,
        scaled by L_e) is comfortably negative => m_w clips to exactly 0
        via max(0, ...), so melt_gate is False and the melting/shed
        machinery is cleanly out of scope for this test."""
        x = np.array([0.02])
        y = np.array([278.0])
        dt = 1.0
        con, mass_total, mass_aerosol = 5.0, 1.0e-6, 1.0e-9
        coef1, coef2 = 0.0, -1.0e-11

        box = _box_ice()
        ice = _ice_bin(
            con=np.array([[con]]),
            mass_total=np.array([[mass_total]]),
            mass_aerosol=np.array([[mass_aerosol]]),
            coef1=np.array([[coef1]]),
            coef2=np.array([[coef2]]),
        )
        flags = activation.ActivationFlags(
            flagp_r=1, flagp_s=1, iflg_inuc=0, iflg_dep=0, iflg_dhf=0, level=1, dt=dt
        )

        result = activation.func_icevap_vec(x, y, box, ice, flags, ESTBAR, ESITBAR)

        # --- golden scalar reference ---
        t_0 = float(AmpsConst.T_0)
        tmp = t_0  # ts_gate False (box.t=280 > T_0)
        e_satw = float(thermo.esat_lk(1, y, ESTBAR, ESITBAR)[0])
        esat_tmp = float(thermo.esat_lk(1, np.array([tmp]), ESTBAR, ESITBAR)[0])  # phase2==1

        d_mean_mass = (coef1 * e_satw / y[0] * (x[0] + 1.0) + coef2 * esat_tmp / tmp) * dt
        water_mass = mass_total - mass_aerosol
        evaporated = water_mass / con + d_mean_mass <= 0.0
        assert not evaporated  # sanity: exercises the "grow/shrink, not evaporated" branch

        used_mi_vapliq_expected = d_mean_mass * con
        ice_left_expected = water_mass + d_mean_mass * con

        l_e = float(AmpsConst.L_e)
        l_f = float(AmpsConst.L_f)
        bracket = l_e * (coef1 * e_satw / y[0] * (x[0] + 1.0) + coef2 * esat_tmp / tmp)
        assert bracket < 0.0  # sanity: melting bracket is negative
        m_w_expected = max(0.0, bracket * dt / l_f)
        assert m_w_expected == 0.0  # sanity: melting cleanly clipped to 0

        np.testing.assert_allclose(result.used_mi_vapliq, [used_mi_vapliq_expected], rtol=1.0e-10)
        np.testing.assert_array_equal(result.used_mi_vap, [0.0])
        np.testing.assert_allclose(result.ice_left, [ice_left_expected], rtol=1.0e-10)
        np.testing.assert_array_equal(result.loss_mi_mlt, [0.0])
        np.testing.assert_array_equal(result.noindep, [1])
        np.testing.assert_array_equal(result.nodhft, [1])
        assert not result.shed_is_placeholder.any()

    def test_flagp_s_off_returns_zeros(self) -> None:
        x = np.array([0.02])
        y = np.array([278.0])
        box = _box_ice()
        ice = _ice_bin()
        flags = activation.ActivationFlags(
            flagp_r=1, flagp_s=0, iflg_inuc=0, iflg_dep=0, iflg_dhf=0, level=1, dt=1.0
        )
        result = activation.func_icevap_vec(x, y, box, ice, flags, ESTBAR, ESITBAR)
        np.testing.assert_array_equal(result.used_mi_vap, [0.0])
        np.testing.assert_array_equal(result.used_mi_vapliq, [0.0])
        np.testing.assert_array_equal(result.ice_left, [0.0])
        np.testing.assert_array_equal(result.noindep, [1])
        np.testing.assert_array_equal(result.nodhft, [1])

    def test_kc04_deposition_and_dhf_branches(self) -> None:
        """The KC04-specific sinks (deposition-freezing "Meyers" path,
        DHF path) -- explicitly highlighted by this task's deliverable
        list. The ice bin itself is deliberately made invalid
        (con=0 < NLMT) so bin_loop1 contributes nothing, isolating these
        two branches."""
        x = np.array([0.05])
        y = np.array([260.0])  # < T_0, needed for both branches' gates

        box = _box_ice(
            t=np.array([260.0]),
            aerosol_dep_con=np.array([100.0]),
            aerosol_dep_mass=np.array([1.0e-6]),
            si_alldep=np.array([0.3]),
            used_ma_dep=np.array([5.0e-8]),
            s_c_dhfmin=np.array([0.02]),
            ds_alldhf=np.array([0.1]),
            akk_lmt_dhf=np.array([0.9]),
            used_ma_dhf=np.array([2.0e-8]),
        )
        ice = _ice_bin(con=np.array([[0.0]]))  # invalid bin -> bin_loop1 no-op
        flags = activation.ActivationFlags(
            flagp_r=1, flagp_s=1, iflg_inuc=1, iflg_dep=2, iflg_dhf=1, level=1, dt=1.0
        )

        result = activation.func_icevap_vec(x, y, box, ice, flags, ESTBAR, ESITBAR)

        e_satw = float(thermo.esat_lk(1, y, ESTBAR, ESITBAR)[0])
        e_sati = float(thermo.esat_lk(2, np.minimum(float(AmpsConst.T_0), y), ESTBAR, ESITBAR)[0])
        r_e = e_satw / e_sati
        xi = r_e * (x[0] + 1.0) - 1.0
        assert xi > 0.0  # sanity: deposition gate condition

        akk_dep_expected = min(1.0, xi / 0.3)
        dep_contribution = akk_dep_expected * 5.0e-8

        akk_dhf_expected = min(1.0, 0.9, (0.05 - 0.02) / 0.1)
        dhf_contribution = akk_dhf_expected * 2.0e-8

        np.testing.assert_allclose(
            result.used_mi_act, [dep_contribution + dhf_contribution], rtol=1.0e-10
        )
        np.testing.assert_array_equal(result.noindep, [0])
        np.testing.assert_array_equal(result.nodhft, [0])
        np.testing.assert_array_equal(result.used_mi_vap, [0.0])
        np.testing.assert_array_equal(result.used_mi_vapliq, [0.0])

    def test_partial_melt_shed_raises_without_opt_in(self) -> None:
        """A bin with substantial ice mass, warm box temperature (so
        melting fires, m_w ~ 2.7e-4 g for these inputs), and a
        `mean_mass` comfortably ABOVE that m_w but still `melt_gate`-
        positive triggers the "partial melt, not whole melt" branch
        (partial-melt-with-shedding) -- must raise unless the caller
        opts in, since ice-shape physics (`cal_semiac_ip`/`get_vip`) is
        not ported anywhere in this codebase (see module docstring)."""
        x = np.array([0.02])
        y = np.array([280.0])
        box = _box_ice(t=np.array([280.0]))
        ice = _ice_bin(
            con=np.array([[1.0]]),
            mass_total=np.array([[1.0e-6]]),
            mass_aerosol=np.array([[0.0]]),
            mean_mass=np.array([[1.0]]),  # >> expected m_w (~2.7e-4) -> NOT whole melt
            coef1=np.array([[1.0e-6]]),  # moderate positive growth => positive m_w
            coef2=np.array([[0.0]]),
        )
        flags = activation.ActivationFlags(
            flagp_r=1, flagp_s=1, iflg_inuc=0, iflg_dep=0, iflg_dhf=0, level=1, dt=1.0
        )

        with pytest.raises(NotImplementedError, match="shed"):
            activation.func_icevap_vec(x, y, box, ice, flags, ESTBAR, ESITBAR)

        result = activation.func_icevap_vec(
            x, y, box, ice, flags, ESTBAR, ESITBAR, allow_shed_placeholder=True
        )
        assert result.shed_is_placeholder[0, 0]
        assert result.loss_mi_mlt[0] == 0.0  # placeholder: no shedding accounted


# ---------------------------------------------------------------------------
# func_vec -- iswitch==2's closed-form residual (independent of the
# ice/liquid physics), plus a basic shape/sanity check for iswitch==0/1.
# ---------------------------------------------------------------------------


class TestFuncVec:
    def test_iswitch2_residual_is_closed_form(self) -> None:
        x = np.array([0.03])
        y = np.array([270.0])
        # used_ma_act=0/sw=0 -> func_liqvap_vec's activation branch is
        # inert, so qr/qi stay exactly 0 and x_n/residual reduce to a
        # closed form this test can check independently.
        box = _box_liq(t=np.array([270.0]), used_ma_act=np.zeros(1), sw=np.zeros(1))
        liq = activation.LiquidBinState(*(np.zeros((1, 1)) for _ in range(8)))
        ice = _ice_bin(con=np.array([[0.0]]))
        flags = activation.ActivationFlags(
            flagp_r=1, flagp_s=1, iflg_inuc=0, iflg_dep=0, iflg_dhf=0, level=1, dt=1.0
        )

        result = activation.func_vec(x, y, 2, box, liq, ice, flags, ESTBAR, ESITBAR)

        e_satw = float(thermo.esat_lk(1, result.y_n, ESTBAR, ESITBAR)[0])
        qv_n = max(0.0, box.qtp[0] - 0.0 - 0.0)
        x_n_expected = box.pressure[0] * qv_n / (float(AmpsConst.Rdvchiarui) + qv_n) / e_satw - 1.0
        expected_residual = -x_n_expected + activation.SW_ALLOW

        np.testing.assert_allclose(result.residual, [expected_residual], rtol=1.0e-10)

    def test_iswitch_invalid_raises(self) -> None:
        x = np.array([0.03])
        y = np.array([270.0])
        box = _box_liq(t=np.array([270.0]))
        liq = activation.LiquidBinState(*(np.zeros((1, 1)) for _ in range(8)))
        ice = _ice_bin(con=np.array([[0.0]]))
        flags = activation.ActivationFlags(
            flagp_r=1, flagp_s=1, iflg_inuc=0, iflg_dep=0, iflg_dhf=0, level=1, dt=1.0
        )
        with pytest.raises(ValueError, match="iswitch"):
            activation.func_vec(x, y, 99, box, liq, ice, flags, ESTBAR, ESITBAR)
