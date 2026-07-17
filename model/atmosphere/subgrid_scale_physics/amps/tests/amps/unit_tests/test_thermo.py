# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import thermo
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.common import dimension as dims


# ---------------------------------------------------------------------------
# Table construction (F1 §3a QSPARM2)
# ---------------------------------------------------------------------------


def _murphy_koop_liquid(t: float) -> float:
    return math.exp(
        54.842763
        - 6763.22 / t
        - 4.210 * math.log(t)
        + 0.000367 * t
        + math.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * math.log(t) + 0.014025 * t)
    )


def _murphy_koop_ice(t: float) -> float:
    return math.exp(9.550426 - 5723.265 / t + 3.53068 * math.log(t) - 0.00728332 * t)


class TestMakeEsatTables:
    def test_shapes(self):
        estbar, esitbar = thermo.make_esat_tables()
        assert estbar.shape == (150,)
        assert esitbar.shape == (111,)

    def test_estbar_first_entry_t164(self):
        estbar, _ = thermo.make_esat_tables()
        expected = _murphy_koop_liquid(164.0)
        assert estbar[0] == pytest.approx(expected, rel=1e-14)

    def test_estbar_last_entry_t313(self):
        estbar, _ = thermo.make_esat_tables()
        expected = _murphy_koop_liquid(313.0)
        assert estbar[149] == pytest.approx(expected, rel=1e-14)

    def test_esitbar_first_entry_t164(self):
        _, esitbar = thermo.make_esat_tables()
        expected = _murphy_koop_ice(164.0)
        assert esitbar[0] == pytest.approx(expected, rel=1e-14)

    def test_esitbar_last_entry_t274(self):
        _, esitbar = thermo.make_esat_tables()
        expected = _murphy_koop_ice(274.0)
        assert esitbar[110] == pytest.approx(expected, rel=1e-14)

    def test_estbar_monotonically_increasing(self):
        estbar, _ = thermo.make_esat_tables()
        assert np.all(np.diff(estbar) > 0)

    def test_esitbar_monotonically_increasing(self):
        _, esitbar = thermo.make_esat_tables()
        assert np.all(np.diff(esitbar) > 0)


# ---------------------------------------------------------------------------
# esat_lk accessor (F1 §3c)
# ---------------------------------------------------------------------------


class TestEsatLk:
    @pytest.fixture(scope="class")
    def tables(self):
        return thermo.make_esat_tables()

    def test_wt_zero_case_t200(self, tables):
        estbar, esitbar = tables
        result = thermo.esat_lk(1, 200.0, estbar, esitbar)
        # 1-based -> 0-based shift: Fortran I = int(200)-163 = 37 (1-based);
        # python idx = I - 1 = int(200)-163-1 = 36.
        expected = estbar[200 - 163 - 1] * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_interpolated_t200_5(self, tables):
        estbar, esitbar = tables
        result = thermo.esat_lk(1, 200.5, estbar, esitbar)
        idx = int(200.5) - 163 - 1  # I-1, I = int(200.5)-163 = 37
        expected = (estbar[idx] * 0.5 + estbar[idx + 1] * 0.5) * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_lower_clamp_t163_2(self, tables):
        estbar, esitbar = tables
        result = thermo.esat_lk(1, 163.2, estbar, esitbar)
        # I = max(1, min(int(163.2)-163, 149)) = max(1, min(0, 149)) = 1
        # wt = clamp(163.2 - 164, 0, 1) = 0
        expected = estbar[0] * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_upper_clamp_t350(self, tables):
        estbar, esitbar = tables
        result = thermo.esat_lk(1, 350.0, estbar, esitbar)
        # I = max(1, min(int(350)-163, 149)) = max(1, min(187, 149)) = 149
        # wt = clamp(350 - (149+163), 0, 1) = clamp(38, 0, 1) = 1
        expected = estbar[149] * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_truncate_vs_clamp_regression_t163_7(self, tables):
        """Named requirement (carry-forward #4): Fortran's truncate-THEN-clamp
        order must be preserved, not clamp-then-truncate. int(163.7)-163=0 ->
        I=max(1,0)=1; wt=max(min(163.7-164,1),0)=0 -> value == estbar[0]*10.
        """
        estbar, esitbar = tables
        result = thermo.esat_lk(1, 163.7, estbar, esitbar)
        expected = estbar[0] * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_ice_phase_interior(self, tables):
        estbar, esitbar = tables
        result = thermo.esat_lk(2, 200.5, estbar, esitbar)
        idx = int(200.5) - 163 - 1
        expected = (esitbar[idx] * 0.5 + esitbar[idx + 1] * 0.5) * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_ice_upper_clamp_t350(self, tables):
        estbar, esitbar = tables
        result = thermo.esat_lk(2, 350.0, estbar, esitbar)
        # J = max(1, min(int(350)-163, 110)) = 110
        expected = esitbar[110] * 10.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_vectorized_matches_scalar(self, tables):
        estbar, esitbar = tables
        t = np.array([163.2, 163.7, 200.0, 200.5, 250.3, 350.0])
        vec = thermo.esat_lk(1, t, estbar, esitbar)
        scalar = np.array([thermo.esat_lk(1, float(ti), estbar, esitbar) for ti in t])
        np.testing.assert_allclose(vec, scalar, rtol=1e-14)

    def test_invalid_phase_raises(self, tables):
        estbar, esitbar = tables
        with pytest.raises(ValueError):
            thermo.esat_lk(3, 200.0, estbar, esitbar)


# ---------------------------------------------------------------------------
# esat_analytic (F1 §3b Lowe & Ficke)
# ---------------------------------------------------------------------------


class TestEsatAnalytic:
    def test_liquid_t273_16(self):
        t = 273.16
        expected = 1000.0 * 6.1070 * math.exp(17.15 * (t - 273.16) / (t - 38.25))
        assert thermo.esat_analytic(1, t) == pytest.approx(expected, rel=1e-14)

    def test_ice_t260(self):
        t = 260.0
        expected = 1000.0 * 6.1064 * math.exp(21.88 * (t - 273.16) / (t - 7.65))
        assert thermo.esat_analytic(2, t) == pytest.approx(expected, rel=1e-14)

    def test_vectorized(self):
        t = np.array([250.0, 270.0, 300.0])
        result = thermo.esat_analytic(1, t)
        expected = np.array(
            [1000.0 * 6.1070 * math.exp(17.15 * (ti - 273.16) / (ti - 38.25)) for ti in t]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            thermo.esat_analytic(3, 250.0)


# ---------------------------------------------------------------------------
# t_from_esat_lk reverse lookup (F1 §3d)
# ---------------------------------------------------------------------------


class TestTFromEsatLk:
    @pytest.fixture(scope="class")
    def tables(self):
        return thermo.make_esat_tables()

    def test_round_trip_liquid(self, tables):
        estbar, esitbar = tables
        esat = thermo.esat_lk(1, 250.3, estbar, esitbar)
        recovered = thermo.t_from_esat_lk(1, 250.0, esat, estbar, esitbar)
        assert recovered == pytest.approx(250.3, abs=1e-9)

    def test_round_trip_ice(self, tables):
        estbar, esitbar = tables
        esat = thermo.esat_lk(2, 200.6, estbar, esitbar)
        recovered = thermo.t_from_esat_lk(2, 200.0, esat, estbar, esitbar)
        assert recovered == pytest.approx(200.6, abs=1e-9)

    def test_round_trip_outside_window_still_found(self, tables):
        """Guess far from the true index (>5 away) exercises the two
        fallback full-scan passes, not just the +-5 window."""
        estbar, esitbar = tables
        esat = thermo.esat_lk(1, 300.4, estbar, esitbar)
        recovered = thermo.t_from_esat_lk(1, 180.0, esat, estbar, esitbar)
        assert recovered == pytest.approx(300.4, abs=1e-9)

    def test_vectorized_round_trip(self, tables):
        estbar, esitbar = tables
        t_true = np.array([200.3, 220.7, 250.1, 300.9])
        esat = thermo.esat_lk(1, t_true, estbar, esitbar)
        recovered = thermo.t_from_esat_lk(1, t_true, esat, estbar, esitbar)
        np.testing.assert_allclose(recovered, t_true, atol=1e-9)

    def test_no_bracket_returns_nan(self, tables):
        estbar, esitbar = tables
        # esat below the entire table's range: no k satisfies the bracket.
        recovered = thermo.t_from_esat_lk(1, 200.0, 0.0, estbar, esitbar)
        assert np.isnan(recovered)


# ---------------------------------------------------------------------------
# Thermo coefficient functions (F1 §3e-3j)
# ---------------------------------------------------------------------------


class TestThermoCoefficients:
    def test_diffusivity(self):
        p, t = 1.0e6, 260.0
        expected = 0.211 * (t / 273.16) ** 1.94 * (1013250.0 / p)
        assert thermo.diffusivity(p, t) == pytest.approx(expected, rel=1e-14)

    def test_thermal_conductivity(self):
        t = 260.0
        expected = (5.69 + 0.017 * (t - 273.16)) * 4.1868 * 1.0e2
        assert thermo.thermal_conductivity(t) == pytest.approx(expected, rel=1e-14)

    def test_dynamic_viscosity_warm_branch(self):
        t = 280.0  # Tc = 6.85 >= 0
        tc = t - 273.15
        expected = (1.718 + 0.0049 * tc) * 1.0e-4
        assert thermo.dynamic_viscosity(t) == pytest.approx(expected, rel=1e-14)

    def test_dynamic_viscosity_cold_branch(self):
        t = 250.0  # Tc = -23.15 < 0
        tc = t - 273.15
        expected = (1.718 + 0.0049 * tc - 1.2e-5 * tc * tc) * 1.0e-4
        assert thermo.dynamic_viscosity(t) == pytest.approx(expected, rel=1e-14)

    def test_sfc_tension_interior(self):
        t = 260.0  # Tc = -13.15, within [-45, 40]
        tc = t - 273.15
        an = (75.93, 0.115, 6.818e-2, 6.511e-3, 2.933e-4, 6.283e-6, 5.285e-8)
        expected = sum(a * tc**i for i, a in enumerate(an))
        assert thermo.sfc_tension(t) == pytest.approx(expected, rel=1e-13)

    def test_sfc_tension_clamps_high(self):
        t = 400.0  # Tc would be 126.85, clamp to 40
        tc = 40.0
        an = (75.93, 0.115, 6.818e-2, 6.511e-3, 2.933e-4, 6.283e-6, 5.285e-8)
        expected = sum(a * tc**i for i, a in enumerate(an))
        assert thermo.sfc_tension(t) == pytest.approx(expected, rel=1e-13)

    def test_sfc_tension_clamps_low(self):
        t = 200.0  # Tc would be -73.15, clamp to -45
        tc = -45.0
        an = (75.93, 0.115, 6.818e-2, 6.511e-3, 2.933e-4, 6.283e-6, 5.285e-8)
        expected = sum(a * tc**i for i, a in enumerate(an))
        assert thermo.sfc_tension(t) == pytest.approx(expected, rel=1e-13)

    def test_gtp_liquid(self):
        t, p, d_v, k_a, es = 260.0, 1.0e6, 0.2, 2.4e5, 2000.0
        r_v = float(AmpsConst.R_v)
        l_e = float(AmpsConst.L_e)
        inv_gtp = r_v * t / (es * d_v) + l_e * (l_e / (r_v * t) - 1.0) / (k_a * t)
        expected = 1.0 / inv_gtp
        assert thermo.gtp(t, p, d_v, k_a, es, 1) == pytest.approx(expected, rel=1e-13)

    def test_gtp_ice(self):
        t, p, d_v, k_a, es = 260.0, 1.0e6, 0.2, 2.4e5, 2000.0
        r_v = float(AmpsConst.R_v)
        l_s = float(AmpsConst.L_s)
        inv_gtp = r_v * t / (es * d_v) + l_s * (l_s / (r_v * t) - 1.0) / (k_a * t)
        expected = 1.0 / inv_gtp
        assert thermo.gtp(t, p, d_v, k_a, es, 2) == pytest.approx(expected, rel=1e-13)

    def test_gtp_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            thermo.gtp(260.0, 1.0e6, 0.2, 2.4e5, 2000.0, 3)

    def test_mod_diffusivity_liquid_uses_local_a_cliq_one(self):
        """NOTE: a_cliq here is the function-local Fortran PARAMETER = 1.0,
        NOT AmpsConst.a_cliq = 0.036 -- see F1 §3j."""
        radius, od_v, t = 1.0e-3, 0.2, 260.0
        pi = float(AmpsConst.PI)
        r_v = float(AmpsConst.R_v)
        del_v = 1.0e-5
        a_c = 1.0
        denom = radius / (radius + del_v) + math.sqrt(2.0 * pi / (r_v * t)) * od_v / (radius * a_c)
        expected = od_v / denom
        assert thermo.mod_diffusivity(1, radius, od_v, t) == pytest.approx(expected, rel=1e-13)

    def test_mod_diffusivity_ice_uses_a_cice1(self):
        radius, od_v, t = 1.0e-3, 0.2, 260.0
        pi = float(AmpsConst.PI)
        r_v = float(AmpsConst.R_v)
        del_v = 1.0e-5
        a_c = 0.5
        denom = radius / (radius + del_v) + math.sqrt(2.0 * pi / (r_v * t)) * od_v / (radius * a_c)
        expected = od_v / denom
        assert thermo.mod_diffusivity(2, radius, od_v, t) == pytest.approx(expected, rel=1e-13)

    def test_mod_thermal_cond(self):
        radius, ok_a, t, den = 1.0e-3, 2.4e5, 260.0, 1.2e-3
        pi = float(AmpsConst.PI)
        m_a = float(AmpsConst.M_a)
        r_u = float(AmpsConst.R_u)
        c_pa = float(AmpsConst.C_pa)
        a_t = 0.96
        del_t = 2.16e-5
        denom = radius / (radius + del_t) + math.sqrt(2.0 * pi * m_a / (r_u * t)) * ok_a / (
            den * radius * a_t * c_pa
        )
        expected = ok_a / denom
        assert thermo.mod_thermal_cond(radius, ok_a, t, den) == pytest.approx(expected, rel=1e-13)

    def test_mgtp_liquid_composes_mod_diffusivity_mod_thermal_cond_gtp(self):
        t, p, radius, d_v, k_a, den, es = 260.0, 1.0e6, 1.0e-3, 0.2, 2.4e5, 1.2e-3, 2000.0
        m_d_v = thermo.mod_diffusivity(1, radius, d_v, t)
        m_k_a = thermo.mod_thermal_cond(radius, k_a, t, den)
        expected = thermo.gtp(t, p, m_d_v, m_k_a, es, 1)
        assert thermo.mgtp(t, p, radius, d_v, k_a, den, es, 1) == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# theta_il closure (F1 §5): cal_til, cal_thetail, diag_t
# ---------------------------------------------------------------------------


class TestThetaIlClosure:
    def test_cal_thetail_matches_formula(self):
        qr, qi, pt, t = 0.001, 0.0005, 9.0e4, 270.0
        p0 = 1.0e5
        l_e, l_s, ra, cp = 2.5e6, 2.8337e6, 287.0, 1004.0
        pit = cp * (pt / p0) ** (ra / cp)
        theta = t * cp / pit
        expected = theta / (1.0 + (l_e * qr + l_s * qi) / (cp * max(t, 253.0)))
        assert thermo.cal_thetail(qr, qi, pt, t, p0=p0) == pytest.approx(expected, rel=1e-13)

    def test_cal_til_matches_formula(self):
        thetail, pt = 268.0, 9.0e4
        p0 = 1.0e5
        ra, cp = 287.0, 1004.0
        pit = cp * (pt / p0) ** (ra / cp)
        expected = pit / cp * thetail
        assert thermo.cal_til(thetail, pt, p0=p0) == pytest.approx(expected, rel=1e-13)

    def test_diag_t_linear_branch(self):
        """P == p00 removes the Exner term; qr=qi=0 removes the heat term
        entirely, so til == thil == T directly (linear branch, T < 253)."""
        thil = 240.0
        p = float(AmpsConst.p00)
        t_out, ierror = thermo.diag_t(thil, p, 0.0, 0.0)
        assert t_out == pytest.approx(240.0, rel=1e-13)
        assert ierror == 0

    def test_diag_t_quadratic_branch_no_condensate(self):
        thil = 260.0
        p = float(AmpsConst.p00)
        t_out, ierror = thermo.diag_t(thil, p, 0.0, 0.0)
        assert t_out == pytest.approx(260.0, rel=1e-13)
        assert ierror == 0

    def test_cal_thetail_diag_t_round_trip_quadratic_branch(self):
        """Full closure round trip per the brief: construct theta_il from a
        known (T, qr, qi) state via cal_thetail, invert with diag_t, assert T
        recovered. cal_thetail (MKS parcel model, F1 §5b) and diag_t (CGS,
        F1 §5a) use *different* (but F1-documented-equivalent) constant
        sets; here they are driven with numerically-consistent equivalents
        (p0 = p00 converted Pa<->CGS via the exact 1 g/(cm s^2) = 0.1 Pa
        factor; cp, r as the exact CGS->MKS conversions of C_pa, R_d; l_e,
        l_s already match exactly between the two constant sets).

        NOTE on what this proves and what it doesn't: with pt == p0 the
        Exner ratio (pt/p0)**(r/cp) is exactly 1.0 on both sides, so this
        case validates FORMULA TRANSCRIPTION only (the heat/(cp*max(T,253))
        closure algebra) -- it says nothing about p0's magnitude, since any
        p0 used consistently forward and backward would round-trip
        identically here. See test_..._pressure_ratio_exercised below for a
        pt != p0 case that genuinely exercises the Exner term (and would
        catch a bug where pt/P were silently ignored, which this degenerate
        case cannot)."""
        t_known = 260.0
        qr, qi = 0.002, 0.001
        p0_pa = 100000.0  # == AmpsConst.p00 (1.0e6 CGS) converted to Pa exactly
        cp_mks = float(AmpsConst.C_pa) * 1.0e-4  # erg/(g K) -> J/(kg K), exact
        r_mks = float(AmpsConst.R_d) * 1.0e-4  # erg/(g K) -> J/(kg K), exact
        l_e_mks = float(AmpsConst.L_e) * 1.0e-4  # erg/g -> J/kg, exact
        l_s_mks = float(AmpsConst.L_s) * 1.0e-4

        thetail = thermo.cal_thetail(
            qr, qi, p0_pa, t_known, p0=p0_pa, cp=cp_mks, r=r_mks, l_e=l_e_mks, l_s=l_s_mks
        )

        p_cgs = float(AmpsConst.p00)  # pt == p0 in Pa <=> P == p00 in CGS
        t_recovered, ierror = thermo.diag_t(thetail, p_cgs, qr, qi)

        assert t_recovered == pytest.approx(t_known, rel=1e-9)
        assert ierror == 0

    def test_cal_thetail_diag_t_round_trip_linear_branch(self):
        """Same construction, but T < 253 K so diag_t takes the linear
        branch. Same pt == p0 caveat as the quadratic-branch case above."""
        t_known = 240.0
        qr, qi = 0.0005, 0.0002
        p0_pa = 100000.0
        cp_mks = float(AmpsConst.C_pa) * 1.0e-4
        r_mks = float(AmpsConst.R_d) * 1.0e-4
        l_e_mks = float(AmpsConst.L_e) * 1.0e-4
        l_s_mks = float(AmpsConst.L_s) * 1.0e-4

        thetail = thermo.cal_thetail(
            qr, qi, p0_pa, t_known, p0=p0_pa, cp=cp_mks, r=r_mks, l_e=l_e_mks, l_s=l_s_mks
        )
        p_cgs = float(AmpsConst.p00)
        t_recovered, ierror = thermo.diag_t(thetail, p_cgs, qr, qi)

        assert t_recovered == pytest.approx(t_known, rel=1e-9)
        assert ierror == 0

    def test_cal_thetail_diag_t_round_trip_pressure_ratio_exercised(self):
        """pt != p0 (pt=8.5e4 Pa vs p0=1.0e5 Pa default), so the Exner ratio
        (pt/p0)**(r/cp) is a genuine non-1.0 factor (~0.955) on both the
        cal_thetail and diag_t sides. Algebraically the ratio still cancels
        exactly for ANY p0 used consistently (theta = T*(p0/pt)**(r/cp),
        then til = thetail*(pt/p0)**(r/cp) recombines to T/(1+heat/(cp*max))
        regardless of p0's value) -- so, like the pt==p0 cases above, this
        still doesn't validate p0's *magnitude* against Fortran output. What
        it adds: it exercises the pressure argument non-trivially, so it
        would catch a bug where cal_thetail/diag_t silently dropped pt/P
        (treated the Exner term as always 1.0), which the pt==p0 cases
        cannot detect since they're degenerate (ratio == 1.0) by
        construction either way."""
        t_known = 260.0
        qr, qi = 0.002, 0.001
        pt_pa = 8.5e4  # != CAL_THETAIL_P0_PA (1.0e5)
        cp_mks = float(AmpsConst.C_pa) * 1.0e-4
        r_mks = float(AmpsConst.R_d) * 1.0e-4
        l_e_mks = float(AmpsConst.L_e) * 1.0e-4
        l_s_mks = float(AmpsConst.L_s) * 1.0e-4

        thetail = thermo.cal_thetail(
            qr, qi, pt_pa, t_known, cp=cp_mks, r=r_mks, l_e=l_e_mks, l_s=l_s_mks
        )

        p_cgs = pt_pa * 10.0  # Pa -> CGS pressure (g/s^2/cm), consistent with pt_pa
        t_recovered, ierror = thermo.diag_t(thetail, p_cgs, qr, qi)

        assert t_recovered == pytest.approx(t_known, rel=1e-9)
        assert ierror == 0


# ---------------------------------------------------------------------------
# DSL deliverable: _esat_lk_dsl (liquid, TILED (Cell,K) table idiom)
# ---------------------------------------------------------------------------


class TestEsatLkDsl:
    NCELLS = 32
    NLEV = 61

    @pytest.fixture(scope="class")
    def estbar(self):
        estbar, _ = thermo.make_esat_tables()
        return estbar

    @pytest.fixture(scope="class")
    def t_field_np(self):
        rng = np.random.default_rng(7)
        return rng.uniform(180.0, 310.0, size=(self.NCELLS, self.NLEV))

    def _run(self, backend, t_field_np, estbar):
        t = gtx.as_field((dims.CellDim, dims.KDim), t_field_np)
        table_tiled_np = thermo.tile_estbar_for_dsl(estbar, self.NCELLS)
        table_tiled = gtx.as_field((dims.CellDim, dims.KDim), table_tiled_np)
        k_index = gtx.as_field(
            (dims.KDim,), np.arange(thermo.ESAT_LK_DSL_TABLE_SIZE, dtype=np.int32)
        )
        out = gtx.as_field((dims.CellDim, dims.KDim), np.zeros_like(t_field_np))

        op = thermo.esat_lk_dsl.with_backend(backend) if backend is not None else thermo.esat_lk_dsl
        op(t, table_tiled, k_index, out=out, offset_provider={})
        return out.asnumpy()

    def test_embedded_matches_numpy(self, t_field_np, estbar):
        estbar_full, esitbar_full = thermo.make_esat_tables()
        expected = thermo.esat_lk(1, t_field_np, estbar_full, esitbar_full)
        got = self._run(None, t_field_np, estbar)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_gtfn_cpu_matches_numpy(self, t_field_np, estbar):
        estbar_full, esitbar_full = thermo.make_esat_tables()
        expected = thermo.esat_lk(1, t_field_np, estbar_full, esitbar_full)
        got = self._run(run_gtfn_cached, t_field_np, estbar)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
