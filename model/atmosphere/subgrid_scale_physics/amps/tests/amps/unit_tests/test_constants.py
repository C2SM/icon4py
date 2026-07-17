# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst


class TestAmpsConst:
    """Test physical constants from mod_amps_const and acc_amps."""

    def test_p00(self):
        """Reference pressure (g/cm/s^2)."""
        assert AmpsConst.p00 == 1.0e6

    def test_T_0(self):
        """Temperature at triple point (K)."""
        assert AmpsConst.T_0 == 273.16

    def test_gg(self):
        """Gravity (cm/s^2)."""
        assert AmpsConst.gg == 980.0

    def test_M_w(self):
        """Molecular weight of water (g/mol)."""
        assert AmpsConst.M_w == 18.016

    def test_R_u(self):
        """Universal gas constant (ergs/deg/g)."""
        assert AmpsConst.R_u == 8.31436e7

    def test_MR(self):
        """Derived: M_w / R_u."""
        assert AmpsConst.MR == 18.016 / 8.31436e7

    def test_den_w(self):
        """Density of water (g/cm^3)."""
        assert AmpsConst.den_w == 1.0

    def test_den_i(self):
        """Density of ice bulk density (g/cm^3)."""
        assert AmpsConst.den_i == 0.91668

    def test_R_d(self):
        """Gas constant for dry air (ergs/deg/g)."""
        assert AmpsConst.R_d == 287.04e4

    def test_R_v(self):
        """Gas constant for vapor (ergs/deg/g)."""
        assert AmpsConst.R_v == 461.5e4

    def test_C_pa(self):
        """Heat capacity of dry air (ergs/deg/g)."""
        assert AmpsConst.C_pa == 1004.64e4

    def test_Racp(self):
        """Derived: R_d / C_pa."""
        assert AmpsConst.Racp == 287.04e4 / 1004.64e4

    def test_Rdvchiarui(self):
        """Derived: R_d / R_v."""
        assert AmpsConst.Rdvchiarui == 287.04e4 / 461.5e4

    def test_M_a(self):
        """Derived: M_w / Rdvchiarui."""
        expected = 18.016 / (287.04e4 / 461.5e4)
        assert AmpsConst.M_a == expected

    def test_c_w(self):
        """Heat content of water (ergs/g)."""
        assert AmpsConst.c_w == 4.187e5

    def test_L_e(self):
        """Latent heat of condensation (ergs/g)."""
        assert AmpsConst.L_e == 2.5e10

    def test_L_f(self):
        """Latent heat of freeze (ergs/g)."""
        assert AmpsConst.L_f == 0.3337e10

    def test_L_s(self):
        """Latent heat of sublimation (ergs/g)."""
        assert AmpsConst.L_s == 2.8337e10

    def test_a_cliq(self):
        """Condensation coefficient for liquid."""
        assert AmpsConst.a_cliq == 0.036

    def test_k_w(self):
        """Thermal conductivity (ergs/s/cm/K)."""
        assert AmpsConst.k_w == 0.58e5

    def test_undef(self):
        """Undefined/missing value flag."""
        assert AmpsConst.undef == 999.9e30

    def test_PI(self):
        """Pi constant."""
        assert AmpsConst.PI == 3.141592653589793238462643

    def test_sq_three(self):
        """sqrt(3)."""
        assert AmpsConst.sq_three == 1.7320508075688772935

    def test_coef3s(self):
        """3 * sqrt(3)."""
        assert AmpsConst.coef3s == 5.196152423

    def test_coef4pi3(self):
        """4*pi/3."""
        assert AmpsConst.coef4pi3 == 4.18879020478639

    def test_coef2p(self):
        """2*pi."""
        assert AmpsConst.coef2p == 6.28318530717959

    def test_coefpi6(self):
        """pi/6."""
        assert AmpsConst.coefpi6 == 0.523598776

    def test_coefsq2p(self):
        """sqrt(2*pi)."""
        assert AmpsConst.coefsq2p == 2.506628274631

    def test_coef3sq3(self):
        """3*sqrt(3)."""
        assert AmpsConst.coef3sq3 == 5.19615242270663

    def test_coef4p(self):
        """4*pi."""
        assert AmpsConst.coef4p == 1.25663706144e1

    def test_coef3i4p1i3(self):
        """(3/(4*pi))^(1/3)."""
        assert AmpsConst.coef3i4p1i3 == 0.62035049089940

    def test_coedpi6(self):
        """pi/6 (double precision)."""
        assert AmpsConst.coedpi6 == 0.523598775598299

    def test_coedsq2p(self):
        """sqrt(2*pi) (double precision)."""
        assert AmpsConst.coedsq2p == 2.506628274631

    def test_coed3sq3(self):
        """3*sqrt(3) (double precision)."""
        assert AmpsConst.coed3sq3 == 5.19615242270663

    def test_coefpi180(self):
        """pi/180."""
        assert AmpsConst.coefpi180 == 0.0174532925199433
