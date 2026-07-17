# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum


class AmpsConst(float, enum.Enum):
    """Physical constants for AMPS microphysics in CGS units.

    All constants are in CGS units unless otherwise noted.
    Derived constants are computed from base constants via expressions.
    """

    # Base constants from mod_amps_const.F90
    p00 = 1.0e6  # reference pressure (g/cm/s^2)
    T_0 = 273.16  # temperature at triple point (K)
    gg = 980.0  # gravity (cm/s^2)
    M_w = 18.016  # molecular weight of water (g/mol)
    R_u = 8.31436e7  # universal gas constant (ergs/deg/g)

    den_w = 1.0  # density of water (g/cm^3)
    den_i = 0.91668  # density of ice, bulk density (g/cm^3)

    R_d = 287.04e4  # gas constant for dry air (ergs/deg/g)
    R_v = 461.5e4  # gas constant for vapor (ergs/deg/g)

    C_pa = 1004.64e4  # heat capacity of dry air (ergs/deg/g)

    c_w = 4.187e5  # heat content of water (ergs/g)

    L_e = 2.5e10  # latent heat of condensation (ergs/g)
    L_f = 0.3337e10  # latent heat of freeze (ergs/g)
    L_s = 2.8337e10  # latent heat of sublimation (ergs/g)

    a_cliq = 0.036  # condensation coefficient for liquid

    k_w = 0.58e5  # thermal conductivity (ergs/s/cm/K)

    undef = 999.9e30  # undefined/missing value

    # Derived constants from mod_amps_const.F90
    MR = M_w / R_u  # M_w / R_u
    Racp = R_d / C_pa  # R_d / C_pa
    Rdvchiarui = R_d / R_v  # R_d / R_v
    M_a = M_w / Rdvchiarui  # M_w / Rdvchiarui

    # Precomputed coefficients from acc_amps.F90
    PI = 3.141592653589793238462643  # pi
    sq_three = 1.7320508075688772935  # sqrt(3)
    coef3s = 5.196152423  # 3*sqrt(3)
    coef4pi3 = 4.18879020478639  # 4*pi/3
    coef2p = 6.28318530717959  # 2*pi
    coefpi6 = 0.523598776  # pi/6
    coefsq2p = 2.506628274631  # sqrt(2*pi)
    coef3sq3 = 5.19615242270663  # 3*sqrt(3)
    coef4p = 1.25663706144e1  # 4*pi
    coef3i4p1i3 = 0.62035049089940  # (3/(4*pi))^(1/3)
    coedpi6 = 0.523598775598299  # pi/6 (double precision)
    coedsq2p = 2.506628274631  # sqrt(2*pi) (double precision)
    coed3sq3 = 5.19615242270663  # 3*sqrt(3) (double precision)
    coefpi180 = 0.0174532925199433  # pi/180
