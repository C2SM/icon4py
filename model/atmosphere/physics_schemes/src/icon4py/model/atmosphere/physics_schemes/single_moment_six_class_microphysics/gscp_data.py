# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Variables used in the grid scale parameterizations (Microphysics).

TODO Cleanup:
- Remove unused constants
- Put back gamma_fct
"""

from typing import Final

import numpy as np
from eve.utils import FrozenNamespace

from icon4py.shared.mo_math_constants import math_const
from icon4py.shared.mo_math_utilities import gamma_fct
from icon4py.shared.mo_physical_constants import phy_const


class GscpData(FrozenNamespace):
    """Constants for the grid scale parameterizations."""

    # Hardcoded switches to select autoconversion and n0s calculation
    # ---------------------------------------------------------------

    iautocon = 1
    isnow_n0temp = 2

    # Epsilons and thresholds
    # -----------------------

    zqmin = 1.0e-15
    """threshold for computations"""
    zeps = 1.0e-15
    """small number"""

    # # More variables
    # # --------------

    # rain_n0_factor = 1.0
    # """COSMO_EU default"""
    # mu_rain = 0.0
    # """COSMO_EU default"""
    # mu_snow = 0.0
    # """COSMO_EU default"""

    # # Even more variables (currently only used in Carmen's scheme for KC05 fall speed)
    # # --------------

    # kc_alpha = 0.5870086
    # """alf, CGS is 0.00739"""
    # kc_beta = 2.45
    # """exponent  in mass-size relation"""
    # kc_gamma = 0.120285
    # """gam, CGS is 0.24"""
    # kc_sigma = 1.85
    # """exponent  in area-size relation"""
    # do_i = 5.83
    # """coefficients for drag correction"""
    # co_i = 0.6
    # """coefficients for turbulence correction"""

    cloud_num = 200.00e06
    """cloud droplet number concentration"""

    # Parameters for autoconversion of cloud water and cloud ice
    # ----------------------------------------------------------

    zccau = 4.0e-4
    """autoconversion coefficient (cloud water to rain)"""
    zciau = 1.0e-3
    """autoconversion coefficient (cloud ice to snow)"""

    zkcau = 9.44e09
    """kernel coeff for SB2001 autoconversion"""
    zkcac = 5.25e00
    """kernel coeff for SB2001 accretion"""
    zcnue = 2.00e00
    """gamma exponent for cloud distribution"""
    zxstar = 2.60e-10
    """separating mass between cloud and rain"""
    zkphi1 = 6.00e02
    """constant in phi-function for autoconversion"""
    zkphi2 = 0.68e00
    """ exponent in phi-function for autoconversion"""
    zkphi3 = 5.00e-05
    """exponent in phi-function for accretion"""

    zhw = 2.270603
    """Howell factor"""
    zecs = 0.9
    """Collection efficiency for snow collecting cloud water"""

    zadi = 0.217
    """Formfactor in the size-mass relation of ice particles"""
    zbdi = 0.302
    """Exponent in the size-mass relation of ice particles"""
    zams = 0.069
    """Formfactor in the mass-size relation of snow particles for cloud ice scheme"""
    zams_gr = 0.069
    """Formfactor in the mass-size relation of snow particles for graupel scheme"""
    zbms = 2.000  # Do not change this, exponent of 2 is hardcoded for isnow_n0temp=2
    """Exponent in the mass-size relation of snow particles"""

    zv1s = 0.50
    """Exponent in the terminal velocity for snow"""

    zami = 130.0
    """Formfactor in the mass-size relation of cloud ice"""
    zn0s0 = 8.0e5
    zn0s1 = 13.5 * 5.65e5
    """parameter in N0S(T)"""
    zn0s2 = -0.107
    """parameter in N0S(T), Field et al"""
    zcac = 1.72
    """(15/32)*(PI**0.5)*(ECR/RHOW)*V0R*AR**(1/8)"""
    zcicri = 1.72
    """(15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)"""
    zcrcri = 1.24e-3
    """(PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)"""
    zcsmel = 1.48e-4
    """4*LHEAT*N0S*AS**(-2/3)/(RHO*lh_f)"""
    zbsmel = 20.32
    """0.26*sqrt(    RHO*v0s/eta)*Gamma(21/8)*AS**(-5/24)"""
    zasmel = 2.43e3
    """DIFF*lh_v*RHO/LHEAT"""

    zcrfrz = 1.68
    """coefficient for raindrop freezing"""
    zcrfrz1 = 9.95e-5
    """FR: 1. coefficient for immersion raindrop freezing: alpha_if"""
    zcrfrz2 = 0.66
    """FR: 2. coefficient for immersion raindrop freezing: a_if"""

    zrho0 = 1.225e0
    """reference air density"""
    zrhow = 1.000e3
    """density of liquid water"""

    zdv = 2.22e-5
    """molecular diffusion coefficient for water vapour"""
    zlheat = 2.40e-2
    """thermal conductivity of dry air"""
    zeta = 1.75e-5
    """kinematic viscosity of air"""

    # Additional parameters
    # ---------------------

    zthet = 248.15
    """temperature for het. nuc. of cloud ice"""
    zthn = 236.15
    """temperature for hom. freezing of cloud water"""
    ztrfrz = 271.15
    """threshold temperature for heterogeneous freezing of raindrops"""
    ztmix = 250.15
    """threshold temperature for mixed-phase cloud freezing of cloud drops (Forbes 2012)"""
    znimax_Thom = 250.0e3
    """FR: maximal number of ice crystals"""
    zmi0 = 1.0e-12
    """initial crystal mass for cloud ice nucleation"""
    zmimax = 1.0e-9
    """maximum mass of cloud ice crystals"""
    zmsmin = 3.0e-9
    """initial mass of snow crystals"""
    zbvi = 0.16
    """v = zvz0i*rhoqi^zbvi"""

    v0snow = 25.0
    """ Sedimentaion velocity of snow """

    v_sedi_rain_min = 0.7
    """minimum terminal fall velocity of rain particles (applied only near the ground) [m/s]"""
    v_sedi_snow_min = 0.1
    """minimum terminal fall velocity of snow particles (applied only near the ground) [m/s]"""
    v_sedi_graupel_min = 0.4
    """minimum terminal fall velocity of graupel particles (applied only near the ground) [m/s]"""

    # Constant exponents in the transfer rate equations
    # -------------------------------------------------

    # DL: TODO. Inline into graupel scheme
    x1o12 = 1.0 / 12.0
    x3o16 = 3.0 / 16.0
    x7o8 = 7.0 / 8.0
    x2o3 = 2.0 / 3.0
    x5o24 = 5.0 / 24.0
    x1o8 = 1.0 / 8.0
    x13o8 = 13.0 / 8.0
    x13o12 = 13.0 / 12.0
    x27o16 = 27.0 / 16.0
    x1o3 = 1.0 / 3.0
    x1o2 = 1.0 / 2.0
    x3o4 = 0.75
    x7o4 = 7.0 / 4.0

    # Parameters relevant to support supercooled liquid water (SLW), sticking efficiency, ...
    # ---------------------------------------------------------------------------------------

    dist_cldtop_ref = 500.0
    """Reference length for distance from cloud top (Forbes 2012)"""
    reduce_dep_ref = 0.1
    """lower bound on snow/ice deposition reduction"""
    zceff_fac = 3.5e-3
    """Scaling factor [1/K] for temperature-dependent cloud ice sticking efficiency"""
    tmin_iceautoconv = 188.15
    """Temperature at which cloud ice autoconversion starts"""


gscp_data: Final = GscpData()


class GscpCoefficients(FrozenNamespace):
    """Workaround until we can pass scalars to scan()."""

    zdt = 600.0  #
    zceff_min = 0.075
    v0snow = 20.0
    zvz0i = 1.25
    mu_rain = 0.0
    rain_n0_factor = 1.0
    icesedi_exp = 0.33

    zconst = (
        gscp_data.zkcau
        / (20.0 * gscp_data.zxstar)
        * (gscp_data.zcnue + 2.0)
        * (gscp_data.zcnue + 4.0)
        / (gscp_data.zcnue + 1.0) ** 2
    )
    ccsrim = 46.98276746732905
    ccsagg = 52.20307496369895
    ccsdep = 99.96257414250321
    ccsvxp = -(gscp_data.zv1s / (gscp_data.zbms + 1.0) + 1.0)
    ccsvel = 46.23061746664488
    ccsvxp = ccsvxp + 1.0
    ccslam = 0.1379999999844696
    ccslxp = 1.0 / (gscp_data.zbms + 1.0)
    ccswxp = 0.1666666666666667
    ccsaxp = -(gscp_data.zv1s + 3.0)
    ccsdxp = -(gscp_data.zv1s + 1.0) / 2.0
    ccshi1 = phy_const.als**2 / (gscp_data.zlheat * phy_const.rv)
    ccdvtp = 4.2213489078749271e-5
    ccidep = 4.0 * gscp_data.zami ** (-gscp_data.x1o3)
    zn0r = 8.0e6 * np.exp(3.2 * mu_rain) * (0.01) ** (-mu_rain)
    zn0r = zn0r * rain_n0_factor
    zcevxp = (mu_rain + 2.0) / (mu_rain + 4.0)
    zcev = 3.0999999914053263e-3
    zbevxp = (2.0 * mu_rain + 5.5) / (2.0 * mu_rain + 8.0) - zcevxp
    zbev = 14.152467883390491
    zvzxp = 0.5 / (mu_rain + 4.0)
    zvz0r = 12.63008787548925

    # DL: Temporary workaround since I cant pass 2D constants
    qi0 = -2.91727579e-08
    qc0 = -3.90868735e-08


gscp_coefficients: Final = GscpCoefficients()


def gscp_set_coefficients(
    igscp,
    zceff_min=0.075,
    v0snow=20.0,
    zvz0i=1.25,
    mu_rain=0.0,
    rain_n0_factor=1.0,
    icesedi_exp=0.33,
    idbg=0,
):
    """Calculate some coefficients for the microphysics schemes.  Usually called only once at model startup.

    Default for optional parameters:
    Values from COSMO
    """

    # For compatilibility TODO: Cleanup
    if v0snow <= -1:
        if igscp == 2:
            v0snow = 20.0
        else:
            assert "Only inwp_gscp supported"

    zconst = (
        gscp_data.zkcau
        / (20.0 * gscp_data.zxstar)
        * (gscp_data.zcnue + 2.0)
        * (gscp_data.zcnue + 4.0)
        / (gscp_data.zcnue + 1.0) ** 2
    )
    ccsrim = (
        0.25 * math_const.pi * gscp_data.zecs * v0snow * gamma_fct(gscp_data.zv1s + 3.0)
    )
    ccsagg = (
        0.25 * math_const.pi * v0snow * gamma_fct(gscp_data.zv1s + 3.0)
    )  # DL: off by -7.105427357601002e-15 (last 2 digits) FORTRAN value is 52.20307496369895
    ccsdep = (
        0.26 * gamma_fct((gscp_data.zv1s + 5.0) / 2.0) * np.sqrt(1.0 / gscp_data.zeta)
    )
    ccsvxp = -(gscp_data.zv1s / (gscp_data.zbms + 1.0) + 1.0)
    ccsvel = (
        gscp_data.zams
        * v0snow
        * gamma_fct(gscp_data.zbms + gscp_data.zv1s + 1.0)
        * (gscp_data.zams * gamma_fct(gscp_data.zbms + 1.0)) ** ccsvxp
    )  # 46.23061746664488
    ccsvxp = ccsvxp + 1.0
    ccslam = gscp_data.zams * gamma_fct(gscp_data.zbms + 1.0)
    ccslxp = 1.0 / (gscp_data.zbms + 1.0)
    ccswxp = (
        gscp_data.zv1s * ccslxp
    )  # Off by -5.551115123125783e-17 original FORTRAN is 0.1666666666666667
    ccsaxp = -(gscp_data.zv1s + 3.0)
    ccsdxp = -(gscp_data.zv1s + 1.0) / 2.0
    ccshi1 = phy_const.als**2 / (gscp_data.zlheat * phy_const.rv)
    ccdvtp = 2.22e-5 * phy_const.tmelt ** (-1.94) * 101325.0
    ccidep = 4.0 * gscp_data.zami ** (-gscp_data.x1o3)
    zn0r = (
        8.0e6 * np.exp(3.2 * mu_rain) * (0.01) ** (-mu_rain)
    )  # empirical relation adapted from Ulbrich (1983)
    zn0r = zn0r * rain_n0_factor  # apply tuning factor to zn0r variable
    zar = (
        math_const.pi * gscp_data.zrhow / 6.0 * zn0r * gamma_fct(mu_rain + 4.0)
    )  # pre-factor in lambda of rain
    zcevxp = (mu_rain + 2.0) / (mu_rain + 4.0)

    zcev = (
        2.0
        * math_const.pi
        * gscp_data.zdv
        / gscp_data.zhw
        * zn0r
        * zar ** (-zcevxp)
        * gamma_fct(mu_rain + 2.0)
    )  # off by 2.168404344971009e-18 FORTRAN: 3.0999999914053263e-3
    zbevxp = (2.0 * mu_rain + 5.5) / (2.0 * mu_rain + 8.0) - zcevxp
    zbev = (
        0.26
        * np.sqrt(gscp_data.zrho0 * 130.0 / gscp_data.zeta)
        * zar ** (-zbevxp)
        * gamma_fct((2.0 * mu_rain + 5.5) / 2.0)
        / gamma_fct(mu_rain + 2.0)
    )  # 14.152467883390491

    zvzxp = 0.5 / (mu_rain + 4.0)  # DL: same
    zvz0r = (
        130.0 * gamma_fct(mu_rain + 4.5) / gamma_fct(mu_rain + 4.0) * zar ** (-zvzxp)
    )  #  Off by 2.6645352591003757e-14 FORTRAN is 12.63008787548925

    return (
        ccsrim,
        ccsagg,
        ccsdep,
        ccsvel,
        ccsvxp,
        ccslam,
        ccslxp,
        ccsaxp,
        ccsdxp,
        ccshi1,
        ccdvtp,
        ccidep,
        ccswxp,
        zconst,
        zcev,
        zbev,
        float(zcevxp),
        float(zbevxp),
        float(zvzxp),
        zvz0r,
        float(zvz0i),
        float(icesedi_exp),
        float(zceff_min),
    )
