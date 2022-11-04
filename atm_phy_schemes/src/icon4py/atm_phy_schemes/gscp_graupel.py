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
"""Prognostic one-moment bulk microphysical parameterization.

Calculate the rates of change of temperature, cloud
water, cloud ice, water vapor, rain, snow and graupel due to cloud microphysical
processes related to the formation of grid scale precipitation. This
includes the sedimentation of rain and snow. The precipitation fluxes at
the surface are also calculated here.

Method:
  Prognostic one-moment bulk microphysical parameterization.
  The sedimentation of rain and snow is computed implicitly.

Equation numbers refer to
Doms, Foerstner, Heise, Herzog, Raschendorfer, Schrodin, Reinhardt, Vogel
(September 2005): "A Description of the Nonhydrostatic Regional Model LM",

TODO: Removed Features
1. lsuper_coolw = False, lred_depgrow = False, lsedi_ice = False
2. isnow_n0temp == 1, and zn0s = const.
3. iautocon == 0 (Kessler)

TODO: David
    1. Test if runnign with GPU backend. normalize
    2. Replace exp(A * log(B)) by B**A. Needs performance check and maybe optimization pass.
    3. Put scan in field operator. --> Disregard *_
    4. Remove workaround for qnc (--> scheme does validate!!!), qc0, qi0 from gscp_data.py and pass explicitly
    5. Remove namespacing, i.e. z and c prefixes
    6. Replace 2D Fields by 1D fields qnc, prr_gsp et al.
    7. Do we really wanna diagnose pri_gsp ?

TODO: GT4Py team
    1. Call field operators from scan --> sat_pres_ice
    2. Re-interate into FORTRAN
"""
# DL. Set somewhere else
import sys
from typing import Final

from eve.utils import FrozenNamespace
from functional.ffront.decorator import program, scan_operator
from functional.ffront.fbuiltins import (
    Field,
    abs,
    exp,
    int32,
    log,
    maximum,
    minimum,
    sqrt,
)

from icon4py.atm_phy_schemes.gscp_data import gscp_coefficients, gscp_data
from icon4py.atm_phy_schemes.mo_convect_tables import conv_table
from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import phy_const


# DL TODO
sys.setrecursionlimit(2000)


class GraupelParametersAndConfiguration(FrozenNamespace):
    """Configuration and local parameters of the graupel scheme."""

    # Configuration
    lsedi_ice = True  # sedimentation of cloud ice (Heymsfield & Donner 1990 *1/3)
    lstickeff = True  # switch for sticking coeff. (work from Guenther Zaengl)
    # lsuper_coolw = True  # switch for supercooled liquid water (work from Felix Rieper)
    # lred_depgrow = True  # separate switch for reduced depositional growth near tops of water clouds (now also used in ICON after correcting the cloud top diagnosis)

    # Local Parameters
    zcsg = 0.5  # coefficient for snow-graupel conversion by riming
    zcrim_g = 4.43
    zrimexp_g = 0.94878
    zcagg_g = 2.46
    zasmel = 2.95e3  # DIFF*lh_v*RHO/LHEAT
    zexpsedg = 0.217  # exponent for graupel sedimentation
    zvz0g = 12.24  # coefficient of sedimentation velocity for graupel
    ztcrit = 3339.5  # factor in calculation of critical temperature


# Statement functions
# -------------------


def fpvsw(ztx):
    """Return saturation vapour pressure over water from temperature."""
    return conv_table.c1es * exp(
        conv_table.c3les * (ztx - conv_table.tmelt) / (ztx - conv_table.c4les)
    )


def fxna(ztx):  # DL: Remove
    """Return number of activate ice crystals from temperature."""
    return 1.0e2 * exp(0.2 * (phy_const.tmelt - ztx))


def fxna_cooper(ztx):  # Dl: Rename
    """Return number of activate ice crystals from temperature.

    Method: Cooper (1986) used by Greg Thompson(2008)
    """
    return 5.0 * exp(0.304 * (phy_const.tmelt - ztx))


def make_normalized(v):
    """
    Set denormals to zero.

    GPU code can't flush to zero double precision denormals. To avoid CPU-GPU differences we'll do it manually.
    TODO: Add pass that replaces exact IF by soft IF that tresholds on denormal.
    """
    # if GT4PyConfig.gpu: #TODO: Test if running with GPU backend
    v = 0.0 if abs(v) <= 2.225073858507201e-308 else v
    return v


local_param: Final = GraupelParametersAndConfiguration()


@scan_operator(
    axis=KDim,
    forward=True,
    init=(
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),  # DL TODO: Use ellipsis operator?
)
def _graupel(
    carry: tuple[
        # float,
        # float,
        # float,
        # float,
        # float,
        # float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],  # DL TODO: Use ellipsis operator?
    # zdt: float,  # time step
    dz: float,  # level thickness
    # Prognostics
    temp: float,
    pres: float,
    rho: float,
    qv: float,
    qi: float,
    qc: float,
    qr: float,
    qs: float,
    qg: float,
    # Number Densities
    qnc: float,  # 2D Field
    #   qi0: float,
    #   qc0: float,
    # Precipitation Fluxes
    pri_gsp: float,  # 2D Field
    prr_gsp: float,  # 2D Field
    prs_gsp: float,  # 2D Field
    prg_gsp: float,  # 2D Field
    # # Precomputed Parameters
    # ccsrim: float,
    # ccsagg: float,
    # ccsdep: float,
    # ccsvel: float,
    # ccsvxp: float,
    # ccslam: float,
    # ccslxp: float,
    # ccsaxp: float,
    # ccsdxp: float,
    # ccshi1: float,
    # ccdvtp: float,
    # ccidep: float,
    # ccswxp: float,
    # zconst: float,
    # zcev: float,
    # zbev: float,
    # zcevxp: float,
    # zbevxp: float,
    # zvzxp: float,
    # zvz0r: float,
    # zvz0i: float,
    # icesedi_exp: float,
    # zceff_min: float,
    # Optional Fields: TODO: Pass optional fields to program
    # ithermo_water: int32,  # TODO: Pass int to prgram
    ddt_tend_t: float,
    ddt_tend_qv: float,
    ddt_tend_qc: float,
    ddt_tend_qi: float,
    ddt_tend_qr: float,
    ddt_tend_qs: float,
    # ddt_tend_qg: float, # DL: Missing from FORTRAN interface
    is_surface: bool,
    # Option Switches
    # l_cv: bool = False,  # if true, cv is used instead of cp
    # lpres_pri: bool = False,  # if true, precipitation rate of ice shall be diagnosed
    # lldiag_ttend: bool = False,  # if true, temperature tendency shall be diagnosed
    # lldiag_qtend: bool = False,  # if true, moisture tendencies shall be diagnosed
):

    # ------------------------------------------------------------------------------
    #  Section 1: Initial setting of local and global variables
    # ------------------------------------------------------------------------------

    # unpack carry
    (
        # qv_kminus1,
        # qc_kminus1,
        # qi_kminus1,
        # qr_kminus1,
        # qs_kminus1,
        # qg_kminus1,
        zpkr_kminus1,
        zpks_kminus1,
        zpkg_kminus1,
        zpki_kminus1,
        zprvr_kminus1,
        zprvs_kminus1,
        zprvg_kminus1,
        zprvi_kminus1,
        zvzr_kminus1,
        zvzs_kminus1,
        zvzi_kminus1,
        zvzg_kminus1,
        dist_cldtop_kminus1,
        zqvsw_up_kminus1,
    ) = carry

    # # Some constant coefficients
    # znimax = znimax_Thom # number of ice crystals at temp threshold for mixed-phase clouds
    # znimix = fxna_cooper(ztmix)
    znimax = gscp_data.znimax_Thom
    znimix = 5.0 * exp(0.304 * (phy_const.tmelt - gscp_data.ztmix))

    # # Precomputations for optimization TODO: Could be done by the GT4PY optimizers?
    # zpvsw0 = fpvsw(t0)  # sat. vap. pressure for t = t0 #DL: This computes to just c3les, no?
    zpvsw0 = conv_table.c1es * exp(
        conv_table.c3les
        * (phy_const.tmelt - phy_const.tmelt)
        / (phy_const.tmelt - conv_table.c4les)
    )
    zlog_10 = log(10.0)  # Natural logarithm  of 10

    ccswxp_ln1o2 = exp(gscp_coefficients.ccswxp * log(0.5))
    zvzxp_ln1o2 = exp(gscp_coefficients.zvzxp * log(0.5))
    zbvi_ln1o2 = exp(gscp_data.zbvi * log(0.5))
    zexpsedg_ln1o2 = exp(local_param.zexpsedg * log(0.5))

    # timestep for calculations
    zdtr = 1.0 / gscp_coefficients.zdt

    # ----------------------------------------------------------------------------
    # Section 2: Check for existence of rain and snow
    #            Initialize microphysics and sedimentation scheme
    # ----------------------------------------------------------------------------

    # DL: Done in ternaly below
    # zcrim = 0.0
    # zcagg = 0.0
    # zbsdep = 0.0
    # zvz0s = 0.0

    zn0s = gscp_data.zn0s0  # TODO: Cleanup
    reduce_dep = 1.0  # FR: Reduction coeff. for dep. growth of rain and ice TODO: Uncomment, cleanup

    # ----------------------------------------------------------------------------
    # 2.1: Preparations for computations and to check the different conditions
    # ----------------------------------------------------------------------------

    # DL: In F90 code, this also serves as scalar replacement at the same time
    # qrg = make_normalized(qr)
    # qsg = make_normalized(qs)
    # qgg = make_normalized(qg)
    # qvg = make_normalized(qv)
    # qcg = make_normalized(qc)
    # qig = make_normalized(qi)
    # tg = make_normalized(t)
    # ppg = make_normalized(p)
    # rhog = make_normalized(rho)
    qrg = qr
    qsg = qs
    qgg = qg
    qvg = qv
    qcg = qc
    qig = qi
    tg = temp
    ppg = pres
    rhog = rho

    # ..for density correction of fall speeds
    z1orhog = 1.0 / rhog
    hlp = log(gscp_data.zrho0 * z1orhog)
    zrho1o2 = exp(hlp * gscp_data.x1o2)
    zrhofac_qi = exp(hlp * gscp_coefficients.icesedi_exp)

    zqrk = qrg * rhog
    zqsk = qsg * rhog
    zqgk = qgg * rhog
    zqik = qig * rhog

    # DL: Masks are precalculated (move back? gt4pys job)
    llqr = True if zqrk > gscp_data.zqmin else False
    llqs = True if zqsk > gscp_data.zqmin else False
    llqg = True if zqgk > gscp_data.zqmin else False
    llqi = True if zqik > gscp_data.zqmin else False

    zdtdh = 0.5 * gscp_coefficients.zdt / dz

    # # DL:  2D arrays that accumulate in k
    zzar = zqrk / zdtdh + zprvr_kminus1 + zpkr_kminus1
    zzas = zqsk / zdtdh + zprvs_kminus1 + zpks_kminus1
    zzag = zqgk / zdtdh + zprvg_kminus1 + zpkg_kminus1
    zzai = zqik / zdtdh + zprvi_kminus1 + zpki_kminus1

    # Reset terminal fall speeds
    zpkr = 0.0
    # zpks = 0.0
    zpkg = 0.0
    zpki = 0.0

    # -------------------------------------------------------------------------
    # qs_prepare:
    # -------------------------------------------------------------------------

    # DL: TODO Refactor: Move into function
    ztc = maximum(minimum(tg - phy_const.tmelt, 0.0), -40.0)
    nnr = 3.0

    hlp = (
        5.065339
        - 0.062659 * ztc
        - 3.032362 * nnr
        + 0.029469 * ztc * nnr
        - 0.000285 * ztc**2
        + 0.312550 * nnr**2
        + 0.000204 * ztc**2 * nnr
        + 0.003199 * ztc * nnr**2
        + 0.000000 * ztc**3
        - 0.015952 * nnr**3
    )
    alf = exp(hlp * zlog_10)
    bet = (
        0.476221
        - 0.015896 * ztc
        + 0.165977 * nnr
        + 0.007468 * ztc * nnr
        - 0.000141 * ztc**2
        + 0.060366 * nnr**2
        + 0.000079 * ztc**2 * nnr
        + 0.000594 * ztc * nnr**2
        + 0.000000 * ztc**3
        - 0.003577 * nnr**3
    )

    # Here is the exponent bms=2.0 hardwired! not ideal! (Uli Blahak)
    m2s = qsg * rhog / gscp_data.zams  # UB rho added as bugfix
    m3s = alf * exp(bet * log(m2s))

    hlp = gscp_data.zn0s1 * exp(gscp_data.zn0s2 * ztc)
    zn0s = 13.50 * m2s * (m2s / m3s) ** 3  # DL: **3 super bad for validation
    zn0s = maximum(zn0s, 0.5 * hlp)
    zn0s = minimum(zn0s, 1.0e2 * hlp)
    zn0s = minimum(zn0s, 1.0e9)
    zn0s = maximum(zn0s, 1.0e6)

    (zcrim, zcagg, zbsdep, zvz0s) = (
        (
            gscp_coefficients.ccsrim * zn0s,
            gscp_coefficients.ccsagg * zn0s,
            gscp_coefficients.ccsdep * sqrt(gscp_coefficients.v0snow),
            gscp_coefficients.ccsvel * exp(gscp_coefficients.ccsvxp * log(zn0s)),
        )
        if llqs
        else (0.0, 0.0, 0.0, 0.0)
    )

    # Alternative implementaion
    # zcrim = ccsrim * zn0s if llqs else 0.0
    # zcagg = ccsagg * zn0s if llqs else 0.0
    # zbsdep = ccsdep * sqrt(v0snow) if llqs else 0.0
    # zvz0s = ccsvel * exp(ccsvxp * log(zn0s)) if llqs else 0.0

    # sedimentation fluxes
    # -------------------------------------------------------------------------
    # qs_sedi:
    # -------------------------------------------------------------------------
    # if llqs:
    zlnqsk = (
        zvz0s * exp(gscp_coefficients.ccswxp * log(zqsk)) * zrho1o2 if llqs else 0.0
    )

    # Prevent terminal fall speed of snow from being zero at the surface level
    zlnqsk = (
        maximum(zlnqsk, gscp_data.v_sedi_snow_min)
        if is_surface
        else zlnqsk
        if llqs
        else 0.0
    )

    zpks = zqsk * zlnqsk if llqs else 0.0
    zvzs = (
        zlnqsk * ccswxp_ln1o2 if zvzs_kminus1 == 0.0 else zvzs_kminus1 if llqs else 0.0
    )

    # -------------------------------------------------------------------------
    # qr_sedi:
    # -------------------------------------------------------------------------
    # if llqr:
    zlnqrk = (
        gscp_coefficients.zvz0r * exp(gscp_coefficients.zvzxp * log(zqrk)) * zrho1o2
        if llqr
        else 0.0
    )

    # Prevent terminal fall speed of rain from being zero at the surface level
    zlnqrk = (
        maximum(zlnqrk, gscp_data.v_sedi_rain_min)
        if is_surface
        else zlnqrk
        if llqr
        else 0.0
    )
    zpkr = zqrk * zlnqrk if llqr else 0.0

    zvzr = (
        zlnqrk * zvzxp_ln1o2 if zvzr_kminus1 == 0.0 else zvzr_kminus1 if llqr else 0.0
    )

    # -------------------------------------------------------------------------
    # qg_sedi:
    # -------------------------------------------------------------------------
    # if llqg:

    zlnqgk = (
        local_param.zvz0g * exp(local_param.zexpsedg * log(zqgk)) * zrho1o2
        if llqg
        else 0.0
    )

    # Prevent terminal fall speed of graupel from being zero at the surface level
    zlnqgk = (
        maximum(zlnqgk, gscp_data.v_sedi_graupel_min)
        if is_surface
        else zlnqgk
        if llqg
        else 0.0
    )
    zpkg = zqgk * zlnqgk if llqg else 0.0

    zvzg = (
        zlnqgk * zexpsedg_ln1o2
        if zvzg_kminus1 == 0.0
        else zvzg_kminus1
        if llqg
        else 0.0
    )

    # -------------------------------------------------------------------------
    # qi_sedi:
    # -------------------------------------------------------------------------
    # IF llqi:
    zlnqik = (
        gscp_coefficients.zvz0i * exp(gscp_data.zbvi * log(zqik)) * zrhofac_qi
        if llqi
        else 0.0
    )
    zpki = zqik * zlnqik if llqi else 0.0

    zvzi = zlnqik * zbvi_ln1o2 if zvzi_kminus1 == 0.0 else zvzi_kminus1 if llqi else 0.0

    # -------------------------------------------------------------------------
    # Prevent terminal fall speeds from being zero at the surface level
    # -------------------------------------------------------------------------

    zvzr = maximum(zvzr, gscp_data.v_sedi_rain_min) if is_surface else zvzr
    zvzs = maximum(zvzs, gscp_data.v_sedi_snow_min) if is_surface else zvzs
    zvzg = maximum(zvzg, gscp_data.v_sedi_graupel_min) if is_surface else zvzg

    # --------------------------------------------------------------------------
    # 2.3: Second part of preparations
    # --------------------------------------------------------------------------

    # zeln7o8qrk = 0.0 #DL: Implemented below in IF (llqr): ic1
    # zeln7o4qrk = 0.0 # Dl: Implemented below in IF (llqr): ic1
    # zeln27o16qrk = 0.0 #DL: Implemented below in IF (llqr): ic1
    # zeln13o8qrk = 0.0 #DL: Implemented below in IF (llqr): ic1
    # zeln3o4qsk = 0.0 #DL: Implemented below 2.5: IF (llqs): ic2
    # zeln8qsk = 0.0 #DL: Implemented below 2.5: IF (llqs): ic2
    # zeln6qgk = 0.0 #DL: implemnted below in IF (llqg): ic3
    # zelnrimexp_g = 0.0 #DL: implemnted below in IF (llqg): ic3
    # zsrmax = 0.0 #DL: Implemented below IF (llqr): ic1
    # zssmax = 0.0 #DL: Implemented below 2.5: IF (llqs): ic2
    # zsgmax = 0.0 #DL: implemnted below in IF (llqg): ic3

    scau = 0.0
    scac = 0.0
    # snuc = 0.0 # DL: Implemented below: 2.8
    scfrz = 0.0
    simelt = 0.0
    sidep = 0.0
    ssdep = 0.0
    sgdep = 0.0
    sdau = 0.0
    srim = 0.0
    srim2 = 0.0
    sshed = 0.0
    sicri = 0.0
    srcri = 0.0
    sagg = 0.0
    sagg2 = 0.0
    siau = 0.0
    ssmelt = 0.0
    sgmelt = 0.0
    sev = 0.0
    sconr = 0.0
    sconsg = 0.0
    srfrz = 0.0

    zpkr = minimum(zpkr, zzar)
    zpks = minimum(zpks, zzas)
    zpkg = minimum(zpkg, maximum(0.0, zzag))
    zpki = minimum(zpki, zzai)

    zzar = zdtdh * (zzar - zpkr)
    zzas = zdtdh * (zzas - zpks)
    zzag = zdtdh * (zzag - zpkg)
    zzai = zdtdh * (zzai - zpki)

    zimr = 1.0 / (1.0 + zvzr * zdtdh)
    zims = 1.0 / (1.0 + zvzs * zdtdh)
    zimg = 1.0 / (1.0 + zvzg * zdtdh)
    zimi = 1.0 / (1.0 + zvzi * zdtdh)

    zqrk = zzar * zimr
    zqsk = zzas * zims
    zqgk = zzag * zimg
    zqik = zzai * zimi

    # zqvsi = sat_pres_ice(tg) / (rhog * r_v * tg) #DL: Todo: Replace
    sat_pres_ice = conv_table.c1es * exp(
        conv_table.c3ies * (tg - phy_const.tmelt) / (tg - conv_table.c4ies)
    )
    zqvsi = sat_pres_ice / (rhog * phy_const.rv * tg)

    llqr = zqrk > gscp_data.zqmin
    llqs = zqsk > gscp_data.zqmin
    llqg = zqgk > gscp_data.zqmin
    llqc = qcg > gscp_data.zqmin
    llqi = qig > gscp_data.zqmin

    # ----------------------------------------------------------------------------
    # 2.4: IF (llqr): ic1
    # ----------------------------------------------------------------------------
    # if llqr:
    zlnqrk = log(zqrk) if llqr else zlnqrk  # DL: TODO: Replace below?
    zsrmax = zzar / rhog * zdtr if llqr else 0.0

    zeln7o8qrk = exp(gscp_data.x7o8 * zlnqrk) if qig + qcg > gscp_data.zqmin else 0.0

    (zeln7o4qrk, zeln27o16qrk) = (
        (exp(gscp_data.x7o4 * zlnqrk), exp(gscp_data.x27o16 * zlnqrk))
        if tg < gscp_data.ztrfrz
        else (0.0, 0.0)
        if llqr
        else (0.0, 0.0)
    )
    zeln13o8qrk = exp(gscp_data.x13o8 * zlnqrk) if llqi else 0.0 if llqr else 0.0

    # ----------------------------------------------------------------------------
    # 2.5: IF (llqs): ic2
    # ----------------------------------------------------------------------------

    # if llqs:
    zlnqsk = (
        log(zqsk) if llqs else zlnqsk
    )  # GZ: shifting this computation ahead of the IF condition changes results! #DL: Replace below?
    zssmax = zzas / rhog * zdtr if llqs else 0.0
    zeln3o4qsk = (
        exp(gscp_data.x3o4 * zlnqsk)
        if qig + qcg > gscp_data.zqmin
        else 0.0
        if llqs
        else 0.0
    )
    zeln8qsk = exp(0.8 * zlnqsk) if llqr else 0.0

    # ----------------------------------------------------------------------------
    # 2.6: IF (llqg): ic3
    # ----------------------------------------------------------------------------

    # if zqgk > zqmin:
    zlnqgk = log(zqgk) if zqgk > gscp_data.zqmin else zlnqgk  # DL: Move below?
    zsgmax = zzag / rhog * zdtr if zqgk > gscp_data.zqmin else 0.0
    zelnrimexp_g = (
        exp(local_param.zrimexp_g * zlnqgk)
        if qig + qcg > gscp_data.zqmin > gscp_data.zqmin
        else 0.0
        if zqgk > gscp_data.zqmin
        else 0.0
    )
    zeln6qgk = exp(0.6 * zlnqgk) if zqgk > gscp_data.zqmin else 0.0

    # # ----------------------------------------------------------------------------
    # # 2.7:  slope of snow PSD and coefficients for depositional growth (llqi,llqs)
    # # ----------------------------------------------------------------------------

    # DL Todo: replace below
    zcsdep = 3.367e-2
    zcidep = 1.3e-5
    zcslam = 1e10

    # if qig > zqmin or zqsk > zqmin:  # DL: Actually the same llqi and llqs
    hlp_bool = True if qig > gscp_data.zqmin else False
    hlp_bool = True if zqsk > gscp_data.zqmin else hlp_bool

    zdvtp = gscp_coefficients.ccdvtp * exp(1.94 * log(tg)) / ppg if hlp_bool else 0.0
    zhi = (
        gscp_coefficients.ccshi1 * zdvtp * rhog * zqvsi / (tg * tg) if hlp_bool else 0.0
    )
    hlp = zdvtp / (1.0 + zhi) if hlp_bool else 0.0
    zcidep = gscp_coefficients.ccidep * hlp if hlp_bool else zcidep

    # if llqs: #DL: TODO Refactor below?
    hlp_bool2 = True if hlp_bool else False if llqs else False
    zcslam = (
        exp(gscp_coefficients.ccslxp * log(gscp_coefficients.ccslam * zn0s / zqsk))
        if hlp_bool2
        else zcslam
    )
    zcslam = minimum(zcslam, 1.0e15) if hlp_bool2 else zcslam
    zcsdep = 4.0 * zn0s * hlp if hlp_bool2 else zcsdep

    zcslam = (
        exp(gscp_coefficients.ccslxp * log(gscp_coefficients.ccslam * zn0s / zqsk))
        if hlp_bool
        else zcslam
    )
    zcslam = minimum(zcslam, 1.0e15) if hlp_bool else zcslam
    zcsdep = 4.0 * zn0s * hlp if hlp_bool else zcsdep

    # # ----------------------------------------------------------------------------
    # # 2.8: Deposition nucleation for low temperatures below a threshold (llqv)
    # # ----------------------------------------------------------------------------

    # if tg < zthet and qvg > 8.0e-6 and qig <= 0.0 and qvg > zqvsi:
    hlp_bool = True if tg < gscp_data.zthet else False
    hlp_bool = False if qvg <= 8.0e-6 else hlp_bool
    hlp_bool = False if qig > 0.0 else hlp_bool

    # DL: Refactor
    # znin = minimum(fxna_cooper(tg), znimax)
    # snuc = zmi0 * z1orhog * znin * zdtr
    znin = minimum(5.0 * exp(0.304 * (phy_const.tmelt - tg)), znimax)
    snuc = gscp_data.zmi0 * z1orhog * znin * zdtr if hlp_bool else 0.0

    # TODO
    # --------------------------------------------------------------------------
    # Section 3: Search for cloudy grid points with cloud water and
    #            calculation of the conversion rates involving qc (ic6)
    # --------------------------------------------------------------------------

    # TODO
    # ------------------------------------------------------------------------
    # Section 4: Search for cold grid points with cloud ice and/or snow and
    #            calculation of the conversion rates involving qi, qs and qg
    # ------------------------------------------------------------------------

    # TODO
    # --------------------------------------------------------------------------
    # Section 6: Search for grid points with rain in subsaturated areas
    #            and calculation of the evaporation rate of rain
    # --------------------------------------------------------------------------

    # TODO
    # --------------------------------------------------------------------------
    # Section 7: Calculate the total tendencies of the prognostic variables.
    #            Update the prognostic variables in the interior domain.
    # --------------------------------------------------------------------------

    zcorr = (
        zsrmax / maximum(zsrmax, sev + srfrz + srcri)
        if sev + srfrz + srcri > 0.0
        else 1.0
    )
    sev = sev * zcorr
    srfrz = srfrz * zcorr
    srcri = srcri * zcorr

    # TODO
    # # limit snow depletion in order to avoid negative values of qs
    # if ssdep <= 0.0:
    #     zcorr = (
    #         zssmax / max(zssmax, ssmelt + sconsg - ssdep) if ssmelt + sconsg - ssdep > 0.0 else 1.0
    #     )
    #     ssmelt = ssmelt * zcorr
    #     sconsg = sconsg *zcorr
    #     ssdep = ssdep * zcorr
    # else:
    #     zcorr = zssmax / max(zssmax, ssmelt + sconsg) if ssmelt + sconsg > 0.0 else 1.0
    #     ssmelt = ssmelt * zcorr
    #     sconsg = sconsg * zcorr

    zqvt = sev - sidep - ssdep - sgdep - snuc - sconr
    zqct = simelt - scau - scfrz - scac - sshed - srim - srim2
    zqit = snuc + scfrz - simelt - sicri + sidep - sdau - sagg - sagg2 - siau
    zqrt = scau + sshed + scac + ssmelt + sgmelt - sev - srcri - srfrz + sconr
    zqst = siau + sdau - ssmelt + srim + ssdep + sagg - sconsg
    zqgt = sagg2 - sgmelt + sicri + srcri + sgdep + srfrz + srim2 + sconsg

    # Update variables
    qig = maximum(0.0, (zzai * z1orhog + zqit * gscp_coefficients.zdt) * zimi)
    qrg = maximum(0.0, (zzar * z1orhog + zqrt * gscp_coefficients.zdt) * zimr)
    qsg = maximum(0.0, (zzas * z1orhog + zqst * gscp_coefficients.zdt) * zims)
    qgg = maximum(0.0, (zzag * z1orhog + zqgt * gscp_coefficients.zdt) * zimg)

    # ----------------------------------------------------------------------
    # Section 10: Complete time step
    # ----------------------------------------------------------------------

    # Store precipitation fluxes and sedimentation velocities for the next level
    zprvr = 0.0 if qrg * rhog * zvzr <= gscp_data.zqmin else qrg * rhog * zvzr
    zprvs = 0.0 if qsg * rhog * zvzs <= gscp_data.zqmin else qsg * rhog * zvzs
    zprvg = 0.0 if qgg * rhog * zvzg <= gscp_data.zqmin else qgg * rhog * zvzg
    zprvi = 0.0 if qig * rhog * zvzi <= gscp_data.zqmin else qig * rhog * zvzi

    # # # DL: This code block inflates errors from 1e-14 to 1e-10
    # # # DL: Also, it is rather expensive
    # # zvzr = (
    # #     0.0
    # #     if qrg + qr[0, 0, +1] <= zqmin
    # #     else zvz0r * exp(zvzxp * log((qrg + qr[0, 0, +1]) * 0.5 * rhog)) * zrho1o2
    # # )
    # # zvzs = (
    # #     0.0
    # #     if qsg + qs[0, 0, +1] <= zqmin
    # #     else zvz0s * exp(ccswxp * log((qsg + qs[0, 0, +1]) * 0.5 * rhog)) * zrho1o2
    # # )
    # # zvzg = (
    # #     0.0
    # #     if qgg + qg[0, 0, +1] <= zqmin
    # #     else zvz0g * exp(zexpsedg * log((qgg + qg[0, 0, +1]) * 0.5 * rhog)) * zrho1o2
    # # )
    # # zvzi = (
    # #     0.0
    # #     if qig + qi[0, 0, +1] <= zqmin
    # #     else zvz0i * exp(zbvi * log((qig + qi[0, 0, +1]) * 0.5 * rhog)) * zrhofac_qi
    # # )

    # # #----------------------------------------------------------------------
    # # # Section 11: Update Tendencies
    # # #----------------------------------------------------------------------

    # # z_heat_cap_r = cvdr if l_cv else cpdr
    # # lvariable_lh = ithermo_water != 0

    # # # Calculate Latent heats if necessary
    # # if lvariable_lh:
    # #     tg = make_normalized(t)
    # #     zlhv = latent_heat_vaporization(tg)
    # #     zlhs = latent_heat_sublimation(tg)
    # # else:
    # #     # Initialize latent heats to constant values.
    # #     zlhv = lh_v
    # #     zlhs = lh_s

    # # # DL: zlhv and zlhs are rather big numbers. Validates badly for a couple of gridpoints
    # # ztt = z_heat_cap_r * (zlhv * (zqct + zqrt) + zlhs * (zqit + zqst + zqgt))

    # # # save input arrays for final tendency calculation
    # # if lldiag_ttend:
    # #     t_in = t

    # # if lldiag_qtend:
    # #     qv_in = qv
    # #     qc_in = qc
    # #     qi_in = qi
    # #     qr_in = qr
    # #     qs_in = qs
    # #     # qg_in = qg

    # # # Update of prognostic variables or tendencies
    # # qr = max(0.0, qrg)
    # # qs = max(0.0, qsg)
    # # qi = max(0.0, qig)
    # # qg = max(0.0, qgg)
    # # t = t + ztt * zdt
    # # qv = max(0.0, qv + zqvt * zdt)
    # # qc = max(0.0, qc + zqct * zdt)

    # # #  ddt_tend_qg = max(-qg_in*zdtr,(qg - qg_in)*zdtr)

    # # if lldiag_ttend:
    # #     ddt_tend_t = (t - t_in) * zdtr

    # # if lldiag_qtend:
    # #     ddt_tend_qv = max(-qv_in * zdtr, (qv - qv_in) * zdtr)
    # #     ddt_tend_qc = max(-qc_in * zdtr, (qc - qc_in) * zdtr)
    # #     ddt_tend_qi = max(-qi_in * zdtr, (qi - qi_in) * zdtr)
    # #     ddt_tend_qr = max(-qr_in * zdtr, (qr - qr_in) * zdtr)
    # #     ddt_tend_qs = max(-qs_in * zdtr, (qs - qs_in) * zdtr)

    # # DL TODO: Temporary REMOVE ONCE IMPLEMENZED!!

    qv = 0.0
    qc = 0.0
    qi = 0.0
    qr = 0.0
    qs = 0.0
    qg = 0.0

    zpkr = 0.0
    zpks = 0.0
    zpkg = 0.0
    zpki = 0.0

    zprvr = 0.0
    zprvs = 0.0
    zprvg = 0.0
    zprvi = 0.0

    zvzr = 0.0
    zvzs = 0.0
    zvzg = 0.0
    zvzi = 0.0

    dist_cldtop = 0.0
    zqvsw_up = 0.0

    return (
        # qv,
        # qc,
        # qi,
        # qr,
        # qs,
        # qg,
        zpkr,
        zpks,
        zpkg,
        zpki,
        zprvr,
        zprvs,
        zprvg,
        zprvi,
        zvzr,
        zvzs,
        zvzi,
        zvzg,
        dist_cldtop,
        zqvsw_up,
    )


@program
# DL: For now definition mirrors FORTRAN interface
def graupel(
    zdt: float,  # time step
    dz: Field[[CellDim, KDim], float],  # level thickness
    # Prognostics
    temp: Field[[CellDim, KDim], float],
    pres: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    qi: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
    qs: Field[[CellDim, KDim], float],
    qg: Field[[CellDim, KDim], float],
    # Number Densities
    qnc: Field[[CellDim, KDim], float],
    qi0: float,
    qc0: float,
    # Precipitation Fluxes
    prr_gsp: Field[[CellDim, KDim], float],
    prs_gsp: Field[[CellDim, KDim], float],
    prg_gsp: Field[[CellDim, KDim], float],
    # Temporaries
    zpkr: Field[[CellDim, KDim], float],
    zpks: Field[[CellDim, KDim], float],
    zpkg: Field[[CellDim, KDim], float],
    zpki: Field[[CellDim, KDim], float],
    zprvr: Field[[CellDim, KDim], float],
    zprvs: Field[[CellDim, KDim], float],
    zprvg: Field[[CellDim, KDim], float],
    zprvi: Field[[CellDim, KDim], float],
    zvzr: Field[[CellDim, KDim], float],
    zvzs: Field[[CellDim, KDim], float],
    zvzg: Field[[CellDim, KDim], float],
    zvzi: Field[[CellDim, KDim], float],
    dist_cldtop: Field[[CellDim, KDim], float],
    zqvsw_up: Field[[CellDim, KDim], float],
    # Precomputed Parameters
    ccsrim: float,
    ccsagg: float,
    ccsdep: float,
    ccsvel: float,
    ccsvxp: float,
    ccslam: float,
    ccslxp: float,
    ccsaxp: float,
    ccsdxp: float,
    ccshi1: float,
    ccdvtp: float,
    ccidep: float,
    ccswxp: float,
    zconst: float,
    zcev: float,
    zbev: float,
    zcevxp: float,
    zbevxp: float,
    zvzxp: float,
    zvz0r: float,
    zvz0i: float,
    icesedi_exp: float,
    zceff_min: float,
    # Optional Fields: TODO: Pass optional fields to program
    ithermo_water: int32,
    pri_gsp: Field[[CellDim, KDim], float],
    ddt_tend_t: Field[[CellDim, KDim], float],
    ddt_tend_qv: Field[[CellDim, KDim], float],
    ddt_tend_qc: Field[[CellDim, KDim], float],
    ddt_tend_qi: Field[[CellDim, KDim], float],
    ddt_tend_qr: Field[[CellDim, KDim], float],
    ddt_tend_qs: Field[[CellDim, KDim], float],
    is_surface: Field[[CellDim, KDim], bool],
    # ddt_tend_qg: Field[[CellDim, KDim], float], # DL: Missing from FORTRAN interface
    # Option Switches
    l_cv: bool = False,  # if true, cv is used instead of cp
    lpres_pri: bool = False,  # if true, precipitation rate of ice shall be diagnosed
    lldiag_ttend: bool = False,  # if true, temperature tendency shall be diagnosed
    lldiag_qtend: bool = False,  # if true, moisture tendencies shall be diagnosed
):
    # Writing to several output fields currently breaks due to gt4py bugs
    _graupel(
        # zdt,
        dz,
        # Prognostics
        temp,
        pres,
        rho,
        qv,
        qi,
        qc,
        qr,
        qs,
        qg,
        # # Number Densities
        qnc,
        # qi0,
        # qc0,
        # # Precipitation Fluxes
        pri_gsp,
        prr_gsp,
        prs_gsp,
        prg_gsp,
        # Precomputed Parameters
        # ccsrim,
        # ccsagg,
        # ccsdep,
        # ccsvel,
        # ccsvxp,
        # ccslam,
        # ccslxp,
        # ccsaxp,
        # ccsdxp,
        # ccshi1,
        # ccdvtp,
        # ccidep,
        # ccswxp,
        # zconst,
        # zcev,
        # zbev,
        # zcevxp,
        # zbevxp,
        # zvzxp,
        # zvz0r,
        # zvz0i,
        # icesedi_exp,
        # zceff_min,
        # # Optional Fields: TODO: Pass optional fields to program
        # ithermo_water,  # TODO: Pass int to prgram
        ddt_tend_t,
        ddt_tend_qv,
        ddt_tend_qc,
        ddt_tend_qi,
        ddt_tend_qr,
        ddt_tend_qs,
        is_surface,
        # # ddt_tend_qg, # DL: Missing from FORTRAN interface
        # # Option Switches
        # l_cv,
        # lpres_pri,
        # lldiag_ttend,
        # lldiag_qtend,
        out=(
            # qv,
            # qc,
            # qi,
            # qr,
            # qs,
            # qg,
            zpkr,
            zpks,
            zpkg,
            zpki,
            zprvr,
            zprvs,
            zprvg,
            zprvi,
            zvzr,
            zvzs,
            zvzg,
            zvzi,
            dist_cldtop,
            zqvsw_up,
        ),
    )
