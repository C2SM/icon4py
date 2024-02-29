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
1. lsuper_coolw = False, lred_depgrow = False, lsedi_ice = False, lstickeff = Flase, lstickeff = False
2. isnow_n0temp == 1, and zn0s = const.
3. iautocon == 0 (Kessler)
4. l_cv = False. Now, we always use cv instead of cp

TODO: Currently unsupported features:
1. lldiag_ttend = True, lldiag_qtend = True (Need IF statment for return in GT4Py)

TODO: David
    1. Test if runnign with GPU backend. normalize
    2. Replace exp(A * log(B)) by B**A. Needs performance check and maybe optimization pass.
    3. Put scan in field operator. --> Disregard unneeded output
    4. Remove workaround for qnc (--> scheme does validate!!!), qc0, qi0 from gscp_data.py and pass explicitly
    5. Remove namespacing, i.e. z and c prefixes
    6. Replace 2D Fields by 1D fields qnc, prr_gsp et al.

TODO: GT4Py team
    1. Call field operators from scan --> sat_pres_ice
    2. Re-interate into FORTRAN
    3. IF statements in return
    4. Allow scan_operator init to initialize to None if no init is needed
"""
# DL. Set somewhere else
import sys
from typing import Final

from eve.utils import FrozenNamespace
from gt4py.next.ffront.decorator import program, scan_operator
from gt4py.next.ffront.fbuiltins import (
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
sys.setrecursionlimit(3000)


class GraupelParametersAndConfiguration(FrozenNamespace):
    """Configuration and local parameters of the graupel scheme."""

    # Configuration
    # lsedi_ice = True  # sedimentation of cloud ice (Heymsfield & Donner 1990 *1/3)
    # lstickeff = True  # switch for sticking coeff. (work from Guenther Zaengl)
    # lsuper_coolw = True  # switch for supercooled liquid water (work from Felix Rieper)
    # lred_depgrow = True  # separate switch for reduced depositional growth near tops of water clouds (now also used in ICON after correcting the cloud top diagnosis)

    lthermo_water_const = True

    # DL: TODO Pass explicitly
    lldiag_ttend = False
    lldiag_qtend = False
    ldass_lhn = True

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
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),  # DL TODO: Use ellipsis operator?
)
def _graupel(
    state_kMinus1: tuple[
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
    qrsflux: float,
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
    ddt_tend_t: float,
    ddt_tend_qv: float,
    ddt_tend_qc: float,
    ddt_tend_qi: float,
    ddt_tend_qr: float,
    ddt_tend_qs: float,
    ddt_tend_qg: float,  # DL: Missing from FORTRAN interface
    is_surface: bool,
    # Option Switches
    # lldiag_ttend: bool,  # if true, temperature tendency shall be diagnosed
    # lldiag_qtend: bool,  # if true, moisture tendencies shall be diagnosed
):

    # ------------------------------------------------------------------------------
    #  Section 1: Initial setting of local and global variables
    # ------------------------------------------------------------------------------

    # unpack carry
    (
        temp_kminus1,
        qv_kminus1,
        qc_kminus1,
        qi_kminus1,
        qr_kminus1,
        qs_kminus1,
        qg_kminus1,
        ddt_tend_t_kminus1,
        ddt_tend_qv_kminus1,
        ddt_tend_qc_kminus1,
        ddt_tend_qi_kminus1,
        ddt_tend_qr_kminus1,
        ddt_tend_qs_kminus1,
        ddt_tend_qg_kminus1,
        prr_gsp_kminus1,
        prs_gsp_kminus1,
        prg_gsp_kminus1,
        pri_gsp_kminus1,
        qrsflux_kminus1,
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
        zvzg_kminus1,
        zvzi_kminus1,
        dist_cldtop_kminus1,
        zqvsw_up_kminus1,
    ) = state_kMinus1

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
    zcrim = 0.0
    zcagg = 0.0
    zbsdep = 0.0
    zvz0s = 0.0

    zn0s = gscp_data.zn0s0  # TODO: Cleanup
    reduce_dep = 1.0  # FR: Reduction coeff. for dep. growth of rain and ice DL: Implemented below in Sec 3

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
    zpks = 0.0
    zpkg = 0.0
    zpki = 0.0

    # -------------------------------------------------------------------------
    # qs_prepare:
    # -------------------------------------------------------------------------

    if llqs:
        # DL: TODO Refactor: Move computation of zn0s into function
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

        zcrim = gscp_coefficients.ccsrim * zn0s
        zcagg = gscp_coefficients.ccsagg * zn0s
        zbsdep = gscp_coefficients.ccsdep * sqrt(gscp_coefficients.v0snow)
        zvz0s = gscp_coefficients.ccsvel * exp(gscp_coefficients.ccsvxp * log(zn0s))
    else:
        (zcrim, zcagg, zbsdep, zvz0s) = (0.0, 0.0, 0.0, 0.0)

    # sedimentation fluxes
    # -------------------------------------------------------------------------
    # qs_sedi:
    # -------------------------------------------------------------------------
    if llqs:
        zlnqsk = zvz0s * exp(gscp_coefficients.ccswxp * log(zqsk)) * zrho1o2

        # Prevent terminal fall speed of snow from being zero at the surface level
        zlnqsk = maximum(zlnqsk, gscp_data.v_sedi_snow_min) if is_surface else zlnqsk

        zpks = zqsk * zlnqsk
        zvzs = zlnqsk * ccswxp_ln1o2 if zvzs_kminus1 == 0.0 else zvzs_kminus1
    else:
        (zpks, zvzs) = (0.0, 0.0)

    # -------------------------------------------------------------------------
    # qr_sedi:
    # -------------------------------------------------------------------------
    if llqr:
        zlnqrk = (
            gscp_coefficients.zvz0r * exp(gscp_coefficients.zvzxp * log(zqrk)) * zrho1o2
        )

        # Prevent terminal fall speed of rain from being zero at the surface level
        zlnqrk = maximum(zlnqrk, gscp_data.v_sedi_rain_min) if is_surface else zlnqrk
        zpkr = zqrk * zlnqrk

        zvzr = zlnqrk * zvzxp_ln1o2 if zvzr_kminus1 == 0.0 else zvzr_kminus1
    else:
        (zpkr, zvzr) = (0.0, 0.0)

    # -------------------------------------------------------------------------
    # qg_sedi:
    # -------------------------------------------------------------------------
    if llqg:

        zlnqgk = local_param.zvz0g * exp(local_param.zexpsedg * log(zqgk)) * zrho1o2

        # Prevent terminal fall speed of graupel from being zero at the surface level
        zlnqgk = maximum(zlnqgk, gscp_data.v_sedi_graupel_min) if is_surface else zlnqgk
        zpkg = zqgk * zlnqgk if llqg else 0.0

        zvzg = zlnqgk * zexpsedg_ln1o2 if zvzg_kminus1 == 0.0 else zvzg_kminus1
    else:
        (zpkg, zvzg) = (0.0, 0.0)

    # -------------------------------------------------------------------------
    # qi_sedi:
    # -------------------------------------------------------------------------
    if llqi:
        zlnqik = gscp_coefficients.zvz0i * exp(gscp_data.zbvi * log(zqik)) * zrhofac_qi
        zpki = zqik * zlnqik

        zvzi = zlnqik * zbvi_ln1o2 if zvzi_kminus1 == 0.0 else zvzi_kminus1
    else:
        (zpki, zvzi) = (0.0, 0.0)

    # -------------------------------------------------------------------------
    # Prevent terminal fall speeds from being zero at the surface level
    # -------------------------------------------------------------------------

    if is_surface:
        zvzr = maximum(zvzr, gscp_data.v_sedi_rain_min)
        zvzs = maximum(zvzs, gscp_data.v_sedi_snow_min)
        zvzg = maximum(zvzg, gscp_data.v_sedi_graupel_min)

    # --------------------------------------------------------------------------
    # 2.3: Second part of preparations
    # --------------------------------------------------------------------------

    zeln7o8qrk = 0.0  # DL: Implemented below in IF (llqr): ic1
    zeln7o4qrk = 0.0  # Dl: Implemented below in IF (llqr): ic1
    zeln13o8qrk = 0.0  # DL: Implemented below in IF (llqr): ic1
    zeln3o4qsk = 0.0  # DL: Implemented below 2.5: IF (llqs): ic2
    zeln8qsk = 0.0  # DL: Implemented below 2.5: IF (llqs): ic2
    zeln6qgk = 0.0  # DL: implemnted below in IF (llqg): ic3
    zelnrimexp_g = 0.0  # DL: implemnted below in IF (llqg): ic3
    zsrmax = 0.0  # DL: Implemented below IF (llqr): ic1
    zssmax = 0.0  # DL: Implemented below 2.5: IF (llqs): ic2
    zsgmax = 0.0  # DL: implemnted below in IF (llqg): ic3

    scau = 0.0
    scac = 0.0
    snuc = 0.0  # DL: Implemented below: 2.8
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

    if llqr:
        zlnqrk = log(zqrk)  # DL: TODO: Implement below?

        # GZ: shifting this computation ahead of the IF condition changes results!
        zsrmax = zzar / rhog * zdtr

        zeln7o8qrk = (
            exp(gscp_data.x7o8 * zlnqrk) if qig + qcg > gscp_data.zqmin else 0.0
        )
        zeln7o4qrk = exp(gscp_data.x7o4 * zlnqrk) if tg < gscp_data.ztrfrz else 0.0

        zeln13o8qrk = exp(gscp_data.x13o8 * zlnqrk) if llqi else 0.0 if llqr else 0.0
    else:
        zeln7o8qrk = 0.0
        zeln7o4qrk = 0.0
        zeln13o8qrk = 0.0
        zsrmax = 0.0

    # ----------------------------------------------------------------------------
    # 2.5: IF (llqs): ic2
    # ----------------------------------------------------------------------------

    if llqs:
        zlnqsk = log(zqsk)  # DL: TODO: Implement below?
        # GZ: shifting this computation ahead of the IF condition changes results!
        zssmax = zzas / rhog * zdtr

        zeln3o4qsk = (
            exp(gscp_data.x3o4 * zlnqsk) if qig + qcg > gscp_data.zqmin else 0.0
        )
        zeln8qsk = exp(0.8 * zlnqsk)
    else:
        zeln3o4qsk, zeln8qsk, zssmax = (0.0, 0.0, 0.0)

    # ----------------------------------------------------------------------------
    # 2.6: IF (llqg): ic3
    # ----------------------------------------------------------------------------

    if zqgk > gscp_data.zqmin:
        zlnqgk = log(zqgk)
        zsgmax = zzag / rhog * zdtr
        zelnrimexp_g = (
            exp(local_param.zrimexp_g * zlnqgk) if qig + qcg > gscp_data.zqmin else 0.0
        )

        zeln6qgk = exp(0.6 * zlnqgk)
    else:
        zelnrimexp_g = 0.0
        zeln6qgk = 0.0
        zsgmax = 0.0

    # ----------------------------------------------------------------------------
    # 2.7:  slope of snow PSD and coefficients for depositional growth (llqi,llqs)
    # ----------------------------------------------------------------------------

    # DL: TODO Refactor this?
    if (qig > gscp_data.zqmin) | (
        zqsk > gscp_data.zqmin
    ):  # DL: TODO FIXFORTRAN same as llqi and llqs
        zdvtp = gscp_coefficients.ccdvtp * exp(1.94 * log(tg)) / ppg
        zhi = gscp_coefficients.ccshi1 * zdvtp * rhog * zqvsi / (tg * tg)
        hlp = zdvtp / (1.0 + zhi)
        zcidep = gscp_coefficients.ccidep * hlp

        if llqs:
            zcslam = exp(
                gscp_coefficients.ccslxp * log(gscp_coefficients.ccslam * zn0s / zqsk)
            )
            zcslam = minimum(zcslam, 1.0e15)
            zcsdep = 4.0 * zn0s * hlp

        else:
            zcsdep = 3.367e-2
            zcidep = 1.3e-5
            zcslam = 1e10
    else:
        zcsdep = 3.367e-2
        zcidep = 1.3e-5
        zcslam = 1e10

    # # ----------------------------------------------------------------------------
    # # 2.8: Deposition nucleation for low temperatures below a threshold (llqv)
    # # ----------------------------------------------------------------------------

    if (tg < gscp_data.zthet) & (qvg > 8.0e-6) & (qig <= 0.0) & (qvg > zqvsi):
        # znin = min(fxna_cooper(tg), znimax) #DL: TODO
        znin = minimum(5.0 * exp(0.304 * (phy_const.tmelt - tg)), znimax)
        snuc = gscp_data.zmi0 * z1orhog * znin * zdtr
    else:
        snuc = 0.0

    # --------------------------------------------------------------------------
    # Section 3: Search for cloudy grid points with cloud water and
    #            calculation of the conversion rates involving qc (ic6)
    # --------------------------------------------------------------------------

    if qcg > gscp_data.zqmin:
        llqs = (
            zqsk > gscp_data.zqmin
        )  # DL: TODO FIXFORTRAN: Why is this recalculated here?

        zscmax = qcg * zdtr
        if tg > gscp_data.zthn:

            # TODO: Candidate for a function
            # Seifert and Beheng (2001) autoconversion rate
            # with constant cloud droplet number concentration qnc
            if qcg > 1.0e-6:
                ztau = minimum(1.0 - qcg / (qcg + qrg), 0.9)
                ztau = maximum(ztau, 1.0e-30)
                hlp = exp(gscp_data.zkphi2 * log(ztau))
                zphi = gscp_data.zkphi1 * hlp * (1.0 - hlp) ** 3
                scau = (
                    gscp_coefficients.zconst
                    * qcg
                    * qcg
                    * qcg
                    * qcg
                    / (qnc * qnc)
                    * (1.0 + zphi / (1.0 - ztau) ** 2)
                )
                zphi = (ztau / (ztau + gscp_data.zkphi3)) ** 4
                scac = gscp_data.zkcac * qcg * qrg * zphi
            else:
                scau = 0.0
                scac = 0.0

            if llqr & (tg < gscp_data.ztrfrz) & (qrg > 0.1 * qcg):
                srfrz = (
                    gscp_data.zcrfrz1
                    * (exp(gscp_data.zcrfrz2 * (gscp_data.ztrfrz - tg)) - 1.0)
                    * zeln7o4qrk
                )

            srim = (
                zcrim * qcg * exp(gscp_coefficients.ccsaxp * log(zcslam))
                if llqs
                else 0.0
            )

            srim2 = local_param.zcrim_g * qcg * zelnrimexp_g
            if tg >= phy_const.tmelt:
                sshed = srim + srim2
                srim = 0.0
                srim2 = 0.0
            else:
                if qcg >= gscp_coefficients.qc0:
                    sconsg = local_param.zcsg * qcg * zeln3o4qsk

            # Check for maximum depletion of cloud water and adjust the
            # transfer rates accordingly
            zcorr = zscmax / maximum(zscmax, scau + scac + srim + srim2 + sshed)
            scau = scau * zcorr
            scac = scac * zcorr
            srim = srim * zcorr
            srim2 = srim2 * zcorr
            sshed = sshed * zcorr
            sconsg = minimum(sconsg, srim + zssmax)

        else:  # hom. freezing of cloud and rain water
            scfrz = zscmax
            srfrz = zsrmax

        # Calculation of heterogeneous nucleation of cloud ice.
        # This is done in this section, because we require water saturation
        # for this process (i.e. the existence of cloud water) to exist.
        # Heterogeneous nucleation is assumed to occur only when no
        # cloud ice is present and the temperature is below a nucleation
        # threshold.

        if (tg <= 267.15) & (not llqi):
            # znin = min(fxna_cooper(tg), znimax)
            znin = minimum(5.0 * exp(0.304 * (phy_const.tmelt - tg)), znimax)
            snuc = gscp_data.zmi0 * z1orhog * znin * zdtr

        # # Calculation of reduction of depositional growth at cloud top (Forbes 2012)
        if not is_surface:
            # znin = minimum(fxna_cooper(tg), znimax)
            znin = minimum(5.0 * exp(0.304 * (phy_const.tmelt - tg)), znimax)
            fnuc = minimum(znin / znimix, 1.0)

            qcgk_1 = qi_kminus1 + qs_kminus1 + qg_kminus1

            # distance from cloud top
            if (qv_kminus1 + qc_kminus1 < zqvsw_up_kminus1) & (
                qcgk_1 < gscp_data.zqmin
            ):
                dist_cldtop = 0.0  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop_kminus1 + dz

            reduce_dep = minimum(
                fnuc
                + (1.0 - fnuc)
                * (
                    gscp_data.reduce_dep_ref
                    + dist_cldtop_kminus1 / gscp_data.dist_cldtop_ref
                ),
                1.0,
            )
        else:
            reduce_dep = 1.0
            dist_cldtop = 0.0
            zqvsw_up = 0.0
    else:
        reduce_dep = 1.0
        dist_cldtop = 0.0
        zqvsw_up = 0.0

    # ------------------------------------------------------------------------
    # Section 4: Search for cold grid points with cloud ice and/or snow and
    #            calculation of the conversion rates involving qi, qs and qg
    # ------------------------------------------------------------------------
    if (qig > gscp_data.zqmin) | (zqsk > gscp_data.zqmin) | (zqgk > gscp_data.zqmin):
        llqs = zqsk > gscp_data.zqmin  # DL: FIXFORTRAN remove?
        llqi = qig > gscp_data.zqmin  # DL: FIXFORTRAN remove?

        if tg <= phy_const.tmelt:  # cold case

            zqvsidiff = qvg - zqvsi
            zsvmax = zqvsidiff * zdtr
            zsvidep = 0.0
            zsvisub = 0.0
            if llqi:

                # znin = minimum(fxna_cooper(tg), znimax) # DL: TODO
                znin = minimum(5.0 * exp(0.304 * (phy_const.tmelt - tg)), znimax)

                # Change in sticking efficiency needed in case of cloud ice sedimentation
                zeff = minimum(exp(0.09 * (tg - phy_const.tmelt)), 1.0)
                zeff = maximum(
                    maximum(zeff, gscp_coefficients.zceff_min),
                    gscp_data.zceff_fac * (tg - gscp_data.tmin_iceautoconv),
                )

                sagg = zeff * qig * zcagg * exp(gscp_coefficients.ccsaxp * log(zcslam))
                sagg2 = zeff * qig * local_param.zcagg_g * zelnrimexp_g
                siau = (
                    zeff * gscp_data.zciau * maximum(qig - gscp_coefficients.qi0, 0.0)
                )
                zmi = maximum(
                    gscp_data.zmi0, minimum(rhog * qig / znin, gscp_data.zmimax)
                )
                znid = rhog * qig / zmi
                zlnlogmi = log(zmi)  # DL: TODO
                sidep = zcidep * znid * exp(0.33 * zlnlogmi) * zqvsidiff

                # for sedimenting quantities the maximum
                # allowed depletion is determined by the predictor value.
                zsimax = zzai * z1orhog * zdtr

                if sidep > 0.0:
                    zsvidep = minimum(sidep * reduce_dep, zsvmax)
                if sidep < 0.0:
                    zsvisub = maximum(sidep, zsvmax)
                    zsvisub = -maximum(zsvisub, -zsimax)

                zlnlogmi = log(gscp_data.zmsmin / zmi)
                zztau = 1.5 * (exp(0.66 * zlnlogmi) - 1.0)
                sdau = zsvidep / zztau
                sicri = gscp_data.zcicri * qig * zeln7o8qrk
                srcri = (
                    gscp_data.zcrcri * (qig / zmi) * zeln13o8qrk
                    if qsg > 1.0e-7
                    else srcri
                )
            else:
                zsimax = 0.0

            zxfac = 1.0 + zbsdep * exp(gscp_coefficients.ccsdxp * log(zcslam))
            ssdep = zcsdep * zxfac * zqvsidiff / (zcslam + gscp_data.zeps) ** 2
            # FR new: depositional growth reduction
            if ssdep > 0.0:
                ssdep = ssdep * reduce_dep

            # GZ: This limitation, which was missing in the original graupel scheme,
            # is crucial for numerical stability in the tropics
            ssdep = minimum(ssdep, zsvmax - zsvidep) if ssdep > 0.0 else ssdep

            # Suppress depositional growth of snow if the existing amount is too small for a
            # a meaningful distiction between cloud ice and snow
            ssdep = minimum(ssdep, 0.0) if qsg <= 1.0e-7 else ssdep

            # ** GZ: this numerical fit should be replaced with a physically more meaningful formulation **
            sgdep = (
                (0.398561 - 0.00152398 * tg + 2554.99 / ppg + 2.6531e-7 * ppg)
                * zqvsidiff
                * zeln6qgk
            )
            # Check for maximal depletion of cloud ice
            # No check is done for depositional autoconversion because
            # this is a always a fraction of the gain rate due to
            # deposition (i.e the sum of this rates is always positive)
            zsisum = siau + sagg + sagg2 + sicri + zsvisub
            zcorr = zsimax / maximum(zsimax, zsisum) if zsimax > 0.0 else 0.0
            sidep = zsvidep - zcorr * zsvisub
            siau = siau * zcorr
            sagg = sagg * zcorr
            sagg2 = sagg2 * zcorr
            sicri = sicri * zcorr
            if zqvsidiff < 0.0:
                ssdep = maximum(ssdep, -zssmax)
                sgdep = maximum(sgdep, -zsgmax)

        else:  # tg > 0 - warm case

            # cloud ice melts instantaneously
            simelt = zzai * z1orhog * zdtr

            zqvsw0 = zpvsw0 / (rhog * phy_const.rv * phy_const.tmelt)
            zqvsw0diff = qvg - zqvsw0
            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if tg > (phy_const.tmelt - local_param.ztcrit * zqvsw0diff):
                # calculate melting rate
                zx1 = (tg - phy_const.tmelt) + local_param.zasmel * zqvsw0diff
                ssmelt = (79.6863 / ppg + 0.612654e-3) * zx1 * zeln8qsk
                ssmelt = minimum(ssmelt, zssmax)
                sgmelt = (12.31698 / ppg + 7.39441e-05) * zx1 * zeln6qgk
                sgmelt = minimum(sgmelt, zsgmax)
                # deposition + melting, ice particle temperature: t0
                # calculation without howell-factor#
                ssdep = (31282.3 / ppg + 0.241897) * zqvsw0diff * zeln8qsk
                sgdep = (0.153907 - ppg * 7.86703e-07) * zqvsw0diff * zeln6qgk
                if zqvsw0diff < 0.0:
                    # melting + evaporation of snow/graupel
                    ssdep = maximum(-zssmax, ssdep)
                    sgdep = maximum(-zsgmax, sgdep)
                    # melt water evaporates
                    ssmelt = ssmelt + ssdep
                    sgmelt = sgmelt + sgdep
                    ssmelt = maximum(ssmelt, 0.0)
                    sgmelt = maximum(sgmelt, 0.0)
                else:
                    # deposition on snow/graupel is interpreted as increase
                    # in rain water ( qv --> qr, sconr)
                    # therefore,  sconr=(zssdep+zsgdep)
                    sconr = ssdep + sgdep
                    ssdep = 0.0
                    sgdep = 0.0
            else:  # if t<t_crit
                # no melting, only evaporation of snow/graupel
                # zqvsw = sat_pres_water(tg) / (rhog * r_v * tg) #DL: TODO
                sat_pres_water = conv_table.c1es * exp(
                    conv_table.c3les
                    * (temp - phy_const.tmelt)
                    / (temp - conv_table.c4les)
                )
                zqvsw = sat_pres_water / (rhog * phy_const.rv * tg)

                zqvsw_up = zqvsw  # DL: TODO refactor?
                zqvsidiff = qvg - zqvsw
                ssdep = (0.28003 - ppg * 0.146293e-6) * zqvsidiff * zeln8qsk
                sgdep = (0.0418521 - ppg * 4.7524e-8) * zqvsidiff * zeln6qgk
                ssdep = maximum(-zssmax, ssdep)
                sgdep = maximum(-zsgmax, sgdep)

    # --------------------------------------------------------------------------
    # Section 6: Search for grid points with rain in subsaturated areas
    #            and calculation of the evaporation rate of rain
    # --------------------------------------------------------------------------

    # zqvsw = sat_pres_water(tg) / (rhog * phy_const.rv * tg) #DL: TODO
    sat_pres_water = conv_table.c1es * exp(
        conv_table.c3les * (temp - phy_const.tmelt) / (temp - conv_table.c4les)
    )
    zqvsw = sat_pres_water / (rhog * phy_const.rv * tg)

    zqvsw_up = zqvsw  # DL TODO: refactor?

    if llqr & (qvg + qcg <= zqvsw):

        zlnqrk = log(zqrk)
        zx1 = 1.0 + gscp_coefficients.zbev * exp(gscp_coefficients.zbevxp * zlnqrk)
        # sev  = zcev*zx1*(zqvsw - qvg) * exp(zcevxp  * zlnqrk)
        # Limit evaporation rate in order to avoid overshoots towards supersaturation
        # the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        temp_c = tg - phy_const.tmelt
        maxevap = (
            (0.61 - 0.0163 * temp_c + 1.111e-4 * temp_c**2)
            * (zqvsw - qvg)
            / gscp_coefficients.zdt
        )
        sev = minimum(
            gscp_coefficients.zcev
            * zx1
            * (zqvsw - qvg)
            * exp(gscp_coefficients.zcevxp * zlnqrk),
            maxevap,
        )

        # Calculation of below-cloud rainwater freezing
        if (tg > gscp_data.zthn) & (tg < gscp_data.ztrfrz):
            srfrz = (
                gscp_data.zcrfrz1
                * (exp(gscp_data.zcrfrz2 * (gscp_data.ztrfrz - tg)) - 1.0)
                * zeln7o4qrk
            )

        else:  # Hom. freezing of rain water
            srfrz = zsrmax

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

    # limit snow depletion in order to avoid negative values of qs
    if ssdep <= 0.0:
        zcorr = (
            zssmax / maximum(zssmax, ssmelt + sconsg - ssdep)
            if ssmelt + sconsg - ssdep > 0.0
            else 1.0
        )
        ssmelt = ssmelt * zcorr
        sconsg = sconsg * zcorr
        ssdep = ssdep * zcorr
    else:
        zcorr = (
            zssmax / maximum(zssmax, ssmelt + sconsg) if ssmelt + sconsg > 0.0 else 1.0
        )
        ssmelt = ssmelt * zcorr
        sconsg = sconsg * zcorr

    zqvt = sev - sidep - ssdep - sgdep - snuc - sconr
    zqct = simelt - scau - scfrz - scac - sshed - srim - srim2
    zqit = snuc + scfrz - simelt - sicri + sidep - sdau - sagg - sagg2 - siau
    zqrt = scau + sshed + scac + ssmelt + sgmelt - sev - srcri - srfrz + sconr
    zqst = siau + sdau - ssmelt + srim + ssdep + sagg - sconsg
    zqgt = sagg2 - sgmelt + sicri + srcri + sgdep + srfrz + srim2 + sconsg

    # Save input arrays for final tendency calculation DL: TODO refactor
    t_in = temp
    qv_in = qv
    qc_in = qc
    qi_in = qi
    qr_in = qr
    qs_in = qs
    qg_in = qg

    # Compute tendencies
    qi_t = (zzai * z1orhog + zqit * gscp_coefficients.zdt) * zimi
    qr_t = (zzar * z1orhog + zqrt * gscp_coefficients.zdt) * zimr
    qs_t = (zzas * z1orhog + zqst * gscp_coefficients.zdt) * zims
    qg_t = (zzag * z1orhog + zqgt * gscp_coefficients.zdt) * zimg

    # Update Variables DL: Refactor?
    qig = maximum(0.0, qi_t)
    qrg = maximum(0.0, qr_t)
    qsg = maximum(0.0, qs_t)
    qgg = maximum(0.0, qg_t)

    # ----------------------------------------------------------------------
    # Section 10: Complete time step
    # ----------------------------------------------------------------------

    if not is_surface:  #  DL: if not surface needed with scan?

        # Store precipitation fluxes and sedimentation velocities for the next level
        zprvr = 0.0 if qrg * rhog * zvzr <= gscp_data.zqmin else qrg * rhog * zvzr
        zprvs = 0.0 if qsg * rhog * zvzs <= gscp_data.zqmin else qsg * rhog * zvzs
        zprvg = 0.0 if qgg * rhog * zvzg <= gscp_data.zqmin else qgg * rhog * zvzg
        zprvi = 0.0 if qig * rhog * zvzi <= gscp_data.zqmin else qig * rhog * zvzi

        # for the latent heat nudging
        if local_param.ldass_lhn:
            qrsflux = zprvr + zprvs + zprvg + zprvi
            qrsflux = 0.5 * (qrsflux + zpkr + zpks + zpkg + zpki)
        else:
            qrsflux = 0.0

        # DL: This code block inflates errors from 1e-14 to 1e-10
        zvzr = (
            0.0
            if qrg + qr_kminus1 <= gscp_data.zqmin
            else gscp_coefficients.zvz0r
            * exp(gscp_coefficients.zvzxp * log((qrg + qr_kminus1) * 0.5 * rhog))
            * zrho1o2
        )
        zvzs = (
            0.0
            if qsg + qs_kminus1 <= gscp_data.zqmin
            else zvz0s
            * exp(gscp_coefficients.ccswxp * log((qsg + qs_kminus1) * 0.5 * rhog))
            * zrho1o2
        )
        zvzg = (
            0.0
            if qgg + qg_kminus1 <= gscp_data.zqmin
            else local_param.zvz0g
            * exp(local_param.zexpsedg * log((qgg + qg_kminus1) * 0.5 * rhog))
            * zrho1o2
        )
        zvzi = (
            0.0
            if qig + qi_kminus1 <= gscp_data.zqmin
            else gscp_coefficients.zvz0i
            * exp(gscp_data.zbvi * log((qig + qi_kminus1) * 0.5 * rhog))
            * zrhofac_qi
        )

    else:

        # DL: TODO, Reset really needed?
        # Delete precipitation fluxes from previous timestep
        prr_gsp = 0.0
        prs_gsp = 0.0
        prg_gsp = 0.0
        pri_gsp = 0.0

        prr_gsp = 0.5 * (qrg * rhog * zvzr + zpkr)
        prs_gsp = 0.5 * (rhog * qsg * zvzs + zpks)
        pri_gsp = 0.5 * (rhog * qig * zvzi + zpki)
        prg_gsp = 0.5 * (qgg * rhog * zvzg + zpkg)

        #  for the latent heat nudging
        qrsflux = prr_gsp + prs_gsp + prg_gsp if local_param.ldass_lhn else 0.0

        (zprvr, zprvs, zprvg, zprvi) = (0.0, 0.0, 0.0, 0.0)

    # ----------------------------------------------------------------------
    # Section 11: Update Tendencies
    # ----------------------------------------------------------------------

    # Calculate Latent heats if necessary
    if local_param.lthermo_water_const:
        # Initialize latent heats to constant values.
        zlhv = phy_const.alv
        zlhs = phy_const.als
    else:
        # tg = make_normalized(t)
        # zlhv = latent_heat_vaporization(tg)
        # zlhs = latent_heat_sublimation(tg)
        zlhv = (
            phy_const.alv
            + (1850.0 - phy_const.clw) * (tg - phy_const.tmelt)
            - phy_const.rv * tg
        )
        zlhs = (
            phy_const.als
            + (1850.0 - 2108.0) * (tg - phy_const.tmelt)
            - phy_const.rdv * tg
        )

    # DL: zlhv and zlhs are rather big numbers. Validates badly for a couple of gridpoints
    ztt = phy_const.rcvd * (zlhv * (zqct + zqrt) + zlhs * (zqit + zqst + zqgt))

    # Diagnose pseudo-tendencies

    # Update of prognostic variables
    qr = maximum(0.0, qrg)
    qs = maximum(0.0, qsg)
    qi = maximum(0.0, qig)
    qg = maximum(0.0, qgg)
    temp = temp + ztt * gscp_coefficients.zdt
    qv = maximum(0.0, qv + zqvt * gscp_coefficients.zdt)
    qc = maximum(0.0, qc + zqct * gscp_coefficients.zdt)

    if local_param.lldiag_ttend:
        ddt_tend_t = temp - t_in * zdtr

    if local_param.lldiag_ttend:
        ddt_tend_qv = maximum(-qv_in * zdtr, (qv - qv_in) * zdtr)
        ddt_tend_qc = maximum(-qc_in * zdtr, (qc - qc_in) * zdtr)
        ddt_tend_qi = maximum(-qi_in * zdtr, (qi - qi_in) * zdtr)
        ddt_tend_qr = maximum(-qr_in * zdtr, (qr - qr_in) * zdtr)
        ddt_tend_qs = maximum(-qs_in * zdtr, (qs - qs_in) * zdtr)
        ddt_tend_qg = maximum(-qg_in * zdtr, (qg - qg_in) * zdtr)

    temp = 10000000.0
    qv = 1.0
    qc = 1.0
    qi = 1.0
    qr = 1.0
    qs = 1.0
    qg = 1.0
    ddt_tend_t = 1.0
    ddt_tend_qv = 1.0
    ddt_tend_qc = 1.0
    ddt_tend_qi = 1.0
    ddt_tend_qr = 1.0
    ddt_tend_qs = 1.0
    ddt_tend_qg = 1.0
    prr_gsp = 1.0
    prs_gsp = 1.0
    prg_gsp = 1.0
    pri_gsp = 1.0
    qrsflux = 1.0
    zpkr = 1.0
    zpks = 1.0
    zpkg = 1.0
    zpki = 1.0
    zprvr = 1.0
    zprvs = 1.0
    zprvg = 1.0
    zprvi = 1.0
    zvzr = 1.0
    zvzs = 1.0
    zvzi = 1.0
    zvzg = 1.0
    dist_cldtop = 1.0
    zqvsw_up = 1.0

    return (
        temp,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        ddt_tend_t,
        ddt_tend_qv,
        ddt_tend_qc,
        ddt_tend_qi,
        ddt_tend_qr,
        ddt_tend_qs,
        ddt_tend_qg,
        prr_gsp,
        prs_gsp,
        prg_gsp,
        pri_gsp,
        qrsflux,
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
    pri_gsp: Field[[CellDim, KDim], float],
    prr_gsp: Field[[CellDim, KDim], float],
    prs_gsp: Field[[CellDim, KDim], float],
    prg_gsp: Field[[CellDim, KDim], float],
    qrsflux: Field[[CellDim, KDim], float],
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
    ddt_tend_t: Field[[CellDim, KDim], float],
    ddt_tend_qv: Field[[CellDim, KDim], float],
    ddt_tend_qc: Field[[CellDim, KDim], float],
    ddt_tend_qi: Field[[CellDim, KDim], float],
    ddt_tend_qr: Field[[CellDim, KDim], float],
    ddt_tend_qs: Field[[CellDim, KDim], float],
    ddt_tend_qg: Field[[CellDim, KDim], float],
    is_surface: Field[[CellDim, KDim], bool],
    # Option Switches
    lldiag_ttend: bool,  # if true, temperature tendency shall be diagnosed
    lldiag_qtend: bool,  # if true, moisture tendencies shall be diagnosed
    num_cells: int32,
    num_levels: int32,
    kstart_moist: int32,
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
        # Number Densities
        qnc,
        # qi0,
        # qc0,
        # Precipitation Fluxes
        pri_gsp,
        prr_gsp,
        prs_gsp,
        prg_gsp,
        qrsflux,
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
        ddt_tend_t,
        ddt_tend_qv,
        ddt_tend_qc,
        ddt_tend_qi,
        ddt_tend_qr,
        ddt_tend_qs,
        ddt_tend_qg,
        is_surface,
        # # Option Switches
        # lldiag_ttend,
        # lldiag_qtend,
        out=(
            temp,
            qv,
            qc,
            qi,
            qr,
            qs,
            qg,
            ddt_tend_t,
            ddt_tend_qv,
            ddt_tend_qc,
            ddt_tend_qi,
            ddt_tend_qr,
            ddt_tend_qs,
            ddt_tend_qg,
            prr_gsp,
            prs_gsp,
            prg_gsp,
            pri_gsp,
            qrsflux,
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
        ),
        domain={CellDim: (0, num_cells), KDim: (kstart_moist, num_levels)},
    )
