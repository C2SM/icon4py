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

TODO: David
    1. Test if runnign with GPU backend

TODO: GT4Py team
1. Pass optional fields to program and scan

2. Defining kwargs in program and scan:
@program
def graupel(field1: Field[[CellDim, KDim], float], field2: Field[[CellDim, KDim], float] = None ):
    if field2 is not None:
        do this
3. Re-interate into FORTRAN
"""
from typing import Final

from eve.utils import FrozenNamespace
from functional.ffront.decorator import program, scan_operator
from functional.ffront.fbuiltins import Field, abs, exp, int32

from icon4py.atm_phy_schemes.gscp_data import gscp_data
from icon4py.atm_phy_schemes.mo_convect_tables import conv_table
from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import phy_const


class GraupelParametersAndConfiguration(FrozenNamespace):
    """Configuration and local parameters of the graupel scheme."""

    # Configuration
    lsedi_ice = (
        True  # switch for sedimentation of cloud ice (Heymsfield & Donner 1990 *1/3)
    )
    lstickeff = True  # switch for sticking coeff. (work from Guenther Zaengl)
    lsuper_coolw = True  # switch for supercooled liquid water (work from Felix Rieper)
    lred_depgrow = True  # separate switch for reduced depositional growth near tops of water clouds (now also used in ICON after correcting the cloud top diagnosis)

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


def fxna(ztx):
    """Return number of activate ice crystals from temperature."""
    return 1.0e2 * exp(0.2 * (conv_table.tmelt - ztx))


def fxna_cooper(ztx):
    """Return number of activate ice crystals from temperature.

    Method: Cooper (1986) used by Greg Thompson(2008)
    """
    return 5.0 * exp(0.304 * (conv_table.tmelt - ztx))


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


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
def _graupel(
    carry: tuple[float, float],
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
    #   qnc: Field[[CellDim], float], #2D Field
    #   qi0: float,
    #   qc0: float,
    # Precipitation Fluxes
    #   prr_gsp: Field[[CellDim], float], #2D Field
    #   prs_gsp: Field[[CellDim], float], #2D Field
    #   prg_gsp: Field[[CellDim], float], #2D Field
    # Temporaries
    zpkr: float,
    zpks: float,
    zpkg: float,
    zpki: float,
    zprvr: float,
    zprvs: float,
    zprvg: float,
    zprvi: float,
    zvzr: float,
    zvzs: float,
    zvzg: float,
    zvzi: float,
    dist_cldtop: float,
    zqvsw_up: float,
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
    # v0snow: float,
    # zvz0i: float,
    # icesedi_exp: float,
    # zams: float,
    # zceff_min: float,
    # Optional Fields: TODO: Pass optional fields to program
    # ithermo_water: int32,  # TODO: Pass int to prgram
    # pri_gsp: Field[[CellDim], float],
    ddt_tend_t: float,
    ddt_tend_qv: float,
    ddt_tend_qc: float,
    ddt_tend_qi: float,
    ddt_tend_qr: float,
    ddt_tend_qs: float,
    # ddt_tend_qg: float, # DL: Missing from FORTRAN interface
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
    qc_kMinus1, qr_kMinus1 = carry

    qc = qc + gscp_data.mma[0]
    return qc, qr


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
    qnc: Field[[CellDim], float],
    qi0: float,
    qc0: float,
    # Precipitation Fluxes
    prr_gsp: Field[[CellDim], float],
    prs_gsp: Field[[CellDim], float],
    prg_gsp: Field[[CellDim], float],
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
    v0snow: float,
    zvz0i: float,
    icesedi_exp: float,
    zams: float,
    zceff_min: float,
    # Optional Fields: TODO: Pass optional fields to program
    ithermo_water: int32,
    pri_gsp: Field[[CellDim], float],
    ddt_tend_t: Field[[CellDim, KDim], float],
    ddt_tend_qv: Field[[CellDim, KDim], float],
    ddt_tend_qc: Field[[CellDim, KDim], float],
    ddt_tend_qi: Field[[CellDim, KDim], float],
    ddt_tend_qr: Field[[CellDim, KDim], float],
    ddt_tend_qs: Field[[CellDim, KDim], float],
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
        # qnc,
        # qi0,
        # qc0,
        # # Precipitation Fluxes
        # prr_gsp,
        # prs_gsp,
        # prg_gsp,
        # Temporaries
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
        # v0snow,
        # zvz0i,
        # icesedi_exp,
        # zams,
        # zceff_min,
        # # Optional Fields: TODO: Pass optional fields to program
        # ithermo_water,  # TODO: Pass int to prgram
        # pri_gsp,
        ddt_tend_t,
        ddt_tend_qv,
        ddt_tend_qc,
        ddt_tend_qi,
        ddt_tend_qr,
        ddt_tend_qs,
        # # ddt_tend_qg, # DL: Missing from FORTRAN interface
        # # Option Switches
        # l_cv,
        # lpres_pri,
        # lldiag_ttend,
        # lldiag_qtend,
        out=(qr, qc),
    )
