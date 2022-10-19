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

TODO: GT4Py team
1. Passing int to program
2. Passing optional fields to program
"""

from functional.ffront.decorator import program, scan_operator
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
def _graupel(
    carry: tuple[float, float],
    qc_in: float,
    qr_in: float,
):

    # unpack carry
    qc_kMinus1, qr_kMinus1 = carry

    return qc_in, qr_in


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
    cloud_num: float,
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
    x13o8: float,
    x1o2: float,
    x27o16: float,
    x3o4: float,
    x7o4: float,
    x7o8: float,
    zbvi: float,
    zcac: float,
    zccau: float,
    zciau: float,
    zcicri: float,
    zcrcri: float,
    zcrfrz: float,
    zcrfrz1: float,
    zcrfrz2: float,
    zeps: float,
    zkcac: float,
    zkphi1: float,
    zkphi2: float,
    zkphi3: float,
    zmi0: float,
    zmimax: float,
    zmsmin: float,
    zn0s0: float,
    zn0s1: float,
    zn0s2: float,
    znimax_Thom: float,
    zqmin: float,
    zrho0: float,
    zthet: float,
    zthn: float,
    ztmix: float,
    ztrfrz: float,
    zvz0i: float,
    icesedi_exp: float,
    zams: float,
    iautocon: int,
    isnow_n0temp: int,
    dist_cldtop_ref: float,
    reduce_dep_ref: float,
    tmin_iceautoconv: float,
    zceff_fac: float,
    zceff_min: float,
    v_sedi_rain_min: float,
    v_sedi_snow_min: float,
    v_sedi_graupel_min: float,
    mma: tuple[float, float, float, float, float, float, float, float, float, float],
    mmb: tuple[float, float, float, float, float, float, float, float, float, float],
    # Optional Fields: TODO: Pass optional fields to program
    # ithermo_water: int,  # TODO: Pass int to prgram
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
    _graupel(qc, qr, out=(qr, qc))
