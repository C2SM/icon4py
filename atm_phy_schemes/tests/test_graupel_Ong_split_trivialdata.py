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
"""Test graupel in standalone mode using data serialized from ICON.

GT4Py hotfix:

In _external_src/gt4py-functional/src/functional/iterator/transforms/pass_manager.py
1. Exchange L49 with: inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
2. Add "return inlined" below
"""

import os

import numpy as np
import sys
from gt4py.next.ffront.fbuiltins import (
    Field,
    float64,
    int32
)

from icon4py.atm_phy_schemes.gscp_graupel_Ong_allinOne_split import _graupel#, _graupel_t_tendency, _graupel_q_tendency, _graupel_flux
from typing import Final
from icon4py.common.dimension import CellDim, KDim
from gt4py.next.iterator.embedded import index_field, np_as_located_field

#import serialbox as ser


def test_graupel_Ong_serialized_data():

    cell_size = 8
    k_size = 8

    
    ser_field_name = (
        "ser_graupel_dz",
        "ser_graupel_temperature",
        "ser_graupel_pres",
        "ser_graupel_rho",
        "ser_graupel_qv",
        "ser_graupel_qc",
        "ser_graupel_qi",
        "ser_graupel_qr",
        "ser_graupel_qs",
        "ser_graupel_qg",
        "ser_graupel_qnc",
        "ser_graupel_ddt_tend_t",
        "ser_graupel_ddt_tend_qv",
        "ser_graupel_ddt_tend_qc",
        "ser_graupel_ddt_tend_qi",
        "ser_graupel_ddt_tend_qr",
        "ser_graupel_ddt_tend_qs",
        "ser_graupel_prr_gsp",
        "ser_graupel_prs_gsp",
        "ser_graupel_pri_gsp",
        "ser_graupel_prg_gsp",
        "ser_graupel_qrsflux"
    )

    mixT_name = (
        "ser_graupel_temperature",
        "ser_graupel_qv",
        "ser_graupel_qc",
        "ser_graupel_qi",
        "ser_graupel_qr",
        "ser_graupel_qs",
        "ser_graupel_qg"
    )

    velocity_field_name = (
        "rhoqrV_old_kup",
        "rhoqsV_old_kup",
        "rhoqgV_old_kup",
        "rhoqiV_old_kup",
        "Vnew_r",
        "Vnew_s",
        "Vnew_g",
        "Vnew_i"
    )


    ser_field = {}
    predict_field = {}
    velocity_field = {}
    for item in ser_field_name:
        predict_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size,k_size), dtype=float64))
    for item in velocity_field_name:
        velocity_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))    

    dt = 2.0
    ser_field["ser_graupel_dz"] =np_as_located_field(CellDim,KDim)(np.ones((cell_size, k_size),dtype=float64)) 
    ser_field["ser_graupel_temperature"] = np_as_located_field(CellDim,KDim)(np.full((cell_size, k_size),fill_value=273.15,dtype=float64))
    ser_field["ser_graupel_pres"] = np_as_located_field(CellDim,KDim)(np.full((cell_size, k_size),fill_value=1.e5,dtype=float64))
    ser_field["ser_graupel_rho"] = np_as_located_field(CellDim,KDim)(np.ones((cell_size, k_size),dtype=float64))
    ser_field["ser_graupel_qv"] = np_as_located_field(CellDim,KDim)(np.full((cell_size, k_size),fill_value=1.e-3,dtype=float64))
    ser_field["ser_graupel_qc"] = np_as_located_field(CellDim,KDim)(np.full((cell_size, k_size),fill_value=1.e-5,dtype=float64))
    ser_field["ser_graupel_qi"] = np_as_located_field(CellDim,KDim)(np.full((cell_size, k_size),fill_value=1.e-5,dtype=float64))
    ser_field["ser_graupel_qr"] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size, k_size),dtype=float64))
    ser_field["ser_graupel_qs"] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size, k_size),dtype=float64))
    ser_field["ser_graupel_qg"] = np_as_located_field(CellDim,KDim)(np.zeros((cell_size, k_size),dtype=float64))
    ser_field["ser_graupel_qnc"] = np_as_located_field(CellDim,KDim)(np.full((cell_size, k_size), fill_value=2.e8, dtype=float64))
    l_cv = True
    lpres_pri = True
    ithermo_water = int32(2)
    ldass_lhn = True
    ldiag_ttend = True
    ldiag_qtend = True
    num_cells = int32(8)
    kstart_moist = int32(0)
    kend = int32(7)
    kend_moist = int32(7)

    # run graupel

    _graupel(
        kstart_moist,
        kend,
        dt,
        ser_field["ser_graupel_dz"],
        ser_field["ser_graupel_temperature"],
        ser_field["ser_graupel_pres"],
        ser_field["ser_graupel_rho"],
        ser_field["ser_graupel_qv"],
        ser_field["ser_graupel_qc"],
        ser_field["ser_graupel_qi"],
        ser_field["ser_graupel_qr"],
        ser_field["ser_graupel_qs"],
        ser_field["ser_graupel_qg"],
        ser_field["ser_graupel_qnc"],
        l_cv,
        ithermo_water,
        out=(
            predict_field["ser_graupel_temperature"],
            predict_field["ser_graupel_qv"],
            predict_field["ser_graupel_qc"],
            predict_field["ser_graupel_qi"],
            predict_field["ser_graupel_qr"],
            predict_field["ser_graupel_qs"],
            predict_field["ser_graupel_qg"],
            # used in graupel scheme, do not output to outer world
            velocity_field["rhoqrV_old_kup"],
            velocity_field["rhoqsV_old_kup"],
            velocity_field["rhoqgV_old_kup"],
            velocity_field["rhoqiV_old_kup"],
            velocity_field["Vnew_r"],
            velocity_field["Vnew_s"],
            velocity_field["Vnew_g"],
            velocity_field["Vnew_i"]
        ),
        offset_provider={}
    )

    '''
    if (const_data["ser_graupel_ldiag_ttend"]):
        _graupel_t_tendency(
            const_data["ser_graupel_dt"],
            predict_field["ser_graupel_temperature"],
            ser_field["ser_graupel_temperature"],
            out=(
                predict_field["ser_graupel_ddt_tend_t"]
            ),
            offset_provider={}
        )

    if (const_data["ser_graupel_ldiag_qtend"]):
        _graupel_q_tendency(
            const_data["ser_graupel_dt"],
            predict_field["ser_graupel_qv"],
            predict_field["ser_graupel_qc"],
            predict_field["ser_graupel_qi"],
            predict_field["ser_graupel_qr"],
            predict_field["ser_graupel_qs"],
            predict_field["ser_graupel_qg"],
            ser_field["ser_graupel_qv"],
            ser_field["ser_graupel_qc"],
            ser_field["ser_graupel_qi"],
            ser_field["ser_graupel_qr"],
            ser_field["ser_graupel_qs"],
            ser_field["ser_graupel_qg"],
            out=(
                predict_field["ser_graupel_ddt_tend_qv"],
                predict_field["ser_graupel_ddt_tend_qc"],
                predict_field["ser_graupel_ddt_tend_qi"],
                predict_field["ser_graupel_ddt_tend_qr"],
                predict_field["ser_graupel_ddt_tend_qs"],
                predict_field["ser_graupel_ddt_tend_qg"]
            ),
            offset_provider={}
        )

    _graupel_flux(
        const_data["ser_graupel_kstart_moist"],
        const_data["ser_graupel_kend"],
        ser_field["ser_graupel_rho"],
        predict_field["ser_graupel_qr"],
        predict_field["ser_graupel_qs"],
        predict_field["ser_graupel_qi"],
        predict_field["ser_graupel_qg"],
        velocity_field["Vnew_r"],
        velocity_field["Vnew_s"],
        velocity_field["Vnew_i"],
        velocity_field["Vnew_g"],
        velocity_field["rhoqrV_old_kup"],
        velocity_field["rhoqsV_old_kup"],
        velocity_field["rhoqiV_old_kup"],
        velocity_field["rhoqgV_old_kup"],
        const_data["ser_graupel_lpres_pri"],
        const_data["ser_graupel_ldass_lhn"],
        out=(
            predict_field["ser_graupel_prr_gsp"],
            predict_field["ser_graupel_prs_gsp"],
            predict_field["ser_graupel_pri_gsp"],
            predict_field["ser_graupel_prg_gsp"],
            predict_field["ser_graupel_qrsflux"]
        ),
        offset_provider={}
    )
    '''
