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
from gt4py.next.ffront.fbuiltins import (
    float64,
    int32
)

from icon4py.atm_phy_schemes.gscp_graupel_Ong import graupel, _graupel, _graupel_scan, _graupel_t_tendency, _graupel_q_tendency, _graupel_flux_scan
from icon4py.model.common.dimension import CellDim, KDim
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn_cpu import (
    run_gtfn,
    run_gtfn_cached,
    run_gtfn_imperative,
)


def test_graupel_Ong_serialized_data():

    # this a test with trivial data

    backend = run_gtfn

    lpres_pri = True
    ldass_lhn = True

    cell_size, k_size = 2, 2


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
        "ser_graupel_prg_gsp",
        "ser_graupel_pri_gsp",
        "ser_graupel_qrsflux"
    )

    ser_redundant_name = (
        "ser_graupel_rho_kup",
        "ser_graupel_Crho1o2_kup",
        "ser_graupel_Crhofac_qi_kup",
        "ser_graupel_Cvz0s_kup",
        "ser_graupel_qvsw_kup",
        "ser_graupel_clddist",
        "ser_graupel_klev"
    )
    ser_tend_name = (
        "ser_graupel_szdep_v2i",
        "ser_graupel_szsub_v2i",
        "ser_graupel_snucl_v2i",
        "ser_graupel_scfrz_c2i",
        "ser_graupel_simlt_i2c",
        "ser_graupel_sicri_i2g",
        "ser_graupel_sidep_v2i",
        "ser_graupel_sdaut_i2s",
        "ser_graupel_saggs_i2s",
        "ser_graupel_saggg_i2g",
        "ser_graupel_siaut_i2s",
        "ser_graupel_ssmlt_s2r",
        "ser_graupel_srims_c2s",
        "ser_graupel_ssdep_v2s",
        "ser_graupel_scosg_s2g",
        "ser_graupel_sgmlt_g2r",
        "ser_graupel_srcri_r2g",
        "ser_graupel_sgdep_v2g",
        "ser_graupel_srfrz_r2g",
        "ser_graupel_srimg_c2g"
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

    # transform serialized input and predicted arrays into GT4py Fields and construct predicted output arrays
    ser_field = {}
    tend_field = {}
    redundant_field = {}
    predict_field = {}
    velocity_field = {}
    for item in ser_field_name:
        if (item == "ser_graupel_temperature"):
            ser_field[item] = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=273.15,dtype=float64))
        elif (item == "ser_graupel_rho"):
            ser_field[item] = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=1.0,dtype=float64))
        elif (item == "ser_graupel_pres"):
            ser_field[item] = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=101325.0,dtype=float64))
        elif (item == "ser_graupel_dz"):
            ser_field[item] = np_as_located_field(CellDim, KDim)(np.full((cell_size, k_size), fill_value=50.0,dtype=float64))
        else:
            ser_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size,k_size), dtype=float64))
        predict_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size,k_size), dtype=float64))
    for item in ser_tend_name:
        tend_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
    for item in ser_redundant_name:
        if (item != "ser_graupel_klev"):
            redundant_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))
        else:
            redundant_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=int32))
    for item in velocity_field_name:
        velocity_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))

    # run graupel

    _graupel_scan(
        0,
        1,
        2.0,
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
        True,
        1,
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
            velocity_field["Vnew_i"],
            redundant_field["ser_graupel_clddist"],
            redundant_field["ser_graupel_rho_kup"],
            redundant_field["ser_graupel_Crho1o2_kup"],
            redundant_field["ser_graupel_Crhofac_qi_kup"],
            redundant_field["ser_graupel_Cvz0s_kup"],
            redundant_field["ser_graupel_qvsw_kup"],
            redundant_field["ser_graupel_klev"],
            tend_field["ser_graupel_szdep_v2i"],
            tend_field["ser_graupel_szsub_v2i"],
            tend_field["ser_graupel_snucl_v2i"],
            tend_field["ser_graupel_scfrz_c2i"],
            tend_field["ser_graupel_simlt_i2c"],
            tend_field["ser_graupel_sicri_i2g"],
            tend_field["ser_graupel_sidep_v2i"],
            tend_field["ser_graupel_sdaut_i2s"],
            tend_field["ser_graupel_saggs_i2s"],
            tend_field["ser_graupel_saggg_i2g"],
            tend_field["ser_graupel_siaut_i2s"],
            tend_field["ser_graupel_ssmlt_s2r"],
            tend_field["ser_graupel_srims_c2s"],
            tend_field["ser_graupel_ssdep_v2s"],
            tend_field["ser_graupel_scosg_s2g"],
            tend_field["ser_graupel_sgmlt_g2r"],
            tend_field["ser_graupel_srcri_r2g"],
            tend_field["ser_graupel_sgdep_v2g"],
            tend_field["ser_graupel_srfrz_r2g"],
            tend_field["ser_graupel_srimg_c2g"]
        ),
        offset_provider={}
    )


    '''
    _graupel(
        0,
        1,
        2.0,
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
        True,
        1,
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

    '''
    graupel(
        0,
        1,
        2.0,
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
        True,
        1,
        velocity_field["rhoqrV_old_kup"],
        velocity_field["rhoqsV_old_kup"],
        velocity_field["rhoqgV_old_kup"],
        velocity_field["rhoqiV_old_kup"],
        velocity_field["Vnew_r"],
        velocity_field["Vnew_s"],
        velocity_field["Vnew_g"],
        velocity_field["Vnew_i"],
        offset_provider={}
    )
    '''

    '''
        redundant_field["ser_graupel_clddist"],
        redundant_field["ser_graupel_rho_kup"],
        redundant_field["ser_graupel_Crho1o2_kup"],
        redundant_field["ser_graupel_Crhofac_qi_kup"],
        redundant_field["ser_graupel_Cvz0s_kup"],
        redundant_field["ser_graupel_qvsw_kup"],
        redundant_field["ser_graupel_klev"],
        tend_field["ser_graupel_szdep_v2i"],
        tend_field["ser_graupel_szsub_v2i"],
        tend_field["ser_graupel_snucl_v2i"],
        tend_field["ser_graupel_scfrz_c2i"],
        tend_field["ser_graupel_simlt_i2c"],
        tend_field["ser_graupel_sicri_i2g"],
        tend_field["ser_graupel_sidep_v2i"],
        tend_field["ser_graupel_sdaut_i2s"],
        tend_field["ser_graupel_saggs_i2s"],
        tend_field["ser_graupel_saggg_i2g"],
        tend_field["ser_graupel_siaut_i2s"],
        tend_field["ser_graupel_ssmlt_s2r"],
        tend_field["ser_graupel_srims_c2s"],
        tend_field["ser_graupel_ssdep_v2s"],
        tend_field["ser_graupel_scosg_s2g"],
        tend_field["ser_graupel_sgmlt_g2r"],
        tend_field["ser_graupel_srcri_r2g"],
        tend_field["ser_graupel_sgdep_v2g"],
        tend_field["ser_graupel_srfrz_r2g"],
        tend_field["ser_graupel_srimg_c2g"],

    '''

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
            ser_field["ser_graupel_qv"],
            ser_field["ser_graupel_qc"],
            ser_field["ser_graupel_qi"],
            ser_field["ser_graupel_qr"],
            ser_field["ser_graupel_qs"],
            out=(
                predict_field["ser_graupel_ddt_tend_qv"],
                predict_field["ser_graupel_ddt_tend_qc"],
                predict_field["ser_graupel_ddt_tend_qi"],
                predict_field["ser_graupel_ddt_tend_qr"],
                predict_field["ser_graupel_ddt_tend_qs"]
            ),
            offset_provider={}
        )


    _graupel_flux_scan(
        const_data["ser_graupel_kstart_moist"],
        const_data["ser_graupel_kend"],
        ser_field["ser_graupel_rho"],
        predict_field["ser_graupel_qr"],
        predict_field["ser_graupel_qs"],
        predict_field["ser_graupel_qg"],
        predict_field["ser_graupel_qi"],
        velocity_field["Vnew_r"],
        velocity_field["Vnew_s"],
        velocity_field["Vnew_g"],
        velocity_field["Vnew_i"],
        velocity_field["rhoqrV_old_kup"],
        velocity_field["rhoqsV_old_kup"],
        velocity_field["rhoqgV_old_kup"],
        velocity_field["rhoqiV_old_kup"],
        lpres_pri,
        ldass_lhn,
        out=(
            predict_field["ser_graupel_prr_gsp"],
            predict_field["ser_graupel_prs_gsp"],
            predict_field["ser_graupel_prg_gsp"],
            predict_field["ser_graupel_pri_gsp"],
            predict_field["ser_graupel_qrsflux"],
            redundant_field["ser_graupel_klev"]
        ),
        offset_provider={}
    )
    '''

test_graupel_Ong_serialized_data()
