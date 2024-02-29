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
import os
import sys

import pytest

from gt4py.next.ffront.fbuiltins import (
    float64,
    int32
)

from icon4py.model.atmosphere.physics_schemes.single_moment_six_class_microphysics.gscp_graupel_Ong import _graupel_scan, _graupel_t_tendency, _graupel_q_tendency, _graupel_flux_scan
from icon4py.model.atmosphere.physics_schemes.single_moment_six_class_microphysics.gscp_graupel_Ong import GraupelGlobalConstants, GraupelFunctionConstants
from typing import Final
from icon4py.model.common.dimension import CellDim, KDim
from gt4py.next import as_field
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached

import serialbox as ser


@pytest.mark.parametrize("date_no,Nblocks,rank,debug,data_output", [(date_i,120,rank_j,False,False) for date_i in range(1) for rank_j in range(5)])
def test_graupel_Ong_serialized_data(date_no,Nblocks,rank,debug,data_output):

    backend = run_gtfn

    lpres_pri = True  # TODO: may need to be read from serialized data, default is True. We now manually set to True for debugging
    ldass_lhn = True  # TODO: may need to be read from serialized data, default is False. We now manually set to True for debugging
    tendency_serialization = True


    #mpi_ranks = np.arange(0, 10, dtype=int)
    initial_date = "2008-09-01T00:00:00.000"
    dates = ("2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000")
    blocks = tuple(i+1 for i in range(Nblocks))
    print()
    print("date no: ", date_no)
    print("rank: ", rank)
    print("Nblocks; ", Nblocks)
    print("debug, output: ", debug, data_output)

    # please put the data in the serialbox directory
    #script_dir = os.path.dirname(__file__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir+'/serialbox/data_dir/'
    #base_dir = "/home/ong/Data/nh_wk_rerun_complete/data_dir/"
    try:
        serializer = ser.Serializer(ser.OpenModeKind.Read, base_dir, "wk__rank"+str(rank))
        savePoints = serializer.savepoint_list()
    except ser.SerialboxError as e:
        print(f"serializer: error: {e}")
        print("Data download link: https://polybox.ethz.ch/index.php/s/8qWeEg5JNAiNeGk")
        sys.exit(1)


    if (debug):
        for item in serializer.get_savepoint("call-graupel-entrance"):
            print(item)
        for item in serializer.get_savepoint("init-graupel"):
            print(item)
        for item in serializer.fieldnames():
            print(item)

        # Nblock = 0
        # for key, value in savePoints.items():
        #    if (key == 'serial_state' and value == 1): Nblock += 1
        # print(item)
        # for item in savePoints:
        #    print(item)

        if (isinstance(savePoints,dict)):
            print("save point is a dictionary")
        elif (isinstance(savePoints,list)):
            print("save point is a list")

        entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[1]]["date"][dates[1]]
        qv1 = serializer.read("ser_graupel_qv", entrance_savePoint)
        entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[2]]["date"][dates[1]]
        qv2 = serializer.read("ser_graupel_qv", entrance_savePoint)
        qv = np.concatenate((qv1, qv2), axis=0)
        for k in range(5):
            for i in range(16):
                print(qv[i][k], qv[i + 16][k], " --- ", qv1[i][k], qv2[i][k])

        # DOES NOT WORK
        #for item in entrance_savePoint:
        #    print(item)

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

    mixT_name = (
        "ser_graupel_temperature",
        "ser_graupel_qv",
        "ser_graupel_qc",
        "ser_graupel_qi",
        "ser_graupel_qr",
        "ser_graupel_qs",
        "ser_graupel_qg",
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

    ser_const_name = (
        "ser_graupel_dt",
        "ser_graupel_qc0",
        "ser_graupel_qi0",
        "ser_graupel_kstart_moist",
        "ser_graupel_l_cv",
        "ser_graupel_ithermo_water",
        "ser_graupel_ldiag_ttend",
        "ser_graupel_ldiag_qtend",
        "ser_graupel_istart",
        "ser_graupel_iend",
        "ser_graupel_kend",
        "ser_graupel_nvec"
    )

    ser_const_type = (
        "float64",
        "float64",
        "float64",
        "int32",
        "bool",
        "int32",
        "bool",
        "bool",
        "int32",
        "int32",
        "int32",
        "int32"
    )

    ser_tune_name = (
        "ser_init_graupel_tune_zceff_min",
        "ser_init_graupel_tune_v0snow",
        "ser_init_graupel_tune_zvz0i",
        "ser_init_graupel_tune_icesedi_exp",
        "ser_init_graupel_tune_mu_rain",
        "ser_init_graupel_tune_rain_n0_factor",
        "ser_init_graupel_iautocon",
        "ser_init_graupel_isnow_n0temp",
        "ser_init_graupel_zqmin",
        "ser_init_graupel_zeps"
    )

    ser_gscp_data_name = (
        "ser_init_graupel_zceff_min",
        "ser_init_graupel_v0snow",
        "ser_init_graupel_zvz0i",
        "ser_init_graupel_icesedi_exp",
        "ser_init_graupel_mu_rain",
        "ser_init_graupel_rain_n0_factor",
        "ser_init_graupel_ccsrim",
        "ser_init_graupel_ccsagg",
        "ser_init_graupel_ccsdep",
        "ser_init_graupel_ccsvel",
        "ser_init_graupel_ccsvxp",
        "ser_init_graupel_ccslam",
        "ser_init_graupel_ccslxp",
        "ser_init_graupel_ccswxp",
        "ser_init_graupel_ccsaxp",
        "ser_init_graupel_ccsdxp",
        "ser_init_graupel_ccshi1",
        "ser_init_graupel_ccdvtp",
        "ser_init_graupel_ccidep",
        "ser_init_graupel_zn0r",
        "ser_init_graupel_zar",
        "ser_init_graupel_zcevxp",
        "ser_init_graupel_zcev",
        "ser_init_graupel_zbevxp",
        "ser_init_graupel_zbev",
        "ser_init_graupel_zvzxp",
        "ser_init_graupel_zvz0r"
    )

    ser_tune_type = (
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int32",
        "float64",
        "float64",
        "float64"
    )

    # construct constant data dictionary
    const_data = {}
    for item in ser_const_name:
        const_data[item] = None

    # construct tuning constant data dictionary
    tune_data = {}
    for item in ser_tune_name:
        tune_data[item] = None

    # construct tuning constant data dictionary
    gscp_data = {}
    for item in ser_gscp_data_name:
        gscp_data[item] = None

    # construct serialized data dictionary
    ser_data = {}
    ref_data = {}
    for item in ser_field_name:
        ser_data[item] = None
        ref_data[item] = None

    # construct serialized tendency data dictionary
    tend_data = {}
    for item in ser_tend_name:
        tend_data[item] = None

    # read constants
    for item_no, item in enumerate(ser_const_name):
        entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[0]]["date"][dates[date_no]]
        const_data[item] = serializer.read(item, entrance_savePoint)[0]
        for i in range(Nblocks - 1):
            entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[i + 1]]["date"][dates[date_no]]
            if (item != "ser_graupel_istart" and item != "ser_graupel_iend" and item != "ser_graupel_nvec"):
                if (const_data[item] != serializer.read(item, entrance_savePoint)[0]):
                    print(const_data[item], serializer.read(item, entrance_savePoint)[0], ser_const_type[item_no])
                    sys.exit(0)

    for item in ser_tune_name:
        init_savePoint = serializer.savepoint["init-graupel"]["serial_state"][1]["date"][initial_date]
        tune_data[item] = serializer.read(item, init_savePoint)[0]

    for item in ser_gscp_data_name:
        init_savePoint = serializer.savepoint["init-graupel"]["serial_state"][1]["date"][initial_date]
        gscp_data[item] = serializer.read(item, init_savePoint)[0]

    # read serialized input and reference data
    for item in ser_field_name:
        entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[0]]["date"][dates[date_no]]
        exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[0]]["date"][dates[date_no]]
        ser_data[item] = serializer.read(item, entrance_savePoint)
        ref_data[item] = serializer.read(item, exit_savePoint)
        for i in range(Nblocks-1):
            entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[i+1]]["date"][dates[date_no]]
            exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[i+1]]["date"][dates[date_no]]
            ser_data[item] = np.concatenate((ser_data[item],serializer.read(item, entrance_savePoint)),axis=0)
            ref_data[item] = np.concatenate((ref_data[item], serializer.read(item, exit_savePoint)), axis=0)

    # read serialized rendency data
    if tendency_serialization:
        for item in ser_tend_name:
            exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[0]]["date"][dates[date_no]]
            tend_data[item] = serializer.read(item, exit_savePoint)
            for i in range(Nblocks-1):
                exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[i + 1]]["date"][dates[date_no]]
                tend_data[item] = np.concatenate((tend_data[item], serializer.read(item, exit_savePoint)), axis=0)

    (cell_size, k_size) = ser_data["ser_graupel_temperature"].shape
    const_data["ser_graupel_kstart_moist"] -= int32(1)
    const_data["ser_graupel_kend"] -= int32(1)
    if (tune_data["ser_init_graupel_tune_v0snow"] < 0.0):
        tune_data["ser_init_graupel_tune_v0snow"] = 20.0 # default value for graupel scheme

    check_graupel_const: Final = GraupelGlobalConstants()
    check_graupel_funcConst: Final = GraupelFunctionConstants()

    # print out some constants and indices
    print("Below is a summary of constants used in the experiment:")
    print("cell and k dimensions  : ", ser_data["ser_graupel_temperature"].shape)
    print("qrsflux shape          : ", ser_data["ser_graupel_qrsflux"].shape)
    print("qnc shape              : ", ser_data["ser_graupel_qnc"].shape)
    print("dt                     : ", const_data["ser_graupel_dt"])
    print("qc0                    : ", const_data["ser_graupel_qc0"], check_graupel_funcConst.GrFuncConst_qc0)
    print("qi0                    : ", const_data["ser_graupel_qi0"], check_graupel_funcConst.GrFuncConst_qi0)
    print("qnc                    : ", ser_data["ser_graupel_qnc"][0])
    print("istart                 : ", const_data["ser_graupel_istart"])
    print("iend                   : ", const_data["ser_graupel_iend"])
    print("kstart_moist           : ", const_data["ser_graupel_kstart_moist"])
    print("kend                   : ", const_data["ser_graupel_kend"])
    print("l_cv                   : ", const_data["ser_graupel_l_cv"])
    print("ithermo_water          : ", const_data["ser_graupel_ithermo_water"])
    print("ldiag_ttend            : ", const_data["ser_graupel_ldiag_ttend"])
    print("ldiag_qtend            : ", const_data["ser_graupel_ldiag_qtend"])
    print("tune_zceff_min         : ", tune_data["ser_init_graupel_tune_zceff_min"], check_graupel_const.GrConst_ceff_min)
    print("tune_v0snow            : ", tune_data["ser_init_graupel_tune_v0snow"], check_graupel_const.GrConst_v0snow)
    print("tune_zvz0i             : ", tune_data["ser_init_graupel_tune_zvz0i"], check_graupel_const.GrConst_vz0i)
    print("tune_icesedi_exp       : ", tune_data["ser_init_graupel_tune_icesedi_exp"], check_graupel_const.GrConst_icesedi_exp)
    print("tune_mu_rain           : ", tune_data["ser_init_graupel_tune_mu_rain"], check_graupel_const.GrConst_mu_rain)
    print("tune_rain_n0_factor    : ", tune_data["ser_init_graupel_tune_rain_n0_factor"], check_graupel_const.GrConst_rain_n0_factor)
    print("iautocon               : ", tune_data["ser_init_graupel_iautocon"], check_graupel_const.GrConst_iautocon)
    print("isnow_n0temp           : ", tune_data["ser_init_graupel_isnow_n0temp"], check_graupel_const.GrConst_isnow_n0temp)
    print("zqmin                  : ", tune_data["ser_init_graupel_zqmin"], check_graupel_const.GrConst_qmin)
    print("zeps                   : ", tune_data["ser_init_graupel_zeps"], check_graupel_const.GrConst_eps)

    print("zceff_min              : ", gscp_data["ser_init_graupel_zceff_min"],check_graupel_const.GrConst_ceff_min)
    print("v0snow                 : ", gscp_data["ser_init_graupel_v0snow"], check_graupel_const.GrConst_v0snow)
    print("zvz0i                  : ", gscp_data["ser_init_graupel_zvz0i"], check_graupel_const.GrConst_vz0i)
    print("icesedi_exp            : ", gscp_data["ser_init_graupel_icesedi_exp"],check_graupel_const.GrConst_icesedi_exp)
    print("mu_rain                : ", gscp_data["ser_init_graupel_mu_rain"], check_graupel_const.GrConst_mu_rain)
    print("rain_n0_factor         : ", gscp_data["ser_init_graupel_rain_n0_factor"],check_graupel_const.GrConst_rain_n0_factor)
    print("ccsrim                 : ", gscp_data["ser_init_graupel_ccsrim"], check_graupel_const.GrConst_ccsrim)
    print("ccsagg                 : ", gscp_data["ser_init_graupel_ccsagg"], check_graupel_const.GrConst_ccsagg)
    print("ccsdep                 : ", gscp_data["ser_init_graupel_ccsdep"], check_graupel_const.GrConst_ccsdep)
    print("ccsvel                 : ", gscp_data["ser_init_graupel_ccsvel"], check_graupel_const.GrConst_ccsvel)
    print("ccsvxp                 : ", gscp_data["ser_init_graupel_ccsvxp"], check_graupel_const.GrConst_ccsvxp)
    print("ccslam                 : ", gscp_data["ser_init_graupel_ccslam"], check_graupel_const.GrConst_ccslam)
    print("ccslxp                 : ", gscp_data["ser_init_graupel_ccslxp"], check_graupel_const.GrConst_ccslxp)
    print("ccswxp                 : ", gscp_data["ser_init_graupel_ccswxp"], check_graupel_const.GrConst_ccswxp)
    print("ccsaxp                 : ", gscp_data["ser_init_graupel_ccsaxp"], check_graupel_const.GrConst_ccsaxp)
    print("ccsdxp                 : ", gscp_data["ser_init_graupel_ccsdxp"], check_graupel_const.GrConst_ccsdxp)
    print("ccshi1                 : ", gscp_data["ser_init_graupel_ccshi1"], check_graupel_const.GrConst_ccshi1)
    print("ccdvtp                 : ", gscp_data["ser_init_graupel_ccdvtp"], check_graupel_const.GrConst_ccdvtp)
    print("ccidep                 : ", gscp_data["ser_init_graupel_ccidep"], check_graupel_const.GrConst_ccidep)
    print("zn0r                   : ", gscp_data["ser_init_graupel_zn0r"], check_graupel_const.GrConst_n0r)
    print("zar                    : ", gscp_data["ser_init_graupel_zar"], check_graupel_const.GrConst_ar)
    print("zcevxp                 : ", gscp_data["ser_init_graupel_zcevxp"], check_graupel_const.GrConst_cevxp)
    print("zcev                   : ", gscp_data["ser_init_graupel_zcev"], check_graupel_const.GrConst_cev)
    print("zbevxp                 : ", gscp_data["ser_init_graupel_zbevxp"], check_graupel_const.GrConst_bevxp)
    print("zbev                   : ", gscp_data["ser_init_graupel_zbev"], check_graupel_const.GrConst_bev)
    print("zvzxp                  : ", gscp_data["ser_init_graupel_zvzxp"], check_graupel_const.GrConst_vzxp)
    print("zvz0r                  : ", gscp_data["ser_init_graupel_zvz0r"], check_graupel_const.GrConst_vz0r)

    print("------------------------------------------------------")
    print()


    # check whether the tuning parameters are the same. If not, exit the program
    def comparison(description: str, a):
        if (description in ser_const_name):
            if (const_data[description] != a):
                print(description, " is NOT equal. Pleas check.")
                sys.exit()
        elif (description in ser_tune_name):
            if (tune_data[description] != a):
                print(description, " is NOT equal. Pleas check.")
                sys.exit()
    comparison("ser_graupel_qc0", check_graupel_funcConst.GrFuncConst_qc0)
    comparison("ser_graupel_qi0", check_graupel_funcConst.GrFuncConst_qi0)
    comparison("ser_init_graupel_tune_zceff_min", check_graupel_const.GrConst_ceff_min)
    comparison("ser_init_graupel_tune_v0snow", check_graupel_const.GrConst_v0snow)
    comparison("ser_init_graupel_tune_zvz0i", check_graupel_const.GrConst_vz0i)
    comparison("ser_init_graupel_tune_icesedi_exp", check_graupel_const.GrConst_icesedi_exp)
    comparison("ser_init_graupel_tune_mu_rain", check_graupel_const.GrConst_mu_rain)
    comparison("ser_init_graupel_tune_rain_n0_factor", check_graupel_const.GrConst_rain_n0_factor)
    comparison("ser_init_graupel_iautocon", check_graupel_const.GrConst_iautocon)
    comparison("ser_init_graupel_isnow_n0temp", check_graupel_const.GrConst_isnow_n0temp)
    comparison("ser_init_graupel_zqmin", check_graupel_const.GrConst_qmin)
    comparison("ser_init_graupel_zeps", check_graupel_const.GrConst_eps)

    # if ldass_lhn is not turned on, the serialized qrsflux has only dimension (1)
    if (not ldass_lhn):
        ser_data["ser_graupel_qrsflux"] = np.zeros((cell_size, k_size), dtype=float64)
        ref_data["ser_graupel_qrsflux"] = np.zeros((cell_size, k_size), dtype=float64)

    # qnc is a constant in Fortran, we transform it into (cell_size, k_size) dimension
    #ser_data["ser_graupel_qnc"] = np.full((cell_size, k_size), fill_value=ser_data["ser_graupel_qnc"][0], dtype=float64)

    # expand the 2D precipitation fluxes (prr, pri, prs, prg) to 3D arrays
    for item in ("ser_graupel_prr_gsp", "ser_graupel_prs_gsp", "ser_graupel_prg_gsp", "ser_graupel_pri_gsp"):
        ser_data[item] = np.expand_dims(ser_data[item], axis=0)
        ref_data[item] = np.expand_dims(ref_data[item], axis=0)
        ser_data[item] = np.flip(np.transpose(np.pad(ser_data[item], [(0, k_size-1), (0, 0)], mode='constant', constant_values=0)),axis=1)
        ref_data[item] = np.flip(np.transpose(np.pad(ref_data[item], [(0, k_size-1), (0, 0)], mode='constant', constant_values=0)),axis=1)

    # checking shape
    for item in ser_field_name:
        if (item != "ser_graupel_qnc" and (cell_size, k_size) != ser_data[item].shape):
            print("The array size is not fixed. The shape of temperature field is ", cell_size, k_size, ", while the shape of ", item, " is ", ser_data[item].shape, "Please check.")
            sys.exit()

    # transform serialized input and predicted arrays into GT4py Fields and construct predicted output arrays
    ser_field = {}
    tend_field = {}
    redundant_field = {}
    predict_field = {}
    velocity_field = {}
    for item in ser_field_name:
        if (item != "ser_graupel_qnc"):
            ser_field[item] = as_field((CellDim, KDim), np.array(ser_data[item], dtype=float64))
            predict_field[item] = as_field((CellDim, KDim), np.zeros((cell_size,k_size), dtype=float64))
        else:
            ser_field[item] = as_field((CellDim,), np.array(ser_data[item], dtype=float64))
            predict_field[item] = as_field((CellDim,), np.zeros(cell_size, dtype=float64))
    for item in ser_tend_name:
        tend_field[item] = as_field((CellDim, KDim), np.zeros((cell_size, k_size), dtype=float64))
    for item in ser_redundant_name:
        if (item != "ser_graupel_klev"):
            redundant_field[item] = as_field((CellDim, KDim), np.zeros((cell_size, k_size), dtype=float64))
        else:
            redundant_field[item] = as_field((CellDim, KDim), np.zeros((cell_size, k_size), dtype=int32))
    for item in velocity_field_name:
        velocity_field[item] = as_field((CellDim, KDim), np.zeros((cell_size, k_size), dtype=float64))

    print("max and min dz: ", ser_field["ser_graupel_dz"].asnumpy().min(), ser_field["ser_graupel_dz"].asnumpy().max())
    print("max and min pres: ", ser_field["ser_graupel_pres"].asnumpy().min(), ser_field["ser_graupel_pres"].asnumpy().max())

    # run graupel

    _graupel_scan(
        const_data["ser_graupel_kstart_moist"],
        const_data["ser_graupel_kend"],
        const_data["ser_graupel_dt"],
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
        const_data["ser_graupel_l_cv"],
        const_data["ser_graupel_ithermo_water"],
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

    if const_data["ser_graupel_ldiag_ttend"]:
        _graupel_t_tendency(
            const_data["ser_graupel_dt"],
            predict_field["ser_graupel_temperature"],
            ser_field["ser_graupel_temperature"],
            out=(
                predict_field["ser_graupel_ddt_tend_t"]
            ),
            offset_provider={}
        )

    if const_data["ser_graupel_ldiag_qtend"]:
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


    diff_data = {}
    for item in mixT_name:
        diff_data[item] = np.abs(predict_field[item].asnumpy() - ref_data[item]) / ref_data[item]
        diff_data[item] = np.where(np.abs(ref_data[item]) <= 1.e-20 , predict_field[item].asnumpy(), diff_data[item])

    if tendency_serialization:
        for item in ser_tend_name:
            diff_data[item] = np.abs(tend_field[item].asnumpy() - tend_data[item]) / tend_data[item]
            diff_data[item] = np.where(np.abs(tend_data[item]) <= 1.e-30, tend_field[item].asnumpy(), tend_data[item])

    if tendency_serialization:
        print("Max predict-ref tendency difference:")
        for item in ser_tend_name:
            print(item, ": ", np.abs(tend_field[item].asnumpy() - tend_data[item]).max(), diff_data[item].max())

    print("Max predict-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(predict_field[item].asnumpy() - ref_data[item]).max(), diff_data[item].max())

    print("Max init-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(ser_field[item].asnumpy() - ref_data[item]).max())

    print("Max init:")
    for item in mixT_name:
        print(item, ": ", np.abs(ser_data[item]).max())

    print("Max ref:")
    for item in mixT_name:
        print(item, ": ", np.abs(ref_data[item]).max())

    print("Max predict:")
    for item in mixT_name:
        print(item, ": ", predict_field[item].asnumpy().max())

    print("Max abs predict:")
    for item in mixT_name:
        print(item, ": ", np.abs(predict_field[item].asnumpy()).max())

    print("Max init-ref total difference:")
    print("qv: ", np.abs(
        ser_field["ser_graupel_qv"].asnumpy() - ref_data["ser_graupel_qv"] +
        ser_field["ser_graupel_qc"].asnumpy() - ref_data["ser_graupel_qc"] +
        ser_field["ser_graupel_qi"].asnumpy() - ref_data["ser_graupel_qi"] +
        ser_field["ser_graupel_qr"].asnumpy() - ref_data["ser_graupel_qr"] +
        ser_field["ser_graupel_qs"].asnumpy() - ref_data["ser_graupel_qs"] +
        ser_field["ser_graupel_qg"].asnumpy() - ref_data["ser_graupel_qg"]
    ).max())

    print("Max predict-init total difference:")
    print("qv: ", np.abs(
        predict_field["ser_graupel_qv"].asnumpy() - ref_data["ser_graupel_qv"] +
        predict_field["ser_graupel_qc"].asnumpy() - ref_data["ser_graupel_qc"] +
        predict_field["ser_graupel_qi"].asnumpy() - ref_data["ser_graupel_qi"] +
        predict_field["ser_graupel_qr"].asnumpy() - ref_data["ser_graupel_qr"] +
        predict_field["ser_graupel_qs"].asnumpy() - ref_data["ser_graupel_qs"] +
        predict_field["ser_graupel_qg"].asnumpy() - ref_data["ser_graupel_qg"]
    ).max())

    if data_output:
        with open(base_dir+'analysis_dz_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    f.write(" {0:.20e} ".format(ser_field["ser_graupel_dz"].asnumpy()[i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_ser_rho_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    f.write(" {0:.20e} ".format(ser_data["ser_graupel_rho"][i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_ref_rho_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    f.write(" {0:.20e} ".format(ref_data["ser_graupel_rho"][i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_predict_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    for item in mixT_name:
                        f.write(" {0:.20e} ".format(predict_field[item].asnumpy()[i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_tend_rank' + str(rank) + '.dat', 'w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in ser_tend_name:
                        f.write(" {0:.20e} ".format(tend_field[item].asnumpy()[i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_redundant_rank' + str(rank) + '.dat', 'w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in ser_redundant_name:
                        f.write(" {0:.20e} ".format(redundant_field[item].asnumpy()[i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_velocity_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in velocity_field_name:
                        f.write(" {0:.20e} ".format(velocity_field[item].asnumpy()[i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_ser_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in mixT_name:
                        f.write(" {0:.20e} ".format(ser_data[item][i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_ref_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in mixT_name:
                        f.write(" {0:.20e} ".format(ref_data[item][i, k]))
                    f.write("\n")

        if tendency_serialization:
            with open(base_dir + 'analysis_ref_tend_rank' + str(rank) + '.dat', 'w') as f:
                for i in range(cell_size):
                    for k in range(k_size):
                        f.write("{0:7d} {1:7d}".format(i, k))
                        for item in ser_tend_name:
                            f.write(" {0:.20e} ".format(tend_data[item][i, k]))
                        f.write("\n")

    # measure differences
    for item in mixT_name:
        print(item)
        # just realized that sometimes the diagnostic variables such as qv tendency output from the
        # _graupel_q_tendency does not pass rtol=1.e-12 even though rtol=1.e-12 can be satisfied
        # for the main graupel scan, why?
        assert(np.allclose(predict_field[item].asnumpy(),ref_data[item], rtol=1e-10, atol=1e-16))

