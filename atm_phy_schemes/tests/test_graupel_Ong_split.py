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
from icon4py.atm_phy_schemes.gscp_graupel_Ong_allinOne_split import GraupelGlobalConstants, GraupelFunctionConstants
from typing import Final
from icon4py.common.dimension import CellDim, KDim
#from icon4py.testutils.utils import to_icon4py_field, zero_field
from gt4py.next.iterator.embedded import index_field, np_as_located_field
#from icon4py.testutils.utils_serialbox import bcolors, field_test

import serialbox as ser


def test_graupel_Ong_serialized_data():

    cell_size = 0
    k_size = 0

    ldass_lhn = False # TODO: to be read from serialized data, default is False

    debug = False
    mpi_ranks = np.arange(0, 10, dtype=int)
    initial_date = "2008-09-01T00:00:00.000"
    dates = ("2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000")
    Nblocks = 80 # 121
    rank = 5
    blocks = tuple(i+1 for i in range(Nblocks))
    print(dates)
    print(blocks)

    base_dir = './Data/nh_wk_rerun3/data_dir/'

    serializer = ser.Serializer(ser.OpenModeKind.Read, base_dir, "wk__rank"+str(rank))
    savePoints = serializer.savepoint_list()

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
        "ser_graupel_nvec",
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
        "int32",
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

    # set date
    date_index = 0

    # construct constant data dictionary
    const_data = {}
    for item in ser_const_name:
        const_data[item] = None

    # construct tuning constant data dictionary
    tune_data = {}
    for item in ser_tune_name:
        tune_data[item] = None

    # construct gscp constant data dictionary
    gscp_data = {}
    for item in ser_gscp_data_name:
        gscp_data[item] = None

    # construct serialized data dictionary
    ser_data = {}
    ref_data = {}
    for item in ser_field_name:
        ser_data[item] = None
        ref_data[item] = None

    # read constants
    for item_no, item in enumerate(ser_const_name):
        entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[0]]["date"][dates[date_index]]
        #if (item == "ser_graupel_qi"):
            # not sure why only qi is a float not a list
        #    const_data[item] = serializer.read(item, entrance_savePoint)
        #else:
        const_data[item] = serializer.read(item, entrance_savePoint)[0]
        #if (item == "ser_graupel_istart" or item == "ser_graupel_iend" or item == "ser_graupel_nvec"):
        #    print(item, ": ", const_data[item])
        #if (item == "ser_graupel_kstart_moist"):
        #    print(item, " at block ",0, ": ", const_data[item])
        for i in range(Nblocks - 1):
            entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[i + 1]]["date"][dates[date_index]]
            if (item != "ser_graupel_istart" and item != "ser_graupel_iend" and item != "ser_graupel_nvec"):
                #if (item == "ser_graupel_qi"):
                    # not sure why only qi is a float not a list
                #    if (const_data[item] != serializer.read(item, entrance_savePoint)):
                #        print(const_data[item], serializer.read(item, entrance_savePoint), ser_const_type[item_no])
                #        sys.exit(0)
                #else:
                if (const_data[item] != serializer.read(item, entrance_savePoint)[0]):
                    print(const_data[item], serializer.read(item, entrance_savePoint)[0], ser_const_type[item_no])
                    sys.exit(0)
            #else:
            #    print(item, ": ", const_data[item])
            #if (item == "ser_graupel_kstart_moist"):
            #    print(item, " at block ", i+1, ": ", serializer.read(item, entrance_savePoint))

    for item in ser_tune_name:
        init_savePoint = serializer.savepoint["init-graupel"]["serial_state"][1]["date"][initial_date]
        tune_data[item] = serializer.read(item, init_savePoint)[0]

    for item in ser_gscp_data_name:
        init_savePoint = serializer.savepoint["init-graupel"]["serial_state"][1]["date"][initial_date]
        gscp_data[item] = serializer.read(item, init_savePoint)[0]

    # read serialized input and reference data
    for item in ser_field_name:
        entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[0]]["date"][dates[date_index]]
        exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[0]]["date"][dates[date_index]]
        ser_data[item] = serializer.read(item, entrance_savePoint)
        ref_data[item] = serializer.read(item, exit_savePoint)
        for i in range(Nblocks-1):
            entrance_savePoint = serializer.savepoint["call-graupel-entrance"]["serial_state"][0]["block_index"][blocks[i+1]]["date"][dates[date_index]]
            exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[i + 1]]["date"][dates[date_index]]
            ser_data[item] = np.concatenate((ser_data[item],serializer.read(item, entrance_savePoint)),axis=0)
            ref_data[item] = np.concatenate((ref_data[item], serializer.read(item, exit_savePoint)), axis=0)

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

    print("------------------   GSCP DATA   ----------------------")

    print("zceff_min              : ", gscp_data["ser_init_graupel_zceff_min"], check_graupel_const.GrConst_ceff_min)
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
    if (const_data["ser_graupel_qc0"] != check_graupel_funcConst.GrFuncConst_qc0):
        print("qc0 is NOT equal. Please check.")
        sys.exit()
    if (const_data["ser_graupel_qi0"] != check_graupel_funcConst.GrFuncConst_qi0):
        print("qi0 is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_tune_zceff_min"] != check_graupel_const.GrConst_ceff_min):
        print("tune_zceff_min is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_tune_v0snow"] != check_graupel_const.GrConst_v0snow):
        print("tune_v0snow is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_tune_zvz0i"] != check_graupel_const.GrConst_vz0i):
        print("tune_zvz0i is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_tune_icesedi_exp"] != check_graupel_const.GrConst_icesedi_exp):
        print("tune_icesedi_exp is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_tune_mu_rain"] != check_graupel_const.GrConst_mu_rain):
        print("tune_mu_rain is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_tune_rain_n0_factor"] != check_graupel_const.GrConst_rain_n0_factor):
        print("tune_rain_n0_factor is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_iautocon"] != check_graupel_const.GrConst_iautocon):
        print("iautocon is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_isnow_n0temp"] != check_graupel_const.GrConst_isnow_n0temp):
        print("isnow_n0temp is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_zqmin"] != check_graupel_const.GrConst_qmin):
        print("zqmin is NOT equal. Please check.")
        sys.exit()
    if (tune_data["ser_init_graupel_zeps"] != check_graupel_const.GrConst_eps):
        print("zeps is NOT equal. Please check.")
        sys.exit()

    # if ldass_lhn is not turned on, the serialized qrsflux has dimension (1)
    if (not ldass_lhn):
        ser_data["ser_graupel_qrsflux"] = np.zeros((cell_size, k_size), dtype=float64)
        ref_data["ser_graupel_qrsflux"] = np.zeros((cell_size, k_size), dtype=float64)

    # qnc is a constant in Fortran, we transform it into (cell_size, k_size) dimension
    ser_data["ser_graupel_qnc"] = np.full((cell_size, k_size), fill_value=ser_data["ser_graupel_qnc"][0], dtype=float64)

    # expand the 2D precipitation fluxes (prr, pri, prs, prg) to 3D arrays
    for item in ("ser_graupel_prr_gsp", "ser_graupel_prs_gsp", "ser_graupel_prg_gsp", "ser_graupel_pri_gsp"):
        ser_data[item] = np.expand_dims(ser_data[item], axis=0)
        ref_data[item] = np.expand_dims(ref_data[item], axis=0)
        ser_data[item] = np.transpose(np.pad(ser_data[item], [(0, k_size-1), (0, 0)], mode='constant', constant_values=0))
        ref_data[item] = np.transpose(np.pad(ref_data[item], [(0, k_size-1), (0, 0)], mode='constant', constant_values=0))

    # checking shape
    for item in ser_field_name:
        #if (item != "ser_graupel_prr_gsp" and item != "ser_graupel_prs_gsp" and item != "ser_graupel_prg_gsp" and item != "ser_graupel_pri_gsp"):
        if ((cell_size, k_size) != ser_data[item].shape):
            print("The array size is not fixed. The shape of temperature field is ", cell_size, k_size, ", while the shape of ", item, " is ", ser_data[item].shape, "Please check.")
            sys.exit()

    # transform serialized input and predicted arrays into GT4py Fields and construct predicted output arrays
    ser_field = {}
    predict_field = {}
    velocity_field = {}
    for item in ser_field_name:
        ser_field[item] = np_as_located_field(CellDim, KDim)(np.array(ser_data[item], dtype=float64))
        predict_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size,k_size), dtype=float64))
    for item in velocity_field_name:
        velocity_field[item] = np_as_located_field(CellDim, KDim)(np.zeros((cell_size, k_size), dtype=float64))

    # run graupel

    _graupel(
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

    print("Max predict-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(predict_field[item].array() - ref_data[item]).max())

    print("Max init-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(ser_field[item].array() - ref_data[item]).max())

    print("Max init:")
    for item in mixT_name:
        print(item, ": ", np.abs(ser_data[item]).max())

    print("Max ref:")
    for item in mixT_name:
        print(item, ": ", np.abs(ref_data[item]).max())

    print("Max predict:")
    for item in mixT_name:
        print(item, ": ", predict_field[item].array().max())

    print("Max abs predict:")
    for item in mixT_name:
        print(item, ": ", np.abs(predict_field[item].array()).max())

    print("Max init-ref total difference (checking conservation):")
    print("qv: ", np.abs(
        ser_field["ser_graupel_qv"].array() - ref_data["ser_graupel_qv"] +
        ser_field["ser_graupel_qc"].array() - ref_data["ser_graupel_qc"] +
        ser_field["ser_graupel_qi"].array() - ref_data["ser_graupel_qi"] +
        ser_field["ser_graupel_qr"].array() - ref_data["ser_graupel_qr"] +
        ser_field["ser_graupel_qs"].array() - ref_data["ser_graupel_qs"] +
        ser_field["ser_graupel_qg"].array() - ref_data["ser_graupel_qg"]
    ).max())

    #for i in range(cell_size):
    #    for k in range(k_size):
    #        print(i, k, ": ", qv_new.array()[i, k], qc_new.array()[i, k], qi_new.array()[i, k], qr_new.array()[i, k],qs_new.array()[i, k], qg_new.array()[i, k])
    #    print()

    with open(base_dir+'analysis_predict_rank'+str(rank)+'.dat','w') as f:
        for i in range(cell_size):
            for k in range(k_size):
                for item in mixT_name:
                    f.write(" {0:.20e} ".format(predict_field[item].array()[i,k]))
                f.write("\n")

    with open(base_dir+'analysis_velocity_rank'+str(rank)+'.dat','w') as f:
        for i in range(cell_size):
            for k in range(k_size):
                for item in velocity_field_name:
                    f.write(" {0:.20e} ".format(velocity_field[item].array()[i, k]))
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


    for item in mixT_name:
        try:
            assert np.allclose(predict_field[item].array(),ref_data[item])
        except Exception as e:
            print(type(e), e.args)
            diff_data = np.abs(predict_field[item].array() - ref_data[item]) / ref_data[item]
            diff_data = np.where(ref_data[item] <= 1.e-20, predict_field[item].array(), diff_data)
            for i in range(cell_size):
                for k in range(k_size):
                    if (np.abs(diff_data[i, k]) > 9.99e-15 and np.abs(diff_data[i, k]) < 9.99e-9):
                        print("diff ref predict data: ", i, k, diff_data[item][i, k], ref_data[item][i, k], predict_field[item].array()[i, k])
