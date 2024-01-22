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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def test_graupel_Ong_analysis():

    data_name = (
        "temperature",
        "qv",
        "qc",
        "qi",
        "qr",
        "qs",
        "qg",
        "ddt_tend_t",
        "ddt_tend_qv",
        "ddt_tend_qc",
        "ddt_tend_qi",
        "ddt_tend_qr",
        "ddt_tend_qs",
        "prr_gsp",
        "prs_gsp",
        "prg_gsp",
        "pri_gsp",
        "qrsflux"
    )

    redundant_name = (
        "rho_kup",
        "Crho1o2_kup",
        "Crhofac_qi_kup",
        "Cvz0s_kup",
        "qvsw_kup",
        "clddist",
        "klev"
    )

    tend_name = (
        "szdep_v2i",
        "szsub_v2i",
        "snucl_v2i",
        "scfrz_c2i",
        "simlt_i2c",
        "sicri_i2g",
        "sidep_v2i",
        "sdaut_i2s",
        "saggs_i2s",
        "saggg_i2g",
        "siaut_i2s",
        "ssmlt_s2r",
        "srims_c2s",
        "ssdep_v2s",
        "scosg_s2g",
        "sgmlt_g2r",
        "srcri_r2g",
        "sgdep_v2g",
        "srfrz_r2g",
        "srimg_c2g"
    )
    '''
    tend_name = (
        "Sidep_v2i",
        "Szdep_v2i",
        "Szsub_v2i",
        "clddist"
    )
    tend_name = (
        "Szdep_v2i",
        "Szsub_v2i",
        "nin",
        "mi",
        "qvsw",
        "qvsi",
        "eff",
        "n0s",
        "rdep",
        "clddist"
    )
    tend_name = (
        "Snucl_v2i",
        "Scfrz_c2i",
        "Simlt_i2c",
        "Sicri_i2g",
        "Sidep_v2i",
        "Sdaut_i2s",
        "Saggs_i2s",
        "Saggg_i2g",
        "Siaut_i2s"
    )
    tend_name = (
        "Ssmlt_s2r",
        "Srims_c2s",
        "Ssdep_v2s",
        "Scosg_s2g",
        "Sgmlt_g2r",
        "Srcri_r2g",
        "Sgdep_v2g",
        "Srfrz_r2g",
        "Srimg_c2g"
    )
    '''

    velocity_name = (
        "rhoqrV_old_kup",
        "rhoqsV_old_kup",
        "rhoqgV_old_kup",
        "rhoqiV_old_kup",
        "Vnew_r",
        "Vnew_s",
        "Vnew_g",
        "Vnew_i"
    )

    tendency_serialization = False

    #master_dir = '/home/ong/Data/nh_wk_rerun_complete/data_dir/'
    script_dir = os.path.dirname(__file__)
    master_dir = script_dir+'/serialbox/data_dir/bug_check/' #conservation_zeroV_modten/

    rank = 5
    k_size = 64
    line_number = 0
    with open(master_dir+'analysis_predict_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_number += 1
    cell_size = int(line_number/k_size)
    print("cell and k dimension: ", cell_size, k_size)

    # construct serialized data dictionary
    predict_data = {}
    velocity_data = {}
    redundant_data = {}
    ser_data = {}
    ref_data = {}
    diff_data = {}
    for item in data_name:
        predict_data[item] = np.zeros((cell_size, k_size), dtype=float)
        ser_data[item] = np.zeros((cell_size,k_size),dtype=float)
        ref_data[item] = np.zeros((cell_size,k_size),dtype=float)
        diff_data[item] = np.zeros((cell_size, k_size), dtype=float)
    for item in velocity_name:
        velocity_data[item] = np.zeros((cell_size, k_size), dtype=float)
    for item in tend_name:
        predict_data[item] = np.zeros((cell_size, k_size), dtype=float)
        ref_data[item] = np.zeros((cell_size, k_size), dtype=float)
        diff_data[item] = np.zeros((cell_size, k_size), dtype=float)
    for item in redundant_name:
        redundant_data[item] = np.zeros((cell_size, k_size), dtype=float)

    with open(master_dir+'analysis_predict_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            #item = int(line_number/cell_size/k_size)
            i = int(line_number/k_size)
            k = line_number % k_size
            try:
                for item in range(len(data_name)):
                    predict_data[data_name[item]][i,k] = float(data[2+item])
            except Exception as e:
                print(type(e), e.args)
                print("predict data: ", line_number, item, i, k)
                sys.exit()
            if (i != int(data[0]) or k != int(data[1])):
                print("algorithm is wrong, please check i and k: ", line_number, item, i, k, int(data[0]), int(data[1]))
                sys.exit()
            line_number += 1

    if ( tendency_serialization ):
        with open(master_dir+'analysis_tend_rank' + str(rank) + '.dat', 'r') as f:
            lines = f.readlines()
            line_number = 0
            for line in lines:
                data = line.split()
                #item = int(line_number/cell_size/k_size)
                i = int(line_number/k_size)
                k = line_number % k_size
                try:
                    for item in range(len(tend_name)):
                        if (str(data[2+item]) == 'nan'):
                            predict_data[tend_name[item]][i,k] = 0.0
                        else:
                            predict_data[tend_name[item]][i,k] = float(data[2+item])
                except Exception as e:
                    print(type(e), e.args)
                    print("predict data: ", line_number, item, i, k)
                    sys.exit()
                line_number += 1

    with open(master_dir+'analysis_velocity_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            # item = int(line_number/cell_size/k_size)
            i = int(line_number / k_size)
            k = line_number % k_size
            try:
                for item in range(len(velocity_name)):
                    velocity_data[velocity_name[item]][i,k] = float(data[2+item])
            except Exception as e:
                print(type(e), e.args)
                print("velocity data: ", line_number, item, i, k)
                sys.exit()
            line_number += 1

    with open(master_dir+'analysis_ser_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            # item = int(line_number/cell_size/k_size)
            i = int(line_number / k_size)
            k = line_number % k_size
            try:
                for item in range(len(data_name)):
                    ser_data[data_name[item]][i, k] = float(data[2 + item])
            except Exception as e:
                print(type(e), e.args)
                print("ser data: ", line_number, item, i, k)
                sys.exit()
            line_number += 1

    with open(master_dir+'analysis_ref_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            # item = int(line_number/cell_size/k_size)
            i = int(line_number / k_size)
            k = line_number % k_size
            try:
                for item in range(len(data_name)):
                    ref_data[data_name[item]][i, k] = float(data[2 + item])
            except Exception as e:
                print(type(e), e.args)
                print("ref data: ", line_number, item, i, k)
                sys.exit()
            line_number += 1

    if ( tendency_serialization ):
        with open(master_dir+'analysis_ref_tend_rank' + str(rank) + '.dat', 'r') as f:
            lines = f.readlines()
            line_number = 0
            for line in lines:
                data = line.split()
                # item = int(line_number/cell_size/k_size)
                i = int(line_number / k_size)
                k = line_number % k_size
                try:
                    for item in range(len(tend_name)):
                        if (str(data[2+item]) == 'nan'):
                            ref_data[tend_name[item]][i,k] = 0.0
                        else:
                            ref_data[tend_name[item]][i, k] = float(data[2 + item])
                except Exception as e:
                    print(type(e), e.args)
                    print("ref data: ", line_number, item, i, k)
                    sys.exit()
                line_number += 1

    with open(master_dir+'analysis_redundant_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            # item = int(line_number/cell_size/k_size)
            i = int(line_number / k_size)
            k = line_number % k_size
            try:
                for item in range(len(redundant_name)):
                    redundant_data[redundant_name[item]][i, k] = float(data[2 + item])
            except Exception as e:
                print(type(e), e.args)
                print("ref data: ", line_number, item, i, k)
                sys.exit()
            line_number += 1

    for item in data_name:
        #diff_data[item] = predict_data[item] - ref_data[item]
        diff_data[item] = np.abs(predict_data[item] - ref_data[item]) / ref_data[item]
        diff_data[item] = np.where(np.abs(ref_data[item]) <= 1.e-20 , predict_data[item], diff_data[item])

    if ( tendency_serialization ):
        for item in tend_name:
            #diff_data[item] = predict_data[item] - ref_data[item]
            diff_data[item] = np.abs(predict_data[item] - ref_data[item]) / ref_data[item]
            diff_data[item] = np.where(np.abs(ref_data[item]) <= 1.e-30, predict_data[item], diff_data[item])

    if ( tendency_serialization ):
        print("Max predict-ref tendency difference:")
        for item in tend_name:
            print(item, ": ", np.abs(predict_data[item] - ref_data[item]).max(), np.abs(diff_data[item]).max())

    print("Max predict-ref difference:")
    for item in data_name:
        print(item, ": ", np.abs(predict_data[item] - ref_data[item]).max(), np.abs(diff_data[item]).max())

    print("Max init-ref difference:")
    for item in data_name:
        print(item, ": ", np.abs(ser_data[item] - ref_data[item]).max())

    print("Max init:")
    for item in data_name:
        print(item, ": ", np.abs(ser_data[item]).max())

    print("Max ref:")
    for item in data_name:
        print(item, ": ", np.abs(ref_data[item]).max())
    for item in tend_name:
        print(item, ": ", np.abs(ref_data[item]).max())

    print("Max predict:")
    for item in data_name:
        print(item, ": ", predict_data[item].max())
    for item in tend_name:
        print(item, ": ", predict_data[item].max())

    print("Min predict:")
    for item in data_name:
        print(item, ": ", predict_data[item].min())
    for item in tend_name:
        print(item, ": ", predict_data[item].min())

    print("Max abs predict:")
    for item in data_name:
        print(item, ": ", np.abs(predict_data[item]).max())
    for item in tend_name:
        print(item, ": ", np.abs(predict_data[item]).max())

    print("Max init-ref total difference:")
    print("qv: ", np.abs(
        ser_data["qv"] - ref_data["qv"] +
        ser_data["qc"] - ref_data["qc"] +
        ser_data["qi"] - ref_data["qi"] +
        ser_data["qr"] - ref_data["qr"] +
        ser_data["qs"] - ref_data["qs"] +
        ser_data["qg"] - ref_data["qg"]
    ).max())

    # =====================
    # checking conservation
    # =====================
    rho = np.zeros((cell_size, k_size), dtype=float)
    dz = np.zeros((cell_size, k_size), dtype=float)

    with open(master_dir+'analysis_ref_rho_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            i = int(line_number/k_size)
            k = line_number % k_size
            try:
                rho[i,k] = float(data[2])
            except Exception as e:
                print(type(e), e.args)
                print("predict data: ", line_number, i, k)
                sys.exit()
            if (i != int(data[0]) or k != int(data[1])):
                print("algorithm is wrong, please check i and k: ", line_number, i, k, int(data[0]), int(data[1]))
                sys.exit()
            line_number += 1

    with open(master_dir+'analysis_dz_rank' + str(rank) + '.dat', 'r') as f:
        lines = f.readlines()
        line_number = 0
        for line in lines:
            data = line.split()
            i = int(line_number/k_size)
            k = line_number % k_size
            try:
                dz[i,k] = float(data[2])
            except Exception as e:
                print(type(e), e.args)
                print("predict data: ", line_number, i, k)
                sys.exit()
            if (i != int(data[0]) or k != int(data[1])):
                print("algorithm is wrong, please check i and k: ", line_number, i, k, int(data[0]), int(data[1]))
                sys.exit()
            line_number += 1

    hydro_init = (
        rho * ser_data["qv"] * dz +
        rho * ser_data["qc"] * dz +
        rho * ser_data["qi"] * dz +
        rho * ser_data["qr"] * dz +
        rho * ser_data["qs"] * dz +
        rho * ser_data["qg"] * dz
    )
    hydro_end = (
        rho * predict_data["qv"] * dz +
        rho * predict_data["qc"] * dz +
        rho * predict_data["qi"] * dz +
        rho * predict_data["qr"] * dz +
        rho * predict_data["qs"] * dz +
        rho * predict_data["qg"] * dz
    )

    dt = 4.0 # TODO to be read from data file
    prec = (
        predict_data["prr_gsp"] +
        predict_data["prs_gsp"] +
        predict_data["prg_gsp"] +
        predict_data["pri_gsp"]
    ) * dt

    print("checking conservation: ",hydro_init.sum(), hydro_end.sum(), prec.sum(), (hydro_init - hydro_end).sum(), (hydro_init - hydro_end - prec).sum())

    #predict_data["qv"] = hydro_init - hydro_end

    fig, ax = plt.subplots()

    x = np.arange(0.0,float(cell_size), step=1.0, dtype=float)
    z = np.arange(0.0, float(k_size), step=1.0, dtype=float)
    x_mesh, z_mesh = np.meshgrid(x,z)
    print(predict_data["qi"].shape)
    print(predict_data["qi"].min(), predict_data["qi"].max(), (predict_data["qi"].max() - predict_data["qi"].min())/10.0)

    #diff_clddist = np.abs(redundant_data["clddist"] - ref_data["clddist"]) / ref_data["clddist"]
    #diff_clddist = np.where(ref_data["clddist"] <= 1.e-30, diff_clddist, diff_clddist)

    for i in range(cell_size):
        for k in range(k_size):
            #if (ref_data["qr"][i,k] > 1.e-12):
            #    print("ref data: ",i,k,ref_data["qr"][i,k])
            #if (diff_data["qr"][i, k] > 1.e-20):
            #    print("diff ref predict data: ",i,k,diff_data["qr"][i,k],ref_data["qr"][i,k], abs(predict_data["qr"][i,k] - ref_data["qr"][i,k])/ref_data["qr"][i,k])
            '''
            "Ssmlt_s2r",
            "Srims_c2s",
            "Ssdep_v2s",
            "Scosg_s2g",
            "Sgmlt_g2r",
            "Srcri_r2g",
            "Sgdep_v2g",
            "Srfrz_r2g",
            "Srimg_c2g"
            ser_graupel_srfrz_r2g
            '''
            #temp = "prr_gsp"
            #if (np.abs(diff_data[temp][i, k]) > 9.99e-10):  # and np.abs(diff_data[temp][i, k]) < 9.99e-9
            #    print("diff ref predict data: ", temp, i, k, diff_data[temp][i, k], ref_data[temp][i, k],predict_data[temp][i, k], " - ",abs(predict_data[temp][i, k] - ref_data[temp][i, k]) / ref_data[temp][i, k])
            #if (str(diff_data[temp][i, k]) == 'nan'):
            #    print("diff ref predict data: ", temp, i, k, diff_data[temp][i, k], ref_data[temp][i, k],predict_data[temp][i, k], " - ",abs(predict_data[temp][i, k] - ref_data[temp][i, k]) / ref_data[temp][i, k])

            '''
            if (k > 7):
                if (np.abs(predict_data["temperature"][i,k]) > 1.e20):
                    print(i,k,predict_data["temperature"][i,k])
            else:
                predict_data["temperature"][i, k] = 0.0

            if (np.abs(velocity_data["Vnew_i"][i,k]) > 1.e-20):
                print(i, k, velocity_data["Vnew_i"][i, k])
            '''

            for temp in ("temperature","qv","qc","qi","qr","qs","qg"):
                if (False and np.abs(diff_data[temp][i, k]) > 9.99e-7):  # and np.abs(diff_data[temp][i, k]) < 9.99e-9
                    #print("diff ref predict data: ", temp, i, k, diff_data[temp][i, k], ref_data[temp][i, k] - ser_data[temp][i, k], ref_data[temp][i, k], predict_data[temp][i, k], " - ", abs(predict_data[temp][i, k] - ref_data[temp][i, k]) / ref_data[temp][i, k])
                    print("diff ref predict data: ", temp, i, k, predict_data[temp][i, k] - ser_data[temp][i, k]," - ", ref_data[temp][i, k] - ser_data[temp][i, k] )
                    #if (str(diff_data[temp][i, k]) == 'nan'):
                    #    print("diff ref predict data: ",i,k,diff_data[temp][i,k],ref_data[temp][i,k], predict_data[temp][i,k])
            if (False and np.abs(diff_data[temp][i, k]) > 9.99e-10): #and np.abs(diff_data[temp][i, k]) < 9.99e-9
                for temp in data_name:
                    print("diff ref predict data: ", temp, i, k, diff_data[temp][i,k],ref_data[temp][i,k], predict_data[temp][i,k], " - ", abs(predict_data[temp][i,k] - ref_data[temp][i,k])/ref_data[temp][i,k])
                for temp in tend_name:
                    print("diff ref predict data: ", temp, i, k, diff_data[temp][i,k],ref_data[temp][i,k], predict_data[temp][i,k], " - ", abs(predict_data[temp][i,k] - ref_data[temp][i,k])/ref_data[temp][i,k])
                print()
            if (False and i == 1065 and k == 35):
                for temp in data_name:
                    print("diff ref predict data: ", temp, i, k, diff_data[temp][i,k],ref_data[temp][i,k], predict_data[temp][i,k], " - ", abs(predict_data[temp][i,k] - ref_data[temp][i,k])/ref_data[temp][i,k])
                for temp in tend_name:
                    print("diff ref predict data: ", temp, i, k, diff_data[temp][i,k],ref_data[temp][i,k], predict_data[temp][i,k], " - ", abs(predict_data[temp][i,k] - ref_data[temp][i,k])/ref_data[temp][i,k])

            #if (np.abs(diff_clddist[i, k]) > 9.99e-15):
            #    print("clddist diff ref predict data: ",i,k,diff_clddist[i,k],ref_data["clddist"][i,k], redundant_data["clddist"][i,k], " - ", abs(redundant_data["clddist"][i,k] - ref_data["clddist"][i,k])/ref_data["clddist"][i,k])

    mixT_name = (
        "temperature",
        "qv",
        "qc",
        "qi",
        "qr",
        "qs",
        "qg"
    )
    # measure differences
    for item in mixT_name:
        print(item, np.allclose(predict_data[item], ref_data[item], rtol=1e-12, atol=1e-16))

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    #diff_data["qi"] = diff_data["qi"] + diff_data["qv"]
    plot_type = 1
    scale = 'log'
    item = "qv"
    velocity = "rhoqsV_old_kup"

    #diff_data[item] = np.abs(ref_data[item] - ser_data[item])

    #print("diff: ", diff_data[item].min(), diff_data[item].max(), (diff_data[item].max() - diff_data[item].min()) / 10.0)
    #print("diff (arg): ", diff_data[item].argmin() / k_size, diff_data[item].argmin() % k_size, diff_data[item].argmax() / k_size, diff_data[item].argmax() % k_size)
    #print("predict: ", predict_data[item].min(), predict_data[item].max(), (predict_data[item].max() - predict_data[item].min()) / 10.0)
    #print("ref: ", ref_data[item].min(), ref_data[item].max(), (ref_data[item].max() - ref_data[item].min()) / 10.0)
    #print("ser: ", ser_data[item].min(), ser_data[item].max(), (ser_data[item].max() - ser_data[item].min()) / 10.0)

    bluemap = plt.get_cmap('Blues')
    redmap = plt.get_cmap('Reds')
    jetmap = plt.get_cmap('jet')
    new_map = truncate_colormap(jetmap, 0.0, 1.0)
    if (plot_type == 1):
        boundary = np.arange(predict_data[item].min(), predict_data[item].max(), step=(predict_data[item].max() - predict_data[item].min())/100.0, dtype=float)
    elif (plot_type == 2):
        boundary = np.arange(ref_data[item].min(), ref_data[item].max(),step=(ref_data[item].max() - ref_data[item].min()) / 100.0, dtype=float)
    elif (plot_type == 3):
        boundary = np.arange(diff_data[item].min(), diff_data[item].max(),step=(diff_data[item].max() - diff_data[item].min()) / 100.0, dtype=float)
    elif (plot_type == 4):
        boundary = np.arange(redundant_data[item].min(), redundant_data[item].max(),step=(redundant_data[item].max() - redundant_data[item].min()) / 100.0, dtype=float)
    elif (plot_type == 5):
        boundary = np.arange(velocity_data[velocity].min(), velocity_data[velocity].max(),step=(velocity_data[velocity].max() - velocity_data[velocity].min()) / 10.0, dtype=float)
    norm = colors.BoundaryNorm(boundary, new_map.N, clip=True)
    if (scale == 'log'):
        if (plot_type == 1):
            cpr = ax.imshow(predict_data[item],interpolation='nearest',cmap=new_map,norm=colors.LogNorm(vmin=1.e-17, vmax=predict_data[item].max()))
        elif (plot_type == 2):
            cpr = ax.imshow(ref_data[item], interpolation='nearest', cmap=new_map,norm=colors.LogNorm(vmin=1.e-17, vmax=ref_data[item].max()))
        elif (plot_type == 3):
            cpr = ax.imshow(diff_data[item], interpolation='nearest', cmap=new_map,norm=colors.LogNorm(vmin=1.e-17, vmax=diff_data[item].max()))
        elif (plot_type == 5):
            cpr = ax.imshow(velocity_data[item], interpolation='nearest', cmap=new_map,norm=colors.LogNorm(vmin=1.e-17, vmax=velocity_data[item].max()))
    elif (scale == 'normal'):
        if (plot_type == 1):
            cpr = ax.imshow(predict_data[item],interpolation='nearest',cmap=new_map,norm=norm)
        elif (plot_type == 2):
            cpr = ax.imshow(ref_data[item], interpolation='nearest', cmap=new_map, norm=norm)
        elif (plot_type == 3):
            cpr = ax.imshow(diff_data[item], interpolation='nearest', cmap=new_map, norm=norm)
        elif (plot_type == 4):
            cpr = ax.imshow(redundant_data[item], interpolation='nearest', cmap=new_map, norm=norm)
        elif (plot_type == 5):
            cpr = ax.imshow(velocity_data[velocity], interpolation='nearest', cmap=new_map, norm=norm)
        #cpr = ax.imshow(velocity_data[velocity], interpolation='nearest', cmap=new_map, norm=norm)
    cb = plt.colorbar(cpr)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    plt.show()

test_graupel_Ong_analysis()
