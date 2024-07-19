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

from icon4py.model.atmosphere.physics_schemes.single_moment_six_class_microphysics import gscp_graupel_Ong as graupel
from typing import Final
from icon4py.model.common.dimension import CellDim, KDim
import gt4py.next as gtx
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics, tracer_state as tracers
from icon4py.model.common.grid import vertical as v_grid, horizontal as h_grid
from icon4py.model.common.test_utils.helpers import dallclose

@pytest.mark.parametrize(
    "experiment, model_top_height,, damping_height, stretch_factor, date",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "48"),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "52"),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "56"),
    ],
)
def test_graupel(
    experiment,
    model_top_height,
    damping_height,
    stretch_factor,
    date,
    data_provider,
    grid_savepoint,
    metrics_savepoint,
    icon_grid,
    lowest_layer_thickness,
):

    backend = run_gtfn

    lpres_pri = True  # TODO: may need to be read from serialized data, default is True. We now manually set to True for debugging
    ldass_lhn = True  # TODO: may need to be read from serialized data, default is False. We now manually set to True for debugging
    tendency_serialization = True

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
    )

    init_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_init()
    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_exit(date=date)

    assert vertical_params.kstart_moist == entry_savepoint.kstart_moist() - 1

    tracer_state = tracers.TracerState(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qr=entry_savepoint.qr(),
        qi=entry_savepoint.qi(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=entry_savepoint.rho(),
        vn=None,
        w=None,
        exner=None,
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=entry_savepoint.temperature(),
        pressure=entry_savepoint.pres(),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=init_savepoint.iautocon(),
        ice_stickeff_min=init_savepoint.ceff_min(),
        ice_v0=init_savepoint.vz0i(),
        ice_sedi_density_factor_exp=init_savepoint.icesedi_exp(),
        snow_v0=init_savepoint.v0snow(),
        rain_mu=init_savepoint.mu_rain(),
        rain_n0=init_savepoint.rain_n0_factor(),
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        config=graupel_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
    )

    assert dallclose(graupel.icon_graupel_params.qmin, init_savepoint.qmin())
    assert dallclose(graupel.icon_graupel_params.eps, init_savepoint.eps())
    assert dallclose(graupel.icon_graupel_params.snow_m0, init_savepoint.ams())
    assert dallclose(graupel_microphysics.ccs[0], init_savepoint.ccsrim(), atol=1.e-8)
    assert dallclose(graupel_microphysics.ccs[1], init_savepoint.ccsagg(), atol=1.e-8)
    assert dallclose(graupel.icon_graupel_params.ccsaxp, init_savepoint.ccsaxp())
    assert dallclose(graupel.icon_graupel_params.ccsdep, init_savepoint.ccsdep(), atol=1.e-7)
    assert dallclose(graupel.icon_graupel_params.ccsdxp, init_savepoint.ccsdxp())
    assert dallclose(graupel_microphysics.ccs[2], init_savepoint.ccsvel(), atol=1.e-8)
    assert dallclose(graupel.icon_graupel_params.ccsvxp, init_savepoint.ccsvxp())
    assert dallclose(graupel.icon_graupel_params.ccslam, init_savepoint.ccslam(), atol=1.e-10)
    assert dallclose(graupel.icon_graupel_params.ccslxp, init_savepoint.ccslxp())
    assert dallclose(graupel.icon_graupel_params.ccshi1, init_savepoint.ccshi1())
    assert dallclose(graupel.icon_graupel_params.ccdvtp, init_savepoint.ccdvtp())
    assert dallclose(graupel.icon_graupel_params.ccidep, init_savepoint.ccidep())
    assert dallclose(graupel_microphysics.rain_vel_coef[2], init_savepoint.cevxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[3], init_savepoint.cev(), atol=1.e-10)
    assert dallclose(graupel_microphysics.rain_vel_coef[4], init_savepoint.bevxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[5], init_savepoint.bev())
    assert dallclose(graupel_microphysics.rain_vel_coef[0], init_savepoint.vzxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[1], init_savepoint.vz0r(), atol=1.e-10)


    # graupel_microphysics.run()
    '''

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

    '''
