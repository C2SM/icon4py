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

import pytest

from icon4py.model.atmosphere.physics.microphysics import single_moment_six_class_gscp_graupel as graupel
from icon4py.model.atmosphere.physics.microphysics import saturation_adjustment
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics, tracer_state as tracers
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils.helpers import dallclose

@pytest.mark.parametrize(
    "experiment, model_top_height,, damping_height, stretch_factor, date",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "48"),
        #(dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "52"),
        #(dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "56"),
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

    dtime = entry_savepoint.dt_microphysics()

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

    saturation_adjustment_config = saturation_adjustment.SaturationAdjustmentConfig()

    print(init_savepoint.iautocon())
    print(init_savepoint.ceff_min())
    print(init_savepoint.vz0i())
    print(init_savepoint.icesedi_exp())
    print(init_savepoint.v0snow())
    print(init_savepoint.mu_rain())
    print(init_savepoint.rain_n0_factor())
    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        do_saturation_adjustment=False,
        liquid_autoconversion_option=init_savepoint.iautocon(),
        ice_stickeff_min=init_savepoint.ceff_min(),
        ice_v0=init_savepoint.vz0i(),
        ice_sedi_density_factor_exp=init_savepoint.icesedi_exp(),
        snow_v0=init_savepoint.v0snow(),
        rain_mu=init_savepoint.mu_rain(),
        rain_n0=init_savepoint.rain_n0_factor(),
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        saturation_adjust_config=saturation_adjustment_config,
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


    from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
    from icon4py.model.common.type_alias import vpfloat, wpfloat
    from icon4py.model.common.dimension import CellDim, KDim
    from icon4py.model.common.grid import horizontal as h_grid
    import gt4py.next as gtx
    temperature_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qv_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qc_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qi_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qr_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qs_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qg_ = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    dist_cldtop = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    rho_kup = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    crho1o2_kup = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    crhofac_qi_kup = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    snow_sed0_kup = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    qvsw_kup = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=vpfloat)
    k_lev = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid, dtype=gtx.int32)
    start_cell_nudging = icon_grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.nudging(CellDim)
    )

    end_cell_local = icon_grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.local(CellDim))
    backend = run_gtfn_cached


    print(icon_grid.num_cells)
    print(start_cell_nudging)
    print(icon_grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.interior(CellDim)
    ))
    print(end_cell_local)
    print(vertical_params.kstart_moist)
    print(icon_grid.num_levels)
    print(grid_savepoint.cells_start_index())
    print(grid_savepoint.cells_end_index())
    print(grid_savepoint.edge_start_index())
    print(grid_savepoint.edge_end_index())
    print(icon_grid.get_end_index(
        CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
    ))
    #for i in range(len(grid_savepoint.global_index(CellDim))):
    #    print(i, grid_savepoint.global_index(CellDim)[i])
    print(grid_savepoint.global_index(CellDim))

    graupel.experiment_icon_graupel(
        vertical_params.kstart_moist,
        gtx.int32(icon_grid.num_levels) - gtx.int32(1),
        graupel_microphysics.config.liquid_autoconversion_option,
        graupel_microphysics.config.snow_intercept_option,
        graupel_microphysics.config.rain_freezing_option,
        graupel_microphysics.config.ice_concentration_option,
        graupel_microphysics.config.do_ice_sedimentation,
        graupel_microphysics.config.ice_autocon_sticking_efficiency_option,
        graupel_microphysics.config.do_reduced_icedeposition,
        graupel_microphysics.config.is_isochoric,
        graupel_microphysics.config.use_constant_water_heat_capacity,
        graupel_microphysics.config.ice_stickeff_min,
        graupel_microphysics.config.ice_v0,
        graupel_microphysics.config.ice_sedi_density_factor_exp,
        graupel_microphysics.config.snow_v0,
        *graupel_microphysics.ccs,
        graupel_microphysics.nimax,
        graupel_microphysics.nimix,
        *graupel_microphysics.rain_vel_coef,
        *graupel_microphysics.sed_dens_factor_coef,
        dtime,
        graupel_microphysics.metric_state.ddqz_z_full,
        diagnostic_state.temperature,
        diagnostic_state.pressure,
        prognostic_state.rho,
        tracer_state.qv,
        tracer_state.qc,
        tracer_state.qi,
        tracer_state.qr,
        tracer_state.qs,
        tracer_state.qg,
        graupel_microphysics.qnc,
        temperature_,
        qv_,
        qc_,
        qi_,
        qr_,
        qs_,
        qg_,
        graupel_microphysics.rhoqrv_old_kup,
        graupel_microphysics.rhoqsv_old_kup,
        graupel_microphysics.rhoqgv_old_kup,
        graupel_microphysics.rhoqiv_old_kup,
        graupel_microphysics.vnew_r,
        graupel_microphysics.vnew_s,
        graupel_microphysics.vnew_g,
        graupel_microphysics.vnew_i,
        dist_cldtop,
        rho_kup,
        crho1o2_kup,
        crhofac_qi_kup,
        snow_sed0_kup,
        qvsw_kup,
        k_lev,
        #start_cell_nudging,
        #end_cell_local,
        gtx.int32(0),
        gtx.int32(5),
        #start_cell_nudging + gtx.int32(1),
        gtx.int32(vertical_params.kstart_moist),
        gtx.int32(vertical_params.kstart_moist+1),
        offset_provider={},
    )


    '''
    graupel_microphysics.run(
        dtime,
        prognostic_state,
        diagnostic_state,
        tracer_state,
    )


    new_temperature = entry_savepoint.temperature().ndarray + graupel_microphysics.temperature_tendency.ndarray * dtime
    new_qv = entry_savepoint.qv().ndarray + graupel_microphysics.qv_tendency.ndarray * dtime
    new_qc = entry_savepoint.qc().ndarray + graupel_microphysics.qc_tendency.ndarray * dtime
    new_qr = entry_savepoint.qr().ndarray + graupel_microphysics.qr_tendency.ndarray * dtime
    new_qi = entry_savepoint.qi().ndarray + graupel_microphysics.qi_tendency.ndarray * dtime
    new_qs = entry_savepoint.qs().ndarray + graupel_microphysics.qs_tendency.ndarray * dtime
    new_qg = entry_savepoint.qg().ndarray + graupel_microphysics.qg_tendency.ndarray * dtime
    # measure differences
    assert dallclose(new_temperature, exit_savepoint.temperature().ndarray)
    assert dallclose(new_qv, exit_savepoint.qv().ndarray)
    assert dallclose(new_qc, exit_savepoint.qc().ndarray)
    assert dallclose(new_qr, exit_savepoint.qr().ndarray)
    assert dallclose(new_qi, exit_savepoint.qi().ndarray)
    assert dallclose(new_qs, exit_savepoint.qs().ndarray)
    assert dallclose(new_qg, exit_savepoint.qg().ndarray)
    assert dallclose(graupel_microphysics.temperature_tendency.ndarray, exit_savepoint.ddt_tend_t().ndarray)
    assert dallclose(graupel_microphysics.qv_tendency.ndarray, exit_savepoint.ddt_tend_qv().ndarray)
    assert dallclose(graupel_microphysics.qc_tendency.ndarray, exit_savepoint.ddt_tend_qc().ndarray)
    assert dallclose(graupel_microphysics.qr_tendency.ndarray, exit_savepoint.ddt_tend_qr().ndarray)
    assert dallclose(graupel_microphysics.qi_tendency.ndarray, exit_savepoint.ddt_tend_qi().ndarray)
    assert dallclose(graupel_microphysics.qs_tendency.ndarray, exit_savepoint.ddt_tend_qs().ndarray)

    assert dallclose(
        graupel_microphysics.ice_precipitation_flux.ndarray[:,icon_grid.num_levels-1],
        exit_savepoint.ice_flux().ndarray
    )
    assert dallclose(
        graupel_microphysics.rain_precipitation_flux.ndarray[:, icon_grid.num_levels - 1],
        exit_savepoint.rain_flux().ndarray
    )
    assert dallclose(
        graupel_microphysics.snow_precipitation_flux.ndarray[:, icon_grid.num_levels - 1],
        exit_savepoint.snow_flux().ndarray
    )
    assert dallclose(
        graupel_microphysics.graupel_precipitation_flux.ndarray[:, icon_grid.num_levels - 1],
        exit_savepoint.graupel_flux().ndarray
    )


    print("Max predict-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(predict_field[item].asnumpy() - ref_data[item]).max(), diff_data[item].max())

    print("Max init-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(ser_field[item].asnumpy() - ref_data[item]).max())


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

    '''
