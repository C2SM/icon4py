# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx
import pytest
from icon4py.model.common.decomposition import definitions
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import (
    dycore_states,
    dycore_utils,
    solve_nonhydro as solve_nh,
)
from icon4py.model.common.interpolation import (
    interpolation_attributes,
    interpolation_factory,
)
from icon4py.model.common.metrics import (
    metrics_attributes,
    metrics_factory,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    test_utils,
)
from icon4py.model.common.grid import geometry as grid_geometry
from icon4py.model.common.grid import geometry_attributes as geometry_meta
from .. import utils
from ..fixtures import *  # noqa: F403

def construct_dummy_decomposition_info(grid, backend) -> definitions.DecompositionInfo:
    """A public helper function to construct a dummy decomposition info object for test cases
    refactored from grid_utils.py"""

    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)

    def _add_dimension(dim: gtx.Dimension):
        indices = data_alloc.index_field(grid, dim, backend=backend)
        owner_mask = xp.ones((grid.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

    decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
    _add_dimension(dims.EdgeDim)
    _add_dimension(dims.VertexDim)
    _add_dimension(dims.CellDim)

    return decomposition_info

# TODO (Yilu): there is the duplication of the vertical_grid_params with the diffusion, probabaly we can move it to other places
@pytest.fixture
def vertical_grid_params(
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
):
    """Group vertical grid configuration parameters into a dictionary."""
    return {
        "lowest_layer_thickness": lowest_layer_thickness,
        "model_top_height": model_top_height,
        "stretch_factor": stretch_factor,
        "damping_height": damping_height,
    }

@pytest.fixture
def metrics_factory_params(
    rayleigh_coeff,
    exner_expol,
    vwind_offctr,
    rayleigh_type,
):
    """Group rayleigh damping configuration parameters into a dictionary."""
    return {
        "rayleigh_coeff": rayleigh_coeff,
        "exner_expol": exner_expol,
        "vwind_offctr": vwind_offctr,
        "rayleigh_type": rayleigh_type,
    }

@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            dt_utils.GLOBAL_EXPERIMENT,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_run_solve_nonhydro_benchmark(
    grid_manager,
    vertical_grid_params,
    metrics_factory_params,
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    at_initial_timestep,
    *,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)

    grid = grid_manager.grid

    # TODO(Yilu): config (do we need to specify the config according to different config?)
    config = solve_nh.NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=24,
        iau_wgt_dyn=1.0,
        fourth_order_divdamp_factor=0.004,
        max_nudging_coefficient=0.375,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config)

    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=vertical_grid_params["lowest_layer_thickness"],
        model_top_height=vertical_grid_params["model_top_height"],
        stretch_factor=vertical_grid_params["stretch_factor"],
        rayleigh_damping_height=vertical_grid_params["damping_height"],
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
        _min_index_flat_horizontal_grad_pressure=sp.nflat_gradp(), # TODO (Yilu)
    )

    dtime = sp.get_metadata("dtime").get("dtime") # TODO (Yilu): set dtime directly
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        mass_flx_me=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
    )

    diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=0.0,
        theta_v_at_cells_on_half_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        perturbed_exner_at_cells_on_model_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        rho_at_cells_on_half_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        exner_tendency_due_to_slow_physics=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        grf_tend_rho=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        grf_tend_thv=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        grf_tend_w=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        mass_flux_at_edges_on_model_levels=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        normal_wind_tendency_due_to_slow_physics_process=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        grf_tend_vn=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ), # TODO (Yilu)
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(current_index), init_savepoint.ddt_w_adv_pc(next_index)
        ), # TODO (Yilu)
        tangential_wind=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        vn_on_half_levels=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        contravariant_correction_at_cells_on_half_levels=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        rho_iau_increment=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        normal_wind_iau_increment=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        exner_iau_increment=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
        exner_dynamical_increment=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, backend=backend
        ),
    )

    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.gemetry_fields

    geometry_field_source = grid_geometry.GridGeometry(
        grid = grid,
        decomposition_info = construct_dummy_decomposition_info(grid, backend),
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs,
    )

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=construct_dummy_decomposition_info(grid, backend),
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=grid,
        vertical_grid=vertical_params,
        decomposition_info=construct_dummy_decomposition_info(grid, backend),
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c), # TODO (Yilu): implement the topography
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=metrics_factory_params["rayleigh_type"],
        rayleigh_coeff=metrics_factory_params["rayleigh_coeff"],
        exner_expol=metrics_factory_params["exner_expol"],
        vwind_offctr=metrics_factory_params["vwind_offctr"],
    )

    interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_field_source.get(interpolation_attributes.C_LIN_E),
        c_intp=interpolation_field_source.get(interpolation_attributes.),
        e_flx_avg=interpolation_field_source.get(interpolation_attributes.E_FLX_AVG),
        geofac_grdiv=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRDIV),
        geofac_rot=interpolation_field_source.get(interpolation_attributes.GEOFAC_ROT),
        pos_on_tplane_e_1=interpolation_field_source.get(interpolation_attributes.POS_ON_TPLANE_E_X),
        pos_on_tplane_e_2=interpolation_field_source.get(interpolation_attributes.POS_ON_TPLANE_E_Y),
        rbf_vec_coeff_e=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_E),
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )

    metric_state_nonhydro = dycore_states.MetricStateNonHydro(
        bdy_halo_c=metrics_field_source.get(metrics_attributes.BDY_HALO_C),
        mask_prog_halo_c=metrics_field_source.get(metrics_attributes.MASK_PROG_HALO_C),
        rayleigh_w=metrics_field_source.get(metrics_attributes.RAYLEIGH_W),
        time_extrapolation_parameter_for_exner=metrics_field_source.get(metrics_attributes.), # TODO (Yilu)
        reference_exner_at_cells_on_model_levels=metrics_field_source.get(metrics_attributes.EXNER_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        wgtfacq_c=metrics_field_source.get(metrics_attributes.WGTFACQ_C),
        inv_ddqz_z_full=metrics_field_source.get(metrics_attributes.INV_DDQZ_Z_FULL),
        reference_rho_at_cells_on_model_levels=metrics_field_source.get(metrics_attributes.), # TODO (Yilu)
        reference_theta_at_cells_on_model_levels=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        exner_w_explicit_weight_parameter=metrics_field_source.get(metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_field_source.get(metrics_attributes.), # TODO (Yilu)
        ddqz_z_half=metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF),
        reference_theta_at_cells_on_half_levels=metrics_field_source.get(metrics_attributes.THETA_REF_MC),# TODO (Yilu)
        d2dexdz2_fac1_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        d2dexdz2_fac2_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        reference_rho_at_edges_on_model_levels=,# TODO (Yilu)
        reference_theta_at_edges_on_model_levels=,# TODO (Yilu)
        ddxn_z_full=metrics_field_source.get(metrics_attributes.DDXN_Z_FULL),
        zdiff_gradp=metrics_field_source.get(metrics_attributes.ZDIFF_GRADP),
        vertoffset_gradp=metrics_field_source.get(metrics_attributes.),# TODO (Yilu)
        pg_edgeidx_dsl=metrics_field_source.get(metrics_attributes.PG_EDGEIDX_DSL),
        pg_exdist=metrics_field_source.get(metrics_attributes.PG_EDGEDIST_DSL),
        ddqz_z_full_e=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL_E),
        ddxt_z_full=metrics_field_source.get(metrics_attributes.DDXT_Z_FULL),
        wgtfac_e=metrics_field_source.get(metrics_attributes.WGTFAC_E),
        wgtfacq_e=metrics_field_source.get(metrics_attributes.WGTFACQ_E),
        exner_w_implicit_weight_parameter=metrics_field_source.get(metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER),
        horizontal_mask_for_3d_divdamp=metrics_field_source.get(metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP),
        scaling_factor_for_3d_divdamp=metrics_field_source.get(metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP),
        coeff1_dwdz=metrics_field_source.get(metrics_attributes.COEFF1_DWDZ),
        coeff2_dwdz=metrics_field_source.get(metrics_attributes.COEFF2_DWDZ),
        coeff_gradekin=metrics_field_source.get(metrics_attributes.COEFF_GRADEKIN),
    )

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    prognostic_states = utils.create_prognostic_states(sp)

    second_order_divdamp_factor = sp.divdamp_fac_o2()
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
        at_first_substep=substep_init == 1,
        at_last_substep=substep_init == ndyn_substeps,
    )


@pytest.mark.datatest
def test_non_hydrostatic_params(savepoint_nonhydro_init):
    config = solve_nh.NonHydrostaticConfig()
    params = solve_nh.NonHydrostaticParams(config)

    assert params.advection_implicit_weight_parameter == savepoint_nonhydro_init.wgt_nnew_vel()
    assert params.advection_explicit_weight_parameter == savepoint_nonhydro_init.wgt_nnow_vel()
    assert params.rhotheta_implicit_weight_parameter == savepoint_nonhydro_init.wgt_nnew_rth()
    assert params.rhotheta_explicit_weight_parameter == savepoint_nonhydro_init.wgt_nnow_rth()

