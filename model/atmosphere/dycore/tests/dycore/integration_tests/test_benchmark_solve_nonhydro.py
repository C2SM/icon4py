# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import pytest


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import model_backends
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import factory
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing.fixtures.benchmark import (
    geometry_field_source,
    interpolation_field_source,
    metrics_field_source,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid_manager


@pytest.fixture(scope="module")
def solve_nonhydro(
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> solve_nh.SolveNonhydro:
    allocator = model_backends.get_allocator(backend_like)
    mesh = geometry_field_source.grid

    config = solve_nh.NonHydrostaticConfig(
        divdamp_order=dycore_states.DivergenceDampingOrder.COMBINED,
        fourth_order_divdamp_factor=0.004,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config)

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator=allocator)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
        mean_cell_area=geometry_field_source.get_scalar(geometry_meta.MEAN_CELL_AREA),
    )
    edge_geometry = grid_states.EdgeParams(
        tangent_orientation=geometry_field_source.get(geometry_meta.TANGENT_ORIENTATION),
        inverse_primal_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.EDGE_LENGTH}"
        ),
        inverse_dual_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ),
        inverse_vertex_vertex_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
        ),
        primal_normal_vert=(
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        ),
        dual_normal_vert=(
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_V),
        ),
        primal_normal_cell=(
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_U),
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_V),
        ),
        dual_normal_cell=(
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_U),
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_V),
        ),
        edge_areas=geometry_field_source.get(geometry_meta.EDGE_AREA),
        coriolis_frequency=geometry_field_source.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_center=(
            geometry_field_source.get(geometry_meta.EDGE_LAT),
            geometry_field_source.get(geometry_meta.EDGE_LON),
        ),
        primal_normal=(
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_U),
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_V),
        ),
    )

    interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_field_source.get(interpolation_attributes.C_LIN_E),
        c_intp=interpolation_field_source.get(interpolation_attributes.CELL_AW_VERTS),
        e_flx_avg=interpolation_field_source.get(interpolation_attributes.E_FLX_AVG),
        geofac_grdiv=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRDIV),
        geofac_rot=interpolation_field_source.get(interpolation_attributes.GEOFAC_ROT),
        pos_on_tplane_e_1=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_X
        ),
        pos_on_tplane_e_2=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_Y
        ),
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
        mask_prog_halo_c=metrics_field_source.get(metrics_attributes.MASK_PROG_HALO_C),
        rayleigh_w=metrics_field_source.get(metrics_attributes.RAYLEIGH_W),
        time_extrapolation_parameter_for_exner=metrics_field_source.get(
            metrics_attributes.EXNER_EXFAC
        ),
        reference_exner_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.EXNER_REF_MC
        ),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        wgtfacq_c=metrics_field_source.get(metrics_attributes.WGTFACQ_C),
        inv_ddqz_z_full=metrics_field_source.get(metrics_attributes.INV_DDQZ_Z_FULL),
        reference_rho_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_MC
        ),
        reference_theta_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_MC
        ),
        exner_w_explicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER
        ),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.D_EXNER_DZ_REF_IC
        ),
        ddqz_z_half=metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF),
        reference_theta_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_IC
        ),
        d2dexdz2_fac1_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        d2dexdz2_fac2_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        reference_rho_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_ME
        ),
        reference_theta_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_ME
        ),
        ddxn_z_full=metrics_field_source.get(metrics_attributes.DDXN_Z_FULL),
        zdiff_gradp=metrics_field_source.get(metrics_attributes.ZDIFF_GRADP),
        vertoffset_gradp=metrics_field_source.get(metrics_attributes.VERTOFFSET_GRADP),
        nflat_gradp=metrics_field_source.get_scalar(metrics_attributes.NFLAT_GRADP),  # type: ignore[arg-type]
        pg_exdist=metrics_field_source.get(metrics_attributes.PG_EXDIST_DSL),
        ddqz_z_full_e=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL_E),
        ddxt_z_full=metrics_field_source.get(metrics_attributes.DDXT_Z_FULL),
        wgtfac_e=metrics_field_source.get(metrics_attributes.WGTFAC_E),
        wgtfacq_e=metrics_field_source.get(metrics_attributes.WGTFACQ_E),
        exner_w_implicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER
        ),
        horizontal_mask_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP
        ),
        scaling_factor_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP
        ),
        coeff1_dwdz=metrics_field_source.get(metrics_attributes.COEFF1_DWDZ),
        coeff2_dwdz=metrics_field_source.get(metrics_attributes.COEFF2_DWDZ),
        coeff_gradekin=metrics_field_source.get(metrics_attributes.COEFF_GRADEKIN),
    )

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=mesh,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_grid,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=geometry_field_source.get(geometry_meta.CELL_OWNER_MASK),
        exchange=decomposition.SingleNodeExchange(),
        backend=backend_like,
        max_nudging_coefficient=0.375,
    )

    return solve_nonhydro


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
@pytest.mark.embedded_remap_error
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
def test_benchmark_solve_nonhydro(  # noqa: PLR0917 [too-many-positional-arguments]
    grid_manager: gm.GridManager,
    solve_nonhydro: solve_nh.SolveNonhydro,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    mesh = grid_manager.grid

    dtime = 10.0 if mesh.limited_area else 90.0

    prepare_fluxes_for_advection = True
    ndyn_substeps = 5
    at_initial_timestep = False
    second_order_divdamp_factor = 0.02

    def random_prognostics() -> states.PrognosticState:
        def random(quantity: type[fw.Quantity]) -> fw.Field[Any]:
            return fw.Field(
                quantity, data_alloc.random_field(mesh, *quantity.dims, allocator=allocator)
            )

        return states.PrognosticState(
            rho=random(qty.RhoOnCellK),
            w=random(qty.WOnCellKHalf),
            vn=random(qty.VnOnEdgeK),
            exner=random(qty.ExnerOnCellK),
            theta_v=random(qty.ThetaVOnCellK),
        )

    current, next_ = random_prognostics(), random_prognostics()
    forcing = fw.allocate(states.DycoreForcing, mesh, allocator)
    prep_adv = fw.allocate(states.PrepAdvection, mesh, allocator)
    dycore_diagnostics = states.DycoreDiagnostics.allocate(mesh, allocator)
    inputs = solve_nh.SolveNonhydro.Input(
        rho=current.rho,
        w=current.w,
        vn=current.vn,
        exner=current.exner,
        theta_v=current.theta_v,
        exner_tendency_due_to_slow_physics=forcing.exner_tendency_due_to_slow_physics,
        normal_wind_tendency_due_to_slow_physics_process=forcing.normal_wind_tendency_due_to_slow_physics_process,
        grf_tend_rho=forcing.grf_tend_rho,
        grf_tend_thv=forcing.grf_tend_thv,
        grf_tend_w=forcing.grf_tend_w,
        grf_tend_vn=forcing.grf_tend_vn,
        rho_iau_increment=forcing.rho_iau_increment,
        normal_wind_iau_increment=forcing.normal_wind_iau_increment,
        exner_iau_increment=forcing.exner_iau_increment,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        prepare_fluxes_for_advection=prepare_fluxes_for_advection,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )
    out = solve_nh.SolveNonhydro.Output(
        rho=next_.rho,
        w=next_.w,
        vn=next_.vn,
        exner=next_.exner,
        theta_v=next_.theta_v,
        vn_traj=prep_adv.vn_traj,
        mass_flx_me=prep_adv.mass_flx_me,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=dycore_diagnostics.perturbed_exner_at_cells_on_model_levels,
        exner_dynamical_increment=dycore_diagnostics.exner_dynamical_increment,
        normal_wind_advective_tendency_predictor=dycore_diagnostics.normal_wind_advective_tendency.predictor,
        normal_wind_advective_tendency_corrector=dycore_diagnostics.normal_wind_advective_tendency.corrector,
        vertical_wind_advective_tendency_predictor=dycore_diagnostics.vertical_wind_advective_tendency.predictor,
        vertical_wind_advective_tendency_corrector=dycore_diagnostics.vertical_wind_advective_tendency.corrector,
    )

    benchmark(
        device_utils.synchronized_function(solve_nonhydro.run, allocator=allocator),
        inputs,
        out,
    )
