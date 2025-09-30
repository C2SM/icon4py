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

from icon4py.model.atmosphere.dycore.dycore_states import DivergenceDampingOrder
from icon4py.model.common.constants import RayleighType


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing
    from gt4py.next.typing import NDArrayObject
import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import utils as common_utils
from icon4py.model.common.decomposition.definitions import DecompositionInfo
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    vertical as v_grid,
)
from icon4py.model.common.initialization.jablonowski_williamson_topography import (
    jablonowski_williamson_topography,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import definitions, grid_utils
from icon4py.model.testing.grid_utils import construct_decomposition_info

from .. import utils
from ..fixtures import *  # noqa: F403


@pytest.mark.embedded_remap_error
@pytest.mark.benchmark
@pytest.mark.parametrize("grid", [definitions.Grids.MCH_OPR_R04B07_DOMAIN01, definitions.Grids.R02B07_GLOBAL])
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
def test_solve_nonhydro_benchmark(
    grid: definitions.GridDescription,
    backend: gtx_typing.Backend | None,
    benchmark: Any,
) -> None:
    dtime = 1.0
    lprep_adv = True
    ndyn_substeps = 5
    at_initial_timestep = True
    second_order_divdamp_factor = 0.0

    config = solve_nh.NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=DivergenceDampingOrder.COMBINED,
        iau_wgt_dyn=1.0,
        fourth_order_divdamp_factor=0.004,
        max_nudging_coefficient=0.375,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config)

    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid, num_levels=80, keep_skip_values=True, backend=backend
    )

    mesh = grid_manager.grid
    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.geometry_fields

    decomposition_info = construct_decomposition_info(mesh, backend)

    geometry_field_source = grid_geometry.GridGeometry(
        grid=mesh,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs,
    )

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
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
        primal_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
        primal_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        dual_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
        dual_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        primal_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_U),
        dual_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_U),
        primal_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_V),
        dual_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_V),
        edge_areas=geometry_field_source.get(geometry_meta.EDGE_AREA),
        coriolis_frequency=geometry_field_source.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_center_lat=geometry_field_source.get(geometry_meta.EDGE_LAT),
        edge_center_lon=geometry_field_source.get(geometry_meta.EDGE_LON),
        primal_normal_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_U),
        primal_normal_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_V),
    )

    topo_c = jablonowski_williamson_topography(
        cell_lat=cell_geometry.cell_center_lat.ndarray,
        u0=35.0,
        backend=backend,
    )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        mass_flx_me=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
    )

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=mesh,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=mesh,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c),
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=RayleighType.KLEMP,
        rayleigh_coeff=5.0,
        exner_expol=0.333,
        vwind_offctr=0.2,
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
        bdy_halo_c=metrics_field_source.get(metrics_attributes.BDY_HALO_C),
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
        nflat_gradp=metrics_field_source.get(metrics_attributes.NFLAT_GRADP),
        pg_edgeidx_dsl=metrics_field_source.get(metrics_attributes.PG_EDGEIDX_DSL),
        pg_exdist=metrics_field_source.get(metrics_attributes.PG_EDGEDIST_DSL),
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

    diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=0.0,
        theta_v_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
        perturbed_exner_at_cells_on_model_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
        rho_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
        exner_tendency_due_to_slow_physics=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
        grf_tend_rho=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        grf_tend_thv=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        grf_tend_w=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        mass_flux_at_edges_on_model_levels=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, backend=backend
        ),
        normal_wind_tendency_due_to_slow_physics_process=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, backend=backend
        ),
        grf_tend_vn=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
            data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
            data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        ),
        tangential_wind=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        vn_on_half_levels=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        contravariant_correction_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
        rho_iau_increment=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        normal_wind_iau_increment=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, backend=backend
        ),
        exner_iau_increment=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        exner_dynamical_increment=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, backend=backend
        ),
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
        owner_mask=data_alloc.random_field(mesh, dims.CellDim, dtype=bool, backend=backend),
        backend=backend,
    )

    prognostic_state_nnow = prognostics.PrognosticState(
        w=data_alloc.random_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, low=0.0, backend=backend
        ),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, backend=backend),
    )
    prognostic_state_nnew = prognostics.PrognosticState(
        w=data_alloc.random_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, low=0.0, backend=backend
        ),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, backend=backend),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, backend=backend),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, backend=backend),
    )

    prognostic_states = common_utils.TimeStepPair(prognostic_state_nnow, prognostic_state_nnew)

    for i_substep in range(ndyn_substeps):
        at_first_substep = i_substep == 0
        at_last_substep = i_substep == ndyn_substeps - 1

        benchmark(
            solve_nonhydro.time_step,
            diagnostic_state_nh,
            prognostic_states,
            prep_adv,
            second_order_divdamp_factor,
            dtime,
            ndyn_substeps,
            at_initial_timestep,
            lprep_adv,
            at_first_substep,
            at_last_substep,
        )
