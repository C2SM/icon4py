# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.velocity_advection_predictor import (
    compute_velocity_advection_in_predictor_step,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests

from .test_velocity_advection_terms import (
    _restore_outside,
    compute_advection_in_horizontal_momentum_numpy,
    compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy,
    compute_diagnostics_from_normal_wind_numpy,
    compute_horizontal_advection_of_w_numpy,
    compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy,
    interpolate_contravariant_correction_to_cells_on_half_levels_numpy,
    interpolate_contravariant_vertical_velocity_to_full_levels_numpy,
)


@pytest.mark.embedded_remap_error
@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestComputeVelocityAdvectionInPredictorStep(stencil_tests.StencilTest):
    PROGRAM = compute_velocity_advection_in_predictor_step
    OUTPUTS = (
        "tangential_wind",
        "tangential_wind_on_half_levels",
        "vn_on_half_levels",
        "horizontal_kinetic_energy_at_edges_on_model_levels",
        "contravariant_correction_at_edges_on_model_levels",
        "contravariant_correction_at_cells_on_half_levels",
        "vertical_wind_advective_tendency",
        "vertical_cfl",
        "normal_wind_advective_tendency",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "end_index_of_damping_layer",
            "skip_compute_predictor_vertical_advection",
            "apply_extra_diffusion_on_vn",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "start_edge_lateral_boundary_level_5",
            "end_edge_halo_level_2",
            "start_cell_lateral_boundary_level_4",
            "end_cell_halo",
            "start_edge_nudging_level_2",
            "end_edge_local",
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "end_index_of_damping_layer",
            "skip_compute_predictor_vertical_advection",
            "apply_extra_diffusion_on_vn",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        tangential_wind: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vertical_wind_advective_tendency: np.ndarray,
        vertical_cfl: np.ndarray,
        normal_wind_advective_tendency: np.ndarray,
        vn: np.ndarray,
        w: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        owner_mask: np.ndarray,
        coriolis_frequency: np.ndarray,
        geofac_rot: np.ndarray,
        coeff_gradekin: np.ndarray,
        c_lin_e: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        area_edge: np.ndarray,
        geofac_grdiv: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        skip_compute_predictor_vertical_advection: bool,
        apply_extra_diffusion_on_vn: bool,
        nflatlev: int,
        end_index_of_damping_layer: int,
        start_edge_lateral_boundary_level_5: int,
        end_edge_halo_level_2: int,
        start_cell_lateral_boundary_level_4: int,
        end_cell_halo: int,
        start_edge_nudging_level_2: int,
        end_edge_local: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        nlev = vertical_end

        (
            tangential_wind_new,
            tangential_wind_on_half_levels_new,
            vn_on_half_levels_new,
            horizontal_kinetic_energy_new,
            contravariant_correction_at_edges_new,
        ) = compute_diagnostics_from_normal_wind_numpy(
            connectivities=connectivities,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nlev=nlev,
        )

        contravariant_correction_at_cells_on_half_levels_new = interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
            connectivities=connectivities,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_new,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            nflatlev=nflatlev,
            nlev=nlev,
        )

        (
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            vertical_cfl_new,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            w=w[:, :-1],
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_new,
            ddqz_z_half=ddqz_z_half[:, :-1],
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        vertical_wind_advective_tendency_new = vertical_wind_advective_tendency[:, :-1]
        if not skip_compute_predictor_vertical_advection:
            horizontal_advection_of_w_at_edges_on_half_levels = (
                compute_horizontal_advection_of_w_numpy(
                    connectivities=connectivities,
                    w=w,
                    tangential_wind_on_half_levels=tangential_wind_on_half_levels_new,
                    vn_on_half_levels=vn_on_half_levels_new,
                    c_intp=c_intp,
                    inv_dual_edge_length=inv_dual_edge_length,
                    inv_primal_edge_length=inv_primal_edge_length,
                    tangent_orientation=tangent_orientation,
                )
            )
            vertical_wind_advective_tendency_new = compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
                connectivities=connectivities,
                vertical_wind_advective_tendency=vertical_wind_advective_tendency_new,
                w=w,
                horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
                contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
                cfl_clipping=cfl_clipping,
                coeff1_dwdz=coeff1_dwdz,
                coeff2_dwdz=coeff2_dwdz,
                e_bln_c_s=e_bln_c_s,
                ddqz_z_half=ddqz_z_half[:, :-1],
                area=area,
                geofac_n2s=geofac_n2s,
                owner_mask=owner_mask,
                scalfac_exdiff=scalfac_exdiff,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
                nlev=nlev,
                end_index_of_damping_layer=end_index_of_damping_layer,
            )

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        normal_wind_advective_tendency_new = compute_advection_in_horizontal_momentum_numpy(
            connectivities=connectivities,
            vn=vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_new,
            tangential_wind=tangential_wind_new,
            coriolis_frequency=coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=vn_on_half_levels_new,
            e_bln_c_s=e_bln_c_s,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        edge_slice = (start_edge_lateral_boundary_level_5, end_edge_halo_level_2)
        cell_slice = (start_cell_lateral_boundary_level_4, end_cell_halo)

        return dict(
            tangential_wind=_restore_outside(
                tangential_wind_new, tangential_wind, edge_slice, (vertical_start, vertical_end)
            ),
            tangential_wind_on_half_levels=_restore_outside(
                tangential_wind_on_half_levels_new,
                tangential_wind_on_half_levels,
                edge_slice,
                (vertical_start, vertical_end),
            ),
            vn_on_half_levels=_restore_outside(
                vn_on_half_levels_new,
                vn_on_half_levels,
                edge_slice,
                (vertical_start, vertical_end + 1),
            ),
            horizontal_kinetic_energy_at_edges_on_model_levels=_restore_outside(
                horizontal_kinetic_energy_new,
                horizontal_kinetic_energy_at_edges_on_model_levels,
                edge_slice,
                (vertical_start, vertical_end),
            ),
            contravariant_correction_at_edges_on_model_levels=_restore_outside(
                contravariant_correction_at_edges_new,
                contravariant_correction_at_edges_on_model_levels,
                edge_slice,
                (nflatlev, vertical_end),
            ),
            contravariant_correction_at_cells_on_half_levels=_restore_outside(
                contravariant_correction_at_cells_on_half_levels_new,
                contravariant_correction_at_cells_on_half_levels,
                cell_slice,
                (vertical_start, vertical_end),
            ),
            vertical_wind_advective_tendency=_restore_outside(
                vertical_wind_advective_tendency_new,
                vertical_wind_advective_tendency,
                cell_slice,
                # ICON computes the tendency over jk = 2..nlev
                # (mo_velocity_advection.f90:598), so the top half level is untouched.
                (vertical_start + 1, vertical_end),
            ),
            vertical_cfl=_restore_outside(
                vertical_cfl_new, vertical_cfl, cell_slice, (vertical_start, vertical_end)
            ),
            normal_wind_advective_tendency=_restore_outside(
                normal_wind_advective_tendency_new,
                normal_wind_advective_tendency,
                (start_edge_nudging_level_2, end_edge_local),
                (vertical_start, vertical_end),
            ),
        )

    @stencil_tests.input_data_fixture(
        params=[
            {
                "skip_compute_predictor_vertical_advection": skip,
                "apply_extra_diffusion_on_vn": diffu,
            }
            for skip, diffu in ((False, True), (True, False))
        ],
        ids=lambda param: (
            f"skip_compute_predictor_vertical_advection[{param['skip_compute_predictor_vertical_advection']}]"
            f"-apply_extra_diffusion_on_vn[{param['apply_extra_diffusion_on_vn']}]"
        ),
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        tangential_wind = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim)
        vn_on_half_levels = data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim)
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            dims.CellDim, dims.KHalfDim
        )
        vertical_wind_advective_tendency = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        vertical_cfl = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        normal_wind_advective_tendency = data_alloc.zero_field(dims.EdgeDim, dims.KDim)

        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        w = data_alloc.random_field(dims.CellDim, dims.KHalfDim)

        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim)
        wgtfac_e = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim)
        wgtfacq_e = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        coeff1_dwdz = data_alloc.random_field(dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(dims.CellDim, dims.KDim)
        c_intp = data_alloc.random_field(dims.VertexDim, dims.V2CDim)
        inv_dual_edge_length = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        inv_primal_edge_length = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        tangent_orientation = data_alloc.random_field(dims.EdgeDim, low=1.0e-5)
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        ddqz_z_half = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        area = data_alloc.random_field(dims.CellDim)
        geofac_n2s = data_alloc.random_field(dims.CellDim, dims.C2E2CODim)
        owner_mask = data_alloc.random_mask(dims.CellDim)
        coriolis_frequency = data_alloc.random_field(dims.EdgeDim)
        geofac_rot = data_alloc.random_field(dims.VertexDim, dims.V2EDim)
        coeff_gradekin = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        c_lin_e = data_alloc.random_field(dims.EdgeDim, dims.E2CDim)
        # low=0.0 makes sure the simplified stencil produces the same result as the numpy version
        ddqz_z_full_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, low=0.0)
        area_edge = data_alloc.random_field(dims.EdgeDim)
        geofac_grdiv = data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim)

        scalfac_exdiff = 10.0
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime

        # values are set to reflect the MCH ch1 experiment. Changing them changes the runtime
        nflatlev = 5
        end_index_of_damping_layer = 12

        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)

        return dict(
            tangential_wind=tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            vertical_cfl=vertical_cfl,
            normal_wind_advective_tendency=normal_wind_advective_tendency,
            vn=vn,
            w=w,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            coriolis_frequency=coriolis_frequency,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            geofac_grdiv=geofac_grdiv,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=request.param[
                "skip_compute_predictor_vertical_advection"
            ],
            apply_extra_diffusion_on_vn=request.param["apply_extra_diffusion_on_vn"],
            nflatlev=nflatlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
            start_edge_lateral_boundary_level_5=grid.start_index(
                edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
            ),
            end_edge_halo_level_2=grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2)),
            start_cell_lateral_boundary_level_4=grid.start_index(
                cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
            ),
            end_cell_halo=grid.end_index(cell_domain(h_grid.Zone.HALO)),
            start_edge_nudging_level_2=grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2)),
            end_edge_local=grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
