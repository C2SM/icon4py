# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np

import icon4py.model.atmosphere.dycore.velocity.velocity_advection_program as velocity_prog
from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.compute_tangential_wind import compute_tangential_wind
from icon4py.model.atmosphere.dycore.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.atmosphere.dycore.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.interpolate_vt_to_interface_edges import (
    interpolate_vt_to_interface_edges,
)
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


class VelocityAdvection:
    def __init__(
        self,
        grid: icon_grid.IconGrid,
        metric_state: solve_nh_states.MetricStateNonHydro,
        interpolation_state: solve_nh_states.InterpolationState,
        vertical_params: v_grid.VerticalGridParams,
        edge_params: h_grid.EdgeParams,
        owner_mask: fa.CellField[bool],
    ):
        self._initialized = False
        self.grid: icon_grid.IconGrid = grid
        self.metric_state: solve_nh_states.MetricStateNonHydro = metric_state
        self.interpolation_state: solve_nh_states.InterpolationState = interpolation_state
        self.vertical_params = vertical_params
        self.edge_params = edge_params
        self.c_owner_mask = owner_mask

        self.cfl_w_limit: float = 0.65
        self.scalfac_exdiff: float = 0.05
        self._allocate_local_fields()

        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_w_v = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, is_halfdim=True, grid=self.grid
        )
        self.z_v_grad_w = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=self.grid)
        self.z_ekinh = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)
        self.z_w_concorr_mc = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid
        )
        self.z_w_con_c = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self.grid
        )
        self.zeta = field_alloc.allocate_zero_field(dims.VertexDim, dims.KDim, grid=self.grid)
        self.z_w_con_c_full = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid
        )
        self.cfl_clipping = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, dtype=bool
        )
        self.levmask = field_alloc.allocate_zero_field(dims.KDim, grid=self.grid, dtype=bool)
        self.vcfl_dsl = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=self.grid)
        self.k_field = field_alloc.allocate_indices(dims.KDim, grid=self.grid, is_halfdim=True)

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_w_concorr_me: fa.EdgeKField[float],
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        ntnd: int,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)
        vertex_domain = h_grid.domain(dims.VertexDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        start_vertex_lateral_boundary_row_2 = self.grid.start_index(
            vertex_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_2)
        )
        end_vertex_halo = self.grid.end_index(vertex_domain(h_grid.Marker.HALO))

        start_edge_lateral_boundary_row_5 = self.grid.start_index(
            edge_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_5)
        )
        start_edge_lateral_boundary_row_7 = self.grid.start_index(
            edge_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_7)
        )
        start_edge_nudging_row_2 = self.grid.start_index(edge_domain(h_grid.Marker.NUDGING_LEVEL_2))
        end_edge_local = self.grid.end_index(edge_domain(h_grid.Marker.LOCAL))

        end_edge_local_minus1 = self.grid.end_index(edge_domain(h_grid.Marker.HALO))
        end_edge_local_minus2 = self.grid.end_index(edge_domain(h_grid.Marker.HALO_LEVEL_2))

        start_cell_lb_plus3 = self.grid.start_index(
            cell_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_4)
        )
        start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Marker.NUDGING))
        end_cell_local = self.grid.end_index(cell_domain(h_grid.Marker.LOCAL))
        end_cell_local_minus1 = self.grid.end_index(cell_domain(h_grid.Marker.HALO))

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=start_vertex_lateral_boundary_row_2,
                horizontal_end=end_vertex_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        mo_math_divrot_rot_vertex_ri_dsl(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=start_vertex_lateral_boundary_row_2,
            horizontal_end=end_vertex_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        compute_tangential_wind(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            vt=diagnostic_state.vt,
            horizontal_start=start_edge_lateral_boundary_row_5,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        interpolate_vn_to_ie_and_compute_ekin_on_edges(
            wgtfac_e=self.metric_state.wgtfac_e,
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=start_edge_lateral_boundary_row_5,
            horizontal_end=end_edge_local_minus2,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            interpolate_vt_to_interface_edges(
                wgtfac_e=self.metric_state.wgtfac_e,
                vt=diagnostic_state.vt,
                z_vt_ie=z_vt_ie,
                horizontal_start=start_edge_lateral_boundary_row_5,
                horizontal_end=end_edge_local_minus2,
                vertical_start=1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        velocity_prog.fused_stencils_4_5(
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_edge_lateral_boundary_row_5,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        velocity_prog.extrapolate_at_top(
            wgtfacq_e=self.metric_state.wgtfacq_e,
            vn=prognostic_state.vn,
            vn_ie=diagnostic_state.vn_ie,
            horizontal_start=start_edge_lateral_boundary_row_5,
            horizontal_end=end_edge_local_minus2,
            vertical_start=self.grid.num_levels,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            compute_horizontal_advection_term_for_vertical_velocity(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=start_edge_lateral_boundary_row_7,
                horizontal_end=end_edge_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        interpolate_to_cell_center(
            interpolant=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            interpolation=self.z_ekinh,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        velocity_prog.fused_stencils_9_10(
            z_w_concorr_me=z_w_concorr_me,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_concorr_mc=self.z_w_concorr_mc,
            wgtfac_c=self.metric_state.wgtfac_c,
            w_concorr_c=diagnostic_state.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        velocity_prog.fused_stencils_11_to_13(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        velocity_prog.fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=gtx.int32(
                max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
            ),
            vertical_end=gtx.int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            velocity_prog.fused_stencils_16_to_17(
                w=prognostic_state.w,
                local_z_v_grad_w=self.z_v_grad_w,
                e_bln_c_s=self.interpolation_state.e_bln_c_s,
                local_z_w_con_c=self.z_w_con_c,
                coeff1_dwdz=self.metric_state.coeff1_dwdz,
                coeff2_dwdz=self.metric_state.coeff2_dwdz,
                ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

            add_extra_diffusion_for_w_con_approaching_cfl(
                levmask=self.levmask,
                cfl_clipping=self.cfl_clipping,
                owner_mask=self.c_owner_mask,
                z_w_con_c=self.z_w_con_c,
                ddqz_z_half=self.metric_state.ddqz_z_half,
                area=cell_areas,
                geofac_n2s=self.interpolation_state.geofac_n2s,
                w=prognostic_state.w,
                ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
                scalfac_exdiff=scalfac_exdiff,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=gtx.int32(
                    max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
                ),
                vertical_end=gtx.int32(self.grid.num_levels - 3),
                offset_provider=self.grid.offset_providers,
            )

        self.levelmask = self.levmask

        compute_advective_normal_wind_tendency(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=start_edge_nudging_row_2,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=start_edge_nudging_row_2,
            horizontal_end=end_edge_local,
            vertical_start=gtx.int32(
                max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
            ),
            vertical_end=gtx.int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )

    def _update_levmask_from_cfl_clipping(self):
        self.levmask = gtx.as_field(
            domain=(dims.KDim,), data=(np.any(self.cfl_clipping.asnumpy(), 0)), dtype=bool
        )

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        ntnd: int,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        vertex_domain = h_grid.domain(dims.VertexDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        start_vertex_lateral_boundary_row_2 = self.grid.start_index(
            vertex_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_2)
        )
        end_vertex_halo = self.grid.end_index(vertex_domain(h_grid.Marker.HALO))
        start_edge_lateral_boundary_row_7 = self.grid.start_index(
            edge_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_7)
        )
        start_edge_nudging_row_2 = self.grid.start_index(edge_domain(h_grid.Marker.NUDGING_LEVEL_2))
        end_edge_local = self.grid.end_index(edge_domain(h_grid.Marker.LOCAL))
        end_edge_halo = self.grid.end_index(edge_domain(h_grid.Marker.HALO))

        start_cell_lateral_boundary_row_3 = self.grid.start_index(
            cell_domain(h_grid.Marker.LATERAL_BOUNDARY_LEVEL_4)
        )

        start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Marker.NUDGING))
        end_cell_local = self.grid.end_index(cell_domain(h_grid.Marker.LOCAL))
        end_cell_halo = self.grid.end_index(cell_domain(h_grid.Marker.HALO))

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=start_vertex_lateral_boundary_row_2,
                horizontal_end=end_vertex_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        mo_math_divrot_rot_vertex_ri_dsl(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=start_vertex_lateral_boundary_row_2,
            horizontal_end=end_vertex_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not vn_only:
            compute_horizontal_advection_term_for_vertical_velocity(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=start_edge_lateral_boundary_row_7,
                horizontal_end=end_edge_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        interpolate_to_cell_center(
            interpolant=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            interpolation=self.z_ekinh,
            horizontal_start=start_cell_lateral_boundary_row_3,
            horizontal_end=end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        velocity_prog.fused_stencils_11_to_13(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lateral_boundary_row_3,
            horizontal_end=end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        velocity_prog.fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_lateral_boundary_row_3,
            horizontal_end=end_cell_halo,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        interpolate_contravariant_vertical_velocity_to_full_levels(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=start_cell_lateral_boundary_row_3,
            horizontal_end=end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        velocity_prog.fused_stencils_16_to_17(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        add_extra_diffusion_for_w_con_approaching_cfl(
            levmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )

        # This behaviour needs to change for multiple blocks
        self.levelmask = self.levmask

        compute_advective_normal_wind_tendency(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=start_edge_nudging_row_2,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=start_edge_nudging_row_2,
            horizontal_end=end_edge_local,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 4),
            offset_provider=self.grid.offset_providers,
        )
