# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: ERA001

from __future__ import annotations

from typing import Optional

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

import icon4py.model.atmosphere.dycore.velocity_advection_stencils as velocity_stencils
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.dycore.stencils import (
    fused_velocity_advection_stencil_1_to_7,
    fused_velocity_advection_stencil_8_to_13,
    fused_velocity_advection_stencil_15_to_18,
    fused_velocity_advection_stencil_19_to_20,
)
from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import (
    horizontal as h_grid,
    icon as icon_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


class VelocityAdvection:
    def __init__(
        self,
        grid: icon_grid.IconGrid,
        metric_state: dycore_states.MetricStateNonHydro,
        interpolation_state: dycore_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_params: grid_states.EdgeParams,
        owner_mask: fa.CellField[bool],
        backend: Optional[gtx_backend.Backend],
    ):
        self.grid: icon_grid.IconGrid = grid
        self._backend = backend
        self.metric_state: dycore_states.MetricStateNonHydro = metric_state
        self.interpolation_state: dycore_states.InterpolationState = interpolation_state
        self.vertical_params = vertical_params
        self.edge_params = edge_params
        self.c_owner_mask = owner_mask

        self.cfl_w_limit: float = 0.65
        self.scalfac_exdiff: float = 0.05
        self._allocate_local_fields()
        self._determine_local_domains()

        self._fused_velocity_advection_stencil_1_to_7_predictor = fused_velocity_advection_stencil_1_to_7.fused_velocity_advection_stencil_1_to_7_predictor.with_backend(
            self._backend
        )
        self._fused_velocity_advection_stencil_1_to_7_corrector = fused_velocity_advection_stencil_1_to_7.fused_velocity_advection_stencil_1_to_7_corrector.with_backend(
            self._backend
        )

        self._fused_velocity_advection_stencil_8_to_13_predictor = fused_velocity_advection_stencil_8_to_13.fused_velocity_advection_stencil_8_to_13_predictor.with_backend(
            self._backend
        )
        self._fused_velocity_advection_stencil_8_to_13_corrector = fused_velocity_advection_stencil_8_to_13.fused_velocity_advection_stencil_8_to_13_corrector.with_backend(
            self._backend
        )

        self._fused_velocity_advection_stencil_15_to_18 = fused_velocity_advection_stencil_15_to_18.fused_velocity_advection_stencil_15_to_18.with_backend(
            self._backend
        )

        self._fused_velocity_advection_stencil_19_to_20 = fused_velocity_advection_stencil_19_to_20.fused_velocity_advection_stencil_19_to_20.with_backend(
            self._backend
        )
        self._compute_horizontal_advection_term_for_vertical_velocity = (
            compute_horizontal_advection_term_for_vertical_velocity.with_backend(self._backend)
        )
        self._interpolate_to_cell_center = interpolate_to_cell_center.with_backend(self._backend)
        self._fused_stencils_9_10 = velocity_stencils.fused_stencils_9_10.with_backend(
            self._backend
        )
        self._fused_stencils_11_to_13 = velocity_stencils.fused_stencils_11_to_13.with_backend(
            self._backend
        )

        self._fused_stencil_14 = velocity_stencils.fused_stencil_14.with_backend(self._backend)
        self._interpolate_contravariant_vertical_velocity_to_full_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels.with_backend(self._backend)
        )
        self._fused_stencils_16_to_17 = velocity_stencils.fused_stencils_16_to_17.with_backend(
            self._backend
        )
        self._add_extra_diffusion_for_w_con_approaching_cfl = (
            add_extra_diffusion_for_w_con_approaching_cfl.with_backend(self._backend)
        )

    def _allocate_local_fields(self):
        self.z_v_grad_w = data_alloc.zero_field(
            self.grid, dims.EdgeDim, dims.KDim, backend=self._backend
        )
        self.z_ekinh = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        self.z_w_concorr_mc = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        self.z_w_con_c = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        self.z_w_con_c_full = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        self.cfl_clipping = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=bool, backend=self._backend
        )
        self.levmask = data_alloc.zero_field(
            self.grid, dims.KDim, dtype=bool, backend=self._backend
        )
        self.vcfl_dsl = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        self.k_field = data_alloc.index_field(
            self.grid, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        self.cell_field = data_alloc.index_field(self.grid, dims.CellDim, backend=self._backend)
        self.edge_field = data_alloc.index_field(self.grid, dims.EdgeDim, backend=self._backend)
        self.vertex_field = data_alloc.index_field(self.grid, dims.VertexDim, backend=self._backend)

    def _determine_local_domains(self):
        vertex_domain = h_grid.domain(dims.VertexDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_vertex_lateral_boundary_level_2 = self.grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_vertex_halo = self.grid.end_index(vertex_domain(h_grid.Zone.HALO))

        self._start_edge_lateral_boundary_level_5 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._start_edge_lateral_boundary_level_7 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        self._start_edge_nudging_level_2 = self.grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )

        self._end_edge_local = self.grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._end_edge_halo = self.grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._end_edge_halo_level_2 = self.grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        self._start_cell_lateral_boundary_level_3 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._start_cell_lateral_boundary_level_4 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_halo = self.grid.end_index(cell_domain(h_grid.Zone.HALO))

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_w_concorr_me: fa.EdgeKField[float],
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        # TODO: rbf array is inverted in serialized data
        rbf_vec_coeff_e = gtx.as_field(
            (dims.EdgeDim, dims.E2C2EDim),
            self.interpolation_state.rbf_vec_coeff_e.asnumpy().transpose(),
            allocator=self._backend,
        )
        self._fused_velocity_advection_stencil_1_to_7_predictor(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=self.metric_state.wgtfac_e,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            wgtfacq_e=self.metric_state.wgtfacq_e,
            nflatlev=self.vertical_params.nflatlev,
            c_intp=self.interpolation_state.c_intp,
            w=prognostic_state.w,
            inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            tangent_orientation=self.edge_params.tangent_orientation,
            z_vt_ie=z_vt_ie,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            z_v_grad_w=self.z_v_grad_w,
            k=self.k_field,
            nlev=gtx.int32(self.grid.num_levels),
            edge=self.edge_field,
            lvn_only=vn_only,
            lateral_boundary_7=self._start_edge_lateral_boundary_level_7,
            halo_1=self._end_edge_halo,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels + 1),
            offset_provider=self.grid.offset_providers,
        )

        self._fused_velocity_advection_stencil_8_to_13_predictor(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=self.metric_state.wgtfac_c,
            w=prognostic_state.w,
            z_w_concorr_mc=self.z_w_concorr_mc,
            w_concorr_c=diagnostic_state.w_concorr_c,
            z_ekinh=self.z_ekinh,
            z_w_con_c=self.z_w_con_c,
            k=self.k_field,
            nlev=self.grid.num_levels,
            nflatlev=self.vertical_params.nflatlev,
            lateral_boundary_3=self._start_cell_lateral_boundary_level_4,
            lateral_boundary_4=self._start_cell_lateral_boundary_level_4,
            end_halo=self._end_cell_halo,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        self._fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=gtx.int32(
                max(3, self.vertical_params.end_index_of_damping_layer - 2) - 1
            ),
            vertical_end=gtx.int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self._update_levmask_from_cfl_clipping()

        self._fused_velocity_advection_stencil_15_to_18(
            z_w_con_c=self.z_w_con_c,
            w=prognostic_state.w,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc.predictor,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_v_grad_w=self.z_v_grad_w,
            levelmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            z_w_con_c_full=self.z_w_con_c_full,
            cell=self.cell_field,
            k=self.k_field,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            cell_lower_bound=self._start_cell_nudging,
            cell_upper_bound=self._end_cell_local,
            nlev=gtx.int32(self.grid.num_levels),
            nrdmax=self.vertical_params.nrdmax,
            lvn_only=vn_only,
            extra_diffu=True,
            start_cell_lateral_boundary=self._start_cell_lateral_boundary_level_4,
            end_cell_halo=self._end_cell_halo,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        self.levelmask = self.levmask

        self._fused_velocity_advection_stencil_19_to_20(
            vn=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            levelmask=self.levelmask,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc.predictor,
            k=self.k_field,
            vertex=self.vertex_field,
            edge=self.edge_field,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=dtime,
            nlev=self.grid.num_levels,
            nrdmax=self.vertical_params.nrdmax,
            start_vertex_lateral_boundary_level_2=self._start_vertex_lateral_boundary_level_2,
            end_vertex_halo=self._end_vertex_halo,
            start_edge_nudging_level_2=self._start_edge_nudging_level_2,
            end_edge_local=self._end_edge_local,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(self.grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

    def _update_levmask_from_cfl_clipping(self):
        xp = data_alloc.import_array_ns(self._backend)
        self.levmask = gtx.as_field(
            domain=(dims.KDim,), data=(xp.any(self.cfl_clipping.ndarray, 0)), dtype=bool
        )

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        z_kin_hor_e: fa.EdgeKField[float],
        z_vt_ie: fa.EdgeKField[float],
        dtime: float,
        cell_areas: fa.CellField[float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        self._fused_velocity_advection_stencil_1_to_7_corrector(
            c_intp=self.interpolation_state.c_intp,
            w=prognostic_state.w,
            inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            tangent_orientation=self.edge_params.tangent_orientation,
            z_vt_ie=z_vt_ie,
            vn_ie=diagnostic_state.vn_ie,
            z_v_grad_w=self.z_v_grad_w,
            edge=self.edge_field,
            vertex=self.vertex_field,
            lateral_boundary_7=self._start_edge_lateral_boundary_level_7,
            halo_1=self._end_edge_halo,
            start_vertex_lateral_boundary_level_2=self._start_vertex_lateral_boundary_level_2,
            end_vertex_halo=self._end_vertex_halo,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels + 1),
            offset_provider=self.grid.offset_providers,
        )

        self._fused_velocity_advection_stencil_8_to_13_corrector(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            wgtfac_c=self.metric_state.wgtfac_c,
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            z_ekinh=self.z_ekinh,
            z_w_con_c=self.z_w_con_c,
            k=self.k_field,
            nlev=self.grid.num_levels,
            nflatlev=self.vertical_params.nflatlev,
            lateral_boundary=self._start_cell_lateral_boundary_level_3,
            end_halo=self._end_cell_halo,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        self._fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=gtx.int32(max(3, self.vertical_params.end_index_of_damping_layer - 2)),
            vertical_end=gtx.int32(self.grid.num_levels - 3),
            offset_provider=self.grid.offset_providers,
        )

        self._update_levmask_from_cfl_clipping()

        self._fused_velocity_advection_stencil_15_to_18(
            z_w_con_c=self.z_w_con_c,
            w=prognostic_state.w,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc.corrector,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_v_grad_w=self.z_v_grad_w,
            levelmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            z_w_con_c_full=self.z_w_con_c_full,
            cell=self.cell_field,
            k=self.k_field,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            cell_lower_bound=self._start_cell_nudging,
            cell_upper_bound=self._end_cell_local,
            nlev=gtx.int32(self.grid.num_levels),
            nrdmax=self.vertical_params.nrdmax,
            lvn_only=False,
            extra_diffu=True,
            start_cell_lateral_boundary=self._start_cell_lateral_boundary_level_4,
            end_cell_halo=self._end_cell_halo,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )
        # This behaviour needs to change for multiple blocks
        self.levelmask = self.levmask

        self._fused_velocity_advection_stencil_19_to_20(
            vn=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            levelmask=self.levelmask,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc.corrector,
            k=self.k_field,
            vertex=self.vertex_field,
            edge=self.edge_field,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=dtime,
            nlev=self.grid.num_levels,
            nrdmax=self.vertical_params.nrdmax,
            start_vertex_lateral_boundary_level_2=self._start_vertex_lateral_boundary_level_2,
            end_vertex_halo=self._end_vertex_halo,
            start_edge_nudging_level_2=self._start_edge_nudging_level_2,
            end_edge_local=self._end_edge_local,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(self.grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )
