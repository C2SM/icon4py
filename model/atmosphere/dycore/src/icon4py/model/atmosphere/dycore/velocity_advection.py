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
    compute_advection_in_horizontal_momentum_equation,
    compute_advection_in_vertical_momentum_equation,
    compute_cell_diagnostics_for_velocity_advection,
    compute_edge_diagnostics_for_velocity_advection,
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

        self._compute_vt_and_khalf_winds_and_horizontal_advection_of_w_and_contravariant_correction = compute_edge_diagnostics_for_velocity_advection.compute_vt_and_khalf_winds_and_horizontal_advection_of_w_and_contravariant_correction.with_backend(
            self._backend
        )
        self._compute_khalf_horizontal_advection_of_w = compute_edge_diagnostics_for_velocity_advection.compute_khalf_horizontal_advection_of_w.with_backend(
            self._backend
        )

        self._compute_horizontal_kinetic_energy_and_khalf_contravariant_terms = compute_cell_diagnostics_for_velocity_advection.compute_horizontal_kinetic_energy_and_khalf_contravariant_terms.with_backend(
            self._backend
        )
        self._compute_horizontal_kinetic_energy_and_khalf_contravariant_corrected_w = compute_cell_diagnostics_for_velocity_advection.compute_horizontal_kinetic_energy_and_khalf_contravariant_corrected_w.with_backend(
            self._backend
        )

        self._compute_advection_in_vertical_momentum_equation = compute_advection_in_vertical_momentum_equation.compute_advection_in_vertical_momentum_equation.with_backend(
            self._backend
        )

        self._compute_advection_in_horizontal_momentum_equation = compute_advection_in_horizontal_momentum_equation.compute_advection_in_horizontal_momentum_equation.with_backend(
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
        self._khalf_horizontal_advection_of_w_at_edge = data_alloc.zero_field(
            self.grid, dims.EdgeDim, dims.KDim, backend=self._backend
        )
        """
        Declared as z_v_grad_w in ICON. vn dw/dn + vt dw/dt. NOTE THAT IT ONLY HAS nlev LEVELS because w[nlevp1-1] is diagnostic.
        """

        self._horizontal_kinetic_energy_at_cell = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        """
        Declared as z_ekinh in ICON.
        """

        self._khalf_contravariant_corrected_w_at_cell = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        """
        Declared as z_w_con_c in ICON. w - (vn dz/dn + vt dz/dt), z is topography height
        """

        self._contravariant_corrected_w_at_cell = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        """
        Declared as z_w_con_c_full in ICON. w - (vn dz/dn + vt dz/dt), z is topography height
        """

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
        skip_compute_predictor_vertical_advection: bool,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        contravariant_correction_at_edge: fa.EdgeKField[float],
        horizontal_kinetic_energy_at_edge: fa.EdgeKField[float],
        khalf_tangential_wind: fa.EdgeKField[float],
        dtime: float,
        cell_areas: fa.CellField[float],
    ):
        """
        Compute some diagnostic variables that are used in the predictor step
        of the dycore and advective tendency of normal and vertical winds.

        Args:
            skip_compute_predictor_vertical_advection: Option to skip computation of advective tendency of vertical wind
            diagnostic_state: DiagnosticStateNonHydro class
            prognostic_state: PrognosticState class
            contravariant_correction_at_edge: Contravariant corrected vertical wind at edge [m s-1]
            horizontal_kinetic_energy_at_edge: Horizontal kinetic energy at edge [m^2 s-2]
            khalf_tangential_wind: tangential wind at edge on k-half levels [m s-1]
            dtime: time step [m s-1]
            cell_areas: cell area [m^2]
        """

        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        self._compute_vt_and_khalf_winds_and_horizontal_advection_of_w_and_contravariant_correction(
            tangential_wind=diagnostic_state.tangential_wind,
            khalf_tangential_wind=khalf_tangential_wind,
            khalf_vn=diagnostic_state.khalf_vn,
            horizontal_kinetic_energy_at_edge=horizontal_kinetic_energy_at_edge,
            contravariant_correction_at_edge=contravariant_correction_at_edge,
            khalf_horizontal_advection_of_w_at_edge=self._khalf_horizontal_advection_of_w_at_edge,
            vn=prognostic_state.vn,
            w=prognostic_state.w,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            wgtfac_e=self.metric_state.wgtfac_e,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            wgtfacq_e=self.metric_state.wgtfacq_e,
            c_intp=self.interpolation_state.c_intp,
            inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            tangent_orientation=self.edge_params.tangent_orientation,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            k=self.k_field,
            edge=self.edge_field,
            nflatlev=self.vertical_params.nflatlev,
            nlev=gtx.int32(self.grid.num_levels),
            lateral_boundary_7=self._start_edge_lateral_boundary_level_7,
            halo_1=self._end_edge_halo,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels + 1),
            offset_provider=self.grid.offset_providers,
        )

        self._compute_horizontal_kinetic_energy_and_khalf_contravariant_terms(
            horizontal_kinetic_energy_at_cell=self._horizontal_kinetic_energy_at_cell,
            khalf_contravariant_correction_at_cell=diagnostic_state.khalf_contravariant_correction_at_cell,
            khalf_contravariant_corrected_w_at_cell=self._khalf_contravariant_corrected_w_at_cell,
            w=prognostic_state.w,
            horizontal_kinetic_energy_at_edge=horizontal_kinetic_energy_at_edge,
            contravariant_correction_at_edge=contravariant_correction_at_edge,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            wgtfac_c=self.metric_state.wgtfac_c,
            k=self.k_field,
            nflatlev=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        # TODO (Chia Rui): rename this stencil
        self._fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self._khalf_contravariant_corrected_w_at_cell,
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

        self._compute_advection_in_vertical_momentum_equation(
            contravariant_corrected_w_at_cell=self._contravariant_corrected_w_at_cell,
            vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.predictor,
            w=prognostic_state.w,
            khalf_contravariant_corrected_w_at_cell=self._khalf_contravariant_corrected_w_at_cell,
            khalf_horizontal_advection_of_w_at_edge=self._khalf_horizontal_advection_of_w_at_edge,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            levelmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            cell=self.cell_field,
            k=self.k_field,
            cell_lower_bound=self._start_cell_nudging,
            cell_upper_bound=self._end_cell_local,
            nlev=gtx.int32(self.grid.num_levels),
            nrdmax=self.vertical_params.nrdmax,
            start_cell_lateral_boundary=self._start_cell_lateral_boundary_level_4,
            end_cell_halo=self._end_cell_halo,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        self.levelmask = self.levmask

        self._compute_advection_in_horizontal_momentum_equation(
            normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.predictor,
            vn=prognostic_state.vn,
            horizontal_kinetic_energy_at_edge=horizontal_kinetic_energy_at_edge,
            horizontal_kinetic_energy_at_cell=self._horizontal_kinetic_energy_at_cell,
            tangential_wind=diagnostic_state.tangential_wind,
            coriolis_frequency=self.edge_params.coriolis_frequency,
            contravariant_corrected_w_at_cell=self._contravariant_corrected_w_at_cell,
            khalf_vn=diagnostic_state.khalf_vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            c_lin_e=self.interpolation_state.c_lin_e,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=dtime,
            levelmask=self.levelmask,
            k=self.k_field,
            vertex=self.vertex_field,
            edge=self.edge_field,
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
        horizontal_kinetic_energy_at_edge: fa.EdgeKField[float],
        khalf_tangential_wind: fa.EdgeKField[float],
        dtime: float,
        cell_areas: fa.CellField[float],
    ):
        """
        Compute some diagnostic variables that are used in the corrector step
        of the dycore and advective tendency of normal and vertical winds.

        Args:
            skip_compute_predictor_vertical_advection: Option to skip computation of advective tendency of vertical wind
            diagnostic_state: DiagnosticStateNonHydro class
            prognostic_state: PrognosticState class
            horizontal_kinetic_energy_at_edge: Horizontal kinetic energy at edge [m^2 s-2]
            khalf_tangential_wind: tangential wind at edge on k-half levels [m s-1]
            dtime: time step [m s-1]
            cell_areas: cell area [m^2]
        """

        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        self._compute_khalf_horizontal_advection_of_w(
            khalf_horizontal_advection_of_w_at_edge=self._khalf_horizontal_advection_of_w_at_edge,
            w=prognostic_state.w,
            khalf_tangential_wind=khalf_tangential_wind,
            khalf_vn=diagnostic_state.khalf_vn,
            c_intp=self.interpolation_state.c_intp,
            inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            tangent_orientation=self.edge_params.tangent_orientation,
            edge=self.edge_field,
            vertex=self.vertex_field,
            lateral_boundary_7=self._start_edge_lateral_boundary_level_7,
            halo_1=self._end_edge_halo,
            start_vertex_lateral_boundary_level_2=self._start_vertex_lateral_boundary_level_2,
            end_vertex_halo=self._end_vertex_halo,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        self._compute_horizontal_kinetic_energy_and_khalf_contravariant_corrected_w(
            horizontal_kinetic_energy_at_cell=self._horizontal_kinetic_energy_at_cell,
            khalf_contravariant_correction_at_cell=diagnostic_state.khalf_contravariant_correction_at_cell,
            khalf_contravariant_corrected_w_at_cell=self._khalf_contravariant_corrected_w_at_cell,
            w=prognostic_state.w,
            horizontal_kinetic_energy_at_edge=horizontal_kinetic_energy_at_edge,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            k=self.k_field,
            nflatlev=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        # TODO (Chia Rui): rename this stencil
        self._fused_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self._khalf_contravariant_corrected_w_at_cell,
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

        self._compute_advection_in_vertical_momentum_equation(
            contravariant_corrected_w_at_cell=self._contravariant_corrected_w_at_cell,
            vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.corrector,
            w=prognostic_state.w,
            khalf_contravariant_corrected_w_at_cell=self._khalf_contravariant_corrected_w_at_cell,
            khalf_horizontal_advection_of_w_at_edge=self._khalf_horizontal_advection_of_w_at_edge,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=False,
            levelmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            cell=self.cell_field,
            k=self.k_field,
            cell_lower_bound=self._start_cell_nudging,
            cell_upper_bound=self._end_cell_local,
            nlev=gtx.int32(self.grid.num_levels),
            nrdmax=self.vertical_params.nrdmax,
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

        self._compute_advection_in_horizontal_momentum_equation(
            normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.corrector,
            vn=prognostic_state.vn,
            horizontal_kinetic_energy_at_edge=horizontal_kinetic_energy_at_edge,
            horizontal_kinetic_energy_at_cell=self._horizontal_kinetic_energy_at_cell,
            tangential_wind=diagnostic_state.tangential_wind,
            coriolis_frequency=self.edge_params.coriolis_frequency,
            contravariant_corrected_w_at_cell=self._contravariant_corrected_w_at_cell,
            khalf_vn=diagnostic_state.khalf_vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            c_lin_e=self.interpolation_state.c_lin_e,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=dtime,
            levelmask=self.levelmask,
            k=self.k_field,
            vertex=self.vertex_field,
            edge=self.edge_field,
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
