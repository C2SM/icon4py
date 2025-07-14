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

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.dycore.stencils import (
    compute_advection_in_horizontal_momentum_equation,
    compute_advection_in_vertical_momentum_equation,
    compute_cell_diagnostics_for_velocity_advection,
    compute_edge_diagnostics_for_velocity_advection,
    compute_maximum_cfl_and_clip_contravariant_vertical_velocity,
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

from icon4py.model.common.io import plots

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
        extras: dict = {},
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

        self._compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction = compute_edge_diagnostics_for_velocity_advection.compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction.with_backend(
            self._backend
        )
        self._compute_horizontal_advection_of_w = compute_edge_diagnostics_for_velocity_advection.compute_horizontal_advection_of_w.with_backend(
            self._backend
        )

        self._interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms = compute_cell_diagnostics_for_velocity_advection.interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms.with_backend(
            self._backend
        )
        self._compute_maximum_cfl_and_clip_contravariant_vertical_velocity = compute_maximum_cfl_and_clip_contravariant_vertical_velocity.compute_maximum_cfl_and_clip_contravariant_vertical_velocity.with_backend(
            self._backend
        )
        self._interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_corrected_w = compute_cell_diagnostics_for_velocity_advection.interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_corrected_w.with_backend(
            self._backend
        )

        self._compute_advection_in_vertical_momentum_equation = compute_advection_in_vertical_momentum_equation.compute_advection_in_vertical_momentum_equation.with_backend(
            self._backend
        )

        self._compute_advection_in_horizontal_momentum_equation = compute_advection_in_horizontal_momentum_equation.compute_advection_in_horizontal_momentum_equation.with_backend(
            self._backend
        )

        #---> IBM
        if "ibm" in extras:
            self._ibm = extras["ibm"]
        #<--- IBM

    def _allocate_local_fields(self):
        self._horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.zero_field(
            self.grid, dims.EdgeDim, dims.KDim, backend=self._backend
        )
        """
        Declared as z_v_grad_w in ICON. vn dw/dn + vt dw/dt. NOTE THAT IT ONLY HAS nlev LEVELS because w[nlevp1-1] is diagnostic.
        """

        self._horizontal_kinetic_energy_at_cells_on_model_levels = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        """
        Declared as z_ekinh in ICON.
        """

        self._contravariant_corrected_w_at_cells_on_half_levels = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        """
        Declared as z_w_con_c in ICON. w - (vn dz/dn + vt dz/dt), z is topography height
        """

        self._contravariant_corrected_w_at_cells_on_model_levels = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        """
        Declared as z_w_con_c_full in ICON. w - (vn dz/dn + vt dz/dt), z is topography height
        """

        self.cfl_clipping = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=bool, backend=self._backend
        )
        self.vcfl_dsl = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )

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
        contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[float],
        horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[float],
        tangential_wind_on_half_levels: fa.EdgeKField[float],
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
            contravariant_correction_at_edges_on_model_levels: Contravariant corrected vertical wind at edge [m s-1]
            horizontal_kinetic_energy_at_edges_on_model_levels: Horizontal kinetic energy at edge [m^2 s-2]
            tangential_wind_on_half_levels: tangential wind at edge on k-half levels [m s-1]
            dtime: time step [m s-1]
            cell_areas: cell area [m^2]
        """

        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        self._compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
            tangential_wind=diagnostic_state.tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            ibm_dvndz_mask=self._ibm.half_edge_mask,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=self._horizontal_advection_of_w_at_edges_on_half_levels,
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
            nflatlev=self.vertical_params.nflatlev,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels + 1),
            offset_provider=self.grid.offset_providers,
        )

        self._interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms(
            horizontal_kinetic_energy_at_cells_on_model_levels=self._horizontal_kinetic_energy_at_cells_on_model_levels,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state.contravariant_correction_at_cells_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=self._contravariant_corrected_w_at_cells_on_half_levels,
            w=prognostic_state.w,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            wgtfac_c=self.metric_state.wgtfac_c,
            nflatlev=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        # TODO most likely this should be inlined into the next function
        self._compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            z_w_con_c=self._contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping=self.cfl_clipping,
            vcfl=self.vcfl_dsl,
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

        # note level_mask removed, because all accesses where additionally checking cfl_clipping
        # TODO(havogt): however, our test data is probably not able to catch cfl_clipping conditons
        self._compute_advection_in_vertical_momentum_equation(
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.predictor,
            w=prognostic_state.w,
            contravariant_corrected_w_at_cells_on_half_levels=self._contravariant_corrected_w_at_cells_on_half_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=self._horizontal_advection_of_w_at_edges_on_half_levels,
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
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            nlev=gtx.int32(self.grid.num_levels),
            end_index_of_damping_layer=self.vertical_params.end_index_of_damping_layer,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        # TODO(havogt): can we move this to the end?
        xp = data_alloc.import_array_ns(self._backend)
        levmask = gtx.as_field(
            domain=(dims.KDim,), data=(xp.any(self.cfl_clipping.ndarray, 0)), dtype=bool
        )
        self._compute_advection_in_horizontal_momentum_equation(
            normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.predictor,
            vn=prognostic_state.vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            horizontal_kinetic_energy_at_cells_on_model_levels=self._horizontal_kinetic_energy_at_cells_on_model_levels,
            tangential_wind=diagnostic_state.tangential_wind,
            coriolis_frequency=self.edge_params.coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
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
            levelmask=levmask,  # TODO(havogt): can we get rid of the levelmask here?
            nlev=self.grid.num_levels,
            end_index_of_damping_layer=self.vertical_params.end_index_of_damping_layer,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )
        #---> IBM
        if self._ibm.DEBUG_LEVEL >= 4:
            plots.pickle_data(
                state={
                    "vn_adv": diagnostic_state.normal_wind_advective_tendency.predictor.asnumpy(),
                    "vn_eh": diagnostic_state.vn_on_half_levels.asnumpy(),
                    "w_wcc_cf": self._contravariant_corrected_w_at_cells_on_model_levels.asnumpy(),
                },
                label=f"advection_predictor",
            )
        #<--- IBM

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[float],
        tangential_wind_on_half_levels: fa.EdgeKField[float],
        dtime: float,
        cell_areas: fa.CellField[float],
    ):
        """
        Compute some diagnostic variables that are used in the corrector step
        of the dycore and advective tendency of normal and vertical winds.

        Args:
            diagnostic_state: DiagnosticStateNonHydro class
            prognostic_state: PrognosticState class
            horizontal_kinetic_energy_at_edges_on_model_levels: Horizontal kinetic energy at edge [m^2 s-2]
            tangential_wind_on_half_levels: tangential wind at edge on k-half levels [m s-1]
            dtime: time step [m s-1]
            cell_areas: cell area [m^2]
        """

        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        self._compute_horizontal_advection_of_w(
            horizontal_advection_of_w_at_edges_on_half_levels=self._horizontal_advection_of_w_at_edges_on_half_levels,
            w=prognostic_state.w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
            c_intp=self.interpolation_state.c_intp,
            inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            tangent_orientation=self.edge_params.tangent_orientation,
            horizontal_start=self._start_edge_lateral_boundary_level_7,
            horizontal_end=self._end_edge_halo,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        #---> IBM
        self._ibm.set_bcs_gradh_w(self._horizontal_advection_of_w_at_edges_on_half_levels)
        #<--- IBM

        self._interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_corrected_w(
            horizontal_kinetic_energy_at_cells_on_model_levels=self._horizontal_kinetic_energy_at_cells_on_model_levels,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state.contravariant_correction_at_cells_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=self._contravariant_corrected_w_at_cells_on_half_levels,
            w=prognostic_state.w,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            nflatlev=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        self._compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            z_w_con_c=self._contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping=self.cfl_clipping,
            vcfl=self.vcfl_dsl,
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

        self._compute_advection_in_vertical_momentum_equation(
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.corrector,
            w=prognostic_state.w,
            contravariant_corrected_w_at_cells_on_half_levels=self._contravariant_corrected_w_at_cells_on_half_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=self._horizontal_advection_of_w_at_edges_on_half_levels,
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
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            nlev=gtx.int32(self.grid.num_levels),
            end_index_of_damping_layer=self.vertical_params.end_index_of_damping_layer,
            horizontal_start=self._start_cell_lateral_boundary_level_4,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        # TODO(havogt): can we move this to the end?
        xp = data_alloc.import_array_ns(self._backend)
        levmask = gtx.as_field(
            domain=(dims.KDim,), data=(xp.any(self.cfl_clipping.ndarray, 0)), dtype=bool
        )
        self._compute_advection_in_horizontal_momentum_equation(
            normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.corrector,
            vn=prognostic_state.vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            horizontal_kinetic_energy_at_cells_on_model_levels=self._horizontal_kinetic_energy_at_cells_on_model_levels,
            tangential_wind=diagnostic_state.tangential_wind,
            coriolis_frequency=self.edge_params.coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
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
            levelmask=levmask,  # TODO(havogt): can we get rid of the levelmask here?
            nlev=self.grid.num_levels,
            end_index_of_damping_layer=self.vertical_params.end_index_of_damping_layer,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )
        #---> IBM
        if self._ibm.DEBUG_LEVEL >= 4:
            plots.pickle_data(
                state={
                    "vn_adv": diagnostic_state.normal_wind_advective_tendency.corrector.asnumpy(),
                    "vn_eh": diagnostic_state.vn_on_half_levels.asnumpy(),
                    "w_wcc_cf": self._contravariant_corrected_w_at_cells_on_model_levels.asnumpy(),
                },
                label=f"advection_corrector",
            )
        #<--- IBM
