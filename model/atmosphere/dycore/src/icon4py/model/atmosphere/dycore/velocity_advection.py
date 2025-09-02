# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: ERA001

from __future__ import annotations

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_horizontal_momentum_equation import (
    compute_advection_in_horizontal_momentum_equation,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_vertical_momentum_equation import (
    compute_advection_in_vertical_momentum_equation,
    compute_contravariant_correction_and_advection_in_vertical_momentum_equation,
)
from icon4py.model.atmosphere.dycore.stencils.compute_derived_horizontal_winds_and_ke_and_contravariant_correction import (
    compute_derived_horizontal_winds_and_ke_and_contravariant_correction,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import (
    horizontal as h_grid,
    icon as icon_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.model_options import setup_program
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
        backend: gtx_backend.Backend | None,
        backend_options: dict | None = None,
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

        if backend_options is None:
            backend_options = backend

        self._compute_derived_horizontal_winds_and_ke_and_contravariant_correction = setup_program(
            backend_options=backend_options,
            program=compute_derived_horizontal_winds_and_ke_and_contravariant_correction,
            constant_args={
                "rbf_vec_coeff_e": self.interpolation_state.rbf_vec_coeff_e,
                "wgtfac_e": self.metric_state.wgtfac_e,
                "ddxn_z_full": self.metric_state.ddxn_z_full,
                "ddxt_z_full": self.metric_state.ddxt_z_full,
                "wgtfacq_e": self.metric_state.wgtfacq_e,
                "c_intp": self.interpolation_state.c_intp,
                "inv_dual_edge_length": self.edge_params.inverse_dual_edge_lengths,
                "inv_primal_edge_length": self.edge_params.inverse_primal_edge_lengths,
                "tangent_orientation": self.edge_params.tangent_orientation,
            },
            variants={
                "skip_compute_predictor_vertical_advection": [True, False],
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_lateral_boundary_level_5,
                "horizontal_end": self._end_edge_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self.grid.num_levels + 1),
                "nflatlev": self.vertical_params.nflatlev,
            },
            offset_provider=self.grid.connectivities,
        )

        # TODO(nfarabullini): add `skip_compute_predictor_vertical_advection` to `variants` once possible
        self._compute_contravariant_correction_and_advection_in_vertical_momentum_equation = setup_program(
            backend_options=backend_options,
            program=compute_contravariant_correction_and_advection_in_vertical_momentum_equation,
            constant_args={
                "coeff1_dwdz": self.metric_state.coeff1_dwdz,
                "coeff2_dwdz": self.metric_state.coeff2_dwdz,
                "e_bln_c_s": self.interpolation_state.e_bln_c_s,
                "wgtfac_c": self.metric_state.wgtfac_c,
                "ddqz_z_half": self.metric_state.ddqz_z_half,
                "geofac_n2s": self.interpolation_state.geofac_n2s,
                "owner_mask": self.c_owner_mask,
            },
            vertical_sizes={
                "end_index_of_damping_layer": self.vertical_params.end_index_of_damping_layer,
                "nflatlev": self.vertical_params.nflatlev,
                "vertical_start": gtx.int32(0),
                "vertical_end": self.grid.num_levels,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_4,
                "horizontal_end": self._end_cell_halo,
            },
            offset_provider=self.grid.connectivities,
        )

        self._compute_advection_in_vertical_momentum_equation = setup_program(
            backend_options=backend_options,
            program=compute_advection_in_vertical_momentum_equation,
            constant_args={
                "coeff1_dwdz": self.metric_state.coeff1_dwdz,
                "coeff2_dwdz": self.metric_state.coeff2_dwdz,
                "c_intp": self.interpolation_state.c_intp,
                "inv_dual_edge_length": self.edge_params.inverse_dual_edge_lengths,
                "inv_primal_edge_length": self.edge_params.inverse_primal_edge_lengths,
                "tangent_orientation": self.edge_params.tangent_orientation,
                "e_bln_c_s": self.interpolation_state.e_bln_c_s,
                "ddqz_z_half": self.metric_state.ddqz_z_half,
                "geofac_n2s": self.interpolation_state.geofac_n2s,
                "owner_mask": self.c_owner_mask,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_4,
                "horizontal_end": self._end_cell_halo,
            },
            vertical_sizes={
                "end_index_of_damping_layer": self.vertical_params.end_index_of_damping_layer,
                "vertical_start": gtx.int32(0),
                "vertical_end": self.grid.num_levels,
            },
            offset_provider=self.grid.connectivities,
        )

        self._compute_advection_in_horizontal_momentum_equation = setup_program(
            backend_options=backend_options,
            program=compute_advection_in_horizontal_momentum_equation,
            constant_args={
                "e_bln_c_s": self.interpolation_state.e_bln_c_s,
                "geofac_rot": self.interpolation_state.geofac_rot,
                "coeff_gradekin": self.metric_state.coeff_gradekin,
                "c_lin_e": self.interpolation_state.c_lin_e,
                "ddqz_z_full_e": self.metric_state.ddqz_z_full_e,
                "area_edge": self.edge_params.edge_areas,
                "tangent_orientation": self.edge_params.tangent_orientation,
                "inv_primal_edge_length": self.edge_params.inverse_primal_edge_lengths,
                "geofac_grdiv": self.interpolation_state.geofac_grdiv,
                "coriolis_frequency": self.edge_params.coriolis_frequency,
            },
            variants={
                "apply_extra_diffusion_on_vn": [False, True],
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_nudging_level_2,
                "horizontal_end": self._end_edge_local,
            },
            vertical_sizes={
                "end_index_of_damping_layer": self.vertical_params.end_index_of_damping_layer,
                "vertical_start": gtx.int32(0),
                "vertical_end": self.grid.num_levels,
            },
            offset_provider=self.grid.connectivities,
        )

    def _allocate_local_fields(self):
        self._horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.zero_field(
            self.grid, dims.EdgeDim, dims.KDim, backend=self._backend, dtype=ta.vpfloat
        )
        """
        Declared as z_v_grad_w in ICON. vn dw/dn + vt dw/dt. NOTE THAT IT ONLY HAS nlev LEVELS because w[nlevp1-1] is diagnostic.
        """

        self._contravariant_corrected_w_at_cells_on_model_levels = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend, dtype=ta.vpfloat
        )
        """
        Declared as z_w_con_c_full in ICON. w - (vn dz/dn + vt dz/dt), z is topography height
        """

        self.vertical_cfl = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend, dtype=ta.vpfloat
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
        contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
        horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
        tangential_wind_on_half_levels: fa.EdgeKField[ta.vpfloat],
        dtime: ta.wpfloat,
        cell_areas: fa.CellField[ta.wpfloat],
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

        self._compute_derived_horizontal_winds_and_ke_and_contravariant_correction(
            tangential_wind=diagnostic_state.tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=self._horizontal_advection_of_w_at_edges_on_half_levels,
            vn=prognostic_state.vn,
            w=prognostic_state.w,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        )

        # TODO(havogt): however, our test data is probably not able to catch cfl_clipping conditons
        self._compute_contravariant_correction_and_advection_in_vertical_momentum_equation(
            contravariant_correction_at_cells_on_half_levels=diagnostic_state.contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.predictor,
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl=self.vertical_cfl,
            w=prognostic_state.w,
            horizontal_advection_of_w_at_edges_on_half_levels=self._horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            area=cell_areas,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        )

        max_vertical_cfl = float(
            self.vertical_cfl.array_ns.max(
                self.vertical_cfl.ndarray[
                    self._start_cell_lateral_boundary_level_4 : self._end_cell_halo, :
                ]
            )
        )
        diagnostic_state.max_vertical_cfl = max(max_vertical_cfl, diagnostic_state.max_vertical_cfl)
        apply_extra_diffusion_on_vn = max_vertical_cfl > cfl_w_limit * dtime
        self._compute_advection_in_horizontal_momentum_equation(
            normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.predictor,
            vn=prognostic_state.vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            tangential_wind=diagnostic_state.tangential_wind,
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        )

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
        tangential_wind_on_half_levels: fa.EdgeKField[ta.vpfloat],
        dtime: ta.wpfloat,
        cell_areas: fa.CellField[ta.wpfloat],
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

        self._compute_advection_in_vertical_momentum_equation(
            vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.corrector,
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl=self.vertical_cfl,
            w=prognostic_state.w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state.contravariant_correction_at_cells_on_half_levels,
            area=cell_areas,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        )

        max_vertical_cfl = float(
            self.vertical_cfl.array_ns.max(
                self.vertical_cfl.ndarray[
                    self._start_cell_lateral_boundary_level_4 : self._end_cell_halo, :
                ]
            )
        )
        diagnostic_state.max_vertical_cfl = max(max_vertical_cfl, diagnostic_state.max_vertical_cfl)
        apply_extra_diffusion_on_vn = max_vertical_cfl > cfl_w_limit * dtime
        self._compute_advection_in_horizontal_momentum_equation(
            normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.corrector,
            vn=prognostic_state.vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            tangential_wind=diagnostic_state.tangential_wind,
            contravariant_corrected_w_at_cells_on_model_levels=self._contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=diagnostic_state.vn_on_half_levels,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        )
