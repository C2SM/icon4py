# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import icon4py.model.common.utils as common_utils
from icon4py.model.common.states import diagnostic_state as diagnostics, prognostic_state as prognostics, tracer_state as tracers
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import icon as icon_grid, horizontal as h_grid, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common.interpolation import interpolation_attributes as interp_attrs
from icon4py.model.standalone_driver import config as driver_config
from icon4py.model.standalone_driver import driver_states
from icon4py.model.common.interpolation.stencils import compute_edge_2_cell_vector_interpolation
from icon4py.model.common.diagnostic_calculations import stencils as diagnostic_stencils
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import single_moment_six_class_gscp_graupel as graupel, saturation_adjustment as satad


log = logging.getLogger(__name__)


class PhysicsDriver:
    def __init__(
        self,
        config: driver_config.DriverConfig,
        static_field_factories: driver_states.StaticFieldFactories,
        vertical_params: v_grid.VerticalGrid,
        grid: icon_grid.IconGrid,
        saturation_adjustment: 
        microphysics
        backend,
    ):
        self.config = config
        self.vertical_params = vertical_params
        self.grid = grid
        self.static_field_factories = static_field_factories
        self.saturation_adjustment = 
        self.microphysics = 
        self.backend = backend

    def _local_fields(self, grid: icon_grid.IconGrid):
        saved_exner = data_alloc.allocate_cell_field(
            grid,
            dims.EdgeDim,
            dims.KDim,
            dtype=ta.wpfloat,
            name="saved_exner",
        )

    def __call__(self, diagnostic_state: diagnostics.DiagnosticState, prognostic_state: common_utils.TimeStepPair[prognostics.PrognosticState], tracer_state: tracers.TracerState):
        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_nudging_level = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        compute_edge_2_cell_vector_interpolation.compute_edge_2_cell_vector_interpolation.with_backend(self.backend)(
            p_e_in=prognostic_state.current.vn,
            ptr_coeff_1=self.static_field_factories.interpolation_field_source.get(interp_attrs.RBF_VEC_COEFF_C1),
            ptr_coeff_2=self.static_field_factories.interpolation_field_source.get(interp_attrs.RBF_VEC_COEFF_C2),
            p_u_out=diagnostic_state.u,
            p_v_out=diagnostic_state.v,
            horizontal_start=1,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.connectivities,
        )  # 1: min_rlcell_int
        
        diagnostic_stencils.diagnose_virtual_temperature_and_temperature_from_exner.with_backend(self.backend)(
            tracer_state,
            prognostic_state.current.theta_v,
            prognostic_state.current.exner,
            diagnostic_state.virtual_temperature,
            diagnostic_state.temperature,
            horizontal_start=start_cell_nudging_level,
            horizontal_end=end_cell_local,
            vertical_start=self.vertical_params.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        # data_alloc.field(saved_exner)  # , 1: min_rlcell
        
        saturation_adjusment()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnose_exner_and_virtual_temperature()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnose_pressure()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        simple_surface()  # only for qv_s, grf_bdywidth_c+1: min_rlcell_int
        # turbulence() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED
        microphysics()
        saturation_adjustment()
        # diagnose_virtual_temperature_and_exner_and_theta_v() # grf_bdywidth_c+1: min_rlcell_int
        # diagnose_pressure() # grf_bdywidth_c+1: min_rlcell_int
        # surface_transfer() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED
        # halo exchange ddt_u_turb and ddt_v_turb
        diagnose_exner_and_theta_v()  # min_rlcell_int-1: min_rlcell_int
        update_vn_from_turb()  # grf_bdywidth_e+1: min_rledge_int
