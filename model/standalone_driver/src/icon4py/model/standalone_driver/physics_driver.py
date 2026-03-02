# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.diagnostic_calculations import stencils as diagnostic_stencils
from icon4py.model.common.grid import (
    geometry_attributes as geo_attrs,
    horizontal as h_grid,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes as interp_attrs
from icon4py.model.common.interpolation.stencils import compute_edge_2_cell_vector_interpolation
from icon4py.model.common.metrics import metrics_attributes as metric_attrs
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_states


log = logging.getLogger(__name__)


class PhysicsDriver:
    def __init__(
        self,
        config: driver_config.DriverConfig,
        static_field_factories: driver_states.StaticFieldFactories,
        vertical_params: v_grid.VerticalGrid,
        grid: icon_grid.IconGrid,
        saturation_adjustment: satad.SaturationAdjustment,
        microphysics: graupel.SingleMomentSixClassIconGraupel,
        backend: model_backends.BackendLike,
    ):
        self.config = config
        self.vertical_params = vertical_params
        self.grid = grid
        self.static_field_factories = static_field_factories
        self.saturation_adjustment = saturation_adjustment
        self.microphysics = microphysics
        self.backend = backend
        self.allocator = model_backends.get_allocator(backend)

    def _local_fields(self, grid: icon_grid.IconGrid):
        self.temperature_tendency = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
        )
        self.u_tendency = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
        )
        self.v_tendency = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
        )
        self.tracer_tendency = tracer_state.TracerStateTendency(
            qv_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
            ),
            qc_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
            ),
            qi_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
            ),
            qr_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
            ),
            qs_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
            ),
            qg_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self.allocator
            ),
        )

    def __call__(
        self,
        diagnostic_state: diagnostics.DiagnosticState,
        prognostic_state: common_utils.TimeStepPair[prognostics.PrognosticState],
        tracers: tracer_state.TracerState,
        perturbed_exner: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ):
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        start_cell_nudging_level = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        start_edge_nudging_level = self.grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        end_edge_local = self.grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        saved_exner = data_alloc.as_field(prognostic_state.current.exner, allocator=self.allocator)

        compute_edge_2_cell_vector_interpolation.compute_edge_2_cell_vector_interpolation.with_backend(
            self.backend
        )(
            p_e_in=prognostic_state.current.vn,
            ptr_coeff_1=self.static_field_factories.interpolation_field_source.get(
                interp_attrs.RBF_VEC_COEFF_C1
            ),
            ptr_coeff_2=self.static_field_factories.interpolation_field_source.get(
                interp_attrs.RBF_VEC_COEFF_C2
            ),
            p_u_out=diagnostic_state.u,
            p_v_out=diagnostic_state.v,
            horizontal_start=1,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.connectivities,
        )  # 1: min_rlcell_int

        diagnostic_stencils.diagnose_virtual_temperature_and_temperature_from_exner.with_backend(
            self.backend
        )(
            virtual_temperature=diagnostic_state.virtual_temperature,
            temperature=diagnostic_state.temperature,
            tracers=tracers,
            theta_v=prognostic_state.current.theta_v,
            exner=prognostic_state.current.exner,
            horizontal_start=start_cell_nudging_level,
            horizontal_end=end_cell_local,
            vertical_start=self.vertical_params.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        # data_alloc.field(saved_exner)  # , 1: min_rlcell

        self.saturation_adjustment.run(
            temperature_tendency=self.temperature_tendency,
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            rho=prognostic_state.current.rho,
            temperature=diagnostic_state.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            dtime=dtime,
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int

        diagnostic_stencils.diagnose_virtual_temperature_and_exner.with_backend(self.backend)(
            virtual_temperature=diagnostic_state.virtual_temperature,
            exner=prognostic_state.current.exner,
            tracers=tracers,
            temperature=diagnostic_state.temperature,
            horizontal_start=start_cell_nudging_level,
            horizontal_end=end_cell_local,
            vertical_start=self.vertical_params.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnostic_stencils.diagnose_surface_pressure.with_backend(self.backend)(
            surface_pressure=diagnostic_state.pressure_at_half_levels,
            exner=prognostic_state.current.exner,
            virtual_temperature=diagnostic_state.virtual_temperature,
            ddqz_z_full=self.static_field_factories.metrics_field_source.get(
                metric_attrs.DDQZ_Z_FULL
            ),
            horizontal_start=start_cell_nudging_level,
            horizontal_end=end_cell_local,
            vertical_start=self.grid.num_levels,
            vertical_end=gtx.int32(self.grid.num_levels + 1),
            offset_provider={"Koff": dims.KDim},
        )

        diagnostic_stencils.diagnose_pressure.with_backend(self.backend)(
            pressure=diagnostic_state.pressure,
            pressure_at_half_levels=diagnostic_state.pressure_at_half_levels,
            surface_pressure=diagnostic_state.surface_pressure,
            virtual_temperature=diagnostic_state.virtual_temperature,
            ddqz_z_full=self.static_field_factories.metrics_field_source.get(
                metric_attrs.DDQZ_Z_FULL
            ),
            horizontal_start=start_cell_nudging_level,
            horizontal_end=end_cell_local,
            vertical_start=self.vertical_params.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int

        # simple_surface()  # only for qv_s, grf_bdywidth_c+1: min_rlcell_int
        # turbulence() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED
        self.microphysics.run(
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            qi_tendency=self.tracer_tendency.qi_tendency,
            qr_tendency=self.tracer_tendency.qr_tendency,
            qs_tendency=self.tracer_tendency.qs_tendency,
            qg_tendency=self.tracer_tendency.qg_tendency,
            temperature_tendency=self.temperature_tendency,
            qv=tracers.qv,
            qc=tracers.qc,
            qi=tracers.qi,
            qr=tracers.qr,
            qs=tracers.qs,
            qg=tracers.qg,
            qnc=tracers.qnc,
            rho=prognostic_state.current.rho,
            temperature=diagnostic_state.temperature,
            pressure=diagnostic_state.pressure,
            dtime=dtime,
        )

        self.saturation_adjustment.run(
            temperature_tendency=self.temperature_tendency,
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            rho=prognostic_state.current.rho,
            temperature=diagnostic_state.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            dtime=dtime,
        )

        diagnostic_stencils.diagnose_exner_and_theta_v_from_virtual_temperature.with_backend(
            self.backend
        )(
            virtual_temperature=diagnostic_state.virtual_temperature,
            exner=prognostic_state.current.exner,
            perturbed_exner=perturbed_exner,
            theta_v=prognostic_state.current.theta_v,
            tracers=tracers,
            temperature=diagnostic_state.temperature,
            rho=prognostic_state.current.rho,
            previous_exner=saved_exner,
            horizontal_start=start_cell_nudging_level,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # TODO (Chia Rui): add diagnose_pressure() here when turbulence is ready # grf_bdywidth_c+1: min_rlcell_int
        # TODO (Chia Rui): surface_transfer() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED YET

        # halo exchange, including ddt_u_turb and ddt_v_turb
        diagnose_exner_and_theta_v()  # min_rlcell_int-1: min_rlcell_int
        diagnostic_stencils.update_vn_from_u_v_tendencies.with_backend(self.backend)(
            vn=prognostic_state.current.vn,
            u_tendency=self.u_tendency,
            v_tendency=self.v_tendency,
            dt=dtime,
            c_lin_e=self.static_field_factories.interpolation_field_source.get(
                interp_attrs.C_LIN_E
            ),
            primal_normal_cell_x=self.static_field_factories.geometry_field_source.get(
                geo_attrs.EDGE_NORMAL_CELL_U
            ),
            primal_normal_cell_y=self.static_field_factories.geometry_field_source.get(
                geo_attrs.EDGE_NORMAL_CELL_V
            ),
            horizontal_start=start_edge_nudging_level,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.connectivities,
        )  # grf_bdywidth_e+1: min_rledge_int
