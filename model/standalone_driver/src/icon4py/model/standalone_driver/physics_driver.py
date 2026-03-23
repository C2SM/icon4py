# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import types

import gt4py.next as gtx

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
from icon4py.model.common.decomposition import definitions as decomposition
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
from icon4py.model.common.states import diagnostic_state, prognostic_state, tracer_state
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_states


log = logging.getLogger(__name__)


class PhysicsDriver:
    def __init__(
        self,
        grid: icon_grid.IconGrid,
        vertical_grid: v_grid.VerticalGrid,
        static_field_factories: driver_states.StaticFieldFactories,
        saturation_adjustment: satad.SaturationAdjustment,
        microphysics: graupel.SingleMomentSixClassIconGraupel,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime = decomposition.single_node_default,
    ):
        self.vertical_grid = vertical_grid
        self.grid = grid
        self.static_field_factories = static_field_factories
        self.saturation_adjustment = saturation_adjustment
        self.microphysics = microphysics
        self._exchange = exchange
        self.backend = backend

    @functools.cached_property
    def _allocator(self) -> gtx.typing.Backend:
        return model_backends.get_allocator(self.backend)

    @functools.cached_property
    def _xp(self) -> types.ModuleType:
        return data_alloc.import_array_ns(self._allocator)

    def _local_fields(self) -> None:
        self.temperature_tendency = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
        )
        self.u_tendency = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
        )
        self.v_tendency = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
        )
        self.tracer_tendency = tracer_state.TracerStateTendency(
            qv_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            ),
            qc_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            ),
            qi_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            ),
            qr_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            ),
            qs_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            ),
            qg_tendency=data_alloc.zero_field(
                self.grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._allocator
            ),
        )

    def __call__(
        self,
        prognostic: prognostic_state.PrognosticState,
        diagnostic: diagnostic_state.DiagnosticState,
        tracers: tracer_state.TracerState,
        perturbed_exner: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        saved_exner = data_alloc.as_field(
            prognostic.exner, allocator=self._allocator
        )  # saved_exner, 1: min_rlcell

        compute_edge_2_cell_vector_interpolation.compute_edge_2_cell_vector_interpolation.with_backend(
            self.backend
        )(
            p_e_in=prognostic.vn,
            ptr_coeff_1=self.static_field_factories.interpolation_field_source.get(
                interp_attrs.RBF_VEC_COEFF_C1
            ),
            ptr_coeff_2=self.static_field_factories.interpolation_field_source.get(
                interp_attrs.RBF_VEC_COEFF_C2
            ),
            p_u_out=diagnostic.u,
            p_v_out=diagnostic.v,
            horizontal_start=1,
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.connectivities,
        )  # 1: min_rlcell_int

        diagnostic_stencils.diagnose_virtual_temperature_and_temperature_from_exner.with_backend(
            self.backend
        )(
            virtual_temperature=diagnostic.virtual_temperature,
            temperature=diagnostic.temperature,
            tracers=tracers,
            theta_v=prognostic.theta_v,
            exner=prognostic.exner,
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.vertical_grid.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int

        self.saturation_adjustment.run(
            temperature_tendency=self.temperature_tendency,
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            rho=prognostic.rho,
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            dtime=dtime,
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnostic_stencils.update_satad_output_from_tendency.with_backend(self.backend)(
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            temperature_tendency=self.temperature_tendency,
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            dtime=dtime,
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.vertical_grid.kstart_moist,
            vertical_end=self.grid.num_levels,
        )

        diagnostic_stencils.diagnose_virtual_temperature_and_exner.with_backend(self.backend)(
            virtual_temperature=diagnostic.virtual_temperature,
            exner=prognostic.exner,
            tracers=tracers,
            temperature=diagnostic.temperature,
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.vertical_grid.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnostic_stencils.diagnose_surface_pressure.with_backend(self.backend)(
            surface_pressure=diagnostic.pressure_at_half_levels,
            exner=prognostic.exner,
            virtual_temperature=diagnostic.virtual_temperature,
            ddqz_z_full=self.static_field_factories.metrics_field_source.get(
                metric_attrs.DDQZ_Z_FULL
            ),
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.grid.num_levels,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={"Koff": dims.KDim},
        )

        diagnostic_stencils.diagnose_pressure.with_backend(self.backend)(
            pressure=diagnostic.pressure,
            pressure_at_half_levels=diagnostic.pressure_at_half_levels,
            surface_pressure=diagnostic.surface_pressure,
            virtual_temperature=diagnostic.virtual_temperature,
            ddqz_z_full=self.static_field_factories.metrics_field_source.get(
                metric_attrs.DDQZ_Z_FULL
            ),
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.vertical_grid.kstart_moist,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int

        # TODO (Chia Rui): simple_surface()  # only for qv_s, grf_bdywidth_c+1: min_rlcell_int
        # TODO (Chia Rui): turbulence() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED

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
            rho=prognostic.rho,
            temperature=diagnostic.temperature,
            pressure=diagnostic.pressure,
            dtime=dtime,
        )
        diagnostic_stencils.update_microphysics_output_from_tendency.with_backend(self.backend)(
            temperature=diagnostic.temperature,
            tracers=tracers,
            temperature_tendency=self.temperature_tendency,
            tracer_tendency=self.tracer_tendency,
            dtime=dtime,
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.vertical_grid.kstart_moist,
            vertical_end=self.grid.num_levels,
        )

        self.saturation_adjustment.run(
            temperature_tendency=self.temperature_tendency,
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            rho=prognostic.rho,
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            dtime=dtime,
        )
        diagnostic_stencils.update_satad_output_from_tendency.with_backend(self.backend)(
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            temperature_tendency=self.temperature_tendency,
            qv_tendency=self.tracer_tendency.qv_tendency,
            qc_tendency=self.tracer_tendency.qc_tendency,
            dtime=dtime,
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=self.vertical_grid.kstart_moist,
            vertical_end=self.grid.num_levels,
        )

        diagnostic_stencils.diagnose_exner_and_theta_v_from_virtual_temperature.with_backend(
            self.backend
        )(
            virtual_temperature=diagnostic.virtual_temperature,
            exner=prognostic.exner,
            perturbed_exner=perturbed_exner,
            theta_v=prognostic.theta_v,
            tracers=tracers,
            temperature=diagnostic.temperature,
            rho=prognostic.rho,
            previous_exner=saved_exner,
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.LOCAL],
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # TODO (Chia Rui): add diagnose_pressure() here when turbulence is ready # grf_bdywidth_c+1: min_rlcell_int
        # TODO (Chia Rui): surface_transfer() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED YET

        # TODO (Chia Rui): (and w if diffusion is applied to w)
        self._exchange.exchange(
            dims.CellDim,
            diagnostic.virtual_temperature,
            perturbed_exner,
            tracers.qv,
            tracers.qc,
            tracers.qr,
            tracers.qi,
            tracers.qs,
            tracers.qg,
            stream=decomposition.DEFAULT_STREAM,
        )
        # TODO (Chia Rui): halo exchange, including ddt_u_turb and ddt_v_turb
        diagnostic_stencils.update_exner_and_theta_v_from_virtual_temperature_in_halo.with_backend(
            self.backend
        )(
            exner=prognostic.exner,
            theta_v=prognostic.theta_v,
            rho=prognostic.rho,
            virtual_temperature=diagnostic.virtual_temperature,
            mask_prog_halo_c=self.static_field_factories.metrics_field_source.get(
                metric_attrs.MASK_PROG_HALO_C
            ),
            horizontal_start=self.grid.cell_start_index[h_grid.Zone.HALO],
            horizontal_end=self.grid.cell_end_index[h_grid.Zone.END],
            vertical_start=0,
            vertical_end=self.grid.num_levels,
        )  # min_rlcell_int-1: min_rlcell_int
        diagnostic_stencils.update_vn_from_u_v_tendencies.with_backend(self.backend)(
            vn=prognostic.vn,
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
            horizontal_start=self.grid.edge_start_index[h_grid.Zone.NUDGING],
            horizontal_end=self.grid.edge_end_index[h_grid.Zone.LOCAL],
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.connectivities,
        )  # grf_bdywidth_e+1: min_rledge_int


def initialize_physics_driver(
    grid: icon_grid.IconGrid,
    vertical_grid: v_grid.VerticalGrid,
    static_field_factories: driver_states.StaticFieldFactories,
    backend: model_backends.BackendLike,
) -> PhysicsDriver:
    saturation_adjustment = satad.SaturationAdjustment(
        config=satad.SaturationAdjustmentConfig(),
        grid=grid,
        vertical_params=vertical_grid,
        metric_state=satad.MetricStateSaturationAdjustment(
            ddqz_z_full=static_field_factories.metrics_field_source.get(metric_attrs.DDQZ_Z_FULL)
        ),
        backend=backend,
    )
    microphysics = graupel.SingleMomentSixClassIconGraupel(
        config=graupel.SingleMomentSixClassIconGraupelConfig(),
        grid=grid,
        vertical_params=vertical_grid,
        metric_state=graupel.MetricStateIconGraupel(
            ddqz_z_full=static_field_factories.metrics_field_source.get(metric_attrs.DDQZ_Z_FULL)
        ),
        backend=backend,
    )
    physics_driver = PhysicsDriver(
        grid=grid,
        vertical_grid=vertical_grid,
        static_field_factories=static_field_factories,
        saturation_adjustment=saturation_adjustment,
        microphysics=microphysics,
        backend=backend,
    )
    return physics_driver
