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
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
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

        self.tendencies: driver_states.TendencyState = driver_states.TendencyState.zero_field(
            self.grid, self._allocator
        )
        self._setup_gt4py_programs()

    @functools.cached_property
    def _allocator(self) -> gtx_typing.Backend:
        return model_backends.get_allocator(self.backend)

    @functools.cached_property
    def _xp(self) -> types.ModuleType:
        return data_alloc.import_array_ns(self._allocator)

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

        self._compute_edge_2_cell_vector_interpolation(
            p_e_in=prognostic.vn,
            p_u_out=diagnostic.u,
            p_v_out=diagnostic.v,
        )  # 1: min_rlcell_int

        self._diagnose_virtual_temperature_and_temperature_from_exner(
            virtual_temperature=diagnostic.virtual_temperature,
            temperature=diagnostic.temperature,
            tracers=tracers,
            theta_v=prognostic.theta_v,
            exner=prognostic.exner,
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int

        self.saturation_adjustment.run(
            temperature_tendency=self.tendencies.temperature_tendency,
            qv_tendency=self.tendencies.tracer_tendency.qv_tendency,
            qc_tendency=self.tendencies.tracer_tendency.qc_tendency,
            rho=prognostic.rho,
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            dtime=dtime,
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        self._update_satad_output_from_tendency(
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            temperature_tendency=self.tendencies.temperature_tendency,
            qv_tendency=self.tendencies.tracer_tendency.qv_tendency,
            qc_tendency=self.tendencies.tracer_tendency.qc_tendency,
            dtime=dtime,
        )

        self._diagnose_virtual_temperature_and_exner(
            virtual_temperature=diagnostic.virtual_temperature,
            exner=prognostic.exner,
            tracers=tracers,
            temperature=diagnostic.temperature,
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        self._diagnose_surface_pressure(
            surface_pressure=diagnostic.pressure_at_half_levels,
            exner=prognostic.exner,
            virtual_temperature=diagnostic.virtual_temperature,
        )

        self._diagnose_pressure(
            pressure=diagnostic.pressure,
            pressure_at_half_levels=diagnostic.pressure_at_half_levels,
            surface_pressure=diagnostic.surface_pressure,
            virtual_temperature=diagnostic.virtual_temperature,
        )  # from kmoist, grf_bdywidth_c+1: min_rlcell_int

        # TODO (Chia Rui): simple_surface()  # only for qv_s, grf_bdywidth_c+1: min_rlcell_int
        # TODO (Chia Rui): turbulence() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED

        self.microphysics.run(
            qv_tendency=self.tendencies.tracer_tendency.qv_tendency,
            qc_tendency=self.tendencies.tracer_tendency.qc_tendency,
            qi_tendency=self.tendencies.tracer_tendency.qi_tendency,
            qr_tendency=self.tendencies.tracer_tendency.qr_tendency,
            qs_tendency=self.tendencies.tracer_tendency.qs_tendency,
            qg_tendency=self.tendencies.tracer_tendency.qg_tendency,
            temperature_tendency=self.tendencies.temperature_tendency,
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
        self._update_microphysics_output_from_tendency(
            temperature=diagnostic.temperature,
            tracers=tracers,
            temperature_tendency=self.tendencies.temperature_tendency,
            tracer_tendency=self.tendencies.tracer_tendency,
            dtime=dtime,
        )

        self.saturation_adjustment.run(
            temperature_tendency=self.tendencies.temperature_tendency,
            qv_tendency=self.tendencies.tracer_tendency.qv_tendency,
            qc_tendency=self.tendencies.tracer_tendency.qc_tendency,
            rho=prognostic.rho,
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            dtime=dtime,
        )
        self._update_satad_output_from_tendency(
            temperature=diagnostic.temperature,
            qv=tracers.qv,
            qc=tracers.qc,
            temperature_tendency=self.tendencies.temperature_tendency,
            qv_tendency=self.tendencies.tracer_tendency.qv_tendency,
            qc_tendency=self.tendencies.tracer_tendency.qc_tendency,
            dtime=dtime,
        )

        self._diagnose_exner_and_theta_v_from_virtual_temperature(
            virtual_temperature=diagnostic.virtual_temperature,
            exner=prognostic.exner,
            perturbed_exner=perturbed_exner,
            theta_v=prognostic.theta_v,
            tracers=tracers,
            temperature=diagnostic.temperature,
            rho=prognostic.rho,
            previous_exner=saved_exner,
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
        self._update_exner_and_theta_v_from_virtual_temperature_in_halo(
            exner=prognostic.exner,
            theta_v=prognostic.theta_v,
            rho=prognostic.rho,
            virtual_temperature=diagnostic.virtual_temperature,
        )  # min_rlcell_int-1: min_rlcell_int
        self._update_vn_from_u_v_tendencies(
            vn=prognostic.vn,
            u_tendency=self.tendencies.u_tendency,
            v_tendency=self.tendencies.v_tendency,
            dt=dtime,
        )  # grf_bdywidth_e+1: min_rledge_int

    def _setup_gt4py_programs(self) -> None:
        self._compute_edge_2_cell_vector_interpolation = model_options.setup_program(
            backend=self.backend,
            program=compute_edge_2_cell_vector_interpolation.compute_edge_2_cell_vector_interpolation,
            constant_args={
                "ptr_coeff_1": self.static_field_factories.interpolation_field_source.get(
                    interp_attrs.RBF_VEC_COEFF_C1
                ),
                "ptr_coeff_2": self.static_field_factories.interpolation_field_source.get(
                    interp_attrs.RBF_VEC_COEFF_C2
                ),
            },
            horizontal_sizes={
                "horizontal_start": gtx.int32(1),
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
            offset_provider=self.grid.connectivities,
        )
        self._diagnose_virtual_temperature_and_temperature_from_exner = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.diagnose_virtual_temperature_and_temperature_from_exner,
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": self.vertical_grid.kstart_moist,
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
        )
        self._update_satad_output_from_tendency = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.update_satad_output_from_tendency,
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": self.vertical_grid.kstart_moist,
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
        )
        self._diagnose_virtual_temperature_and_exner = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.diagnose_virtual_temperature_and_exner,
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": self.vertical_grid.kstart_moist,
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
        )
        self._diagnose_surface_pressure = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.diagnose_surface_pressure,
            constant_args={
                "ddqz_z_full": self.static_field_factories.metrics_field_source.get(
                    metric_attrs.DDQZ_Z_FULL
                ),
            },
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": gtx.int32(self.grid.num_levels),
                "vertical_end": gtx.int32(self.grid.num_levels + 1),
            },
            offset_provider={"Koff": dims.KDim},
        )
        self._diagnose_pressure = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.diagnose_pressure,
            constant_args={
                "ddqz_z_full": self.static_field_factories.metrics_field_source.get(
                    metric_attrs.DDQZ_Z_FULL
                ),
            },
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": self.vertical_grid.kstart_moist,
                "vertical_end": gtx.int32(self.grid.num_levels + 1),
            },
        )
        self._update_microphysics_output_from_tendency = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.update_microphysics_output_from_tendency,
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": self.vertical_grid.kstart_moist,
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
        )
        self._diagnose_exner_and_theta_v_from_virtual_temperature = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.diagnose_exner_and_theta_v_from_virtual_temperature,
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
        )
        self._update_exner_and_theta_v_from_virtual_temperature_in_halo = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.update_exner_and_theta_v_from_virtual_temperature_in_halo,
            constant_args={
                "mask_prog_halo_c": self.static_field_factories.metrics_field_source.get(
                    metric_attrs.MASK_PROG_HALO_C
                ),
            },
            horizontal_sizes={
                "horizontal_start": self.grid.cell_start_index[h_grid.Zone.HALO],
                "horizontal_end": self.grid.cell_end_index[h_grid.Zone.END],
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
        )
        self._update_vn_from_u_v_tendencies = model_options.setup_program(
            backend=self.backend,
            program=diagnostic_stencils.update_vn_from_u_v_tendencies,
            constant_args={
                "c_lin_e": self.static_field_factories.interpolation_field_source.get(
                    interp_attrs.C_LIN_E
                ),
                "primal_normal_cell_x": self.static_field_factories.geometry_field_source.get(
                    geo_attrs.EDGE_NORMAL_CELL_U
                ),
                "primal_normal_cell_y": self.static_field_factories.geometry_field_source.get(
                    geo_attrs.EDGE_NORMAL_CELL_V
                ),
            },
            horizontal_sizes={
                "horizontal_start": self.grid.edge_start_index[h_grid.Zone.NUDGING],
                "horizontal_end": self.grid.edge_end_index[h_grid.Zone.LOCAL],
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
            offset_provider=self.grid.connectivities,
        )


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
