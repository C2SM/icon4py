# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The diagnostics stage of the surface scheme."""

from __future__ import annotations

import logging
import typing

from icon4py.model.atmosphere.subgrid_scale_physics.surface.stencils import diagnostics_stencils
from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.model_options import setup_program


if typing.TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.surface import surface_states
    from icon4py.model.common import field_type_aliases as fa, model_backends
    from icon4py.model.common.grid import base as base_grid


log = logging.getLogger(__name__)


class Diagnostics:
    """
    ``Compute_diagnostics`` of ``t_vdf_sfc`` (mo_vdf_sfc.f90) for the sea surface.

    The surface-layer diagnostics, the roughness length and the Louis exchange coefficients,
    in ICON's order. The roughness length reads the momentum exchange coefficient of the
    previous step from the diagnostic state, which therefore persists between steps.
    """

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        metric_state: surface_states.SurfaceMetricState,
        surface_temperature: fa.CellField[ta.wpfloat],
        backend: model_backends.BackendLike,
        minimum_surface_wind: float,
        surface_layer_fraction: float,
        ocean_roughness_length: float,
        minimum_roughness_length: float,
    ) -> None:
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_sizes = {
            "horizontal_start": grid.start_index(cell_domain(h_grid.Zone.NUDGING)),
            "horizontal_end": grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
        }
        grav = ta.wpfloat(constants.GRAV)
        self._compute_surface_layer_diagnostics = setup_program(
            backend=backend,
            program=diagnostics_stencils.compute_surface_layer_diagnostics,
            constant_args={
                "surface_temperature": surface_temperature,
                "z_mc": metric_state.z_mc,
                "minimum_surface_wind": ta.wpfloat(minimum_surface_wind),
                "surface_layer_fraction": ta.wpfloat(surface_layer_fraction),
                "grav": grav,
            },
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self._compute_sea_surface_roughness = setup_program(
            backend=backend,
            program=diagnostics_stencils.compute_sea_surface_roughness,
            constant_args={
                "ocean_roughness_length": ta.wpfloat(ocean_roughness_length),
                "minimum_roughness_length": ta.wpfloat(minimum_roughness_length),
                "grav": grav,
            },
            variants={"is_initial_time": [True, False]},
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )
        self._compute_surface_exchange_coefficients = setup_program(
            backend=backend,
            program=diagnostics_stencils.compute_surface_exchange_coefficients,
            constant_args={"ddqz_z_half": metric_state.ddqz_z_half, "grav": grav},
            horizontal_sizes=horizontal_sizes,
            offset_provider={},
        )

    def run(
        self,
        *,
        input_state: surface_states.SurfaceInputState,
        diagnostic_state: surface_states.SurfaceDiagnosticState,
        is_initial_time: bool,
    ) -> None:
        """
        Compute the surface diagnostics of the step into `diagnostic_state`.

        `is_initial_time` is ICON's ``l_init .AND. .NOT. isrestart()``: the first step takes
        the configured ocean roughness length, later steps the Charnock relation.
        """
        log.debug("surface diagnostics (Compute_diagnostics): start")
        self._compute_surface_layer_diagnostics(
            u=input_state.u,
            v=input_state.v,
            temperature=input_state.temperature,
            virtual_temperature=input_state.virtual_temperature,
            pressure=input_state.pressure,
            surface_pressure=input_state.surface_pressure,
            wind_speed=diagnostic_state.wind_speed,
            potential_temperature=diagnostic_state.potential_temperature,
            virtual_potential_temperature=diagnostic_state.virtual_potential_temperature,
            saturation_specific_humidity=diagnostic_state.saturation_specific_humidity,
            density=diagnostic_state.density,
            surface_potential_temperature=diagnostic_state.surface_potential_temperature,
            surface_virtual_potential_temperature=diagnostic_state.surface_virtual_potential_temperature,
            richardson_number=diagnostic_state.richardson_number,
        )
        # the roughness length uses the exchange coefficient of the previous step, before the
        # new one is computed ("Uses old value of km_tile", mo_vdf_sfc.f90)
        self._compute_sea_surface_roughness(
            wind_speed=diagnostic_state.wind_speed,
            exchange_coefficient_momentum=diagnostic_state.exchange_coefficient_momentum,
            is_initial_time=is_initial_time,
            roughness_length_momentum=diagnostic_state.roughness_length_momentum,
            roughness_length_heat=diagnostic_state.roughness_length_heat,
        )
        self._compute_surface_exchange_coefficients(
            specific_humidity=input_state.specific_humidity,
            potential_temperature=diagnostic_state.potential_temperature,
            wind_speed=diagnostic_state.wind_speed,
            roughness_length_momentum=diagnostic_state.roughness_length_momentum,
            surface_potential_temperature=diagnostic_state.surface_potential_temperature,
            saturation_specific_humidity=diagnostic_state.saturation_specific_humidity,
            exchange_coefficient_momentum=diagnostic_state.exchange_coefficient_momentum,
            exchange_coefficient_heat=diagnostic_state.exchange_coefficient_heat,
        )
        log.debug("surface diagnostics (Compute_diagnostics): end")
