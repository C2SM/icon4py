# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The surface scheme as a typed component."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from icon4py.model.atmosphere.subgrid_scale_physics.surface import surface, surface_states
from icon4py.model.common import dimension as dims, field, framework as fw
from icon4py.model.common.framework import quantities as qty
from icon4py.model.common.states import surface_flux_state


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    # a type-only import: the surface scheme does not depend on the physics driver
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
    from icon4py.model.atmosphere.subgrid_scale_physics.surface import config as surface_config
    from icon4py.model.common.grid import base as base_grid


class Surface(fw.Component["Surface.Input", "Surface.Output"]):
    """
    The surface scheme: the surface fluxes and the surface temperature from the lowest level of
    the atmosphere. It produces no tendencies.
    """

    class Input(fw.State):
        pressure_ifc: field.Field[qty.PressureOnCellKHalf]
        temperature: field.Field[qty.TemperatureOnCellK]

    class Output(fw.State):
        evapotranspiration: field.Field[qty.EvapotranspirationOnCell]
        """Declared as evap in ICON."""
        sensible_heat_flux: field.Field[qty.SensibleHeatFluxOnCell]
        """Declared as shflx in ICON."""
        latent_heat_flux: field.Field[qty.LatentHeatFluxOnCell]
        """Declared as lhflx in ICON."""
        u_stress: field.Field[qty.UStressOnCell]
        v_stress: field.Field[qty.VStressOnCell]
        q_snocpymlt: field.Field[qty.SnowCanopyMeltHeatOnCell]
        thermal_energy_flux: field.Field[qty.ThermalEnergyFluxOnCell]
        """Declared as ufts in ICON."""
        vapor_energy_flux: field.Field[qty.VaporEnergyFluxOnCell]
        """Declared as ufvs in ICON."""
        surface_temperature: field.Field[qty.SurfaceTemperatureOnCell]
        """Declared as ts in ICON."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: surface_config.SurfaceConfig,
        backend: gtx_typing.Backend | None = None,
        compute: Callable[..., None] | None = None,
    ) -> None:
        self._surface = surface.Surface(grid=grid, config=config, backend=backend)
        self._compute = self._surface.run if compute is None else compute
        self._lowest_level = dims.KDim(grid.num_levels - 1)
        self._surface_level = dims.KHalfDim(grid.num_levels)

    def run(self, input: Input, output: Output) -> Output:
        """One step of the surface scheme, from the surface pressure and the lowest-level air temperature."""
        self._compute(
            input_state=surface_states.SurfaceInputState(
                surface_pressure=input.pressure_ifc[self._surface_level],
                temperature=input.temperature[self._lowest_level],
            ),
            out=surface_flux_state.SurfaceFluxState(
                evapotranspiration=output.evapotranspiration,
                sensible_heat_flux=output.sensible_heat_flux,
                latent_heat_flux=output.latent_heat_flux,
                u_stress=output.u_stress,
                v_stress=output.v_stress,
                q_snocpymlt=output.q_snocpymlt,
                thermal_energy_flux=output.thermal_energy_flux,
                vapor_energy_flux=output.vapor_energy_flux,
            ),
        )
        output.surface_temperature.ndarray[...] = self._surface.surface_temperature.ndarray  # type: ignore[index]  # NDArrayObject declares no __setitem__
        return output


# NOTE: temporary, replaced in part 02 (collect): `fw.collect(Surface.Input, ...)`,
# which picks the leaves by quantity.
def collect_input(state: physics_state.PhysicsState) -> Surface.Input:
    """The surface input from the physics state (no copies)."""
    return Surface.Input(pressure_ifc=state.pressure_ifc, temperature=state.temperature)
