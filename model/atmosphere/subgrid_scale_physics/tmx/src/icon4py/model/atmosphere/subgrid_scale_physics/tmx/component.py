# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tmx turbulent mixing as a typed component."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import field, framework as fw, model_backends, quantity, type_alias as ta
from icon4py.model.common.framework import quantities as qty
from icon4py.model.common.states import surface_flux_state


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    # a type-only import: tmx does not depend on the physics driver
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
    from icon4py.model.common import time
    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import base as base_grid, states as grid_states


class Tmx(fw.Component["Tmx.Input", "Tmx.Output"]):
    """
    The tmx turbulent mixing: the tendencies of temperature, the three moist tracers it diffuses
    and the three wind components, and the turbulent viscosity and diffusivity `km` and `kh`.

    The new state tmx computes alongside the tendencies stays internal: the caller advances the
    model state from the tendencies.
    """

    # TODO(yiluchen1066): on an in-window step on which tmx does not compute, `PhysicsProcess`
    # recycles its last tendencies, as ICON's microphysics does; ICON's tmx interface instead
    # contributes nothing on such a step. No difference while tmx runs every step; revisit when
    # the tmx time step is read from the configuration.

    class Input(fw.State):
        temperature: field.Field[qty.TemperatureOnCellK]
        virtual_temperature: field.Field[qty.VirtualTemperatureOnCellK]
        pressure: field.Field[qty.PressureOnCellK]
        u: field.Field[qty.UOnCellK]
        v: field.Field[qty.VOnCellK]
        w: field.Field[qty.WOnCellKHalf]
        qv: field.Field[qty.QvOnCellK]
        qc: field.Field[qty.QcOnCellK]
        qi: field.Field[qty.QiOnCellK]
        qr: field.Field[qty.QrOnCellK]
        qs: field.Field[qty.QsOnCellK]
        qg: field.Field[qty.QgOnCellK]
        rho: field.Field[qty.RhoOnCellK]
        air_mass: field.Field[qty.AirMassOnCellK]
        evapotranspiration: field.Field[qty.EvapotranspirationOnCell]
        sensible_heat_flux: field.Field[qty.SensibleHeatFluxOnCell]
        latent_heat_flux: field.Field[qty.LatentHeatFluxOnCell]
        u_stress: field.Field[qty.UStressOnCell]
        v_stress: field.Field[qty.VStressOnCell]
        q_snocpymlt: field.Field[qty.SnowCanopyMeltHeatOnCell]
        thermal_energy_flux: field.Field[qty.ThermalEnergyFluxOnCell]
        vapor_energy_flux: field.Field[qty.VaporEnergyFluxOnCell]

    class Output(fw.State):
        tend_temperature: field.Field[qty.TendencyOfTemperatureOnCellK]
        tend_qv: field.Field[qty.TendencyOfQvOnCellK]
        tend_qc: field.Field[qty.TendencyOfQcOnCellK]
        tend_qi: field.Field[qty.TendencyOfQiOnCellK]
        tend_u: field.Field[qty.TendencyOfUOnCellK]
        tend_v: field.Field[qty.TendencyOfVOnCellK]
        tend_w: field.Field[qty.TendencyOfWOnCellKHalf]
        km: field.Field[qty.ExchangeCoefficientForMomentumOnCellK]
        kh: field.Field[qty.ExchangeCoefficientForHeatOnCellK]

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: tmx_config.TmxConfig,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        dtime: time.RelativeTime,
        backend: gtx_typing.Backend | None = None,
        exchange: decomposition.ExchangeRuntime,
    ) -> None:
        self._dt_seconds = ta.wpfloat(dtime.total_seconds())
        self._tmx = tmx.Tmx(
            grid=grid,
            config=config,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=edge_params,
            cell_params=cell_params,
            backend=backend,
            exchange=exchange,
        )
        allocator = model_backends.get_allocator(backend)
        # km and kh are replaced by the output's buffers in `run`
        self._diagnostic_state = tmx_states.TmxDiagnosticState.allocate(grid, allocator=allocator)
        self._new_state = tmx_states.TmxNewState.allocate(grid, allocator=allocator)

    def run(self, input: Input, output: Output) -> Output:
        """One tmx step: the tendencies and `km`, `kh` into `output`; the inputs are read only."""
        self._tmx.run(
            input_state=tmx_states.TmxInputState(
                temperature=input.temperature,
                virtual_temperature=input.virtual_temperature,
                pressure=input.pressure,
                u=input.u,
                v=input.v,
                w=input.w,
                qv=input.qv,
                qc=input.qc,
                qi=input.qi,
                qr=input.qr,
                qs=input.qs,
                qg=input.qg,
                rho=input.rho,
                air_mass=input.air_mass,
            ),
            surface_flux_state=surface_flux_state.SurfaceFluxState(
                evapotranspiration=input.evapotranspiration,
                sensible_heat_flux=input.sensible_heat_flux,
                latent_heat_flux=input.latent_heat_flux,
                u_stress=input.u_stress,
                v_stress=input.v_stress,
                q_snocpymlt=input.q_snocpymlt,
                thermal_energy_flux=input.thermal_energy_flux,
                vapor_energy_flux=input.vapor_energy_flux,
            ),
            diagnostic_state=dataclasses.replace(
                self._diagnostic_state, km=output.km, kh=output.kh
            ),
            tendency_state=tmx_states.TmxTendencyState(
                tend_temperature=output.tend_temperature,
                tend_qv=output.tend_qv,
                tend_qc=output.tend_qc,
                tend_qi=output.tend_qi,
                tend_u=output.tend_u,
                tend_v=output.tend_v,
                tend_w=output.tend_w,
            ),
            new_state=self._new_state,
            dtime=self._dt_seconds,
        )
        return output


def _require[Q: quantity.Quantity](value: field.Field[Q] | None, name: str) -> field.Field[Q]:
    if value is None:
        raise ValueError(f"tmx needs '{name}', which the physics state lacks.")
    return value


# NOTE: temporary, components MWE part 02 (collect): replaced by `fw.collect(Tmx.Input, ...)`,
# which picks the leaves by quantity.
def collect_input(state: physics_state.PhysicsState) -> Tmx.Input:
    """
    The tmx input from the physics state (no copies).

    The air mass is the one the driver hands to the physics, and the surface fluxes are the
    surface process's output: without either the physics state cannot feed tmx.
    """
    return Tmx.Input(
        temperature=state.temperature,
        virtual_temperature=state.virtual_temperature,
        pressure=state.pressure,
        u=state.u,
        v=state.v,
        w=state.w,
        qv=state.qv,
        qc=state.qc,
        qi=state.qi,
        qr=state.qr,
        qs=state.qs,
        qg=state.qg,
        rho=state.rho,
        air_mass=_require(state.air_mass, "air_mass"),
        evapotranspiration=_require(state.evapotranspiration, "evapotranspiration"),
        sensible_heat_flux=_require(state.sensible_heat_flux, "sensible_heat_flux"),
        latent_heat_flux=_require(state.latent_heat_flux, "latent_heat_flux"),
        u_stress=_require(state.u_stress, "u_stress"),
        v_stress=_require(state.v_stress, "v_stress"),
        q_snocpymlt=_require(state.q_snocpymlt, "q_snocpymlt"),
        thermal_energy_flux=_require(state.thermal_energy_flux, "thermal_energy_flux"),
        vapor_energy_flux=_require(state.vapor_energy_flux, "vapor_energy_flux"),
    )
