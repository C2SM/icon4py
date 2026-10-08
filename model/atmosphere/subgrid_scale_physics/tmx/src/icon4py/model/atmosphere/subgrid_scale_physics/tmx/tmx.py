# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tmx turbulent mixing component: one step of `Compute` in ICON's `mo_vdf.f90`."""

from __future__ import annotations

import logging
import typing

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    diagnostics,
    energy_update,
    scalar_diffusion,
    tmx_states,
    wind_diffusion,
)


if typing.TYPE_CHECKING:
    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
    from icon4py.model.common import model_backends
    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import base as base_grid


log = logging.getLogger(__name__)


class Tmx:
    """The tmx turbulent mixing: runs its stages in ICON's order on shared states."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: tmx_config.TmxConfig,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime,
    ) -> None:
        self.diagnostics = diagnostics.Diagnostics(
            grid=grid,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=edge_params,
            cell_params=cell_params,
            backend=backend,
            exchange=exchange,
            turb_prandtl=config.turb_prandtl,
            smag_constant=config.smag_constant,
            max_turb_scale=config.max_turb_scale,
            km_min=config.km_min,
            km_const=config.km_const,
            use_km_const=config.use_km_const,
            louis_constant_b=config.louis_constant_b,
            use_louis=config.use_louis,
            use_louis_land=config.use_louis_land,
            use_louis_ice=config.use_louis_ice,
        )
        self.scalar_diffusion = scalar_diffusion.ScalarDiffusion(
            grid=grid,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=edge_params,
            backend=backend,
            exchange=exchange,
            turb_prandtl=config.turb_prandtl,
            use_scale_turb_energy_flux=config.use_scale_turb_energy_flux,
            scale_turb_energy_flux=config.scale_turb_energy_flux,
        )
        self.wind_diffusion = wind_diffusion.WindDiffusion(
            grid=grid,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=edge_params,
            backend=backend,
            exchange=exchange,
        )
        self.energy_update = energy_update.EnergyUpdate(
            grid=grid,
            metric_state=metric_state,
            backend=backend,
            exchange=exchange,
            dissipation_factor=config.dissipation_factor,
            turb_prandtl=config.turb_prandtl,
            use_km_const=config.use_km_const,
            km_const=config.km_const,
        )

    def run(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: tmx_states.TmxSurfaceFluxState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
        tendency_state: tmx_states.TmxTendencyState,
        new_state: tmx_states.TmxNewState,
        dtime: float,
    ) -> None:
        """
        Run one tmx step: write the tendencies to `tendency_state`, the updated fields to
        `new_state` and the diagnostics to `diagnostic_state`.
        """
        log.debug("tmx step: start")

        self.diagnostics.run(input_state=input_state, diagnostic_state=diagnostic_state)
        self.scalar_diffusion.run_hydrometeor_diffusion(
            input_state=input_state,
            surface_flux_state=surface_flux_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            new_state=new_state,
            dtime=dtime,
        )
        self.scalar_diffusion.run_temperature_diffusion(
            input_state=input_state,
            surface_flux_state=surface_flux_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            new_state=new_state,
            dtime=dtime,
        )
        self.wind_diffusion.run(
            input_state=input_state,
            surface_flux_state=surface_flux_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            new_state=new_state,
            dtime=dtime,
        )
        self.energy_update.run(
            input_state=input_state,
            surface_flux_state=surface_flux_state,
            diagnostic_state=diagnostic_state,
            tendency_state=tendency_state,
            new_state=new_state,
            dtime=dtime,
        )

        log.debug("tmx step: end")
