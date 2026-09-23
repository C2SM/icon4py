# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests of the tmx scalar diffusion component.

Both stages are seeded from the tmx-diagnostics-exit savepoint (``kh_ic``, ``km_ie``) and the
temperature stage also from the tmx-hydro-exit savepoint (the new qv, qc, qi), so that failures
do not cascade between them.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import scalar_diffusion, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.config import TmxConfig
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import (
    RTOL,
    TMX_DATES,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@dataclasses.dataclass
class _Setup:
    component: scalar_diffusion.ScalarDiffusion
    input_state: tmx_states.TmxInputState
    surface_flux_state: tmx_states.TmxSurfaceFluxState
    diagnostic_state: tmx_states.TmxDiagnosticState
    tendency_state: tmx_states.TmxTendencyState
    new_state: tmx_states.TmxNewState


def _setup(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    config: TmxConfig,
) -> _Setup:
    allocator = model_backends.get_allocator(backend)
    diagnostics_exit_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)
    component = scalar_diffusion.ScalarDiffusion(
        grid=icon_grid,
        metric_state=construct_metric_state(
            metrics_savepoint=metrics_savepoint,
            init_savepoint=data_provider.from_savepoint_tmx_init(),
            allocator=allocator,
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        backend=backend,
        exchange=decomposition.SingleNodeExchange(),
        turb_prandtl=config.turb_prandtl,
        energy_type=config.energy_type,
        use_scale_turb_energy_flux=config.use_scale_turb_energy_flux,
        scale_turb_energy_flux=config.scale_turb_energy_flux,
    )
    return _Setup(
        component=component,
        input_state=construct_input_state(data_provider.from_savepoint_tmx_entry(date=date)),
        surface_flux_state=construct_surface_flux_state(
            data_provider.from_savepoint_tmx_surface_fluxes(date=date)
        ),
        diagnostic_state=dataclasses.replace(
            tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator),
            kh_ic=diagnostics_exit_savepoint.kh_ic(),
            km_ie=diagnostics_exit_savepoint.km_ie(),
        ),
        tendency_state=tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator),
        new_state=tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_hydrometeor_diffusion_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: TmxConfig,
    tmx_dtime: float,
) -> None:
    setup = _setup(
        data_provider=data_provider,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
        interpolation_savepoint=interpolation_savepoint,
        icon_grid=icon_grid,
        backend=backend,
        date=date,
        config=tmx_config,
    )
    exit_savepoint = data_provider.from_savepoint_tmx_hydro_exit(date=date)

    setup.component.run_hydrometeor_diffusion(
        input_state=setup.input_state,
        surface_flux_state=setup.surface_flux_state,
        diagnostic_state=setup.diagnostic_state,
        tendency_state=setup.tendency_state,
        new_state=setup.new_state,
        dtime=tmx_dtime,
    )

    fields = (
        (setup.tendency_state.tend_qv, exit_savepoint.tend_qv(), "tend_qv", 5.0e-20),
        (setup.tendency_state.tend_qc, exit_savepoint.tend_qc(), "tend_qc", 5.0e-21),
        (setup.tendency_state.tend_qi, exit_savepoint.tend_qi(), "tend_qi", 3.0e-22),
        (setup.new_state.qv, exit_savepoint.qv_new(), "qv_new", 2.0e-17),
        (setup.new_state.qc, exit_savepoint.qc_new(), "qc_new", 2.0e-18),
        (setup.new_state.qi, exit_savepoint.qi_new(), "qi_new", 7.0e-20),
    )
    for actual, desired, name, atol in fields:
        test_utils.assert_dallclose(
            actual.asnumpy(), desired.asnumpy(), rtol=RTOL, atol=atol, err_msg=name
        )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_temperature_diffusion_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: TmxConfig,
    tmx_dtime: float,
) -> None:
    setup = _setup(
        data_provider=data_provider,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
        interpolation_savepoint=interpolation_savepoint,
        icon_grid=icon_grid,
        backend=backend,
        date=date,
        config=tmx_config,
    )
    hydro_exit_savepoint = data_provider.from_savepoint_tmx_hydro_exit(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_temperature_exit(date=date)
    new_state = dataclasses.replace(
        setup.new_state,
        qv=hydro_exit_savepoint.qv_new(),
        qc=hydro_exit_savepoint.qc_new(),
        qi=hydro_exit_savepoint.qi_new(),
    )

    setup.component.run_temperature_diffusion(
        input_state=setup.input_state,
        surface_flux_state=setup.surface_flux_state,
        diagnostic_state=setup.diagnostic_state,
        tendency_state=setup.tendency_state,
        new_state=new_state,
        dtime=tmx_dtime,
    )

    fields = (
        (setup.component.energy, exit_savepoint.energy(), "energy", 2.0e-10),
        (new_state.temperature, exit_savepoint.ta_new(), "ta_new", 4.0e-13),
        (setup.tendency_state.tend_temperature, exit_savepoint.tend_ta(), "tend_ta", 2.0e-15),
    )
    for actual, desired, name, atol in fields:
        test_utils.assert_dallclose(
            actual.asnumpy(), desired.asnumpy(), rtol=RTOL, atol=atol, err_msg=name
        )
