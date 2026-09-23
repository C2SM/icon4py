# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the tmx wind diffusion component.

Constructs the component from the serialized ICON state (exp.exclaim_ape_aesPhys), with the
diagnostics taken from the tmx-diagnostics-exit savepoint, and verifies one call of ``run``
against the tmx-hor-wind-exit, tmx-vert-wind-exit and tmx-exit savepoints.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states, wind_diffusion
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


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_run_wind_diffusion_single_step(
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
    allocator = model_backends.get_allocator(backend)
    diagnostics_savepoint = data_provider.from_savepoint_tmx_diagnostics_exit(date=date)
    hor_wind_savepoint = data_provider.from_savepoint_tmx_hor_wind_exit(date=date)
    vert_wind_savepoint = data_provider.from_savepoint_tmx_vert_wind_exit(date=date)
    # the tend_wa of tmx-vert-wind-exit misses the horizontal term: it was serialized before
    # the asynchronous GPU kernel adding it had finished; nothing changes tend_wa until tmx-exit
    exit_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

    component = wind_diffusion.WindDiffusion(
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
        solver_type=tmx_config.solver_type,
    )
    diagnostic_state = dataclasses.replace(
        tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator),
        **{
            name: getattr(diagnostics_savepoint, name)()
            for name in (
                "vn",
                "u_vert",
                "v_vert",
                "w_vert",
                "w_ie",
                "rho_ic",
                "div_c",
                "km_c",
                "km_ic",
                "km_iv",
                "km_ie",
            )
        },
    )
    tendency_state = tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator)

    component.run(
        input_state=construct_input_state(data_provider.from_savepoint_tmx_entry(date=date)),
        surface_flux_state=construct_surface_flux_state(
            data_provider.from_savepoint_tmx_surface_fluxes(date=date)
        ),
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        new_state=new_state,
        dtime=tmx_dtime,
    )

    # (computed, reference, absolute tolerance)
    fields = {
        "tend_ua": (tendency_state.tend_u, hor_wind_savepoint.tend_ua(), 4.0e-17),
        "tend_va": (tendency_state.tend_v, hor_wind_savepoint.tend_va(), 4.0e-17),
        "ua_new": (new_state.u, hor_wind_savepoint.ua_new(), 2.0e-14),
        "va_new": (new_state.v, hor_wind_savepoint.va_new(), 1.0e-14),
        "tend_wa": (tendency_state.tend_w, exit_savepoint.tend_wa(), 9.0e-19),
        "wa_new": (new_state.w, vert_wind_savepoint.wa_new(), 3.0e-16),
    }
    for name, (computed, reference, atol) in fields.items():
        test_utils.assert_dallclose(
            computed.asnumpy(), reference.asnumpy(), rtol=RTOL, atol=atol, err_msg=name
        )
