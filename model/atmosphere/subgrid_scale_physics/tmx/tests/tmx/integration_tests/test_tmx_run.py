# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end integration test of the Tmx granule (M6).

Constructs the granule from the serialized ICON state (exp.exclaim_ape_aesPhys)
and verifies one full ``run`` (Stages A to G) from the tmx-entry /
tmx-surface-fluxes savepoints against the tmx-exit savepoint (final
tendencies, dissipation heating and vertically integrated diagnostics).
Unlike the per-stage tests, nothing is seeded from intermediate savepoints:
this exercises the complete Fortran ``Compute`` sequence of mo_vdf.f90.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from .utils import (
    TMX_DATES,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
    verify_full_run_fields,
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
def test_tmx_full_run_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    tmx_config: tmx.TmxConfig,
    tmx_dtime: float,
) -> None:
    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    surface_fluxes_savepoint = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

    granule = tmx.Tmx(
        grid=icon_grid,
        config=tmx_config,
        params=tmx.TmxParams(tmx_config),
        vertical_grid=None,
        metric_state=construct_metric_state(
            metrics_savepoint, init_savepoint, grid_savepoint, allocator
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        backend=backend,
    )

    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator)

    granule.run(
        construct_input_state(entry_savepoint),
        construct_surface_flux_state(surface_fluxes_savepoint),
        diagnostic_state,
        tendency_state,
        new_state,
        tmx_dtime,
    )

    verify_full_run_fields(diagnostic_state, tendency_state, exit_savepoint, icon_grid.num_levels)
