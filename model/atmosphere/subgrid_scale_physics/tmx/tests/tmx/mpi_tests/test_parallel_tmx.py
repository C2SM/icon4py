# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel test of the tmx component.

Runs one step distributed over the MPI ranks and verifies each rank's owned cells against its
slice of the multi-rank serialized reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, parallel_helpers

from ..fixtures import *  # noqa: F403
from ..integration_tests.utils import (
    assert_tmx_exit_fields,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize(
    "experiment_description, date",
    [
        (definitions.Experiments.EXCLAIM_APE_AES, date)
        for date in definitions.Experiments.EXCLAIM_APE_AES.dates[1:]
    ],
)
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_parallel_tmx_run_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment: definitions.Experiment,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    tmx_config = experiment.config.tmx
    assert tmx_config is not None
    allocator = model_backends.get_allocator(backend)

    component = tmx.Tmx(
        grid=icon_grid,
        config=tmx_config,
        metric_state=construct_metric_state(
            metrics_savepoint=metrics_savepoint,
            init_savepoint=data_provider.from_savepoint_tmx_init(),
            allocator=allocator,
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        backend=backend,
        exchange=decomp_defs.create_exchange(process_props, decomposition_info),
    )
    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator)

    component.run(
        input_state=construct_input_state(data_provider.from_savepoint_tmx_entry(date=date)),
        surface_flux_state=construct_surface_flux_state(
            data_provider.from_savepoint_tmx_surface_fluxes(date=date)
        ),
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        new_state=tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator),
        dtime=experiment.config.driver.dtime.total_seconds(),
    )
    _log.info(f"rank={process_props.rank}/{process_props.comm_size}: tmx step done")

    assert_tmx_exit_fields(
        tendency_state=tendency_state,
        diagnostic_state=diagnostic_state,
        exit_savepoint=data_provider.from_savepoint_tmx_exit(date=date),
        use_km_const=tmx_config.use_km_const,
        owner_mask=data_alloc.as_numpy(decomposition_info.owner_mask(dims.CellDim)),
    )
