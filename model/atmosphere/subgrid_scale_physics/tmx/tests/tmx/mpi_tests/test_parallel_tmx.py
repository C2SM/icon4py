# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel test of the Tmx granule.

Runs one full ``Tmx.run`` (Stages A to G) distributed over the MPI ranks, with
GHEX halo exchanges at the Fortran sync points, and verifies each rank's local
fields (including halos) against that rank's slice of the multirank serialized
reference (mpitask{N}_exclaim_ape_aesPhys). The reference data is produced by
the same Fortran run regardless of the decomposition, so agreement here also
implies agreement with the single-rank results of the serial integration tests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.testing import definitions, parallel_helpers

from ..fixtures import *  # noqa: F403
from ..integration_tests.utils import (
    TMX_DATES,
    construct_config,
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


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_parallel_tmx_full_run_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: "
        f"local grid with {icon_grid.num_cells} cells, {icon_grid.num_edges} edges, "
        f"{icon_grid.num_vertices} vertices"
    )

    allocator = model_backends.get_allocator(backend)
    init_savepoint = data_provider.from_savepoint_tmx_init()
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    surface_fluxes_savepoint = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

    exchange = decomp_defs.create_exchange(process_props, decomposition_info)

    config = construct_config(init_savepoint)
    granule = tmx.Tmx(
        grid=icon_grid,
        config=config,
        params=tmx.TmxParams(config),
        vertical_grid=None,
        metric_state=construct_metric_state(
            metrics_savepoint, init_savepoint, grid_savepoint, allocator
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        exchange=exchange,
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
        init_savepoint.dtime(),
    )
    _log.info(f"rank={process_props.rank}/{process_props.comm_size}: tmx run done")

    verify_full_run_fields(diagnostic_state, tendency_state, exit_savepoint, icon_grid.num_levels)
