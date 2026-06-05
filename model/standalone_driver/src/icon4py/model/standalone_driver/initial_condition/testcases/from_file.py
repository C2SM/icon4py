# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import TYPE_CHECKING

import serialbox

from icon4py.model.common import type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver.initial_condition.testcases import utils as testcases_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import geometry as grid_geometry
    from icon4py.model.standalone_driver import driver_states


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FromFileParameters:
    """Parameters for the file-based initial condition."""

    #: Path to the serialised data directory (typically ``<experiment>/ser_data``).
    data_path: pathlib.Path
    #: Number of tracer species stored in the snapshot (0 means no tracers).
    ntracer: int = 0


def _read_prognostics_from_serialbox(
    *,
    data_path: pathlib.Path,
    rank: int,
    grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend,
    ntracer: int,
) -> prognostics.PrognosticState:
    """Read prognostic IC fields directly from a serialbox snapshot.

    Opens the serialbox archive at *data_path* for MPI rank *rank*,
    finds the ``prognostics / initial-state`` savepoint, and fills a
    freshly allocated :class:`~icon4py.model.common.states.prognostic_state.PrognosticState`.

    All array manipulation uses only ``numpy`` and the GT4Py field API;
    there is intentionally no dependency on ``icon4py.model.testing``.
    """
    fname = f"icon_pydycore_rank{rank}"
    ser = serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), fname)
    sp = ser.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()
    log.debug("Reading prognostics initial-state from %s / %s", data_path, fname)

    nc = grid.num_cells
    ne = grid.num_edges
    xp = data_alloc.import_array_ns(backend)

    def read_cell_k(name: str):
        return xp.asarray(xp.squeeze(ser.read(name, sp).astype(float))[:nc, :])

    def read_edge_k(name: str):
        return xp.asarray(xp.squeeze(ser.read(name, sp).astype(float))[:ne, :])

    state = prognostics.initialize_prognostic_state(grid=grid, allocator=backend, ntracer=ntracer)
    state.rho.ndarray[:, :] = read_cell_k("rho_now")
    state.exner.ndarray[:, :] = read_cell_k("exner_now")
    state.theta_v.ndarray[:, :] = read_cell_k("theta_v_now")
    state.vn.ndarray[:, :] = read_edge_k("vn_now")
    state.w.ndarray[:, :] = read_cell_k("w_now")  # shape (nc, num_levels + 1)

    if ntracer > 0:
        tracers_raw = xp.squeeze(ser.read("tracers_now", sp).astype(float))
        for i in range(ntracer):
            state.tracer[i].ndarray[:, :] = xp.asarray(tracers_raw[:nc, :, i])

    return state


def read_from_file(
    *,
    parameters: FromFileParameters,
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,  # unused; kept for API parity
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend,
    lowest_layer_thickness: ta.wpfloat,  # unused; kept for API parity
    model_top_height: ta.wpfloat,  # unused; kept for API parity
    stretch_factor: ta.wpfloat,  # unused; kept for API parity
    damping_height: ta.wpfloat,  # unused; kept for API parity
    exchange: decomposition_defs.ExchangeRuntime,
) -> driver_states.DriverStates:
    """Initialise prognostic state from a serialised ICON initial-condition snapshot.

    Reads ``prognostics / initial-state`` from the serialbox archive located at
    ``parameters.data_path``.  All other physics arguments are accepted but
    ignored so that this function shares the same call signature as the
    analytical initial-condition functions (``jablonowski_williamson``,
    ``gauss3d``, …) and can be stored transparently in
    :attr:`~icon4py.model.standalone_driver.initial_condition.InitialConditionConfig.create`.
    """
    prognostic_state_now = _read_prognostics_from_serialbox(
        data_path=parameters.data_path,
        rank=exchange.my_rank(),
        grid=grid,
        backend=backend,
        ntracer=parameters.ntracer,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=backend)
    return testcases_utils.assemble_driver_states(
        grid=grid,
        allocator=backend,
        backend=backend,
        exchange=exchange,
        interpolation=testcases_utils.extract_interpolation(interpolation_field_source),
        zone_indices_map=testcases_utils.zone_indices(grid),
        metrics_field_source=metrics_field_source,
        prognostic_state_now=prognostic_state_now,
        diagnostic_state=diagnostic_state,
    )
