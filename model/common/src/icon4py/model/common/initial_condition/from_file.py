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
import typing
from typing import TYPE_CHECKING

import numpy as np
import serialbox  # type: ignore[import-untyped]

from icon4py.model.common import dimension as dims, model_backends, time
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.states import (
    nonhydro_states,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.initial_condition.config import ConfigContext


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FromFileConfig:
    """Parameters for the file-based initial condition."""

    data_path: typing.Annotated[
        pathlib.Path,
        common_conf_opt.ConfigOption(
            description="Path to the serialised data directory (typically '<experiment>/ser_data')."
        ),
    ]


def _savepoint_formatted_date(date: time.AbsoluteTime) -> str:
    """Format a datetime the way the savepoints of the serialized data are stamped."""
    return date.replace(tzinfo=None).isoformat(timespec="milliseconds")


def _available_nonhydro_init_dates(serializer: serialbox.Serializer) -> str:
    dates = sorted(
        {
            savepoint.metainfo["date"]
            for savepoint in serializer.savepoint_list()
            if savepoint.name == "solve-nonhydro-init"
        }
    )
    return ", ".join(dates) if dates else "none"


def _read_prognostic_state(
    prognostic_state: prognostics.PrognosticState,
    read_cell_k,
    read_edge_k,
):
    prognostic_state.rho.ndarray[:, :] = read_cell_k("rho_now")
    prognostic_state.exner.ndarray[:, :] = read_cell_k("exner_now")
    prognostic_state.theta_v.ndarray[:, :] = read_cell_k("theta_v_now")
    prognostic_state.vn.ndarray[:, :] = read_edge_k("vn_now")
    prognostic_state.w.ndarray[:, :] = read_cell_k("w_now")


def read_initial_condition_from_file(
    *,
    config: ConfigContext,
    grid: icon_grid.IconGrid,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    decomposition_info: decomposition_defs.DecompositionInfo,
) -> None:
    """Initialise the prognostic and tracer states from the serialized ICON initial state."""
    ic_config = config.initial_condition
    assert isinstance(ic_config, FromFileConfig)
    array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))

    log.info("Reading the initial condition from %s", ic_config.data_path)
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        str(ic_config.data_path),
        "icon_pydycore_rank0",  # always read the single-rank (global) data
    )
    savepoint = serializer.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()

    cell_indices = data_alloc.as_numpy(decomposition_info.global_index(dims.CellDim))
    edge_indices = data_alloc.as_numpy(decomposition_info.global_index(dims.EdgeDim))

    def read_cell_k(name: str) -> data_alloc.NDArray:
        buffer = np.squeeze(serializer.read(name, savepoint).astype(float))
        return array_ns.asarray(buffer[cell_indices, :])

    def read_edge_k(name: str) -> data_alloc.NDArray:
        buffer = np.squeeze(serializer.read(name, savepoint).astype(float))
        return array_ns.asarray(buffer[edge_indices, :])

    _read_prognostic_state(prognostic_state_now, read_cell_k, read_edge_k)

    ntracer = config.ntracer
    if ntracer > 0:
        tracers = np.squeeze(serializer.read("tracers_now", savepoint).astype(float))
        for i, tracer in enumerate(tracer_state_now.active_fields()):
            tracer.field.ndarray[:, :] = array_ns.asarray(tracers[cell_indices, :, i])


def read_restart_from_file(
    *,
    config: ConfigContext,
    grid: icon_grid.IconGrid,
    prognostic_state_now: prognostics.PrognosticState,
    solve_nonhydro_diagnostic_state: nonhydro_states.DiagnosticStateNonHydro,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    decomposition_info: decomposition_defs.DecompositionInfo,
) -> None:
    """
    Initialise the prognostic state from the serialized state of a later time step.

    On a restart (isRestart() in mo_nh_stepping.f90) ICON reads the prognostic
    variables, the perturbed exner function and the advective tendencies of the
    previous time step from its restart file, and skips compute_exner_pert. The
    same quantities are read here from the savepoints written at the beginning
    of the time step that starts at 'start_of_timestepping'. Those savepoints
    are stamped with the date of the end of their time step.
    """
    ic_config = config.initial_condition
    assert isinstance(ic_config, FromFileConfig)
    ntracer = config.ntracer
    if ntracer > 0:
        raise NotImplementedError(
            "restarting with tracers is not supported: the solve-nonhydro savepoints do not "
            "carry them, they are in the advection-init savepoint of the same date."
        )

    array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))
    date = _savepoint_formatted_date(config.start_of_timestepping + config.dtime)

    log.info("Restarting from the serialized state of the time step ending at %s", date)
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        str(ic_config.data_path),
        "icon_pydycore_rank0",  # always read the single-rank (global) data
    )
    try:
        nonhydro_savepoint = (
            serializer.savepoint["solve-nonhydro-init"]
            .istep[1]
            .date[date]
            .dyn_timestep[1]
            .as_savepoint()
        )
        velocity_savepoint = (
            serializer.savepoint["velocity-tendencies-init"]
            .istep[1]
            .date[date]
            .dyn_timestep[1]
            .as_savepoint()
        )
    except serialbox.SerialboxError as err:
        raise ValueError(
            f"there is no serialized state to restart from at {date}. The serialized data "
            f"of this experiment covers: {_available_nonhydro_init_dates(serializer)}."
        ) from err

    cell_indices = data_alloc.as_numpy(decomposition_info.global_index(dims.CellDim))
    edge_indices = data_alloc.as_numpy(decomposition_info.global_index(dims.EdgeDim))

    def read_cell_k(name: str) -> data_alloc.NDArray:
        buffer = np.squeeze(serializer.read(name, nonhydro_savepoint).astype(float))
        return array_ns.asarray(buffer[cell_indices, :])

    def read_edge_k(name: str) -> data_alloc.NDArray:
        buffer = np.squeeze(serializer.read(name, nonhydro_savepoint).astype(float))
        return array_ns.asarray(buffer[edge_indices, :])

    def read_predictor_corrector(
        name: str, indices: np.ndarray
    ) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
        buffer = np.squeeze(serializer.read(name, velocity_savepoint).astype(float))
        return array_ns.asarray(buffer[indices, :, 0]), array_ns.asarray(buffer[indices, :, 1])

    _read_prognostic_state(prognostic_state_now, read_cell_k, read_edge_k)

    solve_nonhydro_diagnostic_state.perturbed_exner_at_cells_on_model_levels.ndarray[:, :] = (
        read_cell_k("exner_pr")
    )

    normal_wind_tendency = read_predictor_corrector("ddt_vn_apc_pc", edge_indices)
    vertical_wind_tendency = read_predictor_corrector("ddt_w_adv_pc", cell_indices)

    solve_nonhydro_diagnostic_state.normal_wind_advective_tendency.predictor.ndarray[:, :] = (
        normal_wind_tendency[0]
    )
    solve_nonhydro_diagnostic_state.normal_wind_advective_tendency.corrector.ndarray[:, :] = (
        normal_wind_tendency[1]
    )
    # The dycore swaps the vertical advective tendency at the first substep of a time step
    # that is not the initial one, and then consumes the predictor without recomputing it.
    # The two time levels are therefore stored swapped.
    solve_nonhydro_diagnostic_state.vertical_wind_advective_tendency.predictor.ndarray[:, :] = (
        vertical_wind_tendency[1]
    )
    solve_nonhydro_diagnostic_state.vertical_wind_advective_tendency.corrector.ndarray[:, :] = (
        vertical_wind_tendency[0]
    )
