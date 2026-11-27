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
from types import ModuleType
from typing import TYPE_CHECKING

import serialbox  # type: ignore[import-untyped]

from icon4py.model.common import dimension as dims, model_backends, state, states, time
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
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


def _read_buffer(
    serializer: serialbox.Serializer,
    savepoint: serialbox.Savepoint,
    name: str,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Read a field of the global (single-rank) archive and move it to the device.

    ``serializer.read`` always returns a numpy array; moving it to the device
    first lets the rank-local selection happen on the device on GPU backends.
    """
    return array_ns.asarray(serializer.read(name, savepoint).astype(float).squeeze())


def _read_hdim_k(
    serializer: serialbox.Serializer,
    savepoint: serialbox.Savepoint,
    name: str,
    indices: data_alloc.NDArray,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Read a (horizontal, K) field and select the rows of this rank."""
    return _read_buffer(serializer, savepoint, name, array_ns)[indices, :]


def _read_predictor_corrector(
    serializer: serialbox.Serializer,
    savepoint: serialbox.Savepoint,
    name: str,
    indices: data_alloc.NDArray,
    array_ns: ModuleType,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Read a (horizontal, K, 2) tendency and return its two time levels for this rank."""
    buffer = _read_buffer(serializer, savepoint, name, array_ns)
    return buffer[indices, :, 0], buffer[indices, :, 1]


def _read_prognostic_state(
    prognostic_state: states.PrognosticState,
    *,
    serializer: serialbox.Serializer,
    savepoint: serialbox.Savepoint,
    cell_indices: data_alloc.NDArray,
    edge_indices: data_alloc.NDArray,
    array_ns: ModuleType,
) -> None:
    def read(name: str, indices: data_alloc.NDArray) -> data_alloc.NDArray:
        return _read_hdim_k(serializer, savepoint, name, indices, array_ns)

    prognostic_state.rho.ndarray[:, :] = read("rho_now", cell_indices)
    prognostic_state.exner.ndarray[:, :] = read("exner_now", cell_indices)
    prognostic_state.theta_v.ndarray[:, :] = read("theta_v_now", cell_indices)
    prognostic_state.vn.ndarray[:, :] = read("vn_now", edge_indices)
    prognostic_state.w.ndarray[:, :] = read("w_now", cell_indices)


def read_initial_condition_from_file(
    *,
    config: ConfigContext,
    grid: icon_grid.IconGrid,
    prognostic_state_now: states.PrognosticState,
    tracer_state_now: states.TracerState,
    backend: model_backends.BackendLike,
    exchange: decomposition_defs.ExchangeRuntime,
    decomposition_info: decomposition_defs.DecompositionInfo,
) -> None:
    """Initialise the prognostic and tracer states from the serialized ICON initial state."""
    ic_config = config.initial_condition
    assert isinstance(ic_config, FromFileConfig)
    array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))

    log.info("Reading the initial condition from %s", ic_config.data_path)
    # TODO(icon4py): the archive name is hardcoded to the single-rank (global) archive
    # because the per-rank archives follow ICON's domain decomposition, not icon4py's,
    # so each rank selects its rows by global index instead. Make this explicit
    # (e.g. in the config) rather than hardcoding 'rank0'.
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        str(ic_config.data_path),
        "icon_pydycore_rank0",
    )
    savepoint = serializer.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()

    cell_indices = decomposition_info.global_index(dims.CellDim)
    edge_indices = decomposition_info.global_index(dims.EdgeDim)

    _read_prognostic_state(
        prognostic_state_now,
        serializer=serializer,
        savepoint=savepoint,
        cell_indices=cell_indices,
        edge_indices=edge_indices,
        array_ns=array_ns,
    )

    if config.ntracer > 0:
        tracers = _read_buffer(serializer, savepoint, "tracers_now", array_ns)
        for i, tracer in enumerate(state.fields(tracer_state_now).values()):
            tracer.ndarray[:, :] = tracers[cell_indices, :, i]


def read_restart_from_file(
    *,
    config: ConfigContext,
    grid: icon_grid.IconGrid,
    prognostic_state_now: states.PrognosticState,
    dycore_diagnostics: states.DycoreDiagnostics,
    backend: model_backends.BackendLike,
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
    if config.ntracer > 0:
        raise NotImplementedError(
            "restarting with tracers is not supported: the solve-nonhydro savepoints do not "
            "carry them, they are in the advection-init savepoint of the same date."
        )

    array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))
    date = _savepoint_formatted_date(config.start_of_timestepping + config.dtime)

    log.info("Restarting from the serialized state of the time step ending at %s", date)
    # TODO(icon4py): the archive name is hardcoded to the single-rank (global) archive
    # because the per-rank archives follow ICON's domain decomposition, not icon4py's,
    # so each rank selects its rows by global index instead. Make this explicit
    # (e.g. in the config) rather than hardcoding 'rank0'.
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        str(ic_config.data_path),
        "icon_pydycore_rank0",
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

    cell_indices = decomposition_info.global_index(dims.CellDim)
    edge_indices = decomposition_info.global_index(dims.EdgeDim)

    _read_prognostic_state(
        prognostic_state_now,
        serializer=serializer,
        savepoint=nonhydro_savepoint,
        cell_indices=cell_indices,
        edge_indices=edge_indices,
        array_ns=array_ns,
    )

    dycore_diagnostics.perturbed_exner_at_cells_on_model_levels.ndarray[:, :] = _read_hdim_k(
        serializer, nonhydro_savepoint, "exner_pr", cell_indices, array_ns
    )

    normal_wind_tendency = _read_predictor_corrector(
        serializer, velocity_savepoint, "ddt_vn_apc_pc", edge_indices, array_ns
    )
    vertical_wind_tendency = _read_predictor_corrector(
        serializer, velocity_savepoint, "ddt_w_adv_pc", cell_indices, array_ns
    )

    normal_wind_pair = dycore_diagnostics.normal_wind_advective_tendency
    normal_wind_pair.predictor.ndarray[:, :] = normal_wind_tendency[0]
    normal_wind_pair.corrector.ndarray[:, :] = normal_wind_tendency[1]
    # The dycore swaps the vertical advective tendency at the first substep of a time step
    # that is not the initial one, and then consumes the predictor without recomputing it.
    # The two time levels are therefore stored swapped.
    vertical_wind_pair = dycore_diagnostics.vertical_wind_advective_tendency
    vertical_wind_pair.predictor.ndarray[:, :] = vertical_wind_tendency[1]
    vertical_wind_pair.corrector.ndarray[:, :] = vertical_wind_tendency[0]
