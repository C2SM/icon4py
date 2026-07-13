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
import types
from collections.abc import Callable
from typing import TYPE_CHECKING

import serialbox  # type: ignore[import-untyped]

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.initial_condition import states as ic_states
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FromFileConfig:
    """Parameters for the file-based initial condition."""

    #: Path to the serialised data directory (typically ``<experiment>/ser_data``).
    data_path: pathlib.Path
    #: Number of tracer species stored in the snapshot (0 means no tracers).
    ntracer: int = 0
    #: Beginning of the simulation. Set by 'ExperimentConfig' from the driver config.
    start_of_simulation: time.AbsoluteTime | None = None
    #: Beginning of the time loop. Differs from 'start_of_simulation' when restarting.
    start_of_timestepping: time.AbsoluteTime | None = None
    #: Model time step. Needed to select the savepoint to restart from.
    dtime: time.RelativeTime | None = None

    @property
    def is_restart(self) -> bool:
        """Whether the time loop starts at a later date than the simulation."""
        return (
            self.start_of_timestepping is not None
            and self.start_of_timestepping != self.start_of_simulation
        )


def _savepoint_date(date: time.AbsoluteTime) -> str:
    """Format a datetime the way the savepoints of the serialized data are stamped."""
    return date.replace(tzinfo=None).isoformat(timespec="milliseconds")


def _open_serializer(data_path: pathlib.Path, rank: int) -> serialbox.Serializer:
    fname = f"icon_pydycore_rank{rank}"
    log.debug("Reading the initial condition from %s / %s", data_path, fname)
    return serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), fname)


def _field_readers(
    ser: serialbox.Serializer,
    savepoint: serialbox.Savepoint,
    grid: icon_grid.IconGrid,
    array_ns: types.ModuleType,
) -> tuple[Callable[[str], data_alloc.NDArray], Callable[[str], data_alloc.NDArray]]:
    """Readers of cell and edge fields. The serialized fields are padded to nproma."""

    def read_cell_k(name: str) -> data_alloc.NDArray:
        buffer = array_ns.squeeze(ser.read(name, savepoint).astype(float))
        return array_ns.asarray(buffer[: grid.num_cells, :])

    def read_edge_k(name: str) -> data_alloc.NDArray:
        buffer = array_ns.squeeze(ser.read(name, savepoint).astype(float))
        return array_ns.asarray(buffer[: grid.num_edges, :])

    return read_cell_k, read_edge_k


def _read_time_levels(
    ser: serialbox.Serializer,
    savepoint: serialbox.Savepoint,
    name: str,
    size: int,
    array_ns: types.ModuleType,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Read both time levels of a predictor-corrector field."""
    buffer = array_ns.squeeze(ser.read(name, savepoint).astype(float))
    return array_ns.asarray(buffer[:size, :, 0]), array_ns.asarray(buffer[:size, :, 1])


def _available_dates(ser: serialbox.Serializer) -> str:
    dates = sorted(
        {
            savepoint.metainfo["date"]
            for savepoint in ser.savepoint_list()
            if savepoint.name == "solve-nonhydro-init"
        }
    )
    return ", ".join(dates) if dates else "none"


def read_initial_condition_from_file(
    *,
    config: FromFileConfig,
    grid: icon_grid.IconGrid,
    prognostic_state_now: prognostics.PrognosticState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> None:
    """Initialise the prognostic state from the serialized ICON initial state."""
    array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))

    ser = _open_serializer(config.data_path, exchange.my_rank())
    savepoint = ser.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()
    read_cell_k, read_edge_k = _field_readers(ser, savepoint, grid, array_ns)

    prognostic_state_now.rho.ndarray[:, :] = read_cell_k("rho_now")
    prognostic_state_now.exner.ndarray[:, :] = read_cell_k("exner_now")
    prognostic_state_now.theta_v.ndarray[:, :] = read_cell_k("theta_v_now")
    prognostic_state_now.vn.ndarray[:, :] = read_edge_k("vn_now")
    prognostic_state_now.w.ndarray[:, :] = read_cell_k("w_now")

    if config.ntracer > 0:
        tracers = array_ns.squeeze(ser.read("tracers_now", savepoint).astype(float))
        for i, tracer in enumerate(prognostic_state_now.tracer.active_fields()):
            tracer.field.ndarray[:, :] = array_ns.asarray(tracers[: grid.num_cells, :, i])


def read_restart_from_file(
    *,
    config: FromFileConfig,
    grid: icon_grid.IconGrid,
    prognostic_state_now: prognostics.PrognosticState,
    dycore_initial_fields: ic_states.DycoreInitialFields,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> None:
    """
    Initialise the prognostic state from the serialized state of a later time step.

    On a restart (isRestart() in mo_nh_stepping.f90) ICON reads the prognostic
    variables, the perturbed exner pressure and the advective tendencies of the
    previous time step from its restart file, and skips both compute_exner_pert and
    the initial diffusion call. The same quantities are read here from the savepoints
    written at the beginning of the time step that starts at 'start_of_timestepping'.
    Those savepoints are stamped with the date of the end of their time step.
    """
    if config.start_of_timestepping is None or config.dtime is None:
        raise ValueError(
            "restarting needs 'start_of_timestepping' and 'dtime': the initial condition "
            "config was not initialized from a driver config."
        )
    if config.ntracer > 0:
        raise NotImplementedError(
            "restarting with tracers is not supported: the solve-nonhydro savepoints do "
            "not carry them."
        )

    array_ns = data_alloc.import_array_ns(model_backends.get_allocator(backend))
    date = _savepoint_date(config.start_of_timestepping + config.dtime)
    ser = _open_serializer(config.data_path, exchange.my_rank())
    log.info("Restarting from the serialized state of the time step ending at %s", date)

    try:
        nonhydro_savepoint = (
            ser.savepoint["solve-nonhydro-init"].istep[1].date[date].dyn_timestep[1].as_savepoint()
        )
        velocity_savepoint = (
            ser.savepoint["velocity-tendencies-init"]
            .istep[1]
            .date[date]
            .dyn_timestep[1]
            .as_savepoint()
        )
    except serialbox.SerialboxError as err:
        raise ValueError(
            f"there is no serialized state to restart from at {date}. The serialized data "
            f"of this experiment covers: {_available_dates(ser)}."
        ) from err

    read_cell_k, read_edge_k = _field_readers(ser, nonhydro_savepoint, grid, array_ns)

    prognostic_state_now.rho.ndarray[:, :] = read_cell_k("rho_now")
    prognostic_state_now.exner.ndarray[:, :] = read_cell_k("exner_now")
    prognostic_state_now.theta_v.ndarray[:, :] = read_cell_k("theta_v_now")
    prognostic_state_now.vn.ndarray[:, :] = read_edge_k("vn_now")
    prognostic_state_now.w.ndarray[:, :] = read_cell_k("w_now")
    dycore_initial_fields.perturbed_exner_at_cells_on_model_levels.ndarray[:, :] = read_cell_k(
        "exner_pr"
    )

    normal_wind_tendency = _read_time_levels(
        ser, velocity_savepoint, "ddt_vn_apc_pc", grid.num_edges, array_ns
    )
    vertical_wind_tendency = _read_time_levels(
        ser, velocity_savepoint, "ddt_w_adv_pc", grid.num_cells, array_ns
    )
    normal = dycore_initial_fields.normal_wind_advective_tendency
    vertical = dycore_initial_fields.vertical_wind_advective_tendency
    normal.predictor.ndarray[:, :] = normal_wind_tendency[0]
    normal.corrector.ndarray[:, :] = normal_wind_tendency[1]
    # The dycore swaps the vertical advective tendency at the first substep of a time
    # step that is not the initial one, and then consumes the predictor without
    # recomputing it. The two time levels are therefore stored swapped.
    vertical.predictor.ndarray[:, :] = vertical_wind_tendency[1]
    vertical.corrector.ndarray[:, :] = vertical_wind_tendency[0]
