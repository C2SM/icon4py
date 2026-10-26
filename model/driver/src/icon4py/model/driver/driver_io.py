# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver-side glue between the model state and the ``icon4py.model.common.io`` module.

The variables a run can write come from four sources:

- the prognostic state (:func:`prognostic_state_to_dataarrays`),
- the active tracers (:func:`tracer_state_to_dataarrays`),
- the standard diagnostics diagnosed from the prognostic state and the tracers
  (``driver_diagnostics``),
- the diagnostic outputs of the physics processes, such as the precipitation fluxes
  (:func:`physics_diagnostic_outputs`, :func:`physics_diagnostics_to_dataarrays`).

:func:`validate_output_variables` checks the field groups of an ``IOConfig`` against
these sources at setup; :class:`OutputStateAssembler` assembles, at a capture step,
exactly the variables the capturing groups request, and :func:`create_io_monitor`
builds the output monitor that writes one file series per field group: an
``IOMonitor`` when the ranks write themselves, an ``IOClient`` shipping the fields to
dedicated IO ranks (``IOConfig.num_io_ranks``), whose side is :func:`run_io_server`.
"""

import logging
import pathlib
import uuid
from collections.abc import Collection, Mapping
from typing import Final

import gt4py.next as gtx
import xarray as xr

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import exceptions, model_backends, time
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.grid import base as grid_base, vertical as v_grid
from icon4py.model.common.io import io as common_io, io_servers, utils as io_utils
from icon4py.model.common.states import (
    data as state_data,
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    static_fields,
    tracer_states,
)
from icon4py.model.driver import config as driver_config, driver_diagnostics, driver_utils


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Prognostic fields
# --------------------------------------------------------------------------------------


#: Prognostic output variables, selected by CF name from the
#: ``states.data.PROGNOSTIC_CF_ATTRIBUTES`` catalog (which also holds fields the driver does
#: not output, e.g. ``tangential_velocity``). The metadata and the state attribute
#: (``icon_var_name``) come from that catalog; this list only selects which entries to emit.
PROGNOSTIC_VARIABLES: Final[list[str]] = [
    "air_density",
    "exner_function",
    "virtual_potential_temperature",
    "upward_air_velocity",
    "normal_velocity",
]


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: Collection[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    selected = PROGNOSTIC_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for name in selected:
        try:
            metadata = state_data.PROGNOSTIC_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown prognostic output variable '{name}'. "
                f"Known variables are: {PROGNOSTIC_VARIABLES}."
            ) from err
        assert metadata.icon_var_name is not None, (
            f"prognostic output '{name}' must declare icon_var_name to be read from the state"
        )
        field = getattr(prognostic_state, metadata.icon_var_name)
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Tracers
# --------------------------------------------------------------------------------------


def tracer_state_to_dataarrays(
    tracer_state: tracer_states.TracerState, variables: Collection[str]
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays of the requested active tracers.

    The variables are the keys of ``states.data.COMMON_TRACER_CF_ATTRIBUTES`` (``qv``,
    ``qc``, ...), which are also the names of the ``TracerState`` fields.
    """
    active: dict[str, gtx.Field] = {
        tracer.name: tracer.field for tracer in tracer_state.active_fields()
    }
    state: dict[str, xr.DataArray] = {}
    for name in variables:
        try:
            metadata = state_data.COMMON_TRACER_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown tracer output variable '{name}'. Known tracers are: "
                f"{', '.join(state_data.COMMON_TRACER_CF_ATTRIBUTES)}."
            ) from err
        if name not in active:
            raise ValueError(f"Tracer output variable '{name}' is not active in the tracer state.")
        state[name] = io_utils.to_data_array(active[name], metadata, to_host=True)
    return state


# --------------------------------------------------------------------------------------
# Physics diagnostics
# --------------------------------------------------------------------------------------


def physics_diagnostic_outputs(
    physics_diagnostics: physics_state.DiagnosticsStore | None,
) -> dict[str, physics_state.DiagnosticOutput]:
    """The diagnostic outputs of the physics processes offered as output variables.

    Keyed by the CF ``standard_name`` of the output's metadata, e.g. ``precipitation_flux``
    for the ``pflx`` output of muphys; surface (purely horizontal) outputs are offered like
    the others. Without physics there are none. The buffers hold the values of the last
    step the process ran, and zeros before that.
    """
    outputs: dict[str, physics_state.DiagnosticOutput] = {}
    if physics_diagnostics is None:
        return outputs
    for output in physics_diagnostics.outputs():
        name = output.metadata.standard_name
        if name in outputs:
            raise ValueError(
                f"Physics diagnostic output '{name}' is provided by more than one process: "
                f"{outputs[name].process_name}, {output.process_name}."
            )
        outputs[name] = output
    return outputs


def physics_diagnostics_to_dataarrays(
    outputs: Mapping[str, physics_state.DiagnosticOutput], variables: Collection[str]
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays of the requested physics diagnostics."""
    return {
        name: io_utils.to_data_array(outputs[name].field, outputs[name].metadata, to_host=True)
        for name in variables
    }


# --------------------------------------------------------------------------------------
# Output configuration
# --------------------------------------------------------------------------------------


def requested_output_variables(io_config: common_io.IOConfig) -> set[str]:
    """The union of the variables of all field groups."""
    return {name for group in io_config.field_groups for name in group.variables}


def output_variable_sources(
    *,
    tracer_config: tracer_states.TracerConfig | None,
    physics_outputs: Mapping[str, physics_state.DiagnosticOutput],
) -> dict[str, list[str]]:
    """Every variable the run can write, with the names of the sources providing it.

    A variable is normally provided by exactly one source; a physics output whose
    standard name coincides with another source's variable name is ambiguous and
    rejected by :func:`validate_output_variables` if a field group requests it.
    """
    sources: dict[str, Collection[str]] = {
        "prognostic": PROGNOSTIC_VARIABLES,
        "diagnostic": driver_diagnostics.DIAGNOSTIC_VARIABLES,
        "tracer": tracer_config.active_names if tracer_config is not None else (),
        "physics": physics_outputs,
    }
    providers: dict[str, list[str]] = {}
    for source, names in sources.items():
        for name in names:
            providers.setdefault(name, []).append(source)
    return providers


def validate_output_variables(
    io_config: common_io.IOConfig, sources: Mapping[str, Collection[str]]
) -> None:
    """Fail at setup if a field group requests a variable the run does not provide.

    ``sources`` maps every available variable to the sources providing it (see
    :func:`output_variable_sources`); a requested variable with several sources is
    rejected as ambiguous.
    """
    for group in io_config.field_groups:
        if unknown := sorted(set(group.variables) - set(sources)):
            raise exceptions.InvalidConfigError(
                f"Unknown output variables in field group '{group.basename}': "
                f"{', '.join(unknown)}. Available output variables are: "
                f"{', '.join(sorted(sources))}."
            )
        for name in group.variables:
            if len(sources[name]) > 1:
                raise exceptions.InvalidConfigError(
                    f"Output variable '{name}' of field group '{group.basename}' is "
                    f"ambiguous: provided by the {' and the '.join(sources[name])} fields."
                )


class OutputStateAssembler:
    """Assembles the output state the capturing field groups request from its sources.

    The available variables are the prognostics, the standard diagnostics, the active
    tracers and the physics diagnostics of the run; the field groups are validated
    against them on construction, so an unknown variable fails before the first time
    step. The diagnostics computer is only built if some group requests a diagnostic.
    """

    def __init__(
        self,
        *,
        io_config: common_io.IOConfig,
        grid: grid_base.Grid,
        backend: gtx.typing.Backend | None,
        static_field_factories: static_fields.StaticFieldFactories,
        tracer_config: tracer_states.TracerConfig | None,
        physics_diagnostics: physics_state.DiagnosticsStore | None,
    ) -> None:
        self._tracer_names: tuple[str, ...] = (
            tracer_config.active_names if tracer_config is not None else ()
        )
        self._physics_outputs = physics_diagnostic_outputs(physics_diagnostics)
        sources = output_variable_sources(
            tracer_config=tracer_config, physics_outputs=self._physics_outputs
        )
        validate_output_variables(io_config, sources)
        self._available = frozenset(sources)

        requested = requested_output_variables(io_config)
        requested_diagnostics = [
            name for name in driver_diagnostics.DIAGNOSTIC_VARIABLES if name in requested
        ]
        self._diagnostics_computer: driver_diagnostics.DiagnosticsComputer | None = (
            driver_diagnostics.DiagnosticsComputer(
                grid=grid,
                backend=backend,
                variables=requested_diagnostics,
                metrics=static_field_factories.metrics,
                interpolation=static_field_factories.interpolation,
            )
            if requested_diagnostics
            else None
        )

    @property
    def available_variables(self) -> frozenset[str]:
        """The variables this run can write."""
        return self._available

    def assemble(
        self,
        variables: Collection[str],
        prognostic_state: prognostics.PrognosticState,
        tracer_state: tracer_states.TracerState,
        diagnostic_state: diagnostics.DiagnosticState,
    ) -> dict[str, xr.DataArray]:
        """The CF/UGRID-annotated DataArrays of exactly the requested variables.

        The requested standard diagnostics are computed into ``diagnostic_state`` first.
        The DataArrays reference the live state (see ``io.utils.to_data_array``): they
        must be written before the next step mutates it.
        """
        requested = set(variables)
        state: dict[str, xr.DataArray] = {}
        if prognostic := [name for name in PROGNOSTIC_VARIABLES if name in requested]:
            state.update(prognostic_state_to_dataarrays(prognostic_state, prognostic))
        if tracers := [name for name in self._tracer_names if name in requested]:
            state.update(tracer_state_to_dataarrays(tracer_state, tracers))
        if diagnostics := [
            name for name in driver_diagnostics.DIAGNOSTIC_VARIABLES if name in requested
        ]:
            assert self._diagnostics_computer is not None
            fields = self._diagnostics_computer.compute(
                prognostic_state, tracer_state, diagnostic_state, diagnostics
            )
            state.update(driver_diagnostics.diagnostic_fields_to_dataarrays(fields))
        if physics := [name for name in self._physics_outputs if name in requested]:
            state.update(physics_diagnostics_to_dataarrays(self._physics_outputs, physics))
        return state


# --------------------------------------------------------------------------------------
# Monitor factory
# --------------------------------------------------------------------------------------


def create_io_monitor(
    *,
    io_config: common_io.IOConfig,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    dtime: time.RelativeTime,
    process_props: decomposition_defs.ProcessProperties,
    decomposition_info: decomposition_defs.DecompositionInfo | None,
    io_ranks: mpi_decomp.IORankSplit | None,
) -> common_io.OutputMonitor:
    """Build the output monitor writing the field groups of ``io_config`` below its
    ``output_path``.

    In a distributed run (multi-rank ``process_props``) ``decomposition_info`` is
    required; each group's ``mode`` selects how the ranks write (see
    ``common_io.OutputMode``).

    With dedicated IO ranks (``io_ranks`` given, on a compute rank) the monitor is an
    ``IOClient`` shipping the fields to this rank's IO rank, which writes them with
    this very configuration (see ``run_io_server``); ``process_props`` are then the
    compute ranks'. None: the ranks write themselves (``IOMonitor``). Either way the
    split must agree with ``io_config.io_ranks_per_node``, the setting it was made for.
    """
    if (io_ranks is None) != (io_config.io_ranks_per_node == 0):
        raise ValueError(
            "The communicator split and 'io_ranks_per_node' of the output configuration "
            "must agree: dedicated IO ranks with both or with neither."
        )
    # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
    grid_id = uuid.UUID(grid.id)
    if io_ranks is not None:
        if decomposition_info is None:
            raise ValueError("Output through dedicated IO ranks requires 'decomposition_info'.")
        if io_ranks.io_rank is None:
            raise ValueError(
                "The output monitor is built on the compute ranks; this rank is an IO rank."
            )
        return io_servers.IOClient(
            config=io_config,
            dtime=dtime,
            grid_id=grid_id,
            horizontal_size=grid.config.horizontal_config,
            decomposition_info=decomposition_info,
            intercomm=io_ranks.intercomm,
            io_rank=io_ranks.io_rank,
        )
    return common_io.IOMonitor(
        config=io_config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        grid_id=grid_id,
        dtime=dtime,
        process_props=process_props,
        decomposition_info=decomposition_info,
    )


def run_io_server(
    *,
    config: driver_config.ExperimentConfig,
    grid_file_path: pathlib.Path,
    io_ranks: mpi_decomp.IORankSplit,
) -> None:
    """Run this dedicated IO rank: write the compute ranks' output until they finish.

    Returns once every served compute rank closed its output. The compute ranks define
    the output (their ``IOClient`` ships the output configuration and the grid
    identity); this rank contributes only what the writers need locally: the vertical
    grid, built on host from the experiment configuration (an IO rank never touches a
    device), and the grid file for the UGRID topology. Raises on any IO-side failure;
    the caller must abort the run, since the compute ranks cannot observe the failure
    (see ``io_servers``).
    """
    server = io_servers.IOServer(io_ranks.intercomm, served_ranks=io_ranks.served_compute_ranks)
    setup = server.receive_setup()
    log.info(
        f"IO rank {io_ranks.process_props.rank} serves compute ranks {list(server.served_ranks)}"
    )
    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=config.vertical_grid,
        allocator=model_backends.get_allocator(None),
    )
    io_monitor = common_io.IOMonitor(
        config=setup.config,
        vertical_size=vertical_grid,
        horizontal_size=setup.horizontal_size,
        grid_file_name=grid_file_path,
        grid_id=setup.grid_id,
        dtime=config.driver.dtime,
        process_props=io_ranks.process_props,
        decomposition_info=setup.decomposition_info,
    )
    server.serve(io_monitor)
