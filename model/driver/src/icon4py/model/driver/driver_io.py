# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver-side glue between the model state and the ``icon4py.model.common.io`` module.

:class:`OutputStateAssembler` assembles, at a capture step, exactly the variables the
capturing field groups request from their sources (``driver_output_sources`` and
``driver_diagnostics``), after :func:`validate_output_variables` checked the groups of
the ``IOConfig`` against them at setup; :func:`create_io_monitor` builds the output
monitor that writes one file series per field group: an ``IOMonitor`` when the ranks
write themselves, an ``IOClient`` shipping the fields to dedicated IO ranks
(``IOConfig.io_ranks_per_node``), whose side is :func:`run_io_server`.
"""

import logging
import pathlib
import uuid
from collections.abc import Collection, Iterable, Mapping

import gt4py.next as gtx
import xarray as xr

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import exceptions, model_backends, time
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.grid import base as grid_base, vertical as v_grid
from icon4py.model.common.io import distributed, io as common_io, io_servers
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    static_fields,
    tracer_states,
)
from icon4py.model.driver import (
    config as driver_config,
    driver_diagnostics,
    driver_output_sources,
    driver_utils,
)


log = logging.getLogger(__name__)


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
        "prognostic": driver_output_sources.PROGNOSTIC_VARIABLES,
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
        physics_diagnostics: Iterable[physics_state.DiagnosticOutput] | None,
    ) -> None:
        self._tracer_names: tuple[str, ...] = (
            tracer_config.active_names if tracer_config is not None else ()
        )
        self._physics_outputs = driver_output_sources.physics_diagnostic_outputs(
            physics_diagnostics
        )
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
        if prognostic := [
            name for name in driver_output_sources.PROGNOSTIC_VARIABLES if name in requested
        ]:
            state.update(
                driver_output_sources.prognostic_state_to_dataarrays(prognostic_state, prognostic)
            )
        if tracers := [name for name in self._tracer_names if name in requested]:
            state.update(driver_output_sources.tracer_state_to_dataarrays(tracer_state, tracers))
        if diagnostics := [
            name for name in driver_diagnostics.DIAGNOSTIC_VARIABLES if name in requested
        ]:
            assert self._diagnostics_computer is not None
            fields = self._diagnostics_computer.compute(
                prognostic_state, tracer_state, diagnostic_state, diagnostics
            )
            state.update(driver_diagnostics.diagnostic_fields_to_dataarrays(fields))
        if physics := [name for name in self._physics_outputs if name in requested]:
            state.update(
                driver_output_sources.physics_diagnostics_to_dataarrays(
                    self._physics_outputs, physics
                )
            )
        return state


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
    compute ranks', and the client ships the sizes of the undecomposed grid (summed
    over the compute communicator, so this is collective on it). None: the ranks write
    themselves (``IOMonitor``). Either way the split must agree with
    ``io_config.io_ranks_per_node``, the setting it was made for.
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
            # the rank's grid holds its halos, so its sizes differ between the compute
            # ranks; the IO rank gets the undecomposed sizes, equal on all of them
            horizontal_size=distributed.global_horizontal_size(process_props, decomposition_info),
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
