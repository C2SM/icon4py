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
the ``IOConfig`` against them at setup; :func:`create_io_monitor` builds the
``IOMonitor`` that writes one file series per field group. A variable is named by the
standard name of its quantity.
"""

import pathlib
import uuid
from collections.abc import Collection, Mapping

import gt4py.next as gtx
import xarray as xr

from icon4py.model.common import exceptions, field, time
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.framework import states
from icon4py.model.common.grid import base as grid_base, vertical as v_grid
from icon4py.model.common.io import io as common_io
from icon4py.model.common.states import static_fields
from icon4py.model.driver import driver_diagnostics, driver_output_sources as output_sources


def requested_output_variables(io_config: common_io.IOConfig) -> set[str]:
    """The union of the variables of all field groups."""
    return {name for group in io_config.field_groups for name in group.variables}


def output_variable_sources(
    *,
    tracer_config: states.TracerConfig | None,
    physics_fields: Mapping[str, field.Field],
) -> dict[str, list[str]]:
    """Every variable the run can write, with the names of the sources providing it.

    A variable is normally provided by exactly one source; one provided by several is
    ambiguous and rejected by :func:`validate_output_variables` if a field group requests it.
    """
    sources: dict[str, Collection[str]] = {
        "prognostic": output_sources.PROGNOSTIC_VARIABLES,
        "diagnostic": driver_diagnostics.DIAGNOSTIC_VARIABLES,
        "tracer": output_sources.tracer_variables(tracer_config),
        "physics": physics_fields,
    }
    providers: dict[str, list[str]] = {}
    for source, names in sources.items():
        for name in names:
            providers.setdefault(name, []).append(source)
    return providers


# NOTE: temporary, replaced in part 06 (io-config): `lookup` raises `UnknownQuantity` at init.
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


# NOTE: temporary, replaced in parts 02 (collect), 05 (derived inputs) and 06 (io-config): goes,
# its work split among `fw.collect`, `derived_by` and the IOMonitor's Input.
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
        tracer_config: states.TracerConfig | None,
        physics_fields: Mapping[str, field.Field] | None,
    ) -> None:
        self._physics_fields: Mapping[str, field.Field] = physics_fields or {}
        sources = output_variable_sources(
            tracer_config=tracer_config, physics_fields=self._physics_fields
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
        prognostic_state: states.PrognosticState,
        tracer_state: states.TracerState,
        diagnostic_state: states.Diagnostics,
    ) -> dict[str, xr.DataArray]:
        """The CF/UGRID-annotated DataArrays of exactly the requested variables.

        The requested standard diagnostics are computed into ``diagnostic_state`` first.
        The DataArrays reference the live state (see ``io.utils.to_data_array``): they
        must be written before the next step mutates it.
        """
        # NOTE: temporary, replaced in part 02 (collect): the leaves are collected by quantity
        requested = set(variables)
        fields = {
            name: field_
            for source in (
                output_sources.by_standard_name(prognostic_state),
                output_sources.by_standard_name(tracer_state),
                self._physics_fields,
            )
            for name, field_ in source.items()
            if name in requested
        }
        # NOTE: temporary, replaced in part 05 (derived inputs): `derived_by` on the IOMonitor's Input
        if diagnostics := [
            name for name in driver_diagnostics.DIAGNOSTIC_VARIABLES if name in requested
        ]:
            assert self._diagnostics_computer is not None
            fields.update(
                self._diagnostics_computer.compute(
                    prognostic_state, tracer_state, diagnostic_state, diagnostics
                )
            )
        # NOTE: temporary, replaced in part 06 (io-config): moves into `IOMonitor.run`
        return output_sources.to_dataarrays(fields)


def create_io_monitor(
    *,
    io_config: common_io.IOConfig,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    dtime: time.RelativeTime,
    process_props: decomposition_defs.ProcessProperties,
    decomposition_info: decomposition_defs.DecompositionInfo | None,
) -> common_io.IOMonitor:
    """Build the ``IOMonitor`` writing the field groups of ``io_config`` below its
    ``output_path``.

    In a distributed run (multi-rank ``process_props``) ``decomposition_info`` is
    required; each group's ``mode`` selects how the ranks write (see
    ``common_io.OutputMode``).
    """
    return common_io.IOMonitor(
        config=io_config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
        grid_id=uuid.UUID(grid.id),
        dtime=dtime,
        process_props=process_props,
        decomposition_info=decomposition_info,
    )
