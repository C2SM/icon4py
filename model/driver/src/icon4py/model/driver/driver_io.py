# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The driver's output: the IO component and its factory.

`IOMonitor` declares every leaf the driver can write; the config selects the variables by
name and `run` hands them, as CF/UGRID-annotated DataArrays, to the common writer
(`icon4py.model.common.io.io.IOMonitor`). The diagnostics among them are derived by the
driver before each store.
"""

import datetime
import pathlib
import uuid
from typing import Any, Final

import xarray as xr

from icon4py.model.common import time
from icon4py.model.common.components import framework as fw, quantities as qty
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base as grid_base, vertical as v_grid
from icon4py.model.common.io import io as common_io, utils as io_utils


# file-name stub of the output file (a counter and the backend's suffix are appended)
DEFAULT_OUTPUT_BASENAME: Final[str] = "icon4py_output"

DEFAULT_OUTPUT_VARIABLES: Final[list[str]] = [
    "air_density",
    "exner_function",
    "virtual_potential_temperature",
    "upward_air_velocity",
    "normal_velocity",
    "eastward_wind",
    "northward_wind",
    "temperature",
    "virtual_temperature",
    "pressure",
]

# A variable is named by its quantity's CF standard_name, except these leaves: the output
# files name them after the keys of the states/data.py CF tables they were written from
# (`exner_function`, not `dimensionless_exner_function`). Kept for file compatibility.
_FILE_NAMES: Final[dict[str, str]] = {
    "exner": "exner_function",
    "temperature": "temperature",
    "virtual_temperature": "virtual_temperature",
    "pressure": "pressure",
}


def _attributes(quantity: type[fw.Quantity]) -> dict[str, str]:
    """The CF attributes of a variable, from its quantity."""
    attributes = {
        "standard_name": quantity.standard_name,
        "long_name": quantity.long_name,
        "units": quantity.units,
    }
    return {key: value for key, value in attributes.items() if value is not None}


class IOMonitor(fw.Component):
    """
    Writes the selected variables through the common writer.

    The writer's schedule advances on every `store`, so `run` must be called at every step;
    the DataArrays are built only when the writer captures.
    """

    class Input(fw.State):
        rho: fw.Field[qty.RhoOnCellK]
        exner: fw.Field[qty.ExnerOnCellK]
        theta_v: fw.Field[qty.ThetaVOnCellK]
        w: fw.Field[qty.WOnCellKHalf]
        vn: fw.Field[qty.VnOnEdgeK]
        u: fw.Field[qty.UOnCellK]
        v: fw.Field[qty.VOnCellK]
        temperature: fw.Field[qty.TemperatureOnCellK]
        virtual_temperature: fw.Field[qty.VirtualTemperatureOnCellK]
        pressure: fw.Field[qty.PressureOnCellK]
        simulation_time: datetime.datetime

    Output = fw.Empty

    def __init__(
        self,
        *,
        grid: grid_base.Grid,
        writer: common_io.IOMonitor,
        variables: list[str],
    ) -> None:
        super().__init__(grid, None)
        self.writer = writer
        leaves: dict[str, str] = {}
        for declaration in IOMonitor.Input.declarations():
            name = _FILE_NAMES.get(declaration.name, declaration.quantity.standard_name)
            assert name is not None, f"Output leaf '{declaration.name}' has no name."
            leaves[name] = declaration.name
        unknown = [name for name in variables if name not in leaves]
        if unknown:
            raise ValueError(
                f"Unknown output variable '{unknown[0]}'. Known variables are: {list(leaves)}."
            )
        # output variable name -> Input leaf name
        self.selected = {name: leaves[name] for name in variables}

    def run(self, inputs: Input, out: fw.Empty | None = None) -> fw.Empty:
        state: dict[str, xr.DataArray] = {}
        if self.writer.at_capture_time():
            for name, leaf in self.selected.items():
                field: fw.Field[Any] = getattr(inputs, leaf)
                state[name] = io_utils.to_data_array(
                    field.data, _attributes(field.quantity), to_host=True
                )
        self.writer.store(state, inputs.simulation_time)
        return self.buffers(out)


def create_io_monitor(
    *,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    dtime: datetime.timedelta,
    variables: list[str] | None = None,
    output_interval: common_io.OutputInterval = time.NumTimeSteps(1),
    output_backend: common_io.OutputBackend = common_io.OutputBackend.ZARR,
    output_mode: common_io.OutputMode = common_io.OutputMode.DISTRIBUTED,
    process_props: decomposition_defs.ProcessProperties,
    decomposition_info: decomposition_defs.DecompositionInfo | None,
) -> IOMonitor:
    """Build an ``IOMonitor`` writing through one field group that holds all output fields.

    ``output_interval`` is either a number of model steps or a simulation-time delta
    (normalized to steps using ``dtime``); it defaults to every step. In a distributed
    run (multi-rank ``process_props``) ``decomposition_info`` is required and
    ``output_mode`` selects how the ranks write (see ``common_io.OutputMode``).
    """
    output_variables = DEFAULT_OUTPUT_VARIABLES if variables is None else variables

    field_groups = [
        common_io.FieldGroupIOConfig(
            output_interval=output_interval,
            basename=DEFAULT_OUTPUT_BASENAME,
            variables=output_variables,
            backend=output_backend,
            mode=output_mode,
            nc_title="ICON4Py output",
            nc_comment="Fields computed by ICON4Py.",
        )
    ]

    config = common_io.IOConfig(output_path=str(output_path), field_groups=field_groups)
    writer = common_io.IOMonitor(
        config=config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
        grid_id=uuid.UUID(grid.id),
        dtime=dtime,
        process_props=process_props,
        decomposition_info=decomposition_info,
    )
    return IOMonitor(grid=grid, writer=writer, variables=output_variables)
