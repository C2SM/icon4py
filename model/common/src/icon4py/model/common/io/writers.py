# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared surface of the output writers.

The writer implementations live in :mod:`icon4py.model.common.io.netcdf_writers`
(serial netCDF files) and :mod:`icon4py.model.common.io.zarr_writers` (serial and
rank-block zarr stores). This module holds what they share so both file formats
stay identical: the :class:`FieldWriter` protocol, the dimension names, and the
coordinate/variable attributes.
"""

import dataclasses
import datetime as dt
import types
import uuid
from typing import Final, Protocol, Required, Self, TypedDict

import xarray as xr

import icon4py.model.common.states.metadata
from icon4py.model.common.grid import base
from icon4py.model.common.io import cf_utils


EDGE: Final[str] = "edge"
VERTEX: Final[str] = "vertex"
CELL: Final[str] = "cell"
MODEL_HALF_LEVEL: Final[str] = "half_level"
MODEL_LEVEL: Final[str] = "level"
TIME: Final[str] = "time"

#: Prefix of the global-index coordinates of rank-block distributed zarr stores.
GLOBAL_INDEX_PREFIX: Final[str] = "global_index"


class GlobalFileAttributes(TypedDict, total=False):
    """
    Global file attributes of an ICON generated netCDF file.

    Attribute map what ICON produces, (including the upper, lower case pattern).
    Omissions (possibly incomplete):
    - 'CDI' used for the supported CDI version (http://mpimet.mpg.de/cdi) since we do not support it

    Additions:
    - 'external_variables': variable used by CF conventions if cell_measure variables are used from an external file'
    """

    #: version of the supported CF conventions
    Conventions: Required[str]  # TODO(halungge): check changelog? latest version is 1.11

    #: unique id of the horizontal grid used in the simulation (from grid file)
    uuidOfHGrid: Required[uuid.UUID]

    #: institution name
    institution: Required[str]

    #: title of the file or simulation
    title: Required[str]

    #: source code repository
    source: Required[str]

    #: path of the binary and generation time stamp of the file
    history: Required[str]

    #: references for publication # TODO(halungge): check if this is the right reference
    references: str
    comment: str
    external_variables: str


@dataclasses.dataclass
class TimeProperties:
    units: str
    calendar: str


class FieldWriter(Protocol):
    """Writer for one output file: create it, append time slices to it, close it."""

    def initialize_dataset(self) -> None: ...

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...


def horizontal_axis_sizes(horizontal: base.HorizontalGridSize) -> dict[str, int]:
    return {
        CELL: horizontal.num_cells,
        EDGE: horizontal.num_edges,
        VERTEX: horizontal.num_vertices,
    }


# ------------------------------------------------------------------------------------
# Coordinate attributes, shared between the writers so the file formats stay identical
# ------------------------------------------------------------------------------------


def time_attributes(time_properties: TimeProperties) -> dict[str, str]:
    return {
        "units": time_properties.units,
        "axis": cf_utils.COARDS_TIME_COORDINATE_NAME,
        "calendar": time_properties.calendar,
        "standard_name": TIME,
        "long_name": TIME,
    }


LEVEL_ATTRIBUTES: Final[dict[str, str]] = {
    "units": "1",
    "positive": "down",
    "long_name": "model full level index",
    "standard_name": cf_utils.LEVEL_STANDARD_NAME,
}

HALF_LEVEL_ATTRIBUTES: Final[dict[str, str]] = {
    "units": "1",
    "positive": "down",
    "long_name": "model half level index",
    "standard_name": icon4py.model.common.states.metadata.INTERFACE_LEVEL_STANDARD_NAME,
}

HEIGHT_ATTRIBUTES: Final[dict[str, str]] = {
    "units": "m",
    "positive": "up",
    "axis": cf_utils.COARDS_VERTICAL_COORDINATE_NAME,
    "long_name": "height value of half levels without topography",
    "standard_name": icon4py.model.common.states.metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME,
}

#: CF/UGRID attributes carried from a field's DataArray onto its file variable.
DATA_VARIABLE_ATTRIBUTES: Final[tuple[str, ...]] = (
    "units",
    "standard_name",
    "long_name",
    "coordinates",
    "mesh",
    "location",
)


def data_variable_attributes(canonical_slice: xr.DataArray) -> dict[str, str]:
    return {name: getattr(canonical_slice, name) for name in DATA_VARIABLE_ATTRIBUTES}


def filter_by_standard_name(model_state: dict, value: str) -> dict:
    return {k: v for k, v in model_state.items() if value == v.standard_name}
