# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared surface of the output writers.

The writer implementations live in :mod:`icon4py.model.common.io.netcdf_writers`
and :mod:`icon4py.model.common.io.zarr_writers`, each supporting serial and
rank-block output. This module holds what they share so both file formats stay
identical: the :class:`FieldWriter` protocol, the dimension names, and the
coordinate/variable attributes.
"""

import dataclasses
import datetime as dt
import types
import uuid
from typing import Final, Protocol, Required, Self, TypedDict

import numpy as np
import xarray as xr

from icon4py.model.common.grid import base
from icon4py.model.common.io import cf_utils
from icon4py.model.common.states import metadata
from icon4py.model.common.utils import data_allocation as data_alloc


EDGE: Final[str] = "edge"
VERTEX: Final[str] = "vertex"
CELL: Final[str] = "cell"
MODEL_HALF_LEVEL: Final[str] = "half_level"
MODEL_LEVEL: Final[str] = "level"
TIME: Final[str] = "time"

#: Prefix of the global-index coordinates of rank-block output (zarr and netCDF).
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
# (the static per-axis attribute dicts live in ``common.states.metadata``)
# ------------------------------------------------------------------------------------


def time_attributes(time_properties: TimeProperties) -> dict[str, str]:
    return {
        "units": time_properties.units,
        "axis": metadata.COARDS_TIME_COORDINATE_NAME,
        "calendar": time_properties.calendar,
        "standard_name": TIME,
        "long_name": TIME,
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

#: The attributes asserting the UGRID mesh association of a variable. The association
#: holds only when the variable's horizontal axis is in the global order of the
#: referenced topology -- which a rank-block store's axes are not.
UGRID_ASSOCIATION_ATTRIBUTES: Final[tuple[str, ...]] = ("mesh", "location")

#: Marker attribute replacing the UGRID association on rank-block variables: their
#: horizontal axes are rank-ordered and padded (see ``distributed.RankBlock``), so a
#: reader must reorder them by the store's ``global_index_<dim>`` coordinate before
#: the mesh of the UGRID topology file applies.
LAYOUT_ATTRIBUTE: Final[str] = "icon4py_layout"
RANK_BLOCK_LAYOUT: Final[str] = "rank_block"


def data_variable_attributes(
    canonical_slice: xr.DataArray, *, rank_block_layout: bool = False
) -> dict[str, str]:
    """CF/UGRID attributes of a field, raising for missing ones.

    Both writers call this before any file mutation: a missing attribute must fail on
    every rank identically, not on the root rank inside a store operation.

    With ``rank_block_layout`` the UGRID association (``mesh``, ``location``) is
    replaced by ``icon4py_layout = "rank_block"``: the variable's horizontal axis is
    rank-ordered and padded, so a UGRID-aware reader would otherwise silently place
    the values on the wrong mesh entities (see ``LAYOUT_ATTRIBUTE``).
    """
    missing = [name for name in DATA_VARIABLE_ATTRIBUTES if name not in canonical_slice.attrs]
    if missing:
        raise ValueError(f"Field is missing the CF attributes: {', '.join(missing)}.")
    attrs = {name: canonical_slice.attrs[name] for name in DATA_VARIABLE_ATTRIBUTES}
    if rank_block_layout:
        for name in UGRID_ASSOCIATION_ATTRIBUTES:
            del attrs[name]
        attrs[LAYOUT_ATTRIBUTE] = RANK_BLOCK_LAYOUT
    return attrs


def canonicalize_time_slice(
    state_to_append: dict[str, xr.DataArray], horizontal_sizes: dict[str, int]
) -> tuple[dict[str, xr.DataArray], dict[str, np.ndarray]]:
    """Canonicalize the fields of a time slice and transfer them to host memory.

    Runs before any file/store mutation, on every rank: a failure (unsupported or
    unknown dimensions, missing CF attributes, a device buffer that cannot be
    converted) must raise identically on all ranks first, or the file would be left
    with a phantom time slice -- and, in rank-block mode, the surviving ranks would
    hang in the next collective.
    """
    canonical_slices: dict[str, xr.DataArray] = {}
    host_data: dict[str, np.ndarray] = {}
    for var_name, new_slice in state_to_append.items():
        canonical_slice = cf_utils.to_canonical_dim_order(new_slice)
        if canonical_slice is None:
            raise ValueError(
                f"Cannot write field '{var_name}': only fields with a horizontal and a "
                f"vertical dimension are supported."
            )
        horizontal_name = str(canonical_slice.dims[-1])
        if horizontal_name not in horizontal_sizes:
            raise ValueError(
                f"Cannot write field '{var_name}': unknown horizontal dimension "
                f"'{horizontal_name}'."
            )
        try:
            data_variable_attributes(canonical_slice)
        except ValueError as err:
            raise ValueError(f"Cannot write field '{var_name}': {err}") from err
        canonical_slices[var_name] = canonical_slice
        host_data[var_name] = data_alloc.as_numpy(canonical_slice.data)
    return canonical_slices, host_data
