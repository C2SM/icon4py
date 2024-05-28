# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import functools
from typing import Final

import cftime
import xarray


LEVEL_NAME: Final[str] = "model_level_number"
INTERFACE_LEVEL_NAME: Final[str] = "interface_model_level_number"
DEFAULT_CALENDAR: Final[str] = "proleptic_gregorian"
DEFAULT_TIME_UNIT: Final[str] = "seconds since 1970-01-01 00:00:00"

"""
CF conventions encourage to use the COARDS conventions for the order of the dimensions: 
    `T` (time), 
    `Z` (height or depth), 
    `Y` (latitude), 
    `X` (longitude).
In the unstructured case `Y` and `X`  combine to the horizontal dimension.
"""
COARDS_T_POS: Final[int] = 0
COARDS_Z_POS: Final[int] = 1
HORIZONTAL_POS: Final[int] = 2

date2num = functools.partial(cftime.date2num, units=DEFAULT_TIME_UNIT, calendar=DEFAULT_CALENDAR)
date2num.__doc__= """Convert a datetime object to a number.

Convenience method that sets units and calendar to the default values.
"""



def to_canonical_dim_order(data: xarray.DataArray) -> xarray.DataArray:
    """Check for spatial dimensions being in canonical order ('T', 'Z', 'Y', 'X') and return them in this order."""
    dims = data.dims
    if len(dims) >= 2:
        if dims[0] in ("cell", "edge", "vertex") and dims[1] in (
            "height",
            "level",
            "interface_level",
        ):
            return data.transpose(dims[1], dims[0], *dims[2:], transpose_coords=True)
        else:
            return data
