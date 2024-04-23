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

from typing import Final

import cftime


LEVEL_NAME: Final[str] = "model_level_number"
INTERFACE_LEVEL_NAME: Final[str] = "interface_model_level_number"
DEFAULT_CALENDAR: Final[str] = "proleptic_gregorian"
DEFAULT_TIME_UNIT: Final[str] = "seconds since 1970-01-01 00:00:00"

"""
CF conventions encourage to use the COARDS conventions for the order of the dimensions: `T` (time), `Z` (height or depth), `Y` (latitude), `X` (longitude).
In the unstructured case `Y` and `X`  combine to the horizontal dimension.
"""
COARDS_T_POS: Final[int] = 0
COARDS_Z_POS: Final[int] = 1
HORIZONTAL_POS: Final[int] = 2


def date2num(date, units=DEFAULT_TIME_UNIT, calendar=DEFAULT_CALENDAR):
    """

    Convert a datetime object to a number.

    Convenience method that makes units and calendar optional and uses the default values.
    """
    return cftime.date2num(date, units=units, calendar=calendar)
