# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Parsing of the time formats used in the ICON namelists."""

from __future__ import annotations

import re

from icon4py.model.common import time


# ISO 8601 duration, restricted to the fixed-length components (weeks, days,
# hours, minutes, seconds). Years and months are intentionally not matched since
# their length is not fixed.
_ISO8601_DURATION = re.compile(
    r"P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?=\d)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
)


def relativetime_from_iso8601(duration: str) -> time.RelativeTime:
    """
    Parse an ISO 8601 duration such as 'PT300S' into a 'time.RelativeTime'.

    Only the components convertible to a fixed duration are supported (weeks,
    days, hours, minutes, seconds).
    """
    match = _ISO8601_DURATION.fullmatch(duration)
    if match is None or not any(match.groups()):
        raise ValueError(f"Invalid ISO 8601 duration: '{duration}'.")
    components = {name: float(value) for name, value in match.groupdict().items() if value}
    return time.RelativeTime(**components)
