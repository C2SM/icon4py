# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import datetime

import pytest

from icon4py.model.common.utils import time_utils


@pytest.mark.parametrize(
    ("duration", "expected_seconds"),
    [
        ("PT300S", 300.0),
        ("PT1H", 3600.0),
        ("PT10M", 600.0),
        ("PT1H30M", 5400.0),
        ("P1DT6H", 108000.0),
        ("PT0.5S", 0.5),
    ],
)
def test_relativetime_from_iso8601_valid(duration: str, expected_seconds: float) -> None:
    assert time_utils.relativetime_from_iso8601(duration) == datetime.timedelta(
        seconds=expected_seconds
    )


@pytest.mark.parametrize("duration", ["", "P", "PT", "P1Y", "P1M", "300", "PT300", "P1DT", "P1WT"])
def test_relativetime_from_iso8601_invalid(duration: str) -> None:
    with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
        time_utils.relativetime_from_iso8601(duration)
