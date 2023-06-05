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

import pytest

from icon4pytools.icon4pygen.exceptions import InvalidConnectivityException
from icon4pytools.icon4pygen.metadata import provide_offset


@pytest.mark.parametrize(
    ("chain", "expected"),
    [
        ("C2E", 3),
        ("C2V", 3),
        ("E2C", 2),
        ("E2V", 2),
        ("V2C", 6),
        ("V2E", 6),
        ("E2C2E", 4),
        ("E2C2EO", 5),
        ("E2C2V", 4),
        ("E2C2V2C", 16),
        ("C2V2C", 12),
        ("C2V2CO", 13),
        ("C2V2C2E", 24),
        ("E2V2E", 10),
        ("E2V2EO", 11),
        ("E2V2E2C", 10),
        ("V2E2C", 6),
        ("V2E2C2V", 6),
        ("V2E2C2VO", 7),
        ("V2E2C2V2E", 30),
        ("V2E2C2V2E2C", 24),
        ("C2E2C", 3),
        ("C2E2CO", 4),
        ("C2E2C2E", 9),
        ("C2E2C2E2C", 9),
        ("C2E2C2E2CO", 10),
        ("C2E2C2E2C2E", 21),
    ],
)
def test_chainsize_neighbors(chain, expected):
    actual = provide_offset(chain)
    assert actual.max_neighbors == expected


def test_unsupported_connectivity_type():
    with pytest.raises(InvalidConnectivityException):
        provide_offset("E2X")
