# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4pytools.icon4pygen.exceptions import InvalidConnectivityException
from icon4pytools.icon4pygen.metadata import provide_offset


# TODO (halungge) that test is in the wrong file: should go to test_metadata.py
@pytest.mark.parametrize(
    ("chain", "expected"),
    [
        ("C2E", 3),
        ("C2V", 3),
        ("E2C", 2),
        ("E2V", 2),
        ("V2C", 6),
        ("V2E", 6),
        ("V2E2V", 6),
        ("E2ECV", 4),
        ("E2EC", 2),
        ("E2C2E", 4),
        ("E2C2EO", 5),
        ("E2C2V", 4),
        ("C2E2C", 3),
        ("C2CEC", 3),
        ("C2E2CO", 4),
        ("C2E2C2E", 9),
        ("C2E2C2E2C", 9),
        ("C2CECEC", 9),
    ],
)
def test_chainsize_neighbors(chain, expected):
    actual = provide_offset(chain)
    assert actual.max_neighbors == expected


@pytest.mark.xfail(
    reason="Test will fail with an AttributeError as InvalidConnectivityException has become unreachable."
)
def test_unsupported_connectivity_type():
    with pytest.raises(InvalidConnectivityException):
        provide_offset("E2X")
