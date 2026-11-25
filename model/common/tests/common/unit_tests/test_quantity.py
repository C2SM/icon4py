# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from tests.common.unit_tests.utils import CELL_K, Pressure, Salt, Vt


def test_quantity_is_a_tag_with_metadata() -> None:
    assert Pressure.dims == CELL_K
    assert Pressure.standard_name == "air_pressure"
    assert Pressure.units == "Pa"
    assert Pressure.precision == "wp"
    assert Salt.standard_name is None
    assert Vt.precision == "vp"
    with pytest.raises(TypeError):
        Pressure()
