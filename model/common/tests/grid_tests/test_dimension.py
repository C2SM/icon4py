# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims

from . import utils


@pytest.mark.parametrize("dim", utils.all_dims())
def test_from_value(dim):
    assert dim == dims.from_value(dim.value)


@pytest.mark.parametrize("value", ["foo", "EDGE", "123", "EdgeDim"])
def test_from_value_with_invalid_name(value):
    assert dims.from_value(value) is None, f"Expected None for invalid value '{value}'"
