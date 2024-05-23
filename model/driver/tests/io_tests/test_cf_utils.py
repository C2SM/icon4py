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

from icon4py.model.driver.io.cf_utils import to_canonical_dim_order

from .test_io import state_values


@pytest.mark.parametrize("input_", state_values())
def test_to_canonical_dim_order(input_):
    input_dims = input_.dims
    output = to_canonical_dim_order(input_)
    assert output.dims == (input_dims[1], input_dims[0])
