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

import numpy as np

from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.diffusion.diffusion_utils import copy_diagnostic_and_prognostics

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def test_copy_diagnostic_and_prognostics():
    mesh = SimpleMesh()
    f1_in = random_field(mesh, CellDim, KDim)
    f1_out = zero_field(mesh, CellDim, KDim)
    f2_in = random_field(mesh, CellDim, KDim)
    f2_out = zero_field(mesh, CellDim, KDim)
    f3_in = random_field(mesh, CellDim, KDim)
    f3_out = zero_field(mesh, CellDim, KDim)
    f4_in = random_field(mesh, CellDim, KDim)
    f4_out = zero_field(mesh, CellDim, KDim)
    f5_in = random_field(mesh, EdgeDim, KDim)
    f5_out = zero_field(mesh, EdgeDim, KDim)
    f6_in = random_field(mesh, CellDim, KDim)
    f6_out = zero_field(mesh, CellDim, KDim)
    f7_in = random_field(mesh, CellDim, KDim)
    f7_out = zero_field(mesh, CellDim, KDim)
    f8_in = random_field(mesh, CellDim, KDim)
    f8_out = zero_field(mesh, CellDim, KDim)

    copy_diagnostic_and_prognostics(
        f1_in,
        f1_out,
        f2_in,
        f2_out,
        f3_in,
        f3_out,
        f4_in,
        f4_out,
        f5_in,
        f5_out,
        f6_in,
        f6_out,
        f7_in,
        f7_out,
        f8_in,
        f8_out,
        offset_provider={},
    )

    assert np.allclose(f1_in, f1_out)
    assert np.allclose(f2_in, f2_out)
    assert np.allclose(f3_in, f3_out)
    assert np.allclose(f4_in, f4_out)
    assert np.allclose(f5_in, f5_out)
    assert np.allclose(f6_in, f6_out)
    assert np.allclose(f7_in, f7_out)
    assert np.allclose(f8_in, f8_out)
