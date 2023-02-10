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

import cffi
import numpy as np
import pytest
from cffi import FFI
from functional.common import Field

from icon4py.common.dimension import KDim, VertexDim
from icon4py.py2f.cffi_utils import to_fields
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


mesh = SimpleMesh()


@to_fields(dim_sizes={VertexDim: mesh.n_vertices, KDim: mesh.k_level})
def scale_field(
    f1: Field[[VertexDim, KDim], float], factor: float
) -> Field[[VertexDim, KDim], float]:
    return factor * f1


@pytest.mark.skip("not implemented")
def test_unpack_args():

    ffi = cffi.FFI()
    f1 = ffi.from_buffer("float*", np.asarray(random_field(mesh, VertexDim, KDim)))
    factor = 0.5
    res = scale_field(f1, factor)
