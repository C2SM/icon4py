# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Exported functions called from `test_boundary_casts.f90`, imported by module name."""

import gt4py.next as gtx
import numpy as np

from icon4py.bindings import icon4py_export
from icon4py.bindings.icon4py_export import WpIn, WpInOut


SomeDim = gtx.Dimension("SomeDim")


@icon4py_export.export
def scale_inout(field: WpInOut[gtx.Field[gtx.Dims[SomeDim], gtx.float32]], factor: gtx.float64):
    field.ndarray[...] *= np.float32(factor)


@icon4py_export.export
def scale_in(field: WpIn[gtx.Field[gtx.Dims[SomeDim], gtx.float32]], factor: gtx.float64):
    field.ndarray[...] *= np.float32(factor)  # must not reach Fortran
