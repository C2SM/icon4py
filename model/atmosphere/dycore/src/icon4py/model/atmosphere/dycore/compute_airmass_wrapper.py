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
from compute_airmass_wrapper_plugin import ffi
from gt4py.next.ffront.fbuiltins import Field, int32
from icon4pytools.py2f.cffi_utils import to_fields

from icon4py.model.atmosphere.dycore.compute_airmass import compute_airmass
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.type_alias import wpfloat


grid = SimpleGrid()
offset_providers = grid.get_all_offset_providers()

nproma = 50000
field_sizes = {EdgeDim: nproma, CellDim: nproma, VertexDim: nproma, KDim: nproma}


def unpack(ptr, size_x, size_y) -> np.ndarray:
    """
    unpacks a 2d c/fortran field into a numpy array.

    :param ptr: c_pointer to the field
    :param size_x: col size (since its called from fortran)
    :param size_y: row size
    :return: a numpy array with shape=(size_y, size_x)
    and dtype = ctype of the pointer
    """
    # for now only 2d, invert for row/column precedence...
    shape = (size_y, size_x)
    length = np.prod(shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ar = np.frombuffer(
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=np.dtype(c_type),
        count=-1,
        offset=0,
    ).reshape(shape)
    return ar


def compute_airmass_wrapper(
    rho_in: Field[[CellDim, KDim], wpfloat],
    ddqz_z_full_in: Field[[CellDim, KDim], wpfloat],
    deepatmo_t1mc_in: Field[[KDim], wpfloat],
    airmass_out: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    compute_airmass(
        rho_in,
        ddqz_z_full_in,
        deepatmo_t1mc_in,
        airmass_out,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        offset_provider={},
    )
