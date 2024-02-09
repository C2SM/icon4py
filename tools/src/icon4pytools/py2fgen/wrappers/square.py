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
# mypy: ignore-errors
# TODO(samkellerhals): Delete file once we can generate wrapper functions for programs. Use this for tests potentially.

# flake8: noqa D104
from gt4py.next.ffront.fbuiltins import Field, float64
from icon4py.model.common.dimension import CellDim, KDim

from icon4pytools.py2fgen.wrappers.square_functions import square_output_param


def square(
    field_ptr: Field[[CellDim, KDim], float64],
    result_ptr: Field[[CellDim, KDim], float64],
):
    """
    simple python function that squares all entries of a field of
    size nx x ny and returns a pointer to the result.

    :param field_ptr:
    :param nx:
    :param ny:
    :param result_ptr:
    :return:
    """
    square_output_param(field_ptr, result_ptr)
