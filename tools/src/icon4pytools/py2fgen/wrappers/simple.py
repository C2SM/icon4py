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
import cProfile
import pstats

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, float64, int32, neighbor_sum
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.caching import CachedProgram
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.settings import backend


# global profiler object
profiler = cProfile.Profile()

grid = SimpleGrid()


def profile_enable():
    profiler.enable()


def profile_disable():
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(f"{__name__}.profile")


@field_operator
def _square(
    inp: Field[[CellDim, KDim], float64],
) -> Field[[CellDim, KDim], float64]:
    return inp**2


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def square(
    inp: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    _square(inp, out=result)


square_cached = CachedProgram(square, with_domain=False)


def square_from_function(
    inp: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    square_cached(inp, result, offset_provider={})


def square_error(
    inp: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    raise Exception("Exception foo occurred")
