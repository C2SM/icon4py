# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# mypy: ignore-errors
import cProfile
import pstats

import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.simple import SimpleGrid


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
    inp: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
) -> gtx.Field[[dims.CEDim, dims.KDim], gtx.float64]:
    return inp**2


@program(grid_type=GridType.UNSTRUCTURED)
def square(
    inp: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
    result: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
):
    _square(inp, out=result)


def square_from_function(
    inp: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
    result: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
):
    square(inp, result, offset_provider={})


def square_error(
    inp: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
    result: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
):
    raise Exception("Exception foo occurred")
