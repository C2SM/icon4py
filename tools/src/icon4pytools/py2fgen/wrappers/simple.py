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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, float64
from icon4py.model.common import dimension as dims
from icon4py.model.common.caching import CachedProgram
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
    inp: Field[[dims.CEDim, dims.KDim], float64],
) -> Field[[dims.CEDim, dims.KDim], float64]:
    return inp**2


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def square(
    inp: Field[[dims.CEDim, dims.KDim], float64],
    result: Field[[dims.CEDim, dims.KDim], float64],
):
    _square(inp, out=result)


square_cached = CachedProgram(square, with_domain=False)


def square_from_function(
    inp: Field[[dims.CEDim, dims.KDim], float64],
    result: Field[[dims.CEDim, dims.KDim], float64],
):
    square_cached(inp, result, offset_provider={})


def square_error(
    inp: Field[[dims.CEDim, dims.KDim], float64],
    result: Field[[dims.CEDim, dims.KDim], float64],
):
    raise Exception("Exception foo occurred")
