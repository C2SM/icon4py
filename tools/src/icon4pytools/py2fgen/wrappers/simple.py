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
from icon4py.model.common.caching import CachedProgram
from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat
from model.common.tests import field_type_aliases as fa


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


def square_from_function(
    inp: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    square(inp, result, offset_provider={})


@field_operator
def _multi_return(
    z_vn_avg: fa.EKwpField,
    mass_fl_e: fa.EKwpField,
    vn_traj: fa.EKwpField,
    mass_flx_me: fa.EKwpField,
    geofac_div: Field[[CEDim], wpfloat],
    z_nabla2_e: fa.EKwpField,
    r_nsubsteps: wpfloat,
) -> tuple[fa.EKwpField, fa.EKwpField]:
    """accumulate_prep_adv_fields stencil formerly known as _mo_solve_nonhydro_stencil_34."""
    vn_traj_wp = vn_traj + r_nsubsteps * z_vn_avg
    mass_flx_me_wp = mass_flx_me + r_nsubsteps * mass_fl_e
    z_temp_wp = neighbor_sum(z_nabla2_e(C2E) * geofac_div(C2CE), axis=C2EDim)  # noqa: F841
    return vn_traj_wp, mass_flx_me_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def multi_return(
    z_vn_avg: fa.EKwpField,
    mass_fl_e: fa.EKwpField,
    vn_traj: fa.EKwpField,
    mass_flx_me: fa.EKwpField,
    geofac_div: Field[[CEDim], wpfloat],
    z_nabla2_e: fa.EKwpField,
    r_nsubsteps: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _multi_return(
        z_vn_avg,
        mass_fl_e,
        vn_traj,
        mass_flx_me,
        geofac_div,
        z_nabla2_e,
        r_nsubsteps,
        out=(vn_traj, mass_flx_me),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


def square_error(
    inp: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    raise Exception("Exception foo occurred")


multi_return_cached = CachedProgram(multi_return)


def multi_return_from_function(
    z_vn_avg: fa.EKwpField,
    mass_fl_e: fa.EKwpField,
    vn_traj: fa.EKwpField,
    mass_flx_me: fa.EKwpField,
    geofac_div: Field[[CEDim], wpfloat],
    z_nabla2_e: fa.EKwpField,
    r_nsubsteps: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    multi_return_cached(
        z_vn_avg,
        mass_fl_e,
        vn_traj,
        mass_flx_me,
        geofac_div,
        z_nabla2_e,
        r_nsubsteps,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        offset_provider=grid.offset_providers,
    )
