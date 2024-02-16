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
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, float64
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _square(
    a: Field[[CellDim, KDim], float64], b: Field[[CellDim, KDim], float64]
) -> Field[[CellDim, KDim], float64]:
    return a * b


@program(grid_type=GridType.UNSTRUCTURED)
def square(
    a: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    _square(a, a, out=result)


def square_from_function(
    a: Field[[CellDim, KDim], float64],
    result: Field[[CellDim, KDim], float64],
):
    square(a, result, offset_provider={})


@field_operator
def _multi_return(
    z_vn_avg: Field[[EdgeDim, KDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    vn_traj: Field[[EdgeDim, KDim], wpfloat],
    mass_flx_me: Field[[EdgeDim, KDim], wpfloat],
    r_nsubsteps: wpfloat,
) -> tuple[Field[[EdgeDim, KDim], wpfloat], Field[[EdgeDim, KDim], wpfloat]]:
    """accumulate_prep_adv_fields stencil formerly known as _mo_solve_nonhydro_stencil_34."""
    vn_traj_wp = vn_traj + r_nsubsteps * z_vn_avg
    mass_flx_me_wp = mass_flx_me + r_nsubsteps * mass_fl_e
    return vn_traj_wp, mass_flx_me_wp


@program(grid_type=GridType.UNSTRUCTURED)
def multi_return(
    z_vn_avg: Field[[EdgeDim, KDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    vn_traj: Field[[EdgeDim, KDim], wpfloat],
    mass_flx_me: Field[[EdgeDim, KDim], wpfloat],
    r_nsubsteps: wpfloat,
):
    _multi_return(
        z_vn_avg,
        mass_fl_e,
        vn_traj,
        mass_flx_me,
        r_nsubsteps,
        out=(vn_traj, mass_flx_me),
    )
