# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.solve_vertical_diffusion_cells import (
    _tridiagonal_back_substitution,
    _tridiagonal_forward_sweep,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _solve_vertical_diffusion_edges(
    a: fa.EdgeKField[wpfloat],
    b: fa.EdgeKField[wpfloat],
    c: fa.EdgeKField[wpfloat],
    rhs: fa.EdgeKField[wpfloat],
    var: fa.EdgeKField[wpfloat],
    tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Implicit vertical diffusion solve on edges (vn diffusion).

    Port of 'diffuse_vertical_implicit' (mo_tmx_numerics.f90), same math as
    '_solve_vertical_diffusion_cells' but with EdgeDim as horizontal dimension:
        b_tot   = 1/dtime + b               (b is 'bb' in the Fortran)
        d       = var/dtime + rhs
        new_var = tridiagonal_solve(a, b_tot, c, d)
        tend    = tend + (new_var - var)/dtime
    """
    rdtime = wpfloat("1.0") / dtime
    b_tot = rdtime + b
    d = var * rdtime + rhs
    c_prime, d_prime = _tridiagonal_forward_sweep(a, b_tot, c, d)
    new_var = _tridiagonal_back_substitution(c_prime, d_prime)
    new_tend = tend + (new_var - var) * rdtime
    return new_var, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_vertical_diffusion_edges(
    a: fa.EdgeKField[wpfloat],
    b: fa.EdgeKField[wpfloat],
    c: fa.EdgeKField[wpfloat],
    rhs: fa.EdgeKField[wpfloat],
    var: fa.EdgeKField[wpfloat],
    new_var: fa.EdgeKField[wpfloat],
    tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_vertical_diffusion_edges(
        a=a,
        b=b,
        c=c,
        rhs=rhs,
        var=var,
        tend=tend,
        dtime=dtime,
        out=(new_var, tend),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
