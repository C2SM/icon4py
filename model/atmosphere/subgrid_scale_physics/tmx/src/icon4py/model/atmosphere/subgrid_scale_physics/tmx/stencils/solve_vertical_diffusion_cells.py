# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), wpfloat("0.0")))
def _tridiagonal_forward_sweep(
    state_kminus1: tuple[wpfloat, wpfloat],
    a: wpfloat,
    b: wpfloat,
    c: wpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    """
    Forward elimination of the Thomas algorithm ('tdma_solver_vec',
    iconmath mo_math_utilities.F90):
        m       = 1 / (b(k) - c_prime(k-1) * a(k))
        c_prime = c(k) * m
        d_prime = (d(k) - d_prime(k-1) * a(k)) * m

    The init state (0, 0) reproduces the Fortran first-row formulas
    c_prime = c/b and d_prime = d/b for any value of a at the first row.
    """
    c_prime_kminus1 = state_kminus1[0]
    d_prime_kminus1 = state_kminus1[1]
    m = wpfloat("1.0") / (b - c_prime_kminus1 * a)
    c_prime = c * m
    d_prime = (d - d_prime_kminus1 * a) * m
    return c_prime, d_prime


@gtx.scan_operator(axis=dims.KDim, forward=False, init=wpfloat("0.0"))
def _tridiagonal_back_substitution(
    x_kplus1: wpfloat,
    c_prime: wpfloat,
    d_prime: wpfloat,
) -> wpfloat:
    """
    Back substitution of the Thomas algorithm:
        x(k) = d_prime(k) - c_prime(k) * x(k+1)

    The init state 0 reproduces the Fortran last-row formula x = d_prime.
    """
    return d_prime - c_prime * x_kplus1


@gtx.field_operator
def _solve_vertical_diffusion_cells(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Implicit vertical diffusion solve on cells.

    Port of 'diffuse_vertical_implicit' (mo_tmx_numerics.f90):
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
def solve_vertical_diffusion_cells(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    new_var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_vertical_diffusion_cells(
        a=a,
        b=b,
        c=c,
        rhs=rhs,
        var=var,
        tend=tend,
        dtime=dtime,
        out=(new_var, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
