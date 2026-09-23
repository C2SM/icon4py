# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Solve the tridiagonal systems ``a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k`` along the vertical
with the Thomas algorithm: a forward sweep followed by a back substitution.

The forward sweep carries ``q = -c'`` (ICON's ``z_q``) and ``d'``. Its init state makes the first row
independent of its sub-diagonal entry, and the back substitution's init state makes the last row
independent of its super-diagonal entry.
"""

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), wpfloat("0.0")))
def _solve_tridiagonal_matrix_forward_sweep(
    state_kminus1: tuple[wpfloat, wpfloat],
    a: wpfloat,
    b: wpfloat,
    c: wpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    q_kminus1, d_prime_kminus1 = state_kminus1
    normalization = wpfloat("1.0") / (b + a * q_kminus1)
    return (wpfloat("0.0") - c) * normalization, (d - a * d_prime_kminus1) * normalization


@gtx.scan_operator(axis=dims.KDim, forward=False, init=wpfloat("0.0"))
def _solve_tridiagonal_matrix_back_substitution(
    x_kplus1: wpfloat,
    q: wpfloat,
    d_prime: wpfloat,
) -> wpfloat:
    return d_prime + x_kplus1 * q


@gtx.scan_operator(axis=dims.KHalfDim, forward=True, init=(wpfloat("0.0"), wpfloat("0.0")))
def _solve_tridiagonal_matrix_forward_sweep_on_half_levels_wp(
    state_kminus1: tuple[wpfloat, wpfloat],
    a: wpfloat,
    b: wpfloat,
    c: wpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    q_kminus1, d_prime_kminus1 = state_kminus1
    normalization = wpfloat("1.0") / (b + a * q_kminus1)
    return (wpfloat("0.0") - c) * normalization, (d - a * d_prime_kminus1) * normalization


@gtx.scan_operator(axis=dims.KHalfDim, forward=False, init=wpfloat("0.0"))
def _solve_tridiagonal_matrix_back_substitution_on_half_levels_wp(
    x_kplus1: wpfloat,
    q: wpfloat,
    d_prime: wpfloat,
) -> wpfloat:
    return d_prime + x_kplus1 * q


@gtx.scan_operator(
    axis=dims.KHalfDim,
    forward=True,
    init=(  # type: ignore[call-overload] # GT4Py misses type hint for tuples here
        vpfloat("0.0"),
        0.0,
    ),
)
def _solve_tridiagonal_matrix_forward_sweep_on_half_levels_mixed_precision(
    state_kminus1: tuple[vpfloat, float],
    a: vpfloat,
    b: vpfloat,
    c: vpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    """The matrix coefficients and ``q`` in vpfloat, ``d'`` in wpfloat."""
    q_kminus1 = astype(state_kminus1[0], vpfloat)
    d_prime_kminus1 = state_kminus1[1]
    normalization = vpfloat("1.0") / (b + a * q_kminus1)
    q = (vpfloat("0.0") - c) * normalization
    d_prime = (d - astype(a, wpfloat) * d_prime_kminus1) * astype(normalization, wpfloat)
    return q, d_prime  # type: ignore[return-value] # return type hints for scan operators broken in GT4Py


@gtx.scan_operator(axis=dims.KHalfDim, forward=False, init=wpfloat("0.0"))
def _solve_tridiagonal_matrix_back_substitution_on_half_levels_mixed_precision(
    x_kplus1: wpfloat,
    q: vpfloat,
    d_prime: wpfloat,
) -> wpfloat:
    return d_prime + x_kplus1 * astype(q, wpfloat)  # type: ignore[return-value] # return type hints for scan operator broken in GT4Py
