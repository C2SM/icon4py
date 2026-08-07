# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Finite difference gradient operators on unstructured grid fields.

Contains normal and tangential gradient computations on edges using
finite difference approximations.
"""

from gt4py import next as gtx

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2V


@gtx.field_operator
def grad_fd_norm(
    psi_c: fa.CellKHalfField[float],
    inv_dual_edge_length: fa.EdgeField[float],
) -> fa.EdgeKHalfField[float]:
    """
    Calculate the gradient value of adjacent interface levels.

    Computes the difference of two offseted values multiplied by a field of the offseted dimension
    Args:
        psi_c: fa.CellKHalfField[float],
        inv_dual_edge_length: Field[Dims[EdgeDim], float],

    Returns: fa.EdgeKHalfField[float]

    """
    grad_norm_psi_e = (psi_c(E2C[1]) - psi_c(E2C[0])) * inv_dual_edge_length
    return grad_norm_psi_e


@gtx.field_operator
def _grad_fd_tang(
    psi_v: fa.VertexKHalfField[float],
    inv_primal_edge_length: fa.EdgeField[float],
    tangent_orientation: fa.EdgeField[float],
) -> fa.EdgeKHalfField[float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e
