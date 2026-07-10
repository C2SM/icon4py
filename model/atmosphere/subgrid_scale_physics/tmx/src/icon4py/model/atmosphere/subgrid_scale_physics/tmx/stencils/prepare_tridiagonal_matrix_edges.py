# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _prepare_tridiagonal_matrix_edges_interior(
    inv_mair: fa.EdgeKField[wpfloat],
    inv_dz: fa.EdgeKField[wpfloat],
    zk: fa.EdgeKField[wpfloat],
    zprefac: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Interior rows of the full-level tridiagonal diffusion matrix on edges.

    Port of the interior loop of 'prepare_diffusion_matrix_wp' (mo_vdf_atmo.f90)
    with lhalflvl=.FALSE. (lvlcorr_a=0, lvlcorr_c=1), applied on the edge grid
    (used for the vn diffusion in 'Compute_diffusion_hor_wind', mo_vdf.f90):
        a(jk) = - zprefac * zk(jk)   * inv_dz(jk)   * inv_mair(jk)
        c(jk) = - zprefac * zk(jk+1) * inv_dz(jk+1) * inv_mair(jk)
        b(jk) = - a(jk) - c(jk)
    """
    a = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    c = wpfloat("0.0") - zprefac * zk(KDim + 1) * inv_dz(KDim + 1) * inv_mair
    b = wpfloat("0.0") - a - c
    return a, b, c


@gtx.field_operator
def _prepare_tridiagonal_matrix_edges_top(
    inv_mair: fa.EdgeKField[wpfloat],
    inv_dz: fa.EdgeKField[wpfloat],
    zk: fa.EdgeKField[wpfloat],
    zprefac: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """Upper boundary row (minlvl): a = 0, b = -c."""
    a = broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim))
    c = wpfloat("0.0") - zprefac * zk(KDim + 1) * inv_dz(KDim + 1) * inv_mair
    b = wpfloat("0.0") - c
    return a, b, c


@gtx.field_operator
def _prepare_tridiagonal_matrix_edges_bottom(
    inv_mair: fa.EdgeKField[wpfloat],
    inv_dz: fa.EdgeKField[wpfloat],
    zk: fa.EdgeKField[wpfloat],
    zprefac: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """Lower boundary row (maxlvl): c = 0, b = -a."""
    a = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    c = broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim))
    b = wpfloat("0.0") - a
    return a, b, c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_tridiagonal_matrix_edges(
    inv_mair: fa.EdgeKField[wpfloat],
    inv_dz: fa.EdgeKField[wpfloat],
    zk: fa.EdgeKField[wpfloat],
    a: fa.EdgeKField[wpfloat],
    b: fa.EdgeKField[wpfloat],
    c: fa.EdgeKField[wpfloat],
    zprefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _prepare_tridiagonal_matrix_edges_interior(
        inv_mair=inv_mair,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        out=(a, b, c),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end - 1),
        },
    )
    _prepare_tridiagonal_matrix_edges_top(
        inv_mair=inv_mair,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        out=(a, b, c),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )
    _prepare_tridiagonal_matrix_edges_bottom(
        inv_mair=inv_mair,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        out=(a, b, c),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
