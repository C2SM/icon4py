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
def _prepare_tridiagonal_matrix_cells_half_interior(
    inv_mair: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    zprefac: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Interior rows of the half-level tridiagonal diffusion matrix (w solve).

    Port of the interior loop of 'prepare_diffusion_matrix_wp' (mo_vdf_atmo.f90)
    with lhalflvl=.TRUE. (lvlcorr_a=-1, lvlcorr_c=0):
        a(jk) = - zprefac * zk(jk-1) * inv_dz(jk-1) * inv_mair(jk)
        c(jk) = - zprefac * zk(jk)   * inv_dz(jk)   * inv_mair(jk)
        b(jk) = - a(jk) - c(jk)

    Here the unknowns live on half levels (rows jk = minlvl..maxlvl, e.g. 2..nlev
    in the Fortran w solve), while zk and inv_dz are full-level fields.
    Note: the extra boundary terms added to b in the w solve (w=0 at top/bottom,
    mo_vdf.f90 'Compute_diffusion_vert_wind') are NOT part of this stencil.
    """
    a = wpfloat("0.0") - zprefac * zk(KDim - 1) * inv_dz(KDim - 1) * inv_mair
    c = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    b = wpfloat("0.0") - a - c
    return a, b, c


@gtx.field_operator
def _prepare_tridiagonal_matrix_cells_half_top(
    inv_mair: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    zprefac: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Upper boundary row (minlvl): a = 0, b = -c."""
    a = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    c = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    b = wpfloat("0.0") - c
    return a, b, c


@gtx.field_operator
def _prepare_tridiagonal_matrix_cells_half_bottom(
    inv_mair: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    zprefac: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Lower boundary row (maxlvl): c = 0, b = -a."""
    a = wpfloat("0.0") - zprefac * zk(KDim - 1) * inv_dz(KDim - 1) * inv_mair
    c = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    b = wpfloat("0.0") - a
    return a, b, c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_tridiagonal_matrix_cells_half(
    inv_mair: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    zprefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _prepare_tridiagonal_matrix_cells_half_interior(
        inv_mair=inv_mair,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        out=(a, b, c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end - 1),
        },
    )
    _prepare_tridiagonal_matrix_cells_half_top(
        inv_mair=inv_mair,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        out=(a, b, c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )
    _prepare_tridiagonal_matrix_cells_half_bottom(
        inv_mair=inv_mair,
        inv_dz=inv_dz,
        zk=zk,
        zprefac=zprefac,
        out=(a, b, c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
