# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vertical_operations import with_boundaries_on_half_levels_on_cells
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


@gtx.field_operator
def _compute_wgtfac_c(
    z_ifc: fa.CellKHalfField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    return with_boundaries_on_half_levels_on_cells(
        top=(z_ifc(dims.KHalfDim + 1) - z_ifc) / (z_ifc(dims.KHalfDim + 2) - z_ifc),
        interior=(z_ifc(dims.KHalfDim - 1) - z_ifc)
        / (z_ifc(dims.KHalfDim - 1) - z_ifc(dims.KHalfDim + 1)),
        bottom=(z_ifc(dims.KHalfDim - 1) - z_ifc) / (z_ifc(dims.KHalfDim - 2) - z_ifc),
        nlev=nlev,
    )


# TODO(halungge): missing test?
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_wgtfac_c(  # noqa: PLR0917 [too-many-positional-arguments]
    wgtfac_c: fa.CellKHalfField[wpfloat],
    z_ifc: fa.CellKHalfField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_wgtfac_c(
        z_ifc=z_ifc,
        nlev=nlev,
        out=wgtfac_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


def _compute_z1_z2_z3(
    z_ifc: data_alloc.NDArray, i1: int, i2: int, i3: int, i4: int
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    z1 = 0.5 * (z_ifc[:, i2] - z_ifc[:, i1])
    z2 = 0.5 * (z_ifc[:, i2] + z_ifc[:, i3]) - z_ifc[:, i1]
    z3 = 0.5 * (z_ifc[:, i3] + z_ifc[:, i4]) - z_ifc[:, i1]
    return z1, z2, z3


@gtx.field_operator
def _quadratic_extrapolation_weights(
    za: fa.CellKField[wpfloat],
    zb: fa.CellKField[wpfloat],
    zc: fa.CellKField[wpfloat],
    zd: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """The three quadratic extrapolation coefficients of ``mo_vertical_grid.f90``.

    ``za`` to ``zd`` are the four interface heights the extrapolation is built from,
    already shifted onto the full level being written, hence full-level fields.
    """
    z1 = wpfloat("0.5") * (zb - za)
    z2 = wpfloat("0.5") * (zb + zc) - za
    z3 = wpfloat("0.5") * (zc + zd) - za
    w3 = z1 * z2 / (z2 - z3) / (z1 - z3)
    w2 = (z1 - w3 * (z1 - z3)) / (z1 - z2)
    return wpfloat("1.0") - (w2 + w3), w2, w3


@gtx.field_operator
def _compute_wgtfacq1_c(z_ifc: fa.CellKHalfField[wpfloat]) -> fa.CellKField[wpfloat]:
    """Top-boundary quadratic extrapolation weights at cell centres.

    Full levels 0..2, each carrying one coefficient. All three are built from the
    same four interface heights (0..3), so the shift differs per level: from full
    level k, interface j sits at ``KDim + (j - k) - 0.5``.
    """
    w1, _, _ = _quadratic_extrapolation_weights(
        z_ifc(dims.KDim - 0.5),
        z_ifc(dims.KDim + 0.5),
        z_ifc(dims.KDim + 1.5),
        z_ifc(dims.KDim + 2.5),
    )
    _, w2, _ = _quadratic_extrapolation_weights(
        z_ifc(dims.KDim - 1.5),
        z_ifc(dims.KDim - 0.5),
        z_ifc(dims.KDim + 0.5),
        z_ifc(dims.KDim + 1.5),
    )
    _, _, w3 = _quadratic_extrapolation_weights(
        z_ifc(dims.KDim - 2.5),
        z_ifc(dims.KDim - 1.5),
        z_ifc(dims.KDim - 0.5),
        z_ifc(dims.KDim + 0.5),
    )
    return concat_where(dims.KDim == 0, w1, concat_where(dims.KDim == 1, w2, w3))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_wgtfacq1_c(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKHalfField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_wgtfacq1_c(
        z_ifc,
        out=wgtfacq1_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_wgtfacq_c_dsl(
    z_ifc: fa.CellKHalfField[wpfloat], nlev: gtx.int32
) -> fa.CellKField[wpfloat]:
    """Surface-boundary quadratic extrapolation weights at cell centres.

    Full levels nlev-3..nlev-1, mirroring :func:`_compute_wgtfacq1_c` at the other
    end of the column: the four interface heights are nlev..nlev-3, and the level
    nearest the surface carries the first coefficient.
    """
    w1, _, _ = _quadratic_extrapolation_weights(
        z_ifc(dims.KDim + 0.5),
        z_ifc(dims.KDim - 0.5),
        z_ifc(dims.KDim - 1.5),
        z_ifc(dims.KDim - 2.5),
    )
    _, w2, _ = _quadratic_extrapolation_weights(
        z_ifc(dims.KDim + 1.5),
        z_ifc(dims.KDim + 0.5),
        z_ifc(dims.KDim - 0.5),
        z_ifc(dims.KDim - 1.5),
    )
    _, _, w3 = _quadratic_extrapolation_weights(
        z_ifc(dims.KDim + 2.5),
        z_ifc(dims.KDim + 1.5),
        z_ifc(dims.KDim + 0.5),
        z_ifc(dims.KDim - 0.5),
    )
    return concat_where(dims.KDim == nlev - 1, w1, concat_where(dims.KDim == nlev - 2, w2, w3))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_wgtfacq_c_dsl(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKHalfField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_wgtfacq_c_dsl(
        z_ifc,
        nlev,
        out=wgtfacq_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
