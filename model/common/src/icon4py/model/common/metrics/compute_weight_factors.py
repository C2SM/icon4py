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


@gtx.field_operator
def _compute_wgtfac_c(
    z_ifc: fa.CellKHalfField[gtx.float64],
    nlev: gtx.int32,
) -> fa.CellKHalfField[gtx.float64]:
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
    wgtfac_c: fa.CellKHalfField[gtx.float64],
    z_ifc: fa.CellKHalfField[gtx.float64],
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


@gtx.field_operator
def _compute_quadratic_extrapolation_weights(
    za: fa.CellKField[gtx.float64],
    zb: fa.CellKField[gtx.float64],
    zc: fa.CellKField[gtx.float64],
    zd: fa.CellKField[gtx.float64],
) -> tuple[fa.CellKField[gtx.float64], fa.CellKField[gtx.float64], fa.CellKField[gtx.float64]]:
    """The three quadratic extrapolation coefficients of ``mo_vertical_grid.f90``.

    ``za`` to ``zd`` are the four interface heights the extrapolation is built from,
    already shifted onto the full level being written, hence full-level fields.
    """
    z1 = 0.5 * (zb - za)
    z2 = 0.5 * (zb + zc) - za
    z3 = 0.5 * (zc + zd) - za
    w3 = z1 * z2 / (z2 - z3) / (z1 - z3)
    w2 = (z1 - w3 * (z1 - z3)) / (z1 - z2)
    return 1.0 - (w2 + w3), w2, w3


@gtx.field_operator
def _compute_wgtfacq1_c(z_ifc: fa.CellKHalfField[gtx.float64]) -> fa.CellKField[gtx.float64]:
    """Top-boundary quadratic extrapolation weights at cell centres.

    Full levels 0..2 carry the weights of interfaces 0..3, one per level. The four
    interfaces are the same for all three, so each is addressed relative to the
    level being written: from full level k, interface j sits at
    ``KDim + (j - k) - 0.5``.
    """
    # TODO(havogt): one-sided masks, because with `==` GT4Py infers the shifted branches over
    # the whole column and dace reads out of bounds, see https://github.com/GridTools/gt4py/issues/2205.
    za = concat_where(
        dims.KDim < 1,
        z_ifc(dims.KDim - 0.5),
        concat_where(dims.KDim < 2, z_ifc(dims.KDim - 1.5), z_ifc(dims.KDim - 2.5)),
    )
    zb = concat_where(
        dims.KDim < 1,
        z_ifc(dims.KDim + 0.5),
        concat_where(dims.KDim < 2, z_ifc(dims.KDim - 0.5), z_ifc(dims.KDim - 1.5)),
    )
    zc = concat_where(
        dims.KDim < 1,
        z_ifc(dims.KDim + 1.5),
        concat_where(dims.KDim < 2, z_ifc(dims.KDim + 0.5), z_ifc(dims.KDim - 0.5)),
    )
    zd = concat_where(
        dims.KDim < 1,
        z_ifc(dims.KDim + 2.5),
        concat_where(dims.KDim < 2, z_ifc(dims.KDim + 1.5), z_ifc(dims.KDim + 0.5)),
    )
    w1, w2, w3 = _compute_quadratic_extrapolation_weights(za, zb, zc, zd)
    return concat_where(dims.KDim < 1, w1, concat_where(dims.KDim < 2, w2, w3))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_wgtfacq1_c(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKHalfField[gtx.float64],
    wgtfacq1_c: fa.CellKField[gtx.float64],
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
def _compute_wgtfacq_c(
    z_ifc: fa.CellKHalfField[gtx.float64], nlev: gtx.int32
) -> fa.CellKField[gtx.float64]:
    """Surface-boundary quadratic extrapolation weights at cell centres.

    Mirrors :func:`_compute_wgtfacq1_c` at the other end of the column: full levels
    nlev-3..nlev-1 carry the weights of interfaces nlev..nlev-3, and the level
    nearest the surface carries the first one.
    """
    # TODO(havogt): one-sided masks, because with `==` GT4Py infers the shifted branches over
    # the whole column and dace reads out of bounds, see https://github.com/GridTools/gt4py/issues/2205.
    za = concat_where(
        dims.KDim >= nlev - 1,
        z_ifc(dims.KDim + 0.5),
        concat_where(dims.KDim >= nlev - 2, z_ifc(dims.KDim + 1.5), z_ifc(dims.KDim + 2.5)),
    )
    zb = concat_where(
        dims.KDim >= nlev - 1,
        z_ifc(dims.KDim - 0.5),
        concat_where(dims.KDim >= nlev - 2, z_ifc(dims.KDim + 0.5), z_ifc(dims.KDim + 1.5)),
    )
    zc = concat_where(
        dims.KDim >= nlev - 1,
        z_ifc(dims.KDim - 1.5),
        concat_where(dims.KDim >= nlev - 2, z_ifc(dims.KDim - 0.5), z_ifc(dims.KDim + 0.5)),
    )
    zd = concat_where(
        dims.KDim >= nlev - 1,
        z_ifc(dims.KDim - 2.5),
        concat_where(dims.KDim >= nlev - 2, z_ifc(dims.KDim - 1.5), z_ifc(dims.KDim - 0.5)),
    )
    w1, w2, w3 = _compute_quadratic_extrapolation_weights(za, zb, zc, zd)
    return concat_where(dims.KDim >= nlev - 1, w1, concat_where(dims.KDim >= nlev - 2, w2, w3))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_wgtfacq_c(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKHalfField[gtx.float64],
    wgtfacq_c: fa.CellKField[gtx.float64],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_wgtfacq_c(
        z_ifc,
        nlev,
        out=wgtfacq_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
