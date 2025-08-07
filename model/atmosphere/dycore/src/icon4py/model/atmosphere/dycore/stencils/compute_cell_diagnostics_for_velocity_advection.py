# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_correction(
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    nflatlev: gtx.int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    horizontal_kinetic_energy_at_cells_on_model_levels = _interpolate_to_cell_center(
        horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
    )

    contravariant_correction_at_cells_model_levels = _interpolate_to_cell_center(
        contravariant_correction_at_edges_on_model_levels, e_bln_c_s
    )

    contravariant_correction_at_cells_on_half_levels = concat_where(
        dims.KDim >= nflatlev + 1,
        _interpolate_cell_field_to_half_levels_vp(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        ),
        contravariant_correction_at_cells_on_half_levels,
    )

    return (
        horizontal_kinetic_energy_at_cells_on_model_levels,
        contravariant_correction_at_cells_on_half_levels,
    )


@gtx.field_operator
def _compute_contravariant_corrected_w(
    w: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    contravariant_corrected_w_at_cells_on_half_levels = concat_where(
        (nflatlev + 1 <= dims.KDim) & (dims.KDim < nlev),
        astype(w, vpfloat) - contravariant_correction_at_cells_on_half_levels,
        astype(w, vpfloat),
    )

    return concat_where(dims.KDim == nlev, 0.0, contravariant_corrected_w_at_cells_on_half_levels)


@gtx.program
def interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms(
    horizontal_kinetic_energy_at_cells_on_model_levels: fa.CellKField[vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Formerly known as fused_velocity_advection_stencil_8_to_13_predictor.

    This interpolates horizontal kinetic energy from edges to cells on models levels
    and compute the contravariant correction term on half levels.
    It also computes the vertical velocity with the contravariant correction term.

    Args:
        - horizontal_kinetic_energy_at_cells_on_model_levels: horizontal kinetic energy computed at cell centers of model levels
        - contravariant_correction_at_cells_on_half_levels: contravariant metric correction at cells on half levels
        - contravariant_corrected_w_at_cells_on_half_levels: contravariant-corrected vertical velocity at cells on half levels
        - w: vertical wind at cell centers
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy computed at edge of model levels
        - contravariant_correction_at_edges_on_model_levels: contravariant metric correction at edge of model levels
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights factors)
        - wgtfac_c: metrics field
        - nflatlev: number of flat levels
        - nlev: total number of vertical levels
        - horizontal_start: start index in the horizontal direction
        - horizontal_end: end index in the horizontal direction
        - vertical_start: start index in the vertical direction
        - vertical_end: end index in the vertical direction

    Returns:
        - horizontal_kinetic_energy_at_cells_on_model_levels
        - contravariant_correction_at_cells_on_half_levels
        - contravariant_corrected_w_at_cells_on_half_levels
    """

    _interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_correction(
        horizontal_kinetic_energy_at_edges_on_model_levels,
        e_bln_c_s,
        contravariant_correction_at_edges_on_model_levels,
        wgtfac_c,
        contravariant_correction_at_cells_on_half_levels,
        nflatlev,
        out=(
            horizontal_kinetic_energy_at_cells_on_model_levels,
            contravariant_correction_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _compute_contravariant_corrected_w(
        w,
        contravariant_correction_at_cells_on_half_levels,
        nflatlev,
        nlev,
        out=contravariant_corrected_w_at_cells_on_half_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program
def interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_corrected_w(
    horizontal_kinetic_energy_at_cells_on_model_levels: fa.CellKField[vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Formerly known as fused_velocity_advection_stencil_8_to_13_corrector.

    This interpolates horizontal kinetic energy from edges to cells on model levels
    The contravariant correction term has been computed at the end of predictor in solve nonhydro
    It also computes the contravariant-corrected vertical velocity at cell centers on half levels

    Args:
        - horizontal_kinetic_energy_at_cells_on_model_levels: horizontal kinetic energy computed at cell centers of model levels
        - contravariant_corrected_w_at_cells_on_half_levels: contravariant-corrected vertical velocity at cells on half levels
        - w: vertical wind at cell centers
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy computed at edge of model levels
        - contravariant_correction_at_cells_on_half_levels: contravariant metric correction at cells on half levels
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights factors)
        - nflatlev: number of flat levels
        - nlev: total number of vertical levels
        - horizontal_start: start index in the horizontal direction
        - horizontal_end: end index in the horizontal direction
        - vertical_start: start index in the vertical direction
        - vertical_end: end index in the vertical direction

    Returns:
        - horizontal_kinetic_energy_at_cells_on_model_levels
        - contravariant_corrected_w_at_cells_on_half_levels
    """

    _interpolate_to_cell_center(
        horizontal_kinetic_energy_at_edges_on_model_levels,
        e_bln_c_s,
        out=horizontal_kinetic_energy_at_cells_on_model_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _compute_contravariant_corrected_w(
        w,
        contravariant_correction_at_cells_on_half_levels,
        nflatlev,
        nlev,
        out=contravariant_corrected_w_at_cells_on_half_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
