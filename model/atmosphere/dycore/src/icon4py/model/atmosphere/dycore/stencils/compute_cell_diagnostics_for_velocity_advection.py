# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.fbuiltins import astype, broadcast, where

from icon4py.model.atmosphere.dycore.stencils.correct_contravariant_vertical_velocity import (
    _correct_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_horizontal_kinetic_energy_and_khalf_contravariant_correction(
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    contravariant_correction_at_edge: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    khalf_contravariant_correction_at_cell: fa.CellKField[vpfloat],
    horizontal_kinetic_energy_at_cell: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nflatlev: gtx.int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    horizontal_kinetic_energy_at_cell = _interpolate_to_cell_center(
        horizontal_kinetic_energy_at_edge, e_bln_c_s
    )

    contravariant_correction_at_cell = where(
        k >= nflatlev,
        _interpolate_to_cell_center(contravariant_correction_at_edge, e_bln_c_s),
        broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )

    khalf_contravariant_correction_at_cell = where(
        nflatlev + 1 <= k,
        _interpolate_to_half_levels_vp(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cell
        ),
        khalf_contravariant_correction_at_cell,
    )

    return (
        horizontal_kinetic_energy_at_cell,
        khalf_contravariant_correction_at_cell,
    )


@field_operator
def _compute_khalf_contravariant_corrected_w(
    w: fa.CellKField[wpfloat],
    khalf_contravariant_correction_at_cell: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    khalf_contravariant_corrected_w_at_cell = where(
        k < nlev, astype(w, vpfloat), broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))
    )

    khalf_contravariant_corrected_w_at_cell = where(
        nflatlev + 1 <= k < nlev,
        _correct_contravariant_vertical_velocity(
            khalf_contravariant_corrected_w_at_cell, khalf_contravariant_correction_at_cell
        ),
        khalf_contravariant_corrected_w_at_cell,
    )

    return khalf_contravariant_corrected_w_at_cell


@gtx.program
def compute_horizontal_kinetic_energy_and_khalf_contravariant_terms(
    horizontal_kinetic_energy_at_cell: fa.CellKField[vpfloat],
    khalf_contravariant_correction_at_cell: fa.CellKField[vpfloat],
    khalf_contravariant_corrected_w_at_cell: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    contravariant_correction_at_edge: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_horizontal_kinetic_energy_and_khalf_contravariant_correction(
        horizontal_kinetic_energy_at_edge,
        e_bln_c_s,
        contravariant_correction_at_edge,
        wgtfac_c,
        khalf_contravariant_correction_at_cell,
        horizontal_kinetic_energy_at_cell,
        k,
        nflatlev,
        out=(
            horizontal_kinetic_energy_at_cell,
            khalf_contravariant_correction_at_cell,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _compute_khalf_contravariant_corrected_w(
        w,
        khalf_contravariant_correction_at_cell,
        k,
        nflatlev,
        nlev,
        out=khalf_contravariant_corrected_w_at_cell,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program
def compute_horizontal_kinetic_energy_and_khalf_contravariant_corrected_w(
    horizontal_kinetic_energy_at_cell: fa.CellKField[vpfloat],
    khalf_contravariant_corrected_w_at_cell: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    khalf_contravariant_correction_at_cell: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    k: fa.KField[gtx.int32],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_to_cell_center(
        horizontal_kinetic_energy_at_edge,
        e_bln_c_s,
        out=horizontal_kinetic_energy_at_cell,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _compute_khalf_contravariant_corrected_w(
        w,
        khalf_contravariant_correction_at_cell,
        k,
        nflatlev,
        nlev,
        out=khalf_contravariant_corrected_w_at_cell,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
