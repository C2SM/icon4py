# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import GridType, field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    vert_idx: fa.KField[int32],
    nlev: int32,
    nflatlev: int32,
) -> fa.CellKField[vpfloat]:
    w_concorr_c = where(
        nflatlev + 1 <= vert_idx < nlev,
        _compute_contravariant_correction_of_w(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        _compute_contravariant_correction_of_w_for_lower_boundary(
            e_bln_c_s, z_w_concorr_me, wgtfacq_c
        ),
    )
    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    vert_idx: fa.KField[int32],
    nlev: int32,
    nflatlev: int32,
    w_concorr_c: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_solve_nonhydro_stencil_39_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        wgtfacq_c,
        vert_idx,
        nlev,
        nflatlev,
        out=w_concorr_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
