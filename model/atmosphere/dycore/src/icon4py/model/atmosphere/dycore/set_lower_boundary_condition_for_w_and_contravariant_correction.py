# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _set_lower_boundary_condition_for_w_and_contravariant_correction(
    w_concorr_c: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_47."""
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    w_nnew_wp = w_concorr_c_wp
    z_contr_w_fl_l_wp = _init_cell_kdim_field_with_zero_wp()
    return w_nnew_wp, z_contr_w_fl_l_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def set_lower_boundary_condition_for_w_and_contravariant_correction(
    w_nnew: fa.CellKField[wpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_lower_boundary_condition_for_w_and_contravariant_correction(
        w_concorr_c,
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
