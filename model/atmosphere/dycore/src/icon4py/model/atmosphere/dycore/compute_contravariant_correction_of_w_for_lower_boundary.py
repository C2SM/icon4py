# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _compute_contravariant_correction_of_w_for_lower_boundary(
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_40."""
    z_w_concorr_me_wp = astype(z_w_concorr_me, wpfloat)

    z_w_concorr_me_offset_1 = z_w_concorr_me_wp(Koff[-1])
    z_w_concorr_me_offset_2 = z_w_concorr_me_wp(Koff[-2])
    z_w_concorr_me_offset_3 = z_w_concorr_me_wp(Koff[-3])

    z_w_concorr_mc_m1 = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_offset_1(C2E), axis=C2EDim)
    z_w_concorr_mc_m2 = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_offset_2(C2E), axis=C2EDim)
    z_w_concorr_mc_m3 = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_offset_3(C2E), axis=C2EDim)

    z_w_concorr_mc_m1_vp = astype(z_w_concorr_mc_m1, vpfloat)
    z_w_concorr_mc_m2_vp = astype(z_w_concorr_mc_m2, vpfloat)
    z_w_concorr_mc_m3_vp = astype(z_w_concorr_mc_m3, vpfloat)
    w_concorr_c_vp = (
        wgtfacq_c(Koff[-1]) * z_w_concorr_mc_m1_vp
        + wgtfacq_c(Koff[-2]) * z_w_concorr_mc_m2_vp
        + wgtfacq_c(Koff[-3]) * z_w_concorr_mc_m3_vp
    )

    return w_concorr_c_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_contravariant_correction_of_w_for_lower_boundary(
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_contravariant_correction_of_w_for_lower_boundary(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfacq_c,
        out=w_concorr_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
