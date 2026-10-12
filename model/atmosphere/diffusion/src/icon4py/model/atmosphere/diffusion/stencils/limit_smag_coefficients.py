# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, maximum, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _limit_smag_coefficients(
    kh_smag: fa.EdgeKField[vpfloat],
    smag_limit: fa.KField[vpfloat],
    smag_offset: vpfloat,
) -> fa.EdgeKField[vpfloat]:
    """Limit the raw Smagorinsky diffusion coefficient to the interval [0, smag_limit].

    The subtraction is performed in ``wpfloat`` to match ICON's evaluation order
    (`mo_nh_diffusion.f90`).
    """
    kh_smag_wp, smag_offset_wp = astype((kh_smag, smag_offset), wpfloat)
    kh_smag_limited = maximum(vpfloat("0.0"), astype(kh_smag_wp - smag_offset_wp, vpfloat))
    kh_smag_limited = minimum(kh_smag_limited, smag_limit)
    return kh_smag_limited


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def limit_smag_coefficients(
    kh_smag: fa.EdgeKField[vpfloat],
    smag_limit: fa.KField[vpfloat],
    smag_offset: vpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _limit_smag_coefficients(
        kh_smag,
        smag_limit,
        smag_offset,
        out=kh_smag,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
