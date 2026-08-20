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
from icon4py.model.common.dimension import KDim
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_wp import (
    _interpolate_edge_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_vn_to_half_levels_with_boundary(
    vn: fa.EdgeKField[wpfloat],
    wgtfac_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e_1: fa.EdgeField[wpfloat],
    wgtfacq1_e_2: fa.EdgeField[wpfloat],
    wgtfacq1_e_3: fa.EdgeField[wpfloat],
    wgtfacq_e_1: fa.EdgeField[wpfloat],
    wgtfacq_e_2: fa.EdgeField[wpfloat],
    wgtfacq_e_3: fa.EdgeField[wpfloat],
    nlev: gtx.int32,
) -> fa.EdgeKField[wpfloat]:
    """
    Interpolate the normal velocity from full levels to half levels (edge interfaces).

    Port of ``interpolate_normal_velocity_edge_interface`` in ICON's ``mo_vdf_atmo.f90``:
    - interior half levels (0 < k < nlev) are linearly interpolated with ``wgtfac_e``
      (reuses the common field operator ``_interpolate_edge_field_to_half_levels_wp``),
    - the top half level (k == 0) is extrapolated quadratically from the first three
      full levels with ``wgtfacq1_e_1/2/3``,
    - the bottom half level (k == nlev) is extrapolated quadratically from the last
      three full levels with ``wgtfacq_e_1/2/3``.

    Each ``concat_where`` branch is evaluated only on its own K region, so the vertical
    (``Koff``) shifts in the branch expressions need to be in bounds only there.

    Args:
        vn: normal velocity on full levels (nlev levels)
        wgtfac_e: interpolation weight on half levels
        wgtfacq1_e_1: top extrapolation weight for full level 0
        wgtfacq1_e_2: top extrapolation weight for full level 1
        wgtfacq1_e_3: top extrapolation weight for full level 2
        wgtfacq_e_1: bottom extrapolation weight for full level nlev - 1
        wgtfacq_e_2: bottom extrapolation weight for full level nlev - 2
        wgtfacq_e_3: bottom extrapolation weight for full level nlev - 3
        nlev: number of full levels

    Returns:
        normal velocity on half levels (nlev + 1 levels)
    """
    vn_ie_interior = _interpolate_edge_field_to_half_levels_wp(wgtfac_e=wgtfac_e, interpolant=vn)
    vn_ie_top = wgtfacq1_e_1 * vn + wgtfacq1_e_2 * vn(KDim + 1) + wgtfacq1_e_3 * vn(KDim + 2)
    vn_ie_bottom = (
        wgtfacq_e_1 * vn(KDim - 1) + wgtfacq_e_2 * vn(KDim - 2) + wgtfacq_e_3 * vn(KDim - 3)
    )
    vn_ie = concat_where(dims.KDim == 0, vn_ie_top, vn_ie_interior)
    return concat_where(dims.KDim == nlev, vn_ie_bottom, vn_ie)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_vn_to_half_levels_with_boundary(
    vn: fa.EdgeKField[wpfloat],
    wgtfac_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e_1: fa.EdgeField[wpfloat],
    wgtfacq1_e_2: fa.EdgeField[wpfloat],
    wgtfacq1_e_3: fa.EdgeField[wpfloat],
    wgtfacq_e_1: fa.EdgeField[wpfloat],
    wgtfacq_e_2: fa.EdgeField[wpfloat],
    wgtfacq_e_3: fa.EdgeField[wpfloat],
    vn_ie: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_vn_to_half_levels_with_boundary(
        vn=vn,
        wgtfac_e=wgtfac_e,
        wgtfacq1_e_1=wgtfacq1_e_1,
        wgtfacq1_e_2=wgtfacq1_e_2,
        wgtfacq1_e_3=wgtfacq1_e_3,
        wgtfacq_e_1=wgtfacq_e_1,
        wgtfacq_e_2=wgtfacq_e_2,
        wgtfacq_e_3=wgtfacq_e_3,
        nlev=nlev,
        out=vn_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
