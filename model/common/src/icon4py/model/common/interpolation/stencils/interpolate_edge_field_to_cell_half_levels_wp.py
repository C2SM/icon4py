# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center_wp import (
    _interpolate_to_cell_center_wp,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_edge_field_to_cell_half_levels_wp(
    interpolant: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Interpolate an edge field on full levels to the cell centers on half levels.

    The field is first averaged from the three C2E neighbor edges to the cell
    center with the bilinear weights ``e_bln_c_s`` (separately on full levels k
    and k - 1), then interpolated vertically to the half level k with the
    weights ``wgtfac_c``:

        out(c, k) = wgtfac_c(c, k) * sum_e e_bln_c_s(c, e) * interpolant(e, k)
                    + (1 - wgtfac_c(c, k))
                      * sum_e e_bln_c_s(c, e) * interpolant(e, k - 1)

    Only the interior half levels are defined; the top (k = 0) and bottom
    (k = nlev) rows must be excluded by the caller's domain.

    Working-precision variant.
    """
    interpolant_c = _interpolate_to_cell_center_wp(interpolant=interpolant, e_bln_c_s=e_bln_c_s)
    return _interpolate_cell_field_to_half_levels_wp(wgtfac_c=wgtfac_c, interpolant=interpolant_c)
