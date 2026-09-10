# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-edge choice between two candidate fluxes by a mask on the edge's upwind cell.

The hybrid scheme (ihadv_tracer=132, mo_advection_hflux.f90 3574-3693) decides per
upwind cell whether its edges get the plain quadratic flux or the WENO flux; both are
computed on all edges here and this stencil picks per edge.
"""

import gt4py.next as gtx
from gt4py.next import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C


@gtx.field_operator
def _select_horizontal_tracer_flux_by_upwind_cell(
    use_first: fa.CellKField[bool],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_flux_first: fa.EdgeKField[ta.wpfloat],
    p_flux_second: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    use_first_upwind = where(p_cell_rel_idx_dsl == 1, use_first(E2C[1]), use_first(E2C[0]))
    return where(use_first_upwind, p_flux_first, p_flux_second)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def select_horizontal_tracer_flux_by_upwind_cell(
    use_first: fa.CellKField[bool],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_flux_first: fa.EdgeKField[ta.wpfloat],
    p_flux_second: fa.EdgeKField[ta.wpfloat],
    p_out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _select_horizontal_tracer_flux_by_upwind_cell(
        use_first=use_first,
        p_cell_rel_idx_dsl=p_cell_rel_idx_dsl,
        p_flux_first=p_flux_first,
        p_flux_second=p_flux_second,
        out=p_out_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
