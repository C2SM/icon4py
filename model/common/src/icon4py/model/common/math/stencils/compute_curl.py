# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import V2E
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_curl(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
) -> fa.VertexKField[wpfloat]:
    return neighbor_sum(vec_e(V2E) * geofac_rot, axis=dims.V2EDim)
