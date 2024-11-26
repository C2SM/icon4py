# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


C_LIN_E: Final[str] = "c_lin_e"  # TODO (@halungge) find proper name
GEOFAC_DIV: Final[str] = "geometrical_factor_for_divergence"
GEOFAC_ROT: Final[str] = "geometrical_factor_for_curl"


attrs: dict[str, model.FieldMetaData] = {
    C_LIN_E: dict(
        standard_name=C_LIN_E,
        long_name=C_LIN_E,  # TODO (@halungge) find proper description
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="c_lin_e",
        dtype=ta.wpfloat,
    ),
    GEOFAC_DIV: dict(
        standard_name=GEOFAC_DIV,
        long_name="geometrical factor for divergence",  # TODO (@halungge) find proper description
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.CellDim, dims.C2EDim),
        icon_var_name="geofac_div",
        dtype=ta.wpfloat,
    ),
    GEOFAC_ROT: dict(
        standard_name=GEOFAC_ROT,
        long_name="geometrical factor for curl",
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.VertexDim, dims.V2EDim),
        icon_var_name="geofac_rot",
        dtype=ta.wpfloat,
    ),
}
