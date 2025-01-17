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


C_LIN_E: Final[str] = "interpolation_coefficient_from_cell_to_edge"
C_BLN_AVG: Final[str] = "bilinear_cell_average_weight"
GEOFAC_DIV: Final[str] = "geometrical_factor_for_divergence"
GEOFAC_ROT: Final[str] = "geometrical_factor_for_curl"
GEOFAC_N2S: Final[str] = "geometrical_factor_for_nabla_2_scalar"
GEOFAC_GRDIV: Final[str] = "geometrical_factor_for_gradient_of_divergence"
GEOFAC_GRG_X: Final[str] = "geometrical_factor_for_green_gauss_gradient_x"
GEOFAC_GRG_Y: Final[str] = "geometrical_factor_for_green_gauss_gradient_y"

attrs: dict[str, model.FieldMetaData] = {
    C_LIN_E: dict(
        standard_name=C_LIN_E,
        long_name="interpolation coefficient from cell to edges",
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="c_lin_e",
        dtype=ta.wpfloat,
    ),
    C_BLN_AVG: dict(
        standard_name=C_BLN_AVG,
        long_name="mass conserving bilinear cell average weight",
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
    GEOFAC_N2S: dict(
        standard_name=GEOFAC_N2S,
        long_name="geometrical factor nabla-2 scalar",
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.CellDim, dims.C2E2CODim),
        icon_var_name="geofac_n2s",
        dtype=ta.wpfloat,
    ),
    GEOFAC_GRDIV: dict(
        standard_name=GEOFAC_GRDIV,
        long_name="geometrical factor for gradient of divergence",
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.EdgeDim, dims.E2C2EODim),
        icon_var_name="geofac_grdiv",
        dtype=ta.wpfloat,
    ),
    GEOFAC_GRG_X: dict(
        standard_name=GEOFAC_GRG_X,
        long_name="geometrical factor for Green Gauss gradient (first component)",
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.CellDim, dims.C2E2CODim),
        icon_var_name="geofac_grg",
        dtype=ta.wpfloat,
    ),
    GEOFAC_GRG_Y: dict(
        standard_name=GEOFAC_GRG_Y,
        long_name="geometrical factor for Green Gauss gradient (second component)",
        units="",  # TODO (@halungge) check or confirm
        dims=(dims.CellDim, dims.C2E2CODim),
        icon_var_name="geofac_grg",
        dtype=ta.wpfloat,
    ),
}
