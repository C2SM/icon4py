# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


EDGE_LON = "grid_longitude_of_edge_midpoint"

EDGE_LAT = "grid_latitude_of_edge_midpoint"

VERTEX_LON = "grid_longitude_of_vertex"

VERTEX_LAT = "grid_latitude_of_vertex"

CELL_LON = "grid_longitude_of_cell_center"
CELL_LAT = "grid_latitude_of_cell_center"
CELL_AREA = "cell_area"
EDGE_AREA = "edge_area"

CORIOLIS_PARAMETER = "coriolis_parameter"

INVERSE_VERTEX_VERTEX_LENGTH = "inverse_of_vertex_vertex_length"

VERTEX_VERTEX_LENGTH = "vertex_vertex_length"

EDGE_LENGTH = "edge_length"

EDGE_PRIMAL_NORMAL_U = "eastward_component_of_edge_normal"
EDGE_PRIMAL_NORMAL_V = "northward_component_of_edge_normal"
TANGENT_ORIENTATION = "edge_orientation"


DUAL_EDGE_LENGTH = "length_of_dual_edge"
EDGE_TANGENT_X = "x_component_of_edge_tangential_unit_vector"
EDGE_TANGENT_Y = "y_component_of_edge_tangential_unit_vector"
EDGE_TANGENT_Z = "z_component_of_edge_tangential_unit_vector"
EDGE_NORMAL_X = "x_component_of_edge_normal_unit_vector"
EDGE_NORMAL_Y = "y_component_of_edge_normal_unit_vector"
EDGE_NORMAL_Z = "z_component_of_edge_normal_unit_vector"
EDGE_NORMAL_VERTEX_U = "eastward_component_of_edge_normal_on_vertex"  # TODO
EDGE_NORMAL_VERTEX_V = "northward_component_of_edge_normal_on_vertex"  # TODO
EDGE_NORMAL_CELL_U = "eastward_component_of_edge_normal_on_cell"  # TODO
EDGE_NORMAL_CELL_V = "northward_component_of_edge_normal_on_cell"  # TODO


attrs: dict[str, model.FieldMetaData] = {
    CELL_LAT: dict(
        standard_name=CELL_LAT,
        units="radians",
        dims=(dims.CellDim,),
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    CELL_LON: dict(
        standard_name=CELL_LON,
        units="radians",
        dims=(dims.CellDim,),
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    VERTEX_LAT: dict(
        standard_name=VERTEX_LAT,
        units="radians",
        dims=(dims.VertexDim,),
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    VERTEX_LON: dict(
        standard_name=VERTEX_LON,
        units="radians",
        dims=(dims.VertexDim,),
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    EDGE_LAT: dict(
        standard_name=EDGE_LAT,
        units="radians",
        dims=(dims.EdgeDim,),
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    EDGE_LON: dict(
        standard_name=EDGE_LON,
        units="radians",
        dims=(dims.EdgeDim,),
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    EDGE_LENGTH: dict(
        standard_name=EDGE_LENGTH,
        long_name="edge length",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="primal_edge_length",
        dtype=ta.wpfloat,
    ),
    DUAL_EDGE_LENGTH: dict(
        standard_name=DUAL_EDGE_LENGTH,
        long_name="length of the dual edge",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="dual_edge_length",
        dtype=ta.wpfloat,
    ),
    VERTEX_VERTEX_LENGTH: dict(
        standard_name=VERTEX_VERTEX_LENGTH,
        long_name="distance between outer vertices of adjacent cells",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="vert_vert_length",
        dtype=ta.wpfloat,
    ),
    EDGE_AREA: dict(
        standard_name=EDGE_AREA,
        long_name="area of quadrilateral spanned by edge and associated dual edge",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name=EDGE_AREA,
        dtype=ta.wpfloat,
    ),
    CORIOLIS_PARAMETER: dict(
        standard_name=CORIOLIS_PARAMETER,
        long_name="coriolis parameter at cell edges",
        units="s-1",
        dims=(dims.EdgeDim,),
        icon_var_name="f_e",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_X: dict(
        standard_name=EDGE_TANGENT_X,
        long_name=EDGE_TANGENT_X,
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="",  # TODO
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_Y: dict(
        standard_name=EDGE_TANGENT_Y,
        long_name=EDGE_TANGENT_Y,
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="",  # TODO
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_Z: dict(
        standard_name=EDGE_NORMAL_Z,
        long_name=EDGE_TANGENT_Z,
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="",  # TODO
        dtype=ta.wpfloat,
    ),
    EDGE_PRIMAL_NORMAL_V: dict(
        standard_name=EDGE_PRIMAL_NORMAL_V,
        long_name=EDGE_PRIMAL_NORMAL_V + " (zonal component)",
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="primal_normal%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_PRIMAL_NORMAL_U: dict(
        standard_name=EDGE_PRIMAL_NORMAL_U,
        long_name=EDGE_PRIMAL_NORMAL_V + " (meridional component)",
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="primal_normal%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_X: dict(
        standard_name=EDGE_NORMAL_X,
        long_name=EDGE_NORMAL_X,
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="primal_cart_normal%x",  # TODO
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_Y: dict(
        standard_name=EDGE_NORMAL_Y,
        long_name=EDGE_NORMAL_Y,
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="primal_cart_normal%y",  # TODO
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_Z: dict(
        standard_name=EDGE_NORMAL_Z,
        long_name=EDGE_NORMAL_Z,
        units="",  # TODO
        dims=(dims.EdgeDim,),
        icon_var_name="primal_cart_normal%z",  # TODO
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_VERTEX_U: dict(
        standard_name=EDGE_NORMAL_VERTEX_U,
        long_name=EDGE_NORMAL_VERTEX_U,
        units="",  # TODO missing
        dims=(dims.EdgeDim, dims.E2C2VDim),
        icon_var_name="primal_normal_vert%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_VERTEX_V: dict(
        standard_name=EDGE_NORMAL_VERTEX_V,
        long_name=EDGE_NORMAL_VERTEX_V,
        units="",  # TODO missing
        dims=(dims.EdgeDim, dims.E2C2VDim),
        icon_var_name="primal_normal_vert%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_CELL_U: dict(
        standard_name=EDGE_NORMAL_CELL_U,
        long_name=EDGE_NORMAL_CELL_U,
        units="",  # TODO missing
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="primal_normal_vert%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_CELL_V: dict(
        standard_name=EDGE_NORMAL_CELL_V,
        long_name=EDGE_NORMAL_CELL_V,
        units="",  # TODO missing
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="primal_normal_vert%v2",
        dtype=ta.wpfloat,
    ),
    TANGENT_ORIENTATION: dict(
        standard_name=TANGENT_ORIENTATION,
        long_name="tangent of rhs aligned with edge tangent",
        units="1",
        dims=(dims.EdgeDim,),
        icon_var_name=TANGENT_ORIENTATION,
        dtype=ta.wpfloat,
    ),
}


def data_for_inverse(metadata: model.FieldMetaData) -> tuple[str, model.FieldMetaData]:
    standard_name = f"inverse_of_{metadata['standard_name']}"
    units = f"{metadata['units']}-1"
    long_name = f"inverse of {metadata.get('long_name')}" if metadata.get("long_name") else ""
    icon_var_name = f"inv_{metadata.get('icon_var_name')}"
    inverse_meta = dict(
        standard_name=standard_name,
        units=units,
        dims=metadata.get("dims"),
        dtype=metadata.get("dtype"),
        long_name=long_name,
        icon_var_name=icon_var_name,
    )

    return standard_name, {k: v for k, v in inverse_meta.items() if v is not None}
