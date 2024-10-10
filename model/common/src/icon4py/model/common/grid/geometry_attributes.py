import icon4py.model.common.dimension as dims


INVERSE_VERTEX_VERTEX_LENGTH = "inverse_vertex_vertex_length"

VERTEX_VERTEX_LENGTH = "vertex_vertex_length"

EDGE_LENGTH = "edge_length"

EDGE_LON = "grid_longitude_of_edge_midpoint"

EDGE_LAT = "grid_latitude_of_edge_midpoint"

VERTEX_LON = "grid_longitude_of_vertex"

VERTEX_LAT = "grid_latitude_of_vertex"

CELL_LON = "grid_longitude_of_cell_center"

CELL_LAT = "grid_latitude_of_cell_center"

attrs = {
    CELL_LAT: dict(standard_name =CELL_LAT,
                 unit="radians", dims=(dims.CellDim,), icon_var_name=""),
    CELL_LON: dict(standard_name=CELL_LON,
                 unit="radians", dims=(dims.CellDim,), icon_var_name=""),

    VERTEX_LAT: dict(standard_name=VERTEX_LAT,
            unit="radians", dims=(dims.VertexDim,), icon_var_name=""),
    VERTEX_LON: dict(standard_name=VERTEX_LON,
            unit="radians", dims=(dims.VertexDim,), icon_var_name=""),

    EDGE_LAT: dict(standard_name=EDGE_LAT,
                   unit="radians", dims=(dims.EdgeDim,), icon_var_name=""),
    EDGE_LON: dict(standard_name=EDGE_LON,
            unit="radians", dims=(dims.EdgeDim,), icon_var_name=""),
    EDGE_LENGTH: dict(standard_name=EDGE_LENGTH, long_name="edge length",
                      unit="m", dims=(dims.EdgeDim,),
                      icon_var_name="primal_edge_length", ),
    VERTEX_VERTEX_LENGTH: dict(standard_name=VERTEX_VERTEX_LENGTH,
                               long_name ="distance between outer vertices of adjacent cells",
                               unit="m", dims=(dims.EdgeDim,),
                               icon_var_name="vert_vert_length"),
    INVERSE_VERTEX_VERTEX_LENGTH: dict(standard_name=INVERSE_VERTEX_VERTEX_LENGTH,
            long_name ="distance between outer vertices of adjacent cells",
            unit="m-1",
            dims=(dims.EdgeDim,),
            icon_var_name="inv_vert_vert_length", )

}