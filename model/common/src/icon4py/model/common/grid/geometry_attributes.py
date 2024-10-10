import icon4py.model.common.dimension as dims


attrs = {
    "grid_latitude_of_cell_center": dict(standard_name = "grid_latitude_of_cell_center",
                                         unit="radians", dims=(dims.CellDim,), icon_var_name=""),
    "grid_longitude_of_cell_center": dict(standard_name="grid_latitude_of_cell_center",
                                          unit="radians", dims=(dims.CellDim,), icon_var_name=""),

    "grid_latitude_of_vertex": dict(standard_name="grid_latitude_of_vertex",
                                          unit="radians", dims=(dims.VertexDim,), icon_var_name=""),
    "grid_longitude_of_vertex": dict(standard_name="grid_longitude_of_vertex",
                                           unit="radians", dims=(dims.VertexDim,), icon_var_name=""),

    "grid_latitude_of_edge_midpoint": dict(standard_name="grid_longitude_of_edge_midpoint",
                                    unit="radians", dims=(dims.EdgeDim,), icon_var_name=""),
    "grid_longitude_of_edge_midpoint": dict(standard_name="grid_longitude_of_edge_midpoint",
                                     unit="radians", dims=(dims.EdgeDim,), icon_var_name=""),
}