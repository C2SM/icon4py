import logging
from grid import ffi
from icon4py.tools.py2fgen import runtime_config, _runtime, _definitions, _conversion

if __debug__:
    logger = logging.getLogger(__name__)
    log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, runtime_config.LOG_LEVEL),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# embedded function imports
from icon4py.tools.py2fgen.wrappers.grid_wrapper import grid_init


@ffi.def_extern()
def grid_init_wrapper(
    cell_starts,
    cell_starts_size_0,
    cell_ends,
    cell_ends_size_0,
    vertex_starts,
    vertex_starts_size_0,
    vertex_ends,
    vertex_ends_size_0,
    edge_starts,
    edge_starts_size_0,
    edge_ends,
    edge_ends_size_0,
    c2e,
    c2e_size_0,
    c2e_size_1,
    e2c,
    e2c_size_0,
    e2c_size_1,
    c2e2c,
    c2e2c_size_0,
    c2e2c_size_1,
    e2c2e,
    e2c2e_size_0,
    e2c2e_size_1,
    e2v,
    e2v_size_0,
    e2v_size_1,
    v2e,
    v2e_size_0,
    v2e_size_1,
    v2c,
    v2c_size_0,
    v2c_size_1,
    e2c2v,
    e2c2v_size_0,
    e2c2v_size_1,
    c2v,
    c2v_size_0,
    c2v_size_1,
    c_owner_mask,
    c_owner_mask_size_0,
    e_owner_mask,
    e_owner_mask_size_0,
    v_owner_mask,
    v_owner_mask_size_0,
    c_glb_index,
    c_glb_index_size_0,
    e_glb_index,
    e_glb_index_size_0,
    v_glb_index,
    v_glb_index_size_0,
    tangent_orientation,
    tangent_orientation_size_0,
    inverse_primal_edge_lengths,
    inverse_primal_edge_lengths_size_0,
    inv_dual_edge_length,
    inv_dual_edge_length_size_0,
    inv_vert_vert_length,
    inv_vert_vert_length_size_0,
    edge_areas,
    edge_areas_size_0,
    f_e,
    f_e_size_0,
    cell_center_lat,
    cell_center_lat_size_0,
    cell_center_lon,
    cell_center_lon_size_0,
    cell_areas,
    cell_areas_size_0,
    primal_normal_vert_x,
    primal_normal_vert_x_size_0,
    primal_normal_vert_x_size_1,
    primal_normal_vert_y,
    primal_normal_vert_y_size_0,
    primal_normal_vert_y_size_1,
    dual_normal_vert_x,
    dual_normal_vert_x_size_0,
    dual_normal_vert_x_size_1,
    dual_normal_vert_y,
    dual_normal_vert_y_size_0,
    dual_normal_vert_y_size_1,
    primal_normal_cell_x,
    primal_normal_cell_x_size_0,
    primal_normal_cell_x_size_1,
    primal_normal_cell_y,
    primal_normal_cell_y_size_0,
    primal_normal_cell_y_size_1,
    dual_normal_cell_x,
    dual_normal_cell_x_size_0,
    dual_normal_cell_x_size_1,
    dual_normal_cell_y,
    dual_normal_cell_y_size_0,
    dual_normal_cell_y_size_1,
    edge_center_lat,
    edge_center_lat_size_0,
    edge_center_lon,
    edge_center_lon_size_0,
    primal_normal_x,
    primal_normal_x_size_0,
    primal_normal_y,
    primal_normal_y_size_0,
    mean_cell_area,
    comm_id,
    num_vertices,
    num_cells,
    num_edges,
    vertical_size,
    limited_area,
    backend,
    on_gpu,
):
    try:
        if __debug__:
            logger.info("Python execution of grid_init started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayInfos

        cell_starts = (cell_starts, (cell_starts_size_0,), False, False)

        cell_ends = (cell_ends, (cell_ends_size_0,), False, False)

        vertex_starts = (vertex_starts, (vertex_starts_size_0,), False, False)

        vertex_ends = (vertex_ends, (vertex_ends_size_0,), False, False)

        edge_starts = (edge_starts, (edge_starts_size_0,), False, False)

        edge_ends = (edge_ends, (edge_ends_size_0,), False, False)

        c2e = (
            c2e,
            (
                c2e_size_0,
                c2e_size_1,
            ),
            on_gpu,
            False,
        )

        e2c = (
            e2c,
            (
                e2c_size_0,
                e2c_size_1,
            ),
            on_gpu,
            False,
        )

        c2e2c = (
            c2e2c,
            (
                c2e2c_size_0,
                c2e2c_size_1,
            ),
            on_gpu,
            False,
        )

        e2c2e = (
            e2c2e,
            (
                e2c2e_size_0,
                e2c2e_size_1,
            ),
            on_gpu,
            False,
        )

        e2v = (
            e2v,
            (
                e2v_size_0,
                e2v_size_1,
            ),
            on_gpu,
            False,
        )

        v2e = (
            v2e,
            (
                v2e_size_0,
                v2e_size_1,
            ),
            on_gpu,
            False,
        )

        v2c = (
            v2c,
            (
                v2c_size_0,
                v2c_size_1,
            ),
            on_gpu,
            False,
        )

        e2c2v = (
            e2c2v,
            (
                e2c2v_size_0,
                e2c2v_size_1,
            ),
            on_gpu,
            False,
        )

        c2v = (
            c2v,
            (
                c2v_size_0,
                c2v_size_1,
            ),
            on_gpu,
            False,
        )

        c_owner_mask = (c_owner_mask, (c_owner_mask_size_0,), False, False)

        e_owner_mask = (e_owner_mask, (e_owner_mask_size_0,), False, False)

        v_owner_mask = (v_owner_mask, (v_owner_mask_size_0,), False, False)

        c_glb_index = (c_glb_index, (c_glb_index_size_0,), False, False)

        e_glb_index = (e_glb_index, (e_glb_index_size_0,), False, False)

        v_glb_index = (v_glb_index, (v_glb_index_size_0,), False, False)

        tangent_orientation = (tangent_orientation, (tangent_orientation_size_0,), on_gpu, False)

        inverse_primal_edge_lengths = (
            inverse_primal_edge_lengths,
            (inverse_primal_edge_lengths_size_0,),
            on_gpu,
            False,
        )

        inv_dual_edge_length = (inv_dual_edge_length, (inv_dual_edge_length_size_0,), on_gpu, False)

        inv_vert_vert_length = (inv_vert_vert_length, (inv_vert_vert_length_size_0,), on_gpu, False)

        edge_areas = (edge_areas, (edge_areas_size_0,), on_gpu, False)

        f_e = (f_e, (f_e_size_0,), on_gpu, False)

        cell_center_lat = (cell_center_lat, (cell_center_lat_size_0,), on_gpu, False)

        cell_center_lon = (cell_center_lon, (cell_center_lon_size_0,), on_gpu, False)

        cell_areas = (cell_areas, (cell_areas_size_0,), on_gpu, False)

        primal_normal_vert_x = (
            primal_normal_vert_x,
            (
                primal_normal_vert_x_size_0,
                primal_normal_vert_x_size_1,
            ),
            on_gpu,
            False,
        )

        primal_normal_vert_y = (
            primal_normal_vert_y,
            (
                primal_normal_vert_y_size_0,
                primal_normal_vert_y_size_1,
            ),
            on_gpu,
            False,
        )

        dual_normal_vert_x = (
            dual_normal_vert_x,
            (
                dual_normal_vert_x_size_0,
                dual_normal_vert_x_size_1,
            ),
            on_gpu,
            False,
        )

        dual_normal_vert_y = (
            dual_normal_vert_y,
            (
                dual_normal_vert_y_size_0,
                dual_normal_vert_y_size_1,
            ),
            on_gpu,
            False,
        )

        primal_normal_cell_x = (
            primal_normal_cell_x,
            (
                primal_normal_cell_x_size_0,
                primal_normal_cell_x_size_1,
            ),
            on_gpu,
            False,
        )

        primal_normal_cell_y = (
            primal_normal_cell_y,
            (
                primal_normal_cell_y_size_0,
                primal_normal_cell_y_size_1,
            ),
            on_gpu,
            False,
        )

        dual_normal_cell_x = (
            dual_normal_cell_x,
            (
                dual_normal_cell_x_size_0,
                dual_normal_cell_x_size_1,
            ),
            on_gpu,
            False,
        )

        dual_normal_cell_y = (
            dual_normal_cell_y,
            (
                dual_normal_cell_y_size_0,
                dual_normal_cell_y_size_1,
            ),
            on_gpu,
            False,
        )

        edge_center_lat = (edge_center_lat, (edge_center_lat_size_0,), on_gpu, False)

        edge_center_lon = (edge_center_lon, (edge_center_lon_size_0,), on_gpu, False)

        primal_normal_x = (primal_normal_x, (primal_normal_x_size_0,), on_gpu, False)

        primal_normal_y = (primal_normal_y, (primal_normal_y_size_0,), on_gpu, False)

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "grid_init constructing `ArrayInfos` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            perf_counters = {}
        else:
            perf_counters = None
        grid_init(
            ffi=ffi,
            perf_counters=perf_counters,
            cell_starts=cell_starts,
            cell_ends=cell_ends,
            vertex_starts=vertex_starts,
            vertex_ends=vertex_ends,
            edge_starts=edge_starts,
            edge_ends=edge_ends,
            c2e=c2e,
            e2c=e2c,
            c2e2c=c2e2c,
            e2c2e=e2c2e,
            e2v=e2v,
            v2e=v2e,
            v2c=v2c,
            e2c2v=e2c2v,
            c2v=c2v,
            c_owner_mask=c_owner_mask,
            e_owner_mask=e_owner_mask,
            v_owner_mask=v_owner_mask,
            c_glb_index=c_glb_index,
            e_glb_index=e_glb_index,
            v_glb_index=v_glb_index,
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            edge_areas=edge_areas,
            f_e=f_e,
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            cell_areas=cell_areas,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_cell_x=dual_normal_cell_x,
            dual_normal_cell_y=dual_normal_cell_y,
            edge_center_lat=edge_center_lat,
            edge_center_lon=edge_center_lon,
            primal_normal_x=primal_normal_x,
            primal_normal_y=primal_normal_y,
            mean_cell_area=mean_cell_area,
            comm_id=comm_id,
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
            vertical_size=vertical_size,
            limited_area=limited_area,
            backend=backend,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "grid_init convert time: %s"
                    % str(perf_counters["convert_end_time"] - perf_counters["convert_start_time"])
                )
                logger.info("grid_init execution time: %s" % str(func_end_time - func_start_time))

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                cell_starts_arr = (
                    _conversion.as_array(ffi, cell_starts, _definitions.INT32)
                    if cell_starts is not None
                    else None
                )
                msg = "shape of cell_starts after computation = %s" % str(
                    cell_starts_arr.shape if cell_starts is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "cell_starts after computation: %s" % str(cell_starts_arr)
                    if cell_starts is not None
                    else "None"
                )
                logger.debug(msg)

                cell_ends_arr = (
                    _conversion.as_array(ffi, cell_ends, _definitions.INT32)
                    if cell_ends is not None
                    else None
                )
                msg = "shape of cell_ends after computation = %s" % str(
                    cell_ends_arr.shape if cell_ends is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "cell_ends after computation: %s" % str(cell_ends_arr)
                    if cell_ends is not None
                    else "None"
                )
                logger.debug(msg)

                vertex_starts_arr = (
                    _conversion.as_array(ffi, vertex_starts, _definitions.INT32)
                    if vertex_starts is not None
                    else None
                )
                msg = "shape of vertex_starts after computation = %s" % str(
                    vertex_starts_arr.shape if vertex_starts is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vertex_starts after computation: %s" % str(vertex_starts_arr)
                    if vertex_starts is not None
                    else "None"
                )
                logger.debug(msg)

                vertex_ends_arr = (
                    _conversion.as_array(ffi, vertex_ends, _definitions.INT32)
                    if vertex_ends is not None
                    else None
                )
                msg = "shape of vertex_ends after computation = %s" % str(
                    vertex_ends_arr.shape if vertex_ends is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vertex_ends after computation: %s" % str(vertex_ends_arr)
                    if vertex_ends is not None
                    else "None"
                )
                logger.debug(msg)

                edge_starts_arr = (
                    _conversion.as_array(ffi, edge_starts, _definitions.INT32)
                    if edge_starts is not None
                    else None
                )
                msg = "shape of edge_starts after computation = %s" % str(
                    edge_starts_arr.shape if edge_starts is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "edge_starts after computation: %s" % str(edge_starts_arr)
                    if edge_starts is not None
                    else "None"
                )
                logger.debug(msg)

                edge_ends_arr = (
                    _conversion.as_array(ffi, edge_ends, _definitions.INT32)
                    if edge_ends is not None
                    else None
                )
                msg = "shape of edge_ends after computation = %s" % str(
                    edge_ends_arr.shape if edge_ends is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "edge_ends after computation: %s" % str(edge_ends_arr)
                    if edge_ends is not None
                    else "None"
                )
                logger.debug(msg)

                c2e_arr = (
                    _conversion.as_array(ffi, c2e, _definitions.INT32) if c2e is not None else None
                )
                msg = "shape of c2e after computation = %s" % str(
                    c2e_arr.shape if c2e is not None else "None"
                )
                logger.debug(msg)
                msg = "c2e after computation: %s" % str(c2e_arr) if c2e is not None else "None"
                logger.debug(msg)

                e2c_arr = (
                    _conversion.as_array(ffi, e2c, _definitions.INT32) if e2c is not None else None
                )
                msg = "shape of e2c after computation = %s" % str(
                    e2c_arr.shape if e2c is not None else "None"
                )
                logger.debug(msg)
                msg = "e2c after computation: %s" % str(e2c_arr) if e2c is not None else "None"
                logger.debug(msg)

                c2e2c_arr = (
                    _conversion.as_array(ffi, c2e2c, _definitions.INT32)
                    if c2e2c is not None
                    else None
                )
                msg = "shape of c2e2c after computation = %s" % str(
                    c2e2c_arr.shape if c2e2c is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "c2e2c after computation: %s" % str(c2e2c_arr) if c2e2c is not None else "None"
                )
                logger.debug(msg)

                e2c2e_arr = (
                    _conversion.as_array(ffi, e2c2e, _definitions.INT32)
                    if e2c2e is not None
                    else None
                )
                msg = "shape of e2c2e after computation = %s" % str(
                    e2c2e_arr.shape if e2c2e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "e2c2e after computation: %s" % str(e2c2e_arr) if e2c2e is not None else "None"
                )
                logger.debug(msg)

                e2v_arr = (
                    _conversion.as_array(ffi, e2v, _definitions.INT32) if e2v is not None else None
                )
                msg = "shape of e2v after computation = %s" % str(
                    e2v_arr.shape if e2v is not None else "None"
                )
                logger.debug(msg)
                msg = "e2v after computation: %s" % str(e2v_arr) if e2v is not None else "None"
                logger.debug(msg)

                v2e_arr = (
                    _conversion.as_array(ffi, v2e, _definitions.INT32) if v2e is not None else None
                )
                msg = "shape of v2e after computation = %s" % str(
                    v2e_arr.shape if v2e is not None else "None"
                )
                logger.debug(msg)
                msg = "v2e after computation: %s" % str(v2e_arr) if v2e is not None else "None"
                logger.debug(msg)

                v2c_arr = (
                    _conversion.as_array(ffi, v2c, _definitions.INT32) if v2c is not None else None
                )
                msg = "shape of v2c after computation = %s" % str(
                    v2c_arr.shape if v2c is not None else "None"
                )
                logger.debug(msg)
                msg = "v2c after computation: %s" % str(v2c_arr) if v2c is not None else "None"
                logger.debug(msg)

                e2c2v_arr = (
                    _conversion.as_array(ffi, e2c2v, _definitions.INT32)
                    if e2c2v is not None
                    else None
                )
                msg = "shape of e2c2v after computation = %s" % str(
                    e2c2v_arr.shape if e2c2v is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "e2c2v after computation: %s" % str(e2c2v_arr) if e2c2v is not None else "None"
                )
                logger.debug(msg)

                c2v_arr = (
                    _conversion.as_array(ffi, c2v, _definitions.INT32) if c2v is not None else None
                )
                msg = "shape of c2v after computation = %s" % str(
                    c2v_arr.shape if c2v is not None else "None"
                )
                logger.debug(msg)
                msg = "c2v after computation: %s" % str(c2v_arr) if c2v is not None else "None"
                logger.debug(msg)

                c_owner_mask_arr = (
                    _conversion.as_array(ffi, c_owner_mask, _definitions.BOOL)
                    if c_owner_mask is not None
                    else None
                )
                msg = "shape of c_owner_mask after computation = %s" % str(
                    c_owner_mask_arr.shape if c_owner_mask is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "c_owner_mask after computation: %s" % str(c_owner_mask_arr)
                    if c_owner_mask is not None
                    else "None"
                )
                logger.debug(msg)

                e_owner_mask_arr = (
                    _conversion.as_array(ffi, e_owner_mask, _definitions.BOOL)
                    if e_owner_mask is not None
                    else None
                )
                msg = "shape of e_owner_mask after computation = %s" % str(
                    e_owner_mask_arr.shape if e_owner_mask is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "e_owner_mask after computation: %s" % str(e_owner_mask_arr)
                    if e_owner_mask is not None
                    else "None"
                )
                logger.debug(msg)

                v_owner_mask_arr = (
                    _conversion.as_array(ffi, v_owner_mask, _definitions.BOOL)
                    if v_owner_mask is not None
                    else None
                )
                msg = "shape of v_owner_mask after computation = %s" % str(
                    v_owner_mask_arr.shape if v_owner_mask is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "v_owner_mask after computation: %s" % str(v_owner_mask_arr)
                    if v_owner_mask is not None
                    else "None"
                )
                logger.debug(msg)

                c_glb_index_arr = (
                    _conversion.as_array(ffi, c_glb_index, _definitions.INT32)
                    if c_glb_index is not None
                    else None
                )
                msg = "shape of c_glb_index after computation = %s" % str(
                    c_glb_index_arr.shape if c_glb_index is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "c_glb_index after computation: %s" % str(c_glb_index_arr)
                    if c_glb_index is not None
                    else "None"
                )
                logger.debug(msg)

                e_glb_index_arr = (
                    _conversion.as_array(ffi, e_glb_index, _definitions.INT32)
                    if e_glb_index is not None
                    else None
                )
                msg = "shape of e_glb_index after computation = %s" % str(
                    e_glb_index_arr.shape if e_glb_index is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "e_glb_index after computation: %s" % str(e_glb_index_arr)
                    if e_glb_index is not None
                    else "None"
                )
                logger.debug(msg)

                v_glb_index_arr = (
                    _conversion.as_array(ffi, v_glb_index, _definitions.INT32)
                    if v_glb_index is not None
                    else None
                )
                msg = "shape of v_glb_index after computation = %s" % str(
                    v_glb_index_arr.shape if v_glb_index is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "v_glb_index after computation: %s" % str(v_glb_index_arr)
                    if v_glb_index is not None
                    else "None"
                )
                logger.debug(msg)

                tangent_orientation_arr = (
                    _conversion.as_array(ffi, tangent_orientation, _definitions.FLOAT64)
                    if tangent_orientation is not None
                    else None
                )
                msg = "shape of tangent_orientation after computation = %s" % str(
                    tangent_orientation_arr.shape if tangent_orientation is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "tangent_orientation after computation: %s" % str(tangent_orientation_arr)
                    if tangent_orientation is not None
                    else "None"
                )
                logger.debug(msg)

                inverse_primal_edge_lengths_arr = (
                    _conversion.as_array(ffi, inverse_primal_edge_lengths, _definitions.FLOAT64)
                    if inverse_primal_edge_lengths is not None
                    else None
                )
                msg = "shape of inverse_primal_edge_lengths after computation = %s" % str(
                    inverse_primal_edge_lengths_arr.shape
                    if inverse_primal_edge_lengths is not None
                    else "None"
                )
                logger.debug(msg)
                msg = (
                    "inverse_primal_edge_lengths after computation: %s"
                    % str(inverse_primal_edge_lengths_arr)
                    if inverse_primal_edge_lengths is not None
                    else "None"
                )
                logger.debug(msg)

                inv_dual_edge_length_arr = (
                    _conversion.as_array(ffi, inv_dual_edge_length, _definitions.FLOAT64)
                    if inv_dual_edge_length is not None
                    else None
                )
                msg = "shape of inv_dual_edge_length after computation = %s" % str(
                    inv_dual_edge_length_arr.shape if inv_dual_edge_length is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "inv_dual_edge_length after computation: %s" % str(inv_dual_edge_length_arr)
                    if inv_dual_edge_length is not None
                    else "None"
                )
                logger.debug(msg)

                inv_vert_vert_length_arr = (
                    _conversion.as_array(ffi, inv_vert_vert_length, _definitions.FLOAT64)
                    if inv_vert_vert_length is not None
                    else None
                )
                msg = "shape of inv_vert_vert_length after computation = %s" % str(
                    inv_vert_vert_length_arr.shape if inv_vert_vert_length is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "inv_vert_vert_length after computation: %s" % str(inv_vert_vert_length_arr)
                    if inv_vert_vert_length is not None
                    else "None"
                )
                logger.debug(msg)

                edge_areas_arr = (
                    _conversion.as_array(ffi, edge_areas, _definitions.FLOAT64)
                    if edge_areas is not None
                    else None
                )
                msg = "shape of edge_areas after computation = %s" % str(
                    edge_areas_arr.shape if edge_areas is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "edge_areas after computation: %s" % str(edge_areas_arr)
                    if edge_areas is not None
                    else "None"
                )
                logger.debug(msg)

                f_e_arr = (
                    _conversion.as_array(ffi, f_e, _definitions.FLOAT64)
                    if f_e is not None
                    else None
                )
                msg = "shape of f_e after computation = %s" % str(
                    f_e_arr.shape if f_e is not None else "None"
                )
                logger.debug(msg)
                msg = "f_e after computation: %s" % str(f_e_arr) if f_e is not None else "None"
                logger.debug(msg)

                cell_center_lat_arr = (
                    _conversion.as_array(ffi, cell_center_lat, _definitions.FLOAT64)
                    if cell_center_lat is not None
                    else None
                )
                msg = "shape of cell_center_lat after computation = %s" % str(
                    cell_center_lat_arr.shape if cell_center_lat is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "cell_center_lat after computation: %s" % str(cell_center_lat_arr)
                    if cell_center_lat is not None
                    else "None"
                )
                logger.debug(msg)

                cell_center_lon_arr = (
                    _conversion.as_array(ffi, cell_center_lon, _definitions.FLOAT64)
                    if cell_center_lon is not None
                    else None
                )
                msg = "shape of cell_center_lon after computation = %s" % str(
                    cell_center_lon_arr.shape if cell_center_lon is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "cell_center_lon after computation: %s" % str(cell_center_lon_arr)
                    if cell_center_lon is not None
                    else "None"
                )
                logger.debug(msg)

                cell_areas_arr = (
                    _conversion.as_array(ffi, cell_areas, _definitions.FLOAT64)
                    if cell_areas is not None
                    else None
                )
                msg = "shape of cell_areas after computation = %s" % str(
                    cell_areas_arr.shape if cell_areas is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "cell_areas after computation: %s" % str(cell_areas_arr)
                    if cell_areas is not None
                    else "None"
                )
                logger.debug(msg)

                primal_normal_vert_x_arr = (
                    _conversion.as_array(ffi, primal_normal_vert_x, _definitions.FLOAT64)
                    if primal_normal_vert_x is not None
                    else None
                )
                msg = "shape of primal_normal_vert_x after computation = %s" % str(
                    primal_normal_vert_x_arr.shape if primal_normal_vert_x is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "primal_normal_vert_x after computation: %s" % str(primal_normal_vert_x_arr)
                    if primal_normal_vert_x is not None
                    else "None"
                )
                logger.debug(msg)

                primal_normal_vert_y_arr = (
                    _conversion.as_array(ffi, primal_normal_vert_y, _definitions.FLOAT64)
                    if primal_normal_vert_y is not None
                    else None
                )
                msg = "shape of primal_normal_vert_y after computation = %s" % str(
                    primal_normal_vert_y_arr.shape if primal_normal_vert_y is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "primal_normal_vert_y after computation: %s" % str(primal_normal_vert_y_arr)
                    if primal_normal_vert_y is not None
                    else "None"
                )
                logger.debug(msg)

                dual_normal_vert_x_arr = (
                    _conversion.as_array(ffi, dual_normal_vert_x, _definitions.FLOAT64)
                    if dual_normal_vert_x is not None
                    else None
                )
                msg = "shape of dual_normal_vert_x after computation = %s" % str(
                    dual_normal_vert_x_arr.shape if dual_normal_vert_x is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "dual_normal_vert_x after computation: %s" % str(dual_normal_vert_x_arr)
                    if dual_normal_vert_x is not None
                    else "None"
                )
                logger.debug(msg)

                dual_normal_vert_y_arr = (
                    _conversion.as_array(ffi, dual_normal_vert_y, _definitions.FLOAT64)
                    if dual_normal_vert_y is not None
                    else None
                )
                msg = "shape of dual_normal_vert_y after computation = %s" % str(
                    dual_normal_vert_y_arr.shape if dual_normal_vert_y is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "dual_normal_vert_y after computation: %s" % str(dual_normal_vert_y_arr)
                    if dual_normal_vert_y is not None
                    else "None"
                )
                logger.debug(msg)

                primal_normal_cell_x_arr = (
                    _conversion.as_array(ffi, primal_normal_cell_x, _definitions.FLOAT64)
                    if primal_normal_cell_x is not None
                    else None
                )
                msg = "shape of primal_normal_cell_x after computation = %s" % str(
                    primal_normal_cell_x_arr.shape if primal_normal_cell_x is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "primal_normal_cell_x after computation: %s" % str(primal_normal_cell_x_arr)
                    if primal_normal_cell_x is not None
                    else "None"
                )
                logger.debug(msg)

                primal_normal_cell_y_arr = (
                    _conversion.as_array(ffi, primal_normal_cell_y, _definitions.FLOAT64)
                    if primal_normal_cell_y is not None
                    else None
                )
                msg = "shape of primal_normal_cell_y after computation = %s" % str(
                    primal_normal_cell_y_arr.shape if primal_normal_cell_y is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "primal_normal_cell_y after computation: %s" % str(primal_normal_cell_y_arr)
                    if primal_normal_cell_y is not None
                    else "None"
                )
                logger.debug(msg)

                dual_normal_cell_x_arr = (
                    _conversion.as_array(ffi, dual_normal_cell_x, _definitions.FLOAT64)
                    if dual_normal_cell_x is not None
                    else None
                )
                msg = "shape of dual_normal_cell_x after computation = %s" % str(
                    dual_normal_cell_x_arr.shape if dual_normal_cell_x is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "dual_normal_cell_x after computation: %s" % str(dual_normal_cell_x_arr)
                    if dual_normal_cell_x is not None
                    else "None"
                )
                logger.debug(msg)

                dual_normal_cell_y_arr = (
                    _conversion.as_array(ffi, dual_normal_cell_y, _definitions.FLOAT64)
                    if dual_normal_cell_y is not None
                    else None
                )
                msg = "shape of dual_normal_cell_y after computation = %s" % str(
                    dual_normal_cell_y_arr.shape if dual_normal_cell_y is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "dual_normal_cell_y after computation: %s" % str(dual_normal_cell_y_arr)
                    if dual_normal_cell_y is not None
                    else "None"
                )
                logger.debug(msg)

                edge_center_lat_arr = (
                    _conversion.as_array(ffi, edge_center_lat, _definitions.FLOAT64)
                    if edge_center_lat is not None
                    else None
                )
                msg = "shape of edge_center_lat after computation = %s" % str(
                    edge_center_lat_arr.shape if edge_center_lat is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "edge_center_lat after computation: %s" % str(edge_center_lat_arr)
                    if edge_center_lat is not None
                    else "None"
                )
                logger.debug(msg)

                edge_center_lon_arr = (
                    _conversion.as_array(ffi, edge_center_lon, _definitions.FLOAT64)
                    if edge_center_lon is not None
                    else None
                )
                msg = "shape of edge_center_lon after computation = %s" % str(
                    edge_center_lon_arr.shape if edge_center_lon is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "edge_center_lon after computation: %s" % str(edge_center_lon_arr)
                    if edge_center_lon is not None
                    else "None"
                )
                logger.debug(msg)

                primal_normal_x_arr = (
                    _conversion.as_array(ffi, primal_normal_x, _definitions.FLOAT64)
                    if primal_normal_x is not None
                    else None
                )
                msg = "shape of primal_normal_x after computation = %s" % str(
                    primal_normal_x_arr.shape if primal_normal_x is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "primal_normal_x after computation: %s" % str(primal_normal_x_arr)
                    if primal_normal_x is not None
                    else "None"
                )
                logger.debug(msg)

                primal_normal_y_arr = (
                    _conversion.as_array(ffi, primal_normal_y, _definitions.FLOAT64)
                    if primal_normal_y is not None
                    else None
                )
                msg = "shape of primal_normal_y after computation = %s" % str(
                    primal_normal_y_arr.shape if primal_normal_y is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "primal_normal_y after computation: %s" % str(primal_normal_y_arr)
                    if primal_normal_y is not None
                    else "None"
                )
                logger.debug(msg)

        if __debug__:
            logger.info("Python execution of grid_init completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0
