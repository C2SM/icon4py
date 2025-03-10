# imports for generated wrapper code
import logging

from grid import ffi

try:
    import cupy as cp  # TODO remove this import
except ImportError:
    cp = None
import gt4py.next as gtx
from gt4py.next.type_system import type_specifications as ts
from icon4py.tools.py2fgen import wrapper_utils

# logger setup
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

if cp is not None:
    logging.info(cp.show_config())

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
    on_gpu,
):
    try:
        logging.info("Python Execution Context Start")

        # Convert ptr to GT4Py fields

        cell_starts = wrapper_utils.ArrayDescriptor(
            cell_starts, shape=(cell_starts_size_0,), on_gpu=False, is_optional=False
        )

        cell_ends = wrapper_utils.ArrayDescriptor(
            cell_ends, shape=(cell_ends_size_0,), on_gpu=False, is_optional=False
        )

        vertex_starts = wrapper_utils.ArrayDescriptor(
            vertex_starts, shape=(vertex_starts_size_0,), on_gpu=False, is_optional=False
        )

        vertex_ends = wrapper_utils.ArrayDescriptor(
            vertex_ends, shape=(vertex_ends_size_0,), on_gpu=False, is_optional=False
        )

        edge_starts = wrapper_utils.ArrayDescriptor(
            edge_starts, shape=(edge_starts_size_0,), on_gpu=False, is_optional=False
        )

        edge_ends = wrapper_utils.ArrayDescriptor(
            edge_ends, shape=(edge_ends_size_0,), on_gpu=False, is_optional=False
        )

        c2e = wrapper_utils.ArrayDescriptor(
            c2e,
            shape=(
                c2e_size_0,
                c2e_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        e2c = wrapper_utils.ArrayDescriptor(
            e2c,
            shape=(
                e2c_size_0,
                e2c_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        c2e2c = wrapper_utils.ArrayDescriptor(
            c2e2c,
            shape=(
                c2e2c_size_0,
                c2e2c_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        e2c2e = wrapper_utils.ArrayDescriptor(
            e2c2e,
            shape=(
                e2c2e_size_0,
                e2c2e_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        e2v = wrapper_utils.ArrayDescriptor(
            e2v,
            shape=(
                e2v_size_0,
                e2v_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        v2e = wrapper_utils.ArrayDescriptor(
            v2e,
            shape=(
                v2e_size_0,
                v2e_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        v2c = wrapper_utils.ArrayDescriptor(
            v2c,
            shape=(
                v2c_size_0,
                v2c_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        e2c2v = wrapper_utils.ArrayDescriptor(
            e2c2v,
            shape=(
                e2c2v_size_0,
                e2c2v_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        c2v = wrapper_utils.ArrayDescriptor(
            c2v,
            shape=(
                c2v_size_0,
                c2v_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        c_owner_mask = wrapper_utils.ArrayDescriptor(
            c_owner_mask, shape=(c_owner_mask_size_0,), on_gpu=False, is_optional=False
        )

        e_owner_mask = wrapper_utils.ArrayDescriptor(
            e_owner_mask, shape=(e_owner_mask_size_0,), on_gpu=False, is_optional=False
        )

        v_owner_mask = wrapper_utils.ArrayDescriptor(
            v_owner_mask, shape=(v_owner_mask_size_0,), on_gpu=False, is_optional=False
        )

        c_glb_index = wrapper_utils.ArrayDescriptor(
            c_glb_index, shape=(c_glb_index_size_0,), on_gpu=False, is_optional=False
        )

        e_glb_index = wrapper_utils.ArrayDescriptor(
            e_glb_index, shape=(e_glb_index_size_0,), on_gpu=False, is_optional=False
        )

        v_glb_index = wrapper_utils.ArrayDescriptor(
            v_glb_index, shape=(v_glb_index_size_0,), on_gpu=False, is_optional=False
        )

        tangent_orientation = wrapper_utils.ArrayDescriptor(
            tangent_orientation,
            shape=(tangent_orientation_size_0,),
            on_gpu=on_gpu,
            is_optional=False,
        )

        inverse_primal_edge_lengths = wrapper_utils.ArrayDescriptor(
            inverse_primal_edge_lengths,
            shape=(inverse_primal_edge_lengths_size_0,),
            on_gpu=on_gpu,
            is_optional=False,
        )

        inv_dual_edge_length = wrapper_utils.ArrayDescriptor(
            inv_dual_edge_length,
            shape=(inv_dual_edge_length_size_0,),
            on_gpu=on_gpu,
            is_optional=False,
        )

        inv_vert_vert_length = wrapper_utils.ArrayDescriptor(
            inv_vert_vert_length,
            shape=(inv_vert_vert_length_size_0,),
            on_gpu=on_gpu,
            is_optional=False,
        )

        edge_areas = wrapper_utils.ArrayDescriptor(
            edge_areas, shape=(edge_areas_size_0,), on_gpu=on_gpu, is_optional=False
        )

        f_e = wrapper_utils.ArrayDescriptor(
            f_e, shape=(f_e_size_0,), on_gpu=on_gpu, is_optional=False
        )

        cell_center_lat = wrapper_utils.ArrayDescriptor(
            cell_center_lat, shape=(cell_center_lat_size_0,), on_gpu=on_gpu, is_optional=False
        )

        cell_center_lon = wrapper_utils.ArrayDescriptor(
            cell_center_lon, shape=(cell_center_lon_size_0,), on_gpu=on_gpu, is_optional=False
        )

        cell_areas = wrapper_utils.ArrayDescriptor(
            cell_areas, shape=(cell_areas_size_0,), on_gpu=on_gpu, is_optional=False
        )

        primal_normal_vert_x = wrapper_utils.ArrayDescriptor(
            primal_normal_vert_x,
            shape=(
                primal_normal_vert_x_size_0,
                primal_normal_vert_x_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        primal_normal_vert_y = wrapper_utils.ArrayDescriptor(
            primal_normal_vert_y,
            shape=(
                primal_normal_vert_y_size_0,
                primal_normal_vert_y_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        dual_normal_vert_x = wrapper_utils.ArrayDescriptor(
            dual_normal_vert_x,
            shape=(
                dual_normal_vert_x_size_0,
                dual_normal_vert_x_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        dual_normal_vert_y = wrapper_utils.ArrayDescriptor(
            dual_normal_vert_y,
            shape=(
                dual_normal_vert_y_size_0,
                dual_normal_vert_y_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        primal_normal_cell_x = wrapper_utils.ArrayDescriptor(
            primal_normal_cell_x,
            shape=(
                primal_normal_cell_x_size_0,
                primal_normal_cell_x_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        primal_normal_cell_y = wrapper_utils.ArrayDescriptor(
            primal_normal_cell_y,
            shape=(
                primal_normal_cell_y_size_0,
                primal_normal_cell_y_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        dual_normal_cell_x = wrapper_utils.ArrayDescriptor(
            dual_normal_cell_x,
            shape=(
                dual_normal_cell_x_size_0,
                dual_normal_cell_x_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        dual_normal_cell_y = wrapper_utils.ArrayDescriptor(
            dual_normal_cell_y,
            shape=(
                dual_normal_cell_y_size_0,
                dual_normal_cell_y_size_1,
            ),
            on_gpu=on_gpu,
            is_optional=False,
        )

        edge_center_lat = wrapper_utils.ArrayDescriptor(
            edge_center_lat, shape=(edge_center_lat_size_0,), on_gpu=on_gpu, is_optional=False
        )

        edge_center_lon = wrapper_utils.ArrayDescriptor(
            edge_center_lon, shape=(edge_center_lon_size_0,), on_gpu=on_gpu, is_optional=False
        )

        primal_normal_x = wrapper_utils.ArrayDescriptor(
            primal_normal_x, shape=(primal_normal_x_size_0,), on_gpu=on_gpu, is_optional=False
        )

        primal_normal_y = wrapper_utils.ArrayDescriptor(
            primal_normal_y, shape=(primal_normal_y_size_0,), on_gpu=on_gpu, is_optional=False
        )

        assert isinstance(limited_area, int)
        limited_area = limited_area != 0

        grid_init(
            ffi=ffi,
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
        )

        # debug info

        msg = "shape of cell_starts after computation = %s" % str(
            cell_starts.shape if cell_starts is not None else "None"
        )
        logging.debug(msg)
        # msg = 'cell_starts after computation: %s' % str(cell_starts.ndarray if cell_starts is not None else "None")
        logging.debug(msg)

        msg = "shape of cell_ends after computation = %s" % str(
            cell_ends.shape if cell_ends is not None else "None"
        )
        logging.debug(msg)
        # msg = 'cell_ends after computation: %s' % str(cell_ends.ndarray if cell_ends is not None else "None")
        logging.debug(msg)

        msg = "shape of vertex_starts after computation = %s" % str(
            vertex_starts.shape if vertex_starts is not None else "None"
        )
        logging.debug(msg)
        # msg = 'vertex_starts after computation: %s' % str(vertex_starts.ndarray if vertex_starts is not None else "None")
        logging.debug(msg)

        msg = "shape of vertex_ends after computation = %s" % str(
            vertex_ends.shape if vertex_ends is not None else "None"
        )
        logging.debug(msg)
        # msg = 'vertex_ends after computation: %s' % str(vertex_ends.ndarray if vertex_ends is not None else "None")
        logging.debug(msg)

        msg = "shape of edge_starts after computation = %s" % str(
            edge_starts.shape if edge_starts is not None else "None"
        )
        logging.debug(msg)
        # msg = 'edge_starts after computation: %s' % str(edge_starts.ndarray if edge_starts is not None else "None")
        logging.debug(msg)

        msg = "shape of edge_ends after computation = %s" % str(
            edge_ends.shape if edge_ends is not None else "None"
        )
        logging.debug(msg)
        # msg = 'edge_ends after computation: %s' % str(edge_ends.ndarray if edge_ends is not None else "None")
        logging.debug(msg)

        msg = "shape of c2e after computation = %s" % str(c2e.shape if c2e is not None else "None")
        logging.debug(msg)
        # msg = 'c2e after computation: %s' % str(c2e.ndarray if c2e is not None else "None")
        logging.debug(msg)

        msg = "shape of e2c after computation = %s" % str(e2c.shape if e2c is not None else "None")
        logging.debug(msg)
        # msg = 'e2c after computation: %s' % str(e2c.ndarray if e2c is not None else "None")
        logging.debug(msg)

        msg = "shape of c2e2c after computation = %s" % str(
            c2e2c.shape if c2e2c is not None else "None"
        )
        logging.debug(msg)
        # msg = 'c2e2c after computation: %s' % str(c2e2c.ndarray if c2e2c is not None else "None")
        logging.debug(msg)

        msg = "shape of e2c2e after computation = %s" % str(
            e2c2e.shape if e2c2e is not None else "None"
        )
        logging.debug(msg)
        # msg = 'e2c2e after computation: %s' % str(e2c2e.ndarray if e2c2e is not None else "None")
        logging.debug(msg)

        msg = "shape of e2v after computation = %s" % str(e2v.shape if e2v is not None else "None")
        logging.debug(msg)
        # msg = 'e2v after computation: %s' % str(e2v.ndarray if e2v is not None else "None")
        logging.debug(msg)

        msg = "shape of v2e after computation = %s" % str(v2e.shape if v2e is not None else "None")
        logging.debug(msg)
        # msg = 'v2e after computation: %s' % str(v2e.ndarray if v2e is not None else "None")
        logging.debug(msg)

        msg = "shape of v2c after computation = %s" % str(v2c.shape if v2c is not None else "None")
        logging.debug(msg)
        # msg = 'v2c after computation: %s' % str(v2c.ndarray if v2c is not None else "None")
        logging.debug(msg)

        msg = "shape of e2c2v after computation = %s" % str(
            e2c2v.shape if e2c2v is not None else "None"
        )
        logging.debug(msg)
        # msg = 'e2c2v after computation: %s' % str(e2c2v.ndarray if e2c2v is not None else "None")
        logging.debug(msg)

        msg = "shape of c2v after computation = %s" % str(c2v.shape if c2v is not None else "None")
        logging.debug(msg)
        # msg = 'c2v after computation: %s' % str(c2v.ndarray if c2v is not None else "None")
        logging.debug(msg)

        msg = "shape of c_owner_mask after computation = %s" % str(
            c_owner_mask.shape if c_owner_mask is not None else "None"
        )
        logging.debug(msg)
        # msg = 'c_owner_mask after computation: %s' % str(c_owner_mask.ndarray if c_owner_mask is not None else "None")
        logging.debug(msg)

        msg = "shape of e_owner_mask after computation = %s" % str(
            e_owner_mask.shape if e_owner_mask is not None else "None"
        )
        logging.debug(msg)
        # msg = 'e_owner_mask after computation: %s' % str(e_owner_mask.ndarray if e_owner_mask is not None else "None")
        logging.debug(msg)

        msg = "shape of v_owner_mask after computation = %s" % str(
            v_owner_mask.shape if v_owner_mask is not None else "None"
        )
        logging.debug(msg)
        # msg = 'v_owner_mask after computation: %s' % str(v_owner_mask.ndarray if v_owner_mask is not None else "None")
        logging.debug(msg)

        msg = "shape of c_glb_index after computation = %s" % str(
            c_glb_index.shape if c_glb_index is not None else "None"
        )
        logging.debug(msg)
        # msg = 'c_glb_index after computation: %s' % str(c_glb_index.ndarray if c_glb_index is not None else "None")
        logging.debug(msg)

        msg = "shape of e_glb_index after computation = %s" % str(
            e_glb_index.shape if e_glb_index is not None else "None"
        )
        logging.debug(msg)
        # msg = 'e_glb_index after computation: %s' % str(e_glb_index.ndarray if e_glb_index is not None else "None")
        logging.debug(msg)

        msg = "shape of v_glb_index after computation = %s" % str(
            v_glb_index.shape if v_glb_index is not None else "None"
        )
        logging.debug(msg)
        # msg = 'v_glb_index after computation: %s' % str(v_glb_index.ndarray if v_glb_index is not None else "None")
        logging.debug(msg)

        msg = "shape of tangent_orientation after computation = %s" % str(
            tangent_orientation.shape if tangent_orientation is not None else "None"
        )
        logging.debug(msg)
        # msg = 'tangent_orientation after computation: %s' % str(tangent_orientation.ndarray if tangent_orientation is not None else "None")
        logging.debug(msg)

        msg = "shape of inverse_primal_edge_lengths after computation = %s" % str(
            inverse_primal_edge_lengths.shape if inverse_primal_edge_lengths is not None else "None"
        )
        logging.debug(msg)
        # msg = 'inverse_primal_edge_lengths after computation: %s' % str(inverse_primal_edge_lengths.ndarray if inverse_primal_edge_lengths is not None else "None")
        logging.debug(msg)

        msg = "shape of inv_dual_edge_length after computation = %s" % str(
            inv_dual_edge_length.shape if inv_dual_edge_length is not None else "None"
        )
        logging.debug(msg)
        # msg = 'inv_dual_edge_length after computation: %s' % str(inv_dual_edge_length.ndarray if inv_dual_edge_length is not None else "None")
        logging.debug(msg)

        msg = "shape of inv_vert_vert_length after computation = %s" % str(
            inv_vert_vert_length.shape if inv_vert_vert_length is not None else "None"
        )
        logging.debug(msg)
        # msg = 'inv_vert_vert_length after computation: %s' % str(inv_vert_vert_length.ndarray if inv_vert_vert_length is not None else "None")
        logging.debug(msg)

        msg = "shape of edge_areas after computation = %s" % str(
            edge_areas.shape if edge_areas is not None else "None"
        )
        logging.debug(msg)
        # msg = 'edge_areas after computation: %s' % str(edge_areas.ndarray if edge_areas is not None else "None")
        logging.debug(msg)

        msg = "shape of f_e after computation = %s" % str(f_e.shape if f_e is not None else "None")
        logging.debug(msg)
        # msg = 'f_e after computation: %s' % str(f_e.ndarray if f_e is not None else "None")
        logging.debug(msg)

        msg = "shape of cell_center_lat after computation = %s" % str(
            cell_center_lat.shape if cell_center_lat is not None else "None"
        )
        logging.debug(msg)
        # msg = 'cell_center_lat after computation: %s' % str(cell_center_lat.ndarray if cell_center_lat is not None else "None")
        logging.debug(msg)

        msg = "shape of cell_center_lon after computation = %s" % str(
            cell_center_lon.shape if cell_center_lon is not None else "None"
        )
        logging.debug(msg)
        # msg = 'cell_center_lon after computation: %s' % str(cell_center_lon.ndarray if cell_center_lon is not None else "None")
        logging.debug(msg)

        msg = "shape of cell_areas after computation = %s" % str(
            cell_areas.shape if cell_areas is not None else "None"
        )
        logging.debug(msg)
        # msg = 'cell_areas after computation: %s' % str(cell_areas.ndarray if cell_areas is not None else "None")
        logging.debug(msg)

        msg = "shape of primal_normal_vert_x after computation = %s" % str(
            primal_normal_vert_x.shape if primal_normal_vert_x is not None else "None"
        )
        logging.debug(msg)
        # msg = 'primal_normal_vert_x after computation: %s' % str(primal_normal_vert_x.ndarray if primal_normal_vert_x is not None else "None")
        logging.debug(msg)

        msg = "shape of primal_normal_vert_y after computation = %s" % str(
            primal_normal_vert_y.shape if primal_normal_vert_y is not None else "None"
        )
        logging.debug(msg)
        # msg = 'primal_normal_vert_y after computation: %s' % str(primal_normal_vert_y.ndarray if primal_normal_vert_y is not None else "None")
        logging.debug(msg)

        msg = "shape of dual_normal_vert_x after computation = %s" % str(
            dual_normal_vert_x.shape if dual_normal_vert_x is not None else "None"
        )
        logging.debug(msg)
        # msg = 'dual_normal_vert_x after computation: %s' % str(dual_normal_vert_x.ndarray if dual_normal_vert_x is not None else "None")
        logging.debug(msg)

        msg = "shape of dual_normal_vert_y after computation = %s" % str(
            dual_normal_vert_y.shape if dual_normal_vert_y is not None else "None"
        )
        logging.debug(msg)
        # msg = 'dual_normal_vert_y after computation: %s' % str(dual_normal_vert_y.ndarray if dual_normal_vert_y is not None else "None")
        logging.debug(msg)

        msg = "shape of primal_normal_cell_x after computation = %s" % str(
            primal_normal_cell_x.shape if primal_normal_cell_x is not None else "None"
        )
        logging.debug(msg)
        # msg = 'primal_normal_cell_x after computation: %s' % str(primal_normal_cell_x.ndarray if primal_normal_cell_x is not None else "None")
        logging.debug(msg)

        msg = "shape of primal_normal_cell_y after computation = %s" % str(
            primal_normal_cell_y.shape if primal_normal_cell_y is not None else "None"
        )
        logging.debug(msg)
        # msg = 'primal_normal_cell_y after computation: %s' % str(primal_normal_cell_y.ndarray if primal_normal_cell_y is not None else "None")
        logging.debug(msg)

        msg = "shape of dual_normal_cell_x after computation = %s" % str(
            dual_normal_cell_x.shape if dual_normal_cell_x is not None else "None"
        )
        logging.debug(msg)
        # msg = 'dual_normal_cell_x after computation: %s' % str(dual_normal_cell_x.ndarray if dual_normal_cell_x is not None else "None")
        logging.debug(msg)

        msg = "shape of dual_normal_cell_y after computation = %s" % str(
            dual_normal_cell_y.shape if dual_normal_cell_y is not None else "None"
        )
        logging.debug(msg)
        # msg = 'dual_normal_cell_y after computation: %s' % str(dual_normal_cell_y.ndarray if dual_normal_cell_y is not None else "None")
        logging.debug(msg)

        msg = "shape of edge_center_lat after computation = %s" % str(
            edge_center_lat.shape if edge_center_lat is not None else "None"
        )
        logging.debug(msg)
        # msg = 'edge_center_lat after computation: %s' % str(edge_center_lat.ndarray if edge_center_lat is not None else "None")
        logging.debug(msg)

        msg = "shape of edge_center_lon after computation = %s" % str(
            edge_center_lon.shape if edge_center_lon is not None else "None"
        )
        logging.debug(msg)
        # msg = 'edge_center_lon after computation: %s' % str(edge_center_lon.ndarray if edge_center_lon is not None else "None")
        logging.debug(msg)

        msg = "shape of primal_normal_x after computation = %s" % str(
            primal_normal_x.shape if primal_normal_x is not None else "None"
        )
        logging.debug(msg)
        # msg = 'primal_normal_x after computation: %s' % str(primal_normal_x.ndarray if primal_normal_x is not None else "None")
        logging.debug(msg)

        msg = "shape of primal_normal_y after computation = %s" % str(
            primal_normal_y.shape if primal_normal_y is not None else "None"
        )
        logging.debug(msg)
        # msg = 'primal_normal_y after computation: %s' % str(primal_normal_y.ndarray if primal_normal_y is not None else "None")
        logging.debug(msg)

        logging.critical("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0
