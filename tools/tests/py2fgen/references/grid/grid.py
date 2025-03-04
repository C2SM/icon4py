# imports for generated wrapper code
import logging

from grid import ffi
import cupy as cp
import gt4py.next as gtx
from gt4py.next.type_system import type_specifications as ts
from icon4py.tools.py2fgen.settings import config
from icon4py.tools.py2fgen import wrapper_utils

xp = config.array_ns

# logger setup
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
logging.info(cp.show_config())

# embedded function imports
from icon4py.tools.py2fgen.wrappers.grid_wrapper import grid_init


C2E = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E2C = gtx.Dimension("C2E2C", kind=gtx.DimensionKind.LOCAL)
C2V = gtx.Dimension("C2V", kind=gtx.DimensionKind.LOCAL)
Cell = gtx.Dimension("Cell", kind=gtx.DimensionKind.HORIZONTAL)
CellGlobalIndex = gtx.Dimension("CellGlobalIndex", kind=gtx.DimensionKind.HORIZONTAL)
CellIndex = gtx.Dimension("CellIndex", kind=gtx.DimensionKind.HORIZONTAL)
E2C = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
E2C2E = gtx.Dimension("E2C2E", kind=gtx.DimensionKind.LOCAL)
E2C2V = gtx.Dimension("E2C2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
Edge = gtx.Dimension("Edge", kind=gtx.DimensionKind.HORIZONTAL)
EdgeGlobalIndex = gtx.Dimension("EdgeGlobalIndex", kind=gtx.DimensionKind.HORIZONTAL)
EdgeIndex = gtx.Dimension("EdgeIndex", kind=gtx.DimensionKind.HORIZONTAL)
V2C = gtx.Dimension("V2C", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
Vertex = gtx.Dimension("Vertex", kind=gtx.DimensionKind.HORIZONTAL)
VertexGlobalIndex = gtx.Dimension("VertexGlobalIndex", kind=gtx.DimensionKind.HORIZONTAL)
VertexIndex = gtx.Dimension("VertexIndex", kind=gtx.DimensionKind.HORIZONTAL)


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
    comm_id,
    global_root,
    global_level,
    num_vertices,
    num_cells,
    num_edges,
    vertical_size,
    limited_area,
):
    try:
        logging.info("Python Execution Context Start")

        # Convert ptr to GT4Py fields

        cell_starts = wrapper_utils.as_field(
            ffi, xp, cell_starts, ts.ScalarKind.INT32, {CellIndex: cell_starts_size_0}, False
        )

        cell_ends = wrapper_utils.as_field(
            ffi, xp, cell_ends, ts.ScalarKind.INT32, {CellIndex: cell_ends_size_0}, False
        )

        vertex_starts = wrapper_utils.as_field(
            ffi, xp, vertex_starts, ts.ScalarKind.INT32, {VertexIndex: vertex_starts_size_0}, False
        )

        vertex_ends = wrapper_utils.as_field(
            ffi, xp, vertex_ends, ts.ScalarKind.INT32, {VertexIndex: vertex_ends_size_0}, False
        )

        edge_starts = wrapper_utils.as_field(
            ffi, xp, edge_starts, ts.ScalarKind.INT32, {EdgeIndex: edge_starts_size_0}, False
        )

        edge_ends = wrapper_utils.as_field(
            ffi, xp, edge_ends, ts.ScalarKind.INT32, {EdgeIndex: edge_ends_size_0}, False
        )

        c2e = wrapper_utils.as_field(
            ffi, xp, c2e, ts.ScalarKind.INT32, {Cell: c2e_size_0, C2E: c2e_size_1}, False
        )

        e2c = wrapper_utils.as_field(
            ffi, xp, e2c, ts.ScalarKind.INT32, {Edge: e2c_size_0, E2C: e2c_size_1}, False
        )

        c2e2c = wrapper_utils.as_field(
            ffi, xp, c2e2c, ts.ScalarKind.INT32, {Cell: c2e2c_size_0, C2E2C: c2e2c_size_1}, False
        )

        e2c2e = wrapper_utils.as_field(
            ffi, xp, e2c2e, ts.ScalarKind.INT32, {Edge: e2c2e_size_0, E2C2E: e2c2e_size_1}, False
        )

        e2v = wrapper_utils.as_field(
            ffi, xp, e2v, ts.ScalarKind.INT32, {Edge: e2v_size_0, E2V: e2v_size_1}, False
        )

        v2e = wrapper_utils.as_field(
            ffi, xp, v2e, ts.ScalarKind.INT32, {Vertex: v2e_size_0, V2E: v2e_size_1}, False
        )

        v2c = wrapper_utils.as_field(
            ffi, xp, v2c, ts.ScalarKind.INT32, {Vertex: v2c_size_0, V2C: v2c_size_1}, False
        )

        e2c2v = wrapper_utils.as_field(
            ffi, xp, e2c2v, ts.ScalarKind.INT32, {Edge: e2c2v_size_0, E2C2V: e2c2v_size_1}, False
        )

        c2v = wrapper_utils.as_field(
            ffi, xp, c2v, ts.ScalarKind.INT32, {Cell: c2v_size_0, C2V: c2v_size_1}, False
        )

        c_owner_mask = wrapper_utils.as_field(
            ffi, xp, c_owner_mask, ts.ScalarKind.BOOL, {Cell: c_owner_mask_size_0}, False
        )

        e_owner_mask = wrapper_utils.as_field(
            ffi, xp, e_owner_mask, ts.ScalarKind.BOOL, {Edge: e_owner_mask_size_0}, False
        )

        v_owner_mask = wrapper_utils.as_field(
            ffi, xp, v_owner_mask, ts.ScalarKind.BOOL, {Vertex: v_owner_mask_size_0}, False
        )

        c_glb_index = wrapper_utils.as_field(
            ffi, xp, c_glb_index, ts.ScalarKind.INT32, {CellGlobalIndex: c_glb_index_size_0}, False
        )

        e_glb_index = wrapper_utils.as_field(
            ffi, xp, e_glb_index, ts.ScalarKind.INT32, {EdgeGlobalIndex: e_glb_index_size_0}, False
        )

        v_glb_index = wrapper_utils.as_field(
            ffi,
            xp,
            v_glb_index,
            ts.ScalarKind.INT32,
            {VertexGlobalIndex: v_glb_index_size_0},
            False,
        )

        assert isinstance(limited_area, int)
        limited_area = limited_area != 0

        grid_init(
            cell_starts,
            cell_ends,
            vertex_starts,
            vertex_ends,
            edge_starts,
            edge_ends,
            c2e,
            e2c,
            c2e2c,
            e2c2e,
            e2v,
            v2e,
            v2c,
            e2c2v,
            c2v,
            c_owner_mask,
            e_owner_mask,
            v_owner_mask,
            c_glb_index,
            e_glb_index,
            v_glb_index,
            comm_id,
            global_root,
            global_level,
            num_vertices,
            num_cells,
            num_edges,
            vertical_size,
            limited_area,
        )

        # debug info

        msg = "shape of cell_starts after computation = %s" % str(
            cell_starts.shape if cell_starts is not None else "None"
        )
        logging.debug(msg)
        msg = "cell_starts after computation: %s" % str(
            cell_starts.ndarray if cell_starts is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of cell_ends after computation = %s" % str(
            cell_ends.shape if cell_ends is not None else "None"
        )
        logging.debug(msg)
        msg = "cell_ends after computation: %s" % str(
            cell_ends.ndarray if cell_ends is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vertex_starts after computation = %s" % str(
            vertex_starts.shape if vertex_starts is not None else "None"
        )
        logging.debug(msg)
        msg = "vertex_starts after computation: %s" % str(
            vertex_starts.ndarray if vertex_starts is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vertex_ends after computation = %s" % str(
            vertex_ends.shape if vertex_ends is not None else "None"
        )
        logging.debug(msg)
        msg = "vertex_ends after computation: %s" % str(
            vertex_ends.ndarray if vertex_ends is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of edge_starts after computation = %s" % str(
            edge_starts.shape if edge_starts is not None else "None"
        )
        logging.debug(msg)
        msg = "edge_starts after computation: %s" % str(
            edge_starts.ndarray if edge_starts is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of edge_ends after computation = %s" % str(
            edge_ends.shape if edge_ends is not None else "None"
        )
        logging.debug(msg)
        msg = "edge_ends after computation: %s" % str(
            edge_ends.ndarray if edge_ends is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of c2e after computation = %s" % str(c2e.shape if c2e is not None else "None")
        logging.debug(msg)
        msg = "c2e after computation: %s" % str(c2e.ndarray if c2e is not None else "None")
        logging.debug(msg)

        msg = "shape of e2c after computation = %s" % str(e2c.shape if e2c is not None else "None")
        logging.debug(msg)
        msg = "e2c after computation: %s" % str(e2c.ndarray if e2c is not None else "None")
        logging.debug(msg)

        msg = "shape of c2e2c after computation = %s" % str(
            c2e2c.shape if c2e2c is not None else "None"
        )
        logging.debug(msg)
        msg = "c2e2c after computation: %s" % str(c2e2c.ndarray if c2e2c is not None else "None")
        logging.debug(msg)

        msg = "shape of e2c2e after computation = %s" % str(
            e2c2e.shape if e2c2e is not None else "None"
        )
        logging.debug(msg)
        msg = "e2c2e after computation: %s" % str(e2c2e.ndarray if e2c2e is not None else "None")
        logging.debug(msg)

        msg = "shape of e2v after computation = %s" % str(e2v.shape if e2v is not None else "None")
        logging.debug(msg)
        msg = "e2v after computation: %s" % str(e2v.ndarray if e2v is not None else "None")
        logging.debug(msg)

        msg = "shape of v2e after computation = %s" % str(v2e.shape if v2e is not None else "None")
        logging.debug(msg)
        msg = "v2e after computation: %s" % str(v2e.ndarray if v2e is not None else "None")
        logging.debug(msg)

        msg = "shape of v2c after computation = %s" % str(v2c.shape if v2c is not None else "None")
        logging.debug(msg)
        msg = "v2c after computation: %s" % str(v2c.ndarray if v2c is not None else "None")
        logging.debug(msg)

        msg = "shape of e2c2v after computation = %s" % str(
            e2c2v.shape if e2c2v is not None else "None"
        )
        logging.debug(msg)
        msg = "e2c2v after computation: %s" % str(e2c2v.ndarray if e2c2v is not None else "None")
        logging.debug(msg)

        msg = "shape of c2v after computation = %s" % str(c2v.shape if c2v is not None else "None")
        logging.debug(msg)
        msg = "c2v after computation: %s" % str(c2v.ndarray if c2v is not None else "None")
        logging.debug(msg)

        msg = "shape of c_owner_mask after computation = %s" % str(
            c_owner_mask.shape if c_owner_mask is not None else "None"
        )
        logging.debug(msg)
        msg = "c_owner_mask after computation: %s" % str(
            c_owner_mask.ndarray if c_owner_mask is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of e_owner_mask after computation = %s" % str(
            e_owner_mask.shape if e_owner_mask is not None else "None"
        )
        logging.debug(msg)
        msg = "e_owner_mask after computation: %s" % str(
            e_owner_mask.ndarray if e_owner_mask is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of v_owner_mask after computation = %s" % str(
            v_owner_mask.shape if v_owner_mask is not None else "None"
        )
        logging.debug(msg)
        msg = "v_owner_mask after computation: %s" % str(
            v_owner_mask.ndarray if v_owner_mask is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of c_glb_index after computation = %s" % str(
            c_glb_index.shape if c_glb_index is not None else "None"
        )
        logging.debug(msg)
        msg = "c_glb_index after computation: %s" % str(
            c_glb_index.ndarray if c_glb_index is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of e_glb_index after computation = %s" % str(
            e_glb_index.shape if e_glb_index is not None else "None"
        )
        logging.debug(msg)
        msg = "e_glb_index after computation: %s" % str(
            e_glb_index.ndarray if e_glb_index is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of v_glb_index after computation = %s" % str(
            v_glb_index.shape if v_glb_index is not None else "None"
        )
        logging.debug(msg)
        msg = "v_glb_index after computation: %s" % str(
            v_glb_index.ndarray if v_glb_index is not None else "None"
        )
        logging.debug(msg)

        logging.critical("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0
