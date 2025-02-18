# imports for generated wrapper code
import logging

from diffusion import ffi
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
from icon4py.tools.py2fgen.wrappers.diffusion_wrapper import diffusion_run
from icon4py.tools.py2fgen.wrappers.diffusion_wrapper import diffusion_init
from icon4py.tools.py2fgen.wrappers.diffusion_wrapper import grid_init_diffusion


C2E = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E2C = gtx.Dimension("C2E2C", kind=gtx.DimensionKind.LOCAL)
C2E2CO = gtx.Dimension("C2E2CO", kind=gtx.DimensionKind.LOCAL)
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
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
V2C = gtx.Dimension("V2C", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
Vertex = gtx.Dimension("Vertex", kind=gtx.DimensionKind.HORIZONTAL)
VertexGlobalIndex = gtx.Dimension("VertexGlobalIndex", kind=gtx.DimensionKind.HORIZONTAL)
VertexIndex = gtx.Dimension("VertexIndex", kind=gtx.DimensionKind.HORIZONTAL)


@ffi.def_extern()
def diffusion_run_wrapper(
    w,
    w_size_0,
    w_size_1,
    vn,
    vn_size_0,
    vn_size_1,
    exner,
    exner_size_0,
    exner_size_1,
    theta_v,
    theta_v_size_0,
    theta_v_size_1,
    rho,
    rho_size_0,
    rho_size_1,
    hdef_ic,
    hdef_ic_size_0,
    hdef_ic_size_1,
    div_ic,
    div_ic_size_0,
    div_ic_size_1,
    dwdx,
    dwdx_size_0,
    dwdx_size_1,
    dwdy,
    dwdy_size_0,
    dwdy_size_1,
    dtime,
    linit,
):
    try:
        logging.info("Python Execution Context Start")

        # Convert ptr to GT4Py fields

        w = wrapper_utils.as_field(
            ffi, xp, w, ts.ScalarKind.FLOAT64, {Cell: w_size_0, K: w_size_1}, False
        )

        vn = wrapper_utils.as_field(
            ffi, xp, vn, ts.ScalarKind.FLOAT64, {Edge: vn_size_0, K: vn_size_1}, False
        )

        exner = wrapper_utils.as_field(
            ffi, xp, exner, ts.ScalarKind.FLOAT64, {Cell: exner_size_0, K: exner_size_1}, False
        )

        theta_v = wrapper_utils.as_field(
            ffi,
            xp,
            theta_v,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_v_size_0, K: theta_v_size_1},
            False,
        )

        rho = wrapper_utils.as_field(
            ffi, xp, rho, ts.ScalarKind.FLOAT64, {Cell: rho_size_0, K: rho_size_1}, False
        )

        hdef_ic = wrapper_utils.as_field(
            ffi, xp, hdef_ic, ts.ScalarKind.FLOAT64, {Cell: hdef_ic_size_0, K: hdef_ic_size_1}, True
        )

        div_ic = wrapper_utils.as_field(
            ffi, xp, div_ic, ts.ScalarKind.FLOAT64, {Cell: div_ic_size_0, K: div_ic_size_1}, True
        )

        dwdx = wrapper_utils.as_field(
            ffi, xp, dwdx, ts.ScalarKind.FLOAT64, {Cell: dwdx_size_0, K: dwdx_size_1}, True
        )

        dwdy = wrapper_utils.as_field(
            ffi, xp, dwdy, ts.ScalarKind.FLOAT64, {Cell: dwdy_size_0, K: dwdy_size_1}, True
        )

        assert isinstance(linit, int)
        linit = linit != 0

        diffusion_run(w, vn, exner, theta_v, rho, hdef_ic, div_ic, dwdx, dwdy, dtime, linit)

        # debug info

        msg = "shape of w after computation = %s" % str(w.shape if w is not None else "None")
        logging.debug(msg)
        msg = "w after computation: %s" % str(w.ndarray if w is not None else "None")
        logging.debug(msg)

        msg = "shape of vn after computation = %s" % str(vn.shape if vn is not None else "None")
        logging.debug(msg)
        msg = "vn after computation: %s" % str(vn.ndarray if vn is not None else "None")
        logging.debug(msg)

        msg = "shape of exner after computation = %s" % str(
            exner.shape if exner is not None else "None"
        )
        logging.debug(msg)
        msg = "exner after computation: %s" % str(exner.ndarray if exner is not None else "None")
        logging.debug(msg)

        msg = "shape of theta_v after computation = %s" % str(
            theta_v.shape if theta_v is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_v after computation: %s" % str(
            theta_v.ndarray if theta_v is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rho after computation = %s" % str(rho.shape if rho is not None else "None")
        logging.debug(msg)
        msg = "rho after computation: %s" % str(rho.ndarray if rho is not None else "None")
        logging.debug(msg)

        msg = "shape of hdef_ic after computation = %s" % str(
            hdef_ic.shape if hdef_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "hdef_ic after computation: %s" % str(
            hdef_ic.ndarray if hdef_ic is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of div_ic after computation = %s" % str(
            div_ic.shape if div_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "div_ic after computation: %s" % str(div_ic.ndarray if div_ic is not None else "None")
        logging.debug(msg)

        msg = "shape of dwdx after computation = %s" % str(
            dwdx.shape if dwdx is not None else "None"
        )
        logging.debug(msg)
        msg = "dwdx after computation: %s" % str(dwdx.ndarray if dwdx is not None else "None")
        logging.debug(msg)

        msg = "shape of dwdy after computation = %s" % str(
            dwdy.shape if dwdy is not None else "None"
        )
        logging.debug(msg)
        msg = "dwdy after computation: %s" % str(dwdy.ndarray if dwdy is not None else "None")
        logging.debug(msg)

        logging.critical("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def diffusion_init_wrapper(
    vct_a,
    vct_a_size_0,
    vct_b,
    vct_b_size_0,
    theta_ref_mc,
    theta_ref_mc_size_0,
    theta_ref_mc_size_1,
    wgtfac_c,
    wgtfac_c_size_0,
    wgtfac_c_size_1,
    e_bln_c_s,
    e_bln_c_s_size_0,
    e_bln_c_s_size_1,
    geofac_div,
    geofac_div_size_0,
    geofac_div_size_1,
    geofac_grg_x,
    geofac_grg_x_size_0,
    geofac_grg_x_size_1,
    geofac_grg_y,
    geofac_grg_y_size_0,
    geofac_grg_y_size_1,
    geofac_n2s,
    geofac_n2s_size_0,
    geofac_n2s_size_1,
    nudgecoeff_e,
    nudgecoeff_e_size_0,
    rbf_coeff_1,
    rbf_coeff_1_size_0,
    rbf_coeff_1_size_1,
    rbf_coeff_2,
    rbf_coeff_2_size_0,
    rbf_coeff_2_size_1,
    mask_hdiff,
    mask_hdiff_size_0,
    mask_hdiff_size_1,
    zd_diffcoef,
    zd_diffcoef_size_0,
    zd_diffcoef_size_1,
    zd_vertoffset,
    zd_vertoffset_size_0,
    zd_vertoffset_size_1,
    zd_vertoffset_size_2,
    zd_intcoef,
    zd_intcoef_size_0,
    zd_intcoef_size_1,
    zd_intcoef_size_2,
    ndyn_substeps,
    rayleigh_damping_height,
    nflat_gradp,
    diffusion_type,
    hdiff_w,
    hdiff_vn,
    zdiffu_t,
    type_t_diffu,
    type_vn_diffu,
    hdiff_efdt_ratio,
    smagorinski_scaling_factor,
    hdiff_temp,
    thslp_zdiffu,
    thhgtd_zdiffu,
    denom_diffu_v,
    nudge_max_coeff,
    itype_sher,
    ltkeshs,
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
    global_root,
    global_level,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
):
    try:
        logging.info("Python Execution Context Start")

        # Convert ptr to GT4Py fields

        vct_a = wrapper_utils.as_field(
            ffi, xp, vct_a, ts.ScalarKind.FLOAT64, {K: vct_a_size_0}, False
        )

        vct_b = wrapper_utils.as_field(
            ffi, xp, vct_b, ts.ScalarKind.FLOAT64, {K: vct_b_size_0}, False
        )

        theta_ref_mc = wrapper_utils.as_field(
            ffi,
            xp,
            theta_ref_mc,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_ref_mc_size_0, K: theta_ref_mc_size_1},
            False,
        )

        wgtfac_c = wrapper_utils.as_field(
            ffi,
            xp,
            wgtfac_c,
            ts.ScalarKind.FLOAT64,
            {Cell: wgtfac_c_size_0, K: wgtfac_c_size_1},
            False,
        )

        e_bln_c_s = wrapper_utils.as_field(
            ffi,
            xp,
            e_bln_c_s,
            ts.ScalarKind.FLOAT64,
            {Cell: e_bln_c_s_size_0, C2E: e_bln_c_s_size_1},
            False,
        )

        geofac_div = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_div,
            ts.ScalarKind.FLOAT64,
            {Cell: geofac_div_size_0, C2E: geofac_div_size_1},
            False,
        )

        geofac_grg_x = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_grg_x,
            ts.ScalarKind.FLOAT64,
            {Cell: geofac_grg_x_size_0, C2E2CO: geofac_grg_x_size_1},
            False,
        )

        geofac_grg_y = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_grg_y,
            ts.ScalarKind.FLOAT64,
            {Cell: geofac_grg_y_size_0, C2E2CO: geofac_grg_y_size_1},
            False,
        )

        geofac_n2s = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_n2s,
            ts.ScalarKind.FLOAT64,
            {Cell: geofac_n2s_size_0, C2E2CO: geofac_n2s_size_1},
            False,
        )

        nudgecoeff_e = wrapper_utils.as_field(
            ffi, xp, nudgecoeff_e, ts.ScalarKind.FLOAT64, {Edge: nudgecoeff_e_size_0}, False
        )

        rbf_coeff_1 = wrapper_utils.as_field(
            ffi,
            xp,
            rbf_coeff_1,
            ts.ScalarKind.FLOAT64,
            {Vertex: rbf_coeff_1_size_0, V2E: rbf_coeff_1_size_1},
            False,
        )

        rbf_coeff_2 = wrapper_utils.as_field(
            ffi,
            xp,
            rbf_coeff_2,
            ts.ScalarKind.FLOAT64,
            {Vertex: rbf_coeff_2_size_0, V2E: rbf_coeff_2_size_1},
            False,
        )

        mask_hdiff = wrapper_utils.as_field(
            ffi,
            xp,
            mask_hdiff,
            ts.ScalarKind.BOOL,
            {Cell: mask_hdiff_size_0, K: mask_hdiff_size_1},
            True,
        )

        zd_diffcoef = wrapper_utils.as_field(
            ffi,
            xp,
            zd_diffcoef,
            ts.ScalarKind.FLOAT64,
            {Cell: zd_diffcoef_size_0, K: zd_diffcoef_size_1},
            True,
        )

        zd_vertoffset = wrapper_utils.as_field(
            ffi,
            xp,
            zd_vertoffset,
            ts.ScalarKind.INT32,
            {Cell: zd_vertoffset_size_0, C2E2C: zd_vertoffset_size_1, K: zd_vertoffset_size_2},
            True,
        )

        zd_intcoef = wrapper_utils.as_field(
            ffi,
            xp,
            zd_intcoef,
            ts.ScalarKind.FLOAT64,
            {Cell: zd_intcoef_size_0, C2E2C: zd_intcoef_size_1, K: zd_intcoef_size_2},
            True,
        )

        assert isinstance(hdiff_w, int)
        hdiff_w = hdiff_w != 0

        assert isinstance(hdiff_vn, int)
        hdiff_vn = hdiff_vn != 0

        assert isinstance(zdiffu_t, int)
        zdiffu_t = zdiffu_t != 0

        assert isinstance(hdiff_temp, int)
        hdiff_temp = hdiff_temp != 0

        assert isinstance(ltkeshs, int)
        ltkeshs = ltkeshs != 0

        tangent_orientation = wrapper_utils.as_field(
            ffi,
            xp,
            tangent_orientation,
            ts.ScalarKind.FLOAT64,
            {Edge: tangent_orientation_size_0},
            False,
        )

        inverse_primal_edge_lengths = wrapper_utils.as_field(
            ffi,
            xp,
            inverse_primal_edge_lengths,
            ts.ScalarKind.FLOAT64,
            {Edge: inverse_primal_edge_lengths_size_0},
            False,
        )

        inv_dual_edge_length = wrapper_utils.as_field(
            ffi,
            xp,
            inv_dual_edge_length,
            ts.ScalarKind.FLOAT64,
            {Edge: inv_dual_edge_length_size_0},
            False,
        )

        inv_vert_vert_length = wrapper_utils.as_field(
            ffi,
            xp,
            inv_vert_vert_length,
            ts.ScalarKind.FLOAT64,
            {Edge: inv_vert_vert_length_size_0},
            False,
        )

        edge_areas = wrapper_utils.as_field(
            ffi, xp, edge_areas, ts.ScalarKind.FLOAT64, {Edge: edge_areas_size_0}, False
        )

        f_e = wrapper_utils.as_field(ffi, xp, f_e, ts.ScalarKind.FLOAT64, {Edge: f_e_size_0}, False)

        cell_center_lat = wrapper_utils.as_field(
            ffi, xp, cell_center_lat, ts.ScalarKind.FLOAT64, {Cell: cell_center_lat_size_0}, False
        )

        cell_center_lon = wrapper_utils.as_field(
            ffi, xp, cell_center_lon, ts.ScalarKind.FLOAT64, {Cell: cell_center_lon_size_0}, False
        )

        cell_areas = wrapper_utils.as_field(
            ffi, xp, cell_areas, ts.ScalarKind.FLOAT64, {Cell: cell_areas_size_0}, False
        )

        primal_normal_vert_x = wrapper_utils.as_field(
            ffi,
            xp,
            primal_normal_vert_x,
            ts.ScalarKind.FLOAT64,
            {Edge: primal_normal_vert_x_size_0, E2C2V: primal_normal_vert_x_size_1},
            False,
        )

        primal_normal_vert_y = wrapper_utils.as_field(
            ffi,
            xp,
            primal_normal_vert_y,
            ts.ScalarKind.FLOAT64,
            {Edge: primal_normal_vert_y_size_0, E2C2V: primal_normal_vert_y_size_1},
            False,
        )

        dual_normal_vert_x = wrapper_utils.as_field(
            ffi,
            xp,
            dual_normal_vert_x,
            ts.ScalarKind.FLOAT64,
            {Edge: dual_normal_vert_x_size_0, E2C2V: dual_normal_vert_x_size_1},
            False,
        )

        dual_normal_vert_y = wrapper_utils.as_field(
            ffi,
            xp,
            dual_normal_vert_y,
            ts.ScalarKind.FLOAT64,
            {Edge: dual_normal_vert_y_size_0, E2C2V: dual_normal_vert_y_size_1},
            False,
        )

        primal_normal_cell_x = wrapper_utils.as_field(
            ffi,
            xp,
            primal_normal_cell_x,
            ts.ScalarKind.FLOAT64,
            {Edge: primal_normal_cell_x_size_0, E2C: primal_normal_cell_x_size_1},
            False,
        )

        primal_normal_cell_y = wrapper_utils.as_field(
            ffi,
            xp,
            primal_normal_cell_y,
            ts.ScalarKind.FLOAT64,
            {Edge: primal_normal_cell_y_size_0, E2C: primal_normal_cell_y_size_1},
            False,
        )

        dual_normal_cell_x = wrapper_utils.as_field(
            ffi,
            xp,
            dual_normal_cell_x,
            ts.ScalarKind.FLOAT64,
            {Edge: dual_normal_cell_x_size_0, E2C: dual_normal_cell_x_size_1},
            False,
        )

        dual_normal_cell_y = wrapper_utils.as_field(
            ffi,
            xp,
            dual_normal_cell_y,
            ts.ScalarKind.FLOAT64,
            {Edge: dual_normal_cell_y_size_0, E2C: dual_normal_cell_y_size_1},
            False,
        )

        edge_center_lat = wrapper_utils.as_field(
            ffi, xp, edge_center_lat, ts.ScalarKind.FLOAT64, {Edge: edge_center_lat_size_0}, False
        )

        edge_center_lon = wrapper_utils.as_field(
            ffi, xp, edge_center_lon, ts.ScalarKind.FLOAT64, {Edge: edge_center_lon_size_0}, False
        )

        primal_normal_x = wrapper_utils.as_field(
            ffi, xp, primal_normal_x, ts.ScalarKind.FLOAT64, {Edge: primal_normal_x_size_0}, False
        )

        primal_normal_y = wrapper_utils.as_field(
            ffi, xp, primal_normal_y, ts.ScalarKind.FLOAT64, {Edge: primal_normal_y_size_0}, False
        )

        diffusion_init(
            vct_a,
            vct_b,
            theta_ref_mc,
            wgtfac_c,
            e_bln_c_s,
            geofac_div,
            geofac_grg_x,
            geofac_grg_y,
            geofac_n2s,
            nudgecoeff_e,
            rbf_coeff_1,
            rbf_coeff_2,
            mask_hdiff,
            zd_diffcoef,
            zd_vertoffset,
            zd_intcoef,
            ndyn_substeps,
            rayleigh_damping_height,
            nflat_gradp,
            diffusion_type,
            hdiff_w,
            hdiff_vn,
            zdiffu_t,
            type_t_diffu,
            type_vn_diffu,
            hdiff_efdt_ratio,
            smagorinski_scaling_factor,
            hdiff_temp,
            thslp_zdiffu,
            thhgtd_zdiffu,
            denom_diffu_v,
            nudge_max_coeff,
            itype_sher,
            ltkeshs,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inv_dual_edge_length,
            inv_vert_vert_length,
            edge_areas,
            f_e,
            cell_center_lat,
            cell_center_lon,
            cell_areas,
            primal_normal_vert_x,
            primal_normal_vert_y,
            dual_normal_vert_x,
            dual_normal_vert_y,
            primal_normal_cell_x,
            primal_normal_cell_y,
            dual_normal_cell_x,
            dual_normal_cell_y,
            edge_center_lat,
            edge_center_lon,
            primal_normal_x,
            primal_normal_y,
            global_root,
            global_level,
            lowest_layer_thickness,
            model_top_height,
            stretch_factor,
        )

        # debug info

        msg = "shape of vct_a after computation = %s" % str(
            vct_a.shape if vct_a is not None else "None"
        )
        logging.debug(msg)
        msg = "vct_a after computation: %s" % str(vct_a.ndarray if vct_a is not None else "None")
        logging.debug(msg)

        msg = "shape of vct_b after computation = %s" % str(
            vct_b.shape if vct_b is not None else "None"
        )
        logging.debug(msg)
        msg = "vct_b after computation: %s" % str(vct_b.ndarray if vct_b is not None else "None")
        logging.debug(msg)

        msg = "shape of theta_ref_mc after computation = %s" % str(
            theta_ref_mc.shape if theta_ref_mc is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_ref_mc after computation: %s" % str(
            theta_ref_mc.ndarray if theta_ref_mc is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of wgtfac_c after computation = %s" % str(
            wgtfac_c.shape if wgtfac_c is not None else "None"
        )
        logging.debug(msg)
        msg = "wgtfac_c after computation: %s" % str(
            wgtfac_c.ndarray if wgtfac_c is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of e_bln_c_s after computation = %s" % str(
            e_bln_c_s.shape if e_bln_c_s is not None else "None"
        )
        logging.debug(msg)
        msg = "e_bln_c_s after computation: %s" % str(
            e_bln_c_s.ndarray if e_bln_c_s is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of geofac_div after computation = %s" % str(
            geofac_div.shape if geofac_div is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_div after computation: %s" % str(
            geofac_div.ndarray if geofac_div is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of geofac_grg_x after computation = %s" % str(
            geofac_grg_x.shape if geofac_grg_x is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_grg_x after computation: %s" % str(
            geofac_grg_x.ndarray if geofac_grg_x is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of geofac_grg_y after computation = %s" % str(
            geofac_grg_y.shape if geofac_grg_y is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_grg_y after computation: %s" % str(
            geofac_grg_y.ndarray if geofac_grg_y is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of geofac_n2s after computation = %s" % str(
            geofac_n2s.shape if geofac_n2s is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_n2s after computation: %s" % str(
            geofac_n2s.ndarray if geofac_n2s is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of nudgecoeff_e after computation = %s" % str(
            nudgecoeff_e.shape if nudgecoeff_e is not None else "None"
        )
        logging.debug(msg)
        msg = "nudgecoeff_e after computation: %s" % str(
            nudgecoeff_e.ndarray if nudgecoeff_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rbf_coeff_1 after computation = %s" % str(
            rbf_coeff_1.shape if rbf_coeff_1 is not None else "None"
        )
        logging.debug(msg)
        msg = "rbf_coeff_1 after computation: %s" % str(
            rbf_coeff_1.ndarray if rbf_coeff_1 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rbf_coeff_2 after computation = %s" % str(
            rbf_coeff_2.shape if rbf_coeff_2 is not None else "None"
        )
        logging.debug(msg)
        msg = "rbf_coeff_2 after computation: %s" % str(
            rbf_coeff_2.ndarray if rbf_coeff_2 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of mask_hdiff after computation = %s" % str(
            mask_hdiff.shape if mask_hdiff is not None else "None"
        )
        logging.debug(msg)
        msg = "mask_hdiff after computation: %s" % str(
            mask_hdiff.ndarray if mask_hdiff is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of zd_diffcoef after computation = %s" % str(
            zd_diffcoef.shape if zd_diffcoef is not None else "None"
        )
        logging.debug(msg)
        msg = "zd_diffcoef after computation: %s" % str(
            zd_diffcoef.ndarray if zd_diffcoef is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of zd_vertoffset after computation = %s" % str(
            zd_vertoffset.shape if zd_vertoffset is not None else "None"
        )
        logging.debug(msg)
        msg = "zd_vertoffset after computation: %s" % str(
            zd_vertoffset.ndarray if zd_vertoffset is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of zd_intcoef after computation = %s" % str(
            zd_intcoef.shape if zd_intcoef is not None else "None"
        )
        logging.debug(msg)
        msg = "zd_intcoef after computation: %s" % str(
            zd_intcoef.ndarray if zd_intcoef is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of tangent_orientation after computation = %s" % str(
            tangent_orientation.shape if tangent_orientation is not None else "None"
        )
        logging.debug(msg)
        msg = "tangent_orientation after computation: %s" % str(
            tangent_orientation.ndarray if tangent_orientation is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of inverse_primal_edge_lengths after computation = %s" % str(
            inverse_primal_edge_lengths.shape if inverse_primal_edge_lengths is not None else "None"
        )
        logging.debug(msg)
        msg = "inverse_primal_edge_lengths after computation: %s" % str(
            inverse_primal_edge_lengths.ndarray
            if inverse_primal_edge_lengths is not None
            else "None"
        )
        logging.debug(msg)

        msg = "shape of inv_dual_edge_length after computation = %s" % str(
            inv_dual_edge_length.shape if inv_dual_edge_length is not None else "None"
        )
        logging.debug(msg)
        msg = "inv_dual_edge_length after computation: %s" % str(
            inv_dual_edge_length.ndarray if inv_dual_edge_length is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of inv_vert_vert_length after computation = %s" % str(
            inv_vert_vert_length.shape if inv_vert_vert_length is not None else "None"
        )
        logging.debug(msg)
        msg = "inv_vert_vert_length after computation: %s" % str(
            inv_vert_vert_length.ndarray if inv_vert_vert_length is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of edge_areas after computation = %s" % str(
            edge_areas.shape if edge_areas is not None else "None"
        )
        logging.debug(msg)
        msg = "edge_areas after computation: %s" % str(
            edge_areas.ndarray if edge_areas is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of f_e after computation = %s" % str(f_e.shape if f_e is not None else "None")
        logging.debug(msg)
        msg = "f_e after computation: %s" % str(f_e.ndarray if f_e is not None else "None")
        logging.debug(msg)

        msg = "shape of cell_center_lat after computation = %s" % str(
            cell_center_lat.shape if cell_center_lat is not None else "None"
        )
        logging.debug(msg)
        msg = "cell_center_lat after computation: %s" % str(
            cell_center_lat.ndarray if cell_center_lat is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of cell_center_lon after computation = %s" % str(
            cell_center_lon.shape if cell_center_lon is not None else "None"
        )
        logging.debug(msg)
        msg = "cell_center_lon after computation: %s" % str(
            cell_center_lon.ndarray if cell_center_lon is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of cell_areas after computation = %s" % str(
            cell_areas.shape if cell_areas is not None else "None"
        )
        logging.debug(msg)
        msg = "cell_areas after computation: %s" % str(
            cell_areas.ndarray if cell_areas is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of primal_normal_vert_x after computation = %s" % str(
            primal_normal_vert_x.shape if primal_normal_vert_x is not None else "None"
        )
        logging.debug(msg)
        msg = "primal_normal_vert_x after computation: %s" % str(
            primal_normal_vert_x.ndarray if primal_normal_vert_x is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of primal_normal_vert_y after computation = %s" % str(
            primal_normal_vert_y.shape if primal_normal_vert_y is not None else "None"
        )
        logging.debug(msg)
        msg = "primal_normal_vert_y after computation: %s" % str(
            primal_normal_vert_y.ndarray if primal_normal_vert_y is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of dual_normal_vert_x after computation = %s" % str(
            dual_normal_vert_x.shape if dual_normal_vert_x is not None else "None"
        )
        logging.debug(msg)
        msg = "dual_normal_vert_x after computation: %s" % str(
            dual_normal_vert_x.ndarray if dual_normal_vert_x is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of dual_normal_vert_y after computation = %s" % str(
            dual_normal_vert_y.shape if dual_normal_vert_y is not None else "None"
        )
        logging.debug(msg)
        msg = "dual_normal_vert_y after computation: %s" % str(
            dual_normal_vert_y.ndarray if dual_normal_vert_y is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of primal_normal_cell_x after computation = %s" % str(
            primal_normal_cell_x.shape if primal_normal_cell_x is not None else "None"
        )
        logging.debug(msg)
        msg = "primal_normal_cell_x after computation: %s" % str(
            primal_normal_cell_x.ndarray if primal_normal_cell_x is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of primal_normal_cell_y after computation = %s" % str(
            primal_normal_cell_y.shape if primal_normal_cell_y is not None else "None"
        )
        logging.debug(msg)
        msg = "primal_normal_cell_y after computation: %s" % str(
            primal_normal_cell_y.ndarray if primal_normal_cell_y is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of dual_normal_cell_x after computation = %s" % str(
            dual_normal_cell_x.shape if dual_normal_cell_x is not None else "None"
        )
        logging.debug(msg)
        msg = "dual_normal_cell_x after computation: %s" % str(
            dual_normal_cell_x.ndarray if dual_normal_cell_x is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of dual_normal_cell_y after computation = %s" % str(
            dual_normal_cell_y.shape if dual_normal_cell_y is not None else "None"
        )
        logging.debug(msg)
        msg = "dual_normal_cell_y after computation: %s" % str(
            dual_normal_cell_y.ndarray if dual_normal_cell_y is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of edge_center_lat after computation = %s" % str(
            edge_center_lat.shape if edge_center_lat is not None else "None"
        )
        logging.debug(msg)
        msg = "edge_center_lat after computation: %s" % str(
            edge_center_lat.ndarray if edge_center_lat is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of edge_center_lon after computation = %s" % str(
            edge_center_lon.shape if edge_center_lon is not None else "None"
        )
        logging.debug(msg)
        msg = "edge_center_lon after computation: %s" % str(
            edge_center_lon.ndarray if edge_center_lon is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of primal_normal_x after computation = %s" % str(
            primal_normal_x.shape if primal_normal_x is not None else "None"
        )
        logging.debug(msg)
        msg = "primal_normal_x after computation: %s" % str(
            primal_normal_x.ndarray if primal_normal_x is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of primal_normal_y after computation = %s" % str(
            primal_normal_y.shape if primal_normal_y is not None else "None"
        )
        logging.debug(msg)
        msg = "primal_normal_y after computation: %s" % str(
            primal_normal_y.ndarray if primal_normal_y is not None else "None"
        )
        logging.debug(msg)

        logging.critical("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def grid_init_diffusion_wrapper(
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

        grid_init_diffusion(
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
