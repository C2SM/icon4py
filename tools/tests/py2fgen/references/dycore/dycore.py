# imports for generated wrapper code
import logging

from dycore import ffi
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
from icon4py.tools.py2fgen.wrappers.dycore_wrapper import solve_nh_run
from icon4py.tools.py2fgen.wrappers.dycore_wrapper import solve_nh_init
from icon4py.tools.py2fgen.wrappers.dycore_wrapper import grid_init


C2E = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E2C = gtx.Dimension("C2E2C", kind=gtx.DimensionKind.LOCAL)
C2E2CO = gtx.Dimension("C2E2CO", kind=gtx.DimensionKind.LOCAL)
C2V = gtx.Dimension("C2V", kind=gtx.DimensionKind.LOCAL)
Cell = gtx.Dimension("Cell", kind=gtx.DimensionKind.HORIZONTAL)
CellIndex = gtx.Dimension("CellIndex", kind=gtx.DimensionKind.HORIZONTAL)
E2C = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
E2C2E = gtx.Dimension("E2C2E", kind=gtx.DimensionKind.LOCAL)
E2C2EO = gtx.Dimension("E2C2EO", kind=gtx.DimensionKind.LOCAL)
E2C2V = gtx.Dimension("E2C2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
Edge = gtx.Dimension("Edge", kind=gtx.DimensionKind.HORIZONTAL)
EdgeIndex = gtx.Dimension("EdgeIndex", kind=gtx.DimensionKind.HORIZONTAL)
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
V2C = gtx.Dimension("V2C", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
Vertex = gtx.Dimension("Vertex", kind=gtx.DimensionKind.HORIZONTAL)
VertexIndex = gtx.Dimension("VertexIndex", kind=gtx.DimensionKind.HORIZONTAL)


@ffi.def_extern()
def solve_nh_run_wrapper(
    rho_now,
    rho_now_size_0,
    rho_now_size_1,
    rho_new,
    rho_new_size_0,
    rho_new_size_1,
    exner_now,
    exner_now_size_0,
    exner_now_size_1,
    exner_new,
    exner_new_size_0,
    exner_new_size_1,
    w_now,
    w_now_size_0,
    w_now_size_1,
    w_new,
    w_new_size_0,
    w_new_size_1,
    theta_v_now,
    theta_v_now_size_0,
    theta_v_now_size_1,
    theta_v_new,
    theta_v_new_size_0,
    theta_v_new_size_1,
    vn_now,
    vn_now_size_0,
    vn_now_size_1,
    vn_new,
    vn_new_size_0,
    vn_new_size_1,
    w_concorr_c,
    w_concorr_c_size_0,
    w_concorr_c_size_1,
    ddt_vn_apc_ntl1,
    ddt_vn_apc_ntl1_size_0,
    ddt_vn_apc_ntl1_size_1,
    ddt_vn_apc_ntl2,
    ddt_vn_apc_ntl2_size_0,
    ddt_vn_apc_ntl2_size_1,
    ddt_w_adv_ntl1,
    ddt_w_adv_ntl1_size_0,
    ddt_w_adv_ntl1_size_1,
    ddt_w_adv_ntl2,
    ddt_w_adv_ntl2_size_0,
    ddt_w_adv_ntl2_size_1,
    theta_v_ic,
    theta_v_ic_size_0,
    theta_v_ic_size_1,
    rho_ic,
    rho_ic_size_0,
    rho_ic_size_1,
    exner_pr,
    exner_pr_size_0,
    exner_pr_size_1,
    exner_dyn_incr,
    exner_dyn_incr_size_0,
    exner_dyn_incr_size_1,
    ddt_exner_phy,
    ddt_exner_phy_size_0,
    ddt_exner_phy_size_1,
    grf_tend_rho,
    grf_tend_rho_size_0,
    grf_tend_rho_size_1,
    grf_tend_thv,
    grf_tend_thv_size_0,
    grf_tend_thv_size_1,
    grf_tend_w,
    grf_tend_w_size_0,
    grf_tend_w_size_1,
    mass_fl_e,
    mass_fl_e_size_0,
    mass_fl_e_size_1,
    ddt_vn_phy,
    ddt_vn_phy_size_0,
    ddt_vn_phy_size_1,
    grf_tend_vn,
    grf_tend_vn_size_0,
    grf_tend_vn_size_1,
    vn_ie,
    vn_ie_size_0,
    vn_ie_size_1,
    vt,
    vt_size_0,
    vt_size_1,
    mass_flx_me,
    mass_flx_me_size_0,
    mass_flx_me_size_1,
    mass_flx_ic,
    mass_flx_ic_size_0,
    mass_flx_ic_size_1,
    vn_traj,
    vn_traj_size_0,
    vn_traj_size_1,
    dtime,
    lprep_adv,
    at_initial_timestep,
    divdamp_fac_o2,
    ndyn_substeps,
    idyn_timestep,
):
    try:
        logging.info("Python Execution Context Start")

        # Convert ptr to GT4Py fields

        rho_now = wrapper_utils.as_field(
            ffi,
            xp,
            rho_now,
            ts.ScalarKind.FLOAT64,
            {Cell: rho_now_size_0, K: rho_now_size_1},
            False,
        )

        rho_new = wrapper_utils.as_field(
            ffi,
            xp,
            rho_new,
            ts.ScalarKind.FLOAT64,
            {Cell: rho_new_size_0, K: rho_new_size_1},
            False,
        )

        exner_now = wrapper_utils.as_field(
            ffi,
            xp,
            exner_now,
            ts.ScalarKind.FLOAT64,
            {Cell: exner_now_size_0, K: exner_now_size_1},
            False,
        )

        exner_new = wrapper_utils.as_field(
            ffi,
            xp,
            exner_new,
            ts.ScalarKind.FLOAT64,
            {Cell: exner_new_size_0, K: exner_new_size_1},
            False,
        )

        w_now = wrapper_utils.as_field(
            ffi, xp, w_now, ts.ScalarKind.FLOAT64, {Cell: w_now_size_0, K: w_now_size_1}, False
        )

        w_new = wrapper_utils.as_field(
            ffi, xp, w_new, ts.ScalarKind.FLOAT64, {Cell: w_new_size_0, K: w_new_size_1}, False
        )

        theta_v_now = wrapper_utils.as_field(
            ffi,
            xp,
            theta_v_now,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_v_now_size_0, K: theta_v_now_size_1},
            False,
        )

        theta_v_new = wrapper_utils.as_field(
            ffi,
            xp,
            theta_v_new,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_v_new_size_0, K: theta_v_new_size_1},
            False,
        )

        vn_now = wrapper_utils.as_field(
            ffi, xp, vn_now, ts.ScalarKind.FLOAT64, {Edge: vn_now_size_0, K: vn_now_size_1}, False
        )

        vn_new = wrapper_utils.as_field(
            ffi, xp, vn_new, ts.ScalarKind.FLOAT64, {Edge: vn_new_size_0, K: vn_new_size_1}, False
        )

        w_concorr_c = wrapper_utils.as_field(
            ffi,
            xp,
            w_concorr_c,
            ts.ScalarKind.FLOAT64,
            {Cell: w_concorr_c_size_0, K: w_concorr_c_size_1},
            False,
        )

        ddt_vn_apc_ntl1 = wrapper_utils.as_field(
            ffi,
            xp,
            ddt_vn_apc_ntl1,
            ts.ScalarKind.FLOAT64,
            {Edge: ddt_vn_apc_ntl1_size_0, K: ddt_vn_apc_ntl1_size_1},
            False,
        )

        ddt_vn_apc_ntl2 = wrapper_utils.as_field(
            ffi,
            xp,
            ddt_vn_apc_ntl2,
            ts.ScalarKind.FLOAT64,
            {Edge: ddt_vn_apc_ntl2_size_0, K: ddt_vn_apc_ntl2_size_1},
            False,
        )

        ddt_w_adv_ntl1 = wrapper_utils.as_field(
            ffi,
            xp,
            ddt_w_adv_ntl1,
            ts.ScalarKind.FLOAT64,
            {Cell: ddt_w_adv_ntl1_size_0, K: ddt_w_adv_ntl1_size_1},
            False,
        )

        ddt_w_adv_ntl2 = wrapper_utils.as_field(
            ffi,
            xp,
            ddt_w_adv_ntl2,
            ts.ScalarKind.FLOAT64,
            {Cell: ddt_w_adv_ntl2_size_0, K: ddt_w_adv_ntl2_size_1},
            False,
        )

        theta_v_ic = wrapper_utils.as_field(
            ffi,
            xp,
            theta_v_ic,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_v_ic_size_0, K: theta_v_ic_size_1},
            False,
        )

        rho_ic = wrapper_utils.as_field(
            ffi, xp, rho_ic, ts.ScalarKind.FLOAT64, {Cell: rho_ic_size_0, K: rho_ic_size_1}, False
        )

        exner_pr = wrapper_utils.as_field(
            ffi,
            xp,
            exner_pr,
            ts.ScalarKind.FLOAT64,
            {Cell: exner_pr_size_0, K: exner_pr_size_1},
            False,
        )

        exner_dyn_incr = wrapper_utils.as_field(
            ffi,
            xp,
            exner_dyn_incr,
            ts.ScalarKind.FLOAT64,
            {Cell: exner_dyn_incr_size_0, K: exner_dyn_incr_size_1},
            False,
        )

        ddt_exner_phy = wrapper_utils.as_field(
            ffi,
            xp,
            ddt_exner_phy,
            ts.ScalarKind.FLOAT64,
            {Cell: ddt_exner_phy_size_0, K: ddt_exner_phy_size_1},
            False,
        )

        grf_tend_rho = wrapper_utils.as_field(
            ffi,
            xp,
            grf_tend_rho,
            ts.ScalarKind.FLOAT64,
            {Cell: grf_tend_rho_size_0, K: grf_tend_rho_size_1},
            False,
        )

        grf_tend_thv = wrapper_utils.as_field(
            ffi,
            xp,
            grf_tend_thv,
            ts.ScalarKind.FLOAT64,
            {Cell: grf_tend_thv_size_0, K: grf_tend_thv_size_1},
            False,
        )

        grf_tend_w = wrapper_utils.as_field(
            ffi,
            xp,
            grf_tend_w,
            ts.ScalarKind.FLOAT64,
            {Cell: grf_tend_w_size_0, K: grf_tend_w_size_1},
            False,
        )

        mass_fl_e = wrapper_utils.as_field(
            ffi,
            xp,
            mass_fl_e,
            ts.ScalarKind.FLOAT64,
            {Edge: mass_fl_e_size_0, K: mass_fl_e_size_1},
            False,
        )

        ddt_vn_phy = wrapper_utils.as_field(
            ffi,
            xp,
            ddt_vn_phy,
            ts.ScalarKind.FLOAT64,
            {Edge: ddt_vn_phy_size_0, K: ddt_vn_phy_size_1},
            False,
        )

        grf_tend_vn = wrapper_utils.as_field(
            ffi,
            xp,
            grf_tend_vn,
            ts.ScalarKind.FLOAT64,
            {Edge: grf_tend_vn_size_0, K: grf_tend_vn_size_1},
            False,
        )

        vn_ie = wrapper_utils.as_field(
            ffi, xp, vn_ie, ts.ScalarKind.FLOAT64, {Edge: vn_ie_size_0, K: vn_ie_size_1}, False
        )

        vt = wrapper_utils.as_field(
            ffi, xp, vt, ts.ScalarKind.FLOAT64, {Edge: vt_size_0, K: vt_size_1}, False
        )

        mass_flx_me = wrapper_utils.as_field(
            ffi,
            xp,
            mass_flx_me,
            ts.ScalarKind.FLOAT64,
            {Edge: mass_flx_me_size_0, K: mass_flx_me_size_1},
            False,
        )

        mass_flx_ic = wrapper_utils.as_field(
            ffi,
            xp,
            mass_flx_ic,
            ts.ScalarKind.FLOAT64,
            {Cell: mass_flx_ic_size_0, K: mass_flx_ic_size_1},
            False,
        )

        vn_traj = wrapper_utils.as_field(
            ffi,
            xp,
            vn_traj,
            ts.ScalarKind.FLOAT64,
            {Edge: vn_traj_size_0, K: vn_traj_size_1},
            False,
        )

        assert isinstance(lprep_adv, int)
        lprep_adv = lprep_adv != 0

        assert isinstance(at_initial_timestep, int)
        at_initial_timestep = at_initial_timestep != 0

        solve_nh_run(
            rho_now,
            rho_new,
            exner_now,
            exner_new,
            w_now,
            w_new,
            theta_v_now,
            theta_v_new,
            vn_now,
            vn_new,
            w_concorr_c,
            ddt_vn_apc_ntl1,
            ddt_vn_apc_ntl2,
            ddt_w_adv_ntl1,
            ddt_w_adv_ntl2,
            theta_v_ic,
            rho_ic,
            exner_pr,
            exner_dyn_incr,
            ddt_exner_phy,
            grf_tend_rho,
            grf_tend_thv,
            grf_tend_w,
            mass_fl_e,
            ddt_vn_phy,
            grf_tend_vn,
            vn_ie,
            vt,
            mass_flx_me,
            mass_flx_ic,
            vn_traj,
            dtime,
            lprep_adv,
            at_initial_timestep,
            divdamp_fac_o2,
            ndyn_substeps,
            idyn_timestep,
        )

        # debug info

        msg = "shape of rho_now after computation = %s" % str(
            rho_now.shape if rho_now is not None else "None"
        )
        logging.debug(msg)
        msg = "rho_now after computation: %s" % str(
            rho_now.ndarray if rho_now is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rho_new after computation = %s" % str(
            rho_new.shape if rho_new is not None else "None"
        )
        logging.debug(msg)
        msg = "rho_new after computation: %s" % str(
            rho_new.ndarray if rho_new is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of exner_now after computation = %s" % str(
            exner_now.shape if exner_now is not None else "None"
        )
        logging.debug(msg)
        msg = "exner_now after computation: %s" % str(
            exner_now.ndarray if exner_now is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of exner_new after computation = %s" % str(
            exner_new.shape if exner_new is not None else "None"
        )
        logging.debug(msg)
        msg = "exner_new after computation: %s" % str(
            exner_new.ndarray if exner_new is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of w_now after computation = %s" % str(
            w_now.shape if w_now is not None else "None"
        )
        logging.debug(msg)
        msg = "w_now after computation: %s" % str(w_now.ndarray if w_now is not None else "None")
        logging.debug(msg)

        msg = "shape of w_new after computation = %s" % str(
            w_new.shape if w_new is not None else "None"
        )
        logging.debug(msg)
        msg = "w_new after computation: %s" % str(w_new.ndarray if w_new is not None else "None")
        logging.debug(msg)

        msg = "shape of theta_v_now after computation = %s" % str(
            theta_v_now.shape if theta_v_now is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_v_now after computation: %s" % str(
            theta_v_now.ndarray if theta_v_now is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of theta_v_new after computation = %s" % str(
            theta_v_new.shape if theta_v_new is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_v_new after computation: %s" % str(
            theta_v_new.ndarray if theta_v_new is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vn_now after computation = %s" % str(
            vn_now.shape if vn_now is not None else "None"
        )
        logging.debug(msg)
        msg = "vn_now after computation: %s" % str(vn_now.ndarray if vn_now is not None else "None")
        logging.debug(msg)

        msg = "shape of vn_new after computation = %s" % str(
            vn_new.shape if vn_new is not None else "None"
        )
        logging.debug(msg)
        msg = "vn_new after computation: %s" % str(vn_new.ndarray if vn_new is not None else "None")
        logging.debug(msg)

        msg = "shape of w_concorr_c after computation = %s" % str(
            w_concorr_c.shape if w_concorr_c is not None else "None"
        )
        logging.debug(msg)
        msg = "w_concorr_c after computation: %s" % str(
            w_concorr_c.ndarray if w_concorr_c is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddt_vn_apc_ntl1 after computation = %s" % str(
            ddt_vn_apc_ntl1.shape if ddt_vn_apc_ntl1 is not None else "None"
        )
        logging.debug(msg)
        msg = "ddt_vn_apc_ntl1 after computation: %s" % str(
            ddt_vn_apc_ntl1.ndarray if ddt_vn_apc_ntl1 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddt_vn_apc_ntl2 after computation = %s" % str(
            ddt_vn_apc_ntl2.shape if ddt_vn_apc_ntl2 is not None else "None"
        )
        logging.debug(msg)
        msg = "ddt_vn_apc_ntl2 after computation: %s" % str(
            ddt_vn_apc_ntl2.ndarray if ddt_vn_apc_ntl2 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddt_w_adv_ntl1 after computation = %s" % str(
            ddt_w_adv_ntl1.shape if ddt_w_adv_ntl1 is not None else "None"
        )
        logging.debug(msg)
        msg = "ddt_w_adv_ntl1 after computation: %s" % str(
            ddt_w_adv_ntl1.ndarray if ddt_w_adv_ntl1 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddt_w_adv_ntl2 after computation = %s" % str(
            ddt_w_adv_ntl2.shape if ddt_w_adv_ntl2 is not None else "None"
        )
        logging.debug(msg)
        msg = "ddt_w_adv_ntl2 after computation: %s" % str(
            ddt_w_adv_ntl2.ndarray if ddt_w_adv_ntl2 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of theta_v_ic after computation = %s" % str(
            theta_v_ic.shape if theta_v_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_v_ic after computation: %s" % str(
            theta_v_ic.ndarray if theta_v_ic is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rho_ic after computation = %s" % str(
            rho_ic.shape if rho_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "rho_ic after computation: %s" % str(rho_ic.ndarray if rho_ic is not None else "None")
        logging.debug(msg)

        msg = "shape of exner_pr after computation = %s" % str(
            exner_pr.shape if exner_pr is not None else "None"
        )
        logging.debug(msg)
        msg = "exner_pr after computation: %s" % str(
            exner_pr.ndarray if exner_pr is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of exner_dyn_incr after computation = %s" % str(
            exner_dyn_incr.shape if exner_dyn_incr is not None else "None"
        )
        logging.debug(msg)
        msg = "exner_dyn_incr after computation: %s" % str(
            exner_dyn_incr.ndarray if exner_dyn_incr is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddt_exner_phy after computation = %s" % str(
            ddt_exner_phy.shape if ddt_exner_phy is not None else "None"
        )
        logging.debug(msg)
        msg = "ddt_exner_phy after computation: %s" % str(
            ddt_exner_phy.ndarray if ddt_exner_phy is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of grf_tend_rho after computation = %s" % str(
            grf_tend_rho.shape if grf_tend_rho is not None else "None"
        )
        logging.debug(msg)
        msg = "grf_tend_rho after computation: %s" % str(
            grf_tend_rho.ndarray if grf_tend_rho is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of grf_tend_thv after computation = %s" % str(
            grf_tend_thv.shape if grf_tend_thv is not None else "None"
        )
        logging.debug(msg)
        msg = "grf_tend_thv after computation: %s" % str(
            grf_tend_thv.ndarray if grf_tend_thv is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of grf_tend_w after computation = %s" % str(
            grf_tend_w.shape if grf_tend_w is not None else "None"
        )
        logging.debug(msg)
        msg = "grf_tend_w after computation: %s" % str(
            grf_tend_w.ndarray if grf_tend_w is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of mass_fl_e after computation = %s" % str(
            mass_fl_e.shape if mass_fl_e is not None else "None"
        )
        logging.debug(msg)
        msg = "mass_fl_e after computation: %s" % str(
            mass_fl_e.ndarray if mass_fl_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddt_vn_phy after computation = %s" % str(
            ddt_vn_phy.shape if ddt_vn_phy is not None else "None"
        )
        logging.debug(msg)
        msg = "ddt_vn_phy after computation: %s" % str(
            ddt_vn_phy.ndarray if ddt_vn_phy is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of grf_tend_vn after computation = %s" % str(
            grf_tend_vn.shape if grf_tend_vn is not None else "None"
        )
        logging.debug(msg)
        msg = "grf_tend_vn after computation: %s" % str(
            grf_tend_vn.ndarray if grf_tend_vn is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vn_ie after computation = %s" % str(
            vn_ie.shape if vn_ie is not None else "None"
        )
        logging.debug(msg)
        msg = "vn_ie after computation: %s" % str(vn_ie.ndarray if vn_ie is not None else "None")
        logging.debug(msg)

        msg = "shape of vt after computation = %s" % str(vt.shape if vt is not None else "None")
        logging.debug(msg)
        msg = "vt after computation: %s" % str(vt.ndarray if vt is not None else "None")
        logging.debug(msg)

        msg = "shape of mass_flx_me after computation = %s" % str(
            mass_flx_me.shape if mass_flx_me is not None else "None"
        )
        logging.debug(msg)
        msg = "mass_flx_me after computation: %s" % str(
            mass_flx_me.ndarray if mass_flx_me is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of mass_flx_ic after computation = %s" % str(
            mass_flx_ic.shape if mass_flx_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "mass_flx_ic after computation: %s" % str(
            mass_flx_ic.ndarray if mass_flx_ic is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vn_traj after computation = %s" % str(
            vn_traj.shape if vn_traj is not None else "None"
        )
        logging.debug(msg)
        msg = "vn_traj after computation: %s" % str(
            vn_traj.ndarray if vn_traj is not None else "None"
        )
        logging.debug(msg)

        logging.critical("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def solve_nh_init_wrapper(
    vct_a,
    vct_a_size_0,
    vct_b,
    vct_b_size_0,
    cell_areas,
    cell_areas_size_0,
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
    edge_areas,
    edge_areas_size_0,
    tangent_orientation,
    tangent_orientation_size_0,
    inverse_primal_edge_lengths,
    inverse_primal_edge_lengths_size_0,
    inverse_dual_edge_lengths,
    inverse_dual_edge_lengths_size_0,
    inverse_vertex_vertex_lengths,
    inverse_vertex_vertex_lengths_size_0,
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
    f_e,
    f_e_size_0,
    c_lin_e,
    c_lin_e_size_0,
    c_lin_e_size_1,
    c_intp,
    c_intp_size_0,
    c_intp_size_1,
    e_flx_avg,
    e_flx_avg_size_0,
    e_flx_avg_size_1,
    geofac_grdiv,
    geofac_grdiv_size_0,
    geofac_grdiv_size_1,
    geofac_rot,
    geofac_rot_size_0,
    geofac_rot_size_1,
    pos_on_tplane_e_1,
    pos_on_tplane_e_1_size_0,
    pos_on_tplane_e_1_size_1,
    pos_on_tplane_e_2,
    pos_on_tplane_e_2_size_0,
    pos_on_tplane_e_2_size_1,
    rbf_vec_coeff_e,
    rbf_vec_coeff_e_size_0,
    rbf_vec_coeff_e_size_1,
    e_bln_c_s,
    e_bln_c_s_size_0,
    e_bln_c_s_size_1,
    rbf_coeff_1,
    rbf_coeff_1_size_0,
    rbf_coeff_1_size_1,
    rbf_coeff_2,
    rbf_coeff_2_size_0,
    rbf_coeff_2_size_1,
    geofac_div,
    geofac_div_size_0,
    geofac_div_size_1,
    geofac_n2s,
    geofac_n2s_size_0,
    geofac_n2s_size_1,
    geofac_grg_x,
    geofac_grg_x_size_0,
    geofac_grg_x_size_1,
    geofac_grg_y,
    geofac_grg_y_size_0,
    geofac_grg_y_size_1,
    nudgecoeff_e,
    nudgecoeff_e_size_0,
    bdy_halo_c,
    bdy_halo_c_size_0,
    mask_prog_halo_c,
    mask_prog_halo_c_size_0,
    rayleigh_w,
    rayleigh_w_size_0,
    exner_exfac,
    exner_exfac_size_0,
    exner_exfac_size_1,
    exner_ref_mc,
    exner_ref_mc_size_0,
    exner_ref_mc_size_1,
    wgtfac_c,
    wgtfac_c_size_0,
    wgtfac_c_size_1,
    wgtfacq_c,
    wgtfacq_c_size_0,
    wgtfacq_c_size_1,
    inv_ddqz_z_full,
    inv_ddqz_z_full_size_0,
    inv_ddqz_z_full_size_1,
    rho_ref_mc,
    rho_ref_mc_size_0,
    rho_ref_mc_size_1,
    theta_ref_mc,
    theta_ref_mc_size_0,
    theta_ref_mc_size_1,
    vwind_expl_wgt,
    vwind_expl_wgt_size_0,
    d_exner_dz_ref_ic,
    d_exner_dz_ref_ic_size_0,
    d_exner_dz_ref_ic_size_1,
    ddqz_z_half,
    ddqz_z_half_size_0,
    ddqz_z_half_size_1,
    theta_ref_ic,
    theta_ref_ic_size_0,
    theta_ref_ic_size_1,
    d2dexdz2_fac1_mc,
    d2dexdz2_fac1_mc_size_0,
    d2dexdz2_fac1_mc_size_1,
    d2dexdz2_fac2_mc,
    d2dexdz2_fac2_mc_size_0,
    d2dexdz2_fac2_mc_size_1,
    rho_ref_me,
    rho_ref_me_size_0,
    rho_ref_me_size_1,
    theta_ref_me,
    theta_ref_me_size_0,
    theta_ref_me_size_1,
    ddxn_z_full,
    ddxn_z_full_size_0,
    ddxn_z_full_size_1,
    zdiff_gradp,
    zdiff_gradp_size_0,
    zdiff_gradp_size_1,
    zdiff_gradp_size_2,
    vertoffset_gradp,
    vertoffset_gradp_size_0,
    vertoffset_gradp_size_1,
    vertoffset_gradp_size_2,
    ipeidx_dsl,
    ipeidx_dsl_size_0,
    ipeidx_dsl_size_1,
    pg_exdist,
    pg_exdist_size_0,
    pg_exdist_size_1,
    ddqz_z_full_e,
    ddqz_z_full_e_size_0,
    ddqz_z_full_e_size_1,
    ddxt_z_full,
    ddxt_z_full_size_0,
    ddxt_z_full_size_1,
    wgtfac_e,
    wgtfac_e_size_0,
    wgtfac_e_size_1,
    wgtfacq_e,
    wgtfacq_e_size_0,
    wgtfacq_e_size_1,
    vwind_impl_wgt,
    vwind_impl_wgt_size_0,
    hmask_dd3d,
    hmask_dd3d_size_0,
    scalfac_dd3d,
    scalfac_dd3d_size_0,
    coeff1_dwdz,
    coeff1_dwdz_size_0,
    coeff1_dwdz_size_1,
    coeff2_dwdz,
    coeff2_dwdz_size_0,
    coeff2_dwdz_size_1,
    coeff_gradekin,
    coeff_gradekin_size_0,
    coeff_gradekin_size_1,
    c_owner_mask,
    c_owner_mask_size_0,
    cell_center_lat,
    cell_center_lat_size_0,
    cell_center_lon,
    cell_center_lon_size_0,
    edge_center_lat,
    edge_center_lat_size_0,
    edge_center_lon,
    edge_center_lon_size_0,
    primal_normal_x,
    primal_normal_x_size_0,
    primal_normal_y,
    primal_normal_y_size_0,
    rayleigh_damping_height,
    itime_scheme,
    iadv_rhotheta,
    igradp_method,
    ndyn_substeps,
    rayleigh_type,
    rayleigh_coeff,
    divdamp_order,
    is_iau_active,
    iau_wgt_dyn,
    divdamp_type,
    divdamp_trans_start,
    divdamp_trans_end,
    l_vert_nested,
    rhotheta_offctr,
    veladv_offctr,
    max_nudging_coeff,
    divdamp_fac,
    divdamp_fac2,
    divdamp_fac3,
    divdamp_fac4,
    divdamp_z,
    divdamp_z2,
    divdamp_z3,
    divdamp_z4,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    nflat_gradp,
    num_levels,
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

        cell_areas = wrapper_utils.as_field(
            ffi, xp, cell_areas, ts.ScalarKind.FLOAT64, {Cell: cell_areas_size_0}, False
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

        edge_areas = wrapper_utils.as_field(
            ffi, xp, edge_areas, ts.ScalarKind.FLOAT64, {Edge: edge_areas_size_0}, False
        )

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

        inverse_dual_edge_lengths = wrapper_utils.as_field(
            ffi,
            xp,
            inverse_dual_edge_lengths,
            ts.ScalarKind.FLOAT64,
            {Edge: inverse_dual_edge_lengths_size_0},
            False,
        )

        inverse_vertex_vertex_lengths = wrapper_utils.as_field(
            ffi,
            xp,
            inverse_vertex_vertex_lengths,
            ts.ScalarKind.FLOAT64,
            {Edge: inverse_vertex_vertex_lengths_size_0},
            False,
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

        f_e = wrapper_utils.as_field(ffi, xp, f_e, ts.ScalarKind.FLOAT64, {Edge: f_e_size_0}, False)

        c_lin_e = wrapper_utils.as_field(
            ffi,
            xp,
            c_lin_e,
            ts.ScalarKind.FLOAT64,
            {Edge: c_lin_e_size_0, E2C: c_lin_e_size_1},
            False,
        )

        c_intp = wrapper_utils.as_field(
            ffi,
            xp,
            c_intp,
            ts.ScalarKind.FLOAT64,
            {Vertex: c_intp_size_0, V2C: c_intp_size_1},
            False,
        )

        e_flx_avg = wrapper_utils.as_field(
            ffi,
            xp,
            e_flx_avg,
            ts.ScalarKind.FLOAT64,
            {Edge: e_flx_avg_size_0, E2C2EO: e_flx_avg_size_1},
            False,
        )

        geofac_grdiv = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_grdiv,
            ts.ScalarKind.FLOAT64,
            {Edge: geofac_grdiv_size_0, E2C2EO: geofac_grdiv_size_1},
            False,
        )

        geofac_rot = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_rot,
            ts.ScalarKind.FLOAT64,
            {Vertex: geofac_rot_size_0, V2E: geofac_rot_size_1},
            False,
        )

        pos_on_tplane_e_1 = wrapper_utils.as_field(
            ffi,
            xp,
            pos_on_tplane_e_1,
            ts.ScalarKind.FLOAT64,
            {Edge: pos_on_tplane_e_1_size_0, E2C: pos_on_tplane_e_1_size_1},
            False,
        )

        pos_on_tplane_e_2 = wrapper_utils.as_field(
            ffi,
            xp,
            pos_on_tplane_e_2,
            ts.ScalarKind.FLOAT64,
            {Edge: pos_on_tplane_e_2_size_0, E2C: pos_on_tplane_e_2_size_1},
            False,
        )

        rbf_vec_coeff_e = wrapper_utils.as_field(
            ffi,
            xp,
            rbf_vec_coeff_e,
            ts.ScalarKind.FLOAT64,
            {Edge: rbf_vec_coeff_e_size_0, E2C2E: rbf_vec_coeff_e_size_1},
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

        geofac_div = wrapper_utils.as_field(
            ffi,
            xp,
            geofac_div,
            ts.ScalarKind.FLOAT64,
            {Cell: geofac_div_size_0, C2E: geofac_div_size_1},
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

        nudgecoeff_e = wrapper_utils.as_field(
            ffi, xp, nudgecoeff_e, ts.ScalarKind.FLOAT64, {Edge: nudgecoeff_e_size_0}, False
        )

        bdy_halo_c = wrapper_utils.as_field(
            ffi, xp, bdy_halo_c, ts.ScalarKind.BOOL, {Cell: bdy_halo_c_size_0}, False
        )

        mask_prog_halo_c = wrapper_utils.as_field(
            ffi, xp, mask_prog_halo_c, ts.ScalarKind.BOOL, {Cell: mask_prog_halo_c_size_0}, False
        )

        rayleigh_w = wrapper_utils.as_field(
            ffi, xp, rayleigh_w, ts.ScalarKind.FLOAT64, {K: rayleigh_w_size_0}, False
        )

        exner_exfac = wrapper_utils.as_field(
            ffi,
            xp,
            exner_exfac,
            ts.ScalarKind.FLOAT64,
            {Cell: exner_exfac_size_0, K: exner_exfac_size_1},
            False,
        )

        exner_ref_mc = wrapper_utils.as_field(
            ffi,
            xp,
            exner_ref_mc,
            ts.ScalarKind.FLOAT64,
            {Cell: exner_ref_mc_size_0, K: exner_ref_mc_size_1},
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

        wgtfacq_c = wrapper_utils.as_field(
            ffi,
            xp,
            wgtfacq_c,
            ts.ScalarKind.FLOAT64,
            {Cell: wgtfacq_c_size_0, K: wgtfacq_c_size_1},
            False,
        )

        inv_ddqz_z_full = wrapper_utils.as_field(
            ffi,
            xp,
            inv_ddqz_z_full,
            ts.ScalarKind.FLOAT64,
            {Cell: inv_ddqz_z_full_size_0, K: inv_ddqz_z_full_size_1},
            False,
        )

        rho_ref_mc = wrapper_utils.as_field(
            ffi,
            xp,
            rho_ref_mc,
            ts.ScalarKind.FLOAT64,
            {Cell: rho_ref_mc_size_0, K: rho_ref_mc_size_1},
            False,
        )

        theta_ref_mc = wrapper_utils.as_field(
            ffi,
            xp,
            theta_ref_mc,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_ref_mc_size_0, K: theta_ref_mc_size_1},
            False,
        )

        vwind_expl_wgt = wrapper_utils.as_field(
            ffi, xp, vwind_expl_wgt, ts.ScalarKind.FLOAT64, {Cell: vwind_expl_wgt_size_0}, False
        )

        d_exner_dz_ref_ic = wrapper_utils.as_field(
            ffi,
            xp,
            d_exner_dz_ref_ic,
            ts.ScalarKind.FLOAT64,
            {Cell: d_exner_dz_ref_ic_size_0, K: d_exner_dz_ref_ic_size_1},
            False,
        )

        ddqz_z_half = wrapper_utils.as_field(
            ffi,
            xp,
            ddqz_z_half,
            ts.ScalarKind.FLOAT64,
            {Cell: ddqz_z_half_size_0, K: ddqz_z_half_size_1},
            False,
        )

        theta_ref_ic = wrapper_utils.as_field(
            ffi,
            xp,
            theta_ref_ic,
            ts.ScalarKind.FLOAT64,
            {Cell: theta_ref_ic_size_0, K: theta_ref_ic_size_1},
            False,
        )

        d2dexdz2_fac1_mc = wrapper_utils.as_field(
            ffi,
            xp,
            d2dexdz2_fac1_mc,
            ts.ScalarKind.FLOAT64,
            {Cell: d2dexdz2_fac1_mc_size_0, K: d2dexdz2_fac1_mc_size_1},
            False,
        )

        d2dexdz2_fac2_mc = wrapper_utils.as_field(
            ffi,
            xp,
            d2dexdz2_fac2_mc,
            ts.ScalarKind.FLOAT64,
            {Cell: d2dexdz2_fac2_mc_size_0, K: d2dexdz2_fac2_mc_size_1},
            False,
        )

        rho_ref_me = wrapper_utils.as_field(
            ffi,
            xp,
            rho_ref_me,
            ts.ScalarKind.FLOAT64,
            {Edge: rho_ref_me_size_0, K: rho_ref_me_size_1},
            False,
        )

        theta_ref_me = wrapper_utils.as_field(
            ffi,
            xp,
            theta_ref_me,
            ts.ScalarKind.FLOAT64,
            {Edge: theta_ref_me_size_0, K: theta_ref_me_size_1},
            False,
        )

        ddxn_z_full = wrapper_utils.as_field(
            ffi,
            xp,
            ddxn_z_full,
            ts.ScalarKind.FLOAT64,
            {Edge: ddxn_z_full_size_0, K: ddxn_z_full_size_1},
            False,
        )

        zdiff_gradp = wrapper_utils.as_field(
            ffi,
            xp,
            zdiff_gradp,
            ts.ScalarKind.FLOAT64,
            {Edge: zdiff_gradp_size_0, E2C: zdiff_gradp_size_1, K: zdiff_gradp_size_2},
            False,
        )

        vertoffset_gradp = wrapper_utils.as_field(
            ffi,
            xp,
            vertoffset_gradp,
            ts.ScalarKind.INT32,
            {
                Edge: vertoffset_gradp_size_0,
                E2C: vertoffset_gradp_size_1,
                K: vertoffset_gradp_size_2,
            },
            False,
        )

        ipeidx_dsl = wrapper_utils.as_field(
            ffi,
            xp,
            ipeidx_dsl,
            ts.ScalarKind.BOOL,
            {Edge: ipeidx_dsl_size_0, K: ipeidx_dsl_size_1},
            False,
        )

        pg_exdist = wrapper_utils.as_field(
            ffi,
            xp,
            pg_exdist,
            ts.ScalarKind.FLOAT64,
            {Edge: pg_exdist_size_0, K: pg_exdist_size_1},
            False,
        )

        ddqz_z_full_e = wrapper_utils.as_field(
            ffi,
            xp,
            ddqz_z_full_e,
            ts.ScalarKind.FLOAT64,
            {Edge: ddqz_z_full_e_size_0, K: ddqz_z_full_e_size_1},
            False,
        )

        ddxt_z_full = wrapper_utils.as_field(
            ffi,
            xp,
            ddxt_z_full,
            ts.ScalarKind.FLOAT64,
            {Edge: ddxt_z_full_size_0, K: ddxt_z_full_size_1},
            False,
        )

        wgtfac_e = wrapper_utils.as_field(
            ffi,
            xp,
            wgtfac_e,
            ts.ScalarKind.FLOAT64,
            {Edge: wgtfac_e_size_0, K: wgtfac_e_size_1},
            False,
        )

        wgtfacq_e = wrapper_utils.as_field(
            ffi,
            xp,
            wgtfacq_e,
            ts.ScalarKind.FLOAT64,
            {Edge: wgtfacq_e_size_0, K: wgtfacq_e_size_1},
            False,
        )

        vwind_impl_wgt = wrapper_utils.as_field(
            ffi, xp, vwind_impl_wgt, ts.ScalarKind.FLOAT64, {Cell: vwind_impl_wgt_size_0}, False
        )

        hmask_dd3d = wrapper_utils.as_field(
            ffi, xp, hmask_dd3d, ts.ScalarKind.FLOAT64, {Edge: hmask_dd3d_size_0}, False
        )

        scalfac_dd3d = wrapper_utils.as_field(
            ffi, xp, scalfac_dd3d, ts.ScalarKind.FLOAT64, {K: scalfac_dd3d_size_0}, False
        )

        coeff1_dwdz = wrapper_utils.as_field(
            ffi,
            xp,
            coeff1_dwdz,
            ts.ScalarKind.FLOAT64,
            {Cell: coeff1_dwdz_size_0, K: coeff1_dwdz_size_1},
            False,
        )

        coeff2_dwdz = wrapper_utils.as_field(
            ffi,
            xp,
            coeff2_dwdz,
            ts.ScalarKind.FLOAT64,
            {Cell: coeff2_dwdz_size_0, K: coeff2_dwdz_size_1},
            False,
        )

        coeff_gradekin = wrapper_utils.as_field(
            ffi,
            xp,
            coeff_gradekin,
            ts.ScalarKind.FLOAT64,
            {Edge: coeff_gradekin_size_0, E2C: coeff_gradekin_size_1},
            False,
        )

        c_owner_mask = wrapper_utils.as_field(
            ffi, xp, c_owner_mask, ts.ScalarKind.BOOL, {Cell: c_owner_mask_size_0}, False
        )

        cell_center_lat = wrapper_utils.as_field(
            ffi, xp, cell_center_lat, ts.ScalarKind.FLOAT64, {Cell: cell_center_lat_size_0}, False
        )

        cell_center_lon = wrapper_utils.as_field(
            ffi, xp, cell_center_lon, ts.ScalarKind.FLOAT64, {Cell: cell_center_lon_size_0}, False
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

        assert isinstance(is_iau_active, int)
        is_iau_active = is_iau_active != 0

        assert isinstance(l_vert_nested, int)
        l_vert_nested = l_vert_nested != 0

        solve_nh_init(
            vct_a,
            vct_b,
            cell_areas,
            primal_normal_cell_x,
            primal_normal_cell_y,
            dual_normal_cell_x,
            dual_normal_cell_y,
            edge_areas,
            tangent_orientation,
            inverse_primal_edge_lengths,
            inverse_dual_edge_lengths,
            inverse_vertex_vertex_lengths,
            primal_normal_vert_x,
            primal_normal_vert_y,
            dual_normal_vert_x,
            dual_normal_vert_y,
            f_e,
            c_lin_e,
            c_intp,
            e_flx_avg,
            geofac_grdiv,
            geofac_rot,
            pos_on_tplane_e_1,
            pos_on_tplane_e_2,
            rbf_vec_coeff_e,
            e_bln_c_s,
            rbf_coeff_1,
            rbf_coeff_2,
            geofac_div,
            geofac_n2s,
            geofac_grg_x,
            geofac_grg_y,
            nudgecoeff_e,
            bdy_halo_c,
            mask_prog_halo_c,
            rayleigh_w,
            exner_exfac,
            exner_ref_mc,
            wgtfac_c,
            wgtfacq_c,
            inv_ddqz_z_full,
            rho_ref_mc,
            theta_ref_mc,
            vwind_expl_wgt,
            d_exner_dz_ref_ic,
            ddqz_z_half,
            theta_ref_ic,
            d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc,
            rho_ref_me,
            theta_ref_me,
            ddxn_z_full,
            zdiff_gradp,
            vertoffset_gradp,
            ipeidx_dsl,
            pg_exdist,
            ddqz_z_full_e,
            ddxt_z_full,
            wgtfac_e,
            wgtfacq_e,
            vwind_impl_wgt,
            hmask_dd3d,
            scalfac_dd3d,
            coeff1_dwdz,
            coeff2_dwdz,
            coeff_gradekin,
            c_owner_mask,
            cell_center_lat,
            cell_center_lon,
            edge_center_lat,
            edge_center_lon,
            primal_normal_x,
            primal_normal_y,
            rayleigh_damping_height,
            itime_scheme,
            iadv_rhotheta,
            igradp_method,
            ndyn_substeps,
            rayleigh_type,
            rayleigh_coeff,
            divdamp_order,
            is_iau_active,
            iau_wgt_dyn,
            divdamp_type,
            divdamp_trans_start,
            divdamp_trans_end,
            l_vert_nested,
            rhotheta_offctr,
            veladv_offctr,
            max_nudging_coeff,
            divdamp_fac,
            divdamp_fac2,
            divdamp_fac3,
            divdamp_fac4,
            divdamp_z,
            divdamp_z2,
            divdamp_z3,
            divdamp_z4,
            lowest_layer_thickness,
            model_top_height,
            stretch_factor,
            nflat_gradp,
            num_levels,
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

        msg = "shape of cell_areas after computation = %s" % str(
            cell_areas.shape if cell_areas is not None else "None"
        )
        logging.debug(msg)
        msg = "cell_areas after computation: %s" % str(
            cell_areas.ndarray if cell_areas is not None else "None"
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

        msg = "shape of edge_areas after computation = %s" % str(
            edge_areas.shape if edge_areas is not None else "None"
        )
        logging.debug(msg)
        msg = "edge_areas after computation: %s" % str(
            edge_areas.ndarray if edge_areas is not None else "None"
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

        msg = "shape of inverse_dual_edge_lengths after computation = %s" % str(
            inverse_dual_edge_lengths.shape if inverse_dual_edge_lengths is not None else "None"
        )
        logging.debug(msg)
        msg = "inverse_dual_edge_lengths after computation: %s" % str(
            inverse_dual_edge_lengths.ndarray if inverse_dual_edge_lengths is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of inverse_vertex_vertex_lengths after computation = %s" % str(
            inverse_vertex_vertex_lengths.shape
            if inverse_vertex_vertex_lengths is not None
            else "None"
        )
        logging.debug(msg)
        msg = "inverse_vertex_vertex_lengths after computation: %s" % str(
            inverse_vertex_vertex_lengths.ndarray
            if inverse_vertex_vertex_lengths is not None
            else "None"
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

        msg = "shape of f_e after computation = %s" % str(f_e.shape if f_e is not None else "None")
        logging.debug(msg)
        msg = "f_e after computation: %s" % str(f_e.ndarray if f_e is not None else "None")
        logging.debug(msg)

        msg = "shape of c_lin_e after computation = %s" % str(
            c_lin_e.shape if c_lin_e is not None else "None"
        )
        logging.debug(msg)
        msg = "c_lin_e after computation: %s" % str(
            c_lin_e.ndarray if c_lin_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of c_intp after computation = %s" % str(
            c_intp.shape if c_intp is not None else "None"
        )
        logging.debug(msg)
        msg = "c_intp after computation: %s" % str(c_intp.ndarray if c_intp is not None else "None")
        logging.debug(msg)

        msg = "shape of e_flx_avg after computation = %s" % str(
            e_flx_avg.shape if e_flx_avg is not None else "None"
        )
        logging.debug(msg)
        msg = "e_flx_avg after computation: %s" % str(
            e_flx_avg.ndarray if e_flx_avg is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of geofac_grdiv after computation = %s" % str(
            geofac_grdiv.shape if geofac_grdiv is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_grdiv after computation: %s" % str(
            geofac_grdiv.ndarray if geofac_grdiv is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of geofac_rot after computation = %s" % str(
            geofac_rot.shape if geofac_rot is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_rot after computation: %s" % str(
            geofac_rot.ndarray if geofac_rot is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of pos_on_tplane_e_1 after computation = %s" % str(
            pos_on_tplane_e_1.shape if pos_on_tplane_e_1 is not None else "None"
        )
        logging.debug(msg)
        msg = "pos_on_tplane_e_1 after computation: %s" % str(
            pos_on_tplane_e_1.ndarray if pos_on_tplane_e_1 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of pos_on_tplane_e_2 after computation = %s" % str(
            pos_on_tplane_e_2.shape if pos_on_tplane_e_2 is not None else "None"
        )
        logging.debug(msg)
        msg = "pos_on_tplane_e_2 after computation: %s" % str(
            pos_on_tplane_e_2.ndarray if pos_on_tplane_e_2 is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rbf_vec_coeff_e after computation = %s" % str(
            rbf_vec_coeff_e.shape if rbf_vec_coeff_e is not None else "None"
        )
        logging.debug(msg)
        msg = "rbf_vec_coeff_e after computation: %s" % str(
            rbf_vec_coeff_e.ndarray if rbf_vec_coeff_e is not None else "None"
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

        msg = "shape of geofac_div after computation = %s" % str(
            geofac_div.shape if geofac_div is not None else "None"
        )
        logging.debug(msg)
        msg = "geofac_div after computation: %s" % str(
            geofac_div.ndarray if geofac_div is not None else "None"
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

        msg = "shape of nudgecoeff_e after computation = %s" % str(
            nudgecoeff_e.shape if nudgecoeff_e is not None else "None"
        )
        logging.debug(msg)
        msg = "nudgecoeff_e after computation: %s" % str(
            nudgecoeff_e.ndarray if nudgecoeff_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of bdy_halo_c after computation = %s" % str(
            bdy_halo_c.shape if bdy_halo_c is not None else "None"
        )
        logging.debug(msg)
        msg = "bdy_halo_c after computation: %s" % str(
            bdy_halo_c.ndarray if bdy_halo_c is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of mask_prog_halo_c after computation = %s" % str(
            mask_prog_halo_c.shape if mask_prog_halo_c is not None else "None"
        )
        logging.debug(msg)
        msg = "mask_prog_halo_c after computation: %s" % str(
            mask_prog_halo_c.ndarray if mask_prog_halo_c is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rayleigh_w after computation = %s" % str(
            rayleigh_w.shape if rayleigh_w is not None else "None"
        )
        logging.debug(msg)
        msg = "rayleigh_w after computation: %s" % str(
            rayleigh_w.ndarray if rayleigh_w is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of exner_exfac after computation = %s" % str(
            exner_exfac.shape if exner_exfac is not None else "None"
        )
        logging.debug(msg)
        msg = "exner_exfac after computation: %s" % str(
            exner_exfac.ndarray if exner_exfac is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of exner_ref_mc after computation = %s" % str(
            exner_ref_mc.shape if exner_ref_mc is not None else "None"
        )
        logging.debug(msg)
        msg = "exner_ref_mc after computation: %s" % str(
            exner_ref_mc.ndarray if exner_ref_mc is not None else "None"
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

        msg = "shape of wgtfacq_c after computation = %s" % str(
            wgtfacq_c.shape if wgtfacq_c is not None else "None"
        )
        logging.debug(msg)
        msg = "wgtfacq_c after computation: %s" % str(
            wgtfacq_c.ndarray if wgtfacq_c is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of inv_ddqz_z_full after computation = %s" % str(
            inv_ddqz_z_full.shape if inv_ddqz_z_full is not None else "None"
        )
        logging.debug(msg)
        msg = "inv_ddqz_z_full after computation: %s" % str(
            inv_ddqz_z_full.ndarray if inv_ddqz_z_full is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rho_ref_mc after computation = %s" % str(
            rho_ref_mc.shape if rho_ref_mc is not None else "None"
        )
        logging.debug(msg)
        msg = "rho_ref_mc after computation: %s" % str(
            rho_ref_mc.ndarray if rho_ref_mc is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of theta_ref_mc after computation = %s" % str(
            theta_ref_mc.shape if theta_ref_mc is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_ref_mc after computation: %s" % str(
            theta_ref_mc.ndarray if theta_ref_mc is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vwind_expl_wgt after computation = %s" % str(
            vwind_expl_wgt.shape if vwind_expl_wgt is not None else "None"
        )
        logging.debug(msg)
        msg = "vwind_expl_wgt after computation: %s" % str(
            vwind_expl_wgt.ndarray if vwind_expl_wgt is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of d_exner_dz_ref_ic after computation = %s" % str(
            d_exner_dz_ref_ic.shape if d_exner_dz_ref_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "d_exner_dz_ref_ic after computation: %s" % str(
            d_exner_dz_ref_ic.ndarray if d_exner_dz_ref_ic is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddqz_z_half after computation = %s" % str(
            ddqz_z_half.shape if ddqz_z_half is not None else "None"
        )
        logging.debug(msg)
        msg = "ddqz_z_half after computation: %s" % str(
            ddqz_z_half.ndarray if ddqz_z_half is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of theta_ref_ic after computation = %s" % str(
            theta_ref_ic.shape if theta_ref_ic is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_ref_ic after computation: %s" % str(
            theta_ref_ic.ndarray if theta_ref_ic is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of d2dexdz2_fac1_mc after computation = %s" % str(
            d2dexdz2_fac1_mc.shape if d2dexdz2_fac1_mc is not None else "None"
        )
        logging.debug(msg)
        msg = "d2dexdz2_fac1_mc after computation: %s" % str(
            d2dexdz2_fac1_mc.ndarray if d2dexdz2_fac1_mc is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of d2dexdz2_fac2_mc after computation = %s" % str(
            d2dexdz2_fac2_mc.shape if d2dexdz2_fac2_mc is not None else "None"
        )
        logging.debug(msg)
        msg = "d2dexdz2_fac2_mc after computation: %s" % str(
            d2dexdz2_fac2_mc.ndarray if d2dexdz2_fac2_mc is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of rho_ref_me after computation = %s" % str(
            rho_ref_me.shape if rho_ref_me is not None else "None"
        )
        logging.debug(msg)
        msg = "rho_ref_me after computation: %s" % str(
            rho_ref_me.ndarray if rho_ref_me is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of theta_ref_me after computation = %s" % str(
            theta_ref_me.shape if theta_ref_me is not None else "None"
        )
        logging.debug(msg)
        msg = "theta_ref_me after computation: %s" % str(
            theta_ref_me.ndarray if theta_ref_me is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddxn_z_full after computation = %s" % str(
            ddxn_z_full.shape if ddxn_z_full is not None else "None"
        )
        logging.debug(msg)
        msg = "ddxn_z_full after computation: %s" % str(
            ddxn_z_full.ndarray if ddxn_z_full is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of zdiff_gradp after computation = %s" % str(
            zdiff_gradp.shape if zdiff_gradp is not None else "None"
        )
        logging.debug(msg)
        msg = "zdiff_gradp after computation: %s" % str(
            zdiff_gradp.ndarray if zdiff_gradp is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vertoffset_gradp after computation = %s" % str(
            vertoffset_gradp.shape if vertoffset_gradp is not None else "None"
        )
        logging.debug(msg)
        msg = "vertoffset_gradp after computation: %s" % str(
            vertoffset_gradp.ndarray if vertoffset_gradp is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ipeidx_dsl after computation = %s" % str(
            ipeidx_dsl.shape if ipeidx_dsl is not None else "None"
        )
        logging.debug(msg)
        msg = "ipeidx_dsl after computation: %s" % str(
            ipeidx_dsl.ndarray if ipeidx_dsl is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of pg_exdist after computation = %s" % str(
            pg_exdist.shape if pg_exdist is not None else "None"
        )
        logging.debug(msg)
        msg = "pg_exdist after computation: %s" % str(
            pg_exdist.ndarray if pg_exdist is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddqz_z_full_e after computation = %s" % str(
            ddqz_z_full_e.shape if ddqz_z_full_e is not None else "None"
        )
        logging.debug(msg)
        msg = "ddqz_z_full_e after computation: %s" % str(
            ddqz_z_full_e.ndarray if ddqz_z_full_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of ddxt_z_full after computation = %s" % str(
            ddxt_z_full.shape if ddxt_z_full is not None else "None"
        )
        logging.debug(msg)
        msg = "ddxt_z_full after computation: %s" % str(
            ddxt_z_full.ndarray if ddxt_z_full is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of wgtfac_e after computation = %s" % str(
            wgtfac_e.shape if wgtfac_e is not None else "None"
        )
        logging.debug(msg)
        msg = "wgtfac_e after computation: %s" % str(
            wgtfac_e.ndarray if wgtfac_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of wgtfacq_e after computation = %s" % str(
            wgtfacq_e.shape if wgtfacq_e is not None else "None"
        )
        logging.debug(msg)
        msg = "wgtfacq_e after computation: %s" % str(
            wgtfacq_e.ndarray if wgtfacq_e is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of vwind_impl_wgt after computation = %s" % str(
            vwind_impl_wgt.shape if vwind_impl_wgt is not None else "None"
        )
        logging.debug(msg)
        msg = "vwind_impl_wgt after computation: %s" % str(
            vwind_impl_wgt.ndarray if vwind_impl_wgt is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of hmask_dd3d after computation = %s" % str(
            hmask_dd3d.shape if hmask_dd3d is not None else "None"
        )
        logging.debug(msg)
        msg = "hmask_dd3d after computation: %s" % str(
            hmask_dd3d.ndarray if hmask_dd3d is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of scalfac_dd3d after computation = %s" % str(
            scalfac_dd3d.shape if scalfac_dd3d is not None else "None"
        )
        logging.debug(msg)
        msg = "scalfac_dd3d after computation: %s" % str(
            scalfac_dd3d.ndarray if scalfac_dd3d is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of coeff1_dwdz after computation = %s" % str(
            coeff1_dwdz.shape if coeff1_dwdz is not None else "None"
        )
        logging.debug(msg)
        msg = "coeff1_dwdz after computation: %s" % str(
            coeff1_dwdz.ndarray if coeff1_dwdz is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of coeff2_dwdz after computation = %s" % str(
            coeff2_dwdz.shape if coeff2_dwdz is not None else "None"
        )
        logging.debug(msg)
        msg = "coeff2_dwdz after computation: %s" % str(
            coeff2_dwdz.ndarray if coeff2_dwdz is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of coeff_gradekin after computation = %s" % str(
            coeff_gradekin.shape if coeff_gradekin is not None else "None"
        )
        logging.debug(msg)
        msg = "coeff_gradekin after computation: %s" % str(
            coeff_gradekin.ndarray if coeff_gradekin is not None else "None"
        )
        logging.debug(msg)

        msg = "shape of c_owner_mask after computation = %s" % str(
            c_owner_mask.shape if c_owner_mask is not None else "None"
        )
        logging.debug(msg)
        msg = "c_owner_mask after computation: %s" % str(
            c_owner_mask.ndarray if c_owner_mask is not None else "None"
        )
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

        logging.critical("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0
