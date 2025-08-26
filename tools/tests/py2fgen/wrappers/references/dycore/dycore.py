import logging
from dycore import ffi
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
from icon4py.tools.py2fgen.wrappers.dycore_wrapper import solve_nh_run
from icon4py.tools.py2fgen.wrappers.dycore_wrapper import solve_nh_init


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
    vn_incr,
    vn_incr_size_0,
    vn_incr_size_1,
    rho_incr,
    rho_incr_size_0,
    rho_incr_size_1,
    exner_incr,
    exner_incr_size_0,
    exner_incr_size_1,
    mass_flx_me,
    mass_flx_me_size_0,
    mass_flx_me_size_1,
    mass_flx_ic,
    mass_flx_ic_size_0,
    mass_flx_ic_size_1,
    vol_flx_ic,
    vol_flx_ic_size_0,
    vol_flx_ic_size_1,
    vn_traj,
    vn_traj_size_0,
    vn_traj_size_1,
    dtime,
    max_vcfl,
    lprep_adv,
    at_initial_timestep,
    divdamp_fac_o2,
    ndyn_substeps_var,
    idyn_timestep,
    on_gpu,
):
    try:
        if __debug__:
            logger.info("Python execution of solve_nh_run started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayInfos

        rho_now = (
            rho_now,
            (
                rho_now_size_0,
                rho_now_size_1,
            ),
            on_gpu,
            False,
        )

        rho_new = (
            rho_new,
            (
                rho_new_size_0,
                rho_new_size_1,
            ),
            on_gpu,
            False,
        )

        exner_now = (
            exner_now,
            (
                exner_now_size_0,
                exner_now_size_1,
            ),
            on_gpu,
            False,
        )

        exner_new = (
            exner_new,
            (
                exner_new_size_0,
                exner_new_size_1,
            ),
            on_gpu,
            False,
        )

        w_now = (
            w_now,
            (
                w_now_size_0,
                w_now_size_1,
            ),
            on_gpu,
            False,
        )

        w_new = (
            w_new,
            (
                w_new_size_0,
                w_new_size_1,
            ),
            on_gpu,
            False,
        )

        theta_v_now = (
            theta_v_now,
            (
                theta_v_now_size_0,
                theta_v_now_size_1,
            ),
            on_gpu,
            False,
        )

        theta_v_new = (
            theta_v_new,
            (
                theta_v_new_size_0,
                theta_v_new_size_1,
            ),
            on_gpu,
            False,
        )

        vn_now = (
            vn_now,
            (
                vn_now_size_0,
                vn_now_size_1,
            ),
            on_gpu,
            False,
        )

        vn_new = (
            vn_new,
            (
                vn_new_size_0,
                vn_new_size_1,
            ),
            on_gpu,
            False,
        )

        w_concorr_c = (
            w_concorr_c,
            (
                w_concorr_c_size_0,
                w_concorr_c_size_1,
            ),
            on_gpu,
            False,
        )

        ddt_vn_apc_ntl1 = (
            ddt_vn_apc_ntl1,
            (
                ddt_vn_apc_ntl1_size_0,
                ddt_vn_apc_ntl1_size_1,
            ),
            on_gpu,
            False,
        )

        ddt_vn_apc_ntl2 = (
            ddt_vn_apc_ntl2,
            (
                ddt_vn_apc_ntl2_size_0,
                ddt_vn_apc_ntl2_size_1,
            ),
            on_gpu,
            False,
        )

        ddt_w_adv_ntl1 = (
            ddt_w_adv_ntl1,
            (
                ddt_w_adv_ntl1_size_0,
                ddt_w_adv_ntl1_size_1,
            ),
            on_gpu,
            False,
        )

        ddt_w_adv_ntl2 = (
            ddt_w_adv_ntl2,
            (
                ddt_w_adv_ntl2_size_0,
                ddt_w_adv_ntl2_size_1,
            ),
            on_gpu,
            False,
        )

        theta_v_ic = (
            theta_v_ic,
            (
                theta_v_ic_size_0,
                theta_v_ic_size_1,
            ),
            on_gpu,
            False,
        )

        rho_ic = (
            rho_ic,
            (
                rho_ic_size_0,
                rho_ic_size_1,
            ),
            on_gpu,
            False,
        )

        exner_pr = (
            exner_pr,
            (
                exner_pr_size_0,
                exner_pr_size_1,
            ),
            on_gpu,
            False,
        )

        exner_dyn_incr = (
            exner_dyn_incr,
            (
                exner_dyn_incr_size_0,
                exner_dyn_incr_size_1,
            ),
            on_gpu,
            False,
        )

        ddt_exner_phy = (
            ddt_exner_phy,
            (
                ddt_exner_phy_size_0,
                ddt_exner_phy_size_1,
            ),
            on_gpu,
            False,
        )

        grf_tend_rho = (
            grf_tend_rho,
            (
                grf_tend_rho_size_0,
                grf_tend_rho_size_1,
            ),
            on_gpu,
            False,
        )

        grf_tend_thv = (
            grf_tend_thv,
            (
                grf_tend_thv_size_0,
                grf_tend_thv_size_1,
            ),
            on_gpu,
            False,
        )

        grf_tend_w = (
            grf_tend_w,
            (
                grf_tend_w_size_0,
                grf_tend_w_size_1,
            ),
            on_gpu,
            False,
        )

        mass_fl_e = (
            mass_fl_e,
            (
                mass_fl_e_size_0,
                mass_fl_e_size_1,
            ),
            on_gpu,
            False,
        )

        ddt_vn_phy = (
            ddt_vn_phy,
            (
                ddt_vn_phy_size_0,
                ddt_vn_phy_size_1,
            ),
            on_gpu,
            False,
        )

        grf_tend_vn = (
            grf_tend_vn,
            (
                grf_tend_vn_size_0,
                grf_tend_vn_size_1,
            ),
            on_gpu,
            False,
        )

        vn_ie = (
            vn_ie,
            (
                vn_ie_size_0,
                vn_ie_size_1,
            ),
            on_gpu,
            False,
        )

        vt = (
            vt,
            (
                vt_size_0,
                vt_size_1,
            ),
            on_gpu,
            False,
        )

        vn_incr = (
            vn_incr,
            (
                vn_incr_size_0,
                vn_incr_size_1,
            ),
            on_gpu,
            True,
        )

        rho_incr = (
            rho_incr,
            (
                rho_incr_size_0,
                rho_incr_size_1,
            ),
            on_gpu,
            True,
        )

        exner_incr = (
            exner_incr,
            (
                exner_incr_size_0,
                exner_incr_size_1,
            ),
            on_gpu,
            True,
        )

        mass_flx_me = (
            mass_flx_me,
            (
                mass_flx_me_size_0,
                mass_flx_me_size_1,
            ),
            on_gpu,
            False,
        )

        mass_flx_ic = (
            mass_flx_ic,
            (
                mass_flx_ic_size_0,
                mass_flx_ic_size_1,
            ),
            on_gpu,
            False,
        )

        vol_flx_ic = (
            vol_flx_ic,
            (
                vol_flx_ic_size_0,
                vol_flx_ic_size_1,
            ),
            on_gpu,
            False,
        )

        vn_traj = (
            vn_traj,
            (
                vn_traj_size_0,
                vn_traj_size_1,
            ),
            on_gpu,
            False,
        )

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "solve_nh_run constructing `ArrayInfos` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            perf_counters = {}
        else:
            perf_counters = None
        solve_nh_run(
            ffi=ffi,
            perf_counters=perf_counters,
            rho_now=rho_now,
            rho_new=rho_new,
            exner_now=exner_now,
            exner_new=exner_new,
            w_now=w_now,
            w_new=w_new,
            theta_v_now=theta_v_now,
            theta_v_new=theta_v_new,
            vn_now=vn_now,
            vn_new=vn_new,
            w_concorr_c=w_concorr_c,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            theta_v_ic=theta_v_ic,
            rho_ic=rho_ic,
            exner_pr=exner_pr,
            exner_dyn_incr=exner_dyn_incr,
            ddt_exner_phy=ddt_exner_phy,
            grf_tend_rho=grf_tend_rho,
            grf_tend_thv=grf_tend_thv,
            grf_tend_w=grf_tend_w,
            mass_fl_e=mass_fl_e,
            ddt_vn_phy=ddt_vn_phy,
            grf_tend_vn=grf_tend_vn,
            vn_ie=vn_ie,
            vt=vt,
            vn_incr=vn_incr,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            mass_flx_me=mass_flx_me,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            vn_traj=vn_traj,
            dtime=dtime,
            max_vcfl=max_vcfl,
            lprep_adv=lprep_adv,
            at_initial_timestep=at_initial_timestep,
            divdamp_fac_o2=divdamp_fac_o2,
            ndyn_substeps_var=ndyn_substeps_var,
            idyn_timestep=idyn_timestep,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "solve_nh_run convert time: %s"
                    % str(perf_counters["convert_end_time"] - perf_counters["convert_start_time"])
                )
                logger.info(
                    "solve_nh_run execution time: %s" % str(func_end_time - func_start_time)
                )

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                rho_now_arr = (
                    _conversion.as_array(ffi, rho_now, _definitions.FLOAT64)
                    if rho_now is not None
                    else None
                )
                msg = "shape of rho_now after computation = %s" % str(
                    rho_now_arr.shape if rho_now is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rho_now after computation: %s" % str(rho_now_arr)
                    if rho_now is not None
                    else "None"
                )
                logger.debug(msg)

                rho_new_arr = (
                    _conversion.as_array(ffi, rho_new, _definitions.FLOAT64)
                    if rho_new is not None
                    else None
                )
                msg = "shape of rho_new after computation = %s" % str(
                    rho_new_arr.shape if rho_new is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rho_new after computation: %s" % str(rho_new_arr)
                    if rho_new is not None
                    else "None"
                )
                logger.debug(msg)

                exner_now_arr = (
                    _conversion.as_array(ffi, exner_now, _definitions.FLOAT64)
                    if exner_now is not None
                    else None
                )
                msg = "shape of exner_now after computation = %s" % str(
                    exner_now_arr.shape if exner_now is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_now after computation: %s" % str(exner_now_arr)
                    if exner_now is not None
                    else "None"
                )
                logger.debug(msg)

                exner_new_arr = (
                    _conversion.as_array(ffi, exner_new, _definitions.FLOAT64)
                    if exner_new is not None
                    else None
                )
                msg = "shape of exner_new after computation = %s" % str(
                    exner_new_arr.shape if exner_new is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_new after computation: %s" % str(exner_new_arr)
                    if exner_new is not None
                    else "None"
                )
                logger.debug(msg)

                w_now_arr = (
                    _conversion.as_array(ffi, w_now, _definitions.FLOAT64)
                    if w_now is not None
                    else None
                )
                msg = "shape of w_now after computation = %s" % str(
                    w_now_arr.shape if w_now is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "w_now after computation: %s" % str(w_now_arr) if w_now is not None else "None"
                )
                logger.debug(msg)

                w_new_arr = (
                    _conversion.as_array(ffi, w_new, _definitions.FLOAT64)
                    if w_new is not None
                    else None
                )
                msg = "shape of w_new after computation = %s" % str(
                    w_new_arr.shape if w_new is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "w_new after computation: %s" % str(w_new_arr) if w_new is not None else "None"
                )
                logger.debug(msg)

                theta_v_now_arr = (
                    _conversion.as_array(ffi, theta_v_now, _definitions.FLOAT64)
                    if theta_v_now is not None
                    else None
                )
                msg = "shape of theta_v_now after computation = %s" % str(
                    theta_v_now_arr.shape if theta_v_now is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "theta_v_now after computation: %s" % str(theta_v_now_arr)
                    if theta_v_now is not None
                    else "None"
                )
                logger.debug(msg)

                theta_v_new_arr = (
                    _conversion.as_array(ffi, theta_v_new, _definitions.FLOAT64)
                    if theta_v_new is not None
                    else None
                )
                msg = "shape of theta_v_new after computation = %s" % str(
                    theta_v_new_arr.shape if theta_v_new is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "theta_v_new after computation: %s" % str(theta_v_new_arr)
                    if theta_v_new is not None
                    else "None"
                )
                logger.debug(msg)

                vn_now_arr = (
                    _conversion.as_array(ffi, vn_now, _definitions.FLOAT64)
                    if vn_now is not None
                    else None
                )
                msg = "shape of vn_now after computation = %s" % str(
                    vn_now_arr.shape if vn_now is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vn_now after computation: %s" % str(vn_now_arr)
                    if vn_now is not None
                    else "None"
                )
                logger.debug(msg)

                vn_new_arr = (
                    _conversion.as_array(ffi, vn_new, _definitions.FLOAT64)
                    if vn_new is not None
                    else None
                )
                msg = "shape of vn_new after computation = %s" % str(
                    vn_new_arr.shape if vn_new is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vn_new after computation: %s" % str(vn_new_arr)
                    if vn_new is not None
                    else "None"
                )
                logger.debug(msg)

                w_concorr_c_arr = (
                    _conversion.as_array(ffi, w_concorr_c, _definitions.FLOAT64)
                    if w_concorr_c is not None
                    else None
                )
                msg = "shape of w_concorr_c after computation = %s" % str(
                    w_concorr_c_arr.shape if w_concorr_c is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "w_concorr_c after computation: %s" % str(w_concorr_c_arr)
                    if w_concorr_c is not None
                    else "None"
                )
                logger.debug(msg)

                ddt_vn_apc_ntl1_arr = (
                    _conversion.as_array(ffi, ddt_vn_apc_ntl1, _definitions.FLOAT64)
                    if ddt_vn_apc_ntl1 is not None
                    else None
                )
                msg = "shape of ddt_vn_apc_ntl1 after computation = %s" % str(
                    ddt_vn_apc_ntl1_arr.shape if ddt_vn_apc_ntl1 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddt_vn_apc_ntl1 after computation: %s" % str(ddt_vn_apc_ntl1_arr)
                    if ddt_vn_apc_ntl1 is not None
                    else "None"
                )
                logger.debug(msg)

                ddt_vn_apc_ntl2_arr = (
                    _conversion.as_array(ffi, ddt_vn_apc_ntl2, _definitions.FLOAT64)
                    if ddt_vn_apc_ntl2 is not None
                    else None
                )
                msg = "shape of ddt_vn_apc_ntl2 after computation = %s" % str(
                    ddt_vn_apc_ntl2_arr.shape if ddt_vn_apc_ntl2 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddt_vn_apc_ntl2 after computation: %s" % str(ddt_vn_apc_ntl2_arr)
                    if ddt_vn_apc_ntl2 is not None
                    else "None"
                )
                logger.debug(msg)

                ddt_w_adv_ntl1_arr = (
                    _conversion.as_array(ffi, ddt_w_adv_ntl1, _definitions.FLOAT64)
                    if ddt_w_adv_ntl1 is not None
                    else None
                )
                msg = "shape of ddt_w_adv_ntl1 after computation = %s" % str(
                    ddt_w_adv_ntl1_arr.shape if ddt_w_adv_ntl1 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddt_w_adv_ntl1 after computation: %s" % str(ddt_w_adv_ntl1_arr)
                    if ddt_w_adv_ntl1 is not None
                    else "None"
                )
                logger.debug(msg)

                ddt_w_adv_ntl2_arr = (
                    _conversion.as_array(ffi, ddt_w_adv_ntl2, _definitions.FLOAT64)
                    if ddt_w_adv_ntl2 is not None
                    else None
                )
                msg = "shape of ddt_w_adv_ntl2 after computation = %s" % str(
                    ddt_w_adv_ntl2_arr.shape if ddt_w_adv_ntl2 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddt_w_adv_ntl2 after computation: %s" % str(ddt_w_adv_ntl2_arr)
                    if ddt_w_adv_ntl2 is not None
                    else "None"
                )
                logger.debug(msg)

                theta_v_ic_arr = (
                    _conversion.as_array(ffi, theta_v_ic, _definitions.FLOAT64)
                    if theta_v_ic is not None
                    else None
                )
                msg = "shape of theta_v_ic after computation = %s" % str(
                    theta_v_ic_arr.shape if theta_v_ic is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "theta_v_ic after computation: %s" % str(theta_v_ic_arr)
                    if theta_v_ic is not None
                    else "None"
                )
                logger.debug(msg)

                rho_ic_arr = (
                    _conversion.as_array(ffi, rho_ic, _definitions.FLOAT64)
                    if rho_ic is not None
                    else None
                )
                msg = "shape of rho_ic after computation = %s" % str(
                    rho_ic_arr.shape if rho_ic is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rho_ic after computation: %s" % str(rho_ic_arr)
                    if rho_ic is not None
                    else "None"
                )
                logger.debug(msg)

                exner_pr_arr = (
                    _conversion.as_array(ffi, exner_pr, _definitions.FLOAT64)
                    if exner_pr is not None
                    else None
                )
                msg = "shape of exner_pr after computation = %s" % str(
                    exner_pr_arr.shape if exner_pr is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_pr after computation: %s" % str(exner_pr_arr)
                    if exner_pr is not None
                    else "None"
                )
                logger.debug(msg)

                exner_dyn_incr_arr = (
                    _conversion.as_array(ffi, exner_dyn_incr, _definitions.FLOAT64)
                    if exner_dyn_incr is not None
                    else None
                )
                msg = "shape of exner_dyn_incr after computation = %s" % str(
                    exner_dyn_incr_arr.shape if exner_dyn_incr is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_dyn_incr after computation: %s" % str(exner_dyn_incr_arr)
                    if exner_dyn_incr is not None
                    else "None"
                )
                logger.debug(msg)

                ddt_exner_phy_arr = (
                    _conversion.as_array(ffi, ddt_exner_phy, _definitions.FLOAT64)
                    if ddt_exner_phy is not None
                    else None
                )
                msg = "shape of ddt_exner_phy after computation = %s" % str(
                    ddt_exner_phy_arr.shape if ddt_exner_phy is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddt_exner_phy after computation: %s" % str(ddt_exner_phy_arr)
                    if ddt_exner_phy is not None
                    else "None"
                )
                logger.debug(msg)

                grf_tend_rho_arr = (
                    _conversion.as_array(ffi, grf_tend_rho, _definitions.FLOAT64)
                    if grf_tend_rho is not None
                    else None
                )
                msg = "shape of grf_tend_rho after computation = %s" % str(
                    grf_tend_rho_arr.shape if grf_tend_rho is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "grf_tend_rho after computation: %s" % str(grf_tend_rho_arr)
                    if grf_tend_rho is not None
                    else "None"
                )
                logger.debug(msg)

                grf_tend_thv_arr = (
                    _conversion.as_array(ffi, grf_tend_thv, _definitions.FLOAT64)
                    if grf_tend_thv is not None
                    else None
                )
                msg = "shape of grf_tend_thv after computation = %s" % str(
                    grf_tend_thv_arr.shape if grf_tend_thv is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "grf_tend_thv after computation: %s" % str(grf_tend_thv_arr)
                    if grf_tend_thv is not None
                    else "None"
                )
                logger.debug(msg)

                grf_tend_w_arr = (
                    _conversion.as_array(ffi, grf_tend_w, _definitions.FLOAT64)
                    if grf_tend_w is not None
                    else None
                )
                msg = "shape of grf_tend_w after computation = %s" % str(
                    grf_tend_w_arr.shape if grf_tend_w is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "grf_tend_w after computation: %s" % str(grf_tend_w_arr)
                    if grf_tend_w is not None
                    else "None"
                )
                logger.debug(msg)

                mass_fl_e_arr = (
                    _conversion.as_array(ffi, mass_fl_e, _definitions.FLOAT64)
                    if mass_fl_e is not None
                    else None
                )
                msg = "shape of mass_fl_e after computation = %s" % str(
                    mass_fl_e_arr.shape if mass_fl_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "mass_fl_e after computation: %s" % str(mass_fl_e_arr)
                    if mass_fl_e is not None
                    else "None"
                )
                logger.debug(msg)

                ddt_vn_phy_arr = (
                    _conversion.as_array(ffi, ddt_vn_phy, _definitions.FLOAT64)
                    if ddt_vn_phy is not None
                    else None
                )
                msg = "shape of ddt_vn_phy after computation = %s" % str(
                    ddt_vn_phy_arr.shape if ddt_vn_phy is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddt_vn_phy after computation: %s" % str(ddt_vn_phy_arr)
                    if ddt_vn_phy is not None
                    else "None"
                )
                logger.debug(msg)

                grf_tend_vn_arr = (
                    _conversion.as_array(ffi, grf_tend_vn, _definitions.FLOAT64)
                    if grf_tend_vn is not None
                    else None
                )
                msg = "shape of grf_tend_vn after computation = %s" % str(
                    grf_tend_vn_arr.shape if grf_tend_vn is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "grf_tend_vn after computation: %s" % str(grf_tend_vn_arr)
                    if grf_tend_vn is not None
                    else "None"
                )
                logger.debug(msg)

                vn_ie_arr = (
                    _conversion.as_array(ffi, vn_ie, _definitions.FLOAT64)
                    if vn_ie is not None
                    else None
                )
                msg = "shape of vn_ie after computation = %s" % str(
                    vn_ie_arr.shape if vn_ie is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vn_ie after computation: %s" % str(vn_ie_arr) if vn_ie is not None else "None"
                )
                logger.debug(msg)

                vt_arr = (
                    _conversion.as_array(ffi, vt, _definitions.FLOAT64) if vt is not None else None
                )
                msg = "shape of vt after computation = %s" % str(
                    vt_arr.shape if vt is not None else "None"
                )
                logger.debug(msg)
                msg = "vt after computation: %s" % str(vt_arr) if vt is not None else "None"
                logger.debug(msg)

                vn_incr_arr = (
                    _conversion.as_array(ffi, vn_incr, _definitions.FLOAT64)
                    if vn_incr is not None
                    else None
                )
                msg = "shape of vn_incr after computation = %s" % str(
                    vn_incr_arr.shape if vn_incr is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vn_incr after computation: %s" % str(vn_incr_arr)
                    if vn_incr is not None
                    else "None"
                )
                logger.debug(msg)

                rho_incr_arr = (
                    _conversion.as_array(ffi, rho_incr, _definitions.FLOAT64)
                    if rho_incr is not None
                    else None
                )
                msg = "shape of rho_incr after computation = %s" % str(
                    rho_incr_arr.shape if rho_incr is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rho_incr after computation: %s" % str(rho_incr_arr)
                    if rho_incr is not None
                    else "None"
                )
                logger.debug(msg)

                exner_incr_arr = (
                    _conversion.as_array(ffi, exner_incr, _definitions.FLOAT64)
                    if exner_incr is not None
                    else None
                )
                msg = "shape of exner_incr after computation = %s" % str(
                    exner_incr_arr.shape if exner_incr is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_incr after computation: %s" % str(exner_incr_arr)
                    if exner_incr is not None
                    else "None"
                )
                logger.debug(msg)

                mass_flx_me_arr = (
                    _conversion.as_array(ffi, mass_flx_me, _definitions.FLOAT64)
                    if mass_flx_me is not None
                    else None
                )
                msg = "shape of mass_flx_me after computation = %s" % str(
                    mass_flx_me_arr.shape if mass_flx_me is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "mass_flx_me after computation: %s" % str(mass_flx_me_arr)
                    if mass_flx_me is not None
                    else "None"
                )
                logger.debug(msg)

                mass_flx_ic_arr = (
                    _conversion.as_array(ffi, mass_flx_ic, _definitions.FLOAT64)
                    if mass_flx_ic is not None
                    else None
                )
                msg = "shape of mass_flx_ic after computation = %s" % str(
                    mass_flx_ic_arr.shape if mass_flx_ic is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "mass_flx_ic after computation: %s" % str(mass_flx_ic_arr)
                    if mass_flx_ic is not None
                    else "None"
                )
                logger.debug(msg)

                vol_flx_ic_arr = (
                    _conversion.as_array(ffi, vol_flx_ic, _definitions.FLOAT64)
                    if vol_flx_ic is not None
                    else None
                )
                msg = "shape of vol_flx_ic after computation = %s" % str(
                    vol_flx_ic_arr.shape if vol_flx_ic is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vol_flx_ic after computation: %s" % str(vol_flx_ic_arr)
                    if vol_flx_ic is not None
                    else "None"
                )
                logger.debug(msg)

                vn_traj_arr = (
                    _conversion.as_array(ffi, vn_traj, _definitions.FLOAT64)
                    if vn_traj is not None
                    else None
                )
                msg = "shape of vn_traj after computation = %s" % str(
                    vn_traj_arr.shape if vn_traj is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vn_traj after computation: %s" % str(vn_traj_arr)
                    if vn_traj is not None
                    else "None"
                )
                logger.debug(msg)

        if __debug__:
            logger.info("Python execution of solve_nh_run completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0


@ffi.def_extern()
def solve_nh_init_wrapper(
    vct_a,
    vct_a_size_0,
    vct_b,
    vct_b_size_0,
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
    rayleigh_damping_height,
    itime_scheme,
    iadv_rhotheta,
    igradp_method,
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
    nudge_max_coeff,
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
    backend,
    on_gpu,
):
    try:
        if __debug__:
            logger.info("Python execution of solve_nh_init started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayInfos

        vct_a = (vct_a, (vct_a_size_0,), on_gpu, False)

        vct_b = (vct_b, (vct_b_size_0,), on_gpu, False)

        c_lin_e = (
            c_lin_e,
            (
                c_lin_e_size_0,
                c_lin_e_size_1,
            ),
            on_gpu,
            False,
        )

        c_intp = (
            c_intp,
            (
                c_intp_size_0,
                c_intp_size_1,
            ),
            on_gpu,
            False,
        )

        e_flx_avg = (
            e_flx_avg,
            (
                e_flx_avg_size_0,
                e_flx_avg_size_1,
            ),
            on_gpu,
            False,
        )

        geofac_grdiv = (
            geofac_grdiv,
            (
                geofac_grdiv_size_0,
                geofac_grdiv_size_1,
            ),
            on_gpu,
            False,
        )

        geofac_rot = (
            geofac_rot,
            (
                geofac_rot_size_0,
                geofac_rot_size_1,
            ),
            on_gpu,
            False,
        )

        pos_on_tplane_e_1 = (
            pos_on_tplane_e_1,
            (
                pos_on_tplane_e_1_size_0,
                pos_on_tplane_e_1_size_1,
            ),
            on_gpu,
            False,
        )

        pos_on_tplane_e_2 = (
            pos_on_tplane_e_2,
            (
                pos_on_tplane_e_2_size_0,
                pos_on_tplane_e_2_size_1,
            ),
            on_gpu,
            False,
        )

        rbf_vec_coeff_e = (
            rbf_vec_coeff_e,
            (
                rbf_vec_coeff_e_size_0,
                rbf_vec_coeff_e_size_1,
            ),
            on_gpu,
            False,
        )

        e_bln_c_s = (
            e_bln_c_s,
            (
                e_bln_c_s_size_0,
                e_bln_c_s_size_1,
            ),
            on_gpu,
            False,
        )

        rbf_coeff_1 = (
            rbf_coeff_1,
            (
                rbf_coeff_1_size_0,
                rbf_coeff_1_size_1,
            ),
            on_gpu,
            False,
        )

        rbf_coeff_2 = (
            rbf_coeff_2,
            (
                rbf_coeff_2_size_0,
                rbf_coeff_2_size_1,
            ),
            on_gpu,
            False,
        )

        geofac_div = (
            geofac_div,
            (
                geofac_div_size_0,
                geofac_div_size_1,
            ),
            on_gpu,
            False,
        )

        geofac_n2s = (
            geofac_n2s,
            (
                geofac_n2s_size_0,
                geofac_n2s_size_1,
            ),
            on_gpu,
            False,
        )

        geofac_grg_x = (
            geofac_grg_x,
            (
                geofac_grg_x_size_0,
                geofac_grg_x_size_1,
            ),
            on_gpu,
            False,
        )

        geofac_grg_y = (
            geofac_grg_y,
            (
                geofac_grg_y_size_0,
                geofac_grg_y_size_1,
            ),
            on_gpu,
            False,
        )

        nudgecoeff_e = (nudgecoeff_e, (nudgecoeff_e_size_0,), on_gpu, False)

        bdy_halo_c = (bdy_halo_c, (bdy_halo_c_size_0,), on_gpu, False)

        mask_prog_halo_c = (mask_prog_halo_c, (mask_prog_halo_c_size_0,), on_gpu, False)

        rayleigh_w = (rayleigh_w, (rayleigh_w_size_0,), on_gpu, False)

        exner_exfac = (
            exner_exfac,
            (
                exner_exfac_size_0,
                exner_exfac_size_1,
            ),
            on_gpu,
            False,
        )

        exner_ref_mc = (
            exner_ref_mc,
            (
                exner_ref_mc_size_0,
                exner_ref_mc_size_1,
            ),
            on_gpu,
            False,
        )

        wgtfac_c = (
            wgtfac_c,
            (
                wgtfac_c_size_0,
                wgtfac_c_size_1,
            ),
            on_gpu,
            False,
        )

        wgtfacq_c = (
            wgtfacq_c,
            (
                wgtfacq_c_size_0,
                wgtfacq_c_size_1,
            ),
            on_gpu,
            False,
        )

        inv_ddqz_z_full = (
            inv_ddqz_z_full,
            (
                inv_ddqz_z_full_size_0,
                inv_ddqz_z_full_size_1,
            ),
            on_gpu,
            False,
        )

        rho_ref_mc = (
            rho_ref_mc,
            (
                rho_ref_mc_size_0,
                rho_ref_mc_size_1,
            ),
            on_gpu,
            False,
        )

        theta_ref_mc = (
            theta_ref_mc,
            (
                theta_ref_mc_size_0,
                theta_ref_mc_size_1,
            ),
            on_gpu,
            False,
        )

        vwind_expl_wgt = (vwind_expl_wgt, (vwind_expl_wgt_size_0,), on_gpu, False)

        d_exner_dz_ref_ic = (
            d_exner_dz_ref_ic,
            (
                d_exner_dz_ref_ic_size_0,
                d_exner_dz_ref_ic_size_1,
            ),
            on_gpu,
            False,
        )

        ddqz_z_half = (
            ddqz_z_half,
            (
                ddqz_z_half_size_0,
                ddqz_z_half_size_1,
            ),
            on_gpu,
            False,
        )

        theta_ref_ic = (
            theta_ref_ic,
            (
                theta_ref_ic_size_0,
                theta_ref_ic_size_1,
            ),
            on_gpu,
            False,
        )

        d2dexdz2_fac1_mc = (
            d2dexdz2_fac1_mc,
            (
                d2dexdz2_fac1_mc_size_0,
                d2dexdz2_fac1_mc_size_1,
            ),
            on_gpu,
            False,
        )

        d2dexdz2_fac2_mc = (
            d2dexdz2_fac2_mc,
            (
                d2dexdz2_fac2_mc_size_0,
                d2dexdz2_fac2_mc_size_1,
            ),
            on_gpu,
            False,
        )

        rho_ref_me = (
            rho_ref_me,
            (
                rho_ref_me_size_0,
                rho_ref_me_size_1,
            ),
            on_gpu,
            False,
        )

        theta_ref_me = (
            theta_ref_me,
            (
                theta_ref_me_size_0,
                theta_ref_me_size_1,
            ),
            on_gpu,
            False,
        )

        ddxn_z_full = (
            ddxn_z_full,
            (
                ddxn_z_full_size_0,
                ddxn_z_full_size_1,
            ),
            on_gpu,
            False,
        )

        zdiff_gradp = (
            zdiff_gradp,
            (
                zdiff_gradp_size_0,
                zdiff_gradp_size_1,
                zdiff_gradp_size_2,
            ),
            on_gpu,
            False,
        )

        vertoffset_gradp = (
            vertoffset_gradp,
            (
                vertoffset_gradp_size_0,
                vertoffset_gradp_size_1,
                vertoffset_gradp_size_2,
            ),
            on_gpu,
            False,
        )

        ipeidx_dsl = (
            ipeidx_dsl,
            (
                ipeidx_dsl_size_0,
                ipeidx_dsl_size_1,
            ),
            on_gpu,
            False,
        )

        pg_exdist = (
            pg_exdist,
            (
                pg_exdist_size_0,
                pg_exdist_size_1,
            ),
            on_gpu,
            False,
        )

        ddqz_z_full_e = (
            ddqz_z_full_e,
            (
                ddqz_z_full_e_size_0,
                ddqz_z_full_e_size_1,
            ),
            on_gpu,
            False,
        )

        ddxt_z_full = (
            ddxt_z_full,
            (
                ddxt_z_full_size_0,
                ddxt_z_full_size_1,
            ),
            on_gpu,
            False,
        )

        wgtfac_e = (
            wgtfac_e,
            (
                wgtfac_e_size_0,
                wgtfac_e_size_1,
            ),
            on_gpu,
            False,
        )

        wgtfacq_e = (
            wgtfacq_e,
            (
                wgtfacq_e_size_0,
                wgtfacq_e_size_1,
            ),
            on_gpu,
            False,
        )

        vwind_impl_wgt = (vwind_impl_wgt, (vwind_impl_wgt_size_0,), on_gpu, False)

        hmask_dd3d = (hmask_dd3d, (hmask_dd3d_size_0,), on_gpu, False)

        scalfac_dd3d = (scalfac_dd3d, (scalfac_dd3d_size_0,), on_gpu, False)

        coeff1_dwdz = (
            coeff1_dwdz,
            (
                coeff1_dwdz_size_0,
                coeff1_dwdz_size_1,
            ),
            on_gpu,
            False,
        )

        coeff2_dwdz = (
            coeff2_dwdz,
            (
                coeff2_dwdz_size_0,
                coeff2_dwdz_size_1,
            ),
            on_gpu,
            False,
        )

        coeff_gradekin = (
            coeff_gradekin,
            (
                coeff_gradekin_size_0,
                coeff_gradekin_size_1,
            ),
            on_gpu,
            False,
        )

        c_owner_mask = (c_owner_mask, (c_owner_mask_size_0,), on_gpu, False)

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "solve_nh_init constructing `ArrayInfos` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            perf_counters = {}
        else:
            perf_counters = None
        solve_nh_init(
            ffi=ffi,
            perf_counters=perf_counters,
            vct_a=vct_a,
            vct_b=vct_b,
            c_lin_e=c_lin_e,
            c_intp=c_intp,
            e_flx_avg=e_flx_avg,
            geofac_grdiv=geofac_grdiv,
            geofac_rot=geofac_rot,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            e_bln_c_s=e_bln_c_s,
            rbf_coeff_1=rbf_coeff_1,
            rbf_coeff_2=rbf_coeff_2,
            geofac_div=geofac_div,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            nudgecoeff_e=nudgecoeff_e,
            bdy_halo_c=bdy_halo_c,
            mask_prog_halo_c=mask_prog_halo_c,
            rayleigh_w=rayleigh_w,
            exner_exfac=exner_exfac,
            exner_ref_mc=exner_ref_mc,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ref_mc=rho_ref_mc,
            theta_ref_mc=theta_ref_mc,
            vwind_expl_wgt=vwind_expl_wgt,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            theta_ref_ic=theta_ref_ic,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            ddxn_z_full=ddxn_z_full,
            zdiff_gradp=zdiff_gradp,
            vertoffset_gradp=vertoffset_gradp,
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            ddqz_z_full_e=ddqz_z_full_e,
            ddxt_z_full=ddxt_z_full,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            vwind_impl_wgt=vwind_impl_wgt,
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            coeff_gradekin=coeff_gradekin,
            c_owner_mask=c_owner_mask,
            rayleigh_damping_height=rayleigh_damping_height,
            itime_scheme=itime_scheme,
            iadv_rhotheta=iadv_rhotheta,
            igradp_method=igradp_method,
            rayleigh_type=rayleigh_type,
            rayleigh_coeff=rayleigh_coeff,
            divdamp_order=divdamp_order,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
            divdamp_type=divdamp_type,
            divdamp_trans_start=divdamp_trans_start,
            divdamp_trans_end=divdamp_trans_end,
            l_vert_nested=l_vert_nested,
            rhotheta_offctr=rhotheta_offctr,
            veladv_offctr=veladv_offctr,
            nudge_max_coeff=nudge_max_coeff,
            divdamp_fac=divdamp_fac,
            divdamp_fac2=divdamp_fac2,
            divdamp_fac3=divdamp_fac3,
            divdamp_fac4=divdamp_fac4,
            divdamp_z=divdamp_z,
            divdamp_z2=divdamp_z2,
            divdamp_z3=divdamp_z3,
            divdamp_z4=divdamp_z4,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            nflat_gradp=nflat_gradp,
            num_levels=num_levels,
            backend=backend,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "solve_nh_init convert time: %s"
                    % str(perf_counters["convert_end_time"] - perf_counters["convert_start_time"])
                )
                logger.info(
                    "solve_nh_init execution time: %s" % str(func_end_time - func_start_time)
                )

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                vct_a_arr = (
                    _conversion.as_array(ffi, vct_a, _definitions.FLOAT64)
                    if vct_a is not None
                    else None
                )
                msg = "shape of vct_a after computation = %s" % str(
                    vct_a_arr.shape if vct_a is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vct_a after computation: %s" % str(vct_a_arr) if vct_a is not None else "None"
                )
                logger.debug(msg)

                vct_b_arr = (
                    _conversion.as_array(ffi, vct_b, _definitions.FLOAT64)
                    if vct_b is not None
                    else None
                )
                msg = "shape of vct_b after computation = %s" % str(
                    vct_b_arr.shape if vct_b is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vct_b after computation: %s" % str(vct_b_arr) if vct_b is not None else "None"
                )
                logger.debug(msg)

                c_lin_e_arr = (
                    _conversion.as_array(ffi, c_lin_e, _definitions.FLOAT64)
                    if c_lin_e is not None
                    else None
                )
                msg = "shape of c_lin_e after computation = %s" % str(
                    c_lin_e_arr.shape if c_lin_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "c_lin_e after computation: %s" % str(c_lin_e_arr)
                    if c_lin_e is not None
                    else "None"
                )
                logger.debug(msg)

                c_intp_arr = (
                    _conversion.as_array(ffi, c_intp, _definitions.FLOAT64)
                    if c_intp is not None
                    else None
                )
                msg = "shape of c_intp after computation = %s" % str(
                    c_intp_arr.shape if c_intp is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "c_intp after computation: %s" % str(c_intp_arr)
                    if c_intp is not None
                    else "None"
                )
                logger.debug(msg)

                e_flx_avg_arr = (
                    _conversion.as_array(ffi, e_flx_avg, _definitions.FLOAT64)
                    if e_flx_avg is not None
                    else None
                )
                msg = "shape of e_flx_avg after computation = %s" % str(
                    e_flx_avg_arr.shape if e_flx_avg is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "e_flx_avg after computation: %s" % str(e_flx_avg_arr)
                    if e_flx_avg is not None
                    else "None"
                )
                logger.debug(msg)

                geofac_grdiv_arr = (
                    _conversion.as_array(ffi, geofac_grdiv, _definitions.FLOAT64)
                    if geofac_grdiv is not None
                    else None
                )
                msg = "shape of geofac_grdiv after computation = %s" % str(
                    geofac_grdiv_arr.shape if geofac_grdiv is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "geofac_grdiv after computation: %s" % str(geofac_grdiv_arr)
                    if geofac_grdiv is not None
                    else "None"
                )
                logger.debug(msg)

                geofac_rot_arr = (
                    _conversion.as_array(ffi, geofac_rot, _definitions.FLOAT64)
                    if geofac_rot is not None
                    else None
                )
                msg = "shape of geofac_rot after computation = %s" % str(
                    geofac_rot_arr.shape if geofac_rot is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "geofac_rot after computation: %s" % str(geofac_rot_arr)
                    if geofac_rot is not None
                    else "None"
                )
                logger.debug(msg)

                pos_on_tplane_e_1_arr = (
                    _conversion.as_array(ffi, pos_on_tplane_e_1, _definitions.FLOAT64)
                    if pos_on_tplane_e_1 is not None
                    else None
                )
                msg = "shape of pos_on_tplane_e_1 after computation = %s" % str(
                    pos_on_tplane_e_1_arr.shape if pos_on_tplane_e_1 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "pos_on_tplane_e_1 after computation: %s" % str(pos_on_tplane_e_1_arr)
                    if pos_on_tplane_e_1 is not None
                    else "None"
                )
                logger.debug(msg)

                pos_on_tplane_e_2_arr = (
                    _conversion.as_array(ffi, pos_on_tplane_e_2, _definitions.FLOAT64)
                    if pos_on_tplane_e_2 is not None
                    else None
                )
                msg = "shape of pos_on_tplane_e_2 after computation = %s" % str(
                    pos_on_tplane_e_2_arr.shape if pos_on_tplane_e_2 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "pos_on_tplane_e_2 after computation: %s" % str(pos_on_tplane_e_2_arr)
                    if pos_on_tplane_e_2 is not None
                    else "None"
                )
                logger.debug(msg)

                rbf_vec_coeff_e_arr = (
                    _conversion.as_array(ffi, rbf_vec_coeff_e, _definitions.FLOAT64)
                    if rbf_vec_coeff_e is not None
                    else None
                )
                msg = "shape of rbf_vec_coeff_e after computation = %s" % str(
                    rbf_vec_coeff_e_arr.shape if rbf_vec_coeff_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rbf_vec_coeff_e after computation: %s" % str(rbf_vec_coeff_e_arr)
                    if rbf_vec_coeff_e is not None
                    else "None"
                )
                logger.debug(msg)

                e_bln_c_s_arr = (
                    _conversion.as_array(ffi, e_bln_c_s, _definitions.FLOAT64)
                    if e_bln_c_s is not None
                    else None
                )
                msg = "shape of e_bln_c_s after computation = %s" % str(
                    e_bln_c_s_arr.shape if e_bln_c_s is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "e_bln_c_s after computation: %s" % str(e_bln_c_s_arr)
                    if e_bln_c_s is not None
                    else "None"
                )
                logger.debug(msg)

                rbf_coeff_1_arr = (
                    _conversion.as_array(ffi, rbf_coeff_1, _definitions.FLOAT64)
                    if rbf_coeff_1 is not None
                    else None
                )
                msg = "shape of rbf_coeff_1 after computation = %s" % str(
                    rbf_coeff_1_arr.shape if rbf_coeff_1 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rbf_coeff_1 after computation: %s" % str(rbf_coeff_1_arr)
                    if rbf_coeff_1 is not None
                    else "None"
                )
                logger.debug(msg)

                rbf_coeff_2_arr = (
                    _conversion.as_array(ffi, rbf_coeff_2, _definitions.FLOAT64)
                    if rbf_coeff_2 is not None
                    else None
                )
                msg = "shape of rbf_coeff_2 after computation = %s" % str(
                    rbf_coeff_2_arr.shape if rbf_coeff_2 is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rbf_coeff_2 after computation: %s" % str(rbf_coeff_2_arr)
                    if rbf_coeff_2 is not None
                    else "None"
                )
                logger.debug(msg)

                geofac_div_arr = (
                    _conversion.as_array(ffi, geofac_div, _definitions.FLOAT64)
                    if geofac_div is not None
                    else None
                )
                msg = "shape of geofac_div after computation = %s" % str(
                    geofac_div_arr.shape if geofac_div is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "geofac_div after computation: %s" % str(geofac_div_arr)
                    if geofac_div is not None
                    else "None"
                )
                logger.debug(msg)

                geofac_n2s_arr = (
                    _conversion.as_array(ffi, geofac_n2s, _definitions.FLOAT64)
                    if geofac_n2s is not None
                    else None
                )
                msg = "shape of geofac_n2s after computation = %s" % str(
                    geofac_n2s_arr.shape if geofac_n2s is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "geofac_n2s after computation: %s" % str(geofac_n2s_arr)
                    if geofac_n2s is not None
                    else "None"
                )
                logger.debug(msg)

                geofac_grg_x_arr = (
                    _conversion.as_array(ffi, geofac_grg_x, _definitions.FLOAT64)
                    if geofac_grg_x is not None
                    else None
                )
                msg = "shape of geofac_grg_x after computation = %s" % str(
                    geofac_grg_x_arr.shape if geofac_grg_x is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "geofac_grg_x after computation: %s" % str(geofac_grg_x_arr)
                    if geofac_grg_x is not None
                    else "None"
                )
                logger.debug(msg)

                geofac_grg_y_arr = (
                    _conversion.as_array(ffi, geofac_grg_y, _definitions.FLOAT64)
                    if geofac_grg_y is not None
                    else None
                )
                msg = "shape of geofac_grg_y after computation = %s" % str(
                    geofac_grg_y_arr.shape if geofac_grg_y is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "geofac_grg_y after computation: %s" % str(geofac_grg_y_arr)
                    if geofac_grg_y is not None
                    else "None"
                )
                logger.debug(msg)

                nudgecoeff_e_arr = (
                    _conversion.as_array(ffi, nudgecoeff_e, _definitions.FLOAT64)
                    if nudgecoeff_e is not None
                    else None
                )
                msg = "shape of nudgecoeff_e after computation = %s" % str(
                    nudgecoeff_e_arr.shape if nudgecoeff_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "nudgecoeff_e after computation: %s" % str(nudgecoeff_e_arr)
                    if nudgecoeff_e is not None
                    else "None"
                )
                logger.debug(msg)

                bdy_halo_c_arr = (
                    _conversion.as_array(ffi, bdy_halo_c, _definitions.BOOL)
                    if bdy_halo_c is not None
                    else None
                )
                msg = "shape of bdy_halo_c after computation = %s" % str(
                    bdy_halo_c_arr.shape if bdy_halo_c is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "bdy_halo_c after computation: %s" % str(bdy_halo_c_arr)
                    if bdy_halo_c is not None
                    else "None"
                )
                logger.debug(msg)

                mask_prog_halo_c_arr = (
                    _conversion.as_array(ffi, mask_prog_halo_c, _definitions.BOOL)
                    if mask_prog_halo_c is not None
                    else None
                )
                msg = "shape of mask_prog_halo_c after computation = %s" % str(
                    mask_prog_halo_c_arr.shape if mask_prog_halo_c is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "mask_prog_halo_c after computation: %s" % str(mask_prog_halo_c_arr)
                    if mask_prog_halo_c is not None
                    else "None"
                )
                logger.debug(msg)

                rayleigh_w_arr = (
                    _conversion.as_array(ffi, rayleigh_w, _definitions.FLOAT64)
                    if rayleigh_w is not None
                    else None
                )
                msg = "shape of rayleigh_w after computation = %s" % str(
                    rayleigh_w_arr.shape if rayleigh_w is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rayleigh_w after computation: %s" % str(rayleigh_w_arr)
                    if rayleigh_w is not None
                    else "None"
                )
                logger.debug(msg)

                exner_exfac_arr = (
                    _conversion.as_array(ffi, exner_exfac, _definitions.FLOAT64)
                    if exner_exfac is not None
                    else None
                )
                msg = "shape of exner_exfac after computation = %s" % str(
                    exner_exfac_arr.shape if exner_exfac is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_exfac after computation: %s" % str(exner_exfac_arr)
                    if exner_exfac is not None
                    else "None"
                )
                logger.debug(msg)

                exner_ref_mc_arr = (
                    _conversion.as_array(ffi, exner_ref_mc, _definitions.FLOAT64)
                    if exner_ref_mc is not None
                    else None
                )
                msg = "shape of exner_ref_mc after computation = %s" % str(
                    exner_ref_mc_arr.shape if exner_ref_mc is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "exner_ref_mc after computation: %s" % str(exner_ref_mc_arr)
                    if exner_ref_mc is not None
                    else "None"
                )
                logger.debug(msg)

                wgtfac_c_arr = (
                    _conversion.as_array(ffi, wgtfac_c, _definitions.FLOAT64)
                    if wgtfac_c is not None
                    else None
                )
                msg = "shape of wgtfac_c after computation = %s" % str(
                    wgtfac_c_arr.shape if wgtfac_c is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "wgtfac_c after computation: %s" % str(wgtfac_c_arr)
                    if wgtfac_c is not None
                    else "None"
                )
                logger.debug(msg)

                wgtfacq_c_arr = (
                    _conversion.as_array(ffi, wgtfacq_c, _definitions.FLOAT64)
                    if wgtfacq_c is not None
                    else None
                )
                msg = "shape of wgtfacq_c after computation = %s" % str(
                    wgtfacq_c_arr.shape if wgtfacq_c is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "wgtfacq_c after computation: %s" % str(wgtfacq_c_arr)
                    if wgtfacq_c is not None
                    else "None"
                )
                logger.debug(msg)

                inv_ddqz_z_full_arr = (
                    _conversion.as_array(ffi, inv_ddqz_z_full, _definitions.FLOAT64)
                    if inv_ddqz_z_full is not None
                    else None
                )
                msg = "shape of inv_ddqz_z_full after computation = %s" % str(
                    inv_ddqz_z_full_arr.shape if inv_ddqz_z_full is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "inv_ddqz_z_full after computation: %s" % str(inv_ddqz_z_full_arr)
                    if inv_ddqz_z_full is not None
                    else "None"
                )
                logger.debug(msg)

                rho_ref_mc_arr = (
                    _conversion.as_array(ffi, rho_ref_mc, _definitions.FLOAT64)
                    if rho_ref_mc is not None
                    else None
                )
                msg = "shape of rho_ref_mc after computation = %s" % str(
                    rho_ref_mc_arr.shape if rho_ref_mc is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rho_ref_mc after computation: %s" % str(rho_ref_mc_arr)
                    if rho_ref_mc is not None
                    else "None"
                )
                logger.debug(msg)

                theta_ref_mc_arr = (
                    _conversion.as_array(ffi, theta_ref_mc, _definitions.FLOAT64)
                    if theta_ref_mc is not None
                    else None
                )
                msg = "shape of theta_ref_mc after computation = %s" % str(
                    theta_ref_mc_arr.shape if theta_ref_mc is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "theta_ref_mc after computation: %s" % str(theta_ref_mc_arr)
                    if theta_ref_mc is not None
                    else "None"
                )
                logger.debug(msg)

                vwind_expl_wgt_arr = (
                    _conversion.as_array(ffi, vwind_expl_wgt, _definitions.FLOAT64)
                    if vwind_expl_wgt is not None
                    else None
                )
                msg = "shape of vwind_expl_wgt after computation = %s" % str(
                    vwind_expl_wgt_arr.shape if vwind_expl_wgt is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vwind_expl_wgt after computation: %s" % str(vwind_expl_wgt_arr)
                    if vwind_expl_wgt is not None
                    else "None"
                )
                logger.debug(msg)

                d_exner_dz_ref_ic_arr = (
                    _conversion.as_array(ffi, d_exner_dz_ref_ic, _definitions.FLOAT64)
                    if d_exner_dz_ref_ic is not None
                    else None
                )
                msg = "shape of d_exner_dz_ref_ic after computation = %s" % str(
                    d_exner_dz_ref_ic_arr.shape if d_exner_dz_ref_ic is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "d_exner_dz_ref_ic after computation: %s" % str(d_exner_dz_ref_ic_arr)
                    if d_exner_dz_ref_ic is not None
                    else "None"
                )
                logger.debug(msg)

                ddqz_z_half_arr = (
                    _conversion.as_array(ffi, ddqz_z_half, _definitions.FLOAT64)
                    if ddqz_z_half is not None
                    else None
                )
                msg = "shape of ddqz_z_half after computation = %s" % str(
                    ddqz_z_half_arr.shape if ddqz_z_half is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddqz_z_half after computation: %s" % str(ddqz_z_half_arr)
                    if ddqz_z_half is not None
                    else "None"
                )
                logger.debug(msg)

                theta_ref_ic_arr = (
                    _conversion.as_array(ffi, theta_ref_ic, _definitions.FLOAT64)
                    if theta_ref_ic is not None
                    else None
                )
                msg = "shape of theta_ref_ic after computation = %s" % str(
                    theta_ref_ic_arr.shape if theta_ref_ic is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "theta_ref_ic after computation: %s" % str(theta_ref_ic_arr)
                    if theta_ref_ic is not None
                    else "None"
                )
                logger.debug(msg)

                d2dexdz2_fac1_mc_arr = (
                    _conversion.as_array(ffi, d2dexdz2_fac1_mc, _definitions.FLOAT64)
                    if d2dexdz2_fac1_mc is not None
                    else None
                )
                msg = "shape of d2dexdz2_fac1_mc after computation = %s" % str(
                    d2dexdz2_fac1_mc_arr.shape if d2dexdz2_fac1_mc is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "d2dexdz2_fac1_mc after computation: %s" % str(d2dexdz2_fac1_mc_arr)
                    if d2dexdz2_fac1_mc is not None
                    else "None"
                )
                logger.debug(msg)

                d2dexdz2_fac2_mc_arr = (
                    _conversion.as_array(ffi, d2dexdz2_fac2_mc, _definitions.FLOAT64)
                    if d2dexdz2_fac2_mc is not None
                    else None
                )
                msg = "shape of d2dexdz2_fac2_mc after computation = %s" % str(
                    d2dexdz2_fac2_mc_arr.shape if d2dexdz2_fac2_mc is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "d2dexdz2_fac2_mc after computation: %s" % str(d2dexdz2_fac2_mc_arr)
                    if d2dexdz2_fac2_mc is not None
                    else "None"
                )
                logger.debug(msg)

                rho_ref_me_arr = (
                    _conversion.as_array(ffi, rho_ref_me, _definitions.FLOAT64)
                    if rho_ref_me is not None
                    else None
                )
                msg = "shape of rho_ref_me after computation = %s" % str(
                    rho_ref_me_arr.shape if rho_ref_me is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "rho_ref_me after computation: %s" % str(rho_ref_me_arr)
                    if rho_ref_me is not None
                    else "None"
                )
                logger.debug(msg)

                theta_ref_me_arr = (
                    _conversion.as_array(ffi, theta_ref_me, _definitions.FLOAT64)
                    if theta_ref_me is not None
                    else None
                )
                msg = "shape of theta_ref_me after computation = %s" % str(
                    theta_ref_me_arr.shape if theta_ref_me is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "theta_ref_me after computation: %s" % str(theta_ref_me_arr)
                    if theta_ref_me is not None
                    else "None"
                )
                logger.debug(msg)

                ddxn_z_full_arr = (
                    _conversion.as_array(ffi, ddxn_z_full, _definitions.FLOAT64)
                    if ddxn_z_full is not None
                    else None
                )
                msg = "shape of ddxn_z_full after computation = %s" % str(
                    ddxn_z_full_arr.shape if ddxn_z_full is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddxn_z_full after computation: %s" % str(ddxn_z_full_arr)
                    if ddxn_z_full is not None
                    else "None"
                )
                logger.debug(msg)

                zdiff_gradp_arr = (
                    _conversion.as_array(ffi, zdiff_gradp, _definitions.FLOAT64)
                    if zdiff_gradp is not None
                    else None
                )
                msg = "shape of zdiff_gradp after computation = %s" % str(
                    zdiff_gradp_arr.shape if zdiff_gradp is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "zdiff_gradp after computation: %s" % str(zdiff_gradp_arr)
                    if zdiff_gradp is not None
                    else "None"
                )
                logger.debug(msg)

                vertoffset_gradp_arr = (
                    _conversion.as_array(ffi, vertoffset_gradp, _definitions.INT32)
                    if vertoffset_gradp is not None
                    else None
                )
                msg = "shape of vertoffset_gradp after computation = %s" % str(
                    vertoffset_gradp_arr.shape if vertoffset_gradp is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vertoffset_gradp after computation: %s" % str(vertoffset_gradp_arr)
                    if vertoffset_gradp is not None
                    else "None"
                )
                logger.debug(msg)

                ipeidx_dsl_arr = (
                    _conversion.as_array(ffi, ipeidx_dsl, _definitions.BOOL)
                    if ipeidx_dsl is not None
                    else None
                )
                msg = "shape of ipeidx_dsl after computation = %s" % str(
                    ipeidx_dsl_arr.shape if ipeidx_dsl is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ipeidx_dsl after computation: %s" % str(ipeidx_dsl_arr)
                    if ipeidx_dsl is not None
                    else "None"
                )
                logger.debug(msg)

                pg_exdist_arr = (
                    _conversion.as_array(ffi, pg_exdist, _definitions.FLOAT64)
                    if pg_exdist is not None
                    else None
                )
                msg = "shape of pg_exdist after computation = %s" % str(
                    pg_exdist_arr.shape if pg_exdist is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "pg_exdist after computation: %s" % str(pg_exdist_arr)
                    if pg_exdist is not None
                    else "None"
                )
                logger.debug(msg)

                ddqz_z_full_e_arr = (
                    _conversion.as_array(ffi, ddqz_z_full_e, _definitions.FLOAT64)
                    if ddqz_z_full_e is not None
                    else None
                )
                msg = "shape of ddqz_z_full_e after computation = %s" % str(
                    ddqz_z_full_e_arr.shape if ddqz_z_full_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddqz_z_full_e after computation: %s" % str(ddqz_z_full_e_arr)
                    if ddqz_z_full_e is not None
                    else "None"
                )
                logger.debug(msg)

                ddxt_z_full_arr = (
                    _conversion.as_array(ffi, ddxt_z_full, _definitions.FLOAT64)
                    if ddxt_z_full is not None
                    else None
                )
                msg = "shape of ddxt_z_full after computation = %s" % str(
                    ddxt_z_full_arr.shape if ddxt_z_full is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "ddxt_z_full after computation: %s" % str(ddxt_z_full_arr)
                    if ddxt_z_full is not None
                    else "None"
                )
                logger.debug(msg)

                wgtfac_e_arr = (
                    _conversion.as_array(ffi, wgtfac_e, _definitions.FLOAT64)
                    if wgtfac_e is not None
                    else None
                )
                msg = "shape of wgtfac_e after computation = %s" % str(
                    wgtfac_e_arr.shape if wgtfac_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "wgtfac_e after computation: %s" % str(wgtfac_e_arr)
                    if wgtfac_e is not None
                    else "None"
                )
                logger.debug(msg)

                wgtfacq_e_arr = (
                    _conversion.as_array(ffi, wgtfacq_e, _definitions.FLOAT64)
                    if wgtfacq_e is not None
                    else None
                )
                msg = "shape of wgtfacq_e after computation = %s" % str(
                    wgtfacq_e_arr.shape if wgtfacq_e is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "wgtfacq_e after computation: %s" % str(wgtfacq_e_arr)
                    if wgtfacq_e is not None
                    else "None"
                )
                logger.debug(msg)

                vwind_impl_wgt_arr = (
                    _conversion.as_array(ffi, vwind_impl_wgt, _definitions.FLOAT64)
                    if vwind_impl_wgt is not None
                    else None
                )
                msg = "shape of vwind_impl_wgt after computation = %s" % str(
                    vwind_impl_wgt_arr.shape if vwind_impl_wgt is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "vwind_impl_wgt after computation: %s" % str(vwind_impl_wgt_arr)
                    if vwind_impl_wgt is not None
                    else "None"
                )
                logger.debug(msg)

                hmask_dd3d_arr = (
                    _conversion.as_array(ffi, hmask_dd3d, _definitions.FLOAT64)
                    if hmask_dd3d is not None
                    else None
                )
                msg = "shape of hmask_dd3d after computation = %s" % str(
                    hmask_dd3d_arr.shape if hmask_dd3d is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "hmask_dd3d after computation: %s" % str(hmask_dd3d_arr)
                    if hmask_dd3d is not None
                    else "None"
                )
                logger.debug(msg)

                scalfac_dd3d_arr = (
                    _conversion.as_array(ffi, scalfac_dd3d, _definitions.FLOAT64)
                    if scalfac_dd3d is not None
                    else None
                )
                msg = "shape of scalfac_dd3d after computation = %s" % str(
                    scalfac_dd3d_arr.shape if scalfac_dd3d is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "scalfac_dd3d after computation: %s" % str(scalfac_dd3d_arr)
                    if scalfac_dd3d is not None
                    else "None"
                )
                logger.debug(msg)

                coeff1_dwdz_arr = (
                    _conversion.as_array(ffi, coeff1_dwdz, _definitions.FLOAT64)
                    if coeff1_dwdz is not None
                    else None
                )
                msg = "shape of coeff1_dwdz after computation = %s" % str(
                    coeff1_dwdz_arr.shape if coeff1_dwdz is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "coeff1_dwdz after computation: %s" % str(coeff1_dwdz_arr)
                    if coeff1_dwdz is not None
                    else "None"
                )
                logger.debug(msg)

                coeff2_dwdz_arr = (
                    _conversion.as_array(ffi, coeff2_dwdz, _definitions.FLOAT64)
                    if coeff2_dwdz is not None
                    else None
                )
                msg = "shape of coeff2_dwdz after computation = %s" % str(
                    coeff2_dwdz_arr.shape if coeff2_dwdz is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "coeff2_dwdz after computation: %s" % str(coeff2_dwdz_arr)
                    if coeff2_dwdz is not None
                    else "None"
                )
                logger.debug(msg)

                coeff_gradekin_arr = (
                    _conversion.as_array(ffi, coeff_gradekin, _definitions.FLOAT64)
                    if coeff_gradekin is not None
                    else None
                )
                msg = "shape of coeff_gradekin after computation = %s" % str(
                    coeff_gradekin_arr.shape if coeff_gradekin is not None else "None"
                )
                logger.debug(msg)
                msg = (
                    "coeff_gradekin after computation: %s" % str(coeff_gradekin_arr)
                    if coeff_gradekin is not None
                    else "None"
                )
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

        if __debug__:
            logger.info("Python execution of solve_nh_init completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0
