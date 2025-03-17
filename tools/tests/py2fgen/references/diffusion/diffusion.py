import logging
from diffusion import ffi
from icon4py.tools.py2fgen import utils, runtime_config, _runtime, _definitions

if __debug__:
    logger = logging.getLogger(__name__)
    log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, runtime_config.LOG_LEVEL),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# embedded function imports
from icon4py.tools.py2fgen.wrappers.diffusion_wrapper import diffusion_run
from icon4py.tools.py2fgen.wrappers.diffusion_wrapper import diffusion_init


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
    on_gpu,
):
    try:
        if __debug__:
            logger.info("Python execution of diffusion_run started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayDescriptors

        w = (
            w,
            (
                w_size_0,
                w_size_1,
            ),
            on_gpu,
            False,
        )

        vn = (
            vn,
            (
                vn_size_0,
                vn_size_1,
            ),
            on_gpu,
            False,
        )

        exner = (
            exner,
            (
                exner_size_0,
                exner_size_1,
            ),
            on_gpu,
            False,
        )

        theta_v = (
            theta_v,
            (
                theta_v_size_0,
                theta_v_size_1,
            ),
            on_gpu,
            False,
        )

        rho = (
            rho,
            (
                rho_size_0,
                rho_size_1,
            ),
            on_gpu,
            False,
        )

        hdef_ic = (
            hdef_ic,
            (
                hdef_ic_size_0,
                hdef_ic_size_1,
            ),
            on_gpu,
            True,
        )

        div_ic = (
            div_ic,
            (
                div_ic_size_0,
                div_ic_size_1,
            ),
            on_gpu,
            True,
        )

        dwdx = (
            dwdx,
            (
                dwdx_size_0,
                dwdx_size_1,
            ),
            on_gpu,
            True,
        )

        dwdy = (
            dwdy,
            (
                dwdy_size_0,
                dwdy_size_1,
            ),
            on_gpu,
            True,
        )

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "diffusion_run constructing `ArrayDescriptors` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            meta = {}
        else:
            meta = None
        diffusion_run(
            ffi=ffi,
            meta=meta,
            w=w,
            vn=vn,
            exner=exner,
            theta_v=theta_v,
            rho=rho,
            hdef_ic=hdef_ic,
            div_ic=div_ic,
            dwdx=dwdx,
            dwdy=dwdy,
            dtime=dtime,
            linit=linit,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "diffusion_run convert time: %s"
                    % str(meta["convert_end_time"] - meta["convert_start_time"])
                )
                logger.info(
                    "diffusion_run execution time: %s" % str(func_end_time - func_start_time)
                )

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                msg = "shape of w after computation = %s" % str(
                    w.shape if w is not None else "None"
                )
                logger.debug(msg)
                msg = "w after computation: %s" % str(
                    utils.as_array(ffi, w, _definitions.FLOAT64) if w is not None else "None"
                )
                logger.debug(msg)

                msg = "shape of vn after computation = %s" % str(
                    vn.shape if vn is not None else "None"
                )
                logger.debug(msg)
                msg = "vn after computation: %s" % str(
                    utils.as_array(ffi, vn, _definitions.FLOAT64) if vn is not None else "None"
                )
                logger.debug(msg)

                msg = "shape of exner after computation = %s" % str(
                    exner.shape if exner is not None else "None"
                )
                logger.debug(msg)
                msg = "exner after computation: %s" % str(
                    utils.as_array(ffi, exner, _definitions.FLOAT64)
                    if exner is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of theta_v after computation = %s" % str(
                    theta_v.shape if theta_v is not None else "None"
                )
                logger.debug(msg)
                msg = "theta_v after computation: %s" % str(
                    utils.as_array(ffi, theta_v, _definitions.FLOAT64)
                    if theta_v is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of rho after computation = %s" % str(
                    rho.shape if rho is not None else "None"
                )
                logger.debug(msg)
                msg = "rho after computation: %s" % str(
                    utils.as_array(ffi, rho, _definitions.FLOAT64) if rho is not None else "None"
                )
                logger.debug(msg)

                msg = "shape of hdef_ic after computation = %s" % str(
                    hdef_ic.shape if hdef_ic is not None else "None"
                )
                logger.debug(msg)
                msg = "hdef_ic after computation: %s" % str(
                    utils.as_array(ffi, hdef_ic, _definitions.FLOAT64)
                    if hdef_ic is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of div_ic after computation = %s" % str(
                    div_ic.shape if div_ic is not None else "None"
                )
                logger.debug(msg)
                msg = "div_ic after computation: %s" % str(
                    utils.as_array(ffi, div_ic, _definitions.FLOAT64)
                    if div_ic is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of dwdx after computation = %s" % str(
                    dwdx.shape if dwdx is not None else "None"
                )
                logger.debug(msg)
                msg = "dwdx after computation: %s" % str(
                    utils.as_array(ffi, dwdx, _definitions.FLOAT64) if dwdx is not None else "None"
                )
                logger.debug(msg)

                msg = "shape of dwdy after computation = %s" % str(
                    dwdy.shape if dwdy is not None else "None"
                )
                logger.debug(msg)
                msg = "dwdy after computation: %s" % str(
                    utils.as_array(ffi, dwdy, _definitions.FLOAT64) if dwdy is not None else "None"
                )
                logger.debug(msg)

        if __debug__:
            logger.info("Python execution of diffusion_run completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
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
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    backend,
    on_gpu,
):
    try:
        if __debug__:
            logger.info("Python execution of diffusion_init started.")

        if __debug__:
            if runtime_config.PROFILING:
                unpack_start_time = _runtime.perf_counter()

        # ArrayDescriptors

        vct_a = (vct_a, (vct_a_size_0,), on_gpu, False)

        vct_b = (vct_b, (vct_b_size_0,), on_gpu, False)

        theta_ref_mc = (
            theta_ref_mc,
            (
                theta_ref_mc_size_0,
                theta_ref_mc_size_1,
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

        e_bln_c_s = (
            e_bln_c_s,
            (
                e_bln_c_s_size_0,
                e_bln_c_s_size_1,
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

        geofac_n2s = (
            geofac_n2s,
            (
                geofac_n2s_size_0,
                geofac_n2s_size_1,
            ),
            on_gpu,
            False,
        )

        nudgecoeff_e = (nudgecoeff_e, (nudgecoeff_e_size_0,), on_gpu, False)

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

        mask_hdiff = (
            mask_hdiff,
            (
                mask_hdiff_size_0,
                mask_hdiff_size_1,
            ),
            on_gpu,
            True,
        )

        zd_diffcoef = (
            zd_diffcoef,
            (
                zd_diffcoef_size_0,
                zd_diffcoef_size_1,
            ),
            on_gpu,
            True,
        )

        zd_vertoffset = (
            zd_vertoffset,
            (
                zd_vertoffset_size_0,
                zd_vertoffset_size_1,
                zd_vertoffset_size_2,
            ),
            on_gpu,
            True,
        )

        zd_intcoef = (
            zd_intcoef,
            (
                zd_intcoef_size_0,
                zd_intcoef_size_1,
                zd_intcoef_size_2,
            ),
            on_gpu,
            True,
        )

        if __debug__:
            if runtime_config.PROFILING:
                allocate_end_time = _runtime.perf_counter()
                logger.info(
                    "diffusion_init constructing `ArrayDescriptors` time: %s"
                    % str(allocate_end_time - unpack_start_time)
                )

                func_start_time = _runtime.perf_counter()

        if __debug__ and runtime_config.PROFILING:
            meta = {}
        else:
            meta = None
        diffusion_init(
            ffi=ffi,
            meta=meta,
            vct_a=vct_a,
            vct_b=vct_b,
            theta_ref_mc=theta_ref_mc,
            wgtfac_c=wgtfac_c,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            geofac_n2s=geofac_n2s,
            nudgecoeff_e=nudgecoeff_e,
            rbf_coeff_1=rbf_coeff_1,
            rbf_coeff_2=rbf_coeff_2,
            mask_hdiff=mask_hdiff,
            zd_diffcoef=zd_diffcoef,
            zd_vertoffset=zd_vertoffset,
            zd_intcoef=zd_intcoef,
            ndyn_substeps=ndyn_substeps,
            rayleigh_damping_height=rayleigh_damping_height,
            nflat_gradp=nflat_gradp,
            diffusion_type=diffusion_type,
            hdiff_w=hdiff_w,
            hdiff_vn=hdiff_vn,
            zdiffu_t=zdiffu_t,
            type_t_diffu=type_t_diffu,
            type_vn_diffu=type_vn_diffu,
            hdiff_efdt_ratio=hdiff_efdt_ratio,
            smagorinski_scaling_factor=smagorinski_scaling_factor,
            hdiff_temp=hdiff_temp,
            thslp_zdiffu=thslp_zdiffu,
            thhgtd_zdiffu=thhgtd_zdiffu,
            denom_diffu_v=denom_diffu_v,
            nudge_max_coeff=nudge_max_coeff,
            itype_sher=itype_sher,
            ltkeshs=ltkeshs,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            backend=backend,
        )

        if __debug__:
            if runtime_config.PROFILING:
                func_end_time = _runtime.perf_counter()
                logger.info(
                    "diffusion_init convert time: %s"
                    % str(meta["convert_end_time"] - meta["convert_start_time"])
                )
                logger.info(
                    "diffusion_init execution time: %s" % str(func_end_time - func_start_time)
                )

        if __debug__:
            if logger.isEnabledFor(logging.DEBUG):

                msg = "shape of vct_a after computation = %s" % str(
                    vct_a.shape if vct_a is not None else "None"
                )
                logger.debug(msg)
                msg = "vct_a after computation: %s" % str(
                    utils.as_array(ffi, vct_a, _definitions.FLOAT64)
                    if vct_a is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of vct_b after computation = %s" % str(
                    vct_b.shape if vct_b is not None else "None"
                )
                logger.debug(msg)
                msg = "vct_b after computation: %s" % str(
                    utils.as_array(ffi, vct_b, _definitions.FLOAT64)
                    if vct_b is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of theta_ref_mc after computation = %s" % str(
                    theta_ref_mc.shape if theta_ref_mc is not None else "None"
                )
                logger.debug(msg)
                msg = "theta_ref_mc after computation: %s" % str(
                    utils.as_array(ffi, theta_ref_mc, _definitions.FLOAT64)
                    if theta_ref_mc is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of wgtfac_c after computation = %s" % str(
                    wgtfac_c.shape if wgtfac_c is not None else "None"
                )
                logger.debug(msg)
                msg = "wgtfac_c after computation: %s" % str(
                    utils.as_array(ffi, wgtfac_c, _definitions.FLOAT64)
                    if wgtfac_c is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of e_bln_c_s after computation = %s" % str(
                    e_bln_c_s.shape if e_bln_c_s is not None else "None"
                )
                logger.debug(msg)
                msg = "e_bln_c_s after computation: %s" % str(
                    utils.as_array(ffi, e_bln_c_s, _definitions.FLOAT64)
                    if e_bln_c_s is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of geofac_div after computation = %s" % str(
                    geofac_div.shape if geofac_div is not None else "None"
                )
                logger.debug(msg)
                msg = "geofac_div after computation: %s" % str(
                    utils.as_array(ffi, geofac_div, _definitions.FLOAT64)
                    if geofac_div is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of geofac_grg_x after computation = %s" % str(
                    geofac_grg_x.shape if geofac_grg_x is not None else "None"
                )
                logger.debug(msg)
                msg = "geofac_grg_x after computation: %s" % str(
                    utils.as_array(ffi, geofac_grg_x, _definitions.FLOAT64)
                    if geofac_grg_x is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of geofac_grg_y after computation = %s" % str(
                    geofac_grg_y.shape if geofac_grg_y is not None else "None"
                )
                logger.debug(msg)
                msg = "geofac_grg_y after computation: %s" % str(
                    utils.as_array(ffi, geofac_grg_y, _definitions.FLOAT64)
                    if geofac_grg_y is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of geofac_n2s after computation = %s" % str(
                    geofac_n2s.shape if geofac_n2s is not None else "None"
                )
                logger.debug(msg)
                msg = "geofac_n2s after computation: %s" % str(
                    utils.as_array(ffi, geofac_n2s, _definitions.FLOAT64)
                    if geofac_n2s is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of nudgecoeff_e after computation = %s" % str(
                    nudgecoeff_e.shape if nudgecoeff_e is not None else "None"
                )
                logger.debug(msg)
                msg = "nudgecoeff_e after computation: %s" % str(
                    utils.as_array(ffi, nudgecoeff_e, _definitions.FLOAT64)
                    if nudgecoeff_e is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of rbf_coeff_1 after computation = %s" % str(
                    rbf_coeff_1.shape if rbf_coeff_1 is not None else "None"
                )
                logger.debug(msg)
                msg = "rbf_coeff_1 after computation: %s" % str(
                    utils.as_array(ffi, rbf_coeff_1, _definitions.FLOAT64)
                    if rbf_coeff_1 is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of rbf_coeff_2 after computation = %s" % str(
                    rbf_coeff_2.shape if rbf_coeff_2 is not None else "None"
                )
                logger.debug(msg)
                msg = "rbf_coeff_2 after computation: %s" % str(
                    utils.as_array(ffi, rbf_coeff_2, _definitions.FLOAT64)
                    if rbf_coeff_2 is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of mask_hdiff after computation = %s" % str(
                    mask_hdiff.shape if mask_hdiff is not None else "None"
                )
                logger.debug(msg)
                msg = "mask_hdiff after computation: %s" % str(
                    utils.as_array(ffi, mask_hdiff, _definitions.BOOL)
                    if mask_hdiff is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of zd_diffcoef after computation = %s" % str(
                    zd_diffcoef.shape if zd_diffcoef is not None else "None"
                )
                logger.debug(msg)
                msg = "zd_diffcoef after computation: %s" % str(
                    utils.as_array(ffi, zd_diffcoef, _definitions.FLOAT64)
                    if zd_diffcoef is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of zd_vertoffset after computation = %s" % str(
                    zd_vertoffset.shape if zd_vertoffset is not None else "None"
                )
                logger.debug(msg)
                msg = "zd_vertoffset after computation: %s" % str(
                    utils.as_array(ffi, zd_vertoffset, _definitions.INT32)
                    if zd_vertoffset is not None
                    else "None"
                )
                logger.debug(msg)

                msg = "shape of zd_intcoef after computation = %s" % str(
                    zd_intcoef.shape if zd_intcoef is not None else "None"
                )
                logger.debug(msg)
                msg = "zd_intcoef after computation: %s" % str(
                    utils.as_array(ffi, zd_intcoef, _definitions.FLOAT64)
                    if zd_intcoef is not None
                    else "None"
                )
                logger.debug(msg)

        if __debug__:
            logger.info("Python execution of diffusion_init completed.")

    except Exception as e:
        logger.exception(f"A Python error occurred: {e}")
        return 1

    return 0
