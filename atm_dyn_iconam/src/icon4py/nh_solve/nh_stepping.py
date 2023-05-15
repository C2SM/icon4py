
class NonhydroStepping:

    def __init__(self):


    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state: MetricState,
        metric_state_nonhydro: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        a_vec: Field[[KDim], float],
        enh_smag_fac: Field[[KDim], float],
        fac: tuple,
        z: tuple,
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """


    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):

    def perform_nh_stepping(self):

    def _perform_nh_timeloop(self):

    def _integrate_nh(self, num_steps):
        # if jg > 1:
        #     jgp = p_patch(jg).parent_id
        if n_dom_start == 0:
            jgp = 0
        else:
            jgp = 1

        if jg == 1 and l_limited_area and linit_dyn(jg):
            n_save = nsav2(jg)
            n_now = nnow(jg)

        for jstep in range(num_steps):
            # if ifeedback_type == 1 and jstep == 1 and jg > 1:
            #     n_now = nnow(jg)
            #     n_save = nsav2(jg)

            jstep_adv(jg).marchuk_order = jstep_adv(jg).marchuk_order + 1

            # if p_patch(jg).n_childdom > 0 and ndyn_substeps_var(jg) > 1:
            #     lbdy_nudging = False
            #     lnest_active = False
            #     for jn in range(p_patch(jg).n_childdom):
            #         jgc = p_patch(jg).child_id(jn)
            #         if p_patch(jgc).ldom_active:
            #             lnest_active = True
            #             lbdy_nudging = True if not lfeedback(jgc) else False
            #
            #     n_now = nnow(jg)
            #     n_save = nsav1(jg)
            #
            #     if lnest_active:
            #         # save_progvars(jg, p_nh_state(jg) % prog(n_now), p_nh_state(jg) % prog(n_save))

            n_now_rcf = nnow_rcf(jg)
            n_new_rcf = nnew_rcf(jg)

            # messy calls and stuff

            if itime_scheme == 1:
                prepare_tracer()

                # airmass_now
                compute_airmass()
                # airmass_new
                compute_airmass()

                step_advection()
            else:
                ## WHAT IS THIS?????
                # IF(iforcing == iheldsuarez) THEN
                #     CALL held_suarez_nh_interface(p_nh_state(jg) % prog(nnow(jg)), p_patch(jg), &
                #     p_int_state(jg), p_nh_state(jg) % metrics, &
                #     p_nh_state(jg) % diag)
                # ENDIF

                init_ddt_vn_diagnostics(p_nh_state(jg).diag)

                if ldynamics and not ltestcase and linit_dyn(jg) and diffusion_config(jg).lhdiff_vn and init_mode not in (MODE_IAU, MODE_IAU_OLD):
                    diffusion()

                if ldynamics:
                    perform_dyn_substepping()

                    if diffusion_config(jg).lhdiff_vn and lhdiff_rcf:
                        diffusion()

                if ltransport:
                    step_advection()

                    if iprog_aero >= 1:
                        sync_patch_array()
                        aerosol_2D_advection()
                        sync_patch_array()
                        aerosol_2D_diffusion()

                # if messy
                main_tracer_afteradv()

                ## WHAT IS THIS???
                if iforcing == iaes:
                    interface_iconam_aes()

                    jn = 1
                    # for jn in range(p_patch(jg).n_childdom):
                    jgc = p_patch(jg).child_id(jn)

                    ## WHAT IS THIS??
                    # IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

                    if lredgrid_phys(jgc):
                        if patch_weight(jgc) > 0:
                            # lcall_rrg = isNextTriggerTimeInRange()
                        else:
                            # lcall_rrg = isNextTriggerTimeInRange()
                    else:
                        # lcall_rrg = True

                    if lcall_rrg:
                        interpol_rrg_grf()

                    if lcall_rrg and atm_phy_nwp_config(jgc).latm_above_top
                        copy_rrg_ubc()
                # messy and testing interface stuff
            if l_limited_area and not l_global_nudging:
                tsrat = float(ndyn_substeps, wp)

                if latbc_config.itype_latbc > 0:
                    if latbc_config.nudge_hydro_pres: sync_patch_array_mult()

                if num_prefetch_proc >= 1:
                    latbc.update_intp_wgt(datetime_local(jg).ptr)

                    if jg == 1:
                        limarea_nudging_latbdy()

                    if nudging_config(jg).ltype(indg_type.ubn):
                        if jg == 1:
                            ptr_latbc_data_atm_old = latbc.latbc_data(latbc.prev_latbc_tlev()).atm
                            ptr_latbc_data_atm_new = latbc.latbc_data(latbc.new_latbc_tlev).atm
                        else:
                            ptr_latbc_data_atm_old = latbc.latbc_data(latbc.prev_latbc_tlev()).atm_child(jg)
                            ptr_latbc_data_atm_new = latbc.latbc_data(latbc.new_latbc_tlev).atm_child(jg)

                    limarea_nudging_upbdy()
                else:
                    if jg == 1:
                        limarea_nudging_latbdy()

            elif l_global_nudging and jg==1:
                nudging_interface()

            # if p_patch(jg).n_childdom > 0:
            #     lnest_active = False
            #
            #     for jn in range(p_patch(jg).n_childdom):
            #         jgc = p_patch(jg).child_id(jn)
            #         if p_patch(jgc).ldom_active:
            #             lnest_active = True

            # if p_patch(jg).n_childdom > 0 and lnest_active:
            #     if ndyn_substeps_var(jg) == 1:
            #         n_now_grf = nnow(jg)
            #     else:
            #         n_now_grf = nsav1(jg)
            #
            #     rdt_loc = 1 / dt_loc
            #     dt_sub = dt_loc / 2
            #     rdtmflx_loc = 1 / (dt_loc * (float(max(1, ndyn_substeps_var(jg) - 1), wp) / float(ndyn_substeps_var(jg), wp)))
            #
            #     compute_tendencies(jg, nnew(jg), n_now_grf, n_new_rcf, n_now_rcf, rdt_loc, rdtmflx_loc)
            #
            #     for jn in range(p_patch(jg).n_childdom):
            #         jgc = p_patch(jg).child_id(jn)
            #
            #         if p_patch(jgc).ldom_active:
            #             boundary_interpolation()
            #
            #     for jn in range(p_patch(jg).n_childdom):
            #         jgc = p_patch(jg).child_id(jn)
            #
            #         if lfeedback(jgc) and l_density_nudging and grf_intmethod_e <= 4:
            #             prep_rho_bdy_nudging(jg, jgc)
            #         elif not lfeedback(jgc):
            #             prep_bdy_nudging(jg, jgc)
            #
            #     for jn in range(p_patch(jg).n_childdom):
            #         jgc = p_patch(jg).child_id(jn)
            #
            #         if p_patch(jgc).domain_is_owned:
            #             if proc_split:
            #                 push_glob_comm(p_patch(jgc).comm, p_patch(jgc).proc0)
            #             integrate_nh(datetime_local, jgc, nstep_global, iau_iter, dt_sub, mtime_dt_sub, nsteps_nest, latbc)
            #             if proc_split: pop_glob_comm()

                # feedback loop, do we need this???
            if jg == 1 and is_avgFG_time(datetime_local(jg).ptr):
                average_first_guess()

            if test_mode <= 0:
                swap(nnow(jg), nnew(jg))
                swap(nnow_rcf(jg), nnew_rcf(jg))

            # if p_patch(jg).n_childdom > 0:
            #
            #     for jn in range(p_patch(jg).n_childdom):
            #         jgc = p_patch(jg).child_id(jn)
            #
            #         if not p_patch(jgc).ldom_active and start_time(jgc) <= sim_time < end_time(jgc):
            #             p_patch(jgc).ldom_active = True
            #             gpu_d2h_nh_nwp(p_patch(jg), prm_diag(jg), ext_data=ext_data(jg))
            #             i_am_accel_node = False
            #             jstep_adv(jgc).marchuk_order = 0
            #             datetime_local(jgc).ptr = datetime_local(jg).ptr
            #             linit_dyn(jgc) = True
            #             dt_sub = dt_loc / 2
            #
            #             if atm_phy_nwp_config(jgc).inwp_surface == 1:
            #                 aggregate_landvars(p_patch(jg), ext_data(jg), p_lnd_state(jg).prog_lnd(nnow_rcf(jg)), p_lnd_state(jg).diag_lnd)
            #
            #             initialize_nest(jg, jgc)
            #             hydro_adjust_const_thetav()
            #             init_exner_pr(jgc, nnow(jgc), use_acc= False)
            #             init_mode_soil = 1
            #
            #              if iforcing == inwp:
            #                 init_nwp_phy()
            #                 init_cloud_aero_cpl(datetime_local(jgc).ptr, p_patch(jgc), p_nh_state(jgc).metrics, ext_data(jgc), prm_diag(jgc))
            #
            #                 if iprog_aero >= 1: setup_aerosol_advection(p_patch(jgc))
            #
            #         compute_airmass(p_patch=p_patch(jgc), p_metrics = p_nh_state(jgc).metrics, rho = p_nh_state(jgc).prog(nnow(jgc)).rho, airmass = p_nh_state(jgc).diag.airmass_new)
            #
            #         if lredgrid_phys(jgc):
            #             interpol_rrg_grf(jg, jgc, jn, nnow_rcf(jg))
            #              if atm_phy_nwp_config(jgc).latm_above_top:
            #                 copy_rrg_ubc(jg, jgc)
            #
            #         init_slowphysics(datetime_local(jgc).ptr, jgc, dt_sub, lacc=False)
            #         gpu_h2d_nh_nwp(p_patch(jg), prm_diag(jg), ext_data=ext_data(jg))
            #         gpu_h2d_nh_nwp(p_patch(jgc), prm_diag(jgc), ext_data=ext_data(jgc), phy_params=phy_params(jgc), atm_phy_nwp_config=atm_phy_nwp_config(jg))
            #         i_am_accel_node = my_process_is_work()

        # messy stuff


    def _perform_dyn_substepping(self):

        if idiv_method == 1 and (ltransport or n_childdom > 0 and grf_intmethod_e >= 5):
            lprep_adv = True
        else:
            lprep_adv = False

        for nstep in range(ndyn_substeps_var):

            ndyn_substeps_tot = (jstep - 1) * ndyn_substeps_var(jg) + nstep

            if nstep == 1:
                lclean_mflx=True
            lrecompute = lclean_mflx
            if nstep == ndyn_substeps_var(jg):
                llast = True

            if n_childdom > 0 and nstep == 1:
                lsave_mflx = True

            if not ldeepatmo:
                SolveNonhydro.time_step()

            linit_dyn(jg) = False

            if diffusion_config(jg).lhdiff_vn and not lhdiff_rcf:
                diffusion.time_step()

            # compute tracer prep
            # compute swap

        lready_for_checkpoint = True
        if (init_mode not in (mo_impl_constants.MODE_IAU, mo_impl_constants.MODE_IAU_OLD)) and (cur_time <= dt_iau):
            lready_for_checkpoint = False

        compute_airmass()


