!>
!! Initializes and controls the time stepping in the nonhydrostatic model.
!!
!!
!! @par Revision History
!! Initial release by Almut Gassmann, MPI-M (27009-02-06)
!!
!! @par Copyright and License
!!
!! This code is subject to the DWD and MPI-M-Software-License-Agreement in
!! its most recent form.
!! Please see the file LICENSE in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the
!! headers of the routines.
!!
!! The time stepping does eventually perform an IAU with the follwing
!! characteristics:
!!
!! IAU iteration
!!
!!                     input
!!                       /
!!                      /
!!                     /
!!          ........../
!!         /
!!        /
!!       /
!!      /
!!     /
!!  -90min               0min              90min
!! ---|------------------|------------------|------------->
!!    |//////////////////| - - - - - - - - - - - - - - - ->
!!                       /       free forecast (iteration = false)
!!                      /
!!                     /
!!          ........../
!!         /   reset
!!        /
!!       /
!!      /
!!     /
!!  -90min               0min              90min
!! ---|------------------|------------------|------------->
!!    |//////////////////|//////////////////| free forecast
!!
!!    \_______IAU________/
!!
!----------------------------
#include "omp_definitions.inc"
!----------------------------

MODULE mo_nh_stepping
!-------------------------------------------------------------------------
!
!    ProTeX FORTRAN source: Style 2
!    modified for ICON project, DWD/MPI-M 2006
!
!-------------------------------------------------------------------------
!
!

  USE mo_kind,                     ONLY: wp, vp
  USE mo_io_units
  USE mo_nonhydro_state,           ONLY: p_nh_state, p_nh_state_lists
  USE mo_nonhydrostatic_config,    ONLY: lhdiff_rcf, itime_scheme, divdamp_order,                     &
    &                                    divdamp_fac, divdamp_fac_o2, ih_clch, ih_clcm, kstart_moist, &
    &                                    ndyn_substeps, ndyn_substeps_var, ndyn_substeps_max
  USE mo_diffusion_config,         ONLY: diffusion_config
  USE mo_dynamics_config,          ONLY: nnow,nnew, nnow_rcf, nnew_rcf, nsav1, nsav2, idiv_method, &
    &                                    ldeepatmo
  USE mo_io_config,                ONLY: is_totint_time, n_diag, var_in_output, checkpoint_on_demand
  USE mo_parallel_config,          ONLY: nproma, itype_comm, num_prefetch_proc, proc0_offloading
  USE mo_run_config,               ONLY: ltestcase, dtime, nsteps, ldynamics, ltransport,   &
    &                                    ntracer, iforcing, msg_level, test_mode,           &
    &                                    output_mode, lart, luse_radarfwo, ldass_lhn
  USE mo_advection_config,         ONLY: advection_config
  USE mo_timer,                    ONLY: ltimer, timers_level, timer_start, timer_stop,   &
    &                                    timer_total, timer_model_init, timer_nudging,    &
    &                                    timer_bdy_interp, timer_feedback, timer_nesting, &
    &                                    timer_integrate_nh, timer_nh_diagnostics,        &
    &                                    timer_iconam_aes, timer_dace_coupling
  USE mo_atm_phy_nwp_config,       ONLY: dt_phy, atm_phy_nwp_config, iprog_aero, setup_nwp_diag_events
  USE mo_ensemble_pert_config,     ONLY: compute_ensemble_pert, use_ensemble_pert
  USE mo_nwp_phy_init,             ONLY: init_nwp_phy, init_cloud_aero_cpl
  USE mo_nwp_phy_state,            ONLY: prm_diag, prm_nwp_tend, phy_params, prm_nwp_stochconv
  USE mo_lnd_nwp_config,           ONLY: nlev_soil, nlev_snow, sstice_mode, sst_td_filename, &
    &                                    ci_td_filename, frsi_min
  USE mo_nwp_lnd_state,            ONLY: p_lnd_state
  USE mo_ext_data_state,           ONLY: ext_data
  USE mo_limarea_config,           ONLY: latbc_config
  USE mo_model_domain,             ONLY: p_patch, t_patch, p_patch_local_parent
  USE mo_time_config,              ONLY: time_config
  USE mo_grid_config,              ONLY: n_dom, lfeedback, ifeedback_type, l_limited_area, &
    &                                    n_dom_start, lredgrid_phys, start_time, end_time, patch_weight
  USE mo_gribout_config,           ONLY: gribout_config
  USE mo_nh_testcases_nml,         ONLY: is_toy_chem, ltestcase_update
  USE mo_nh_dcmip_terminator,      ONLY: dcmip_terminator_interface
  USE mo_nh_supervise,             ONLY: supervise_total_integrals_nh, print_maxwinds,  &
    &                                    init_supervise_nh, finalize_supervise_nh
  USE mo_intp_data_strc,           ONLY: p_int_state, t_int_state, p_int_state_local_parent
  USE mo_intp_rbf,                 ONLY: rbf_vec_interpol_cell
  USE mo_intp,                     ONLY: verts2cells_scalar
  USE mo_grf_intp_data_strc,       ONLY: p_grf_state, p_grf_state_local_parent
  USE mo_gridref_config,           ONLY: l_density_nudging, grf_intmethod_e
  USE mo_grf_bdyintp,              ONLY: interpol_scal_grf
  USE mo_nh_nest_utilities,        ONLY: compute_tendencies, boundary_interpolation,    &
                                         prep_bdy_nudging, nest_boundary_nudging,       &
                                         prep_rho_bdy_nudging, density_boundary_nudging,&
                                         limarea_nudging_latbdy,                        &
                                         limarea_nudging_upbdy, save_progvars
  USE mo_nh_feedback,              ONLY: feedback, relax_feedback, lhn_feedback
  USE mo_exception,                ONLY: message, message_text, finish
  USE mo_impl_constants,           ONLY: SUCCESS, inoforcing, iheldsuarez, inwp, iaes,         &
    &                                    MODE_IAU, MODE_IAU_OLD, SSTICE_CLIM,                  &
    &                                    MODE_IFSANA,MODE_COMBINED,MODE_COSMO,MODE_ICONVREMAP, &
    &                                    SSTICE_AVG_MONTHLY, SSTICE_AVG_DAILY, SSTICE_INST,    &
    &                                    max_dom, min_rlcell, min_rlvert, ismag, iprog
  USE mo_math_divrot,              ONLY: rot_vertex, div_avg !, div
  USE mo_solve_nonhydro,           ONLY: solve_nh
  USE mo_update_dyn_scm,           ONLY: add_slowphys_scm
  USE mo_advection_stepping,       ONLY: step_advection
  USE mo_advection_aerosols,       ONLY: aerosol_2D_advection, setup_aerosol_advection
  USE mo_aerosol_util,             ONLY: aerosol_2D_diffusion
  USE mo_nh_dtp_interface,         ONLY: prepare_tracer, compute_airmass
  USE mo_nh_diffusion,             ONLY: diffusion
  USE mo_memory_log,               ONLY: memory_log_add
  USE mo_mpi,                      ONLY: proc_split, push_glob_comm, pop_glob_comm, &
       &                                 p_comm_work, my_process_is_mpi_workroot,   &
       &                                 my_process_is_mpi_test, my_process_is_work_only, i_am_accel_node
#ifdef HAVE_RADARFWO
  USE mo_emvorado_interface,       ONLY: emvorado_radarfwo
#endif
#ifdef NOMPI
  USE mo_mpi,                      ONLY: my_process_is_mpi_all_seq
#endif

  USE mo_sync,                     ONLY: sync_patch_array_mult, sync_patch_array, SYNC_C, SYNC_E, global_max
  USE mo_nh_interface_nwp,         ONLY: nwp_nh_interface
#ifndef __NO_AES__
  USE mo_interface_iconam_aes,     ONLY: interface_iconam_aes
  USE mo_aes_phy_memory,           ONLY: prm_tend
#endif
  USE mo_phys_nest_utilities,      ONLY: interpol_phys_grf, feedback_phys_diag, interpol_rrg_grf, copy_rrg_ubc
  USE mo_nh_diagnose_pres_temp,    ONLY: diagnose_pres_temp
  USE mo_nh_held_suarez_interface, ONLY: held_suarez_nh_interface
  USE mo_master_config,            ONLY: isRestart, getModelBaseDir
  USE mo_restart_nml_and_att,      ONLY: getAttributesForRestarting
  USE mo_key_value_store,          ONLY: t_key_value_store
  USE mo_meteogram_config,         ONLY: meteogram_output_config
  USE mo_meteogram_output,         ONLY: meteogram_sample_vars, meteogram_is_sample_step
  USE mo_name_list_output,         ONLY: write_name_list_output, istime4name_list_output, istime4name_list_output_dom
  USE mo_name_list_output_init,    ONLY: output_file
  USE mo_pp_scheduler,             ONLY: new_simulation_status, pp_scheduler_process
  USE mo_pp_tasks,                 ONLY: t_simulation_status

  USE mo_nwp_sfc_utils,            ONLY: aggregate_landvars, process_sst_and_seaice
  USE mo_reader_sst_sic,           ONLY: t_sst_sic_reader
  USE mo_interpolate_time,         ONLY: t_time_intp
  USE mo_nh_init_nest_utils,       ONLY: initialize_nest
  USE mo_nh_init_utils,            ONLY: compute_iau_wgt, save_initial_state, restore_initial_state
  USE mo_hydro_adjust,             ONLY: hydro_adjust_const_thetav
  USE mo_td_ext_data,              ONLY: update_nwp_phy_bcs, set_sst_and_seaice
  USE mo_initicon_types,           ONLY: t_pi_atm
  USE mo_initicon_config,          ONLY: init_mode, timeshift, init_mode_soil, iterate_iau, dt_iau
  USE mo_synsat_config,            ONLY: lsynsat
  USE mo_rttov_interface,          ONLY: rttov_driver, copy_rttov_ubc
#ifndef __NO_ICON_LES__
  USE mo_les_config,               ONLY: les_config
  USE mo_turbulent_diagnostic,     ONLY: calculate_turbulent_diagnostics, &
                                         write_vertical_profiles, write_time_series, &
                                         les_cloud_diag
#endif
  USE mo_restart,                  ONLY: t_RestartDescriptor, createRestartDescriptor, deleteRestartDescriptor
  USE mo_restart_util,             ONLY: check_for_checkpoint
  USE mo_prepadv_types,            ONLY: t_prepare_adv
  USE mo_prepadv_state,            ONLY: prep_adv, jstep_adv
  USE mo_action,                   ONLY: reset_act
  USE mo_output_event_handler,     ONLY: get_current_jfile
  USE mo_nwp_diagnosis,            ONLY: nwp_diag_for_output, nwp_opt_diagnostics
  USE mo_opt_diagnostics,          ONLY: update_opt_acc, reset_opt_acc, &
    &                                    calc_mean_opt_acc, p_nh_opt_diag
  USE mo_var_list_register_utils,  ONLY: vlr_print_vls
  USE mo_async_latbc_utils,        ONLY: recv_latbc_data
  USE mo_async_latbc_types,        ONLY: t_latbc_data
  USE mo_nonhydro_types,           ONLY: t_nh_state, t_nh_diag
  USE mo_fortran_tools,            ONLY: swap, copy, init
  USE mtime,                       ONLY: datetime, newDatetime, deallocateDatetime, datetimeToString,     &
       &                                 timedelta, newTimedelta, deallocateTimedelta, timedeltaToString, &
       &                                 MAX_DATETIME_STR_LEN, MAX_TIMEDELTA_STR_LEN, newDatetime,        &
       &                                 MAX_MTIME_ERROR_STR_LEN, no_error, mtime_strerror,               &
       &                                 OPERATOR(-), OPERATOR(+), OPERATOR(>), OPERATOR(*),              &
       &                                 ASSIGNMENT(=), OPERATOR(==), OPERATOR(>=), OPERATOR(/=),         &
       &                                 event, eventGroup, newEvent,                                     &
       &                                 addEventToEventGroup,                                            &
       &                                 getTotalSecondsTimedelta, getTimedeltaFromDatetime
  USE mo_util_mtime,               ONLY: mtime_utils, assumePrevMidnight, FMT_DDHHMMSS_DAYSEP, &
    &                                    getElapsedSimTimeInSeconds, is_event_active
  USE mo_event_manager,            ONLY: addEventGroup, getEventGroup, printEventGroup
  USE mo_phy_events,               ONLY: mtime_ctrl_physics
  USE mo_derived_variable_handling, ONLY: update_statistics, statistics_active_on_dom
#ifdef MESSY
  USE messy_main_channel_bi,       ONLY: messy_channel_write_output &
    &                                  , IOMODE_RST
  USE messy_main_tracer_bi,        ONLY: main_tracer_beforeadv, main_tracer_afteradv
#ifdef MESSYTIMER
  USE messy_main_timer_bi,         ONLY: messy_timer_reset_time

#endif
#endif

  USE mo_radar_data_state,         ONLY: lhn_fields
  USE mo_assimilation_config,      ONLY: assimilation_config

#if defined( _OPENACC )
  USE mo_nonhydro_gpu_types,       ONLY: h2d_icon, d2h_icon, devcpy_grf_state
  USE mo_mpi,                      ONLY: my_process_is_work
  USE mo_acc_device_management,    ONLY: printGPUMem
#endif
  USE mo_loopindices,              ONLY: get_indices_c, get_indices_v
  USE mo_nh_testcase_interface,    ONLY: nh_testcase_interface
  USE mo_upatmo_config,            ONLY: upatmo_config
  USE mo_nh_deepatmo_solve,        ONLY: solve_nh_deepatmo
  USE mo_upatmo_impl_const,        ONLY: idamtr, iUpatmoPrcStat
#ifndef __NO_ICON_UPATMO__
  USE mo_upatmo_state,             ONLY: prm_upatmo
  USE mo_upatmo_flowevent_utils,   ONLY: t_upatmoRestartAttributes,      &
    &                                    upatmoRestartAttributesPrepare, &
    &                                    upatmoRestartAttributesGet,     &
    &                                    upatmoRestartAttributesDeallocate
#endif
  use mo_icon2dace,                ONLY: mec_Event, init_dace_op, run_dace_op, dace_op_init
  USE mo_extpar_config,            ONLY: generate_td_filename
  USE mo_nudging_config,           ONLY: nudging_config, l_global_nudging, indg_type
  USE mo_nudging,                  ONLY: nudging_interface
  USE mo_opt_nwp_diagnostics,      ONLY: compute_field_dbz3d_lin
  USE mo_nwp_gpu_util,             ONLY: gpu_d2h_nh_nwp, gpu_h2d_nh_nwp, devcpy_nwp, hostcpy_nwp
  USE mo_nwp_diagnosis,            ONLY: nwp_diag_global

  IMPLICIT NONE

  PRIVATE

  !> module name string
  CHARACTER(LEN=*), PARAMETER :: modname = 'mo_nh_stepping'


  ! additional flow control variables that need to be dimensioned with the
  ! number of model domains
  LOGICAL, ALLOCATABLE :: linit_dyn(:)  ! determines whether dynamics must be initialized
                                        ! on given patch

  LOGICAL :: lready_for_checkpoint = .FALSE.
  ! event handling manager, wrong place, have to move later

  TYPE(eventGroup), POINTER :: checkpointEventGroup => NULL()

  PUBLIC :: perform_nh_stepping

  TYPE(t_sst_sic_reader), ALLOCATABLE, TARGET :: sst_reader(:)
  TYPE(t_sst_sic_reader), ALLOCATABLE, TARGET :: sic_reader(:)
  TYPE(t_time_intp),      ALLOCATABLE         :: sst_intp(:)
  TYPE(t_time_intp),      ALLOCATABLE         :: sic_intp(:)
  REAL(wp),               ALLOCATABLE         :: sst_dat(:,:,:,:)
  REAL(wp),               ALLOCATABLE         :: sic_dat(:,:,:,:)

  TYPE t_datetime_ptr
    TYPE(datetime), POINTER :: ptr => NULL()
  END TYPE t_datetime_ptr

  CONTAINS

  !-------------------------------------------------------------------------
  !>
  !! Organizes nonhydrostatic time stepping
  !! Currently we assume to have only one grid level.
  SUBROUTINE perform_nh_stepping (mtime_current, latbc)
    !
    TYPE(datetime),     POINTER       :: mtime_current     !< current datetime (mtime)
    TYPE(t_latbc_data), INTENT(INOUT) :: latbc             !< data structure for async latbc prefetching

  TYPE(t_simulation_status)            :: simulation_status

  CHARACTER(len=*), PARAMETER ::  &
    &  routine = modname//':perform_nh_stepping'
  CHARACTER(filename_max) :: sst_td_file !< file name for reading in
  CHARACTER(filename_max) :: ci_td_file

  INTEGER                              :: jg, jgc, jn
  INTEGER                              :: month, year
  LOGICAL                              :: is_mpi_workroot
  LOGICAL                              :: l_exist
  is_mpi_workroot = my_process_is_mpi_workroot()


!!$  INTEGER omp_get_num_threads
!!$  INTEGER omp_get_max_threads
!!$  INTEGER omp_get_max_active_levels
!-----------------------------------------------------------------------

#if defined(MESSY) && defined(_OPENACC)
   CALL finish (routine, 'MESSY:  OpenACC version currently not implemented')
#endif

  CALL allocate_nh_stepping (mtime_current)


  ! Compute diagnostic dynamics fields for initial output and physics initialization
  CALL diag_for_output_dyn ()


  ! diagnose airmass from \rho(now) for both restart and non-restart runs
  ! airmass_new required by initial physics call (i.e. by radheat in init_slowphysics)
  ! airmass_now not needed, since ddt_temp_dyn is not computed during the
  ! initial slow physics call.
  DO jg=1, n_dom
    CALL compute_airmass(p_patch   = p_patch(jg),                       & !in
      &                  p_metrics = p_nh_state(jg)%metrics,            & !in
      &                  rho       = p_nh_state(jg)%prog(nnow(jg))%rho, & !in
      &                  airmass   = p_nh_state(jg)%diag%airmass_new    ) !inout


    ! initialize exner_pr if the model domain is active
    IF (p_patch(jg)%ldom_active .AND. .NOT. isRestart()) CALL init_exner_pr(jg, nnow(jg), use_acc=.FALSE.)
  ENDDO




  ! Initialize time-dependent ensemble perturbations if necessary
  IF (use_ensemble_pert .AND. gribout_config(1)%perturbationNumber >= 1) THEN
    CALL compute_ensemble_pert(p_patch(1:), ext_data, prm_diag, phy_params, mtime_current, .FALSE.)
  ENDIF



#if defined( _OPENACC )
    ! initialize GPU for NWP and AES
    i_am_accel_node = my_process_is_work()    ! Activate GPUs
    IF (i_am_accel_node) THEN
      CALL h2d_icon( p_int_state, p_int_state_local_parent, p_patch, p_patch_local_parent, &
      &            p_nh_state, prep_adv, advection_config, iforcing, lacc=.TRUE. )
      IF (n_dom > 1 .OR. l_limited_area) THEN
        CALL devcpy_grf_state (p_grf_state, .TRUE., lacc=.TRUE.)
        CALL devcpy_grf_state (p_grf_state_local_parent, .TRUE., lacc=.TRUE.)
      ELSEIF (ANY(lredgrid_phys)) THEN
        CALL devcpy_grf_state (p_grf_state_local_parent, .TRUE., lacc=.TRUE.)
      ENDIF
    ENDIF
#endif

  SELECT CASE (iforcing)
  CASE (iaes)
    IF (.NOT.isRestart()) THEN
      CALL init_slowphysics (mtime_current, 1, dtime, lacc=.TRUE.)
    END IF
  END SELECT ! iforcing


  !------------------------------------------------------------------
  !  get and write out some of the initial values
  !------------------------------------------------------------------
  IF (.NOT.isRestart() .AND. (mtime_current >= time_config%tc_exp_startdate)) THEN

    ! Compute diagnostic 3D radar reflectivity (in linear units) if some derived output variables are present in any namelist.
    ! has to be computed before pp_scheduler_process(simulation_status) below!
    DO jg = 1, n_dom

      IF (.NOT. p_patch(jg)%ldom_active) CYCLE

      IF ( var_in_output(jg)%dbz .OR. var_in_output(jg)%dbz850 .OR. &
           var_in_output(jg)%dbzlmx_low .OR. var_in_output(jg)%dbzcmax ) THEN

        CALL compute_field_dbz3d_lin (jg, p_patch(jg),                                                  &
             &                        p_nh_state(jg)%prog(nnow(jg)), p_nh_state(jg)%prog(nnow_rcf(jg)), &
             &                        p_nh_state(jg)%diag, prm_diag(jg), prm_diag(jg)%dbz3d_lin, lacc=.TRUE. )

      END IF

    END DO

    !--------------------------------------------------------------------------
    ! loop over the list of internal post-processing tasks, e.g.
    ! interpolate selected fields to p- and/or z-levels
    simulation_status = new_simulation_status(l_first_step   = .TRUE.,                  &
      &                                       l_output_step  = .TRUE.,                  &
      &                                       l_dom_active   = p_patch(1:)%ldom_active, &
      &                                       i_timelevel_dyn= nnow, i_timelevel_phy= nnow_rcf)
    CALL pp_scheduler_process(simulation_status)


    CALL update_statistics
    IF (p_nh_opt_diag(1)%acc%l_any_m) THEN
#ifdef _OPENACC
      CALL finish (routine, 'update_opt_acc: OpenACC version currently not tested')
#endif
      CALL update_opt_acc(p_nh_opt_diag(1)%acc,            & ! it is ported to OpenACC but untested
        &                 p_nh_state(1)%prog(nnow_rcf(1)), &
        &                 p_nh_state(1)%prog(nnow(1))%rho, &
        &                 p_nh_state(1)%diag,              &
        &                 p_patch(1)%cells%owned,          &
        &                 p_patch(1)%nlev                  )
    END IF

    IF (output_mode%l_nml) THEN
      CALL write_name_list_output(jstep=0, lacc=i_am_accel_node)
    END IF

    !-----------------------------------------------
    ! Pass "initialized analysis" or "analysis" when
    ! time step 0 cannot be reached in time stepping
    !-----------------------------------------------

    IF (p_nh_opt_diag(1)%acc%l_any_m) THEN
#ifdef _OPENACC
      CALL finish (routine, 'reset_opt_acc: OpenACC version currently not ported')
#endif
      CALL reset_opt_acc(p_nh_opt_diag(1)%acc)
    END IF

    ! sample meteogram output
    DO jg = 1, n_dom
      IF (output_mode%l_nml        .AND. &    ! meteogram output is only initialized for nml output
        & p_patch(jg)%ldom_active  .AND. &
        & meteogram_is_sample_step( meteogram_output_config(jg), 0 ) ) THEN
#ifdef _OPENACC
        CALL finish (routine, 'meteogram_sample_vars: OpenACC version currently not ported')
#endif
        CALL meteogram_sample_vars(jg, 0, time_config%tc_startdate)
      END IF
    END DO
#ifdef MESSY
    ! MESSy initial output
!    CALL messy_write_output
#endif

  END IF ! not isRestart()

  CALL perform_nh_timeloop (mtime_current, latbc)

#if defined( _OPENACC )
  IF (i_am_accel_node) THEN
    CALL d2h_icon( p_int_state, p_int_state_local_parent, p_patch, p_patch_local_parent, &
      &            p_nh_state, prep_adv, advection_config, iforcing, lacc=.TRUE. )
    IF (n_dom > 1 .OR. l_limited_area) THEN
       CALL devcpy_grf_state (p_grf_state, .FALSE., lacc=.TRUE.)
       CALL devcpy_grf_state (p_grf_state_local_parent, .FALSE., lacc=.TRUE.)
    ELSEIF (ANY(lredgrid_phys)) THEN
       CALL devcpy_grf_state (p_grf_state_local_parent, .FALSE., lacc=.TRUE.)
    ENDIF
  ENDIF
  i_am_accel_node = .FALSE.                 ! Deactivate GPUs
#endif

  END SUBROUTINE perform_nh_stepping
  !-------------------------------------------------------------------------
  !>
  !! Organizes nonhydrostatic time stepping
  !! Currently we assume to have only one grid level.
  SUBROUTINE perform_nh_timeloop (mtime_current, latbc)
    !
    CHARACTER(len=*), PARAMETER :: routine = modname//':perform_nh_timeloop'
    TYPE(t_latbc_data),     INTENT(INOUT)  :: latbc !< data structure for async latbc prefetching
    TYPE(datetime),         POINTER        :: mtime_current     ! current datetime (mtime)

  INTEGER                              :: jg, jn, jgc
  INTEGER                              :: ierr
  LOGICAL                              :: l_compute_diagnostic_quants,  &
    &                                     l_nml_output, l_nml_output_dom(max_dom), lprint_timestep, &
    &                                     lwrite_checkpoint, lcfl_watch_mode
  TYPE(t_simulation_status)            :: simulation_status
  TYPE(datetime),   POINTER            :: mtime_old         ! copy of current datetime (mtime)

  INTEGER                              :: i, iau_iter
  REAL(wp)                             :: elapsed_time_global
  INTEGER                              :: jstep   ! step number
  INTEGER                              :: jstep0  ! step for which the restart file
                                                  ! was produced
  INTEGER                              :: kstep   ! step number relative to restart step
  INTEGER                              :: jstep_shift ! start counter for time loop
  INTEGER, ALLOCATABLE                 :: output_jfile(:)

  TYPE(timedelta), POINTER             :: model_time_step => NULL()

  TYPE(datetime), POINTER              :: eventStartDate    => NULL(), &
       &                                  eventEndDate      => NULL()
  TYPE(datetime), POINTER              :: checkpointRefDate => NULL(), &
       &                                  restartRefDate    => NULL()
  TYPE(timedelta), POINTER             :: eventInterval     => NULL()
  TYPE(event), POINTER                 :: checkpointEvent   => NULL()
  TYPE(event), POINTER                 :: restartEvent      => NULL()
  TYPE(event), POINTER                 :: lpi_max_Event     => NULL()
  TYPE(event), POINTER                 :: celltracks_Event  => NULL()
  TYPE(event), POINTER                 :: dbz_Event         => NULL()

  INTEGER                              :: checkpointEvents
  LOGICAL                              :: lret
  TYPE(t_datetime_ptr)                 :: datetime_current(max_dom)
  TYPE(t_key_value_store), POINTER :: restartAttributes
  CLASS(t_RestartDescriptor), POINTER  :: restartDescriptor

  CHARACTER(LEN=MAX_TIMEDELTA_STR_LEN)   :: td_string
  CHARACTER(LEN=MAX_DATETIME_STR_LEN)    :: dt_string, dstring
  CHARACTER(len=MAX_MTIME_ERROR_STR_LEN) :: errstring

  REAL(wp)                             :: sim_time     !< elapsed simulation time

  LOGICAL :: l_isStartdate, l_isExpStopdate, l_isRestart, l_isCheckpoint, l_doWriteRestart
  LOGICAL :: lstop_on_demand = .FALSE. , lchkp_allowed = .FALSE.

  REAL(wp), ALLOCATABLE :: elapsedTime(:)  ! time elapsed since last call of
                                           ! NWP physics routines. For restart purposes.
#ifndef __NO_ICON_UPATMO__
  TYPE(t_upatmoRestartAttributes) :: upatmoRestartAttributes
#endif
  TYPE(datetime)                      :: target_datetime  ! target date for for update of clim.
                                                          ! lower boundary conditions in NWP mode
  TYPE(datetime)                      :: ref_datetime     ! reference datetime for computing
                                                          ! climatological SST increments
  TYPE(datetime)                      :: latbc_read_datetime  ! validity time of next lbc input file

  LOGICAL :: l_accumulation_step

!!$  INTEGER omp_get_num_threads


!-----------------------------------------------------------------------
  ! calculate elapsed simulation time in seconds
  sim_time = getElapsedSimTimeInSeconds(mtime_current)
  iau_iter = 0

  ! allocate temporary variable for restarting purposes
  IF (output_mode%l_nml) THEN
    ALLOCATE(output_jfile(SIZE(output_file)), STAT=ierr)
    IF (ierr /= SUCCESS)  CALL finish (routine, 'ALLOCATE failed!')
  ENDIF

  IF (timeshift%dt_shift < 0._wp  .AND. .NOT. isRestart()) THEN
    jstep_shift = NINT(timeshift%dt_shift/dtime)
    !Model start shifted backwards by ', ABS(jstep_shift),' time steps'

    atm_phy_nwp_config(:)%lcalc_acc_avg = .FALSE.
  ELSE
    jstep_shift = 0
  ENDIF

  mtime_old => newDatetime(mtime_current)
  DO jg=1, n_dom
    datetime_current(jg)%ptr => newDatetime(mtime_current)
  END DO

  restartDescriptor => createRestartDescriptor("atm")

  jstep0 = 0

  CALL getAttributesForRestarting(restartAttributes)
  ! get start counter for time loop from restart file:
  IF (isRestart()) CALL restartAttributes%get("jstep", jstep0)

  ! for debug purposes print var lists: for msg_level >= 13 short and for >= 20 long format
  IF  (.NOT. ltestcase .AND. msg_level >= 13) &
    & CALL vlr_print_vls(lshort=(msg_level < 20))

  ! Check if current number of dynamics substeps is larger than the default value
  ! (this can happen for restarted runs only at this point)
  IF (ANY(ndyn_substeps_var(1:n_dom) > ndyn_substeps)) THEN
    lcfl_watch_mode = .TRUE.
  ELSE
    lcfl_watch_mode = .FALSE.
  ENDIF

  ! init routine for mo_nh_supervise module (eg. opening of files)
  CALL init_supervise_nh()

  eventStartDate => time_config%tc_exp_startdate
  eventEndDate   => time_config%tc_exp_stopdate

  ! for debugging purposes the referenece (anchor) date for checkpoint
  ! and restart may be switched to be relative to current jobs start
  ! date instead of the experiments start date.

  IF (time_config%is_relative_time) THEN
    checkpointRefDate => time_config%tc_startdate
    restartRefDate    => time_config%tc_startdate
  ELSE
    checkpointRefDate => time_config%tc_exp_startdate
    restartRefDate    => time_config%tc_exp_startdate
  ENDIF

  ! --- --- create checkpointing event
  eventInterval  => time_config%tc_dt_checkpoint
  checkpointEvent => newEvent('checkpoint', checkpointRefDate, eventStartDate, eventEndDate, eventInterval, errno=ierr)

  ! --- --- create restart event, ie. checkpoint + model stop
  eventInterval  => time_config%tc_dt_restart
  restartEvent => newEvent('restart', restartRefDate, eventStartDate, eventEndDate, eventInterval, errno=ierr)

  CALL printEventGroup(checkpointEvents)

  ! Create mtime events for optional NWP diagnostics
  CALL setup_nwp_diag_events(lpi_max_Event, celltracks_Event, dbz_Event)

  ! set time loop properties
  model_time_step => time_config%tc_dt_model

  jstep = jstep0+jstep_shift+1

  TIME_LOOP: DO
    ! Check if a nested domain needs to be turned off
    DO jg=2, n_dom
      IF (p_patch(jg)%ldom_active .AND. (sim_time >= end_time(jg))) THEN
        p_patch(jg)%ldom_active = .FALSE.
        !domain ',jg,' stopped at time ',sim_time
      ENDIF
    ENDDO

    ! Update time-dependent ensemble perturbations if necessary
    IF (use_ensemble_pert .AND. gribout_config(1)%perturbationNumber >= 1) THEN
      CALL compute_ensemble_pert(p_patch(1:), ext_data, prm_diag, phy_params, mtime_current, .TRUE.)
    ENDIF

    ! update model date and time mtime based
    mtime_current = mtime_current + model_time_step

    ! provisional implementation for checkpoint+stop on demand
    IF (checkpoint_on_demand) CALL check_for_checkpoint(lready_for_checkpoint, lchkp_allowed, lstop_on_demand)

    IF (lstop_on_demand) THEN
      ! --- --- create restart event, ie. checkpoint + model stop
      eventInterval  => model_time_step
      restartEvent => newEvent('restart', restartRefDate, eventStartDate, mtime_current, eventInterval, errno=ierr)
      IF (ierr /= no_Error) THEN
        CALL mtime_strerror(ierr, errstring)
        CALL finish('perform_nh_timeloop', "event 'restart': "//errstring)
      ENDIF
      CALL message('perform_nh_timeloop', "checkpoint+stop forced during runtime")
      lret = addEventToEventGroup(restartEvent, checkpointEventGroup)
    ENDIF

    ! store state of output files for restarting purposes
    IF (output_mode%l_nml .AND. jstep>=0 ) THEN
      DO i=1,SIZE(output_file)
        output_jfile(i) = get_current_jfile(output_file(i)%out_event)
      END DO
    ENDIF

    ! turn on calculation of averaged and accumulated quantities at the first regular time step
    IF (jstep-jstep0 == 1) atm_phy_nwp_config(:)%lcalc_acc_avg = .TRUE.

    lprint_timestep = msg_level > 2 .OR. MOD(jstep,25) == 0

    ! always print the first and the last time step
    lprint_timestep = lprint_timestep .OR. (jstep == jstep0+1) .OR. (jstep == jstep0+nsteps)

    !--------------------------------------------------------------------------
    ! Set output flags
    !--------------------------------------------------------------------------

    l_nml_output = output_mode%l_nml .AND. jstep >= 0 .AND. istime4name_list_output(jstep)

    DO jg = 1, n_dom
      l_nml_output_dom(jg) = output_mode%l_nml .AND. jstep >= 0 .AND. istime4name_list_output_dom(jg=jg, jstep=jstep)
    END DO

    ! Computation of diagnostic quantities may also be necessary for
    ! meteogram sampling:
!DR Note that this may be incorrect for meteograms in case that
!DR meteogram_output_config is not the same for all domains.
    !RW Computing diagnostics for mvstream could be done per dom, now it runs every timestep on all domains.
    l_compute_diagnostic_quants = l_nml_output
    DO jg = 1, n_dom
      l_compute_diagnostic_quants = l_compute_diagnostic_quants .OR. &
        &          statistics_active_on_dom(jg) .OR. &
        &          (meteogram_is_sample_step(meteogram_output_config(jg), jstep ) .AND. output_mode%l_nml)
    END DO
    l_compute_diagnostic_quants = jstep >= 0 .AND. l_compute_diagnostic_quants .AND. &
      &                           .NOT. output_mode%l_none

    ! Calculations for enhanced sound-wave and gravity-wave damping during the spinup phase
    ! if mixed second-order/fourth-order divergence damping (divdamp_order=24) is chosen.
    ! Includes increased vertical wind off-centering during the first 2 hours of integration.
    IF (divdamp_order==24) THEN
      elapsed_time_global = (REAL(jstep,wp)-0.5_wp)*dtime
      IF (elapsed_time_global <= 7200._wp+0.5_wp*dtime .AND. .NOT. ltestcase) THEN
        CALL update_spinup_damping(elapsed_time_global)
      ELSE
        divdamp_fac_o2 = 0._wp
      ENDIF
    ENDIF

    !--------------------------------------------------------------------------
    !
    ! dynamics stepping
    !
    CALL integrate_nh(datetime_current, 1, jstep-jstep_shift, iau_iter, dtime, model_time_step, 1, latbc)
    ! --------------------------------------------------------------------------------
    !
    ! Compute diagnostics for output if necessary
    !
    IF ((l_compute_diagnostic_quants .OR. iforcing==iaes .OR. iforcing==inoforcing)) THEN
      CALL diag_for_output_dyn ()
    ENDIF
    ! Adapt number of dynamics substeps if necessary
    !
    IF (lcfl_watch_mode .OR. MOD(jstep-jstep_shift,5) == 0) THEN
      IF (ANY((/MODE_IFSANA,MODE_COMBINED,MODE_COSMO,MODE_ICONVREMAP/) == init_mode)) THEN
        ! For interpolated initial conditions, apply more restrictive criteria for timestep reduction during the spinup phase
        CALL set_ndyn_substeps(lcfl_watch_mode,jstep <= 100)
      ELSE
        CALL set_ndyn_substeps(lcfl_watch_mode,.FALSE.)
      ENDIF
    ENDIF

    !--------------------------------------------------------------------------
    ! loop over the list of internal post-processing tasks, e.g.
    ! interpolate selected fields to p- and/or z-levels
    !
    ! Mean sea level pressure needs to be computed also at
    ! no-output-steps for accumulation purposes; set by l_accumulation_step
    l_accumulation_step = (iforcing == iaes) .OR. ANY(statistics_active_on_dom(:))
    simulation_status = new_simulation_status(l_output_step  = l_nml_output,             &
      &                                       l_last_step    = (jstep==(nsteps+jstep0)), &
      &                                       l_accumulation_step = l_accumulation_step, &
      &                                       l_dom_active   = p_patch(1:)%ldom_active,  &
      &                                       i_timelevel_dyn= nnow, i_timelevel_phy= nnow_rcf)
    CALL pp_scheduler_process(simulation_status)

#ifdef MESSY
    DO jg = 1, n_dom
      CALL messy_write_output(jg)
    END DO
#endif

    ! update accumlated values
    CALL update_statistics
    IF (p_nh_opt_diag(1)%acc%l_any_m) THEN
#ifdef _OPENACC
      CALL finish (routine, 'update_opt_acc: OpenACC version currently not implemented')
#endif
      CALL update_opt_acc(p_nh_opt_diag(1)%acc,            &
        &                 p_nh_state(1)%prog(nnow_rcf(1)), &
        &                 p_nh_state(1)%prog(nnow(1))%rho, &
        &                 p_nh_state(1)%diag,              &
        &                 p_patch(1)%cells%owned,          &
        &                 p_patch(1)%nlev)
      IF (l_nml_output) CALL calc_mean_opt_acc(p_nh_opt_diag(1)%acc)
    END IF
    ! output of results
    ! note: nnew has been replaced by nnow here because the update
    IF (l_nml_output) THEN
      CALL write_name_list_output(jstep, lacc=i_am_accel_node)
    ENDIF

    ! sample meteogram output
    DO jg = 1, n_dom
      IF (output_mode%l_nml        .AND. &    ! meteogram output is only initialized for nml output
        & p_patch(jg)%ldom_active  .AND. .NOT. (jstep == 0 .AND. iau_iter == 2) .AND. &
        & meteogram_is_sample_step(meteogram_output_config(jg), jstep)) THEN
#ifdef _OPENACC
        CALL finish (routine, 'meteogram_sample_vars: OpenACC version currently not implemented')
#endif
        CALL meteogram_sample_vars(jg, jstep, mtime_current)
      END IF
    END DO

    ! Diagnostics: computation of total integrals
    !
    ! Diagnostics computation is not yet properly MPI-parallelized
    !
    IF (output_mode%l_totint .AND. is_totint_time(current_step =jstep,   &
      &                                           restart_step = jstep0, &
      &                                           n_diag       = n_diag, &
      &                                           n_steps      = nsteps) ) THEN

      kstep = jstep-jstep0

#ifdef NOMPI
      IF (my_process_is_mpi_all_seq()) &
#endif
        CALL supervise_total_integrals_nh( kstep, p_patch(1:), p_nh_state, p_int_state(1:), &
        &                                  nnow(1:n_dom), nnow_rcf(1:n_dom), jstep == (nsteps+jstep0), lacc=i_am_accel_node)
    ENDIF
    ! re-initialize MAX/MIN fields with 'resetval'
    ! must be done AFTER output

    CALL reset_act%execute(slack=dtime, mtime_date=mtime_current)
    !--------------------------------------------------------------------------
    ! Write restart file
    !--------------------------------------------------------------------------
    ! check whether time has come for writing restart file

    !
    ! default is to assume we do not write a checkpoint/restart file
    lwrite_checkpoint = .FALSE.
    ! if thwe model is not supposed to write output, do not write checkpoints
    IF (.NOT. output_mode%l_none ) THEN
      ! to clarify the decision tree we use shorter and more expressive names:

      l_isStartdate    = (time_config%tc_startdate == mtime_current)
      l_isExpStopdate  = (time_config%tc_exp_stopdate == mtime_current)
      l_isRestart      = is_event_active(restartEvent, mtime_current, proc0_offloading)
      l_isCheckpoint   = is_event_active(checkpointEvent, mtime_current, proc0_offloading)
      l_doWriteRestart = time_config%tc_write_restart

      IF ( (l_isRestart .OR. l_isCheckpoint)                     &
           &  .AND.                                                        &
           !  and the current date differs from the start date
           &        .NOT. l_isStartdate                                    &
           &  .AND.                                                        &
           !  and end of run has not been reached or restart writing has been disabled
           &        (.NOT. l_isExpStopdate .OR. l_doWriteRestart)          &
           & ) THEN
        lwrite_checkpoint = .TRUE.
      END IF
    END IF

    !--------------------------------------------------------------------
    ! Pass forecast state at selected steps to DACE observation operators
    !--------------------------------------------------------------------
    IF (lwrite_checkpoint) THEN
      CALL diag_for_output_dyn ()
        DO jg = 1, n_dom

#ifndef __NO_ICON_UPATMO__
            ! upper-atmosphere physics
            IF (upatmo_config(jg)%nwp_phy%l_phy_stat( iUpatmoPrcStat%enabled )) THEN
              CALL upatmoRestartAttributesPrepare(jg, upatmoRestartAttributes, prm_upatmo(jg), mtime_current)
            ENDIF
#endif
            CALL restartDescriptor%updatePatch(p_patch(jg), &
              & opt_t_elapsed_phy          = elapsedTime,                &
              & opt_ndyn_substeps          = ndyn_substeps_var(jg),      &
              & opt_jstep_adv_marchuk_order= jstep_adv(jg)%marchuk_order,&
              & opt_depth_lnd              = nlev_soil,                  &
              & opt_nlev_snow              = nlev_snow,                  &
#ifndef __NO_ICON_UPATMO__
              & opt_upatmo_restart_atts    = upatmoRestartAttributes,    &
#endif
              & opt_ndom                   = n_dom )

        ENDDO

        ! trigger writing of restart files. note that the nest
        ! boundary has not been updated. therefore data in the
        ! boundary region may be older than the data in the prognostic
        ! region. However this has no effect on the prognostic result.
        CALL restartDescriptor%writeRestart(mtime_current, jstep, opt_output_jfile = output_jfile)

#ifdef MESSY
        CALL messy_channel_write_output(IOMODE_RST)
!       CALL messy_ncregrid_write_restart
#endif

#ifndef __NO_ICON_UPATMO__
        IF (ANY(upatmo_config(:)%nwp_phy%l_phy_stat( iUpatmoPrcStat%enabled ))) THEN
          CALL upatmoRestartAttributesDeallocate(upatmoRestartAttributes)
        ENDIF
#endif
    END IF  ! lwrite_checkpoint

#ifdef MESSYTIMER
    ! timer sync
    CALL messy_timer_reset_time
#endif

    ! prefetch boundary data if necessary
    IF(num_prefetch_proc >= 1 .AND. latbc_config%itype_latbc > 0 .AND. &
    &  .NOT.(jstep == 0 .AND. iau_iter == 1) ) THEN
      latbc_read_datetime = latbc%mtime_last_read + latbc%delta_dtime
      CALL recv_latbc_data(latbc               = latbc,              &
         &                  p_patch             = p_patch(1:),        &
         &                  p_nh_state          = p_nh_state(1),      &
         &                  p_int               = p_int_state(1),     &
         &                  cur_datetime        = mtime_current,      &
         &                  latbc_read_datetime = latbc_read_datetime,&
         &                  lcheck_read         = .TRUE.,             &
         &                  tlev                = latbc%new_latbc_tlev)
    ENDIF

    IF (mtime_current >= time_config%tc_stopdate .OR. lstop_on_demand) THEN
       ! leave time loop
       EXIT TIME_LOOP
    END IF

    ! Reset model to initial state if IAU iteration is selected and the first iteration cycle has been completed
     jstep = jstep + 1
    sim_time = getElapsedSimTimeInSeconds(mtime_current)

  ENDDO TIME_LOOP

  ! clean-up routine for mo_nh_supervise module (eg. closing of files)
  CALL finalize_supervise_nh()

  END SUBROUTINE perform_nh_timeloop
  !-----------------------------------------------------------------------------
  !>
  !! integrate_nh
  !!
  !! Performs dynamics time stepping:  Rotational modes (helicity bracket) and
  !! divergent modes (Poisson bracket) are split using Strang splitting.
  RECURSIVE SUBROUTINE integrate_nh (datetime_local, jg, nstep_global,   &
    &                                iau_iter, dt_loc, mtime_dt_loc, num_steps, latbc )

    CHARACTER(len=*), PARAMETER :: routine = modname//':integrate_nh'

    TYPE(t_datetime_ptr)    :: datetime_local(:)     !< current datetime in mtime format (for each patch)

    INTEGER , INTENT(IN)    :: jg           !< current grid level
    INTEGER , INTENT(IN)    :: nstep_global !< counter of global time step
    INTEGER , INTENT(IN)    :: num_steps    !< number of time steps to be executed
    INTEGER , INTENT(IN)    :: iau_iter     !< counter for IAU iteration
    REAL(wp), INTENT(IN)    :: dt_loc       !< time step applicable to local grid level
    TYPE(timedelta), POINTER :: mtime_dt_loc !< time step applicable to local grid level (mtime format)
    TYPE(t_latbc_data), TARGET, INTENT(INOUT) :: latbc

    ! Local variables

    ! Time levels
    INTEGER :: n_now_grf, n_now, n_save
    INTEGER :: n_now_rcf, n_new_rcf         ! accounts for reduced calling frequencies (rcf)

    INTEGER :: jstep, jgp, jgc, jn

    REAL(wp):: dt_sub                ! (advective) timestep for next finer grid level
    TYPE(timedelta), POINTER :: mtime_dt_sub
    REAL(wp):: rdt_loc,  rdtmflx_loc ! inverse time step for local grid level

    LOGICAL :: lnest_active, lcall_rrg, lbdy_nudging

    INTEGER, PARAMETER :: nsteps_nest=2 ! number of time steps executed in nested domain

    REAL(wp)                             :: sim_time !< elapsed simulation time on this grid level

    TYPE(t_pi_atm), POINTER :: ptr_latbc_data_atm_old, ptr_latbc_data_atm_new

    ! calculate elapsed simulation time in seconds (local time for
    ! this domain!)
    sim_time = getElapsedSimTimeInSeconds(datetime_local(jg)%ptr)

    !--------------------------------------------------------------------------
    ! This timer must not be called in nested domain because the model crashes otherwise
    ! Determine parent domain ID
    IF ( jg > 1) THEN
      jgp = p_patch(jg)%parent_id
    ELSE IF (n_dom_start == 0) THEN
      jgp = 0
    ELSE
      jgp = 1
    ENDIF

    ! If the limited-area mode is used, save initial state in the coarse domain
    ! The save time level is later on used for boundary relaxation in the case of
    ! fixed boundary conditions.
    ! If time-dependent data from a driving model are provided,
    ! they should be written to the save time level, so that the relaxation routine
    ! automatically does the right thing

    IF (jg == 1 .AND. l_limited_area .AND. linit_dyn(jg)) THEN

      n_save = nsav2(jg)
      n_now = nnow(jg)

    ENDIF

    ! This executes one time step for the global domain and two steps for nested domains
    JSTEP_LOOP: DO jstep = 1, num_steps

      IF (ifeedback_type == 1 .AND. (jstep == 1) .AND. jg > 1 ) THEN

          !FEEDBACK (nesting): OpenACC version currently not implemented
        ! Save prognostic variables at current timestep to compute
        ! feedback increments (not needed in global domain)
        n_now = nnow(jg)
        n_save = nsav2(jg)
      ENDIF

      ! update several switches which decide upon
      ! - switching order of operators in case of Marchuk-splitting
      !
      ! simplified setting (may be removed lateron)
      jstep_adv(jg)%marchuk_order = jstep_adv(jg)%marchuk_order + 1

      IF ( p_patch(jg)%n_childdom > 0 .AND. ndyn_substeps_var(jg) > 1) THEN
        lbdy_nudging = .FALSE.
        lnest_active = .FALSE.
        DO jn = 1, p_patch(jg)%n_childdom
          jgc = p_patch(jg)%child_id(jn)
          IF (p_patch(jgc)%ldom_active) THEN
            lnest_active = .TRUE.
            IF (.NOT. lfeedback(jgc)) lbdy_nudging = .TRUE.
          ENDIF
        ENDDO

        ! Save prognostic variables at current timestep to compute
        ! interpolation tendencies
        n_now  = nnow(jg)
        n_save = nsav1(jg)

        IF (lnest_active) THEN ! optimized copy restricted to nest boundary points
          CALL save_progvars(jg,p_nh_state(jg)%prog(n_now),p_nh_state(jg)%prog(n_save))
        ENDIF
      ENDIF

      ! Set local variable for rcf-time levels
      n_now_rcf = nnow_rcf(jg)
      n_new_rcf = nnew_rcf(jg)

#ifdef MESSY
#ifdef _OPENACC
      CALL finish (routine, 'MESSY:  OpenACC version currently not implemented')
#endif
      CALL messy_global_start(jg)
      CALL messy_local_start(jg)
      CALL messy_vdiff(jg)
#endif
      !
      ! Update model date (for local patch!) - Note that for the
      ! top-level patch, this is omitted, since the update has already
      ! happened in the calling subroutine.
      datetime_local(jg)%ptr = datetime_local(jg)%ptr + mtime_dt_loc
      sim_time = getElapsedSimTimeInSeconds(datetime_local(jg)%ptr)

      IF (itime_scheme == 1) THEN
        !------------------
        ! Pure advection
        !------------------

        ! Print control output for maximum horizontal and vertical wind speed
        !
        ! 2 Cases:
        ! msg_level E [12, inf[: print max/min output for every domain and every transport step
        ! msg_level E [ 8,  11]: print max/min output for global domain and every transport step
        IF (msg_level >= 12 .OR. msg_level >= 8 .AND. jg == 1) THEN
          CALL print_maxwinds(p_patch(jg), p_nh_state(jg)%prog(nnow(jg))%vn,   &
            p_nh_state(jg)%prog(nnow(jg))%w, lacc=.TRUE.)
        ENDIF

#ifdef MESSY
        CALL main_tracer_beforeadv
#endif

        ! Update nh-testcases
        IF (ltestcase_update) THEN
#ifdef _OPENACC
          CALL finish (routine, 'nh_testcase_interface: OpenACC version currently not implemented')
#endif
          CALL nh_testcase_interface( nstep_global,                &  !in
            &                         dt_loc,                      &  !in
            &                         sim_time,                    &  !in
            &                         p_patch(jg),                 &  !in
            &                         p_nh_state(jg),              &  !inout
            &                         p_int_state(jg),             &  !in
            &                         jstep_adv(jg)%marchuk_order  )  !in
        ENDIF

        ! Diagnose some velocity-related quantities for the tracer
        ! transport scheme
        CALL prepare_tracer( p_patch(jg), p_nh_state(jg)%prog(nnow(jg)),  &! in
          &         p_nh_state(jg)%prog(nnew(jg)),                        &! in
          &         p_nh_state(jg)%metrics, p_int_state(jg),              &! in
          &         ndyn_substeps_var(jg), .TRUE., .TRUE.,                &! in
          &         advection_config(jg)%lfull_comp,                      &! in
          &         p_nh_state(jg)%diag,                                  &! inout
          &         prep_adv(jg)%vn_traj, prep_adv(jg)%mass_flx_me,       &! inout
          &         prep_adv(jg)%mass_flx_ic                              )! inout

        ! airmass_now
        CALL compute_airmass(p_patch   = p_patch(jg),                       & !in
          &                  p_metrics = p_nh_state(jg)%metrics,            & !in
          &                  rho       = p_nh_state(jg)%prog(nnow(jg))%rho, & !in
          &                  airmass   = p_nh_state(jg)%diag%airmass_now    ) !inout

        ! airmass_new
        CALL compute_airmass(p_patch   = p_patch(jg),                       & !in
          &                  p_metrics = p_nh_state(jg)%metrics,            & !in
          &                  rho       = p_nh_state(jg)%prog(nnew(jg))%rho, & !in
          &                  airmass   = p_nh_state(jg)%diag%airmass_new    ) !inout

        CALL step_advection(                                                 &
          &       p_patch           = p_patch(jg),                           & !in
          &       p_int_state       = p_int_state(jg),                       & !in
          &       p_dtime           = dt_loc,                                & !in
          &       k_step            = jstep_adv(jg)%marchuk_order,           & !in
          &       p_tracer_now      = p_nh_state(jg)%prog(n_now_rcf)%tracer, & !in
          &       p_mflx_contra_h   = prep_adv(jg)%mass_flx_me,              & !in
          &       p_vn_contra_traj  = prep_adv(jg)%vn_traj,                  & !in
          &       p_mflx_contra_v   = prep_adv(jg)%mass_flx_ic,              & !in
          &       p_cellhgt_mc_now  = p_nh_state(jg)%metrics%ddqz_z_full,    & !in
          &       p_rhodz_new       = p_nh_state(jg)%diag%airmass_new,       & !in
          &       p_rhodz_now       = p_nh_state(jg)%diag%airmass_now,       & !in
          &       p_grf_tend_tracer = p_nh_state(jg)%diag%grf_tend_tracer,   & !in
          &       p_tracer_new      = p_nh_state(jg)%prog(n_new_rcf)%tracer, & !inout
          &       p_mflx_tracer_h   = p_nh_state(jg)%diag%hfl_tracer,        & !out
          &       p_mflx_tracer_v   = p_nh_state(jg)%diag%vfl_tracer,        & !out
          &       rho_incr          = p_nh_state(jg)%diag%rho_incr,          & !in
          &       q_ubc             = prep_adv(jg)%q_ubc,                    & !in
          &       q_int             = prep_adv(jg)%q_int,                    & !out
          &       opt_ddt_tracer_adv= p_nh_state(jg)%diag%ddt_tracer_adv,    & !optout
          &       opt_deepatmo_t1mc = p_nh_state(jg)%metrics%deepatmo_t1mc,  & !optin
          &       opt_deepatmo_t2mc = p_nh_state(jg)%metrics%deepatmo_t2mc   ) !optin

#ifdef MESSY
        CALL main_tracer_afteradv
#endif

      ELSE  ! itime_scheme /= 1
        ! artificial forcing (Held-Suarez test forcing)
        !!!!!!!!
        ! re-check: iadv_rcf -> ndynsubsteps
        !!!!!!!!
        IF ( iforcing == iheldsuarez) THEN
          CALL held_suarez_nh_interface (p_nh_state(jg)%prog(nnow(jg)), p_patch(jg), &
                                         p_int_state(jg),p_nh_state(jg)%metrics,  &
                                         p_nh_state(jg)%diag)
        ENDIF

        ! Set diagnostic fields, which collect dynamics tendencies over all substeps, to zero
        CALL init_ddt_vn_diagnostics(p_nh_state(jg)%diag)

        ! For real-data runs, perform an extra diffusion call before the first time
        ! step because no other filtering of the interpolated velocity field is done
        !
        IF (ldynamics .AND. .NOT.ltestcase .AND. linit_dyn(jg) .AND. diffusion_config(jg)%lhdiff_vn .AND. &
            init_mode /= MODE_IAU .AND. init_mode /= MODE_IAU_OLD) THEN

          ! Use here the model time step dt_loc, for which the diffusion is computed here.
          CALL diffusion(p_nh_state(jg)%prog(nnow(jg)), p_nh_state(jg)%diag,       &
            p_nh_state(jg)%metrics, p_patch(jg), p_int_state(jg), dt_loc, .TRUE.)

        ENDIF

        !IF (itype_comm == 1) THEN

          IF (ldynamics) THEN
            ! dynamics integration with substepping
            !
            CALL perform_dyn_substepping (p_patch(jg), p_nh_state(jg), p_int_state(jg), &
              &                           prep_adv(jg), jstep, iau_iter, dt_loc, datetime_local(jg)%ptr)
            ! diffusion at physics time steps
            !
            IF (diffusion_config(jg)%lhdiff_vn .AND. lhdiff_rcf) THEN
              CALL diffusion(p_nh_state(jg)%prog(nnew(jg)), p_nh_state(jg)%diag,     &
                &            p_nh_state(jg)%metrics, p_patch(jg), p_int_state(jg),   &
                &            dt_loc, .FALSE.)
            ENDIF

          !ELSE IF (iforcing == inwp) THEN
          ! dynamics for ldynamics off, option of coriolis force, typically used for SCM and similar test cases
        !itype_comm /= 1 currently not implemented'
        !ENDIF


#ifdef MESSY
        CALL main_tracer_beforeadv
#endif


        ! 5. tracer advection
        !-----------------------
        IF ( ltransport) THEN


          IF (msg_level >= 12) THEN
            WRITE(message_text,'(a,i2)') 'call advection  DOM:',jg
            CALL message('integrate_nh', message_text)
          ENDIF

          CALL step_advection(                                                &
            &       p_patch           = p_patch(jg),                          & !in
            &       p_int_state       = p_int_state(jg),                      & !in
            &       p_dtime           = dt_loc,                               & !in
            &       k_step            = jstep_adv(jg)%marchuk_order,          & !in
            &       p_tracer_now      = p_nh_state(jg)%prog(n_now_rcf)%tracer,& !in
            &       p_mflx_contra_h   = prep_adv(jg)%mass_flx_me,             & !in
            &       p_vn_contra_traj  = prep_adv(jg)%vn_traj,                 & !in
            &       p_mflx_contra_v   = prep_adv(jg)%mass_flx_ic,             & !in
            &       p_cellhgt_mc_now  = p_nh_state(jg)%metrics%ddqz_z_full,   & !in
            &       p_rhodz_new       = p_nh_state(jg)%diag%airmass_new,      & !in
            &       p_rhodz_now       = p_nh_state(jg)%diag%airmass_now,      & !in
            &       p_grf_tend_tracer = p_nh_state(jg)%diag%grf_tend_tracer,  & !in
            &       p_tracer_new      = p_nh_state(jg)%prog(n_new_rcf)%tracer,& !inout
            &       p_mflx_tracer_h   = p_nh_state(jg)%diag%hfl_tracer,       & !out
            &       p_mflx_tracer_v   = p_nh_state(jg)%diag%vfl_tracer,       & !out
            &       rho_incr          = p_nh_state(jg)%diag%rho_incr,         & !in
            &       q_ubc             = prep_adv(jg)%q_ubc,                   & !in
            &       q_int             = prep_adv(jg)%q_int,                   & !out
            &       opt_ddt_tracer_adv= p_nh_state(jg)%diag%ddt_tracer_adv,   & !out
            &       opt_deepatmo_t1mc = p_nh_state(jg)%metrics%deepatmo_t1mc, & !optin
            &       opt_deepatmo_t2mc = p_nh_state(jg)%metrics%deepatmo_t2mc  ) !optin

          IF (iprog_aero >= 1) THEN

#ifdef _OPENACC
            CALL finish (routine, 'aerosol_2D_advection: OpenACC version currently not implemented')
#endif
            CALL sync_patch_array(SYNC_C, p_patch(jg), prm_diag(jg)%aerosol)
            CALL aerosol_2D_advection( p_patch(jg), p_int_state(jg), iprog_aero,   & !in
              &          dt_loc, prm_diag(jg)%aerosol, prep_adv(jg)%vn_traj,       & !in, inout, in
              &          prep_adv(jg)%mass_flx_me, prep_adv(jg)%mass_flx_ic,       & !in
              &          p_nh_state(jg)%metrics%ddqz_z_full_e,                     & !in
              &          p_nh_state(jg)%diag%airmass_now,                          & !in
              &          p_nh_state(jg)%diag%airmass_new                           ) !in
            CALL sync_patch_array(SYNC_C, p_patch(jg), prm_diag(jg)%aerosol)
            CALL aerosol_2D_diffusion( p_patch(jg), p_int_state(jg), nproma, prm_diag(jg)%aerosol)
          ENDIF


        ENDIF !ltransport

#ifdef MESSY
        CALL main_tracer_afteradv
#endif
        IF ( iforcing==iaes  ) THEN
            ! aes physics
            !
            CALL interface_iconam_aes(     dt_loc                                    & !in
                &                         ,datetime_local(jg)%ptr                    & !in
                &                         ,p_patch(jg)                               & !in
                &                         ,p_int_state(jg)                           & !in
                &                         ,p_nh_state(jg)%metrics                    & !in
                &                         ,p_nh_state(jg)%prog(nnew(jg))             & !inout
                &                         ,p_nh_state(jg)%prog(n_new_rcf)            & !inout
                &                         ,p_nh_state(jg)%diag                       )

            !
#endif

          ! Boundary interpolation of land state variables entering into radiation computation
          ! if a reduced grid is used in the child domain(s)
          DO jn = 1, p_patch(jg)%n_childdom

            jgc = p_patch(jg)%child_id(jn)
            IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

            IF ( lredgrid_phys(jgc) ) THEN
              ! Determine if radiation in the nested domain will be triggered
              ! during the subsequent two (small) time steps.
              ! The time range is given by ]mtime_current, mtime_current+slack]
              IF (patch_weight(jgc) > 0._wp) THEN
                ! in this case, broadcasts of mtime_current and nextActive are necessary.
                ! They are encapsulated in isNextTriggerTimeInRange
                lcall_rrg = atm_phy_nwp_config(jgc)%phyProc_rad%isNextTriggerTimeInRange( &
                  &                                             mtime_current = datetime_local(jgc)%ptr, &
                  &                                             slack         = mtime_dt_loc, &
                  &                                             p_source      = p_patch(jgc)%proc0, &
                  &                                             comm          = p_comm_work)

              ELSE
                lcall_rrg = atm_phy_nwp_config(jgc)%phyProc_rad%isNextTriggerTimeInRange( &
                  &                                             mtime_current = datetime_local(jgc)%ptr, &
                  &                                             slack         = mtime_dt_loc)
              ENDIF

            ELSE
              lcall_rrg = .FALSE.
            ENDIF

            IF (lcall_rrg) THEN
              CALL interpol_rrg_grf(jg, jgc, jn, nnew_rcf(jg), lacc=.TRUE.)
            ENDIF
            IF (lcall_rrg .AND. atm_phy_nwp_config(jgc)%latm_above_top) THEN
              CALL copy_rrg_ubc(jg, jgc)
            ENDIF

          ENDDO
        ENDIF !iforcing

        ! Terminator toy chemistry
        !
        ! So far it can only be activated for testcases and not for real-cases,
        ! since the initialization is done in init_nh_testcase. However,
        ! nothing speaks against combining toy chemistry with real case runs.
        IF (ltestcase .AND. is_toy_chem) THEN
#ifdef _OPENACC
          CALL finish (routine, 'dcmip_terminator_interface: OpenACC version currently not implemented')
#endif
          CALL dcmip_terminator_interface (p_patch(jg),            & !in
            &                              p_nh_state(jg)%metrics, & !in
            &                              p_nh_state(jg)%prog,    & !inout
            &                              p_nh_state(jg)%diag,    & !inout
            &                              datetime_local(jg)%ptr, & !in
            &                              dt_loc                  ) !in
        ENDIF

        ! Update nh-testcases
        IF (ltestcase_update) THEN
#ifdef _OPENACC
          CALL finish (routine, 'nh_testcase_interface: OpenACC version currently not implemented')
#endif
          CALL nh_testcase_interface( nstep_global,                &  !in
            &                         dt_loc,                      &  !in
            &                         sim_time,                    &  !in
            &                         p_patch(jg),                 &  !in
            &                         p_nh_state(jg),              &  !inout
            &                         p_int_state(jg),             &  !in
            &                         jstep_adv(jg)%marchuk_order  )  !in
        ENDIF

#ifdef MESSY
        CALL messy_physc(jg)
#endif

      ENDIF  ! itime_scheme
      !
      ! lateral nudging and optional upper boundary nudging in limited area mode
      !
      IF ( (l_limited_area .AND. (.NOT. l_global_nudging)) ) THEN
        IF (latbc_config%itype_latbc > 0) THEN  ! use time-dependent boundary data
          IF (num_prefetch_proc == 0) THEN
            !Synchronous latBC input has been disabled'
            !exit
          END IF

          IF (latbc_config%nudge_hydro_pres) CALL sync_patch_array_mult(SYNC_C, p_patch(jg), 2, &
            p_nh_state(jg)%diag%pres, p_nh_state(jg)%diag%temp, opt_varname="diag%pres and diag%temp")

          ! update the linear time interpolation weights
          ! latbc%lc1
          ! latbc%lc2
          CALL latbc%update_intp_wgt(datetime_local(jg)%ptr)

          IF (jg==1) THEN
            ! lateral boundary nudging (for DOM01 only)
            CALL limarea_nudging_latbdy(p_patch(jg),p_nh_state(jg)%prog(nnew(jg)),  &
              &  p_nh_state(jg)%prog(n_new_rcf)%tracer,                             &
              &  p_nh_state(jg)%metrics,p_nh_state(jg)%diag,p_int_state(jg),        &
              &  p_latbc_old=latbc%latbc_data(latbc%prev_latbc_tlev())%atm,         &
              &  p_latbc_new=latbc%latbc_data(latbc%new_latbc_tlev)%atm,            &
              &  lc1=latbc%lc1, lc2=latbc%lc2)
          ENDIF

          IF (nudging_config(jg)%ltype(indg_type%ubn)) THEN
            ! set pointer to upper boundary nudging data
            IF (jg==1) THEN
              ptr_latbc_data_atm_old =>latbc%latbc_data(latbc%prev_latbc_tlev())%atm
              ptr_latbc_data_atm_new =>latbc%latbc_data(latbc%new_latbc_tlev   )%atm
            ELSE
              ptr_latbc_data_atm_old =>latbc%latbc_data(latbc%prev_latbc_tlev())%atm_child(jg)
              ptr_latbc_data_atm_new =>latbc%latbc_data(latbc%new_latbc_tlev   )%atm_child(jg)
            ENDIF
            !
            ! upper boundary nudging
            CALL limarea_nudging_upbdy(p_patch(jg),p_nh_state(jg)%prog(nnew(jg)),   &
              &  p_nh_state(jg)%prog(n_new_rcf)%tracer,                             &
              &  p_nh_state(jg)%metrics,p_nh_state(jg)%diag,p_int_state(jg),        &
              &  p_latbc_old=ptr_latbc_data_atm_old,                                &
              &  p_latbc_new=ptr_latbc_data_atm_new,                                &
              &  lc1=latbc%lc1, lc2=latbc%lc2)
          ENDIF

        ELSE  ! constant lateral boundary data

          IF (jg==1) THEN
            ! Model state is nudged towards constant state along the lateral boundaries
            ! Currently only implemented for the base domain
            !
            CALL limarea_nudging_latbdy(p_patch(jg),p_nh_state(jg)%prog(nnew(jg)),  &
              &                         p_nh_state(jg)%prog(n_new_rcf)%tracer,      &
              &                         p_nh_state(jg)%metrics,p_nh_state(jg)%diag,p_int_state(jg), &
              &                         p_latbc_const=p_nh_state(jg)%prog(nsav2(jg)))
          ENDIF

        ENDIF

      ELSE IF (l_global_nudging .AND. jg==1) THEN

#ifdef _OPENACC
        CALL finish (routine, 'nudging_interface: OpenACC version currently not implemented')
#endif
        ! Apply global nudging
        CALL nudging_interface( p_patch          = p_patch(jg),            & !in
          &                     p_nh_state       = p_nh_state(jg),         & !inout
          &                     latbc            = latbc,                  & !in
          &                     mtime_datetime   = datetime_local(jg)%ptr, & !in
          &                     nnew             = nnew(jg),               & !in
          &                     nnew_rcf         = n_new_rcf,              & !in
          &                     upatmo_config    = upatmo_config(jg),      & !in
          &                     nudging_config   = nudging_config(jg)      ) !inout

      ENDIF
      ! Check if at least one of the nested domains is active
      !
      IF (p_patch(jg)%n_childdom > 0) THEN
        lnest_active = .FALSE.
        DO jn = 1, p_patch(jg)%n_childdom
          jgc = p_patch(jg)%child_id(jn)
          IF (p_patch(jgc)%ldom_active) lnest_active = .TRUE.
        ENDDO
      ENDIF

      ! If there are nested domains...
      IF (p_patch(jg)%n_childdom > 0 .AND. lnest_active ) THEN

        IF (ndyn_substeps_var(jg) == 1) THEN
          n_now_grf  = nnow(jg)
        ELSE
          n_now_grf  = nsav1(jg)
        ENDIF

        rdt_loc     = 1._wp/dt_loc
        dt_sub      = dt_loc/2._wp    ! (adv.) time step on next refinement level
        mtime_dt_sub => newTimedelta(mtime_dt_loc)
        mtime_dt_sub = mtime_dt_sub*0.5_wp
        rdtmflx_loc = 1._wp/(dt_loc*(REAL(MAX(1,ndyn_substeps_var(jg)-1),wp)/REAL(ndyn_substeps_var(jg),wp)))

        ! Compute time tendencies for interpolation to refined mesh boundaries
        CALL compute_tendencies (jg,nnew(jg),n_now_grf,n_new_rcf,n_now_rcf, &
          &                      rdt_loc,rdtmflx_loc)
        ! Loop over nested domains
        DO jn = 1, p_patch(jg)%n_childdom

          jgc = p_patch(jg)%child_id(jn)

          ! Interpolate tendencies to lateral boundaries of refined mesh (jgc)
          IF (p_patch(jgc)%ldom_active) THEN
            CALL boundary_interpolation(jg, jgc,                   &
              &  n_now_grf,nnow(jgc),n_now_rcf,nnow_rcf(jgc),      &
              &  p_patch(1:),p_nh_state(:),prep_adv(:),p_grf_state(1:))
          ENDIF

        ENDDO

        ! prep_bdy_nudging can not be called using delayed requests!
        DO jn = 1, p_patch(jg)%n_childdom

          jgc = p_patch(jg)%child_id(jn)
          IF (.NOT. p_patch(jgc)%ldom_active) CYCLE
          ! If feedback is turned off for child domain, compute parent-child
          ! differences for boundary nudging
          !
          IF (lfeedback(jgc) .AND. l_density_nudging .AND. grf_intmethod_e <= 4) THEN
            CALL prep_rho_bdy_nudging(jg,jgc)
          ELSE IF (.NOT. lfeedback(jgc)) THEN
            CALL prep_bdy_nudging(jg,jgc)
          ENDIF
        ENDDO

        DO jn = 1, p_patch(jg)%n_childdom

          jgc = p_patch(jg)%child_id(jn)
          IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

          IF(p_patch(jgc)%domain_is_owned) THEN
            IF(proc_split) CALL push_glob_comm(p_patch(jgc)%comm, p_patch(jgc)%proc0)
            ! Recursive call to process_grid_level for child grid level
            CALL integrate_nh( datetime_local, jgc, nstep_global, iau_iter, &
              &                dt_sub, mtime_dt_sub, nsteps_nest, latbc )
            IF(proc_split) CALL pop_glob_comm()
          ENDIF

        ENDDO

        DO jn = 1, p_patch(jg)%n_childdom

          ! Call feedback to copy averaged prognostic variables from refined mesh back
          ! to the coarse mesh (i.e. from jgc to jg)
          jgc = p_patch(jg)%child_id(jn)
          IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

          IF (lfeedback(jgc)) THEN
            IF (ifeedback_type == 1) THEN
              CALL feedback(p_patch, p_nh_state, p_int_state, p_grf_state, p_lnd_state, &
                &           jgc, jg)
            ELSE
                CALL relax_feedback(  p_patch(n_dom_start:n_dom),            &
                  & p_nh_state(1:n_dom), p_int_state(n_dom_start:n_dom),     &
                  & p_grf_state(n_dom_start:n_dom), jgc, jg, dt_loc)

            ENDIF
            IF (ldass_lhn) THEN
              IF (assimilation_config(jgc)%dass_lhn%isActive(datetime_local(jgc)%ptr)) THEN
                CALL lhn_feedback(p_patch(n_dom_start:n_dom), lhn_fields, &
                  p_grf_state(n_dom_start:n_dom), jgc, jg)
              END IF
            ENDIF
            ! Note: the last argument of "feedback" ensures that tracer feedback is
            ! only done for those time steps in which transport and microphysics are called
          ENDIF
        ENDDO
      ENDIF



      IF (test_mode <= 0) THEN ! ... normal execution of time stepping
        ! Finally, switch between time levels now and new for next time step
        CALL swap(nnow(jg), nnew(jg))

        ! Special treatment for processes (i.e. advection) which can be treated with
        ! reduced calling frequency. Switch between time levels now and new immediately
        ! AFTER the last transport timestep.
        CALL swap(nnow_rcf(jg), nnew_rcf(jg))

      ENDIF


      ! Check if nested domains have to be activated
      IF ( p_patch(jg)%n_childdom > 0 ) THEN

        ! Loop over nested domains
        DO jn = 1, p_patch(jg)%n_childdom
          jgc = p_patch(jg)%child_id(jn)

          IF ( .NOT. p_patch(jgc)%ldom_active .AND. &
            &  (sim_time >= start_time(jgc))  .AND. &
            &  (sim_time <  end_time(jgc))) THEN
            p_patch(jgc)%ldom_active = .TRUE.

            jstep_adv(jgc)%marchuk_order = 0
            datetime_local(jgc)%ptr      = datetime_local(jg)%ptr
            linit_dyn(jgc)               = .TRUE.
            dt_sub                       = dt_loc/2._wp

            IF (  atm_phy_nwp_config(jgc)%inwp_surface == 1 ) THEN
              CALL aggregate_landvars(p_patch(jg), ext_data(jg),                &
                p_lnd_state(jg)%prog_lnd(nnow_rcf(jg)), p_lnd_state(jg)%diag_lnd, &
                lacc=.TRUE.)
            ENDIF

#ifdef _OPENACC
            IF (msg_level >= 7) &
              & CALL message (routine, 'NESTING online init: Switching to CPU for initialization')

            ! The online initialization of the nest runs on CPU only.
            CALL gpu_d2h_nh_nwp(p_patch(jg), prm_diag(jg), ext_data=ext_data(jg), lacc=i_am_accel_node)
            i_am_accel_node = .FALSE. ! disable the execution of ACC kernels
#endif
            CALL initialize_nest(jg, jgc)

            ! Apply hydrostatic adjustment, using downward integration
            ! (deep-atmosphere modification should enter implicitly via reference state)
            CALL hydro_adjust_const_thetav(p_patch(jgc), p_nh_state(jgc)%metrics, .TRUE.,    &
              p_nh_state(jgc)%prog(nnow(jgc))%rho, p_nh_state(jgc)%prog(nnow(jgc))%exner,    &
              p_nh_state(jgc)%prog(nnow(jgc))%theta_v )

            CALL init_exner_pr(jgc, nnow(jgc), use_acc=.FALSE.)

            ! Activate cold-start mode in TERRA-init routine irrespective of what has been used for the global domain
            init_mode_soil = 1


            ! init airmass_new (diagnose airmass from \rho(now)). airmass_now not needed
            CALL compute_airmass(p_patch   = p_patch(jgc),                        & !in
              &                  p_metrics = p_nh_state(jgc)%metrics,             & !in
              &                  rho       = p_nh_state(jgc)%prog(nnow(jgc))%rho, & !in
              &                  airmass   = p_nh_state(jgc)%diag%airmass_new     ) !inout

            IF ( lredgrid_phys(jgc) ) THEN
              CALL interpol_rrg_grf(jg, jgc, jn, nnow_rcf(jg), lacc=.FALSE.)
              IF (atm_phy_nwp_config(jgc)%latm_above_top) THEN
                CALL copy_rrg_ubc(jg, jgc)
              ENDIF
            ENDIF

#ifdef _OPENACC
            i_am_accel_node = my_process_is_work()
            CALL gpu_h2d_nh_nwp(p_patch(jg), prm_diag(jg), ext_data=ext_data(jg), lacc=i_am_accel_node) ! necessary as Halo-Data can be modified
            CALL gpu_h2d_nh_nwp(p_patch(jgc), prm_diag(jgc), ext_data=ext_data(jgc), phy_params=phy_params(jgc), &
                                atm_phy_nwp_config=atm_phy_nwp_config(jg), lacc=i_am_accel_node)
            IF (msg_level >= 7) &
              & CALL message (routine, 'NESTING online init: Switching back to GPU')
#endif

            CALL init_slowphysics (datetime_local(jgc)%ptr, jgc, dt_sub, lacc=.TRUE.)
            ! jg: use opt_id to account for multiple childs that can be initialized at once
            ! jgc: opt_id not needed as jgc should be only initialized once

          ENDIF
        ENDDO
      ENDIF

#ifdef MESSY
      CALL messy_local_end(jg)
      CALL messy_global_end(jg)
#endif

    ENDDO JSTEP_LOOP
  END SUBROUTINE integrate_nh

  !>
  !! Performs dynamical core substepping with respect to physics/transport.
  !!
  !! Perform dynamical core substepping with respect to physics/transport.
  !! Number of substeps is given by ndyn_substeps.
  SUBROUTINE perform_dyn_substepping (p_patch, p_nh_state, p_int_state, prep_adv, &
    &                                 jstep, iau_iter, dt_phy, mtime_current)

    TYPE(t_patch)       ,INTENT(INOUT) :: p_patch

    TYPE(t_nh_state)    ,INTENT(INOUT) :: p_nh_state

    TYPE(t_int_state)   ,INTENT(IN)    :: p_int_state

    TYPE(t_prepare_adv) ,INTENT(INOUT) :: prep_adv

    INTEGER             ,INTENT(IN)    :: jstep     ! number of current (large) time step
                                                    ! performed in current domain
    INTEGER             ,INTENT(IN)    :: iau_iter  ! counter for IAU iteration
    REAL(wp)            ,INTENT(IN)    :: dt_phy    ! physics time step for current patch

    TYPE(datetime)      ,INTENT(IN)    :: mtime_current

    CHARACTER(len=*), PARAMETER :: routine = modname//':perform_dyn_substepping'

    ! local variables
    INTEGER                  :: jg                ! domain ID
    INTEGER                  :: nstep             ! timestep counter
    INTEGER                  :: ndyn_substeps_tot ! total number of dynamics substeps
                                                  ! since last boundary update
    REAL(wp)                 :: dt_dyn            ! dynamics time step
    REAL(wp)                 :: cur_time          ! current time (for IAU)

    LOGICAL                  :: lclean_mflx       ! .TRUE.: first substep
    LOGICAL                  :: l_recompute       ! .TRUE.: recompute velocity tendencies for predictor
                                                  ! (first substep)
    LOGICAL                  :: lsave_mflx
    LOGICAL                  :: lprep_adv         !.TRUE.: do computations for preparing tracer advection in solve_nh
    LOGICAL                  :: llast             !.TRUE.: this is the last substep
    TYPE(timeDelta) :: time_diff
    !-------------------------------------------------------------------------

    ! get domain ID
    jg = p_patch%id

    ! compute dynamics timestep
    dt_dyn = dt_phy/ndyn_substeps_var(jg)

    IF ( idiv_method == 1 .AND. (ltransport .OR. p_patch%n_childdom > 0 .AND. grf_intmethod_e >= 5)) THEN
      lprep_adv = .TRUE. ! do computations for preparing tracer advection in solve_nh
    ELSE
      lprep_adv = .FALSE.
    ENDIF

    ! perform dynamics substepping
    !
    SUBSTEPS: DO nstep = 1, ndyn_substeps_var(jg)

      ! Print control output for maximum horizontal and vertical wind speed
      !
      ! 3 Cases:
      ! msg_level E [12, inf[: print max/min output for every domain and every substep
      ! msg_level E [ 8,  11]: print max/min output for global domain and every substep
      ! msg_level E [ 5,   7]: print max/min output for global domain and first substep
      !
      IF (msg_level >= 12 &
        & .OR. msg_level >= 8 .AND. jg == 1 &
        & .OR. msg_level >= 5 .AND. jg == 1 .AND. nstep == 1) THEN
        CALL print_maxwinds(p_patch, p_nh_state%prog(nnow(jg))%vn,   &
          p_nh_state%prog(nnow(jg))%w, lacc=i_am_accel_node)
      ENDIF

      ! total number of dynamics substeps since last boundary update
      ! applicable to refined domains only
      ndyn_substeps_tot = (jstep-1)*ndyn_substeps_var(jg) + nstep

      ! nullify prep_adv fields at first substep
      lclean_mflx = (nstep==1)
      l_recompute = lclean_mflx

      ! logical checking for the last substep
      llast = (nstep==ndyn_substeps_var(jg))

      ! save massflux at first substep
      lsave_mflx = (p_patch%n_childdom > 0 .AND. nstep == 1 )



      ! integrate dynamical core
      !IF (.NOT. ldeepatmo) THEN ! shallow atmosphere
        CALL solve_nh(p_nh_state, p_patch, p_int_state, prep_adv,     &
          &           nnow(jg), nnew(jg), linit_dyn(jg), l_recompute, &
          &           lsave_mflx, lprep_adv, lclean_mflx,             &
          &           nstep, ndyn_substeps_tot-1, dt_dyn)
      !END IF

      ! now reset linit_dyn to .FALSE.
      linit_dyn(jg) = .FALSE.

      ! compute diffusion at every dynamics substep (.NOT. lhdiff_rcf)
      IF (diffusion_config(jg)%lhdiff_vn .AND. .NOT. lhdiff_rcf) THEN

        ! Use here the dynamics substep time step dt_dyn, for which the diffusion is computed here.
        CALL diffusion(p_nh_state%prog(nnew(jg)), p_nh_state%diag, &
          &            p_nh_state%metrics, p_patch, p_int_state,   &
          &            dt_dyn, .FALSE.)

      ENDIF

      IF (advection_config(jg)%lfull_comp) &

        CALL prepare_tracer( p_patch, p_nh_state%prog(nnow(jg)),        &! in
          &                  p_nh_state%prog(nnew(jg)),                 &! in
          &                  p_nh_state%metrics, p_int_state,           &! in
          &                  ndyn_substeps_var(jg), llast, lclean_mflx, &! in
          &                  advection_config(jg)%lfull_comp,           &! in
          &                  p_nh_state%diag,                           &! inout
          &                  prep_adv%vn_traj, prep_adv%mass_flx_me,    &! inout
          &                  prep_adv%mass_flx_ic                       )! inout


      ! Finally, switch between time levels now and new for next iteration
      !
      ! Note, that we do not swap during the very last iteration.
      ! This final swap is postponed till the end of the integration step.
      IF ( .NOT. llast ) THEN
        CALL swap(nnow(jg), nnew(jg))
      ENDIF

    END DO SUBSTEPS

    IF ( ANY((/MODE_IAU,MODE_IAU_OLD/)==init_mode) ) THEN
      IF (cur_time > dt_iau) lready_for_checkpoint = .TRUE.
    ELSE
      lready_for_checkpoint = .TRUE.
    ENDIF

    ! airmass_new
    CALL compute_airmass(p_patch   = p_patch,                       & !in
      &                  p_metrics = p_nh_state%metrics,            & !in
      &                  rho       = p_nh_state%prog(nnew(jg))%rho, & !in
      &                  airmass   = p_nh_state%diag%airmass_new    ) !inout


  END SUBROUTINE perform_dyn_substepping


  !-------------------------------------------------------------------------
  !>
  !! Driver routine for initial call of physics routines.
  !! Apart from the full set of slow physics parameterizations, also turbulent transfer is
  !! called, in order to have proper transfer coefficients available at the initial time step.
  !!
  !! This had to be moved ahead of the initial output for the physics fields to be more complete
  RECURSIVE SUBROUTINE init_slowphysics (mtime_current, jg, dt_loc, lacc)

    CHARACTER(len=*), PARAMETER :: routine = modname//':init_slowphysics'

    TYPE(datetime), POINTER :: mtime_current
    INTEGER , INTENT(IN)    :: jg           !< current grid level
    REAL(wp), INTENT(IN)    :: dt_loc       !< time step applicable to local grid level
    LOGICAL, INTENT(IN) :: lacc

    ! Local variables
    INTEGER                             :: n_now_rcf
    INTEGER                             :: jgp, jgc, jn
    REAL(wp)                            :: dt_sub       !< (advective) timestep for next finer grid level


    ! Determine parent domain ID
    IF ( jg > 1) THEN
      jgp = p_patch(jg)%parent_id
    ELSE IF (n_dom_start == 0) THEN
      jgp = 0
    ELSE
      jgp = 1
    ENDIF

    ! Set local variable for rcf-time levels
    n_now_rcf = nnow_rcf(jg)


    !initial call of (slow) physics, domain ', jg

    SELECT CASE (iforcing)

    CASE (inwp) ! iforcing
      !
      ! nwp physics, slow physics forcing
      CALL nwp_nh_interface(atm_phy_nwp_config(jg)%lcall_phy(:), & !in
          &                  .TRUE.,                             & !in
          &                  lredgrid_phys(jg),                  & !in
          &                  dt_loc,                             & !in
          &                  dt_phy(jg,:),                       & !in
          &                  mtime_current,                      & !in
          &                  p_patch(jg)  ,                      & !in
          &                  p_int_state(jg),                    & !in
          &                  p_nh_state(jg)%metrics ,            & !in
          &                  p_patch(jgp),                       & !in
          &                  ext_data(jg)           ,            & !in
          &                  p_nh_state(jg)%prog(nnow(jg)) ,     & !inout
          &                  p_nh_state(jg)%prog(n_now_rcf) ,    & !inout
          &                  p_nh_state(jg)%prog(n_now_rcf) ,    & !inout
          &                  p_nh_state(jg)%diag,                & !inout
          &                  prm_diag  (jg),                     & !inout
          &                  prm_nwp_tend(jg)                ,   &
          &                  prm_nwp_stochconv(jg),              &
          &                  p_lnd_state(jg)%diag_lnd,           &
          &                  p_lnd_state(jg)%prog_lnd(n_now_rcf),& !inout
          &                  p_lnd_state(jg)%prog_lnd(n_now_rcf),& !inout
          &                  p_lnd_state(jg)%prog_wtr(n_now_rcf),& !inout
          &                  p_lnd_state(jg)%prog_wtr(n_now_rcf),& !inout
          &                  p_nh_state_lists(jg)%prog_list(n_now_rcf), & !in
          &                  lacc=lacc) !in

    END SELECT ! iforcing

    ! Boundary interpolation of land state variables entering into radiation computation
    ! if a reduced grid is used in the child domain(s)
    DO jn = 1, p_patch(jg)%n_childdom

      jgc = p_patch(jg)%child_id(jn)
      IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

      IF ( lredgrid_phys(jgc) ) THEN
        CALL interpol_rrg_grf(jg, jgc, jn, nnow_rcf(jg), lacc=lacc)
        IF (atm_phy_nwp_config(jgc)%latm_above_top) THEN
          CALL copy_rrg_ubc(jg, jgc)
        ENDIF
      ENDIF
    ENDDO

    IF (p_patch(jg)%n_childdom > 0) THEN

      dt_sub     = dt_loc/2._wp    ! dyn. time step on next refinement level

      DO jn = 1, p_patch(jg)%n_childdom

        jgc = p_patch(jg)%child_id(jn)
        IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

        IF(p_patch(jgc)%domain_is_owned) THEN
          IF(proc_split) CALL push_glob_comm(p_patch(jgc)%comm, p_patch(jgc)%proc0)
          CALL init_slowphysics( mtime_current, jgc, dt_sub, lacc=lacc )
          IF(proc_split) CALL pop_glob_comm()
        ENDIF

      ENDDO

    ENDIF

  END SUBROUTINE init_slowphysics

  !-------------------------------------------------------------------------
  !>
  !! Diagnostic computations for output - dynamics fields
  !!
  !! This routine encapsulates calls to diagnostic computations required at output
  !! times only
  SUBROUTINE diag_for_output_dyn ()

    CHARACTER(len=*), PARAMETER ::  &
     &  routine = 'mo_nh_stepping:diag_for_output_dyn'

    ! Local variables
    INTEGER :: jg, jgc, jn ! loop indices
    INTEGER :: jc, jv, jk, jb
    INTEGER :: rl_start, rl_end
    INTEGER :: i_startblk, i_endblk, i_startidx, i_endidx
    INTEGER :: nlev
    INTEGER :: idamtr_t1mc_divh, idamtr_t1mc_gradh

    REAL(wp), DIMENSION(:,:,:), POINTER  :: p_vn   => NULL()

    DO jg = 1, n_dom

      IF (.NOT. p_patch(jg)%domain_is_owned .OR. .NOT. p_patch(jg)%ldom_active) CYCLE

      nlev = p_patch(jg)%nlev

      p_vn  => p_nh_state(jg)%prog(nnow(jg))%vn


      ! Reconstruct zonal and meridional wind components
      !
      ! - wind
      CALL rbf_vec_interpol_cell(p_vn,p_patch(jg),p_int_state(jg),&
                                 p_nh_state(jg)%diag%u,p_nh_state(jg)%diag%v)
      !
      ! - wind tendencies, if fields exist, testing for the ua component is sufficient
      !
      IF (p_nh_state(jg)%diag%ddt_ua_dyn_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_dyn)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_dyn, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_dyn, &
              &                     p_nh_state(jg)%diag%ddt_va_dyn)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_dmp_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_dmp)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_dmp, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_dmp, &
              &                     p_nh_state(jg)%diag%ddt_va_dmp)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_hdf_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_hdf)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_hdf, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_hdf, &
              &                     p_nh_state(jg)%diag%ddt_va_hdf)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_adv_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_adv)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_adv, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_adv, &
              &                     p_nh_state(jg)%diag%ddt_va_adv)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_cor_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_cor)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_cor, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_cor, &
              &                     p_nh_state(jg)%diag%ddt_va_cor)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_pgr_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_pgr)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_pgr, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_pgr, &
              &                     p_nh_state(jg)%diag%ddt_va_pgr)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_phd_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_phd)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_phd, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_phd, &
              &                     p_nh_state(jg)%diag%ddt_va_phd)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_cen_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_cen)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_cen, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_cen, &
              &                     p_nh_state(jg)%diag%ddt_va_cen)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_iau_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_iau)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_iau, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_iau, &
              &                     p_nh_state(jg)%diag%ddt_va_iau)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_ray_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_ray)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_ray, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_ray, &
              &                     p_nh_state(jg)%diag%ddt_va_ray)
      END IF
      !
      IF (p_nh_state(jg)%diag%ddt_ua_grf_is_associated) THEN
         CALL sync_patch_array(SYNC_E, p_patch(jg), p_nh_state(jg)%diag%ddt_vn_grf)
         CALL rbf_vec_interpol_cell(p_nh_state(jg)%diag%ddt_vn_grf, &
              &                     p_patch(jg), p_int_state(jg),   &
              &                     p_nh_state(jg)%diag%ddt_ua_grf, &
              &                     p_nh_state(jg)%diag%ddt_va_grf)
      END IF


      !CALL div(p_vn, p_patch(jg), p_int_state(jg), p_nh_state(jg)%diag%div)
      CALL div_avg(p_vn, p_patch(jg), p_int_state(jg),p_int_state(jg)%c_bln_avg,&
                                                          p_nh_state(jg)%diag%div)

      CALL rot_vertex (p_vn, p_patch(jg), p_int_state(jg), p_nh_state(jg)%diag%omega_z)

      ! Diagnose relative vorticity on cells
      CALL verts2cells_scalar(p_nh_state(jg)%diag%omega_z, p_patch(jg), &
        p_int_state(jg)%verts_aw_cells, p_nh_state(jg)%diag%vor)

      CALL diagnose_pres_temp (p_nh_state(jg)%metrics, p_nh_state(jg)%prog(nnow(jg)), &
        &                      p_nh_state(jg)%prog(nnow_rcf(jg)),                     &
        &                      p_nh_state(jg)%diag,p_patch(jg),                       &
        &                      opt_calc_temp=.TRUE.,                                  &
        &                      opt_calc_pres=.TRUE.,                                  &
        &                      opt_lconstgrav=upatmo_config(jg)%dyn%l_constgrav       )

    ENDDO ! jg-loop

    ! Fill boundaries of nested domains
    DO jg = n_dom, 1, -1

      IF (.NOT. p_patch(jg)%domain_is_owned .OR. p_patch(jg)%n_childdom == 0) CYCLE
      IF (.NOT. p_patch(jg)%ldom_active) CYCLE

      CALL sync_patch_array_mult(SYNC_C, p_patch(jg), 3, p_nh_state(jg)%diag%u,      &
        p_nh_state(jg)%diag%v, p_nh_state(jg)%diag%div, opt_varname="u, v and div")

      DO jn = 1, p_patch(jg)%n_childdom
        jgc = p_patch(jg)%child_id(jn)
        IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

        CALL interpol_scal_grf (p_patch(jg), p_patch(jgc), p_grf_state(jg)%p_dom(jn), 3, &
             p_nh_state(jg)%diag%u, p_nh_state(jgc)%diag%u, p_nh_state(jg)%diag%v,       &
             p_nh_state(jgc)%diag%v, p_nh_state(jg)%diag%div, p_nh_state(jgc)%diag%div)

      ENDDO

    ENDDO ! jg-loop
  END SUBROUTINE diag_for_output_dyn
  !-------------------------------------------------------------------------
  !>
  !! Wrapper for computation of aggregated land variables
  !!
  SUBROUTINE aggr_landvars(lacc)

    LOGICAL, INTENT(IN)   :: lacc

    ! Local variables
    INTEGER :: jg ! loop indices

    DO jg = 1, n_dom

      IF (.NOT. p_patch(jg)%domain_is_owned) CYCLE
      IF (.NOT. p_patch(jg)%ldom_active) CYCLE

      IF (  atm_phy_nwp_config(jg)%inwp_surface == 1 ) THEN
        CALL aggregate_landvars( p_patch(jg), ext_data(jg),                 &
             p_lnd_state(jg)%prog_lnd(nnow_rcf(jg)), p_lnd_state(jg)%diag_lnd, &
             lacc=lacc)
      ENDIF

    ENDDO ! jg-loop
  END SUBROUTINE aggr_landvars

  !-------------------------------------------------------------------------
  !>
  !! Fills nest boundary cells for physics fields
  !!
  SUBROUTINE fill_nestlatbc_phys(lacc)

    LOGICAL,   INTENT(IN)   :: lacc
    ! Local variables
    INTEGER :: jg, jgc, jn ! loop indices

#ifdef _OPENACC
    IF( lacc /= i_am_accel_node ) CALL finish ( 'fill_nestlatbc_phys', 'lacc /= i_am_accel_node' )
#endif

    ! Fill boundaries of nested domains
    DO jg = n_dom, 1, -1

      IF (.NOT. p_patch(jg)%domain_is_owned .OR. p_patch(jg)%n_childdom == 0) CYCLE
      IF (.NOT. p_patch(jg)%ldom_active) CYCLE

      CALL sync_patch_array(SYNC_C, p_patch(jg), p_nh_state(jg)%prog(nnow_rcf(jg))%tke)

      DO jn = 1, p_patch(jg)%n_childdom
        jgc = p_patch(jg)%child_id(jn)
        IF (.NOT. p_patch(jgc)%ldom_active) CYCLE

        !$ACC WAIT
        CALL interpol_phys_grf(ext_data, jg, jgc, jn, lacc=lacc)

        IF (lfeedback(jgc) .AND. ifeedback_type==1) CALL feedback_phys_diag(jgc, jg, lacc=lacc)

        CALL interpol_scal_grf (p_patch(jg), p_patch(jgc), p_grf_state(jg)%p_dom(jn), 1, &
           p_nh_state(jg)%prog(nnow_rcf(jg))%tke, p_nh_state(jgc)%prog(nnow_rcf(jgc))%tke)

      ENDDO

    ENDDO ! jg-loop
  END SUBROUTINE fill_nestlatbc_phys


  !-------------------------------------------------------------------------
  !>
  !! Update of vertical wind offcentering and divergence damping
  !!
  !! This routine handles the increased sound-wave damping (by increasing the vertical wind offcentering)
  !! and mixed second-order/fourth-order divergence damping during the initial spinup phase
  SUBROUTINE update_spinup_damping(elapsed_time)

    REAL(wp), INTENT(IN) :: elapsed_time
    REAL(wp) :: time1, time2

    time1 = 1800._wp  ! enhanced damping during the first half hour of integration
    time2 = 7200._wp  ! linear decrease of enhanced damping until time2

    IF (elapsed_time <= time1) THEN ! apply slightly super-implicit weights
      divdamp_fac_o2 = 8._wp*divdamp_fac
    ELSE IF (elapsed_time <= time2) THEN ! linearly decrease minimum weights to 0.5
      divdamp_fac_o2 = 8._wp*divdamp_fac*(time2-elapsed_time)/(time2-time1)
    ELSE
      divdamp_fac_o2 = 0._wp
    ENDIF


  END SUBROUTINE update_spinup_damping


  !-------------------------------------------------------------------------
  !> Auxiliary routine to encapsulate initialization of exner_pr variable
  !!
  SUBROUTINE init_exner_pr(jg, nnow, use_acc)
    INTEGER, INTENT(IN) :: jg, nnow ! domain ID / time step indicator
    LOGICAL, INTENT(IN) :: use_acc  ! if True, use openACC
    INTEGER :: i,j,k,ie,je,ke

   ie = SIZE(p_nh_state(jg)%diag%exner_pr, 1)
   je = SIZE(p_nh_state(jg)%diag%exner_pr, 2)
   ke = SIZE(p_nh_state(jg)%diag%exner_pr, 3)

  !$ACC PARALLEL PRESENT(p_nh_state) ASYNC(1) IF(use_acc)
  !$ACC LOOP GANG VECTOR COLLAPSE(3)
  DO k = 1, ke
    DO j = 1, je
      DO i = 1, ie
        p_nh_state(jg)%diag%exner_pr(i,j,k) = &
          & p_nh_state(jg)%prog(nnow)%exner(i,j,k) - &
          & REAL(p_nh_state(jg)%metrics%exner_ref_mc(i,j,k), wp)
      END DO
    END DO
  END DO
  !$ACC END PARALLEL

  END SUBROUTINE init_exner_pr

  !-------------------------------------------------------------------------
  !> Driver routine to reset the model to its initial state if IAU iteration is selected
  !!
  SUBROUTINE reset_to_initial_state(datetime_current)

    TYPE(datetime), POINTER :: datetime_current
    INTEGER :: jg

    nnow(:)     = 1
    nnow_rcf(:) = 1
    nnew(:)     = 2
    nnew_rcf(:) = 2

   ! Reset model to initial state, repeat IAU with full incrementation window


    atm_phy_nwp_config(:)%lcalc_acc_avg = .FALSE.

    CALL restore_initial_state(p_patch(1:), p_nh_state, prm_diag, prm_nwp_tend, prm_nwp_stochconv, p_lnd_state, ext_data, lhn_fields)

    ! Reinitialize time-dependent ensemble perturbations if necessary
    IF (use_ensemble_pert .AND. gribout_config(1)%perturbationNumber >= 1) THEN
      CALL compute_ensemble_pert(p_patch(1:), ext_data, prm_diag, phy_params, datetime_current, .FALSE.)
    ENDIF

    DO jg=1, n_dom
      IF (.NOT. p_patch(jg)%ldom_active) CYCLE

      CALL diagnose_pres_temp (p_nh_state(jg)%metrics, p_nh_state(jg)%prog(nnow(jg)), &
        &                      p_nh_state(jg)%prog(nnow_rcf(jg)),                     &
        &                      p_nh_state(jg)%diag,p_patch(jg),                       &
        &                      opt_calc_temp=.TRUE.,                                  &
        &                      opt_calc_pres=.TRUE.,                                  &
        &                      opt_lconstgrav=upatmo_config(jg)%dyn%l_constgrav       )

      CALL rbf_vec_interpol_cell(p_nh_state(jg)%prog(nnow(jg))%vn,p_patch(jg),p_int_state(jg),&
                                 p_nh_state(jg)%diag%u,p_nh_state(jg)%diag%v)

      ! init airmass_new (diagnose airmass from \rho(now)). airmass_now not needed
      CALL compute_airmass(p_patch   = p_patch(jg),                       & !in
        &                  p_metrics = p_nh_state(jg)%metrics,            & !in
        &                  rho       = p_nh_state(jg)%prog(nnow(jg))%rho, & !in
        &                  airmass   = p_nh_state(jg)%diag%airmass_new    ) !inout

      CALL init_exner_pr(jg, nnow(jg), use_acc=.TRUE.)

#ifdef _OPENACC
      CALL message('reset_to_initial_state', "Copy data to CPU for init_nwp_phy")
      CALL gpu_d2h_nh_nwp(p_patch(jg), prm_diag(jg), ext_data=ext_data(jg), lacc=i_am_accel_node)
      i_am_accel_node = .FALSE.
#endif
      CALL init_nwp_phy(                            &
           & p_patch(jg)                           ,&
           & p_nh_state(jg)%metrics                ,&
           & p_nh_state(jg)%prog(nnow(jg))         ,&
           & p_nh_state(jg)%diag                   ,&
           & prm_diag(jg)                          ,&
           & prm_nwp_tend(jg)                      ,&
           & p_lnd_state(jg)%prog_lnd(nnow_rcf(jg)),&
           & p_lnd_state(jg)%prog_lnd(nnew_rcf(jg)),&
           & p_lnd_state(jg)%prog_wtr(nnow_rcf(jg)),&
           & p_lnd_state(jg)%prog_wtr(nnew_rcf(jg)),&
           & p_lnd_state(jg)%diag_lnd              ,&
           & ext_data(jg)                          ,&
           & phy_params(jg)                        ,&
           & datetime_current                      ,&
           & lreset=.TRUE.                          )

#ifdef _OPENACC
      CALL message('reset_to_initial_state', "Copy reinitialized data back to GPU after init_nwp_phy")
      i_am_accel_node = my_process_is_work()
      CALL gpu_h2d_nh_nwp(p_patch(jg), prm_diag(jg), ext_data=ext_data(jg), phy_params=phy_params(jg), &
                          atm_phy_nwp_config=atm_phy_nwp_config(jg), lacc=i_am_accel_node)
#endif

    ENDDO

    CALL aggr_landvars(lacc=.TRUE.)

    CALL init_slowphysics (datetime_current, 1, dtime, lacc=.TRUE.)

    CALL fill_nestlatbc_phys(lacc=.TRUE.)

  END SUBROUTINE reset_to_initial_state

  !-------------------------------------------------------------------------
  !> Control routine for adaptive number of dynamic substeps
  !!
  SUBROUTINE set_ndyn_substeps(lcfl_watch_mode,lspinup)

    LOGICAL, INTENT(INOUT) :: lcfl_watch_mode
    LOGICAL, INTENT(IN) :: lspinup

    INTEGER :: jg, ndyn_substeps_enh
    REAL(wp) :: mvcfl(n_dom), thresh1_cfl, thresh2_cfl
    LOGICAL :: lskip

    lskip = .FALSE.

    thresh1_cfl = MERGE(0.95_wp,1.05_wp,lspinup)
    thresh2_cfl = MERGE(0.90_wp,0.95_wp,lspinup)
    ndyn_substeps_enh = MERGE(1,0,lspinup)

    mvcfl(1:n_dom) = p_nh_state(1:n_dom)%diag%max_vcfl_dyn

    p_nh_state(1:n_dom)%diag%max_vcfl_dyn = 0._vp

    mvcfl = global_max(mvcfl)
    IF (ANY(mvcfl(1:n_dom) > 0.85_wp) .AND. .NOT. lcfl_watch_mode) THEN
      !High CFL number for vertical advection in dynamical core, entering watch mode
      lcfl_watch_mode = .TRUE.
    ENDIF

    IF (lcfl_watch_mode) THEN
      DO jg = 1, n_dom
        IF (mvcfl(jg) > 0.95_wp .OR. ndyn_substeps_var(jg) > ndyn_substeps) THEN
          !Maximum vertical CFL number in domain '
        ENDIF
        IF (mvcfl(jg) > thresh1_cfl) THEN
          ndyn_substeps_var(jg) = MIN(ndyn_substeps_var(jg)+1,ndyn_substeps_max+ndyn_substeps_enh)
          advection_config(jg)%ivcfl_max = MIN(ndyn_substeps_var(jg),ndyn_substeps_max)
          !Number of dynamics substeps in domain '
        ENDIF
        IF (ndyn_substeps_var(jg) > ndyn_substeps .AND.                                            &
            mvcfl(jg)*REAL(ndyn_substeps_var(jg),wp)/REAL(ndyn_substeps_var(jg)-1,wp) < thresh2_cfl) THEN
          ndyn_substeps_var(jg) = ndyn_substeps_var(jg)-1
          advection_config(jg)%ivcfl_max = ndyn_substeps_var(jg)
          !Number of dynamics substeps in domain jg,' decreased to ', ndyn_substeps_var(jg)
          lskip = .TRUE.
        ENDIF
      ENDDO
    ENDIF

    IF (ALL(ndyn_substeps_var(1:n_dom) == ndyn_substeps) .AND. ALL(mvcfl(1:n_dom) < 0.8_wp) .AND. &
        lcfl_watch_mode .AND. .NOT. lskip) THEN
      !CFL number for vertical advection has decreased, leaving watch mode
      lcfl_watch_mode = .FALSE.
    ENDIF

  END SUBROUTINE set_ndyn_substeps

  !-------------------------------------------------------------------------
  SUBROUTINE allocate_nh_stepping(mtime_current)

    TYPE(datetime),     POINTER          :: mtime_current     !< current datetime (mtime)

    INTEGER                              :: jg
    INTEGER                              :: ist
    CHARACTER(len=32)       :: attname   ! attribute name
    TYPE(t_key_value_store), POINTER :: restartAttributes
    CHARACTER(len=*), PARAMETER :: routine = modname//': perform_nh_stepping'

    !-----------------------------------------------------------------------
    ! allocate axiliary fields for transport
    !
    ALLOCATE(jstep_adv(n_dom), STAT=ist )
    IF (ist /= SUCCESS) THEN
      CALL finish(routine, 'allocation for jstep_adv failed' )
    ENDIF

    ! allocate flow control variables for transport and slow physics calls
    ALLOCATE(linit_dyn(n_dom), STAT=ist )
    IF (ist /= SUCCESS) THEN
      CALL finish(routine, 'allocation for flow control variables failed')
    ENDIF
    !
    ! initialize
    CALL getAttributesForRestarting(restartAttributes)
    IF (restartAttributes%is_init) THEN
      !
      ! Get attributes from restart file
      DO jg = 1,n_dom
        !ndyn_substeps_DOM',jg
        CALL restartAttributes%get(attname, ndyn_substeps_var(jg))
        !jstep_adv_marchuk_order_DOM',jg
        CALL restartAttributes%get(attname, jstep_adv(jg)%marchuk_order)
      ENDDO
      linit_dyn(:)      = .FALSE.
    ELSE
      jstep_adv(:)%marchuk_order = 0
      linit_dyn(:)               = .TRUE.
    ENDIF

    DO jg=1, n_dom

#ifndef __NO_ICON_UPATMO__
      ! upper-atmosphere physics
      IF (isRestart() .AND. upatmo_config(jg)%nwp_phy%l_phy_stat( iUpatmoPrcStat%enabled )) THEN
        CALL upatmoRestartAttributesGet(jg, prm_upatmo(jg), mtime_current)
      ENDIF
#endif
    ENDDO

  END SUBROUTINE allocate_nh_stepping
  !-------------------------------------------------------------------------

  !-------------------------------------------------------------------------
  SUBROUTINE init_ddt_vn_diagnostics(p_nh_diag)
    TYPE(t_nh_diag), INTENT(inout) :: p_nh_diag  !< p_nh_state(jg)%diag
  END SUBROUTINE init_ddt_vn_diagnostics

  !-----------------------------------------------------------------------------

END MODULE mo_nh_stepping

