!>
!! mo_solve_nonhydro
!!
!! This module contains the nonhydrostatic dynamical core for the triangular version
!! Its routines were previously contained in mo_divergent_modes and mo_vector_operations
!! but have been extracted for better memory efficiency
!!
!! @author Guenther Zaengl, DWD
!!
!! @par Revision History
!! Initial release by Guenther Zaengl (2010-10-13) based on earlier work
!! by Almut Gassmann, MPI-M
!! Modification by William Sawyer, CSCS (2015-02-06)
!! - OpenACC implementation
!!
!! @par Copyright and License
!!
!! This code is subject to the DWD and MPI-M-Software-License-Agreement in
!! its most recent form.
!! Please see the file LICENSE in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the
!! headers of the routines.
!!

!----------------------------
#include "omp_definitions.inc"
!----------------------------

MODULE mo_solve_nonhydro

  USE mo_kind,                 ONLY: wp, vp
  USE mo_nonhydrostatic_config,ONLY: itime_scheme,iadv_rhotheta, igradp_method, l_open_ubc, &
                                     kstart_moist, lhdiff_rcf, divdamp_order,               &
                                     divdamp_fac, divdamp_fac2, divdamp_fac3, divdamp_fac4, &
                                     divdamp_z, divdamp_z2, divdamp_z3, divdamp_z4,         &
                                     divdamp_type, rayleigh_type, rhotheta_offctr,          &
                                     veladv_offctr, divdamp_fac_o2, kstart_dd3d, ndyn_substeps_var
  USE mo_dynamics_config,   ONLY: idiv_method
  USE mo_parallel_config,   ONLY: nproma, p_test_run, itype_comm, use_dycore_barrier, &
    & cpu_min_nproma
  USE mo_run_config,        ONLY: ltimer, timers_level, lvert_nest
  USE mo_model_domain,      ONLY: t_patch
  USE mo_grid_config,       ONLY: l_limited_area
  USE mo_gridref_config,    ONLY: grf_intmethod_e
  USE mo_interpol_config,   ONLY: nudge_max_coeff
  USE mo_intp_data_strc,    ONLY: t_int_state
  USE mo_intp,              ONLY: cells2verts_scalar
  USE mo_nonhydro_types,    ONLY: t_nh_state
  USE mo_physical_constants,ONLY: cpd, rd, cvd, cvd_o_rd, grav, rd_o_cpd, p0ref
  USE mo_math_gradients,    ONLY: grad_green_gauss_cell
  USE mo_velocity_advection,ONLY: velocity_tendencies
  USE mo_math_constants,    ONLY: dbl_eps
  USE mo_math_divrot,       ONLY: div_avg
  USE mo_vertical_grid,     ONLY: nrdmax, nflat_gradp
  USE mo_init_vgrid,        ONLY: nflatlev
  USE mo_loopindices,       ONLY: get_indices_c, get_indices_e
  USE mo_impl_constants,    ONLY: min_rlcell_int, min_rledge_int, min_rlvert_int, &
    &                             min_rlcell, min_rledge, RAYLEIGH_CLASSIC, RAYLEIGH_KLEMP
  USE mo_impl_constants_grf,ONLY: grf_bdywidth_c, grf_bdywidth_e
  USE mo_advection_hflux,   ONLY: upwind_hflux_miura3
  USE mo_advection_traj,    ONLY: t_back_traj, btraj_compute_o1
  USE mo_sync,              ONLY: SYNC_E, SYNC_C, sync_patch_array,                             &
                                  sync_patch_array_mult, sync_patch_array_mult_mp
  USE mo_mpi,               ONLY: my_process_is_mpi_all_seq, work_mpi_barrier, i_am_accel_node
  USE mo_timer,             ONLY: timer_solve_nh, timer_barrier, timer_start, timer_stop,       &
                                  timer_solve_nh_cellcomp, timer_solve_nh_edgecomp,             &
                                  timer_solve_nh_vnupd, timer_solve_nh_vimpl, timer_solve_nh_exch
  USE mo_exception,         ONLY: message
  USE mo_icon_comm_lib,     ONLY: icon_comm_sync
  USE mo_vertical_coord_table,ONLY: vct_a
  USE mo_prepadv_types,     ONLY: t_prepare_adv
  USE mo_initicon_config,   ONLY: is_iau_active, iau_wgt_dyn
  USE mo_fortran_tools,     ONLY: init_zero_contiguous_dp, init_zero_contiguous_sp ! Import both for mixed prec.
#ifdef _OPENACC
  USE mo_mpi,               ONLY: my_process_is_work
#endif


  USE cudafor
  USE nvtx

  IMPLICIT NONE

  PRIVATE


  REAL(wp), PARAMETER :: rd_o_cvd = 1._wp / cvd_o_rd
  REAL(wp), PARAMETER :: cpd_o_rd = 1._wp / rd_o_cpd
  REAL(wp), PARAMETER :: rd_o_p0ref = rd / p0ref
  REAL(wp), PARAMETER :: grav_o_cpd = grav / cpd

  PUBLIC :: solve_nh

  CONTAINS


  !>
  !! solve_nh
  !!
  !! Main solver routine for nonhydrostatic dynamical core
  !!
  !! @par Revision History
  !! Development started by Guenther Zaengl on 2010-02-03
  !! Modification by Sebastian Borchert, DWD (2017-07-07)
  !! (Dear developer, for computational efficiency reasons, a copy of this subroutine
  !! exists in 'src/atm_dyn_iconam/mo_nh_deepatmo_solve'. If you would change something here,
  !! please consider to apply your development there, too, in order to help preventing
  !! the copy from diverging and becoming a code corpse sooner or later. Thank you!)
  !!
  SUBROUTINE solve_nh (p_nh, p_patch, p_int, prep_adv, nnow, nnew, l_init, l_recompute, lsave_mflx, &
                       lprep_adv, lclean_mflx, idyn_timestep, jstep, dtime)

    TYPE(t_nh_state),    TARGET, INTENT(INOUT) :: p_nh
    TYPE(t_int_state),   TARGET, INTENT(IN)    :: p_int
    TYPE(t_patch),       TARGET, INTENT(INOUT) :: p_patch
    TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv

    ! Initialization switch that has to be .TRUE. at the initial time step only (not for restart)
    LOGICAL,                   INTENT(IN)    :: l_init
    ! Switch to recompute velocity tendencies after a physics call irrespective of the time scheme option
    LOGICAL,                   INTENT(IN)    :: l_recompute
    ! Switch if mass flux needs to be saved for nest boundary interpolation tendency computation
    LOGICAL,                   INTENT(IN)    :: lsave_mflx
    ! Switch if preparations for tracer advection shall be computed
    LOGICAL,                   INTENT(IN)    :: lprep_adv
    ! Switch if mass fluxes computed for tracer advection need to be reinitialized
    LOGICAL,                   INTENT(IN)    :: lclean_mflx
    ! Counter of dynamics time step within a large time step (ranges from 1 to ndyn_substeps)
    INTEGER,                   INTENT(IN)    :: idyn_timestep
    ! Time step count since last boundary interpolation (ranges from 0 to 2*ndyn_substeps-1)
    INTEGER,                   INTENT(IN)    :: jstep
    ! Time levels
    INTEGER,                   INTENT(IN)    :: nnow, nnew
    ! Dynamics time step
    REAL(wp),                  INTENT(IN)    :: dtime

    ! Local variables
    INTEGER  :: jb, jk, jc, je, jks, jg
    INTEGER  :: nlev, nlevp1              !< number of full levels
    INTEGER  :: i_startblk, i_endblk, i_startidx, i_endidx, ishift
    INTEGER  :: rl_start, rl_end, istep, ntl1, ntl2, nvar, nshift, nshift_total
    INTEGER  :: i_startblk_2, i_endblk_2, i_startidx_2, i_endidx_2, rl_start_2, rl_end_2
    INTEGER  :: ic, ie, ilc0, ibc0, ikp1, ikp2

    REAL(wp) :: z_theta_v_fl_e  (nproma,p_patch%nlev  ,p_patch%nblks_e), &
                z_theta_v_e     (nproma,p_patch%nlev  ,p_patch%nblks_e), &
                z_rho_e         (nproma,p_patch%nlev  ,p_patch%nblks_e), &
                z_mass_fl_div   (nproma,p_patch%nlev  ,p_patch%nblks_c), & ! used for idiv_method=2 only
                z_theta_v_fl_div(nproma,p_patch%nlev  ,p_patch%nblks_c), & ! used for idiv_method=2 only
                z_theta_v_v     (nproma,p_patch%nlev  ,p_patch%nblks_v), & ! used for iadv_rhotheta=1 only
                z_rho_v         (nproma,p_patch%nlev  ,p_patch%nblks_v)    ! used for iadv_rhotheta=1 only

#if !defined (__LOOP_EXCHANGE) && !defined (__SX__)
    TYPE(t_back_traj), SAVE :: btraj
#endif

    ! The data type vp (variable precision) is by default the same as wp but reduces
    ! to single precision when the __MIXED_PRECISION cpp flag is set at compile time

    REAL(vp) :: z_th_ddz_exner_c(nproma,p_patch%nlev,p_patch%nblks_c), &
                z_dexner_dz_c (2,nproma,p_patch%nlev,p_patch%nblks_c), &
                z_vt_ie         (nproma,p_patch%nlev,p_patch%nblks_e), &
                z_kin_hor_e     (nproma,p_patch%nlev,p_patch%nblks_e), &
                z_exner_ex_pr (nproma,p_patch%nlevp1,p_patch%nblks_c), & ! nlevp1 is intended here
                z_gradh_exner   (nproma,p_patch%nlev,p_patch%nblks_e), &
                z_rth_pr      (2,nproma,p_patch%nlev,p_patch%nblks_c), &
                z_grad_rth    (4,nproma,p_patch%nlev,p_patch%nblks_c), &
                z_w_concorr_me  (nproma,p_patch%nlev,p_patch%nblks_e)

    ! This field in addition has reversed index order (vertical first) for optimization
#ifdef __LOOP_EXCHANGE
    REAL(vp) :: z_graddiv_vn    (p_patch%nlev,nproma,p_patch%nblks_e)
#else
    REAL(vp) :: z_graddiv_vn    (nproma,p_patch%nlev,p_patch%nblks_e)
#endif

    REAL(wp) :: z_w_expl        (nproma,p_patch%nlevp1),          &
                z_thermal_exp   (nproma,p_patch%nblks_c),         &
                z_vn_avg        (nproma,p_patch%nlev  ),          &
                z_mflx_top      (nproma,p_patch%nblks_c),         &
                z_contr_w_fl_l  (nproma,p_patch%nlevp1),          &
                z_rho_expl      (nproma,p_patch%nlev  ),          &
                z_exner_expl    (nproma,p_patch%nlev  )
    REAL(wp) :: z_theta_tavg_m1, z_theta_tavg, z_rho_tavg_m1, z_rho_tavg
    REAL(wp) :: z_thermal_exp_local ! local variable to use in OpenACC loop



    ! The data type vp (variable precision) is by default the same as wp but reduces
    ! to single precision when the __MIXED_PRECISION cpp flag is set at compile time

    ! TODO :  of these, fairly easy to scalarize:  z_theta_v_pr_ic
    REAL(vp) :: z_alpha         (nproma,p_patch%nlevp1),          &
                z_beta          (nproma,p_patch%nlev  ),          &
                z_q             (nproma,p_patch%nlev  ),          &
                z_graddiv2_vn   (nproma,p_patch%nlev  ),          &
                z_theta_v_pr_ic (nproma,p_patch%nlevp1),          &
                z_exner_ic      (nproma,p_patch%nlevp1),          &
                z_w_concorr_mc  (nproma,p_patch%nlev  ),          &
                z_flxdiv_mass   (nproma,p_patch%nlev  ),          &
                z_flxdiv_theta  (nproma,p_patch%nlev  ),          &
                z_hydro_corr    (nproma,p_patch%nblks_e)

    REAL(vp) :: z_a, z_b, z_c, z_g, z_gamma,      &
                z_w_backtraj, z_theta_v_pr_mc_m1, z_theta_v_pr_mc

    REAL(vp) :: z_w_concorr_mc_m0, z_w_concorr_mc_m1, z_w_concorr_mc_m2

    REAL(wp) :: z_theta1, z_theta2, wgt_nnow_vel, wgt_nnew_vel,     &
               dt_shift, wgt_nnow_rth, wgt_nnew_rth, dthalf,        &
               r_nsubsteps, r_dtimensubsteps, scal_divdamp_o2,      &
               alin, dz32, df32, dz42, df42, bqdr, aqdr,            &
               zf, dzlin, dzqdr
    REAL(wp) :: dt_linintp_ubc               ! time increment for linear interpolation of nest UBC
    REAL(wp) :: z_raylfac(nrdmax(p_patch%id))
    REAL(wp) :: z_ntdistv_bary_1, distv_bary_1, z_ntdistv_bary_2, distv_bary_2

    REAL(wp), DIMENSION(p_patch%nlev) :: scal_divdamp, bdy_divdamp, enh_divdamp_fac
    REAL(vp) :: z_dwdz_dd(nproma,kstart_dd3d(p_patch%id):p_patch%nlev,p_patch%nblks_c)

    ! Local variables for normal wind tendencies and differentials
    REAL(wp) :: z_ddt_vn_dyn, z_ddt_vn_apc, z_ddt_vn_cor, &
      &         z_ddt_vn_pgr, z_ddt_vn_ray,               &
      &         z_d_vn_dmp, z_d_vn_iau

    !--------------------------------------------------------------------------
    ! OUT/INOUT FIELDS DSL
    !

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: exner_pr_before
    REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_c) :: z_exner_ex_pr_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1) :: z_exner_ic_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_dexner_dz_c_1_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_rth_pr_1_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_rth_pr_2_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: rho_ic_before
    REAL(wp), DIMENSION(nproma,p_patch%nlevp1) :: z_theta_v_pr_ic_before
    REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_c) :: theta_v_ic_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_th_ddz_exner_c_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_th_ddz_exner_c_before_new

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_dexner_dz_c_2_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) ::z_rho_e_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) ::z_theta_v_e_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_graddiv_vn_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_gradh_exner_before

    INTEGER, DIMENSION(:,:,:,:), POINTER :: ikoffset_dsl

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_hydro_corr_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_hydro_corr_dsl

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: vn_nnew_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_graddiv2_vn_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: vn_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_vn_avg_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: vt_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: mass_fl_e_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_theta_v_fl_e_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: vn_traj_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: mass_flx_me_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_w_concorr_me_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_e) :: vn_ie_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_vt_ie_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_kin_hor_e_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_c) :: w_concorr_c_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_flxdiv_mass_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_flxdiv_theta_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1) :: z_w_expl_before
    REAL(wp), DIMENSION(nproma,p_patch%nlevp1) :: z_contr_w_fl_l_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_beta_before
    REAL(wp), DIMENSION(nproma,p_patch%nlevp1) :: z_alpha_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1) :: w_nnew_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_rho_expl_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_exner_expl_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev) :: z_q_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: w_before

    REAL(wp), DIMENSION(nproma,p_patch%nblks_c) :: w_1

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: rho_new_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: theta_v_new_before
    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: exner_new_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: z_dwdz_dd_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_c) :: mass_flx_ic_before

    REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_c) :: exner_dyn_incr_before

    REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_c) :: w_new_before

    !
    ! OUT/INOUT FIELDS DSL
    !--------------------------------------------------------------------------


    INTEGER :: nproma_gradp, nblks_gradp, npromz_gradp, nlen_gradp, jk_start
    LOGICAL :: lcompute, lcleanup, lvn_only, lvn_pos

    ! Local variables to control vertical nesting
    LOGICAL :: l_vert_nested, l_child_vertnest

    ! Pointers
    INTEGER, POINTER   &
#ifdef HAVE_FC_ATTRIBUTE_CONTIGUOUS
      , CONTIGUOUS     &
#endif
      ::               &
      ! to cell indices
      icidx(:,:,:), icblk(:,:,:), &
      ! to edge indices
      ieidx(:,:,:), ieblk(:,:,:), &
      ! to vertex indices
      ividx(:,:,:), ivblk(:,:,:), &
      ! to vertical neighbor indices for pressure gradient computation
      ikidx(:,:,:,:),             &
      ! to quad edge indices
      iqidx(:,:,:), iqblk(:,:,:), &
      ! for igradp_method = 3
      iplev(:), ipeidx(:), ipeblk(:)
#if !defined (__LOOP_EXCHANGE) && !defined (__SX__)
! These convenience pointers are needed to avoid PGI trying to copy derived type instance btraj back from device to host
    INTEGER, POINTER  :: p_cell_idx(:,:,:), p_cell_blk(:,:,:)
    REAL(vp), POINTER :: p_distv_bary(:,:,:,:)
#endif

    !-------------------------------------------------------------------
    IF (use_dycore_barrier) THEN
      CALL timer_start(timer_barrier)
      CALL work_mpi_barrier()
      CALL timer_stop(timer_barrier)
    ENDIF
    !-------------------------------------------------------------------

#if !defined (__LOOP_EXCHANGE) && !defined (__SX__)
    CALL btraj%construct(nproma,p_patch%nlev,p_patch%nblks_e,2)
! These convenience pointers are needed to avoid PGI trying to copy derived type instance btraj back from device to host
    p_cell_idx   => btraj%cell_idx
    p_cell_blk   => btraj%cell_blk
    p_distv_bary => btraj%distv_bary
#endif

    jg = p_patch%id

    IF (lvert_nest .AND. (p_patch%nshift_total > 0)) THEN
      l_vert_nested = .TRUE.
      nshift_total  = p_patch%nshift_total
    ELSE
      l_vert_nested = .FALSE.
      nshift_total  = 0
    ENDIF
    IF (lvert_nest .AND. p_patch%n_childdom > 0 .AND.              &
      (p_patch%nshift_child > 0 .OR. p_patch%nshift_total > 0)) THEN
      l_child_vertnest = .TRUE.
      nshift = p_patch%nshift_child + 1
    ELSE
      l_child_vertnest = .FALSE.
      nshift = 0
    ENDIF
    dthalf  = 0.5_wp*dtime

    CALL message('DSL', 'start running dycore kernels')
    IF (ltimer) CALL timer_start(timer_solve_nh)

    ! Inverse value of ndyn_substeps for tracer advection precomputations
    r_nsubsteps = 1._wp/REAL(ndyn_substeps_var(jg),wp)

    ! Inverse value of dtime * ndyn_substeps_var
    r_dtimensubsteps = 1._wp/(dtime*REAL(ndyn_substeps_var(jg),wp))

    ! number of vertical levels
    nlev   = p_patch%nlev
    nlevp1 = p_patch%nlevp1

    ! Set pointers to neighbor cells
    icidx => p_patch%edges%cell_idx
    icblk => p_patch%edges%cell_blk

    ! Set pointers to neighbor edges
    ieidx => p_patch%cells%edge_idx
    ieblk => p_patch%cells%edge_blk

    ! Set pointers to vertices of an edge
    ividx => p_patch%edges%vertex_idx
    ivblk => p_patch%edges%vertex_blk

    ! Set pointer to vertical neighbor indices for pressure gradient
    ikidx => p_nh%metrics%vertidx_gradp
    ikoffset_dsl => p_nh%metrics%vertoffset_gradp_dsl

    ! Set pointers to quad edges
    iqidx => p_patch%edges%quad_idx
    iqblk => p_patch%edges%quad_blk

    ! DA: moved from below to here to get into the same ACC data section
    iplev  => p_nh%metrics%pg_vertidx
    ipeidx => p_nh%metrics%pg_edgeidx
    ipeblk => p_nh%metrics%pg_edgeblk


    ! Precompute Rayleigh damping factor
    DO jk = 2, nrdmax(jg)
       z_raylfac(jk) = 1.0_wp/(1.0_wp+dtime*p_nh%metrics%rayleigh_w(jk))
    ENDDO

    ! Fourth-order divergence damping
    !
    ! The divergence damping factor enh_divdamp_fac is defined as a profile in height z
    ! above sea level with 4 height sections:
    !
    ! enh_divdamp_fac(z) = divdamp_fac                                              !               z <= divdamp_z
    ! enh_divdamp_fac(z) = divdamp_fac  + (z-divdamp_z )* alin                      ! divdamp_z  <= z <= divdamp_z2
    ! enh_divdamp_fac(z) = divdamp_fac2 + (z-divdamp_z2)*(aqdr+(z-divdamp_z2)*bqdr) ! divdamp_z2 <= z <= divdamp_z4
    ! enh_divdamp_fac(z) = divdamp_fac4                                             ! divdamp_z4 <= z
    !
    alin = (divdamp_fac2-divdamp_fac)/(divdamp_z2-divdamp_z)
    !
    df32 = divdamp_fac3-divdamp_fac2; dz32 = divdamp_z3-divdamp_z2
    df42 = divdamp_fac4-divdamp_fac2; dz42 = divdamp_z4-divdamp_z2
    !
    bqdr = (df42*dz32-df32*dz42)/(dz32*dz42*(dz42-dz32))
    aqdr = df32/dz32-bqdr*dz32
    !
    DO jk = 1, nlev
      jks = jk + nshift_total
      zf = 0.5_wp*(vct_a(jks)+vct_a(jks+1))
      dzlin = MIN(divdamp_z2-divdamp_z ,MAX(0._wp,zf-divdamp_z ))
      dzqdr = MIN(divdamp_z4-divdamp_z2,MAX(0._wp,zf-divdamp_z2))
      !
      IF (divdamp_order == 24) THEN
        enh_divdamp_fac(jk) = MAX( 0._wp, divdamp_fac + dzlin*alin + dzqdr*(aqdr+dzqdr*bqdr) - 0.25_wp*divdamp_fac_o2 )
      ELSE
        enh_divdamp_fac(jk) =             divdamp_fac + dzlin*alin + dzqdr*(aqdr+dzqdr*bqdr)
      ENDIF
    ENDDO

    scal_divdamp(:) = - enh_divdamp_fac(:) * p_patch%geometry_info%mean_cell_area**2

    ! Time increment for backward-shifting of lateral boundary mass flux
    dt_shift = dtime*REAL(2*ndyn_substeps_var(jg)-1,wp)/2._wp    ! == dt_phy - 0.5*dtime

    ! Time increment for linear interpolation of nest UBC.
    ! The linear interpolation is of the form
    ! \phi(t) = \phi0 + (t-t0)*dphi/dt, with t=(jstep+0.5)*dtime, and t0=dt_phy
    !
    ! dt_linintp_ubc == (t-t0)
    dt_linintp_ubc = jstep*dtime - dt_shift

    ! Coefficient for reduced fourth-order divergence damping along nest boundaries
    bdy_divdamp(:) = 0.75_wp/(nudge_max_coeff + dbl_eps)*ABS(scal_divdamp(:))



    ! scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
    ! delta_x**2 is approximated by the mean cell area
    scal_divdamp_o2 = divdamp_fac_o2 * p_patch%geometry_info%mean_cell_area


    IF (p_test_run) THEN
      !$ACC KERNELS IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
      z_rho_e     = 0._wp
      z_theta_v_e = 0._wp
      z_dwdz_dd   = 0._wp
      z_graddiv_vn= 0._wp
      !$ACC END KERNELS
    ENDIF

    ! Set time levels of ddt_adv fields for call to velocity_tendencies
    IF (itime_scheme >= 4) THEN ! Velocity advection averaging nnow and nnew tendencies
      ntl1 = nnow
      ntl2 = nnew
    ELSE                        ! Velocity advection is taken at nnew only
      ntl1 = 1
      ntl2 = 1
    ENDIF

    ! Weighting coefficients for velocity advection if tendency averaging is used
    ! The off-centering specified here turned out to be beneficial to numerical
    ! stability in extreme situations
    wgt_nnow_vel = 0.5_wp - veladv_offctr ! default value for veladv_offctr is 0.25
    wgt_nnew_vel = 0.5_wp + veladv_offctr

    ! Weighting coefficients for rho and theta at interface levels in the corrector step
    ! This empirically determined weighting minimizes the vertical wind off-centering
    ! needed for numerical stability of vertical sound wave propagation
    wgt_nnew_rth = 0.5_wp + rhotheta_offctr ! default value for rhotheta_offctr is -0.1
    wgt_nnow_rth = 1._wp - wgt_nnew_rth

    DO istep = 1, 2

      IF (istep == 1) THEN ! predictor step
        IF (itime_scheme >= 6 .OR. l_init .OR. l_recompute) THEN
          IF (itime_scheme < 6 .AND. .NOT. l_init) THEN
            lvn_only = .TRUE. ! Recompute only vn tendency
          ELSE
            lvn_only = .FALSE.
          ENDIF
          CALL velocity_tendencies(p_nh%prog(nnow),p_patch,p_int,p_nh%metrics,p_nh%diag,z_w_concorr_me, &
            z_kin_hor_e,z_vt_ie,ntl1,istep,lvn_only,dtime)
        ENDIF
        nvar = nnow
      ELSE                 ! corrector step
        lvn_only = .FALSE.
        CALL velocity_tendencies(p_nh%prog(nnew),p_patch,p_int,p_nh%metrics,p_nh%diag,z_w_concorr_me, &
          z_kin_hor_e,z_vt_ie,ntl2,istep,lvn_only,dtime)
        nvar = nnew
      ENDIF


      ! Preparations for igradp_method = 3/5 (reformulated extrapolation below the ground)
      IF (istep == 1 .AND. (igradp_method == 3 .OR. igradp_method == 5)) THEN

        nproma_gradp = cpu_min_nproma(nproma,256)
        nblks_gradp  = INT(p_nh%metrics%pg_listdim/nproma_gradp)
        npromz_gradp = MOD(p_nh%metrics%pg_listdim,nproma_gradp)
        IF (npromz_gradp > 0) THEN
          nblks_gradp = nblks_gradp + 1
        ELSE
          npromz_gradp = nproma_gradp
        ENDIF

      ENDIF

      IF (timers_level > 5) CALL timer_start(timer_solve_nh_cellcomp)

      ! Computations on mass points

      rl_start = 3
      IF (istep == 1) THEN
        rl_end = min_rlcell_int - 1
      ELSE ! halo points are not needed in step 2
        rl_end = min_rlcell_int
      ENDIF

      i_startblk = p_patch%cells%start_block(rl_start)
      i_endblk   = p_patch%cells%end_block(rl_end)

      ! DSL: Instead of calling init_zero_contiguous_dp to set z_rth_pr to zero,
      ! introduce a stencil that does the same thing, but does not touch the
      ! padding, so it can be verified.

      rl_start_2 = 1
      rl_end_2 = min_rlcell

      i_startblk_2 = p_patch%cells%start_block(rl_start_2)
      i_endblk_2   = p_patch%cells%end_block(rl_end_2)

      ! initialize nest boundary points of z_rth_pr with zero
      IF (istep == 1 .AND. (jg > 1 .OR. l_limited_area)) THEN

        CALL get_indices_c(p_patch, 1, i_startblk_2, i_endblk_2, &
                           i_startidx_2, i_endidx_2, rl_start_2, rl_end_2)

            call wrap_run_mo_solve_nonhydro_stencil_01(z_rth_pr_1=z_rth_pr(:,:,1,1), z_rth_pr_2=z_rth_pr(:,:,1,2), &
                z_rth_pr_1_before=z_rth_pr_1_before(:,:,1), z_rth_pr_2_before=z_rth_pr_2_before(:,:,1), &
                vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx_2, horizontal_upper=i_endidx_2)

      ENDIF

      DO jb = i_startblk, i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
          i_startidx, i_endidx, rl_start, rl_end)

        IF (istep == 1) THEN ! to be executed in predictor step only


        call wrap_run_mo_solve_nonhydro_stencil_02(exner_exfac=p_nh%metrics%exner_exfac(:,:,1), exner=p_nh%prog(nnow)%exner(:,:,1), &
                exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1), exner_pr=p_nh%diag%exner_pr(:,:,1), z_exner_ex_pr=z_exner_ex_pr(:,:,1), &
                exner_pr_before=exner_pr_before(:,:,1), z_exner_ex_pr_before=z_exner_ex_pr_before(:,:,1), vertical_lower=1, &
                vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          ! The purpose of the extra level of exner_pr is to simplify coding for
          ! igradp_method=4/5. It is multiplied with zero and thus actually not used


    call wrap_run_mo_solve_nonhydro_stencil_03(z_exner_ex_pr=z_exner_ex_pr(:,:,1), z_exner_ex_pr_before=z_exner_ex_pr_before(:,:,1), vertical_lower=nlevp1, vertical_upper=nlevp1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)



          IF (igradp_method <= 3) THEN
            ! Perturbation Exner pressure on bottom half level
!DIR$ IVDEP



    call wrap_run_mo_solve_nonhydro_stencil_04(wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1), z_exner_ex_pr=z_exner_ex_pr(:,:,1), &
        z_exner_ic=z_exner_ic(:,:), z_exner_ic_before=z_exner_ic_before(:,:), z_exner_ic_abs_tol=1e-17_wp, &
        vertical_lower=nlevp1, vertical_upper=nlevp1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

! WS: moved full z_exner_ic calculation here to avoid OpenACC dependency on jk+1 below
!     possibly GZ will want to consider the cache ramifications of this change for CPU

    call wrap_run_mo_solve_nonhydro_stencil_05(wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1), z_exner_ex_pr=z_exner_ex_pr(:,:,1), &
          z_exner_ic=z_exner_ic(:,:), z_exner_ic_before=z_exner_ic_before(:,:), z_exner_ic_rel_tol=1e-9_wp, &
          vertical_lower=MAX(2,nflatlev(jg)), vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)



                ! First vertical derivative of perturbation Exner pressure
    call wrap_run_mo_solve_nonhydro_stencil_06(z_exner_ic=z_exner_ic(:,:), inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1), z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1), &
            z_dexner_dz_c_1_before=z_dexner_dz_c_1_before(:,:,1), vertical_lower=MAX(2,nflatlev(jg)), vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

            IF (nflatlev(jg) == 1) THEN
              ! Perturbation Exner pressure on top half level

            ENDIF

          ENDIF


    call wrap_run_mo_solve_nonhydro_stencil_07(rho=p_nh%prog(nnow)%rho(:,:,1), rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1), theta_v=p_nh%prog(nnow)%theta_v(:,:,1), &
         theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1), z_rth_pr_1=z_rth_pr(:,:,1,1), z_rth_pr_2=z_rth_pr(:,:,1,2), z_rth_pr_1_before=z_rth_pr_1_before(:,:,1), &
         z_rth_pr_2_before=z_rth_pr_2_before(:,:,1), vertical_lower=1, vertical_upper=1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)


    call wrap_run_mo_solve_nonhydro_stencil_08(wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1), &
            rho=p_nh%prog(nnow)%rho(:,:,1), rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1), theta_v=p_nh%prog(nnow)%theta_v(:,:,1), &
            theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1), rho_ic=p_nh%diag%rho_ic(:,:,1), z_rth_pr_1=z_rth_pr(:,:,1,1), z_rth_pr_2=z_rth_pr(:,:,1,2), &
            rho_ic_before=rho_ic_before(:,:,1), z_rth_pr_1_before=z_rth_pr_1_before(:,:,1), z_rth_pr_2_before=z_rth_pr_2_before(:,:,1), &
            vertical_lower=2, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)


    call wrap_run_mo_solve_nonhydro_stencil_09(wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1), z_rth_pr_2=z_rth_pr(:,:,1,2), &
             theta_v=p_nh%prog(nnow)%theta_v(:,:,1), vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1), exner_pr=p_nh%diag%exner_pr(:,:,1), &
             d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1), ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1), &
             z_theta_v_pr_ic=z_theta_v_pr_ic(:,:), theta_v_ic=p_nh%diag%theta_v_ic(:,:,1), z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1), &
             z_theta_v_pr_ic_before=z_theta_v_pr_ic_before(:,:), theta_v_ic_before=theta_v_ic_before(:,:,1), z_th_ddz_exner_c_before=z_th_ddz_exner_c_before(:,:,1), &
             z_theta_v_pr_ic_rel_tol=1e-8_wp, z_th_ddz_exner_c_rel_tol=2e-8_wp, vertical_lower=2, vertical_upper=nlev, horizontal_lower=i_startidx, &
             horizontal_upper=i_endidx)


        ELSE  ! istep = 2 - in this step, an upwind-biased discretization is used for rho_ic and theta_v_ic
          ! in order to reduce the numerical dispersion errors



      call wrap_run_mo_solve_nonhydro_stencil_10(dtime=dtime, wgt_nnew_rth=wgt_nnew_rth, wgt_nnow_rth=wgt_nnow_rth, &
              w=p_nh%prog(nnew)%w(:,:,1), w_concorr_c=p_nh%diag%w_concorr_c(:,:,1), ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1), &
              rho_now=p_nh%prog(nnow)%rho(:,:,1), rho_var=p_nh%prog(nvar)%rho(:,:,1), theta_now=p_nh%prog(nnow)%theta_v(:,:,1), theta_var=p_nh%prog(nvar)%theta_v(:,:,1), &
              wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1), theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1), vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1), &
              exner_pr=p_nh%diag%exner_pr(:,:,1), d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1), &
              rho_ic=p_nh%diag%rho_ic(:,:,1), z_theta_v_pr_ic=z_theta_v_pr_ic(:,:), z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1), theta_v_ic=p_nh%diag%theta_v_ic(:,:,1), &
              rho_ic_before=rho_ic_before(:,:,1), z_theta_v_pr_ic_before=z_theta_v_pr_ic_before(:,:), z_th_ddz_exner_c_before=z_th_ddz_exner_c_before(:,:,1), theta_v_ic_before=theta_v_ic_before(:,:,1), &
              z_th_ddz_exner_c_rel_tol=5e-6_wp, z_theta_v_pr_ic_rel_tol=5e-7_wp,&
              vertical_lower=2, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDIF ! istep = 1/2

        ! rho and theta at top level (in case of vertical nesting, upper boundary conditions
        !                             are set in the vertical solver loop)
        IF (l_open_ubc .AND. .NOT. l_vert_nested) THEN
          IF ( istep == 1 ) THEN

          ELSE ! ISTEP == 2

          ENDIF

        ENDIF

        IF (istep == 1) THEN

          ! Perturbation theta at top and surface levels

    call wrap_run_mo_solve_nonhydro_stencil_11_lower(z_theta_v_pr_ic=z_theta_v_pr_ic(:,:), &
                                                     z_theta_v_pr_ic_before=z_theta_v_pr_ic_before(:,:), &
                                                     vertical_lower=1, vertical_upper=1, &
                                                     horizontal_lower=i_startidx, horizontal_upper=i_endidx)



    call wrap_run_mo_solve_nonhydro_stencil_11_upper(wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1), z_rth_pr=z_rth_pr(:,:,1,2), &
        theta_ref_ic=p_nh%metrics%theta_ref_ic(:,:,1), z_theta_v_pr_ic=z_theta_v_pr_ic(:,:), &
        theta_v_ic=p_nh%diag%theta_v_ic(:,:,1), z_theta_v_pr_ic_before=z_theta_v_pr_ic_before(:,:), &
        theta_v_ic_before=theta_v_ic_before(:,:,1), z_theta_v_pr_ic_rel_tol=1e-9_wp , vertical_lower=nlevp1, vertical_upper=nlevp1, &
        horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          IF (igradp_method <= 3) THEN


                ! Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
    call wrap_run_mo_solve_nonhydro_stencil_12(z_theta_v_pr_ic=z_theta_v_pr_ic(:,:), d2dexdz2_fac1_mc=p_nh%metrics%d2dexdz2_fac1_mc(:,:,1), &
              d2dexdz2_fac2_mc=p_nh%metrics%d2dexdz2_fac2_mc(:,:,1), z_rth_pr_2=z_rth_pr(:,:,1,2), z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2), &
              z_dexner_dz_c_2_before=z_dexner_dz_c_2_before(:,:,1), z_dexner_dz_c_2_rel_tol=1e-9_wp,  vertical_lower=nflat_gradp(jg), &
              vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
          ENDIF

        ENDIF ! istep == 1

      ENDDO

      IF (istep == 1) THEN
        ! Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        ! at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme
        rl_start = min_rlcell_int - 2
        rl_end   = min_rlcell_int - 2

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)


    call wrap_run_mo_solve_nonhydro_stencil_13(rho=p_nh%prog(nnow)%rho(:,:,1), rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1), &
            theta_v=p_nh%prog(nnow)%theta_v(:,:,1), theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1), z_rth_pr_1=z_rth_pr(:,:,1,1), &
            z_rth_pr_2=z_rth_pr(:,:,1,2), z_rth_pr_1_before=z_rth_pr_1_before(:,:,1), z_rth_pr_2_before=z_rth_pr_2_before(:,:,1), &
            vertical_lower=1, vertical_upper= nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDDO

      ENDIF

      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_cellcomp)
        CALL timer_start(timer_solve_nh_vnupd)
      ENDIF

      ! Compute rho and theta at edges for horizontal flux divergence term
      IF (istep == 1) THEN
        IF (iadv_rhotheta == 1) THEN ! Simplified Miura scheme
          ! Compute density and potential temperature at vertices
          CALL cells2verts_scalar(p_nh%prog(nnow)%rho,p_patch, p_int%cells_aw_verts, &
            z_rho_v, opt_rlend=min_rlvert_int-1)
          CALL cells2verts_scalar(p_nh%prog(nnow)%theta_v,p_patch, p_int%cells_aw_verts, &
            z_theta_v_v, opt_rlend=min_rlvert_int-1)

        ELSE IF (iadv_rhotheta == 2) THEN ! Miura second-order upwind scheme

#if !defined (__LOOP_EXCHANGE) && !defined (__SX__) && !defined (_OPENACC)
          ! Compute backward trajectory - code is inlined for cache-based machines (see below)
          CALL btraj_compute_o1( btraj      = btraj,                 & !inout
            &                   ptr_p       = p_patch,               & !in
            &                   ptr_int     = p_int,                 & !in
            &                   p_vn        = p_nh%prog(nnow)%vn,    & !in
            &                   p_vt        = p_nh%diag%vt,          & !in
            &                   p_dthalf    = 0.5_wp*dtime,          & !in
            &                   opt_rlstart = 7,                     & !in
            &                   opt_rlend   = min_rledge_int-1,      & !in
            &                   opt_acc_async = .TRUE.               ) !in
#endif

          ! Compute Green-Gauss gradients for rho and theta
!TODO: grad_green_gauss_cell adjust...
          CALL grad_green_gauss_cell(z_rth_pr, p_patch, p_int, z_grad_rth,    &
            opt_rlstart=3, opt_rlend=min_rlcell_int-1, opt_acc_async=.TRUE.)

        ELSE IF (iadv_rhotheta == 3) THEN ! Third-order Miura scheme (does not perform well yet)

          lcompute =.TRUE.
          lcleanup =.FALSE.
          ! First call: compute backward trajectory with wind at time level nnow

          CALL upwind_hflux_miura3(p_patch, p_nh%prog(nnow)%rho, p_nh%prog(nnow)%vn, &
            p_nh%prog(nnow)%vn, REAL(p_nh%diag%vt,wp), dtime, p_int,    &
            lcompute, lcleanup, 0, z_rho_e,                    &
            opt_rlstart=7, opt_lout_edge=.TRUE. )

          ! Second call: compute only reconstructed value for flux divergence
          lcompute =.FALSE.
          lcleanup =.TRUE.
          CALL upwind_hflux_miura3(p_patch, p_nh%prog(nnow)%theta_v, p_nh%prog(nnow)%vn, &
            p_nh%prog(nnow)%vn, REAL(p_nh%diag%vt,wp), dtime, p_int,        &
            lcompute, lcleanup, 0, z_theta_v_e,                    &
            opt_rlstart=7, opt_lout_edge=.TRUE. )

        ENDIF
      ENDIF ! istep = 1

      IF (istep == 1) THEN
        ! Compute 'edge values' of density and virtual potential temperature for horizontal
        ! flux divergence term; this is included in upwind_hflux_miura3 for option 3
        IF (iadv_rhotheta <= 2) THEN

          rl_start = min_rledge_int-2
          ! Initialize halo edges with zero in order to avoid access of uninitialized array elements
          i_startblk = p_patch%edges%start_block(rl_start)
          IF (idiv_method == 1) THEN
            rl_end   = min_rledge_int-2
            i_endblk = p_patch%edges%end_block(rl_end)
          ELSE
            rl_end   = min_rledge_int-3
            i_endblk = p_patch%edges%end_block(rl_end)
          ENDIF

          IF (i_endblk >= i_startblk) THEN
            ! DSL: Instead of calling init_zero_contiguous_dp to set z_rho_e and
            ! z_theta_v_e to zero, introduce a stencil that does the same thing,
            ! but does not touch the padding, so it can be verified.

            CALL get_indices_e(p_patch, 1, i_startblk, i_endblk, &
                               i_startidx, i_endidx, rl_start, rl_end)


            call wrap_run_mo_solve_nonhydro_stencil_14(z_rho_e=z_rho_e(:,:,1), z_theta_v_e=z_theta_v_e(:,:,1), &
                z_rho_e_before=z_rho_e_before(:,:,1), z_theta_v_e_before=z_theta_v_e_before(:,:,1), &
                vertical_lower=1, vertical_upper= nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
          ENDIF

          rl_start = 7
          rl_end   = min_rledge_int-1

          i_startblk = p_patch%edges%start_block(rl_start)
          i_endblk   = p_patch%edges%end_block  (rl_end)

          ! initialize also nest boundary points with zero
          IF (jg > 1 .OR. l_limited_area) THEN
            ! DSL: Instead of calling init_zero_contiguous_dp to set z_rho_e and
            ! z_theta_v_e to zero, introduce a stencil that does the same thing,
            ! but does not touch the padding, so it can be verified.

            rl_start_2 = 1
            rl_end_2   = min_rledge_int-1

            i_startblk_2 = p_patch%edges%start_block(rl_start_2)
            i_endblk_2   = p_patch%edges%end_block  (rl_end_2)

            CALL get_indices_e(p_patch, 1, i_startblk_2, i_endblk_2, &
                               i_startidx_2, i_endidx_2, rl_start_2, rl_end_2)

            call wrap_run_mo_solve_nonhydro_stencil_15(z_rho_e=z_rho_e(:,:,1), z_theta_v_e=z_theta_v_e(:,:,1), &
                z_rho_e_before=z_rho_e_before(:,:,1), z_theta_v_e_before=z_theta_v_e_before(:,:,1), vertical_lower=1, vertical_upper= nlev, horizontal_lower=i_startidx_2, horizontal_upper=i_endidx_2)

          ENDIF

          DO jb = i_startblk, i_endblk

            CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

            IF (iadv_rhotheta == 2) THEN
              ! Operations from upwind_hflux_miura are inlined in order to process both
              ! fields in one step

                  ! line and block indices of upwind neighbor cell

                  ! distances from upwind mass point to the end point of the backward trajectory
                  ! in edge-normal and tangential directions


                  ! rotate distance vectors into local lat-lon coordinates:
                  !
                  ! component in longitudinal direction


                  ! component in latitudinal direction

                  ! Calculate "edge values" of rho and theta_v
                  ! Note: z_rth_pr contains the perturbation values of rho and theta_v,
                  ! and the corresponding gradients are stored in z_grad_rth.


                  ! Calculate "edge values" of rho and theta_v
                  ! Note: z_rth_pr contains the perturbation values of rho and theta_v,
                  ! and the corresponding gradients are stored in z_grad_rth.


            ELSE ! iadv_rhotheta = 1


                  ! Compute upwind-biased values for rho and theta starting from centered differences
                  ! Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                  ! at a second-order accurate FV discretization, but twice the length is needed for numerical
                  ! stability

            call wrap_run_mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(p_dthalf=0.5_wp*dtime, p_vn=p_nh%prog(nnow)%vn(:,:,1), p_vt=p_nh%diag%vt(:,:,1), &
              pos_on_tplane_e_1=p_int%pos_on_tplane_e(:,:,1,1), pos_on_tplane_e_2=p_int%pos_on_tplane_e(:,:,1,2), &
              primal_normal_cell_1=p_patch%edges%primal_normal_cell_x(:,:,1), &
              dual_normal_cell_1=p_patch%edges%dual_normal_cell_x(:,:,1), &
              primal_normal_cell_2=p_patch%edges%primal_normal_cell_y(:,:,1), &
              dual_normal_cell_2=p_patch%edges%dual_normal_cell_y(:,:,1), &
              rho_ref_me=p_nh%metrics%rho_ref_me(:,:,1), &
              theta_ref_me=p_nh%metrics%theta_ref_me(:,:,1), &
              z_grad_rth_1=z_grad_rth(:,:,1,1), z_grad_rth_2=z_grad_rth(:,:,1,2), z_grad_rth_3=z_grad_rth(:,:,1,3), z_grad_rth_4=z_grad_rth(:,:,1,4), &
              z_rth_pr_1=z_rth_pr(:,:,1,1), z_rth_pr_2=z_rth_pr(:,:,1,2), z_rho_e=z_rho_e(:,:,1), z_theta_v_e=z_theta_v_e(:,:,1), &
              z_rho_e_before=z_rho_e_before(:,:,1), z_theta_v_e_before=z_theta_v_e_before(:,:,1), &
              vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx,horizontal_upper=i_endidx)

          ENDDO

        ENDIF

      ELSE IF (istep == 2 .AND. lhdiff_rcf .AND. divdamp_type >= 3) THEN ! apply div damping on 3D divergence

        ! add dw/dz contribution to divergence damping term

        rl_start = 7
        rl_end   = min_rledge_int-2

        i_startblk = p_patch%edges%start_block(rl_start)
        i_endblk   = p_patch%edges%end_block  (rl_end)

        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)


      call wrap_run_mo_solve_nonhydro_stencil_17(hmask_dd3d=p_nh%metrics%hmask_dd3d(:,1), scalfac_dd3d=p_nh%metrics%scalfac_dd3d(:), &
               inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1), z_dwdz_dd=z_dwdz_dd(:,:,1), z_graddiv_vn=z_graddiv_vn(:,:,1), &
               z_graddiv_vn_before=z_graddiv_vn_before(:,:,1), vertical_lower=kstart_dd3d(jg), vertical_upper=nlev, &
               horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDDO

      ENDIF ! istep = 1/2

      ! Remaining computations at edge points

      rl_start = grf_bdywidth_e + 1   ! boundary update follows below
      rl_end   = min_rledge_int

      i_startblk = p_patch%edges%start_block(rl_start)
      i_endblk   = p_patch%edges%end_block(rl_end)

      IF (istep == 1) THEN


        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! Store values at nest interface levels
          IF (idyn_timestep == 1 .AND. l_child_vertnest) THEN

          ENDIF


    call wrap_run_mo_solve_nonhydro_stencil_18(inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1), &
             z_exner_ex_pr=z_exner_ex_pr(:,:,1), z_gradh_exner=z_gradh_exner(:,:,1), z_gradh_exner_before=z_gradh_exner_before(:,:,1), &
             vertical_lower=1, vertical_upper=nflatlev(jg)-1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          IF (igradp_method <= 3) THEN

                ! horizontal gradient of Exner pressure, including metric correction

    call wrap_run_mo_solve_nonhydro_stencil_19(inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1), &
         z_exner_ex_pr=z_exner_ex_pr(:,:,1), ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1), c_lin_e=p_int%c_lin_e(:,:,1), &
         z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1), z_gradh_exner=z_gradh_exner(:,:,1), z_gradh_exner_before=z_gradh_exner_before(:,:,1), &
         z_gradh_exner_rel_tol=1e-10_wp, vertical_lower=nflatlev(jg), vertical_upper=nflat_gradp(jg), horizontal_lower=i_startidx, &
         horizontal_upper=i_endidx)


                ! horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction

    call wrap_run_mo_solve_nonhydro_stencil_20(inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1), z_exner_ex_pr=z_exner_ex_pr(:,:,1), &
          zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1), ikoffset=ikoffset_dsl(:,:,:,1), z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1), z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2), &
          z_gradh_exner=z_gradh_exner(:,:,1), z_gradh_exner_before=z_gradh_exner_before(:,:,1), &
          vertical_lower=nflat_gradp(jg)+1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          ELSE IF (igradp_method == 4 .OR. igradp_method == 5) THEN
                ! horizontal gradient of Exner pressure, cubic/quadratic interpolation
          ENDIF

          ! compute hydrostatically approximated correction term that replaces downward extrapolation
          IF (igradp_method == 3) THEN


    call wrap_run_mo_solve_nonhydro_stencil_21(grav_o_cpd=grav_o_cpd, theta_v=p_nh%prog(nnow)%theta_v(:,:,1), ikoffset=ikoffset_dsl(:,:,:,1), &
          zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1), theta_v_ic=p_nh%diag%theta_v_ic(:,:,1), inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1), &
          inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1), z_hydro_corr=z_hydro_corr_dsl(:,:,1), z_hydro_corr_before=z_hydro_corr_before(:,:,1), &
          vertical_lower=nlev, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          ELSE IF (igradp_method == 5) THEN


          ENDIF

        ENDDO


      ENDIF ! istep = 1


      IF (istep == 1 .AND. (igradp_method == 3 .OR. igradp_method == 5)) THEN


    rl_start_2 = grf_bdywidth_e+1
    rl_end_2   = min_rledge

    i_startblk_2 = p_patch%edges%start_block(rl_start_2)
    i_endblk_2   = p_patch%edges%end_block(rl_end_2)

    CALL get_indices_e(p_patch, 1, i_startblk_2, i_endblk_2, &
                       i_startidx_2, i_endidx_2, rl_start_2, rl_end_2)

    call wrap_run_mo_solve_nonhydro_stencil_22( &
      ipeidx_dsl=p_nh%metrics%pg_edgeidx_dsl(:,:,1), &
      pg_exdist=p_nh%metrics%pg_exdist_dsl(:,:,1), &
      z_hydro_corr=z_hydro_corr_dsl(:,nlev,1), &
      z_gradh_exner=z_gradh_exner(:,:,1), &
      z_gradh_exner_before=z_gradh_exner_before(:,:,1), &
      vertical_lower=1, &
      vertical_upper=nlev, &
      horizontal_lower=i_startidx_2, &
      horizontal_upper=i_endidx_2 &
    )

      ENDIF


      ! Update horizontal velocity field: advection, Coriolis force, pressure-gradient term, and physics


      DO jb = i_startblk, i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
          i_startidx, i_endidx, rl_start, rl_end)

        IF ((itime_scheme >= 4) .AND. istep == 2) THEN ! use temporally averaged velocity advection terms


    call wrap_run_mo_solve_nonhydro_stencil_23(cpd=cpd, dtime=dtime, wgt_nnew_vel=wgt_nnew_vel, wgt_nnow_vel=wgt_nnow_vel, &
                vn_nnow=p_nh%prog(nnow)%vn(:,:,1), ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1), &
                ddt_vn_adv_ntl2=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl2), ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1), &
                z_theta_v_e=z_theta_v_e(:,:,1), z_gradh_exner=z_gradh_exner(:,:,1), vn_nnew=p_nh%prog(nnew)%vn(:,:,1), &
                vn_nnew_before=vn_nnew_before(:,:,1), vn_nnew_rel_tol=5e-8_wp, vertical_lower=1, vertical_upper=nlev, &
                horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ELSE


    call wrap_run_mo_solve_nonhydro_stencil_24(cpd=cpd, dtime=dtime, &
            vn_nnow=p_nh%prog(nnow)%vn(:,:,1), ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1), &
            ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1), &
            z_theta_v_e=z_theta_v_e(:,:,1), z_gradh_exner=z_gradh_exner(:,:,1), vn_nnew=p_nh%prog(nnew)%vn(:,:,1), &
            vn_nnew_before=vn_nnew_before(:,:,1), vn_nnew_rel_tol=5e-10_wp, vertical_lower=1, &
            vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ENDIF

        IF (lhdiff_rcf .AND. istep == 2 .AND. (divdamp_order == 4 .OR. divdamp_order == 24)) THEN ! fourth-order divergence damping
        ! Compute gradient of divergence of gradient of divergence for fourth-order divergence damping

          call wrap_run_mo_solve_nonhydro_stencil_25(geofac_grdiv=p_int%geofac_grdiv(:,:,1), z_graddiv_vn=z_graddiv_vn(:,:,1), &
            z_graddiv2_vn=z_graddiv2_vn(:,:), z_graddiv2_vn_before=z_graddiv2_vn_before(:,:), z_graddiv2_vn_rel_tol=1e-7_wp, &
            vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDIF

        IF (lhdiff_rcf .AND. istep == 2) THEN
          ! apply divergence damping if diffusion is not called every sound-wave time step
          IF (divdamp_order == 2 .OR. (divdamp_order == 24 .AND. scal_divdamp_o2 > 1.e-6_wp) ) THEN ! 2nd-order divergence damping


    call wrap_run_mo_solve_nonhydro_stencil_26(scal_divdamp_o2=scal_divdamp_o2, z_graddiv_vn=z_graddiv_vn(:,:,1), &
                  vn=p_nh%prog(nnew)%vn(:,:,1), vn_before=vn_before(:,:,1), vertical_lower=1, vn_rel_tol=1e-10_wp, &
                  vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
          ENDIF
          IF (divdamp_order == 4 .OR. (divdamp_order == 24 .AND. divdamp_fac_o2 <= 4._wp*divdamp_fac) ) THEN
            IF (l_limited_area .OR. jg > 1) THEN
              ! fourth-order divergence damping with reduced damping coefficient along nest boundary
              ! (scal_divdamp is negative whereas bdy_divdamp is positive; decreasing the divergence
              ! damping along nest boundaries is beneficial because this reduces the interference
              ! with the increased diffusion applied in nh_diffusion)


    call wrap_run_mo_solve_nonhydro_stencil_27(scal_divdamp=scal_divdamp(:), bdy_divdamp=bdy_divdamp(:), &
                nudgecoeff_e=p_int%nudgecoeff_e(:,1), z_graddiv2_vn=z_graddiv2_vn(:,:), vn=p_nh%prog(nnew)%vn(:,:,1), &
                vn_before=vn_before(:,:,1), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, &
                horizontal_upper=i_endidx)

            ELSE ! fourth-order divergence damping


              call wrap_run_mo_solve_nonhydro_4th_order_divdamp(scal_divdamp=scal_divdamp(:), &
                          z_graddiv2_vn=z_graddiv2_vn(:,:), vn=p_nh%prog(nnew)%vn(:,:,1), &
                          vn_before=vn_before(:,:,1), vn_rel_tol=1e-10_wp, vertical_lower=1, vertical_upper=nlev, &
                          horizontal_lower=i_startidx, horizontal_upper=i_endidx)
            ENDIF
          ENDIF
        ENDIF

        IF (is_iau_active) THEN ! add analysis increment from data assimilation


        call wrap_run_mo_solve_nonhydro_stencil_28(iau_wgt_dyn=iau_wgt_dyn, vn_incr=p_nh%diag%vn_incr(:,:,1), &
              vn=p_nh%prog(nnew)%vn(:,:,1), vn_before=vn_before(:,:,1), vertical_lower=1, vertical_upper=nlev, &
              horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDIF

        ! Classic Rayleigh damping mechanism for vn (requires reference state !!)
        !
        IF ( rayleigh_type == RAYLEIGH_CLASSIC ) THEN


        ENDIF
      ENDDO

      ! Boundary update of horizontal velocity
      IF (istep == 1 .AND. (l_limited_area .OR. jg > 1)) THEN
        rl_start = 1
        rl_end   = grf_bdywidth_e

        i_startblk = p_patch%edges%start_block(rl_start)
        i_endblk   = p_patch%edges%end_block(rl_end)


        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
            i_startidx, i_endidx, rl_start, rl_end)


    call wrap_run_mo_solve_nonhydro_stencil_29(dtime=dtime, grf_tend_vn=p_nh%diag%grf_tend_vn(:,:,1), vn_now=p_nh%prog(nnow)%vn(:,:,1), &
              vn_new=p_nh%prog(nnew)%vn(:,:,1), vn_new_before=vn_before(:,:,1), &
              vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDDO


      ENDIF

      ! Preparations for nest boundary interpolation of mass fluxes from parent domain
      IF (jg > 1 .AND. grf_intmethod_e >= 5 .AND. idiv_method == 1 .AND. jstep == 0 .AND. istep == 1) THEN



      ENDIF



      !-------------------------
      ! communication phase
      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_vnupd)
        CALL timer_start(timer_solve_nh_exch)
      ENDIF

      IF (itype_comm == 1) THEN
        IF (istep == 1) THEN
          CALL sync_patch_array_mult(SYNC_E,p_patch,2,p_nh%prog(nnew)%vn,z_rho_e,opt_varname="vn_nnew and z_rho_e")
        ELSE
          CALL sync_patch_array(SYNC_E,p_patch,p_nh%prog(nnew)%vn,opt_varname="vn_nnew")
        ENDIF
      ENDIF

      IF (idiv_method == 2 .AND. istep == 1) THEN
        CALL sync_patch_array(SYNC_E,p_patch,z_theta_v_e,opt_varname="z_theta_v_e")
      END IF

      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_exch)
        CALL timer_start(timer_solve_nh_edgecomp)
      ENDIF
      ! end communication phase
      !-------------------------
      rl_start = 5
      rl_end   = min_rledge_int - 2

      i_startblk = p_patch%edges%start_block(rl_start)
      i_endblk   = p_patch%edges%end_block(rl_end)

      rl_start_2 = 1
      rl_end_2   = min_rledge

      i_startblk_2 = p_patch%edges%start_block(rl_start_2)
      i_endblk_2   = p_patch%edges%end_block(rl_end_2)

      CALL get_indices_e(p_patch, 1, i_startblk_2, i_endblk_2, &
                         i_startidx_2, i_endidx_2, rl_start_2, rl_end_2)

      DO jb = i_startblk, i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                         i_startidx, i_endidx, rl_start, rl_end)

        IF (istep == 1) THEN


              ! Average normal wind components in order to get nearly second-order accurate divergence
              ! Compute gradient of divergence of vn for divergence damping
              ! RBF reconstruction of tangential wind component

    call wrap_run_mo_solve_nonhydro_stencil_30(e_flx_avg=p_int%e_flx_avg(:,:,1), vn=p_nh%prog(nnew)%vn(:,:,1), &
              geofac_grdiv=p_int%geofac_grdiv(:,:,1), rbf_vec_coeff_e=p_int%rbf_vec_coeff_e_dsl(:,:,1), z_vn_avg=z_vn_avg(:,:), &
              z_graddiv_vn=z_graddiv_vn(:,:,1),  vt=p_nh%diag%vt(:,:,1), z_vn_avg_before=z_vn_avg_before(:,:), &
              z_graddiv_vn_before=z_graddiv_vn_before(:,:,1), vt_before=vt_before(:,:,1), &
              z_vn_avg_rel_tol=3e-7_wp, z_graddiv_vn_rel_tol=1e-4_wp, z_graddiv_vn_abs_tol=1e-20_wp, vt_rel_tol=5e-8_wp, &
              vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ELSE IF (itime_scheme >= 5) THEN


        ELSE

    call wrap_run_mo_solve_nonhydro_stencil_31(e_flx_avg=p_int%e_flx_avg(:,:,1), vn=p_nh%prog(nnew)%vn(:,:,1), &
          z_vn_avg=z_vn_avg(:,:), z_vn_avg_before=z_vn_avg_before(:,:), z_vn_avg_rel_tol=3e-7_wp, vertical_lower=1, &
          vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ENDIF

        IF (idiv_method == 1) THEN  ! Compute fluxes at edges using averaged velocities
                                  ! corresponding computation for idiv_method=2 follows later

    call wrap_run_mo_solve_nonhydro_stencil_32(z_rho_e=z_rho_e(:,:,1), z_vn_avg=z_vn_avg(:,:), ddqz_z_full_e=p_nh%metrics%ddqz_z_full_e(:,:,1), &
          z_theta_v_e=z_theta_v_e(:,:,1), mass_fl_e=p_nh%diag%mass_fl_e(:,:,1), z_theta_v_fl_e=z_theta_v_fl_e(:,:,1), &
          mass_fl_e_before=mass_fl_e_before(:,:,1), z_theta_v_fl_e_before=z_theta_v_fl_e_before(:,:,1), vertical_lower=1, &
          vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          IF (lsave_mflx .AND. istep == 2) THEN ! store mass flux for nest boundary interpolation

          ENDIF

          IF (lprep_adv .AND. istep == 2) THEN ! Preprations for tracer advection
            IF (lclean_mflx) THEN

              call wrap_run_mo_solve_nonhydro_stencil_33(vn_traj=prep_adv%vn_traj(:,:,1), &
                  mass_flx_me=prep_adv%mass_flx_me(:,:,1), vn_traj_before=vn_traj_before(:,:,1), &
                  mass_flx_me_before=mass_flx_me_before(:,:,1), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx_2, horizontal_upper=i_endidx_2)

            ENDIF


    call wrap_run_mo_solve_nonhydro_stencil_34(r_nsubsteps=r_nsubsteps, z_vn_avg=z_vn_avg(:,:), mass_fl_e=p_nh%diag%mass_fl_e(:,:,1), &
              vn_traj=prep_adv%vn_traj(:,:,1), mass_flx_me=prep_adv%mass_flx_me(:,:,1), vn_traj_before=vn_traj_before(:,:,1), &
              mass_flx_me_before=mass_flx_me_before(:,:,1), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, &
              horizontal_upper=i_endidx)

          ENDIF

        ENDIF

        IF (istep == 1 .OR. itime_scheme >= 5) THEN
          ! Compute contravariant correction for vertical velocity at full levels


    call wrap_run_mo_solve_nonhydro_stencil_35(vn=p_nh%prog(nnew)%vn(:,:,1), ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1), &
              ddxt_z_full=p_nh%metrics%ddxt_z_full(:,:,1), vt=p_nh%diag%vt(:,:,1), z_w_concorr_me=z_w_concorr_me(:,:,1), &
              z_w_concorr_me_before=z_w_concorr_me_before(:,:,1), &
              z_w_concorr_me_rel_tol=1e-8_wp, vertical_lower=nflatlev(jg), &
              vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ENDIF

        IF (istep == 1) THEN
          ! Interpolate vn to interface levels and compute horizontal part of kinetic energy on edges
          ! (needed in velocity tendencies called at istep=2)


    call wrap_run_mo_solve_nonhydro_stencil_36(wgtfac_e=p_nh%metrics%wgtfac_e(:,:,1), vn=p_nh%prog(nnew)%vn(:,:,1), &
          vt=p_nh%diag%vt(:,:,1), vn_ie=p_nh%diag%vn_ie(:,:,1), z_vt_ie=z_vt_ie(:,:,1), z_kin_hor_e=z_kin_hor_e(:,:,1), &
          vn_ie_before=vn_ie_before(:,:,1), z_vt_ie_before=z_vt_ie_before(:,:,1), z_kin_hor_e_before=z_kin_hor_e_before(:,:,1), &
          vn_ie_rel_tol=1e-8_wp, z_vt_ie_abs_tol=1e-13_wp, vertical_lower=2, &
          vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          IF (.NOT. l_vert_nested) THEN
            ! Top and bottom levels
!DIR$ IVDEP

    call wrap_run_mo_solve_nonhydro_stencil_37(vn=p_nh%prog(nnew)%vn(:,:,1), vt=p_nh%diag%vt(:,:,1), &
          vn_ie=p_nh%diag%vn_ie(:,:,1), z_vt_ie=z_vt_ie(:,:,1), z_kin_hor_e=z_kin_hor_e(:,:,1), vn_ie_before=vn_ie_before(:,:,1), &
          z_vt_ie_before=z_vt_ie_before(:,:,1), z_kin_hor_e_before=z_kin_hor_e_before(:,:,1), vertical_lower=1, &
          vertical_upper=1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
    call wrap_run_mo_solve_nonhydro_stencil_38(vn=p_nh%prog(nnew)%vn(:,:,1), wgtfacq_e=p_nh%metrics%wgtfacq_e_dsl(:,:,1), &
          vn_ie=p_nh%diag%vn_ie(:,:,1), vn_ie_before=vn_ie_before(:,:,1), vn_ie_rel_tol=5e-10_wp, vertical_lower=nlevp1, &
          vertical_upper=nlevp1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          ELSE
            ! vn_ie(jk=1) is extrapolated using parent domain information in this case

          ENDIF
        ENDIF

      ENDDO


      ! Apply mass fluxes across lateral nest boundary interpolated from parent domain
      IF (jg > 1 .AND. grf_intmethod_e >= 5 .AND. idiv_method == 1) THEN



          ! This is needed for tracer mass consistency along the lateral boundaries
          IF (lprep_adv .AND. istep == 2) THEN ! subtract mass flux added previously...

          ENDIF

          IF (lprep_adv .AND. istep == 2) THEN ! ... and add the corrected one again

          ENDIF

        ENDDO

      ENDIF


      ! It turned out that it is sufficient to compute the contravariant correction in the
      ! predictor step at time level n+1; repeating the calculation in the corrector step
      ! has negligible impact on the results except in very-high resolution runs with extremely steep mountains
      IF (istep == 1 .OR. itime_scheme >= 5) THEN

        rl_start = 3
        rl_end = min_rlcell_int - 1

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

!
! This is one of the very few code divergences for OPENACC (see comment below)
!
        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
            i_startidx, i_endidx, rl_start, rl_end)

          ! ... and to interface levels


    call wrap_run_mo_solve_nonhydro_stencil_39(e_bln_c_s=p_int%e_bln_c_s(:,:,1), z_w_concorr_me=z_w_concorr_me(:,:,1), &
               wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1), w_concorr_c=p_nh%diag%w_concorr_c(:,:,1), w_concorr_c_before=w_concorr_c_before(:,:,1), &
               w_concorr_c_rel_tol=1e-11_wp, w_concorr_c_abs_tol=1e-15_wp, &
               vertical_lower=nflatlev(jg)+1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)


    call wrap_run_mo_solve_nonhydro_stencil_40(e_bln_c_s=p_int%e_bln_c_s(:,:,1), z_w_concorr_me=z_w_concorr_me(:,:,1), &
          wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1), w_concorr_c=p_nh%diag%w_concorr_c(:,:,1), &
          w_concorr_c_before=w_concorr_c_before(:,:,1), w_concorr_c_rel_tol=1e-7_wp, &
          vertical_lower=nlev+1, vertical_upper=nlev+1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDDO

      ENDIF

      IF (idiv_method == 2) THEN ! Compute fluxes at edges from original velocities
        rl_start = 7
        rl_end = min_rledge_int - 3

        i_startblk = p_patch%edges%start_block(rl_start)
        i_endblk   = p_patch%edges%end_block(rl_end)

        IF (jg > 1 .OR. l_limited_area) THEN

          CALL init_zero_contiguous_dp(&
               z_theta_v_fl_e(1,1,p_patch%edges%start_block(5)),                &
               nproma * nlev * (i_startblk - p_patch%edges%start_block(5) + 1), &
               opt_acc_async=.TRUE., lacc=i_am_accel_node)
        ENDIF

        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
            i_startidx, i_endidx, rl_start, rl_end)

          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1,nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx

              p_nh%diag%mass_fl_e(je,jk,jb) = z_rho_e(je,jk,jb)         &
                * p_nh%prog(nnew)%vn(je,jk,jb) * p_nh%metrics%ddqz_z_full_e(je,jk,jb)
              z_theta_v_fl_e(je,jk,jb)= p_nh%diag%mass_fl_e(je,jk,jb)   &
                * z_theta_v_e(je,jk,jb)

            ENDDO
          ENDDO
          !$ACC END PARALLEL

        ENDDO

      ENDIF  ! idiv_method = 2


      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_edgecomp)
        CALL timer_start(timer_solve_nh_vimpl)
      ENDIF

      IF (idiv_method == 2) THEN ! use averaged divergence - idiv_method=1 is inlined for better cache efficiency

        ! horizontal divergences of rho and rhotheta are processed in one step for efficiency
        CALL div_avg(p_nh%diag%mass_fl_e, p_patch, p_int, p_int%c_bln_avg, z_mass_fl_div, &
                     opt_in2=z_theta_v_fl_e, opt_out2=z_theta_v_fl_div, opt_rlstart=4,    &
                     opt_rlend=min_rlcell_int)
      ENDIF


      rl_start = grf_bdywidth_c+1
      rl_end   = min_rlcell_int

      i_startblk = p_patch%cells%start_block(rl_start)
      i_endblk   = p_patch%cells%end_block(rl_end)

      IF (l_vert_nested) THEN
        jk_start = 2
      ELSE
        jk_start = 1
      ENDIF

      DO jb = i_startblk, i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        IF (idiv_method == 1) THEN
        ! horizontal divergences of rho and rhotheta are inlined and processed in one step for efficiency


    call wrap_run_mo_solve_nonhydro_stencil_41(geofac_div=p_int%geofac_div(:,:,1), mass_fl_e=p_nh%diag%mass_fl_e(:,:,1), &
          z_theta_v_fl_e=z_theta_v_fl_e(:,:,1), z_flxdiv_mass=z_flxdiv_mass(:,:), z_flxdiv_theta=z_flxdiv_theta(:,:), &
          z_flxdiv_mass_before=z_flxdiv_mass_before(:,:), z_flxdiv_theta_before=z_flxdiv_theta_before(:,:), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ELSE ! idiv_method = 2 - just copy values to local 2D array


        ENDIF

        ! upper boundary conditions for rho_ic and theta_v_ic in the case of vertical nesting
        !
        ! kept constant during predictor/corrector step, and linearly interpolated for
        ! each dynamics substep.
        ! Hence, copying them every dynamics substep during the predictor step (istep=1) is sufficient.
        IF (l_vert_nested .AND. istep == 1) THEN

        ENDIF

        ! Start of vertically implicit solver part for sound-wave terms;
        ! advective terms and gravity-wave terms are treated explicitly
        !
        IF (istep == 2 .AND. (itime_scheme >= 4)) THEN


    call wrap_run_mo_solve_nonhydro_stencil_42( &
      cpd=cpd, &
      dtime=dtime, &
      wgt_nnew_vel=wgt_nnew_vel, &
      wgt_nnow_vel=wgt_nnow_vel, &
      z_w_expl=z_w_expl(:,:), &
      w_nnow=p_nh%prog(nnow)%w(:,:,jb), &
      ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1), &
      ddt_w_adv_ntl2=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl2), &
      z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb), &
      z_contr_w_fl_l=z_contr_w_fl_l(:,:), &
      rho_ic=p_nh%diag%rho_ic(:,:,jb), &
      w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb), &
      vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb), &
      z_w_expl_before=z_w_expl_before(:,:), &
      z_contr_w_fl_l_before=z_contr_w_fl_l_before(:,:), &
      z_w_expl_rel_tol=1e-11_wp, vertical_lower=2, &
      vertical_upper=nlev, horizontal_lower=i_startidx, &
      horizontal_upper=i_endidx)

        ELSE


    call wrap_run_mo_solve_nonhydro_stencil_43( &
      cpd=cpd, &
      dtime=dtime, &
      z_w_expl=z_w_expl(:,:), &
      w_nnow=p_nh%prog(nnow)%w(:,:,jb), &
      ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1), &
      z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb), &
      z_contr_w_fl_l=z_contr_w_fl_l(:,:), &
      rho_ic=p_nh%diag%rho_ic(:,:,jb), &
      w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb), &
      vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb), &
      z_w_expl_before=z_w_expl_before(:,:), &
      z_contr_w_fl_l_before=z_contr_w_fl_l_before(:,:), &
      vertical_lower=2, vertical_upper=nlev, &
      horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDIF

        ! Solver coefficients

    call wrap_run_mo_solve_nonhydro_stencil_44( &
      cvd=cvd, &
      dtime=dtime, &
      rd=rd, &
      z_beta=z_beta(:,:), &
      exner_nnow=p_nh%prog(nnow)%exner(:,:,jb), &
      rho_nnow=p_nh%prog(nnow)%rho(:,:,jb), &
      theta_v_nnow=p_nh%prog(nnow)%theta_v(:,:,jb), &
      inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb), &
      z_alpha=z_alpha(:,:), &
      vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,jb), &
      theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb), &
      rho_ic=p_nh%diag%rho_ic(:,:,jb), &
      z_beta_before=z_beta_before(:,:), &
      z_alpha_before=z_alpha_before(:,:), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)



    call wrap_run_mo_solve_nonhydro_stencil_45(z_alpha=z_alpha(:,:), z_alpha_before=z_alpha_before(:,:), &
        vertical_lower=nlevp1, vertical_upper=nlevp1, &
        horizontal_lower=i_startidx, horizontal_upper=i_endidx)
    call wrap_run_mo_solve_nonhydro_stencil_45_b(z_q=z_q(:,:), z_q_before=z_q_before(:,:), &
        vertical_lower=1, vertical_upper=1, &
        horizontal_lower=i_startidx, horizontal_upper=i_endidx)


        ! upper boundary condition for w (interpolated from parent domain in case of vertical nesting)
        ! Note: the upper b.c. reduces to w(1) = 0 in the absence of diabatic heating
        IF (l_open_ubc .AND. .NOT. l_vert_nested) THEN

        ELSE IF (.NOT. l_open_ubc .AND. .NOT. l_vert_nested) THEN


    call wrap_run_mo_solve_nonhydro_stencil_46( &
      w_nnew=p_nh%prog(nnew)%w(:,:,jb), &
      z_contr_w_fl_l=z_contr_w_fl_l(:,:), &
      w_nnew_before=w_nnew_before(:,:), &
      z_contr_w_fl_l_before=z_contr_w_fl_l_before(:,:), vertical_lower=1, vertical_upper=1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ELSE  ! l_vert_nested

        ENDIF

        ! lower boundary condition for w, consistent with contravariant correction

    call wrap_run_mo_solve_nonhydro_stencil_47( &
      w_nnew=p_nh%prog(nnew)%w(:,:,jb), &
      z_contr_w_fl_l=z_contr_w_fl_l(:,:), &
      w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb), &
      w_nnew_before=w_nnew_before(:,:), &
      z_contr_w_fl_l_before=z_contr_w_fl_l_before(:,:), &
      vertical_lower=nlevp1, vertical_upper=nlevp1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)


        ! Explicit parts of density and Exner pressure
        !
        ! Top level first

    call wrap_run_mo_solve_nonhydro_stencil_48( &
      dtime=dtime, &
      z_rho_expl=z_rho_expl(:,:), &
      z_exner_expl=z_exner_expl(:,:), &
      rho_nnow=p_nh%prog(nnow)%rho(:,:,jb), &
      inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb), &
      z_flxdiv_mass=z_flxdiv_mass(:,:), &
      z_contr_w_fl_l=z_contr_w_fl_l(:,:), &
      exner_pr=p_nh%diag%exner_pr(:,:,jb), &
      z_beta=z_beta(:,:), &
      z_flxdiv_theta=z_flxdiv_theta(:,:), &
      theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb), &
      ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb), &
      z_rho_expl_before=z_rho_expl_before(:,:), &
      z_exner_expl_before=z_exner_expl_before(:,:), &
      vertical_lower=1, vertical_upper=1, &
      horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ! Other levels

    call wrap_run_mo_solve_nonhydro_stencil_49( &
      dtime=dtime, &
      z_rho_expl=z_rho_expl(:,:), &
      z_exner_expl=z_exner_expl(:,:), &
      rho_nnow=p_nh%prog(nnow)%rho(:,:  ,jb), &
      inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:  ,jb), &
      z_flxdiv_mass=z_flxdiv_mass(:,:), &
      z_contr_w_fl_l=z_contr_w_fl_l(:,:), &
      exner_pr=p_nh%diag%exner_pr(:,:,jb), &
      z_beta=z_beta(:,:), &
      z_flxdiv_theta=z_flxdiv_theta(:,:), &
      theta_v_ic=p_nh%diag%theta_v_ic(:,:  ,jb), &
      ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb), &
      z_rho_expl_before=z_rho_expl_before(:,:), &
      z_exner_expl_before=z_exner_expl_before(:,:), &
      vertical_lower=2, vertical_upper=nlev, &
      horizontal_lower=i_startidx, horizontal_upper=i_endidx)


        IF (is_iau_active) THEN ! add analysis increments from data assimilation to density and exner pressure


    call wrap_run_mo_solve_nonhydro_stencil_50( &
      iau_wgt_dyn=iau_wgt_dyn, &
      z_rho_expl=z_rho_expl(:,:), &
      z_exner_expl=z_exner_expl(:,:), &
      rho_incr=p_nh%diag%rho_incr(:,:,jb), &
      exner_incr=p_nh%diag%exner_incr(:,:,jb), &
      z_rho_expl_before=z_rho_expl_before(:,:), &
      z_exner_expl_before=z_exner_expl_before(:,:), &
      vertical_lower=1, vertical_upper=nlev, &
      horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ENDIF

        !
        ! Solve tridiagonal matrix for w
        !
! TODO: not parallelized


    call wrap_run_mo_solve_nonhydro_stencil_52(cpd=cpd, dtime=dtime, vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1), &
              theta_v_ic=p_nh%diag%theta_v_ic(:,:,1), ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1), z_alpha=z_alpha(:,:), &
              z_beta=z_beta(:,:), z_w_expl=z_w_expl(:,:), z_exner_expl=z_exner_expl(:,:), z_q=z_q(:,:), w=p_nh%prog(nnew)%w(:,:,1), &
              z_q_before=z_q_before(:,:), w_before=w_before(:,:,1), &
              vertical_lower=2, vertical_upper=nlev, &
              horizontal_lower=i_startidx, horizontal_upper=i_endidx)


    call wrap_run_mo_solve_nonhydro_stencil_53(z_q=z_q, w=p_nh%prog(nnew)%w(:,:,1), w_before=w_before(:,:,1), &
              vertical_lower=2, vertical_upper=nlev, &
              horizontal_lower=i_startidx, horizontal_upper=i_endidx)


        ! Rayleigh damping mechanism (Klemp,Dudhia,Hassiotis: MWR136,pp.3987-4004)
        !
        IF ( rayleigh_type == RAYLEIGH_KLEMP ) THEN

!$ACC PARALLEL IF( i_am_accel_node ) DEFAULT(NONE) ASYNC(1)
!$ACC LOOP GANG VECTOR COLLAPSE(1)
DO jc = 1, nproma
  w_1(jc,jb) = p_nh%prog(nnew)%w(jc,1,jb)
ENDDO
!$ACC END PARALLEL

    call wrap_run_mo_solve_nonhydro_stencil_54(z_raylfac=z_raylfac(:), w_1=w_1(:,1), w=p_nh%prog(nnew)%w(:,:,1), &
          w_before=w_before(:,:,1), vertical_lower=2, vertical_upper=nrdmax(jg), horizontal_lower=i_startidx, &
          horizontal_upper=i_endidx)

        ! Classic Rayleigh damping mechanism for w (requires reference state !!)
        !
        ELSE IF ( rayleigh_type == RAYLEIGH_CLASSIC ) THEN


        ENDIF

        ! Results for thermodynamic variables

    call wrap_run_mo_solve_nonhydro_stencil_55(cvd_o_rd=cvd_o_rd, dtime=dtime, z_rho_expl=z_rho_expl(:,:), &
          vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1), inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1), &
          rho_ic=p_nh%diag%rho_ic(:,:,1), w=p_nh%prog(nnew)%w(:,:,1), z_exner_expl=z_exner_expl(:,:), &
          exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1), z_alpha=z_alpha(:,:), z_beta=z_beta, rho_now=p_nh%prog(nnow)%rho(:,:,1), &
          theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1), exner_now=p_nh%prog(nnow)%exner(:,:,1), rho_new=p_nh%prog(nnew)%rho(:,:,1), &
          exner_new=p_nh%prog(nnew)%exner(:,:,1), theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1), rho_new_before=rho_new_before(:,:,1), &
          exner_new_before=exner_new_before(:,:,1), theta_v_new_before=theta_v_new_before(:,:,1), vertical_lower=jk_start, &
          vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

        ! Special treatment of uppermost layer in the case of vertical nesting
        IF (l_vert_nested) THEN

        ENDIF


        ! compute dw/dz for divergence damping term
        IF (lhdiff_rcf .AND. istep == 1 .AND. divdamp_type >= 3) THEN


    call wrap_run_mo_solve_nonhydro_stencil_56_63(inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1), w=p_nh%prog(nnew)%w(:,:,1), &
            w_concorr_c=p_nh%diag%w_concorr_c(:,:,1), z_dwdz_dd=z_dwdz_dd(:,:,1), z_dwdz_dd_before=z_dwdz_dd_before(:,:,1), &
            vertical_lower=kstart_dd3d(jg), vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ENDIF

        ! Preparations for tracer advection
        IF (lprep_adv .AND. istep == 2) THEN
          IF (lclean_mflx) THEN

              call wrap_run_mo_solve_nonhydro_stencil_57(mass_flx_ic=prep_adv%mass_flx_ic(:,:,1), &
                  mass_flx_ic_before=mass_flx_ic_before(:,:,1), &
                  vertical_lower=1, vertical_upper=nlev, &
                  horizontal_lower=i_startidx, horizontal_upper=i_endidx)
          ENDIF

    call wrap_run_mo_solve_nonhydro_stencil_58(r_nsubsteps=r_nsubsteps, z_contr_w_fl_l=z_contr_w_fl_l(:,:), rho_ic=p_nh%diag%rho_ic(:,:,1), &
          vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1), w=p_nh%prog(nnew)%w(:,:,1), mass_flx_ic=prep_adv%mass_flx_ic(:,:,1), &
          mass_flx_ic_before=mass_flx_ic_before(:,:,1), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          IF (l_vert_nested) THEN
            ! Use mass flux which has been interpolated to the upper nest boundary.
            ! This mass flux is also seen by the mass continuity equation (rho).
            ! Hence, by using the same mass flux for the tracer mass continuity equations,
            ! consistency with continuity (CWC) is ensured.

        ENDIF

        ! store dynamical part of exner time increment in exner_dyn_incr
        ! the conversion into a temperature tendency is done in the NWP interface
        IF (istep == 1 .AND. idyn_timestep == 1) THEN


    call wrap_run_mo_solve_nonhydro_stencil_59(exner=p_nh%prog(nnow)%exner(:,:,1), exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1), &
          exner_dyn_incr_before=exner_dyn_incr_before(:,:,1), vertical_lower=kstart_moist(jg), &
          vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ELSE IF (istep == 2 .AND. idyn_timestep == ndyn_substeps_var(jg)) THEN

    call wrap_run_mo_solve_nonhydro_stencil_60(dtime=dtime, ndyn_substeps_var=real(ndyn_substeps_var(jg), wp), exner=p_nh%prog(nnew)%exner(:,:,1), &
          ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,1), exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1), exner_dyn_incr_before=exner_dyn_incr_before(:,:,1), &
          vertical_lower=kstart_moist(jg), vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ENDIF

        IF (istep == 2 .AND. l_child_vertnest) THEN
          ! Store values at nest interface levels

        ENDIF

      ENDDO

      ! Boundary update in case of nesting
      IF (l_limited_area .OR. jg > 1) THEN

        rl_start = 1
        rl_end   = grf_bdywidth_c

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! non-MPI-parallelized (serial) case
          IF (istep == 1 .AND. my_process_is_mpi_all_seq() ) THEN



          ELSE IF (istep == 1 ) THEN

            ! In the MPI-parallelized case, only rho and w are updated here,
            ! and theta_v is preliminarily stored on exner in order to save
            ! halo communications


    call wrap_run_mo_solve_nonhydro_stencil_61(dtime=dtime, rho_now=p_nh%prog(nnow)%rho(:,:,1), grf_tend_rho=p_nh%diag%grf_tend_rho(:,:,1), &
            theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1), grf_tend_thv=p_nh%diag%grf_tend_thv(:,:,1), w_now=p_nh%prog(nnow)%w(:,:,1), &
            grf_tend_w=p_nh%diag%grf_tend_w(:,:,1), rho_new=p_nh%prog(nnew)%rho(:,:,1), exner_new=p_nh%prog(nnew)%exner(:,:,1), &
            w_new=p_nh%prog(nnew)%w(:,:,1), rho_new_before=rho_new_before(:,:,1), exner_new_before=exner_new_before(:,:,1), &
          w_new_before=w_new_before(:,:,1), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)


    call wrap_run_mo_solve_nonhydro_stencil_62(dtime=dtime, w_now=p_nh%prog(nnow)%w(:,:,1), grf_tend_w=p_nh%diag%grf_tend_w(:,:,1), &
              w_new=p_nh%prog(nnew)%w(:,:,1), w_new_before=w_new_before(:,:,1), &
              vertical_lower=nlevp1, vertical_upper=nlevp1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)

          ENDIF

          ! compute dw/dz for divergence damping term
          IF (lhdiff_rcf .AND. istep == 1 .AND. divdamp_type >= 3) THEN


    call wrap_run_mo_solve_nonhydro_stencil_56_63(inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1), w=p_nh%prog(nnew)%w(:,:,1), &
          w_concorr_c=p_nh%diag%w_concorr_c(:,:,1), z_dwdz_dd=z_dwdz_dd(:,:,1), z_dwdz_dd_before=z_dwdz_dd_before(:,:,1), &
          vertical_lower=kstart_dd3d(jg), vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
          ENDIF

          ! Preparations for tracer advection
          !
          ! Note that the vertical mass flux at nest boundary points is required in case that
          ! vertical tracer transport precedes horizontal tracer transport.
          IF (lprep_adv .AND. istep == 2) THEN
            IF (lclean_mflx) THEN

              call wrap_run_mo_solve_nonhydro_stencil_64(mass_flx_ic=prep_adv%mass_flx_ic(:,:,1), &
                  mass_flx_ic_before=mass_flx_ic_before(:,:,1), &
		  vertical_lower=1, vertical_upper=nlevp1, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
            ENDIF


    call wrap_run_mo_solve_nonhydro_stencil_65(r_nsubsteps=r_nsubsteps, rho_ic=p_nh%diag%rho_ic(:,:,1), &
          vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1), vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1), &
          w_now=p_nh%prog(nnow)%w(:,:,1), w_new=p_nh%prog(nnew)%w(:,:,1), w_concorr_c=p_nh%diag%w_concorr_c(:,:,1), &
          mass_flx_ic=prep_adv%mass_flx_ic(:,:,1), mass_flx_ic_before=mass_flx_ic_before(:,:,1), vertical_lower=1, vertical_upper=nlev, &
          horizontal_lower=i_startidx, horizontal_upper=i_endidx)

            IF (l_vert_nested) THEN

            ENDIF
          ENDIF

        ENDDO

      ENDIF



      !-------------------------
      ! communication phase

      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_vimpl)
        CALL timer_start(timer_solve_nh_exch)
      ENDIF

      IF (itype_comm == 1) THEN
        IF (istep == 1) THEN
          IF (lhdiff_rcf .AND. divdamp_type >= 3) THEN
            ! Synchronize w and vertical contribution to divergence damping
            CALL sync_patch_array_mult(SYNC_C,p_patch,2,p_nh%prog(nnew)%w,z_dwdz_dd, &
                 &                     opt_varname="w_nnew and z_dwdz_dd")
          ELSE
            ! Only w needs to be synchronized
            CALL sync_patch_array(SYNC_C,p_patch,p_nh%prog(nnew)%w,opt_varname="w_nnew")
          ENDIF
        ELSE ! istep = 2: synchronize all prognostic variables
          CALL sync_patch_array_mult(SYNC_C,p_patch,3,p_nh%prog(nnew)%rho, &
            p_nh%prog(nnew)%exner,p_nh%prog(nnew)%w,opt_varname="rho, exner, w_nnew")
        ENDIF
      ENDIF

      IF (timers_level > 5) CALL timer_stop(timer_solve_nh_exch)

      ! end communication phase
      !-------------------------

    ENDDO ! istep-loop


    ! The remaining computations are needed for MPI-parallelized applications only
    IF ( .NOT. my_process_is_mpi_all_seq() ) THEN

      IF (l_limited_area .OR. jg > 1) THEN

        ! Index list over halo points lying in the boundary interpolation zone
        ! Note: this list typically contains at most 10 grid points

    rl_start = min_rlcell_int - 1
    rl_end   = min_rlcell

    CALL get_indices_c(p_patch, 1, 1, 1, &
                       i_startidx, i_endidx, rl_start, rl_end)

    call wrap_run_mo_solve_nonhydro_stencil_66(rd_o_cvd=rd_o_cvd, rd_o_p0ref=rd_o_p0ref, &
            bdy_halo_c=p_nh%metrics%mask_prog_halo_c_dsl_low_refin(:,1), rho=p_nh%prog(nnew)%rho(:,:,1), &
            theta_v=p_nh%prog(nnew)%theta_v(:,:,1), exner=p_nh%prog(nnew)%exner(:,:,1), &
            theta_v_before=theta_v_new_before(:,:,1), exner_before=exner_new_before(:,:,1), &
            vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, &
            horizontal_upper=i_endidx)


        rl_start = 1
        rl_end   = grf_bdywidth_c

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)


    call wrap_run_mo_solve_nonhydro_stencil_67(rd_o_cvd=rd_o_cvd, rd_o_p0ref=rd_o_p0ref, rho=p_nh%prog(nnew)%rho(:,:,1), &
            theta_v=p_nh%prog(nnew)%theta_v(:,:,1), exner=p_nh%prog(nnew)%exner(:,:,1), theta_v_before=theta_v_new_before(:,:,1), &
            exner_before=exner_new_before(:,:,1), vertical_lower=1, vertical_upper=nlev, horizontal_lower=i_startidx, horizontal_upper=i_endidx)
        ENDDO

      ENDIF

      rl_start = min_rlcell_int - 1
      rl_end   = min_rlcell

      i_startblk = p_patch%cells%start_block(rl_start)
      i_endblk   = p_patch%cells%end_block(rl_end)

      DO jb = i_startblk, i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                         i_startidx, i_endidx, rl_start, rl_end)

    call wrap_run_mo_solve_nonhydro_stencil_68( &
      mask_prog_halo_c=p_nh%metrics%mask_prog_halo_c(:,1), &
      rho_now=p_nh%prog(nnow)%rho(:,:,1), &
      theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1), &
      exner_new=p_nh%prog(nnew)%exner(:,:,1), &
      exner_now=p_nh%prog(nnow)%exner(:,:,1), &
      rho_new=p_nh%prog(nnew)%rho(:,:,1), &
      theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1), &
      cvd_o_rd=cvd_o_rd, &
      theta_v_new_before=theta_v_new_before(:,:,1), &
      vertical_lower=1, &
      vertical_upper=nlev, &
      horizontal_lower=i_startidx, &
      horizontal_upper=i_endidx &
    )


      ENDDO


    ENDIF  ! .NOT. my_process_is_mpi_all_seq()

    IF (ltimer) CALL timer_stop(timer_solve_nh)
    CALL message('DSL', 'all dycore kernels ran')



#if !defined (__LOOP_EXCHANGE) && !defined (__SX__)
    CALL btraj%destruct()
#endif

  END SUBROUTINE solve_nh

END MODULE mo_solve_nonhydro
