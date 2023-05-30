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
  !$ser verbatim USE mo_nonhydro_state, ONLY: jstep_ptr, nstep_ptr, mo_solve_nonhydro_ctr
#ifdef _OPENACC
  USE mo_mpi,               ONLY: my_process_is_work
#endif

  USE mo_mpi, ONLY: get_my_mpi_work_id

  USE cudafor
  USE nvtx

  IMPLICIT NONE

  PRIVATE


  REAL(wp), PARAMETER :: rd_o_cvd = 1._wp / cvd_o_rd
  REAL(wp), PARAMETER :: cpd_o_rd = 1._wp / rd_o_cpd
  REAL(wp), PARAMETER :: rd_o_p0ref = rd / p0ref
  REAL(wp), PARAMETER :: grav_o_cpd = grav / cpd

  PUBLIC :: solve_nh

#ifdef _CRAYFTN
#define __CRAY_FTN_VERSION (_RELEASE_MAJOR * 100 + _RELEASE_MINOR)
#endif

  ! On the vectorizing DWD-NEC the diagnostics for the tendencies of the normal wind
  ! from terms xyz, ddt_vn_xyz, is disabled by default due to the fear that the
  ! conditional storage in conditionally allocated global fields is attempted even if
  ! the condition is not given and therefore the global field not allocated. If this
  ! happens, this would results in a corrupted memory.
  ! (Requested by G. Zaengl based on earlier problems with similar constructs.)
#ifndef __SX__
#define __ENABLE_DDT_VN_XYZ__
#endif

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
#ifdef __SWAPDIM
    REAL(vp) :: z_th_ddz_exner_c(nproma,p_patch%nlev  ,p_patch%nblks_c), &
                z_dexner_dz_c   (nproma,p_patch%nlev  ,p_patch%nblks_c,2), &
                z_vt_ie         (nproma,p_patch%nlev  ,p_patch%nblks_e), &
                z_kin_hor_e     (nproma,p_patch%nlev  ,p_patch%nblks_e), &
                z_exner_ex_pr   (nproma,p_patch%nlevp1,p_patch%nblks_c), & 
                z_gradh_exner   (nproma,p_patch%nlev  ,p_patch%nblks_e), &
                z_rth_pr        (nproma,p_patch%nlev  ,p_patch%nblks_c,2), &
                z_grad_rth      (nproma,p_patch%nlev  ,p_patch%nblks_c,4), &
                z_w_concorr_me  (nproma,p_patch%nlev  ,p_patch%nblks_e)
#else
    REAL(vp) :: z_th_ddz_exner_c(nproma,p_patch%nlev,p_patch%nblks_c), &
                z_dexner_dz_c (2,nproma,p_patch%nlev,p_patch%nblks_c), &
                z_vt_ie         (nproma,p_patch%nlev,p_patch%nblks_e), &
                z_kin_hor_e     (nproma,p_patch%nlev,p_patch%nblks_e), &
                z_exner_ex_pr (nproma,p_patch%nlevp1,p_patch%nblks_c), & ! nlevp1 is intended here
                z_gradh_exner   (nproma,p_patch%nlev,p_patch%nblks_e), &
                z_rth_pr      (2,nproma,p_patch%nlev,p_patch%nblks_c), &
                z_grad_rth    (4,nproma,p_patch%nlev,p_patch%nblks_c), &
                z_w_concorr_me  (nproma,p_patch%nlev,p_patch%nblks_e)
#endif
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
                z_hydro_corr    (nproma,p_patch%nlev,p_patch%nblks_e)

    REAL(vp) :: z_a, z_b, z_c, z_g, z_gamma,      &
                z_w_backtraj, z_theta_v_pr_mc_m1, z_theta_v_pr_mc

#ifdef _OPENACC
    REAL(vp) :: z_w_concorr_mc_m0, z_w_concorr_mc_m1, z_w_concorr_mc_m2
#endif

    REAL(wp) :: z_theta1, z_theta2, wgt_nnow_vel, wgt_nnew_vel,     &
               dt_shift, wgt_nnow_rth, wgt_nnew_rth, dthalf,        &
               r_nsubsteps, r_dtimensubsteps, scal_divdamp_o2,      &
               alin, dz32, df32, dz42, df42, bqdr, aqdr,            &
               zf, dzlin, dzqdr
    ! time shifts for linear interpolation of nest UBC
    REAL(wp) :: dt_linintp_ubc, dt_linintp_ubc_nnow, dt_linintp_ubc_nnew
    REAL(wp) :: z_raylfac(nrdmax(p_patch%id))
    REAL(wp) :: z_ntdistv_bary_1, distv_bary_1, z_ntdistv_bary_2, distv_bary_2

    REAL(wp), DIMENSION(p_patch%nlev) :: scal_divdamp, bdy_divdamp, enh_divdamp_fac
    REAL(vp) :: z_dwdz_dd(nproma,kstart_dd3d(p_patch%id):p_patch%nlev,p_patch%nblks_c)

    ! Local variables for normal wind tendencies and differentials
    REAL(wp) :: z_ddt_vn_dyn, z_ddt_vn_apc, z_ddt_vn_cor, &
      &         z_ddt_vn_pgr, z_ddt_vn_ray,               &
      &         z_d_vn_dmp, z_d_vn_iau

    REAL(wp), DIMENSION(nproma, p_patch%nblks_c) :: w_1
    !--------------------------------------------------------------------------
    ! OUT/INOUT FIELDS DSL
    !


    
    INTEGER, DIMENSION(:,:,:,:), POINTER :: ikoffset_dsl

    !
    ! OUT/INOUT FIELDS DSL
    !--------------------------------------------------------------------------

#ifdef __INTEL_COMPILER
!DIR$ ATTRIBUTES ALIGN : 64 :: z_theta_v_fl_e,z_theta_v_e,z_rho_e,z_mass_fl_div
!DIR$ ATTRIBUTES ALIGN : 64 :: z_theta_v_fl_div,z_theta_v_v,z_rho_v,z_dwdz_dd
!DIR$ ATTRIBUTES ALIGN : 64 :: z_th_ddz_exner_c,z_dexner_dz_c,z_vt_ie,z_kin_hor_e
!DIR$ ATTRIBUTES ALIGN : 64 :: z_exner_ex_pr,z_gradh_exner,z_rth_pr,z_grad_rth
!DIR$ ATTRIBUTES ALIGN : 64 :: z_w_concorr_me,z_graddiv_vn,z_w_expl,z_thermal_exp
!DIR$ ATTRIBUTES ALIGN : 64 :: z_vn_avg,z_mflx_top,z_contr_w_fl_l,z_rho_expl
!DIR$ ATTRIBUTES ALIGN : 64 :: z_exner_expl,z_alpha,z_beta,z_q,z_graddiv2_vn
!DIR$ ATTRIBUTES ALIGN : 64 :: z_theta_v_pr_ic,z_exner_ic,z_w_concorr_mc
!DIR$ ATTRIBUTES ALIGN : 64 :: z_flxdiv_mass,z_flxdiv_theta,z_hydro_corr
!DIR$ ATTRIBUTES ALIGN : 64 :: z_raylfac,scal_divdamp,bdy_divdamp,enh_divdamp_fac
#endif

    INTEGER :: nproma_gradp, nblks_gradp, npromz_gradp, nlen_gradp, jk_start
    LOGICAL :: lcompute, lcleanup, lvn_only, lvn_pos

    ! Local variables to control vertical nesting
    LOGICAL :: l_vert_nested, l_child_vertnest

    ! Pointers
    INTEGER, POINTER, CONTIGUOUS :: &
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
#ifdef __SX__
      REAL(wp) :: z_rho_tavg_m1_v(nproma), z_theta_tavg_m1_v(nproma)
      REAL(vp) :: z_theta_v_pr_mc_m1_v(nproma)
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

    ! Set pointers to quad edges
    iqidx => p_patch%edges%quad_idx
    iqblk => p_patch%edges%quad_blk

    ! DA: moved from below to here to get into the same ACC data section
    iplev  => p_nh%metrics%pg_vertidx
    ipeidx => p_nh%metrics%pg_edgeidx
    ipeblk => p_nh%metrics%pg_edgeblk

    !$ser verbatim mo_solve_nonhydro_ctr = mo_solve_nonhydro_ctr + 1

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
    dt_linintp_ubc      = jstep*dtime - dt_shift ! valid for center of current time step
    dt_linintp_ubc_nnow = dt_linintp_ubc - 0.5_wp*dtime
    dt_linintp_ubc_nnew = dt_linintp_ubc + 0.5_wp*dtime

    ! Coefficient for reduced fourth-order divergence damping along nest boundaries
    bdy_divdamp(:) = 0.75_wp/(nudge_max_coeff + dbl_eps)*ABS(scal_divdamp(:))

    !$ACC DATA CREATE(z_kin_hor_e, z_vt_ie, z_w_concorr_me, z_mass_fl_div, z_theta_v_fl_e, z_theta_v_fl_div) &
    !$ACC   CREATE(z_dexner_dz_c, z_exner_ex_pr, z_gradh_exner, z_rth_pr, z_grad_rth) &
    !$ACC   CREATE(z_theta_v_pr_ic, z_th_ddz_exner_c, z_w_concorr_mc) &
    !$ACC   CREATE(z_vn_avg, z_rho_e, z_theta_v_e, z_dwdz_dd, z_thermal_exp, z_mflx_top) &
    !$ACC   CREATE(z_exner_ic, z_alpha, z_beta, z_q, z_contr_w_fl_l, z_exner_expl) &
    !$ACC   CREATE(z_flxdiv_mass, z_flxdiv_theta, z_rho_expl, z_w_expl) &
    !$ACC   CREATE(z_rho_v, z_theta_v_v, z_graddiv_vn, z_hydro_corr, z_graddiv2_vn) &
    !$ACC   CREATE(w_1) &
    !$ACC   COPYIN(nflatlev, nflat_gradp, kstart_dd3d, kstart_moist, nrdmax) &
    !$ACC   COPYIN(z_raylfac, ndyn_substeps_var, scal_divdamp, bdy_divdamp) &
#ifndef __LOOP_EXCHANGE
    !$ACC   PRESENT(p_cell_idx, p_cell_blk, p_distv_bary) &
#endif
    !$ACC   PRESENT(prep_adv, p_int, p_patch, p_nh) &
    !$ACC   PRESENT(icidx, icblk, ividx, ivblk, ieidx, ieblk, ikidx, iqidx, iqblk) &
    !$ACC   PRESENT(ipeidx, ipeblk, iplev) &
    !$ACC   IF(i_am_accel_node)


    ! scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
    ! delta_x**2 is approximated by the mean cell area
    scal_divdamp_o2 = divdamp_fac_o2 * p_patch%geometry_info%mean_cell_area


    IF (p_test_run) THEN
      !$ACC KERNELS IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
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

!$NEC sparse
    DO istep = 1, 2

      IF (istep == 1) THEN ! predictor step
        IF (itime_scheme >= 6 .OR. l_init .OR. l_recompute) THEN
          IF (itime_scheme < 6 .AND. .NOT. l_init) THEN
            lvn_only = .TRUE. ! Recompute only vn tendency
          ELSE
            lvn_only = .FALSE.
          ENDIF
          CALL velocity_tendencies(p_nh%prog(nnow),p_patch,p_int,p_nh%metrics,p_nh%diag,z_w_concorr_me, &
            z_kin_hor_e,z_vt_ie,ntl1,istep,lvn_only,dtime,dt_linintp_ubc_nnow)
        ENDIF
        nvar = nnow
      ELSE                 ! corrector step
        lvn_only = .FALSE.
        CALL velocity_tendencies(p_nh%prog(nnew),p_patch,p_int,p_nh%metrics,p_nh%diag,z_w_concorr_me, &
          z_kin_hor_e,z_vt_ie,ntl2,istep,lvn_only,dtime,dt_linintp_ubc_nnew)
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
!$OMP PARALLEL PRIVATE (rl_start,rl_end,i_startblk,i_endblk)

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


    !$ser init directory="." prefix="liskov-serialisation" mpi_rank=get_my_mpi_work_id()

    !$ser savepoint mo_solve_nonhydro_stencil_01_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)
!$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jk = 1, nlev
              DO jc = i_startidx_2, i_endidx_2
                z_rth_pr(jc,jk,1,1) = 0._wp
                z_rth_pr(jc,jk,1,2) = 0._wp
              ENDDO
            ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_01_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)
!$OMP BARRIER
      ENDIF

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc,z_exner_ic,z_theta_v_pr_ic,z_w_backtraj,&
!$OMP            z_theta_v_pr_mc_m1,z_theta_v_pr_mc,z_rho_tavg_m1,z_rho_tavg, &
#ifdef __SX__
!$OMP            z_rho_tavg_m1_v,z_theta_tavg_m1_v,z_theta_v_pr_mc_m1_v, &
#endif
!$OMP            z_theta_tavg_m1,z_theta_tavg,z_thermal_exp_local) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk, i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
          i_startidx, i_endidx, rl_start, rl_end)

        IF (istep == 1) THEN ! to be executed in predictor step only


    !$ser savepoint mo_solve_nonhydro_stencil_02_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing exner_exfac=p_nh%metrics%exner_exfac(:,:,1)'

    !$ser data exner_exfac=p_nh%metrics%exner_exfac(:,:,1)

    PRINT *, 'Serializing exner=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)'

    !$ser data exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,1)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              ! temporally extrapolated perturbation Exner pressure (used for horizontal gradients only)
              z_exner_ex_pr(jc,jk,jb) = (1._wp + p_nh%metrics%exner_exfac(jc,jk,jb)) *    &
                (p_nh%prog(nnow)%exner(jc,jk,jb) - p_nh%metrics%exner_ref_mc(jc,jk,jb)) - &
                 p_nh%metrics%exner_exfac(jc,jk,jb) * p_nh%diag%exner_pr(jc,jk,jb)

              ! non-extrapolated perturbation Exner pressure, saved in exner_pr for the next time step
              p_nh%diag%exner_pr(jc,jk,jb) = p_nh%prog(nnow)%exner(jc,jk,jb) - &
                                              p_nh%metrics%exner_ref_mc(jc,jk,jb)

            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_02_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing exner_exfac=p_nh%metrics%exner_exfac(:,:,1)'

    !$ser data exner_exfac=p_nh%metrics%exner_exfac(:,:,1)

    PRINT *, 'Serializing exner=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)'

    !$ser data exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,1)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

          ! The purpose of the extra level of exner_pr is to simplify coding for
          ! igradp_method=4/5. It is multiplied with zero and thus actually not used


    !$ser savepoint mo_solve_nonhydro_stencil_03_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)
          !$ACC KERNELS IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          z_exner_ex_pr(i_startidx:i_endidx,nlevp1,jb) = 0._wp
          !$ACC END KERNELS

    !$ser savepoint mo_solve_nonhydro_stencil_03_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

          IF (l_open_ubc .AND. .NOT. l_vert_nested) THEN
            ! Compute contribution of thermal expansion to vertical wind at model top
            ! Isothermal expansion is assumed

#ifdef _OPENACC
! Exchanging loop order to remove data dep
! TODO: evaluate if this makes sense
            !$ACC PARALLEL IF(i_am_accel_node) PRIVATE(z_thermal_exp_local) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              z_thermal_exp_local = 0._wp
              DO jk = 1, nlev
                   z_thermal_exp_local= z_thermal_exp_local + cvd_o_rd                        &
                    * p_nh%diag%ddt_exner_phy(jc,jk,jb)                                       &
                    /  (p_nh%prog(nnow)%exner(jc,jk,jb)*p_nh%metrics%inv_ddqz_z_full(jc,jk,jb))
              ENDDO
              z_thermal_exp(jc,jb) = z_thermal_exp_local
            ENDDO
            !$ACC END PARALLEL

#else
            z_thermal_exp(:,jb) = 0._wp
            DO jk = 1, nlev
!DIR$ IVDEP
              DO jc = i_startidx, i_endidx
                z_thermal_exp(jc,jb) = z_thermal_exp(jc,jb) + cvd_o_rd                      &
                  * p_nh%diag%ddt_exner_phy(jc,jk,jb)                                       &
                  /  (p_nh%prog(nnow)%exner(jc,jk,jb)*p_nh%metrics%inv_ddqz_z_full(jc,jk,jb))
              ENDDO
            ENDDO
#endif

          ENDIF

          IF (igradp_method <= 3) THEN
            ! Perturbation Exner pressure on bottom half level
!DIR$ IVDEP


    !$ser savepoint mo_solve_nonhydro_stencil_04_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)'

    !$ser data wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing z_exner_ic=z_exner_ic(:,:)'

    !$ser accdata z_exner_ic=z_exner_ic(:,:)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              z_exner_ic(jc,nlevp1) =                                         &
                p_nh%metrics%wgtfacq_c(jc,1,jb)*z_exner_ex_pr(jc,nlev  ,jb) + &
                p_nh%metrics%wgtfacq_c(jc,2,jb)*z_exner_ex_pr(jc,nlev-1,jb) + &
                p_nh%metrics%wgtfacq_c(jc,3,jb)*z_exner_ex_pr(jc,nlev-2,jb)
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_04_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)'

    !$ser data wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing z_exner_ic=z_exner_ic(:,:)'

    !$ser accdata z_exner_ic=z_exner_ic(:,:)

! WS: moved full z_exner_ic calculation here to avoid OpenACC dependency on jk+1 below
!     possibly GZ will want to consider the cache ramifications of this change for CPU

    !$ser savepoint mo_solve_nonhydro_stencil_05_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing z_exner_ic=z_exner_ic(:,:)'

    !$ser accdata z_exner_ic=z_exner_ic(:,:)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR TILE(32, 4)
            DO jk = nlev, MAX(2,nflatlev(jg)), -1
!DIR$ IVDEP
              DO jc = i_startidx, i_endidx
                ! Exner pressure on remaining half levels for metric correction term
                z_exner_ic(jc,jk) =                                                    &
                         p_nh%metrics%wgtfac_c(jc,jk,jb) *z_exner_ex_pr(jc,jk  ,jb) +  &
                  (1._vp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_exner_ex_pr(jc,jk-1,jb)
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_05_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing z_exner_ic=z_exner_ic(:,:)'

    !$ser accdata z_exner_ic=z_exner_ic(:,:)



    !$ser savepoint mo_solve_nonhydro_stencil_06_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_exner_ic=z_exner_ic(:,:)'

    !$ser accdata z_exner_ic=z_exner_ic(:,:)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)'

    !$ser accdata z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR TILE(32, 4)
            DO jk = nlev, MAX(2,nflatlev(jg)), -1
!DIR$ IVDEP
              DO jc = i_startidx, i_endidx

                ! First vertical derivative of perturbation Exner pressure
#ifdef __SWAPDIM
                z_dexner_dz_c(jc,jk,jb,1) =                     &
#else
                z_dexner_dz_c(1,jc,jk,jb) =                     &
#endif
                  (z_exner_ic(jc,jk) - z_exner_ic(jc,jk+1)) *   &
                  p_nh%metrics%inv_ddqz_z_full(jc,jk,jb)
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_06_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_exner_ic=z_exner_ic(:,:)'

    !$ser accdata z_exner_ic=z_exner_ic(:,:)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)'

    !$ser accdata z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)

            IF (nflatlev(jg) == 1) THEN
              ! Perturbation Exner pressure on top half level
!DIR$ IVDEP
              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG VECTOR
              DO jc = i_startidx, i_endidx
                z_exner_ic(jc,1) =                                          &
                  p_nh%metrics%wgtfacq1_c(jc,1,jb)*z_exner_ex_pr(jc,1,jb) + &
                  p_nh%metrics%wgtfacq1_c(jc,2,jb)*z_exner_ex_pr(jc,2,jb) + &
                  p_nh%metrics%wgtfacq1_c(jc,3,jb)*z_exner_ex_pr(jc,3,jb)

                ! First vertical derivative of perturbation Exner pressure
#ifdef __SWAPDIM
                z_dexner_dz_c(jc,1,jb,1) =                    &
#else
                z_dexner_dz_c(1,jc,1,jb) =                    &
#endif
                  (z_exner_ic(jc,1) - z_exner_ic(jc,2)) *   &
                  p_nh%metrics%inv_ddqz_z_full(jc,1,jb)
              ENDDO
              !$ACC END PARALLEL
            ENDIF

          ENDIF


    !$ser savepoint mo_solve_nonhydro_stencil_07_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rho=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)'

    !$ser data rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
#ifdef __SWAPDIM
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            z_rth_pr(jc,1,jb,1) = p_nh%prog(nnow)%rho(jc,1,jb) - &
              p_nh%metrics%rho_ref_mc(jc,1,jb)
            z_rth_pr(jc,1,jb,2) = p_nh%prog(nnow)%theta_v(jc,1,jb) - &
              p_nh%metrics%theta_ref_mc(jc,1,jb)
          ENDDO
#else
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            z_rth_pr(1,jc,1,jb) =  p_nh%prog(nnow)%rho(jc,1,jb) - &
              p_nh%metrics%rho_ref_mc(jc,1,jb)
            z_rth_pr(2,jc,1,jb) =  p_nh%prog(nnow)%theta_v(jc,1,jb) - &
              p_nh%metrics%theta_ref_mc(jc,1,jb)
          ENDDO
#endif
          !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_07_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rho=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)'

    !$ser data rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)


    !$ser savepoint mo_solve_nonhydro_stencil_08_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing rho=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)'

    !$ser data rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR TILE(32, 4)
          DO jk = 2, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              ! density at interface levels for vertical flux divergence computation
              p_nh%diag%rho_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *p_nh%prog(nnow)%rho(jc,jk  ,jb) + &
                                    (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*p_nh%prog(nnow)%rho(jc,jk-1,jb)

              ! perturbation density and virtual potential temperature at main levels for horizontal flux divergence term
              ! (needed in the predictor step only)
#ifdef __SWAPDIM
              z_rth_pr(jc,jk,jb,1) =  p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
              z_rth_pr(jc,jk,jb,2) =  p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
#else
              z_rth_pr(1,jc,jk,jb) =  p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
              z_rth_pr(2,jc,jk,jb) =  p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
#endif
#ifdef _OPENACC
            ENDDO
          ENDDO
          !$ACC END PARALLEL
#endif


    !$ser savepoint mo_solve_nonhydro_stencil_08_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing rho=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)'

    !$ser data rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)


    !$ser savepoint mo_solve_nonhydro_stencil_09_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,1)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,1)

    PRINT *, 'Serializing d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)'

    !$ser data d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)

    PRINT *, 'Serializing ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)'

    !$ser data ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)
#ifdef _OPENACC
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 2, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
#endif

              ! perturbation virtual potential temperature at interface levels
#ifdef __SWAPDIM
              z_theta_v_pr_ic(jc,jk) =                                           &
                       p_nh%metrics%wgtfac_c(jc,jk,jb) *z_rth_pr(jc,jk  ,jb,2) + &
                (1._vp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_rth_pr(jc,jk-1,jb,2)
#else
              z_theta_v_pr_ic(jc,jk) =                                           &
                       p_nh%metrics%wgtfac_c(jc,jk,jb) *z_rth_pr(2,jc,jk  ,jb) + &
                (1._vp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_rth_pr(2,jc,jk-1,jb)
#endif
              ! virtual potential temperature at interface levels
              p_nh%diag%theta_v_ic(jc,jk,jb) =                                                &
                       p_nh%metrics%wgtfac_c(jc,jk,jb) *p_nh%prog(nnow)%theta_v(jc,jk  ,jb) + &
                (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*p_nh%prog(nnow)%theta_v(jc,jk-1,jb)

              ! vertical pressure gradient * theta_v
              z_th_ddz_exner_c(jc,jk,jb) = p_nh%metrics%vwind_expl_wgt(jc,jb)* &
                p_nh%diag%theta_v_ic(jc,jk,jb) * (p_nh%diag%exner_pr(jc,jk-1,jb)-      &
                p_nh%diag%exner_pr(jc,jk,jb)) / p_nh%metrics%ddqz_z_half(jc,jk,jb) +   &
                z_theta_v_pr_ic(jc,jk)*p_nh%metrics%d_exner_dz_ref_ic(jc,jk,jb)
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_09_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,1)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,1)

    PRINT *, 'Serializing d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)'

    !$ser data d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)

    PRINT *, 'Serializing ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)'

    !$ser data ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)


        ELSE  ! istep = 2 - in this step, an upwind-biased discretization is used for rho_ic and theta_v_ic
          ! in order to reduce the numerical dispersion errors
#ifdef __SX__
          ! precompute values for jk = 1 which are previous values in first iteration of jk compute loop
          jk = 2
            DO jc = i_startidx, i_endidx
              z_rho_tavg_m1_v(jc) = wgt_nnow_rth*p_nh%prog(nnow)%rho(jc,jk-1,jb) + &
                              wgt_nnew_rth*p_nh%prog(nvar)%rho(jc,jk-1,jb)
              z_theta_tavg_m1_v(jc) = wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,jk-1,jb) + &
                                wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,jk-1,jb)
              z_theta_v_pr_mc_m1_v(jc)  = z_theta_tavg_m1_v(jc) - p_nh%metrics%theta_ref_mc(jc,jk-1,jb)
            ENDDO
#endif


    !$ser savepoint mo_solve_nonhydro_stencil_10_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing wgt_nnew_rth=wgt_nnew_rth'

    !$ser data wgt_nnew_rth=wgt_nnew_rth

    PRINT *, 'Serializing wgt_nnow_rth=wgt_nnow_rth'

    !$ser data wgt_nnow_rth=wgt_nnow_rth

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)'

    !$ser data ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_var=p_nh%prog(nvar)%rho(:,:,1)'

    !$ser data rho_var=p_nh%prog(nvar)%rho(:,:,1)

    PRINT *, 'Serializing theta_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_var=p_nh%prog(nvar)%theta_v(:,:,1)'

    !$ser data theta_var=p_nh%prog(nvar)%theta_v(:,:,1)

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,1)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,1)

    PRINT *, 'Serializing d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)'

    !$ser data d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR TILE(128, *) &
          !$ACC   PRIVATE(z_w_backtraj, z_rho_tavg_m1, z_theta_tavg_m1, z_rho_tavg) &
          !$ACC   PRIVATE(z_theta_tavg, z_theta_v_pr_mc_m1, z_theta_v_pr_mc)
          DO jk = 2, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              ! backward trajectory - use w(nnew) in order to be at the same time level as w_concorr
              z_w_backtraj = - (p_nh%prog(nnew)%w(jc,jk,jb) - p_nh%diag%w_concorr_c(jc,jk,jb)) * &
                dtime*0.5_wp/p_nh%metrics%ddqz_z_half(jc,jk,jb)

              ! temporally averaged density and virtual potential temperature depending on rhotheta_offctr
              ! (see pre-computation above)
#ifndef __SX__
              z_rho_tavg_m1 = wgt_nnow_rth*p_nh%prog(nnow)%rho(jc,jk-1,jb) + &
                              wgt_nnew_rth*p_nh%prog(nvar)%rho(jc,jk-1,jb)
              z_theta_tavg_m1 = wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,jk-1,jb) + &
                                wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,jk-1,jb)
#else
              z_rho_tavg_m1   = z_rho_tavg_m1_v(jc)
              z_theta_tavg_m1 = z_theta_tavg_m1_v(jc)
#endif

              z_rho_tavg = wgt_nnow_rth*p_nh%prog(nnow)%rho(jc,jk,jb) + &
                           wgt_nnew_rth*p_nh%prog(nvar)%rho(jc,jk,jb)
              z_theta_tavg = wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,jk,jb) + &
                             wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,jk,jb)

              ! density at interface levels for vertical flux divergence computation
              p_nh%diag%rho_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *z_rho_tavg    + &
                                    (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_rho_tavg_m1 + &
                z_w_backtraj*(z_rho_tavg_m1-z_rho_tavg)

              ! perturbation virtual potential temperature at main levels
#ifndef __SX__
              z_theta_v_pr_mc_m1  = z_theta_tavg_m1 - p_nh%metrics%theta_ref_mc(jc,jk-1,jb)
#else
              z_theta_v_pr_mc_m1 = z_theta_v_pr_mc_m1_v(jc)
#endif
              z_theta_v_pr_mc     = z_theta_tavg    - p_nh%metrics%theta_ref_mc(jc,jk,jb)

              ! perturbation virtual potential temperature at interface levels
              z_theta_v_pr_ic(jc,jk) =                                       &
                       p_nh%metrics%wgtfac_c(jc,jk,jb) *z_theta_v_pr_mc +    &
                (1._vp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_theta_v_pr_mc_m1

              ! virtual potential temperature at interface levels
              p_nh%diag%theta_v_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *z_theta_tavg    +  &
                                        (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_theta_tavg_m1 +  &
                z_w_backtraj*(z_theta_tavg_m1-z_theta_tavg)

              ! vertical pressure gradient * theta_v
              z_th_ddz_exner_c(jc,jk,jb) = p_nh%metrics%vwind_expl_wgt(jc,jb)* &
                p_nh%diag%theta_v_ic(jc,jk,jb) * (p_nh%diag%exner_pr(jc,jk-1,jb)-      &
                p_nh%diag%exner_pr(jc,jk,jb)) / p_nh%metrics%ddqz_z_half(jc,jk,jb) +   &
                z_theta_v_pr_ic(jc,jk)*p_nh%metrics%d_exner_dz_ref_ic(jc,jk,jb)

#ifdef __SX__
              ! save current values as previous values for next iteration
              z_rho_tavg_m1_v(jc) = z_rho_tavg
              z_theta_tavg_m1_v(jc) = z_theta_tavg
              z_theta_v_pr_mc_m1_v(jc) = z_theta_v_pr_mc
#endif

            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_10_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing wgt_nnew_rth=wgt_nnew_rth'

    !$ser data wgt_nnew_rth=wgt_nnew_rth

    PRINT *, 'Serializing wgt_nnow_rth=wgt_nnow_rth'

    !$ser data wgt_nnow_rth=wgt_nnow_rth

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)'

    !$ser data ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_var=p_nh%prog(nvar)%rho(:,:,1)'

    !$ser data rho_var=p_nh%prog(nvar)%rho(:,:,1)

    PRINT *, 'Serializing theta_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_var=p_nh%prog(nvar)%theta_v(:,:,1)'

    !$ser data theta_var=p_nh%prog(nvar)%theta_v(:,:,1)

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,1)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,1)

    PRINT *, 'Serializing d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)'

    !$ser data d_exner_dz_ref_ic=p_nh%metrics%d_exner_dz_ref_ic(:,:,1)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,1)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

        ENDIF ! istep = 1/2

        ! rho and theta at top level (in case of vertical nesting, upper boundary conditions
        !                             are set in the vertical solver loop)
        IF (l_open_ubc .AND. .NOT. l_vert_nested) THEN
          IF ( istep == 1 ) THEN
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
!DIR$ IVDEP
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              p_nh%diag%theta_v_ic(jc,1,jb) = &
                p_nh%metrics%theta_ref_ic(jc,1,jb)                   + &
#ifdef __SWAPDIM
                p_nh%metrics%wgtfacq1_c(jc,1,jb)*z_rth_pr(jc,1,jb,2) + &
                p_nh%metrics%wgtfacq1_c(jc,2,jb)*z_rth_pr(jc,2,jb,2) + &
                p_nh%metrics%wgtfacq1_c(jc,3,jb)*z_rth_pr(jc,3,jb,2)
#else
                p_nh%metrics%wgtfacq1_c(jc,1,jb)*z_rth_pr(2,jc,1,jb) + &
                p_nh%metrics%wgtfacq1_c(jc,2,jb)*z_rth_pr(2,jc,2,jb) + &
                p_nh%metrics%wgtfacq1_c(jc,3,jb)*z_rth_pr(2,jc,3,jb)
#endif
            ENDDO
            !$ACC END PARALLEL
          ELSE ! ISTEP == 2
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
!DIR$ IVDEP
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              p_nh%diag%theta_v_ic(jc,1,jb) = p_nh%metrics%theta_ref_ic(jc,1,jb) + &
                p_nh%metrics%wgtfacq1_c(jc,1,jb)* ( wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,1,jb) +     &
                wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,1,jb) - p_nh%metrics%theta_ref_mc(jc,1,jb) ) + &
                p_nh%metrics%wgtfacq1_c(jc,2,jb)*( wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,2,jb) +      &
                wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,2,jb) - p_nh%metrics%theta_ref_mc(jc,2,jb) ) + &
                p_nh%metrics%wgtfacq1_c(jc,3,jb)*( wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,3,jb) +      &
                wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,3,jb) - p_nh%metrics%theta_ref_mc(jc,3,jb) )
            ENDDO
            !$ACC END PARALLEL
          ENDIF
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
!DIR$ IVDEP
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            p_nh%diag%rho_ic(jc,1,jb) =  wgt_nnow_rth*(                        &
              p_nh%metrics%wgtfacq1_c(jc,1,jb)*p_nh%prog(nnow)%rho(jc,1,jb) +  &
              p_nh%metrics%wgtfacq1_c(jc,2,jb)*p_nh%prog(nnow)%rho(jc,2,jb) +  &
              p_nh%metrics%wgtfacq1_c(jc,3,jb)*p_nh%prog(nnow)%rho(jc,3,jb))+  &
              wgt_nnew_rth * (                                                 &
              p_nh%metrics%wgtfacq1_c(jc,1,jb)*p_nh%prog(nvar)%rho(jc,1,jb) +  &
              p_nh%metrics%wgtfacq1_c(jc,2,jb)*p_nh%prog(nvar)%rho(jc,2,jb) +  &
              p_nh%metrics%wgtfacq1_c(jc,3,jb)*p_nh%prog(nvar)%rho(jc,3,jb) )
          ENDDO
          !$ACC END PARALLEL
        ENDIF

        IF (istep == 1) THEN

          ! Perturbation theta at top and surface levels

    !$ser savepoint mo_solve_nonhydro_stencil_11_lower_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
!DIR$ IVDEP
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            z_theta_v_pr_ic(jc,1)      = 0._wp
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_11_lower_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)



    !$ser savepoint mo_solve_nonhydro_stencil_11_upper_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)'

    !$ser data wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)

    PRINT *, 'Serializing z_rth_pr=z_rth_pr(:,:,1,2)'

    !$ser data z_rth_pr=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing theta_ref_ic=p_nh%metrics%theta_ref_ic(:,:,1)'

    !$ser data theta_ref_ic=p_nh%metrics%theta_ref_ic(:,:,1)

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)
!$ACC PARALLEL IF( i_am_accel_node )  DEFAULT(PRESENT) ASYNC(1)
!DIR$ IVDEP
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            z_theta_v_pr_ic(jc,nlevp1) =                                   &
#ifdef __SWAPDIM
              p_nh%metrics%wgtfacq_c(jc,1,jb)*z_rth_pr(jc,nlev  ,jb,2) +     &
              p_nh%metrics%wgtfacq_c(jc,2,jb)*z_rth_pr(jc,nlev-1,jb,2) +   &
              p_nh%metrics%wgtfacq_c(jc,3,jb)*z_rth_pr(jc,nlev-2,jb,2)
#else
              p_nh%metrics%wgtfacq_c(jc,1,jb)*z_rth_pr(2,jc,nlev  ,jb) +     &
              p_nh%metrics%wgtfacq_c(jc,2,jb)*z_rth_pr(2,jc,nlev-1,jb) +   &
              p_nh%metrics%wgtfacq_c(jc,3,jb)*z_rth_pr(2,jc,nlev-2,jb)
#endif
            p_nh%diag%theta_v_ic(jc,nlevp1,jb) =                                  &
              p_nh%metrics%theta_ref_ic(jc,nlevp1,jb) + z_theta_v_pr_ic(jc,nlevp1)
          ENDDO
          !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_11_upper_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)'

    !$ser data wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)

    PRINT *, 'Serializing z_rth_pr=z_rth_pr(:,:,1,2)'

    !$ser data z_rth_pr=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing theta_ref_ic=p_nh%metrics%theta_ref_ic(:,:,1)'

    !$ser data theta_ref_ic=p_nh%metrics%theta_ref_ic(:,:,1)

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

          IF (igradp_method <= 3) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_12_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing d2dexdz2_fac1_mc=p_nh%metrics%d2dexdz2_fac1_mc(:,:,1)'

    !$ser data d2dexdz2_fac1_mc=p_nh%metrics%d2dexdz2_fac1_mc(:,:,1)

    PRINT *, 'Serializing d2dexdz2_fac2_mc=p_nh%metrics%d2dexdz2_fac2_mc(:,:,1)'

    !$ser data d2dexdz2_fac2_mc=p_nh%metrics%d2dexdz2_fac2_mc(:,:,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)'

    !$ser accdata z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR TILE(32, 4)
            DO jk = nflat_gradp(jg), nlev
!DIR$ IVDEP
              DO jc = i_startidx, i_endidx
                ! Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
#ifdef __SWAPDIM
                z_dexner_dz_c(jc,jk,jb,2) = -0.5_vp *                              &
                  ((z_theta_v_pr_ic(jc,jk) - z_theta_v_pr_ic(jc,jk+1)) *           &
                  p_nh%metrics%d2dexdz2_fac1_mc(jc,jk,jb) + z_rth_pr(jc,jk,jb,2) * &
#else
                z_dexner_dz_c(2,jc,jk,jb) = -0.5_vp *                              &
                  ((z_theta_v_pr_ic(jc,jk) - z_theta_v_pr_ic(jc,jk+1)) *           &
                  p_nh%metrics%d2dexdz2_fac1_mc(jc,jk,jb) + z_rth_pr(2,jc,jk,jb) * &
#endif
                  p_nh%metrics%d2dexdz2_fac2_mc(jc,jk,jb))
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_12_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)'

    !$ser accdata z_theta_v_pr_ic=z_theta_v_pr_ic(:,:)

    PRINT *, 'Serializing d2dexdz2_fac1_mc=p_nh%metrics%d2dexdz2_fac1_mc(:,:,1)'

    !$ser data d2dexdz2_fac1_mc=p_nh%metrics%d2dexdz2_fac1_mc(:,:,1)

    PRINT *, 'Serializing d2dexdz2_fac2_mc=p_nh%metrics%d2dexdz2_fac2_mc(:,:,1)'

    !$ser data d2dexdz2_fac2_mc=p_nh%metrics%d2dexdz2_fac2_mc(:,:,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)'

    !$ser accdata z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)
          ENDIF

        ENDIF ! istep == 1

      ENDDO
!$OMP END DO NOWAIT

      IF (istep == 1) THEN
        ! Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        ! at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme
        rl_start = min_rlcell_int - 2
        rl_end   = min_rlcell_int - 2

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)


    !$ser savepoint mo_solve_nonhydro_stencil_13_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rho=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)'

    !$ser data rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
#ifdef __SWAPDIM
              z_rth_pr(jc,jk,jb,1) = p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
              z_rth_pr(jc,jk,jb,2) = p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
#else
              z_rth_pr(1,jc,jk,jb) = p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
              z_rth_pr(2,jc,jk,jb) = p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
#endif
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_13_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rho=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)'

    !$ser data rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)'

    !$ser data theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

        ENDDO
!$OMP END DO NOWAIT

      ENDIF
!$OMP END PARALLEL

      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_cellcomp)
        CALL timer_start(timer_solve_nh_vnupd)
      ENDIF

      ! Compute rho and theta at edges for horizontal flux divergence term
      IF (istep == 1) THEN
        IF (iadv_rhotheta == 1) THEN ! Simplified Miura scheme
          !DA: TODO: remove the wait after everything is async
          !$ACC WAIT
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
#ifdef __MIXED_PRECISION
            &                   p_vt        = REAL(p_nh%diag%vt,wp), & !in    ! this results in differences in distv_bary, not sure why...
#else
            &                   p_vt        = p_nh%diag%vt,          & !in
#endif
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

          !DA: TODO: remove the wait after everything is async
          !$ACC WAIT

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

!$OMP PARALLEL PRIVATE (rl_start,rl_end,i_startblk,i_endblk)
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


    !$ser savepoint mo_solve_nonhydro_stencil_14_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jk = 1, nlev
              DO jc = i_startidx, i_endidx
                z_rho_e(jc,jk,1) = 0._wp
                z_theta_v_e(jc,jk,1) = 0._wp
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_14_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)
          ENDIF
!$OMP BARRIER

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


    !$ser savepoint mo_solve_nonhydro_stencil_15_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jk = 1, nlev
              DO jc = i_startidx_2, i_endidx_2
                z_rho_e(jc,jk,1) = 0._wp
                z_theta_v_e(jc,jk,1) = 0._wp
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_15_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)
!$OMP BARRIER
          ENDIF

!$OMP DO PRIVATE(jb,jk,je,i_startidx,i_endidx,ilc0,ibc0,lvn_pos,&
!$OMP            z_ntdistv_bary_1,z_ntdistv_bary_2,distv_bary_1,distv_bary_2) ICON_OMP_DEFAULT_SCHEDULE
          DO jb = i_startblk, i_endblk

            CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

            IF (iadv_rhotheta == 2) THEN
              ! Operations from upwind_hflux_miura are inlined in order to process both
              ! fields in one step


    !$ser savepoint mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing p_dthalf=0.5_wp*dtime'

    !$ser data p_dthalf=0.5_wp*dtime

    PRINT *, 'Serializing p_vn=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data p_vn=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing p_vt=p_nh%diag%vt(:,:,1)'

    !$ser data p_vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing primal_normal_cell_1=p_patch%edges%primal_normal_cell_x(:,:,1)'

    !$ser data primal_normal_cell_1=p_patch%edges%primal_normal_cell_x(:,:,1)

    PRINT *, 'Serializing dual_normal_cell_1=p_patch%edges%dual_normal_cell_x(:,:,1)'

    !$ser data dual_normal_cell_1=p_patch%edges%dual_normal_cell_x(:,:,1)

    PRINT *, 'Serializing primal_normal_cell_2=p_patch%edges%primal_normal_cell_y(:,:,1)'

    !$ser data primal_normal_cell_2=p_patch%edges%primal_normal_cell_y(:,:,1)

    PRINT *, 'Serializing dual_normal_cell_2=p_patch%edges%dual_normal_cell_y(:,:,1)'

    !$ser data dual_normal_cell_2=p_patch%edges%dual_normal_cell_y(:,:,1)

    PRINT *, 'Serializing rho_ref_me=p_nh%metrics%rho_ref_me(:,:,1)'

    !$ser data rho_ref_me=p_nh%metrics%rho_ref_me(:,:,1)

    PRINT *, 'Serializing theta_ref_me=p_nh%metrics%theta_ref_me(:,:,1)'

    !$ser data theta_ref_me=p_nh%metrics%theta_ref_me(:,:,1)

    PRINT *, 'Serializing z_grad_rth_1=z_grad_rth(:,:,1,1)'

    !$ser data z_grad_rth_1=z_grad_rth(:,:,1,1)

    PRINT *, 'Serializing z_grad_rth_2=z_grad_rth(:,:,1,2)'

    !$ser data z_grad_rth_2=z_grad_rth(:,:,1,2)

    PRINT *, 'Serializing z_grad_rth_3=z_grad_rth(:,:,1,3)'

    !$ser data z_grad_rth_3=z_grad_rth(:,:,1,3)

    PRINT *, 'Serializing z_grad_rth_4=z_grad_rth(:,:,1,4)'

    !$ser data z_grad_rth_4=z_grad_rth(:,:,1,4)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)


              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
#if defined (__LOOP_EXCHANGE) || defined (__SX__) || defined (_OPENACC)
              ! For cache-based machines, also the back-trajectory computation is inlined to improve efficiency
              !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4) &
              !$ACC   PRIVATE(lvn_pos, ilc0, ibc0, z_ntdistv_bary_1, z_ntdistv_bary_2, distv_bary_1, distv_bary_2)
#ifdef __LOOP_EXCHANGE
              DO je = i_startidx, i_endidx
!DIR$ IVDEP, PREFERVECTOR
                DO jk = 1, nlev
#else
              DO jk = 1, nlev
                DO je = i_startidx, i_endidx
#endif
                  lvn_pos = p_nh%prog(nnow)%vn(je,jk,jb) >= 0._wp

                  ! line and block indices of upwind neighbor cell
                  ilc0 = MERGE(p_patch%edges%cell_idx(je,jb,1),p_patch%edges%cell_idx(je,jb,2),lvn_pos)
                  ibc0 = MERGE(p_patch%edges%cell_blk(je,jb,1),p_patch%edges%cell_blk(je,jb,2),lvn_pos)

                  ! distances from upwind mass point to the end point of the backward trajectory
                  ! in edge-normal and tangential directions
                  z_ntdistv_bary_1 =  - ( p_nh%prog(nnow)%vn(je,jk,jb) * dthalf +    &
                    MERGE(p_int%pos_on_tplane_e(je,jb,1,1), p_int%pos_on_tplane_e(je,jb,2,1),lvn_pos))

                  z_ntdistv_bary_2 =  - ( p_nh%diag%vt(je,jk,jb) * dthalf +    &
                    MERGE(p_int%pos_on_tplane_e(je,jb,1,2), p_int%pos_on_tplane_e(je,jb,2,2),lvn_pos))

                  ! rotate distance vectors into local lat-lon coordinates:
                  !
                  ! component in longitudinal direction
                  distv_bary_1 =                                                                     &
                        z_ntdistv_bary_1*MERGE(p_patch%edges%primal_normal_cell(je,jb,1)%v1,         &
                                               p_patch%edges%primal_normal_cell(je,jb,2)%v1,lvn_pos) &
                      + z_ntdistv_bary_2*MERGE(p_patch%edges%dual_normal_cell(je,jb,1)%v1,           &
                                               p_patch%edges%dual_normal_cell(je,jb,2)%v1,lvn_pos)

                  ! component in latitudinal direction
                  distv_bary_2 =                                                                     & 
                        z_ntdistv_bary_1*MERGE(p_patch%edges%primal_normal_cell(je,jb,1)%v2,         &
                                               p_patch%edges%primal_normal_cell(je,jb,2)%v2,lvn_pos) &
                      + z_ntdistv_bary_2*MERGE(p_patch%edges%dual_normal_cell(je,jb,1)%v2,           &
                                               p_patch%edges%dual_normal_cell(je,jb,2)%v2,lvn_pos)


                  ! Calculate "edge values" of rho and theta_v
                  ! Note: z_rth_pr contains the perturbation values of rho and theta_v,
                  ! and the corresponding gradients are stored in z_grad_rth.
#ifdef __SWAPDIM
                  z_rho_e(je,jk,jb) =                                                     &
                    REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,1) &
                    + distv_bary_1 * z_grad_rth(ilc0,jk,ibc0,1) &
                    + distv_bary_2 * z_grad_rth(ilc0,jk,ibc0,2)
                  z_theta_v_e(je,jk,jb) =                                                   &
                    REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,2) &
                    + distv_bary_1 * z_grad_rth(ilc0,jk,ibc0,3)                             &
                    + distv_bary_2 * z_grad_rth(ilc0,jk,ibc0,4)
#else
                  z_rho_e(je,jk,jb) = REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp) &
                    +                      z_rth_pr(1,ilc0,jk,ibc0)              &
                    + distv_bary_1 * z_grad_rth(1,ilc0,jk,ibc0)                  &
                    + distv_bary_2 * z_grad_rth(2,ilc0,jk,ibc0)

                  z_theta_v_e(je,jk,jb) = REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) &
                    +                          z_rth_pr(2,ilc0,jk,ibc0)                &
                    + distv_bary_1 * z_grad_rth(3,ilc0,jk,ibc0)                        &
                    + distv_bary_2 * z_grad_rth(4,ilc0,jk,ibc0)
#endif
                ENDDO   ! loop over vertical levels
              ENDDO ! loop over edges
#else
              !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4) PRIVATE(ilc0, ibc0)
              DO jk = 1, nlev
                DO je = i_startidx, i_endidx

                  ilc0 = p_cell_idx(je,jk,jb)
                  ibc0 = p_cell_blk(je,jk,jb)

                  ! Calculate "edge values" of rho and theta_v
                  ! Note: z_rth_pr contains the perturbation values of rho and theta_v,
                  ! and the corresponding gradients are stored in z_grad_rth.
#ifdef __SWAPDIM
                  z_rho_e(je,jk,jb) =                                                       &
                    REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,1)   &
                    + p_distv_bary(je,jk,jb,1) * z_grad_rth(ilc0,jk,ibc0,1)             &
                    + p_distv_bary(je,jk,jb,2) * z_grad_rth(ilc0,jk,ibc0,2)
                  z_theta_v_e(je,jk,jb) =                                                   &
                    REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,2) &
                    + p_distv_bary(je,jk,jb,1) * z_grad_rth(ilc0,jk,ibc0,3)             &
                    + p_distv_bary(je,jk,jb,2) * z_grad_rth(ilc0,jk,ibc0,4)
#else
                  z_rho_e(je,jk,jb) = REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp)     &
                    +                            z_rth_pr(1,ilc0,jk,ibc0)            &
                    + p_distv_bary(je,jk,jb,1) * z_grad_rth(1,ilc0,jk,ibc0)      &
                    + p_distv_bary(je,jk,jb,2) * z_grad_rth(2,ilc0,jk,ibc0)
                  z_theta_v_e(je,jk,jb) = REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) &
                    +                            z_rth_pr(2,ilc0,jk,ibc0)              &
                    + p_distv_bary(je,jk,jb,1) * z_grad_rth(3,ilc0,jk,ibc0)        &
                    + p_distv_bary(je,jk,jb,2) * z_grad_rth(4,ilc0,jk,ibc0)
#endif

                ENDDO ! loop over edges
              ENDDO   ! loop over vertical levels
#endif
              !$ACC END PARALLEL

            ELSE ! iadv_rhotheta = 1

              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
              DO je = i_startidx, i_endidx
!DIR$ IVDEP
                DO jk = 1, nlev
#else
              DO jk = 1, nlev
                DO je = i_startidx, i_endidx
#endif

                  ! Compute upwind-biased values for rho and theta starting from centered differences
                  ! Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                  ! at a second-order accurate FV discretization, but twice the length is needed for numerical
                  ! stability
                  z_rho_e(je,jk,jb) =                                                                          &
                    p_int%c_lin_e(je,1,jb)*p_nh%prog(nnow)%rho(icidx(je,jb,1),jk,icblk(je,jb,1)) +             &
                    p_int%c_lin_e(je,2,jb)*p_nh%prog(nnow)%rho(icidx(je,jb,2),jk,icblk(je,jb,2)) -             &
                    dtime * (p_nh%prog(nnow)%vn(je,jk,jb)*p_patch%edges%inv_dual_edge_length(je,jb)*           &
                   (p_nh%prog(nnow)%rho(icidx(je,jb,2),jk,icblk(je,jb,2)) -                                    &
                    p_nh%prog(nnow)%rho(icidx(je,jb,1),jk,icblk(je,jb,1)) ) + p_nh%diag%vt(je,jk,jb) *         &
                    p_patch%edges%inv_primal_edge_length(je,jb) * p_patch%edges%tangent_orientation(je,jb) *   &
                   (z_rho_v(ividx(je,jb,2),jk,ivblk(je,jb,2)) - z_rho_v(ividx(je,jb,1),jk,ivblk(je,jb,1)) ) )

                  z_theta_v_e(je,jk,jb) =                                                                          &
                    p_int%c_lin_e(je,1,jb)*p_nh%prog(nnow)%theta_v(icidx(je,jb,1),jk,icblk(je,jb,1)) +             &
                    p_int%c_lin_e(je,2,jb)*p_nh%prog(nnow)%theta_v(icidx(je,jb,2),jk,icblk(je,jb,2)) -             &
                    dtime * (p_nh%prog(nnow)%vn(je,jk,jb)*p_patch%edges%inv_dual_edge_length(je,jb)*               &
                   (p_nh%prog(nnow)%theta_v(icidx(je,jb,2),jk,icblk(je,jb,2)) -                                    &
                    p_nh%prog(nnow)%theta_v(icidx(je,jb,1),jk,icblk(je,jb,1)) ) + p_nh%diag%vt(je,jk,jb) *         &
                    p_patch%edges%inv_primal_edge_length(je,jb) * p_patch%edges%tangent_orientation(je,jb) *       &
                   (z_theta_v_v(ividx(je,jb,2),jk,ivblk(je,jb,2)) - z_theta_v_v(ividx(je,jb,1),jk,ivblk(je,jb,1)) ))

                ENDDO ! loop over edges
              ENDDO   ! loop over vertical levels
              !$ACC END PARALLEL
            ENDIF
            

    !$ser savepoint mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing p_dthalf=0.5_wp*dtime'

    !$ser data p_dthalf=0.5_wp*dtime

    PRINT *, 'Serializing p_vn=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data p_vn=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing p_vt=p_nh%diag%vt(:,:,1)'

    !$ser data p_vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing primal_normal_cell_1=p_patch%edges%primal_normal_cell_x(:,:,1)'

    !$ser data primal_normal_cell_1=p_patch%edges%primal_normal_cell_x(:,:,1)

    PRINT *, 'Serializing dual_normal_cell_1=p_patch%edges%dual_normal_cell_x(:,:,1)'

    !$ser data dual_normal_cell_1=p_patch%edges%dual_normal_cell_x(:,:,1)

    PRINT *, 'Serializing primal_normal_cell_2=p_patch%edges%primal_normal_cell_y(:,:,1)'

    !$ser data primal_normal_cell_2=p_patch%edges%primal_normal_cell_y(:,:,1)

    PRINT *, 'Serializing dual_normal_cell_2=p_patch%edges%dual_normal_cell_y(:,:,1)'

    !$ser data dual_normal_cell_2=p_patch%edges%dual_normal_cell_y(:,:,1)

    PRINT *, 'Serializing rho_ref_me=p_nh%metrics%rho_ref_me(:,:,1)'

    !$ser data rho_ref_me=p_nh%metrics%rho_ref_me(:,:,1)

    PRINT *, 'Serializing theta_ref_me=p_nh%metrics%theta_ref_me(:,:,1)'

    !$ser data theta_ref_me=p_nh%metrics%theta_ref_me(:,:,1)

    PRINT *, 'Serializing z_grad_rth_1=z_grad_rth(:,:,1,1)'

    !$ser data z_grad_rth_1=z_grad_rth(:,:,1,1)

    PRINT *, 'Serializing z_grad_rth_2=z_grad_rth(:,:,1,2)'

    !$ser data z_grad_rth_2=z_grad_rth(:,:,1,2)

    PRINT *, 'Serializing z_grad_rth_3=z_grad_rth(:,:,1,3)'

    !$ser data z_grad_rth_3=z_grad_rth(:,:,1,3)

    PRINT *, 'Serializing z_grad_rth_4=z_grad_rth(:,:,1,4)'

    !$ser data z_grad_rth_4=z_grad_rth(:,:,1,4)

    PRINT *, 'Serializing z_rth_pr_1=z_rth_pr(:,:,1,1)'

    !$ser accdata z_rth_pr_1=z_rth_pr(:,:,1,1)

    PRINT *, 'Serializing z_rth_pr_2=z_rth_pr(:,:,1,2)'

    !$ser accdata z_rth_pr_2=z_rth_pr(:,:,1,2)

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

          ENDDO
!$OMP END DO

        ENDIF

      ELSE IF (istep == 2 .AND. lhdiff_rcf .AND. divdamp_type >= 3) THEN ! apply div damping on 3D divergence

        ! add dw/dz contribution to divergence damping term

        rl_start = 7
        rl_end   = min_rledge_int-2

        i_startblk = p_patch%edges%start_block(rl_start)
        i_endblk   = p_patch%edges%end_block  (rl_end)

!$OMP DO PRIVATE(jb,jk,je,i_startidx,i_endidx) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)


    !$ser savepoint mo_solve_nonhydro_stencil_17_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing hmask_dd3d=p_nh%metrics%hmask_dd3d(:,1)'

    !$ser data hmask_dd3d=p_nh%metrics%hmask_dd3d(:,1)

    PRINT *, 'Serializing scalfac_dd3d=p_nh%metrics%scalfac_dd3d(:)'

    !$ser data scalfac_dd3d=p_nh%metrics%scalfac_dd3d(:)

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_dwdz_dd=z_dwdz_dd(:,:,1)'

    !$ser accdata z_dwdz_dd=z_dwdz_dd(:,:,1)

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
!DIR$ IVDEP, PREFERVECTOR
            DO jk = kstart_dd3d(jg), nlev
              z_graddiv_vn(jk,je,jb) = z_graddiv_vn(jk,je,jb) +  p_nh%metrics%hmask_dd3d(je,jb)*            &
                p_nh%metrics%scalfac_dd3d(jk) * p_patch%edges%inv_dual_edge_length(je,jb)*                  &
                ( z_dwdz_dd(icidx(je,jb,2),jk,icblk(je,jb,2)) - z_dwdz_dd(icidx(je,jb,1),jk,icblk(je,jb,1)) )
#else
          DO jk = kstart_dd3d(jg), nlev
            DO je = i_startidx, i_endidx
              z_graddiv_vn(je,jk,jb) = z_graddiv_vn(je,jk,jb) +  p_nh%metrics%hmask_dd3d(je,jb)*            &
                p_nh%metrics%scalfac_dd3d(jk) * p_patch%edges%inv_dual_edge_length(je,jb)*                  &
                ( z_dwdz_dd(icidx(je,jb,2),jk,icblk(je,jb,2)) - z_dwdz_dd(icidx(je,jb,1),jk,icblk(je,jb,1)) )
#endif
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_17_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing hmask_dd3d=p_nh%metrics%hmask_dd3d(:,1)'

    !$ser data hmask_dd3d=p_nh%metrics%hmask_dd3d(:,1)

    PRINT *, 'Serializing scalfac_dd3d=p_nh%metrics%scalfac_dd3d(:)'

    !$ser data scalfac_dd3d=p_nh%metrics%scalfac_dd3d(:)

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_dwdz_dd=z_dwdz_dd(:,:,1)'

    !$ser accdata z_dwdz_dd=z_dwdz_dd(:,:,1)

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

        ENDDO
!$OMP END DO

      ENDIF ! istep = 1/2

      ! Remaining computations at edge points

      rl_start = grf_bdywidth_e + 1   ! boundary update follows below
      rl_end   = min_rledge_int

      i_startblk = p_patch%edges%start_block(rl_start)
      i_endblk   = p_patch%edges%end_block(rl_end)

      IF (istep == 1) THEN

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,z_theta1,z_theta2,ikp1,ikp2) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! Store values at nest interface levels; this is done here for the first sub-time step,
          ! the final averaging is done in mo_nh_nest_utilities:compute_tendencies
          IF (idyn_timestep == 1 .AND. l_child_vertnest) THEN
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR
            DO je = i_startidx, i_endidx
              p_nh%diag%vn_ie_int(je,1,jb) = p_nh%diag%vn_ie(je,nshift,jb)
            ENDDO
            !$ACC END PARALLEL
          ENDIF


    !$ser savepoint mo_solve_nonhydro_stencil_18_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
            DO jk = 1, nflatlev(jg)-1
#else
          DO jk = 1, nflatlev(jg)-1
            DO je = i_startidx, i_endidx
#endif
              ! horizontal gradient of Exner pressure where coordinate surfaces are flat
              z_gradh_exner(je,jk,jb) = p_patch%edges%inv_dual_edge_length(je,jb)* &
               (z_exner_ex_pr(icidx(je,jb,2),jk,icblk(je,jb,2)) -                  &
                z_exner_ex_pr(icidx(je,jb,1),jk,icblk(je,jb,1)) )
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_18_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

          IF (igradp_method <= 3) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_19_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)'

    !$ser data ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)

    PRINT *, 'Serializing c_lin_e=p_int%c_lin_e(:,:,1)'

    !$ser data c_lin_e=p_int%c_lin_e(:,:,1)

    PRINT *, 'Serializing z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)'

    !$ser accdata z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR COLLAPSE(2)
#ifdef __LOOP_EXCHANGE
            DO je = i_startidx, i_endidx
!DIR$ IVDEP
              DO jk = nflatlev(jg), nflat_gradp(jg)
#else
!$NEC outerloop_unroll(8)
            DO jk = nflatlev(jg), nflat_gradp(jg)
              DO je = i_startidx, i_endidx
#endif
                ! horizontal gradient of Exner pressure, including metric correction
                z_gradh_exner(je,jk,jb) = p_patch%edges%inv_dual_edge_length(je,jb)*         &
                 (z_exner_ex_pr(icidx(je,jb,2),jk,icblk(je,jb,2)) -                          &
                  z_exner_ex_pr(icidx(je,jb,1),jk,icblk(je,jb,1)) ) -                        &
                  p_nh%metrics%ddxn_z_full(je,jk,jb) *                                       &
#ifdef __SWAPDIM
                 (p_int%c_lin_e(je,1,jb)*z_dexner_dz_c(icidx(je,jb,1),jk,icblk(je,jb,1),1) + &
                  p_int%c_lin_e(je,2,jb)*z_dexner_dz_c(icidx(je,jb,2),jk,icblk(je,jb,2),1))
#else
                 (p_int%c_lin_e(je,1,jb)*z_dexner_dz_c(1,icidx(je,jb,1),jk,icblk(je,jb,1)) + &
                  p_int%c_lin_e(je,2,jb)*z_dexner_dz_c(1,icidx(je,jb,2),jk,icblk(je,jb,2)))
#endif
              ENDDO
            ENDDO
            !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_19_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)'

    !$ser data ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)

    PRINT *, 'Serializing c_lin_e=p_int%c_lin_e(:,:,1)'

    !$ser data c_lin_e=p_int%c_lin_e(:,:,1)

    PRINT *, 'Serializing z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)'

    !$ser accdata z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)


    !$ser savepoint mo_solve_nonhydro_stencil_20_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)'

    !$ser data zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)

    PRINT *, 'Serializing z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)'

    !$ser accdata z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)

    PRINT *, 'Serializing z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)'

    !$ser accdata z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
            DO je = i_startidx, i_endidx
!DIR$ IVDEP, PREFERVECTOR
              DO jk = nflat_gradp(jg)+1, nlev
#else
!$NEC outerloop_unroll(8)
            DO jk = nflat_gradp(jg)+1, nlev
              DO je = i_startidx, i_endidx
#endif
                ! horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
                z_gradh_exner(je,jk,jb) = p_patch%edges%inv_dual_edge_length(je,jb)*          &
                  (z_exner_ex_pr(icidx(je,jb,2),ikidx(2,je,jk,jb),icblk(je,jb,2)) +           &
                   p_nh%metrics%zdiff_gradp(2,je,jk,jb)*                                      &
#ifdef __SWAPDIM
                  (z_dexner_dz_c(icidx(je,jb,2),ikidx(2,je,jk,jb),icblk(je,jb,2),1) +         &
                   p_nh%metrics%zdiff_gradp(2,je,jk,jb)*                                      &
                   z_dexner_dz_c(icidx(je,jb,2),ikidx(2,je,jk,jb),icblk(je,jb,2),2)) -        &
                  (z_exner_ex_pr(icidx(je,jb,1),ikidx(1,je,jk,jb),icblk(je,jb,1)) +           &
                   p_nh%metrics%zdiff_gradp(1,je,jk,jb)*                                      &
                  (z_dexner_dz_c(icidx(je,jb,1),ikidx(1,je,jk,jb),icblk(je,jb,1),1) +         &
                   p_nh%metrics%zdiff_gradp(1,je,jk,jb)* &
                   z_dexner_dz_c(icidx(je,jb,1),ikidx(1,je,jk,jb),icblk(je,jb,1),2))))
#else
                  (z_dexner_dz_c(1,icidx(je,jb,2),ikidx(2,je,jk,jb),icblk(je,jb,2)) +         &
                   p_nh%metrics%zdiff_gradp(2,je,jk,jb)*                                      &
                   z_dexner_dz_c(2,icidx(je,jb,2),ikidx(2,je,jk,jb),icblk(je,jb,2))) -        &
                  (z_exner_ex_pr(icidx(je,jb,1),ikidx(1,je,jk,jb),icblk(je,jb,1)) +           &
                   p_nh%metrics%zdiff_gradp(1,je,jk,jb)*                                      &
                  (z_dexner_dz_c(1,icidx(je,jb,1),ikidx(1,je,jk,jb),icblk(je,jb,1)) +         &
                   p_nh%metrics%zdiff_gradp(1,je,jk,jb)*                                      &
                   z_dexner_dz_c(2,icidx(je,jb,1),ikidx(1,je,jk,jb),icblk(je,jb,1)))))
#endif
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_20_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_exner_ex_pr=z_exner_ex_pr(:,:,1)'

    !$ser accdata z_exner_ex_pr=z_exner_ex_pr(:,:,1)

    PRINT *, 'Serializing zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)'

    !$ser data zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)

    PRINT *, 'Serializing z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)'

    !$ser accdata z_dexner_dz_c_1=z_dexner_dz_c(:,:,1,1)

    PRINT *, 'Serializing z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)'

    !$ser accdata z_dexner_dz_c_2=z_dexner_dz_c(:,:,1,2)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

          ELSE IF (igradp_method == 4 .OR. igradp_method == 5) THEN

            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
            DO je = i_startidx, i_endidx
              DO jk = nflatlev(jg), nlev
#else
            DO jk = nflatlev(jg), nlev
              DO je = i_startidx, i_endidx
#endif
                ! horizontal gradient of Exner pressure, cubic/quadratic interpolation
                z_gradh_exner(je,jk,jb) =  p_patch%edges%inv_dual_edge_length(je,jb)*   &
                  (z_exner_ex_pr(icidx(je,jb,2),ikidx(2,je,jk,jb)-1,icblk(je,jb,2)) *   &
                   p_nh%metrics%coeff_gradp(5,je,jk,jb) +                               &
                   z_exner_ex_pr(icidx(je,jb,2),ikidx(2,je,jk,jb)  ,icblk(je,jb,2)) *   &
                   p_nh%metrics%coeff_gradp(6,je,jk,jb) +                               &
                   z_exner_ex_pr(icidx(je,jb,2),ikidx(2,je,jk,jb)+1,icblk(je,jb,2)) *   &
                   p_nh%metrics%coeff_gradp(7,je,jk,jb) +                               &
                   z_exner_ex_pr(icidx(je,jb,2),ikidx(2,je,jk,jb)+2,icblk(je,jb,2)) *   &
                   p_nh%metrics%coeff_gradp(8,je,jk,jb) -                               &
                  (z_exner_ex_pr(icidx(je,jb,1),ikidx(1,je,jk,jb)-1,icblk(je,jb,1)) *   &
                   p_nh%metrics%coeff_gradp(1,je,jk,jb) +                               &
                   z_exner_ex_pr(icidx(je,jb,1),ikidx(1,je,jk,jb)  ,icblk(je,jb,1)) *   &
                   p_nh%metrics%coeff_gradp(2,je,jk,jb) +                               &
                   z_exner_ex_pr(icidx(je,jb,1),ikidx(1,je,jk,jb)+1,icblk(je,jb,1)) *   &
                   p_nh%metrics%coeff_gradp(3,je,jk,jb) +                               &
                   z_exner_ex_pr(icidx(je,jb,1),ikidx(1,je,jk,jb)+2,icblk(je,jb,1)) *   &
                  p_nh%metrics%coeff_gradp(4,je,jk,jb)) )

              ENDDO
            ENDDO
            !$ACC END PARALLEL
          ENDIF

          ! compute hydrostatically approximated correction term that replaces downward extrapolation
          IF (igradp_method == 3) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_21_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing grav_o_cpd=grav_o_cpd'

    !$ser data grav_o_cpd=grav_o_cpd

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)'

    !$ser data zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_hydro_corr=z_hydro_corr(:,:,1)'

    !$ser accdata z_hydro_corr=z_hydro_corr(:,:,1)

            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR PRIVATE(z_theta1, z_theta2)
            DO je = i_startidx, i_endidx

              z_theta1 = &
                 p_nh%prog(nnow)%theta_v(icidx(je,jb,1),ikidx(1,je,nlev,jb),icblk(je,jb,1)) +  &
                 p_nh%metrics%zdiff_gradp(1,je,nlev,jb)*                                       &
                (p_nh%diag%theta_v_ic(icidx(je,jb,1),ikidx(1,je,nlev,jb),  icblk(je,jb,1)) -   &
                 p_nh%diag%theta_v_ic(icidx(je,jb,1),ikidx(1,je,nlev,jb)+1,icblk(je,jb,1))) *  &
                 p_nh%metrics%inv_ddqz_z_full(icidx(je,jb,1),ikidx(1,je,nlev,jb),icblk(je,jb,1))

              z_theta2 = &
                 p_nh%prog(nnow)%theta_v(icidx(je,jb,2),ikidx(2,je,nlev,jb),icblk(je,jb,2)) +  &
                 p_nh%metrics%zdiff_gradp(2,je,nlev,jb)*                                       &
                (p_nh%diag%theta_v_ic(icidx(je,jb,2),ikidx(2,je,nlev,jb),  icblk(je,jb,2)) -   &
                 p_nh%diag%theta_v_ic(icidx(je,jb,2),ikidx(2,je,nlev,jb)+1,icblk(je,jb,2))) *  &
                 p_nh%metrics%inv_ddqz_z_full(icidx(je,jb,2),ikidx(2,je,nlev,jb),icblk(je,jb,2))

              z_hydro_corr(je,nlev,jb) = grav_o_cpd*p_patch%edges%inv_dual_edge_length(je,jb)*    &
                (z_theta2-z_theta1)*4._wp/(z_theta1+z_theta2)**2

            ENDDO
            !$ACC END PARALLEL
    

    !$ser savepoint mo_solve_nonhydro_stencil_21_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing grav_o_cpd=grav_o_cpd'

    !$ser data grav_o_cpd=grav_o_cpd

    PRINT *, 'Serializing theta_v=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)'

    !$ser data zdiff_gradp=p_nh%metrics%zdiff_gradp_dsl(:,:,:,1)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)'

    !$ser data inv_dual_edge_length=p_patch%edges%inv_dual_edge_length(:,1)

    PRINT *, 'Serializing z_hydro_corr=z_hydro_corr(:,:,1)'

    !$ser accdata z_hydro_corr=z_hydro_corr(:,:,1)

          ELSE IF (igradp_method == 5) THEN

            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR PRIVATE(ikp1, ikp2, z_theta1, z_theta2)
            DO je = i_startidx, i_endidx

              ikp1 = MIN(nlev,ikidx(1,je,nlev,jb)+2)
              ikp2 = MIN(nlev,ikidx(2,je,nlev,jb)+2)

              z_theta1 =                                                                       &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,1),ikidx(1,je,nlev,jb)-1,icblk(je,jb,1)) * &
                p_nh%metrics%coeff_gradp(1,je,nlev,jb) +                                         &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,1),ikidx(1,je,nlev,jb)  ,icblk(je,jb,1)) * &
                p_nh%metrics%coeff_gradp(2,je,nlev,jb) +                                         &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,1),ikidx(1,je,nlev,jb)+1,icblk(je,jb,1)) * &
                p_nh%metrics%coeff_gradp(3,je,nlev,jb) +                                         &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,1),ikp1                 ,icblk(je,jb,1)) * &
                p_nh%metrics%coeff_gradp(4,je,nlev,jb)

              z_theta2 =                                                                       &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,2),ikidx(2,je,nlev,jb)-1,icblk(je,jb,2)) * &
                p_nh%metrics%coeff_gradp(5,je,nlev,jb) +                                         &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,2),ikidx(2,je,nlev,jb)  ,icblk(je,jb,2)) * &
                p_nh%metrics%coeff_gradp(6,je,nlev,jb) +                                         &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,2),ikidx(2,je,nlev,jb)+1,icblk(je,jb,2)) * &
                p_nh%metrics%coeff_gradp(7,je,nlev,jb) +                                         &
                p_nh%prog(nnow)%theta_v(icidx(je,jb,2),ikp2                 ,icblk(je,jb,2)) * &
                p_nh%metrics%coeff_gradp(8,je,nlev,jb)

              z_hydro_corr(je,nlev,jb) = grav_o_cpd*p_patch%edges%inv_dual_edge_length(je,jb)*    &
                (z_theta2-z_theta1)*4._wp/(z_theta1+z_theta2)**2

            ENDDO
            !$ACC END PARALLEL
          ENDIF

        ENDDO
!$OMP END DO

      ENDIF ! istep = 1


      IF (istep == 1 .AND. (igradp_method == 3 .OR. igradp_method == 5)) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_22_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing ipeidx_dsl=p_nh%metrics%pg_edgeidx_dsl(:,:,1)'

    !$ser data ipeidx_dsl=p_nh%metrics%pg_edgeidx_dsl(:,:,1)

    PRINT *, 'Serializing pg_exdist=p_nh%metrics%pg_exdist_dsl(:,:,1)'

    !$ser data pg_exdist=p_nh%metrics%pg_exdist_dsl(:,:,1)

    PRINT *, 'Serializing z_hydro_corr=z_hydro_corr(:,:,1)'

    !$ser accdata z_hydro_corr=z_hydro_corr(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)
!$OMP DO PRIVATE(jb,je,ie,nlen_gradp,ishift) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = 1, nblks_gradp
          IF (jb == nblks_gradp) THEN
            nlen_gradp = npromz_gradp
          ELSE
            nlen_gradp = nproma_gradp
          ENDIF
          ishift = (jb-1)*nproma_gradp
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
!$NEC ivdep
          !$ACC LOOP GANG VECTOR
          DO je = 1, nlen_gradp
            ie = ishift+je

            z_gradh_exner(ipeidx(ie),iplev(ie),ipeblk(ie))  =              &
              z_gradh_exner(ipeidx(ie),iplev(ie),ipeblk(ie)) +             &
              p_nh%metrics%pg_exdist(ie)*z_hydro_corr(ipeidx(ie),nlev,ipeblk(ie))

          ENDDO
          !$ACC END PARALLEL
        ENDDO
!$OMP END DO

    rl_start_2 = grf_bdywidth_e+1
    rl_end_2   = min_rledge

    i_startblk_2 = p_patch%edges%start_block(rl_start_2)
    i_endblk_2   = p_patch%edges%end_block(rl_end_2)

    CALL get_indices_e(p_patch, 1, i_startblk_2, i_endblk_2, &
                       i_startidx_2, i_endidx_2, rl_start_2, rl_end_2)


    !$ser savepoint mo_solve_nonhydro_stencil_22_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing ipeidx_dsl=p_nh%metrics%pg_edgeidx_dsl(:,:,1)'

    !$ser data ipeidx_dsl=p_nh%metrics%pg_edgeidx_dsl(:,:,1)

    PRINT *, 'Serializing pg_exdist=p_nh%metrics%pg_exdist_dsl(:,:,1)'

    !$ser data pg_exdist=p_nh%metrics%pg_exdist_dsl(:,:,1)

    PRINT *, 'Serializing z_hydro_corr=z_hydro_corr(:,:,1)'

    !$ser accdata z_hydro_corr=z_hydro_corr(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

      ENDIF


      ! Update horizontal velocity field: advection, Coriolis force, pressure-gradient term, and physics

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,z_graddiv2_vn,                                                   &
!$OMP            z_ddt_vn_dyn, z_ddt_vn_apc, z_ddt_vn_cor, z_ddt_vn_pgr, z_ddt_vn_ray, z_d_vn_dmp, z_d_vn_iau  &
!$OMP           ) ICON_OMP_DEFAULT_SCHEDULE

      DO jb = i_startblk, i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
          i_startidx, i_endidx, rl_start, rl_end)

        IF ((itime_scheme >= 4) .AND. istep == 2) THEN ! use temporally averaged velocity advection terms


    !$ser savepoint mo_solve_nonhydro_stencil_23_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing wgt_nnew_vel=wgt_nnew_vel'

    !$ser data wgt_nnew_vel=wgt_nnew_vel

    PRINT *, 'Serializing wgt_nnow_vel=wgt_nnow_vel'

    !$ser data wgt_nnow_vel=wgt_nnow_vel

    PRINT *, 'Serializing vn_nnow=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data vn_nnow=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)'

    !$ser data ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)

    PRINT *, 'Serializing ddt_vn_adv_ntl2=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl2)'

    !$ser data ddt_vn_adv_ntl2=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl2)

    PRINT *, 'Serializing ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)'

    !$ser data ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

    PRINT *, 'Serializing vn_nnew=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn_nnew=p_nh%prog(nnew)%vn(:,:,1)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)

          !$ACC LOOP GANG(STATIC: 1) VECTOR PRIVATE(z_ddt_vn_dyn, z_ddt_vn_apc, z_ddt_vn_cor, z_ddt_vn_pgr) TILE(32, 4)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_ddt_vn_apc                      =  p_nh%diag%ddt_vn_apc_pc(je,jk,jb,ntl1)*wgt_nnow_vel  &
                &                                 +p_nh%diag%ddt_vn_apc_pc(je,jk,jb,ntl2)*wgt_nnew_vel
              z_ddt_vn_pgr                      = -cpd*z_theta_v_e(je,jk,jb)*z_gradh_exner(je,jk,jb)
              !
              z_ddt_vn_dyn                      =  z_ddt_vn_apc                   & ! advection plus Coriolis
                &                                 +z_ddt_vn_pgr                   & ! pressure gradient
                &                                 +p_nh%diag%ddt_vn_phy(je,jk,jb)   ! physics applied in dynamics
              !
              p_nh%prog(nnew)%vn(je,jk,jb)      =  p_nh%prog(nnow)%vn(je,jk,jb)   + dtime       * z_ddt_vn_dyn
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF (p_nh%diag%ddt_vn_adv_is_associated .OR. p_nh%diag%ddt_vn_cor_is_associated) THEN
                z_ddt_vn_cor                    =  p_nh%diag%ddt_vn_cor_pc(je,jk,jb,ntl1)*wgt_nnow_vel  &
                  &                               +p_nh%diag%ddt_vn_cor_pc(je,jk,jb,ntl2)*wgt_nnew_vel
                !
                IF (p_nh%diag%ddt_vn_adv_is_associated) THEN
                  p_nh%diag%ddt_vn_adv(je,jk,jb)=  p_nh%diag%ddt_vn_adv(je,jk,jb) + r_nsubsteps *(z_ddt_vn_apc-z_ddt_vn_cor)
                END IF
                !
                IF (p_nh%diag%ddt_vn_cor_is_associated) THEN
                  p_nh%diag%ddt_vn_cor(je,jk,jb)=  p_nh%diag%ddt_vn_cor(je,jk,jb) + r_nsubsteps * z_ddt_vn_cor
                END IF
                !
              END IF
              !
              IF (p_nh%diag%ddt_vn_pgr_is_associated) THEN
                p_nh%diag%ddt_vn_pgr(je,jk,jb)  =  p_nh%diag%ddt_vn_pgr(je,jk,jb) + r_nsubsteps * z_ddt_vn_pgr
              END IF
              !
              IF (p_nh%diag%ddt_vn_phd_is_associated) THEN
                p_nh%diag%ddt_vn_phd(je,jk,jb)  =  p_nh%diag%ddt_vn_phd(je,jk,jb) + r_nsubsteps * p_nh%diag%ddt_vn_phy(je,jk,jb)
              END IF
              !
              IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + r_nsubsteps * z_ddt_vn_dyn
              END IF
#endif
              !
            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_23_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing wgt_nnew_vel=wgt_nnew_vel'

    !$ser data wgt_nnew_vel=wgt_nnew_vel

    PRINT *, 'Serializing wgt_nnow_vel=wgt_nnow_vel'

    !$ser data wgt_nnow_vel=wgt_nnow_vel

    PRINT *, 'Serializing vn_nnow=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data vn_nnow=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)'

    !$ser data ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)

    PRINT *, 'Serializing ddt_vn_adv_ntl2=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl2)'

    !$ser data ddt_vn_adv_ntl2=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl2)

    PRINT *, 'Serializing ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)'

    !$ser data ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

    PRINT *, 'Serializing vn_nnew=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn_nnew=p_nh%prog(nnew)%vn(:,:,1)

        ELSE


    !$ser savepoint mo_solve_nonhydro_stencil_24_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing vn_nnow=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data vn_nnow=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)'

    !$ser data ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)

    PRINT *, 'Serializing ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)'

    !$ser data ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

    PRINT *, 'Serializing vn_nnew=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn_nnew=p_nh%prog(nnew)%vn(:,:,1)
!$ACC PARALLEL IF(i_am_accel_node)  DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              p_nh%prog(nnew)%vn(je,jk,jb)      =  p_nh%prog(nnow)%vn(je,jk,jb)   + dtime *                 &
                &                                ( p_nh%diag%ddt_vn_apc_pc(je,jk,jb,ntl1)                   &
                &                                 -cpd*z_theta_v_e(je,jk,jb)*z_gradh_exner(je,jk,jb)        &
                &                                 +p_nh%diag%ddt_vn_phy(je,jk,jb)                        )
              !
            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_24_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing vn_nnow=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data vn_nnow=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)'

    !$ser data ddt_vn_adv_ntl1=p_nh%diag%ddt_vn_apc_pc(:,:,1,ntl1)

    PRINT *, 'Serializing ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)'

    !$ser data ddt_vn_phy=p_nh%diag%ddt_vn_phy(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

    PRINT *, 'Serializing z_gradh_exner=z_gradh_exner(:,:,1)'

    !$ser accdata z_gradh_exner=z_gradh_exner(:,:,1)

    PRINT *, 'Serializing vn_nnew=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn_nnew=p_nh%prog(nnew)%vn(:,:,1)
        ENDIF

        IF (lhdiff_rcf .AND. istep == 2 .AND. (divdamp_order == 4 .OR. divdamp_order == 24)) THEN ! fourth-order divergence damping
        ! Compute gradient of divergence of gradient of divergence for fourth-order divergence damping


    !$ser savepoint mo_solve_nonhydro_stencil_25_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing geofac_grdiv=p_int%geofac_grdiv(:,:,1)'

    !$ser data geofac_grdiv=p_int%geofac_grdiv(:,:,1)

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

    PRINT *, 'Serializing z_graddiv2_vn=z_graddiv2_vn(:,:)'

    !$ser accdata z_graddiv2_vn=z_graddiv2_vn(:,:)
!$ACC PARALLEL IF( i_am_accel_node )  DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
!DIR$ IVDEP
            DO jk = 1, nlev
              z_graddiv2_vn(je,jk) = p_int%geofac_grdiv(je,1,jb)*z_graddiv_vn(jk,je,jb)      &
                + p_int%geofac_grdiv(je,2,jb)*z_graddiv_vn(jk,iqidx(je,jb,1),iqblk(je,jb,1)) &
                + p_int%geofac_grdiv(je,3,jb)*z_graddiv_vn(jk,iqidx(je,jb,2),iqblk(je,jb,2)) &
                + p_int%geofac_grdiv(je,4,jb)*z_graddiv_vn(jk,iqidx(je,jb,3),iqblk(je,jb,3)) &
                + p_int%geofac_grdiv(je,5,jb)*z_graddiv_vn(jk,iqidx(je,jb,4),iqblk(je,jb,4))
#else
!$NEC outerloop_unroll(6)
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              z_graddiv2_vn(je,jk) = p_int%geofac_grdiv(je,1,jb)*z_graddiv_vn(je,jk,jb)      &
                + p_int%geofac_grdiv(je,2,jb)*z_graddiv_vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%geofac_grdiv(je,3,jb)*z_graddiv_vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%geofac_grdiv(je,4,jb)*z_graddiv_vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%geofac_grdiv(je,5,jb)*z_graddiv_vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))
#endif

            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_25_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing geofac_grdiv=p_int%geofac_grdiv(:,:,1)'

    !$ser data geofac_grdiv=p_int%geofac_grdiv(:,:,1)

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

    PRINT *, 'Serializing z_graddiv2_vn=z_graddiv2_vn(:,:)'

    !$ser accdata z_graddiv2_vn=z_graddiv2_vn(:,:)

        ENDIF

        IF (lhdiff_rcf .AND. istep == 2) THEN
          ! apply divergence damping if diffusion is not called every sound-wave time step
          IF (divdamp_order == 2 .OR. (divdamp_order == 24 .AND. scal_divdamp_o2 > 1.e-6_wp) ) THEN ! 2nd-order divergence damping


    !$ser savepoint mo_solve_nonhydro_stencil_26_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing scal_divdamp_o2=scal_divdamp_o2'

    !$ser data scal_divdamp_o2=scal_divdamp_o2

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)
!$ACC PARALLEL IF(i_am_accel_node)  DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4) PRIVATE(z_d_vn_dmp)
            DO jk = 1, nlev
!DIR$ IVDEP
              DO je = i_startidx, i_endidx
                !
#ifdef __LOOP_EXCHANGE
                z_d_vn_dmp = scal_divdamp_o2*z_graddiv_vn(jk,je,jb)
#else
                z_d_vn_dmp = scal_divdamp_o2*z_graddiv_vn(je,jk,jb)
#endif
                !
                p_nh%prog(nnew)%vn(je,jk,jb)      =  p_nh%prog(nnew)%vn(je,jk,jb)   + z_d_vn_dmp
                !
#ifdef __ENABLE_DDT_VN_XYZ__
                IF (p_nh%diag%ddt_vn_dmp_is_associated) THEN
                  p_nh%diag%ddt_vn_dmp(je,jk,jb)  =  p_nh%diag%ddt_vn_dmp(je,jk,jb) + z_d_vn_dmp * r_dtimensubsteps
                END IF
                !
                IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                  p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + z_d_vn_dmp * r_dtimensubsteps
                END IF
#endif
                !
              ENDDO
            ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_26_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing scal_divdamp_o2=scal_divdamp_o2'

    !$ser data scal_divdamp_o2=scal_divdamp_o2

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

          ENDIF
          IF (divdamp_order == 4 .OR. (divdamp_order == 24 .AND. divdamp_fac_o2 <= 4._wp*divdamp_fac) ) THEN
            IF (l_limited_area .OR. jg > 1) THEN
              ! fourth-order divergence damping with reduced damping coefficient along nest boundary
              ! (scal_divdamp is negative whereas bdy_divdamp is positive; decreasing the divergence
              ! damping along nest boundaries is beneficial because this reduces the interference
              ! with the increased diffusion applied in nh_diffusion)


    !$ser savepoint mo_solve_nonhydro_stencil_27_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing scal_divdamp=scal_divdamp(:)'

    !$ser data scal_divdamp=scal_divdamp(:)

    PRINT *, 'Serializing bdy_divdamp=bdy_divdamp(:)'

    !$ser data bdy_divdamp=bdy_divdamp(:)

    PRINT *, 'Serializing nudgecoeff_e=p_int%nudgecoeff_e(:,1)'

    !$ser data nudgecoeff_e=p_int%nudgecoeff_e(:,1)

    PRINT *, 'Serializing z_graddiv2_vn=z_graddiv2_vn(:,:)'

    !$ser accdata z_graddiv2_vn=z_graddiv2_vn(:,:)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)
!$ACC PARALLEL IF(i_am_accel_node)  DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4) PRIVATE(z_d_vn_dmp)
              DO jk = 1, nlev
!DIR$ IVDEP
!$NEC ivdep
                DO je = i_startidx, i_endidx
                  !
                  z_d_vn_dmp = (scal_divdamp(jk)+bdy_divdamp(jk)*p_int%nudgecoeff_e(je,jb))*z_graddiv2_vn(je,jk)
                  !
                  p_nh%prog(nnew)%vn(je,jk,jb)      =  p_nh%prog(nnew)%vn(je,jk,jb)   + z_d_vn_dmp
                  !
#ifdef __ENABLE_DDT_VN_XYZ__
                  IF (p_nh%diag%ddt_vn_dmp_is_associated) THEN
                    p_nh%diag%ddt_vn_dmp(je,jk,jb)  =  p_nh%diag%ddt_vn_dmp(je,jk,jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
                  !
                  IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                    p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
#endif
                  !
                ENDDO
              ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_27_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing scal_divdamp=scal_divdamp(:)'

    !$ser data scal_divdamp=scal_divdamp(:)

    PRINT *, 'Serializing bdy_divdamp=bdy_divdamp(:)'

    !$ser data bdy_divdamp=bdy_divdamp(:)

    PRINT *, 'Serializing nudgecoeff_e=p_int%nudgecoeff_e(:,1)'

    !$ser data nudgecoeff_e=p_int%nudgecoeff_e(:,1)

    PRINT *, 'Serializing z_graddiv2_vn=z_graddiv2_vn(:,:)'

    !$ser accdata z_graddiv2_vn=z_graddiv2_vn(:,:)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

            ELSE ! fourth-order divergence damping


    !$ser savepoint mo_solve_nonhydro_4th_order_divdamp_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing scal_divdamp=scal_divdamp(:)'

    !$ser data scal_divdamp=scal_divdamp(:)

    PRINT *, 'Serializing z_graddiv2_vn=z_graddiv2_vn(:,:)'

    !$ser accdata z_graddiv2_vn=z_graddiv2_vn(:,:)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

!$ACC PARALLEL IF(i_am_accel_node)  DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4) PRIVATE(z_d_vn_dmp)
              DO jk = 1, nlev
!DIR$ IVDEP
                DO je = i_startidx, i_endidx
                  !
                  z_d_vn_dmp = scal_divdamp(jk)*z_graddiv2_vn(je,jk)
                  !
                  p_nh%prog(nnew)%vn(je,jk,jb)      =  p_nh%prog(nnew)%vn(je,jk,jb)   + z_d_vn_dmp
                  !
#ifdef __ENABLE_DDT_VN_XYZ__
                  IF (p_nh%diag%ddt_vn_dmp_is_associated) THEN
                    p_nh%diag%ddt_vn_dmp(je,jk,jb)  =  p_nh%diag%ddt_vn_dmp(je,jk,jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
                  !
                  IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                    p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
#endif
                  !
                ENDDO
              ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_4th_order_divdamp_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing scal_divdamp=scal_divdamp(:)'

    !$ser data scal_divdamp=scal_divdamp(:)

    PRINT *, 'Serializing z_graddiv2_vn=z_graddiv2_vn(:,:)'

    !$ser accdata z_graddiv2_vn=z_graddiv2_vn(:,:)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)
            ENDIF
          ENDIF
        ENDIF

        IF (is_iau_active) THEN ! add analysis increment from data assimilation


    !$ser savepoint mo_solve_nonhydro_stencil_28_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing iau_wgt_dyn=iau_wgt_dyn'

    !$ser data iau_wgt_dyn=iau_wgt_dyn

    PRINT *, 'Serializing vn_incr=p_nh%diag%vn_incr(:,:,1)'

    !$ser data vn_incr=p_nh%diag%vn_incr(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)
  !$ACC PARALLEL IF(i_am_accel_node)  DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4) PRIVATE(z_d_vn_iau)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_d_vn_iau = iau_wgt_dyn*p_nh%diag%vn_incr(je,jk,jb)
              !
              p_nh%prog(nnew)%vn(je,jk,jb)        =  p_nh%prog(nnew)%vn(je,jk,jb)   + z_d_vn_iau
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF (istep == 2) THEN
                IF (p_nh%diag%ddt_vn_iau_is_associated) THEN
                  p_nh%diag%ddt_vn_iau(je,jk,jb)  =  p_nh%diag%ddt_vn_iau(je,jk,jb) + z_d_vn_iau * r_dtimensubsteps
                END IF
                !
                IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                  p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + z_d_vn_iau * r_dtimensubsteps
                END IF
              END IF
#endif
              !
            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_28_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing iau_wgt_dyn=iau_wgt_dyn'

    !$ser data iau_wgt_dyn=iau_wgt_dyn

    PRINT *, 'Serializing vn_incr=p_nh%diag%vn_incr(:,:,1)'

    !$ser data vn_incr=p_nh%diag%vn_incr(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

        ENDIF

        ! Classic Rayleigh damping mechanism for vn (requires reference state !!)
        !
        IF ( rayleigh_type == RAYLEIGH_CLASSIC ) THEN

          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2) PRIVATE(z_ddt_vn_ray)
          DO jk = 1, nrdmax(jg)
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_ddt_vn_ray = -p_nh%metrics%rayleigh_vn(jk) * (p_nh%prog(nnew)%vn(je,jk,jb) - p_nh%ref%vn_ref(je,jk,jb))
              !
              p_nh%prog(nnew)%vn(je,jk,jb)        =  p_nh%prog(nnew)%vn(je,jk,jb)   + z_ddt_vn_ray * dtime
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF (istep == 2) THEN
                IF (p_nh%diag%ddt_vn_ray_is_associated) THEN
                  p_nh%diag%ddt_vn_ray(je,jk,jb)  =  p_nh%diag%ddt_vn_ray(je,jk,jb) + z_ddt_vn_ray * r_nsubsteps
                END IF
                !
                IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                  p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + z_ddt_vn_ray * r_nsubsteps
                END IF
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL
        ENDIF
      ENDDO
!$OMP END DO

      ! Boundary update of horizontal velocity
      IF (istep == 1 .AND. (l_limited_area .OR. jg > 1)) THEN
        rl_start = 1
        rl_end   = grf_bdywidth_e

        i_startblk = p_patch%edges%start_block(rl_start)
        i_endblk   = p_patch%edges%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
            i_startidx, i_endidx, rl_start, rl_end)


    !$ser savepoint mo_solve_nonhydro_stencil_29_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing grf_tend_vn=p_nh%diag%grf_tend_vn(:,:,1)'

    !$ser data grf_tend_vn=p_nh%diag%grf_tend_vn(:,:,1)

    PRINT *, 'Serializing vn_now=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data vn_now=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing vn_new=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn_new=p_nh%prog(nnew)%vn(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              p_nh%prog(nnew)%vn(je,jk,jb)      =  p_nh%prog(nnow)%vn(je,jk,jb)   + p_nh%diag%grf_tend_vn(je,jk,jb) * dtime
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF (p_nh%diag%ddt_vn_grf_is_associated) THEN
                p_nh%diag%ddt_vn_grf(je,jk,jb)  =  p_nh%diag%ddt_vn_grf(je,jk,jb) + p_nh%diag%grf_tend_vn(je,jk,jb) * r_nsubsteps
              END IF
              !
              IF (p_nh%diag%ddt_vn_dyn_is_associated) THEN
                p_nh%diag%ddt_vn_dyn(je,jk,jb)  =  p_nh%diag%ddt_vn_dyn(je,jk,jb) + p_nh%diag%grf_tend_vn(je,jk,jb) * r_nsubsteps
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_29_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing grf_tend_vn=p_nh%diag%grf_tend_vn(:,:,1)'

    !$ser data grf_tend_vn=p_nh%diag%grf_tend_vn(:,:,1)

    PRINT *, 'Serializing vn_now=p_nh%prog(nnow)%vn(:,:,1)'

    !$ser data vn_now=p_nh%prog(nnow)%vn(:,:,1)

    PRINT *, 'Serializing vn_new=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn_new=p_nh%prog(nnew)%vn(:,:,1)

        ENDDO
!$OMP END DO

      ENDIF

      ! Preparations for nest boundary interpolation of mass fluxes from parent domain
      IF (jg > 1 .AND. grf_intmethod_e >= 5 .AND. idiv_method == 1 .AND. jstep == 0 .AND. istep == 1) THEN

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG

!$OMP DO PRIVATE(ic,je,jb,jk) ICON_OMP_DEFAULT_SCHEDULE
        DO ic = 1, p_nh%metrics%bdy_mflx_e_dim
          je = p_nh%metrics%bdy_mflx_e_idx(ic)
          jb = p_nh%metrics%bdy_mflx_e_blk(ic)
!DIR$ IVDEP
          !$ACC LOOP VECTOR
          DO jk = 1, nlev
            p_nh%diag%grf_bdy_mflx(jk,ic,2) = p_nh%diag%grf_tend_mflx(je,jk,jb)
            p_nh%diag%grf_bdy_mflx(jk,ic,1) = prep_adv%mass_flx_me(je,jk,jb) - dt_shift*p_nh%diag%grf_bdy_mflx(jk,ic,2)
          ENDDO

        ENDDO
!$OMP END DO

        !$ACC END PARALLEL

      ENDIF

!$OMP END PARALLEL


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

!$OMP PARALLEL PRIVATE (rl_start,rl_end,i_startblk,i_endblk)
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

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,z_vn_avg) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk, i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                         i_startidx, i_endidx, rl_start, rl_end)

        IF (istep == 1) THEN



    !$ser savepoint mo_solve_nonhydro_stencil_30_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_flx_avg=p_int%e_flx_avg(:,:,1)'

    !$ser data e_flx_avg=p_int%e_flx_avg(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing geofac_grdiv=p_int%geofac_grdiv(:,:,1)'

    !$ser data geofac_grdiv=p_int%geofac_grdiv(:,:,1)

    PRINT *, 'Serializing rbf_vec_coeff_e=p_int%rbf_vec_coeff_e_dsl(:,:,1)'

    !$ser data rbf_vec_coeff_e=p_int%rbf_vec_coeff_e_dsl(:,:,1)

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)

          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
!DIR$ IVDEP
            DO jk = 1, nlev
#else
!$NEC outerloop_unroll(8)
          DO jk = 1, nlev
!$NEC vovertake
            DO je = i_startidx, i_endidx
#endif
              ! Average normal wind components in order to get nearly second-order accurate divergence
              z_vn_avg(je,jk) = p_int%e_flx_avg(je,1,jb)*p_nh%prog(nnew)%vn(je,jk,jb)           &
                + p_int%e_flx_avg(je,2,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%e_flx_avg(je,3,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%e_flx_avg(je,4,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%e_flx_avg(je,5,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))

              ! Compute gradient of divergence of vn for divergence damping
#ifdef __LOOP_EXCHANGE
              z_graddiv_vn(jk,je,jb) = p_int%geofac_grdiv(je,1,jb)*p_nh%prog(nnew)%vn(je,jk,jb)    &
#else
              z_graddiv_vn(je,jk,jb) = p_int%geofac_grdiv(je,1,jb)*p_nh%prog(nnew)%vn(je,jk,jb)    &
#endif
              + p_int%geofac_grdiv(je,2,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%geofac_grdiv(je,3,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%geofac_grdiv(je,4,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%geofac_grdiv(je,5,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))

              ! RBF reconstruction of tangential wind component
              p_nh%diag%vt(je,jk,jb) = p_int%rbf_vec_coeff_e(1,je,jb)  &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%rbf_vec_coeff_e(2,je,jb)                       &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%rbf_vec_coeff_e(3,je,jb)                       &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%rbf_vec_coeff_e(4,je,jb)                       &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))
            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_30_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_flx_avg=p_int%e_flx_avg(:,:,1)'

    !$ser data e_flx_avg=p_int%e_flx_avg(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing geofac_grdiv=p_int%geofac_grdiv(:,:,1)'

    !$ser data geofac_grdiv=p_int%geofac_grdiv(:,:,1)

    PRINT *, 'Serializing rbf_vec_coeff_e=p_int%rbf_vec_coeff_e_dsl(:,:,1)'

    !$ser data rbf_vec_coeff_e=p_int%rbf_vec_coeff_e_dsl(:,:,1)

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)

    PRINT *, 'Serializing z_graddiv_vn=z_graddiv_vn(:,:,1)'

    !$ser accdata z_graddiv_vn=z_graddiv_vn(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

        ELSE IF (itime_scheme >= 5) THEN
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
!DIR$ IVDEP
            DO jk = 1, nlev
#else
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
#endif
              ! Average normal wind components in order to get nearly second-order accurate divergence
              z_vn_avg(je,jk) = p_int%e_flx_avg(je,1,jb)*p_nh%prog(nnew)%vn(je,jk,jb)           &
                + p_int%e_flx_avg(je,2,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%e_flx_avg(je,3,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%e_flx_avg(je,4,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%e_flx_avg(je,5,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))

              ! RBF reconstruction of tangential wind component
              p_nh%diag%vt(je,jk,jb) = p_int%rbf_vec_coeff_e(1,je,jb)  &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%rbf_vec_coeff_e(2,je,jb)                       &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%rbf_vec_coeff_e(3,je,jb)                       &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%rbf_vec_coeff_e(4,je,jb)                       &
                * p_nh%prog(nnew)%vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))

            ENDDO
          ENDDO
!$ACC END PARALLEL

        ELSE


    !$ser savepoint mo_solve_nonhydro_stencil_31_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_flx_avg=p_int%e_flx_avg(:,:,1)'

    !$ser data e_flx_avg=p_int%e_flx_avg(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)
!$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
!DIR$ IVDEP
            DO jk = 1, nlev
#else
!$NEC outerloop_unroll(8)
          DO jk = 1, nlev
!$NEC vovertake
            DO je = i_startidx, i_endidx
#endif
              ! Average normal wind components in order to get nearly second-order accurate divergence
              z_vn_avg(je,jk) = p_int%e_flx_avg(je,1,jb)*p_nh%prog(nnew)%vn(je,jk,jb)           &
                + p_int%e_flx_avg(je,2,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,1),jk,iqblk(je,jb,1)) &
                + p_int%e_flx_avg(je,3,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,2),jk,iqblk(je,jb,2)) &
                + p_int%e_flx_avg(je,4,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,3),jk,iqblk(je,jb,3)) &
                + p_int%e_flx_avg(je,5,jb)*p_nh%prog(nnew)%vn(iqidx(je,jb,4),jk,iqblk(je,jb,4))
            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_31_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_flx_avg=p_int%e_flx_avg(:,:,1)'

    !$ser data e_flx_avg=p_int%e_flx_avg(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)
        ENDIF

        IF (idiv_method == 1) THEN  ! Compute fluxes at edges using averaged velocities
                                  ! corresponding computation for idiv_method=2 follows later


    !$ser savepoint mo_solve_nonhydro_stencil_32_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)

    PRINT *, 'Serializing ddqz_z_full_e=p_nh%metrics%ddqz_z_full_e(:,:,1)'

    !$ser data ddqz_z_full_e=p_nh%metrics%ddqz_z_full_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

    PRINT *, 'Serializing mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)'

    !$ser accdata mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)'

    !$ser accdata z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)
!$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
          DO jk = 1,nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx

              p_nh%diag%mass_fl_e(je,jk,jb) = z_rho_e(je,jk,jb) *        &
                z_vn_avg(je,jk) * p_nh%metrics%ddqz_z_full_e(je,jk,jb)
              z_theta_v_fl_e(je,jk,jb) = p_nh%diag%mass_fl_e(je,jk,jb) * &
                z_theta_v_e(je,jk,jb)

            ENDDO
          ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_32_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_rho_e=z_rho_e(:,:,1)'

    !$ser accdata z_rho_e=z_rho_e(:,:,1)

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)

    PRINT *, 'Serializing ddqz_z_full_e=p_nh%metrics%ddqz_z_full_e(:,:,1)'

    !$ser data ddqz_z_full_e=p_nh%metrics%ddqz_z_full_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_e=z_theta_v_e(:,:,1)'

    !$ser accdata z_theta_v_e=z_theta_v_e(:,:,1)

    PRINT *, 'Serializing mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)'

    !$ser accdata mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)'

    !$ser accdata z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)

          IF (lsave_mflx .AND. istep == 2) THEN ! store mass flux for nest boundary interpolation
#ifndef _OPENACC
            DO je = i_startidx, i_endidx
              IF (p_patch%edges%refin_ctrl(je,jb) <= -4 .AND. p_patch%edges%refin_ctrl(je,jb) >= -6) THEN
!DIR$ IVDEP
                DO jk=1,nlev
                  p_nh%diag%mass_fl_e_sv(je,jk,jb) = p_nh%diag%mass_fl_e(je,jk,jb)
                ENDDO
              ENDIF
            ENDDO
#else
              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
              DO jk=1,nlev
                DO je = i_startidx, i_endidx
                  IF (p_patch%edges%refin_ctrl(je,jb) <= -4 .AND. p_patch%edges%refin_ctrl(je,jb) >= -6) THEN
                    p_nh%diag%mass_fl_e_sv(je,jk,jb) = p_nh%diag%mass_fl_e(je,jk,jb)
                  ENDIF
                ENDDO
              ENDDO
              !$ACC END PARALLEL
#endif
          ENDIF

          IF (lprep_adv .AND. istep == 2) THEN ! Preprations for tracer advection
            IF (lclean_mflx) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_33_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn_traj=prep_adv%vn_traj(:,:,1)'

    !$ser accdata vn_traj=prep_adv%vn_traj(:,:,1)

    PRINT *, 'Serializing mass_flx_me=prep_adv%mass_flx_me(:,:,1)'

    !$ser accdata mass_flx_me=prep_adv%mass_flx_me(:,:,1)
!$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
              DO jk = 1, nlev
!$NEC ivdep
                DO je = i_startidx, i_endidx
                  prep_adv%vn_traj(je,jk,jb)     = 0._wp
                  prep_adv%mass_flx_me(je,jk,jb) = 0._wp
                ENDDO
              ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_33_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn_traj=prep_adv%vn_traj(:,:,1)'

    !$ser accdata vn_traj=prep_adv%vn_traj(:,:,1)

    PRINT *, 'Serializing mass_flx_me=prep_adv%mass_flx_me(:,:,1)'

    !$ser accdata mass_flx_me=prep_adv%mass_flx_me(:,:,1)

            ENDIF


    !$ser savepoint mo_solve_nonhydro_stencil_34_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing r_nsubsteps=r_nsubsteps'

    !$ser data r_nsubsteps=r_nsubsteps

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)

    PRINT *, 'Serializing mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)'

    !$ser accdata mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)

    PRINT *, 'Serializing vn_traj=prep_adv%vn_traj(:,:,1)'

    !$ser accdata vn_traj=prep_adv%vn_traj(:,:,1)

    PRINT *, 'Serializing mass_flx_me=prep_adv%mass_flx_me(:,:,1)'

    !$ser accdata mass_flx_me=prep_adv%mass_flx_me(:,:,1)
!$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG(STATIC: 1) VECTOR TILE(32, 4)
            DO jk = 1, nlev
!$NEC ivdep
              DO je = i_startidx, i_endidx
                prep_adv%vn_traj(je,jk,jb)     = prep_adv%vn_traj(je,jk,jb)     + r_nsubsteps*z_vn_avg(je,jk)
                prep_adv%mass_flx_me(je,jk,jb) = prep_adv%mass_flx_me(je,jk,jb) + r_nsubsteps*p_nh%diag%mass_fl_e(je,jk,jb)
              ENDDO
            ENDDO
!$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_34_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing r_nsubsteps=r_nsubsteps'

    !$ser data r_nsubsteps=r_nsubsteps

    PRINT *, 'Serializing z_vn_avg=z_vn_avg(:,:)'

    !$ser accdata z_vn_avg=z_vn_avg(:,:)

    PRINT *, 'Serializing mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)'

    !$ser accdata mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)

    PRINT *, 'Serializing vn_traj=prep_adv%vn_traj(:,:,1)'

    !$ser accdata vn_traj=prep_adv%vn_traj(:,:,1)

    PRINT *, 'Serializing mass_flx_me=prep_adv%mass_flx_me(:,:,1)'

    !$ser accdata mass_flx_me=prep_adv%mass_flx_me(:,:,1)

          ENDIF

        ENDIF

        IF (istep == 1 .OR. itime_scheme >= 5) THEN
          ! Compute contravariant correction for vertical velocity at full levels


    !$ser savepoint mo_solve_nonhydro_stencil_35_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)'

    !$ser data ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)

    PRINT *, 'Serializing ddxt_z_full=p_nh%metrics%ddxt_z_full(:,:,1)'

    !$ser data ddxt_z_full=p_nh%metrics%ddxt_z_full(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing z_w_concorr_me=z_w_concorr_me(:,:,1)'

    !$ser accdata z_w_concorr_me=z_w_concorr_me(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = nflatlev(jg), nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              z_w_concorr_me(je,jk,jb) =                                          &
                p_nh%prog(nnew)%vn(je,jk,jb)*p_nh%metrics%ddxn_z_full(je,jk,jb) + &
                p_nh%diag%vt(je,jk,jb)      *p_nh%metrics%ddxt_z_full(je,jk,jb)
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_35_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)'

    !$ser data ddxn_z_full=p_nh%metrics%ddxn_z_full(:,:,1)

    PRINT *, 'Serializing ddxt_z_full=p_nh%metrics%ddxt_z_full(:,:,1)'

    !$ser data ddxt_z_full=p_nh%metrics%ddxt_z_full(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing z_w_concorr_me=z_w_concorr_me(:,:,1)'

    !$ser accdata z_w_concorr_me=z_w_concorr_me(:,:,1)
        ENDIF

        IF (istep == 1) THEN
          ! Interpolate vn to interface levels and compute horizontal part of kinetic energy on edges
          ! (needed in velocity tendencies called at istep=2)


    !$ser savepoint mo_solve_nonhydro_stencil_36_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_e=p_nh%metrics%wgtfac_e(:,:,1)'

    !$ser data wgtfac_e=p_nh%metrics%wgtfac_e(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing vn_ie=p_nh%diag%vn_ie(:,:,1)'

    !$ser accdata vn_ie=p_nh%diag%vn_ie(:,:,1)

    PRINT *, 'Serializing z_vt_ie=z_vt_ie(:,:,1)'

    !$ser accdata z_vt_ie=z_vt_ie(:,:,1)

    PRINT *, 'Serializing z_kin_hor_e=z_kin_hor_e(:,:,1)'

    !$ser accdata z_kin_hor_e=z_kin_hor_e(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
!$NEC outerloop_unroll(3)
          DO jk = 2, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              p_nh%diag%vn_ie(je,jk,jb) =                                                    &
                           p_nh%metrics%wgtfac_e(je,jk,jb) *p_nh%prog(nnew)%vn(je,jk  ,jb) + &
                  (1._wp - p_nh%metrics%wgtfac_e(je,jk,jb))*p_nh%prog(nnew)%vn(je,jk-1,jb)
              z_vt_ie(je,jk,jb) =                                                      &
                           p_nh%metrics%wgtfac_e(je,jk,jb) *p_nh%diag%vt(je,jk  ,jb) + &
                  (1._wp - p_nh%metrics%wgtfac_e(je,jk,jb))*p_nh%diag%vt(je,jk-1,jb)
              z_kin_hor_e(je,jk,jb) = 0.5_wp*(p_nh%prog(nnew)%vn(je,jk,jb)**2 + p_nh%diag%vt(je,jk,jb)**2)
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_36_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing wgtfac_e=p_nh%metrics%wgtfac_e(:,:,1)'

    !$ser data wgtfac_e=p_nh%metrics%wgtfac_e(:,:,1)

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing vn_ie=p_nh%diag%vn_ie(:,:,1)'

    !$ser accdata vn_ie=p_nh%diag%vn_ie(:,:,1)

    PRINT *, 'Serializing z_vt_ie=z_vt_ie(:,:,1)'

    !$ser accdata z_vt_ie=z_vt_ie(:,:,1)

    PRINT *, 'Serializing z_kin_hor_e=z_kin_hor_e(:,:,1)'

    !$ser accdata z_kin_hor_e=z_kin_hor_e(:,:,1)

          IF (.NOT. l_vert_nested) THEN
            ! Top and bottom levels
!DIR$ IVDEP


    !$ser savepoint mo_solve_nonhydro_stencil_37_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing vn_ie=p_nh%diag%vn_ie(:,:,1)'

    !$ser accdata vn_ie=p_nh%diag%vn_ie(:,:,1)

    PRINT *, 'Serializing z_vt_ie=z_vt_ie(:,:,1)'

    !$ser accdata z_vt_ie=z_vt_ie(:,:,1)

    PRINT *, 'Serializing z_kin_hor_e=z_kin_hor_e(:,:,1)'

    !$ser accdata z_kin_hor_e=z_kin_hor_e(:,:,1)


    !$ser savepoint mo_solve_nonhydro_stencil_38_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing wgtfacq_e=p_nh%metrics%wgtfacq_e_dsl(:,:,1)'

    !$ser data wgtfacq_e=p_nh%metrics%wgtfacq_e_dsl(:,:,1)

    PRINT *, 'Serializing vn_ie=p_nh%diag%vn_ie(:,:,1)'

    !$ser accdata vn_ie=p_nh%diag%vn_ie(:,:,1)

            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO je = i_startidx, i_endidx
              ! Quadratic extrapolation at the top turned out to cause numerical instability in pathological cases,
              ! thus we use a no-gradient condition in the upper half layer
              p_nh%diag%vn_ie(je,1,jb) = p_nh%prog(nnew)%vn(je,1,jb)
              ! vt_ie(jk=1) is actually unused, but we need it for convenience of implementation
              z_vt_ie(je,1,jb) = p_nh%diag%vt(je,1,jb)
              !
              z_kin_hor_e(je,1,jb) = 0.5_wp*(p_nh%prog(nnew)%vn(je,1,jb)**2 + p_nh%diag%vt(je,1,jb)**2)
              p_nh%diag%vn_ie(je,nlevp1,jb) =                           &
                p_nh%metrics%wgtfacq_e(je,1,jb)*p_nh%prog(nnew)%vn(je,nlev,jb) +   &
                p_nh%metrics%wgtfacq_e(je,2,jb)*p_nh%prog(nnew)%vn(je,nlev-1,jb) + &
                p_nh%metrics%wgtfacq_e(je,3,jb)*p_nh%prog(nnew)%vn(je,nlev-2,jb)
            ENDDO
            !$ACC END PARALLEL
    

    !$ser savepoint mo_solve_nonhydro_stencil_37_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing vt=p_nh%diag%vt(:,:,1)'

    !$ser accdata vt=p_nh%diag%vt(:,:,1)

    PRINT *, 'Serializing vn_ie=p_nh%diag%vn_ie(:,:,1)'

    !$ser accdata vn_ie=p_nh%diag%vn_ie(:,:,1)

    PRINT *, 'Serializing z_vt_ie=z_vt_ie(:,:,1)'

    !$ser accdata z_vt_ie=z_vt_ie(:,:,1)

    PRINT *, 'Serializing z_kin_hor_e=z_kin_hor_e(:,:,1)'

    !$ser accdata z_kin_hor_e=z_kin_hor_e(:,:,1)

    !$ser savepoint mo_solve_nonhydro_stencil_38_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing vn=p_nh%prog(nnew)%vn(:,:,1)'

    !$ser accdata vn=p_nh%prog(nnew)%vn(:,:,1)

    PRINT *, 'Serializing wgtfacq_e=p_nh%metrics%wgtfacq_e_dsl(:,:,1)'

    !$ser data wgtfacq_e=p_nh%metrics%wgtfacq_e_dsl(:,:,1)

    PRINT *, 'Serializing vn_ie=p_nh%diag%vn_ie(:,:,1)'

    !$ser accdata vn_ie=p_nh%diag%vn_ie(:,:,1)

          ELSE
            ! vn_ie(jk=1) is interpolated horizontally from the parent domain, and linearly interpolated in time
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              p_nh%diag%vn_ie(je,1,jb) = p_nh%diag%vn_ie_ubc(je,1,jb)+dt_linintp_ubc_nnew*p_nh%diag%vn_ie_ubc(je,2,jb)
              ! vt_ie(jk=1) is actually unused, but we need it for convenience of implementation
              z_vt_ie(je,1,jb) = p_nh%diag%vt(je,1,jb)
              !
              z_kin_hor_e(je,1,jb) = 0.5_wp*(p_nh%prog(nnew)%vn(je,1,jb)**2 + p_nh%diag%vt(je,1,jb)**2)
              p_nh%diag%vn_ie(je,nlevp1,jb) =                           &
                p_nh%metrics%wgtfacq_e(je,1,jb)*p_nh%prog(nnew)%vn(je,nlev,jb) +   &
                p_nh%metrics%wgtfacq_e(je,2,jb)*p_nh%prog(nnew)%vn(je,nlev-1,jb) + &
                p_nh%metrics%wgtfacq_e(je,3,jb)*p_nh%prog(nnew)%vn(je,nlev-2,jb)
            ENDDO
            !$ACC END PARALLEL
          ENDIF
        ENDIF

      ENDDO
!$OMP END DO

      ! Apply mass fluxes across lateral nest boundary interpolated from parent domain
      IF (jg > 1 .AND. grf_intmethod_e >= 5 .AND. idiv_method == 1) THEN

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        ! PGI 21.2 requires GANG-VECTOR on this level. (Having the jk as VECTOR crashes.)
        ! PRIVATE clause is required as je,jb are used in each vector thread.
        !$ACC LOOP GANG VECTOR PRIVATE(je, jb)

!$OMP DO PRIVATE(ic,je,jb,jk) ICON_OMP_DEFAULT_SCHEDULE
        DO ic = 1, p_nh%metrics%bdy_mflx_e_dim
          je = p_nh%metrics%bdy_mflx_e_idx(ic)
          jb = p_nh%metrics%bdy_mflx_e_blk(ic)

          ! This is needed for tracer mass consistency along the lateral boundaries
          IF (lprep_adv .AND. istep == 2) THEN ! subtract mass flux added previously...
            !$ACC LOOP SEQ
!$NEC ivdep
            DO jk = 1, nlev
              prep_adv%mass_flx_me(je,jk,jb) = prep_adv%mass_flx_me(je,jk,jb) - r_nsubsteps*p_nh%diag%mass_fl_e(je,jk,jb)
              prep_adv%vn_traj(je,jk,jb)     = prep_adv%vn_traj(je,jk,jb) - r_nsubsteps*p_nh%diag%mass_fl_e(je,jk,jb) / &
                (z_rho_e(je,jk,jb) * p_nh%metrics%ddqz_z_full_e(je,jk,jb))
            ENDDO
          ENDIF

!DIR$ IVDEP
          !$ACC LOOP SEQ
!$NEC ivdep
          DO jk = 1, nlev
            p_nh%diag%mass_fl_e(je,jk,jb) = p_nh%diag%grf_bdy_mflx(jk,ic,1) + &
              REAL(jstep,wp)*dtime*p_nh%diag%grf_bdy_mflx(jk,ic,2)
            z_theta_v_fl_e(je,jk,jb) = p_nh%diag%mass_fl_e(je,jk,jb) * z_theta_v_e(je,jk,jb)
          ENDDO

          IF (lprep_adv .AND. istep == 2) THEN ! ... and add the corrected one again
            !$ACC LOOP SEQ
!$NEC ivdep
            DO jk = 1, nlev
              prep_adv%mass_flx_me(je,jk,jb) = prep_adv%mass_flx_me(je,jk,jb) + r_nsubsteps*p_nh%diag%mass_fl_e(je,jk,jb)
              prep_adv%vn_traj(je,jk,jb)     = prep_adv%vn_traj(je,jk,jb) + r_nsubsteps*p_nh%diag%mass_fl_e(je,jk,jb) / &
                (z_rho_e(je,jk,jb) * p_nh%metrics%ddqz_z_full_e(je,jk,jb))
            ENDDO
          ENDIF

        ENDDO
!$OMP END DO

      !$ACC END PARALLEL

      ENDIF


      ! It turned out that it is sufficient to compute the contravariant correction in the
      ! predictor step at time level n+1; repeating the calculation in the corrector step
      ! has negligible impact on the results except in very-high resolution runs with extremely steep mountains
      IF (istep == 1 .OR. itime_scheme >= 5) THEN

        rl_start = 3
        rl_end = min_rlcell_int - 1

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

#ifdef _OPENACC
!
! This is one of the very few code divergences for OPENACC (see comment below)
!
        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
            i_startidx, i_endidx, rl_start, rl_end)

          ! ... and to interface levels


    !$ser savepoint mo_solve_nonhydro_stencil_39_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_bln_c_s=p_int%e_bln_c_s(:,:,1)'

    !$ser data e_bln_c_s=p_int%e_bln_c_s(:,:,1)

    PRINT *, 'Serializing z_w_concorr_me=z_w_concorr_me(:,:,1)'

    !$ser accdata z_w_concorr_me=z_w_concorr_me(:,:,1)

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR TILE(32, 4) PRIVATE(z_w_concorr_mc_m1, z_w_concorr_mc_m0)
          DO jk = nflatlev(jg)+1, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              ! COMMENT: this optimization yields drastically better performance in an OpenACC context
              ! Interpolate contravariant correction to cell centers...
              z_w_concorr_mc_m1 =  &
                p_int%e_bln_c_s(jc,1,jb)*z_w_concorr_me(ieidx(jc,jb,1),jk-1,ieblk(jc,jb,1)) + &
                p_int%e_bln_c_s(jc,2,jb)*z_w_concorr_me(ieidx(jc,jb,2),jk-1,ieblk(jc,jb,2)) + &
                p_int%e_bln_c_s(jc,3,jb)*z_w_concorr_me(ieidx(jc,jb,3),jk-1,ieblk(jc,jb,3))
              z_w_concorr_mc_m0 =  &
                p_int%e_bln_c_s(jc,1,jb)*z_w_concorr_me(ieidx(jc,jb,1),jk,ieblk(jc,jb,1)) + &
                p_int%e_bln_c_s(jc,2,jb)*z_w_concorr_me(ieidx(jc,jb,2),jk,ieblk(jc,jb,2)) + &
                p_int%e_bln_c_s(jc,3,jb)*z_w_concorr_me(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))
              p_nh%diag%w_concorr_c(jc,jk,jb) =                                &
                p_nh%metrics%wgtfac_c(jc,jk,jb)*z_w_concorr_mc_m0 +        &
                (1._vp - p_nh%metrics%wgtfac_c(jc,jk,jb))*z_w_concorr_mc_m1
            ENDDO
          ENDDO
          !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_39_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_bln_c_s=p_int%e_bln_c_s(:,:,1)'

    !$ser data e_bln_c_s=p_int%e_bln_c_s(:,:,1)

    PRINT *, 'Serializing z_w_concorr_me=z_w_concorr_me(:,:,1)'

    !$ser accdata z_w_concorr_me=z_w_concorr_me(:,:,1)

    PRINT *, 'Serializing wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)'

    !$ser data wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)


    !$ser savepoint mo_solve_nonhydro_stencil_40_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_bln_c_s=p_int%e_bln_c_s(:,:,1)'

    !$ser data e_bln_c_s=p_int%e_bln_c_s(:,:,1)

    PRINT *, 'Serializing z_w_concorr_me=z_w_concorr_me(:,:,1)'

    !$ser accdata z_w_concorr_me=z_w_concorr_me(:,:,1)

    PRINT *, 'Serializing wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)'

    !$ser data wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR PRIVATE(z_w_concorr_mc_m2, z_w_concorr_mc_m1, z_w_concorr_mc_m0)
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            ! Interpolate contravariant correction to cell centers...
            z_w_concorr_mc_m2 =  &
              p_int%e_bln_c_s(jc,1,jb)*z_w_concorr_me(ieidx(jc,jb,1),nlev-2,ieblk(jc,jb,1)) + &
              p_int%e_bln_c_s(jc,2,jb)*z_w_concorr_me(ieidx(jc,jb,2),nlev-2,ieblk(jc,jb,2)) + &
              p_int%e_bln_c_s(jc,3,jb)*z_w_concorr_me(ieidx(jc,jb,3),nlev-2,ieblk(jc,jb,3))

            z_w_concorr_mc_m1 =  &
              p_int%e_bln_c_s(jc,1,jb)*z_w_concorr_me(ieidx(jc,jb,1),nlev-1,ieblk(jc,jb,1)) + &
              p_int%e_bln_c_s(jc,2,jb)*z_w_concorr_me(ieidx(jc,jb,2),nlev-1,ieblk(jc,jb,2)) + &
              p_int%e_bln_c_s(jc,3,jb)*z_w_concorr_me(ieidx(jc,jb,3),nlev-1,ieblk(jc,jb,3))

            z_w_concorr_mc_m0   =  &
              p_int%e_bln_c_s(jc,1,jb)*z_w_concorr_me(ieidx(jc,jb,1),nlev,ieblk(jc,jb,1)) + &
              p_int%e_bln_c_s(jc,2,jb)*z_w_concorr_me(ieidx(jc,jb,2),nlev,ieblk(jc,jb,2)) + &
              p_int%e_bln_c_s(jc,3,jb)*z_w_concorr_me(ieidx(jc,jb,3),nlev,ieblk(jc,jb,3))

            p_nh%diag%w_concorr_c(jc,nlevp1,jb) =                         &
              p_nh%metrics%wgtfacq_c(jc,1,jb)*z_w_concorr_mc_m0 +         &
              p_nh%metrics%wgtfacq_c(jc,2,jb)*z_w_concorr_mc_m1 +       &
              p_nh%metrics%wgtfacq_c(jc,3,jb)*z_w_concorr_mc_m2
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_40_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing e_bln_c_s=p_int%e_bln_c_s(:,:,1)'

    !$ser data e_bln_c_s=p_int%e_bln_c_s(:,:,1)

    PRINT *, 'Serializing z_w_concorr_me=z_w_concorr_me(:,:,1)'

    !$ser accdata z_w_concorr_me=z_w_concorr_me(:,:,1)

    PRINT *, 'Serializing wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)'

    !$ser data wgtfacq_c=p_nh%metrics%wgtfacq_c_dsl(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

        ENDDO
#else
!
! OMP-only code
!
!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc,z_w_concorr_mc) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        ! Interpolate contravariant correction to cell centers...
#ifdef __LOOP_EXCHANGE
          DO jc = i_startidx, i_endidx
!DIR$ IVDEP
            DO jk = nflatlev(jg), nlev
#else
          DO jk = nflatlev(jg), nlev
            DO jc = i_startidx, i_endidx
#endif

              z_w_concorr_mc(jc,jk) =  &
                p_int%e_bln_c_s(jc,1,jb)*z_w_concorr_me(ieidx(jc,jb,1),jk,ieblk(jc,jb,1)) + &
                p_int%e_bln_c_s(jc,2,jb)*z_w_concorr_me(ieidx(jc,jb,2),jk,ieblk(jc,jb,2)) + &
                p_int%e_bln_c_s(jc,3,jb)*z_w_concorr_me(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))

            ENDDO
          ENDDO

          ! ... and to interface levels
          DO jk = nflatlev(jg)+1, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              p_nh%diag%w_concorr_c(jc,jk,jb) =                                &
                p_nh%metrics%wgtfac_c(jc,jk,jb)*z_w_concorr_mc(jc,jk) +        &
               (1._vp - p_nh%metrics%wgtfac_c(jc,jk,jb))*z_w_concorr_mc(jc,jk-1)
            ENDDO
          ENDDO
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            p_nh%diag%w_concorr_c(jc,nlevp1,jb) =                         &
              p_nh%metrics%wgtfacq_c(jc,1,jb)*z_w_concorr_mc(jc,nlev) +   &
              p_nh%metrics%wgtfacq_c(jc,2,jb)*z_w_concorr_mc(jc,nlev-1) + &
              p_nh%metrics%wgtfacq_c(jc,3,jb)*z_w_concorr_mc(jc,nlev-2)
          ENDDO

        ENDDO
!$OMP END DO
#endif
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
!$OMP BARRIER
        ENDIF

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
            i_startidx, i_endidx, rl_start, rl_end)

          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
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
!$OMP END DO

      ENDIF  ! idiv_method = 2

!$OMP END PARALLEL

      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_edgecomp)
        CALL timer_start(timer_solve_nh_vimpl)
      ENDIF

      IF (idiv_method == 2) THEN ! use averaged divergence - idiv_method=1 is inlined for better cache efficiency

!TODO remove the wait after everything is ASYNC(1)
        !$ACC WAIT

        ! horizontal divergences of rho and rhotheta are processed in one step for efficiency
        CALL div_avg(p_nh%diag%mass_fl_e, p_patch, p_int, p_int%c_bln_avg, z_mass_fl_div, &
                     opt_in2=z_theta_v_fl_e, opt_out2=z_theta_v_fl_div, opt_rlstart=4,    &
                     opt_rlend=min_rlcell_int)
      ENDIF

!$OMP PARALLEL PRIVATE (rl_start,rl_end,i_startblk,i_endblk,jk_start)

      rl_start = grf_bdywidth_c+1
      rl_end   = min_rlcell_int

      i_startblk = p_patch%cells%start_block(rl_start)
      i_endblk   = p_patch%cells%end_block(rl_end)

      IF (l_vert_nested) THEN
        jk_start = 2
      ELSE
        jk_start = 1
      ENDIF

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc,z_w_expl,z_contr_w_fl_l,z_rho_expl,z_exner_expl, &
!$OMP   z_a,z_b,z_c,z_g,z_q,z_alpha,z_beta,z_gamma,ic,z_flxdiv_mass,z_flxdiv_theta  ) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk, i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        IF (idiv_method == 1) THEN
        ! horizontal divergences of rho and rhotheta are inlined and processed in one step for efficiency


    !$ser savepoint mo_solve_nonhydro_stencil_41_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing geofac_div=p_int%geofac_div(:,:,1)'

    !$ser data geofac_div=p_int%geofac_div(:,:,1)

    PRINT *, 'Serializing mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)'

    !$ser accdata mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)'

    !$ser accdata z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)

    PRINT *, 'Serializing z_flxdiv_mass=z_flxdiv_mass(:,:)'

    !$ser accdata z_flxdiv_mass=z_flxdiv_mass(:,:)

    PRINT *, 'Serializing z_flxdiv_theta=z_flxdiv_theta(:,:)'

    !$ser accdata z_flxdiv_theta=z_flxdiv_theta(:,:)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
#ifdef __LOOP_EXCHANGE
          DO jc = i_startidx, i_endidx
!DIR$ IVDEP, PREFERVECTOR
            DO jk = 1, nlev
#else
!$NEC outerloop_unroll(8)
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
#endif
              z_flxdiv_mass(jc,jk) =  &
                p_nh%diag%mass_fl_e(ieidx(jc,jb,1),jk,ieblk(jc,jb,1)) * p_int%geofac_div(jc,1,jb) + &
                p_nh%diag%mass_fl_e(ieidx(jc,jb,2),jk,ieblk(jc,jb,2)) * p_int%geofac_div(jc,2,jb) + &
                p_nh%diag%mass_fl_e(ieidx(jc,jb,3),jk,ieblk(jc,jb,3)) * p_int%geofac_div(jc,3,jb)

              z_flxdiv_theta(jc,jk) =  &
                z_theta_v_fl_e(ieidx(jc,jb,1),jk,ieblk(jc,jb,1)) * p_int%geofac_div(jc,1,jb) + &
                z_theta_v_fl_e(ieidx(jc,jb,2),jk,ieblk(jc,jb,2)) * p_int%geofac_div(jc,2,jb) + &
                z_theta_v_fl_e(ieidx(jc,jb,3),jk,ieblk(jc,jb,3)) * p_int%geofac_div(jc,3,jb)
            END DO
          END DO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_41_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing geofac_div=p_int%geofac_div(:,:,1)'

    !$ser data geofac_div=p_int%geofac_div(:,:,1)

    PRINT *, 'Serializing mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)'

    !$ser accdata mass_fl_e=p_nh%diag%mass_fl_e(:,:,1)

    PRINT *, 'Serializing z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)'

    !$ser accdata z_theta_v_fl_e=z_theta_v_fl_e(:,:,1)

    PRINT *, 'Serializing z_flxdiv_mass=z_flxdiv_mass(:,:)'

    !$ser accdata z_flxdiv_mass=z_flxdiv_mass(:,:)

    PRINT *, 'Serializing z_flxdiv_theta=z_flxdiv_theta(:,:)'

    !$ser accdata z_flxdiv_theta=z_flxdiv_theta(:,:)

        ELSE ! idiv_method = 2 - just copy values to local 2D array

          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
              z_flxdiv_mass(jc,jk)  = z_mass_fl_div(jc,jk,jb)
              z_flxdiv_theta(jc,jk) = z_theta_v_fl_div(jc,jk,jb)
            END DO
          END DO
          !$ACC END PARALLEL

        ENDIF

        ! upper boundary conditions for rho_ic and theta_v_ic in the case of vertical nesting
        !
        ! kept constant during predictor/corrector step, and linearly interpolated for 
        ! each dynamics substep. 
        ! Hence, copying them every dynamics substep during the predictor step (istep=1) is sufficient. 
        IF (l_vert_nested .AND. istep == 1) THEN
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx

            p_nh%diag%theta_v_ic(jc,1,jb) = p_nh%diag%theta_v_ic_ubc(jc,jb,1)  &
              &                           + dt_linintp_ubc * p_nh%diag%theta_v_ic_ubc(jc,jb,2)

            p_nh%diag%rho_ic(jc,1,jb) = p_nh%diag%rho_ic_ubc(jc,jb,1)  &
              &                       + dt_linintp_ubc * p_nh%diag%rho_ic_ubc(jc,jb,2)
 
            z_mflx_top(jc,jb) = p_nh%diag%mflx_ic_ubc(jc,jb,1)  &
              &               + dt_linintp_ubc * p_nh%diag%mflx_ic_ubc(jc,jb,2)

          ENDDO
          !$ACC END PARALLEL
        ENDIF

        ! Start of vertically implicit solver part for sound-wave terms;
        ! advective terms and gravity-wave terms are treated explicitly
        !
        IF (istep == 2 .AND. (itime_scheme >= 4)) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_42_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing wgt_nnew_vel=wgt_nnew_vel'

    !$ser data wgt_nnew_vel=wgt_nnew_vel

    PRINT *, 'Serializing wgt_nnow_vel=wgt_nnow_vel'

    !$ser data wgt_nnow_vel=wgt_nnow_vel

    PRINT *, 'Serializing z_w_expl=z_w_expl(:,:)'

    !$ser accdata z_w_expl=z_w_expl(:,:)

    PRINT *, 'Serializing w_nnow=p_nh%prog(nnow)%w(:,:,jb)'

    !$ser data w_nnow=p_nh%prog(nnow)%w(:,:,jb)

    PRINT *, 'Serializing ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)'

    !$ser data ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)

    PRINT *, 'Serializing ddt_w_adv_ntl2=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl2)'

    !$ser data ddt_w_adv_ntl2=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl2)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,jb)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,jb)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 2, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx

              ! explicit part for w - use temporally averaged advection terms for better numerical stability
              ! the explicit weight for the pressure-gradient term is already included in z_th_ddz_exner_c
              z_w_expl(jc,jk) = p_nh%prog(nnow)%w(jc,jk,jb) + dtime *   &
                (wgt_nnow_vel*p_nh%diag%ddt_w_adv_pc(jc,jk,jb,ntl1) +   &
                 wgt_nnew_vel*p_nh%diag%ddt_w_adv_pc(jc,jk,jb,ntl2)     &
                 -cpd*z_th_ddz_exner_c(jc,jk,jb) )

              ! contravariant vertical velocity times density for explicit part
              z_contr_w_fl_l(jc,jk) = p_nh%diag%rho_ic(jc,jk,jb)*(-p_nh%diag%w_concorr_c(jc,jk,jb) &
                + p_nh%metrics%vwind_expl_wgt(jc,jb)*p_nh%prog(nnow)%w(jc,jk,jb) )

            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_42_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing wgt_nnew_vel=wgt_nnew_vel'

    !$ser data wgt_nnew_vel=wgt_nnew_vel

    PRINT *, 'Serializing wgt_nnow_vel=wgt_nnow_vel'

    !$ser data wgt_nnow_vel=wgt_nnow_vel

    PRINT *, 'Serializing z_w_expl=z_w_expl(:,:)'

    !$ser accdata z_w_expl=z_w_expl(:,:)

    PRINT *, 'Serializing w_nnow=p_nh%prog(nnow)%w(:,:,jb)'

    !$ser data w_nnow=p_nh%prog(nnow)%w(:,:,jb)

    PRINT *, 'Serializing ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)'

    !$ser data ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)

    PRINT *, 'Serializing ddt_w_adv_ntl2=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl2)'

    !$ser data ddt_w_adv_ntl2=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl2)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,jb)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,jb)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)

        ELSE


    !$ser savepoint mo_solve_nonhydro_stencil_43_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_w_expl=z_w_expl(:,:)'

    !$ser accdata z_w_expl=z_w_expl(:,:)

    PRINT *, 'Serializing w_nnow=p_nh%prog(nnow)%w(:,:,jb)'

    !$ser data w_nnow=p_nh%prog(nnow)%w(:,:,jb)

    PRINT *, 'Serializing ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)'

    !$ser data ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,jb)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,jb)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 2, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx

              ! explicit part for w
              z_w_expl(jc,jk) = p_nh%prog(nnow)%w(jc,jk,jb) + dtime *                &
                (p_nh%diag%ddt_w_adv_pc(jc,jk,jb,ntl1)-cpd*z_th_ddz_exner_c(jc,jk,jb))

              ! contravariant vertical velocity times density for explicit part
              z_contr_w_fl_l(jc,jk) = p_nh%diag%rho_ic(jc,jk,jb)*(-p_nh%diag%w_concorr_c(jc,jk,jb) &
                + p_nh%metrics%vwind_expl_wgt(jc,jb)*p_nh%prog(nnow)%w(jc,jk,jb) )

            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_43_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_w_expl=z_w_expl(:,:)'

    !$ser accdata z_w_expl=z_w_expl(:,:)

    PRINT *, 'Serializing w_nnow=p_nh%prog(nnow)%w(:,:,jb)'

    !$ser data w_nnow=p_nh%prog(nnow)%w(:,:,jb)

    PRINT *, 'Serializing ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)'

    !$ser data ddt_w_adv_ntl1=p_nh%diag%ddt_w_adv_pc(:,:,jb,ntl1)

    PRINT *, 'Serializing z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)'

    !$ser accdata z_th_ddz_exner_c=z_th_ddz_exner_c(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,jb)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,jb)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,jb)

        ENDIF

        ! Solver coefficients

    !$ser savepoint mo_solve_nonhydro_stencil_44_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cvd=cvd'

    !$ser data cvd=cvd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing rd=rd'

    !$ser data rd=rd

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing exner_nnow=p_nh%prog(nnow)%exner(:,:,jb)'

    !$ser data exner_nnow=p_nh%prog(nnow)%exner(:,:,jb)

    PRINT *, 'Serializing rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)'

    !$ser data rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)

    PRINT *, 'Serializing theta_v_nnow=p_nh%prog(nnow)%theta_v(:,:,jb)'

    !$ser data theta_v_nnow=p_nh%prog(nnow)%theta_v(:,:,jb)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,jb)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,jb)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,jb)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,jb)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR COLLAPSE(2)
        DO jk = 1, nlev
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            z_beta(jc,jk)=dtime*rd*p_nh%prog(nnow)%exner(jc,jk,jb) /                 &
              (cvd*p_nh%prog(nnow)%rho(jc,jk,jb)*p_nh%prog(nnow)%theta_v(jc,jk,jb)) * &
              p_nh%metrics%inv_ddqz_z_full(jc,jk,jb)

            z_alpha(jc,jk)= p_nh%metrics%vwind_impl_wgt(jc,jb)*         &
              &  p_nh%diag%theta_v_ic(jc,jk,jb)*p_nh%diag%rho_ic(jc,jk,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_44_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cvd=cvd'

    !$ser data cvd=cvd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing rd=rd'

    !$ser data rd=rd

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing exner_nnow=p_nh%prog(nnow)%exner(:,:,jb)'

    !$ser data exner_nnow=p_nh%prog(nnow)%exner(:,:,jb)

    PRINT *, 'Serializing rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)'

    !$ser data rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)

    PRINT *, 'Serializing theta_v_nnow=p_nh%prog(nnow)%theta_v(:,:,jb)'

    !$ser data theta_v_nnow=p_nh%prog(nnow)%theta_v(:,:,jb)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,jb)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,jb)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,jb)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,jb)



    !$ser savepoint mo_solve_nonhydro_stencil_45_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)


    !$ser savepoint mo_solve_nonhydro_stencil_45_b_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_q=z_q(:,:)'

    !$ser accdata z_q=z_q(:,:)

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR
        DO jc = i_startidx, i_endidx
          z_alpha(jc,nlevp1) = 0.0_wp
          !
          ! Note: z_q is used in the tridiagonal matrix solver for w below.
          !       z_q(1) is always zero, irrespective of w(1)=0 or w(1)/=0
          !       z_q(1)=0 is equivalent to cp(slev)=c(slev)/b(slev) in mo_math_utilities:tdma_solver_vec 
          z_q(jc,1) = 0._vp
        ENDDO
        !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_45_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    !$ser savepoint mo_solve_nonhydro_stencil_45_b_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_q=z_q(:,:)'

    !$ser accdata z_q=z_q(:,:)


        ! upper boundary condition for w (interpolated from parent domain in case of vertical nesting)
        ! Note: the upper b.c. reduces to w(1) = 0 in the absence of diabatic heating
        IF (l_open_ubc .AND. .NOT. l_vert_nested) THEN
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            p_nh%prog(nnew)%w(jc,1,jb) = z_thermal_exp(jc,jb)
            z_contr_w_fl_l(jc,1) = p_nh%diag%rho_ic(jc,1,jb)*p_nh%prog(nnow)%w(jc,1,jb)   &
              * p_nh%metrics%vwind_expl_wgt(jc,jb)
          ENDDO
          !$ACC END PARALLEL
        ELSE IF (.NOT. l_open_ubc .AND. .NOT. l_vert_nested) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_46_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing w_nnew=p_nh%prog(nnew)%w(:,:,jb)'

    !$ser accdata w_nnew=p_nh%prog(nnew)%w(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            p_nh%prog(nnew)%w(jc,1,jb) = 0._wp
            z_contr_w_fl_l(jc,1)       = 0._wp
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_46_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing w_nnew=p_nh%prog(nnew)%w(:,:,jb)'

    !$ser accdata w_nnew=p_nh%prog(nnew)%w(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

        ELSE  ! l_vert_nested
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            ! UBC for w: horizontally interpolated from the parent interface level, 
            !            and linearly interpolated in time.
            p_nh%prog(nnew)%w(jc,1,jb) = p_nh%diag%w_ubc(jc,jb,1)  &
              &                        + dt_linintp_ubc_nnew * p_nh%diag%w_ubc(jc,jb,2)
            !
            z_contr_w_fl_l(jc,1) = z_mflx_top(jc,jb) * p_nh%metrics%vwind_expl_wgt(jc,jb)
          ENDDO
          !$ACC END PARALLEL
        ENDIF

        ! lower boundary condition for w, consistent with contravariant correction

    !$ser savepoint mo_solve_nonhydro_stencil_47_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing w_nnew=p_nh%prog(nnew)%w(:,:,jb)'

    !$ser accdata w_nnew=p_nh%prog(nnew)%w(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
        DO jc = i_startidx, i_endidx
          p_nh%prog(nnew)%w(jc,nlevp1,jb) = p_nh%diag%w_concorr_c(jc,nlevp1,jb)
          z_contr_w_fl_l(jc,nlevp1)       = 0.0_wp
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_47_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing w_nnew=p_nh%prog(nnew)%w(:,:,jb)'

    !$ser accdata w_nnew=p_nh%prog(nnew)%w(:,:,jb)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,jb)


        ! Explicit parts of density and Exner pressure
        !
        ! Top level first


    !$ser savepoint mo_solve_nonhydro_stencil_48_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)'

    !$ser data rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)

    PRINT *, 'Serializing z_flxdiv_mass=z_flxdiv_mass(:,:)'

    !$ser accdata z_flxdiv_mass=z_flxdiv_mass(:,:)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,jb)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,jb)

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing z_flxdiv_theta=z_flxdiv_theta(:,:)'

    !$ser accdata z_flxdiv_theta=z_flxdiv_theta(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)

    PRINT *, 'Serializing ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)'

    !$ser data ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
        DO jc = i_startidx, i_endidx
          z_rho_expl(jc,1)=        p_nh%prog(nnow)%rho(jc,1,jb)   &
            &        -dtime*p_nh%metrics%inv_ddqz_z_full(jc,1,jb) &
            &                            *(z_flxdiv_mass(jc,1)    &
            &                            +z_contr_w_fl_l(jc,1   ) &
            &                            -z_contr_w_fl_l(jc,2   ))

          z_exner_expl(jc,1)=     p_nh%diag%exner_pr(jc,1,jb)      &
            &      -z_beta (jc,1)*(z_flxdiv_theta(jc,1)            &
            & +p_nh%diag%theta_v_ic(jc,1,jb)*z_contr_w_fl_l(jc,1)  &
            & -p_nh%diag%theta_v_ic(jc,2,jb)*z_contr_w_fl_l(jc,2)) &
            & +dtime*p_nh%diag%ddt_exner_phy(jc,1,jb)
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_48_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)'

    !$ser data rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)

    PRINT *, 'Serializing z_flxdiv_mass=z_flxdiv_mass(:,:)'

    !$ser accdata z_flxdiv_mass=z_flxdiv_mass(:,:)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,jb)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,jb)

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing z_flxdiv_theta=z_flxdiv_theta(:,:)'

    !$ser accdata z_flxdiv_theta=z_flxdiv_theta(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)

    PRINT *, 'Serializing ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)'

    !$ser data ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)

        ! Other levels

    !$ser savepoint mo_solve_nonhydro_stencil_49_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)'

    !$ser data rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)

    PRINT *, 'Serializing z_flxdiv_mass=z_flxdiv_mass(:,:)'

    !$ser accdata z_flxdiv_mass=z_flxdiv_mass(:,:)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,jb)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,jb)

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing z_flxdiv_theta=z_flxdiv_theta(:,:)'

    !$ser accdata z_flxdiv_theta=z_flxdiv_theta(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)

    PRINT *, 'Serializing ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)'

    !$ser data ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR COLLAPSE(2)
        DO jk = 2, nlev
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            z_rho_expl(jc,jk)=       p_nh%prog(nnow)%rho(jc,jk  ,jb) &
              &        -dtime*p_nh%metrics%inv_ddqz_z_full(jc,jk  ,jb) &
              &                            *(z_flxdiv_mass(jc,jk     ) &
              &                            +z_contr_w_fl_l(jc,jk     ) &
              &                             -z_contr_w_fl_l(jc,jk+1   ))

            z_exner_expl(jc,jk)=    p_nh%diag%exner_pr(jc,jk,jb) - z_beta(jc,jk) &
              &                             *(z_flxdiv_theta(jc,jk)              &
              &   +p_nh%diag%theta_v_ic(jc,jk  ,jb)*z_contr_w_fl_l(jc,jk  )      &
              &   -p_nh%diag%theta_v_ic(jc,jk+1,jb)*z_contr_w_fl_l(jc,jk+1))     &
              &   +dtime*p_nh%diag%ddt_exner_phy(jc,jk,jb)

          ENDDO
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_49_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)'

    !$ser data rho_nnow=p_nh%prog(nnow)%rho(:,:,jb)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,jb)

    PRINT *, 'Serializing z_flxdiv_mass=z_flxdiv_mass(:,:)'

    !$ser accdata z_flxdiv_mass=z_flxdiv_mass(:,:)

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing exner_pr=p_nh%diag%exner_pr(:,:,jb)'

    !$ser accdata exner_pr=p_nh%diag%exner_pr(:,:,jb)

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing z_flxdiv_theta=z_flxdiv_theta(:,:)'

    !$ser accdata z_flxdiv_theta=z_flxdiv_theta(:,:)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,jb)

    PRINT *, 'Serializing ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)'

    !$ser data ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,jb)


        IF (is_iau_active) THEN ! add analysis increments from data assimilation to density and exner pressure
          

    !$ser savepoint mo_solve_nonhydro_stencil_50_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing iau_wgt_dyn=iau_wgt_dyn'

    !$ser data iau_wgt_dyn=iau_wgt_dyn

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing rho_incr=p_nh%diag%rho_incr(:,:,jb)'

    !$ser data rho_incr=p_nh%diag%rho_incr(:,:,jb)

    PRINT *, 'Serializing exner_incr=p_nh%diag%exner_incr(:,:,jb)'

    !$ser data exner_incr=p_nh%diag%exner_incr(:,:,jb)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              z_rho_expl(jc,jk)   = z_rho_expl(jc,jk)   + iau_wgt_dyn*p_nh%diag%rho_incr(jc,jk,jb)
              z_exner_expl(jc,jk) = z_exner_expl(jc,jk) + iau_wgt_dyn*p_nh%diag%exner_incr(jc,jk,jb)
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_50_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing iau_wgt_dyn=iau_wgt_dyn'

    !$ser data iau_wgt_dyn=iau_wgt_dyn

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing rho_incr=p_nh%diag%rho_incr(:,:,jb)'

    !$ser data rho_incr=p_nh%diag%rho_incr(:,:,jb)

    PRINT *, 'Serializing exner_incr=p_nh%diag%exner_incr(:,:,jb)'

    !$ser data exner_incr=p_nh%diag%exner_incr(:,:,jb)

        ENDIF

        !
        ! Solve tridiagonal matrix for w
        !
! TODO: not parallelized


    !$ser savepoint mo_solve_nonhydro_stencil_52_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

    PRINT *, 'Serializing ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)'

    !$ser data ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing z_w_expl=z_w_expl(:,:)'

    !$ser accdata z_w_expl=z_w_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing z_q=z_q(:,:)'

    !$ser accdata z_q=z_q(:,:)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP SEQ
!$NEC outerloop_unroll(8)
        DO jk = 2, nlev
!DIR$ IVDEP
!$NEC ivdep
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            z_gamma = dtime*cpd*p_nh%metrics%vwind_impl_wgt(jc,jb)*    &
              p_nh%diag%theta_v_ic(jc,jk,jb)/p_nh%metrics%ddqz_z_half(jc,jk,jb)
            z_a  = -z_gamma*z_beta(jc,jk-1)*z_alpha(jc,jk-1)
            z_c = -z_gamma*z_beta(jc,jk  )*z_alpha(jc,jk+1)
            z_b = 1.0_vp+z_gamma*z_alpha(jc,jk) &
              *(z_beta(jc,jk-1)+z_beta(jc,jk))
            z_g = 1.0_vp/(z_b+z_a*z_q(jc,jk-1))
            z_q(jc,jk) = - z_c*z_g
            p_nh%prog(nnew)%w(jc,jk,jb) = z_w_expl(jc,jk) - z_gamma  &
              &      *(z_exner_expl(jc,jk-1)-z_exner_expl(jc,jk))
            p_nh%prog(nnew)%w(jc,jk,jb) = (p_nh%prog(nnew)%w(jc,jk,jb)  &
              -z_a*p_nh%prog(nnew)%w(jc,jk-1,jb))*z_g
          ENDDO
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_52_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cpd=cpd'

    !$ser data cpd=cpd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)'

    !$ser accdata theta_v_ic=p_nh%diag%theta_v_ic(:,:,1)

    PRINT *, 'Serializing ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)'

    !$ser data ddqz_z_half=p_nh%metrics%ddqz_z_half(:,:,1)

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    PRINT *, 'Serializing z_beta=z_beta(:,:)'

    !$ser accdata z_beta=z_beta(:,:)

    PRINT *, 'Serializing z_w_expl=z_w_expl(:,:)'

    !$ser accdata z_w_expl=z_w_expl(:,:)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing z_q=z_q(:,:)'

    !$ser accdata z_q=z_q(:,:)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)


    !$ser savepoint mo_solve_nonhydro_stencil_53_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_q=z_q'

    !$ser accdata z_q=z_q

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP SEQ
        DO jk = nlev-1, 2, -1
!DIR$ IVDEP
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx
            p_nh%prog(nnew)%w(jc,jk,jb) = p_nh%prog(nnew)%w(jc,jk,jb)&
              &             +p_nh%prog(nnew)%w(jc,jk+1,jb)*z_q(jc,jk)
          ENDDO
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_53_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_q=z_q'

    !$ser accdata z_q=z_q

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

        ! Rayleigh damping mechanism (Klemp,Dudhia,Hassiotis: MWR136,pp.3987-4004)
        !
        IF ( rayleigh_type == RAYLEIGH_KLEMP ) THEN

!$ACC PARALLEL IF( i_am_accel_node ) DEFAULT(PRESENT) ASYNC(1)
!$ACC LOOP GANG VECTOR COLLAPSE(1)
DO jc = 1, nproma
  w_1(jc,jb) = p_nh%prog(nnew)%w(jc,1,jb)
ENDDO
!$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_54_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_raylfac=z_raylfac(:)'

    !$ser data z_raylfac=z_raylfac(:)

    PRINT *, 'Serializing w_1=w_1(:,1)'

    !$ser data w_1=w_1(:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 2, nrdmax(jg)
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              p_nh%prog(nnew)%w(jc,jk,jb) = z_raylfac(jk)*p_nh%prog(nnew)%w(jc,jk,jb) +    &
                                            (1._wp-z_raylfac(jk))*p_nh%prog(nnew)%w(jc,1,jb)
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_54_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing z_raylfac=z_raylfac(:)'

    !$ser data z_raylfac=z_raylfac(:)

    PRINT *, 'Serializing w_1=w_1(:,1)'

    !$ser data w_1=w_1(:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

        ! Classic Rayleigh damping mechanism for w (requires reference state !!)
        !
        ELSE IF ( rayleigh_type == RAYLEIGH_CLASSIC ) THEN

          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 2, nrdmax(jg)
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              p_nh%prog(nnew)%w(jc,jk,jb) = p_nh%prog(nnew)%w(jc,jk,jb)       &
                &                         - dtime*p_nh%metrics%rayleigh_w(jk) &
                &                         * ( p_nh%prog(nnew)%w(jc,jk,jb)     &
                &                         - p_nh%ref%w_ref(jc,jk,jb) )
            ENDDO
          ENDDO
          !$ACC END PARALLEL
        ENDIF

        ! Results for thermodynamic variables

    !$ser savepoint mo_solve_nonhydro_stencil_55_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cvd_o_rd=cvd_o_rd'

    !$ser data cvd_o_rd=cvd_o_rd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)'

    !$ser data exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    PRINT *, 'Serializing z_beta=z_beta'

    !$ser accdata z_beta=z_beta

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing exner_now=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser data exner_now=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing rho_new=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser accdata rho_new=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing exner_new=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner_new=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG VECTOR TILE(128, 1)
!$NEC outerloop_unroll(8)
        DO jk = jk_start, nlev
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx

            ! density
            p_nh%prog(nnew)%rho(jc,jk,jb) = z_rho_expl(jc,jk)              &
              - p_nh%metrics%vwind_impl_wgt(jc,jb)*dtime                   &
              * p_nh%metrics%inv_ddqz_z_full(jc,jk,jb)                     &
              *(p_nh%diag%rho_ic(jc,jk  ,jb)*p_nh%prog(nnew)%w(jc,jk  ,jb) &
              - p_nh%diag%rho_ic(jc,jk+1,jb)*p_nh%prog(nnew)%w(jc,jk+1,jb))

            ! exner
            p_nh%prog(nnew)%exner(jc,jk,jb) = z_exner_expl(jc,jk) &
              + p_nh%metrics%exner_ref_mc(jc,jk,jb)-z_beta(jc,jk) &
              *(z_alpha(jc,jk  )*p_nh%prog(nnew)%w(jc,jk  ,jb)    &
              - z_alpha(jc,jk+1)*p_nh%prog(nnew)%w(jc,jk+1,jb))

            ! theta
            p_nh%prog(nnew)%theta_v(jc,jk,jb) = p_nh%prog(nnow)%rho(jc,jk,jb)*p_nh%prog(nnow)%theta_v(jc,jk,jb) &
              *( (p_nh%prog(nnew)%exner(jc,jk,jb)/p_nh%prog(nnow)%exner(jc,jk,jb)-1.0_wp) * cvd_o_rd+1.0_wp   ) &
              / p_nh%prog(nnew)%rho(jc,jk,jb)

          ENDDO
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_55_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing cvd_o_rd=cvd_o_rd'

    !$ser data cvd_o_rd=cvd_o_rd

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing z_rho_expl=z_rho_expl(:,:)'

    !$ser accdata z_rho_expl=z_rho_expl(:,:)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing z_exner_expl=z_exner_expl(:,:)'

    !$ser accdata z_exner_expl=z_exner_expl(:,:)

    PRINT *, 'Serializing exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)'

    !$ser data exner_ref_mc=p_nh%metrics%exner_ref_mc(:,:,1)

    PRINT *, 'Serializing z_alpha=z_alpha(:,:)'

    !$ser accdata z_alpha=z_alpha(:,:)

    PRINT *, 'Serializing z_beta=z_beta'

    !$ser accdata z_beta=z_beta

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing exner_now=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser data exner_now=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing rho_new=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser accdata rho_new=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing exner_new=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner_new=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)

        ! Special treatment of uppermost layer in the case of vertical nesting
        IF (l_vert_nested) THEN
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx

            ! density
            p_nh%prog(nnew)%rho(jc,1,jb) = z_rho_expl(jc,1)                             &
              - p_nh%metrics%vwind_impl_wgt(jc,jb)*dtime                                &
              * p_nh%metrics%inv_ddqz_z_full(jc,1,jb)                                   &
              *(z_mflx_top(jc,jb) - p_nh%diag%rho_ic(jc,2,jb)*p_nh%prog(nnew)%w(jc,2,jb))

            ! exner
            p_nh%prog(nnew)%exner(jc,1,jb) = z_exner_expl(jc,1)                  &
              + p_nh%metrics%exner_ref_mc(jc,1,jb)-z_beta(jc,1)                  &
              *(p_nh%metrics%vwind_impl_wgt(jc,jb)*p_nh%diag%theta_v_ic(jc,1,jb) &
              * z_mflx_top(jc,jb) - z_alpha(jc,2)*p_nh%prog(nnew)%w(jc,2,jb))

            ! theta
            p_nh%prog(nnew)%theta_v(jc,1,jb) = p_nh%prog(nnow)%rho(jc,1,jb)*p_nh%prog(nnow)%theta_v(jc,1,jb) &
              *( (p_nh%prog(nnew)%exner(jc,1,jb)/p_nh%prog(nnow)%exner(jc,1,jb)-1.0_wp) * cvd_o_rd+1.0_wp  ) &
              /p_nh%prog(nnew)%rho(jc,1,jb)

          ENDDO
          !$ACC END PARALLEL
        ENDIF


        ! compute dw/dz for divergence damping term
        IF (lhdiff_rcf .AND. istep == 1 .AND. divdamp_type >= 3) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_56_63_8509f33c-e632-43a4-a676-38bb1a1dc21b_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing z_dwdz_dd=z_dwdz_dd(:,:,1)'

    !$ser accdata z_dwdz_dd=z_dwdz_dd(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR TILE(32, 4)
          DO jk = kstart_dd3d(jg), nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              z_dwdz_dd(jc,jk,jb) = p_nh%metrics%inv_ddqz_z_full(jc,jk,jb) *          &
                ( (p_nh%prog(nnew)%w(jc,jk,jb)-p_nh%prog(nnew)%w(jc,jk+1,jb)) -       &
                (p_nh%diag%w_concorr_c(jc,jk,jb)-p_nh%diag%w_concorr_c(jc,jk+1,jb)) )
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_56_63_8509f33c-e632-43a4-a676-38bb1a1dc21b_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing z_dwdz_dd=z_dwdz_dd(:,:,1)'

    !$ser accdata z_dwdz_dd=z_dwdz_dd(:,:,1)
        ENDIF

        ! Preparations for tracer advection
        IF (lprep_adv .AND. istep == 2) THEN
          IF (lclean_mflx) THEN 


    !$ser savepoint mo_solve_nonhydro_stencil_57_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR COLLAPSE(2)
            DO jk = 1, nlev
!$NEC ivdep
              DO jc = i_startidx, i_endidx
                prep_adv%mass_flx_ic(jc,jk,jb) = 0._wp
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_57_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)

          ENDIF


    !$ser savepoint mo_solve_nonhydro_stencil_58_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing r_nsubsteps=r_nsubsteps'

    !$ser data r_nsubsteps=r_nsubsteps

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = jk_start, nlev
!$NEC ivdep
            DO jc = i_startidx, i_endidx
              prep_adv%mass_flx_ic(jc,jk,jb) = prep_adv%mass_flx_ic(jc,jk,jb) + r_nsubsteps * ( z_contr_w_fl_l(jc,jk) + &
                p_nh%diag%rho_ic(jc,jk,jb) * p_nh%metrics%vwind_impl_wgt(jc,jb) * p_nh%prog(nnew)%w(jc,jk,jb) )
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_58_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing r_nsubsteps=r_nsubsteps'

    !$ser data r_nsubsteps=r_nsubsteps

    PRINT *, 'Serializing z_contr_w_fl_l=z_contr_w_fl_l(:,:)'

    !$ser accdata z_contr_w_fl_l=z_contr_w_fl_l(:,:)

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)

          IF (l_vert_nested) THEN
            ! Use mass flux which has been interpolated to the upper nest boundary.
            ! This mass flux is also seen by the mass continuity equation (rho).
            ! Hence, by using the same mass flux for the tracer mass continuity equations,
            ! consistency with continuity (CWC) is ensured.
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              prep_adv%mass_flx_ic(jc,1,jb) = prep_adv%mass_flx_ic(jc,1,jb) + &
                r_nsubsteps * z_mflx_top(jc,jb)
            ENDDO
            !$ACC END PARALLEL
          ENDIF
        ENDIF

        ! store dynamical part of exner time increment in exner_dyn_incr
        ! the conversion into a temperature tendency is done in the NWP interface
        IF (istep == 1 .AND. idyn_timestep == 1) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_59_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing exner=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)'

    !$ser accdata exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = kstart_moist(jg), nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              p_nh%diag%exner_dyn_incr(jc,jk,jb) = p_nh%prog(nnow)%exner(jc,jk,jb)
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_59_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing exner=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)'

    !$ser accdata exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)

        ELSE IF (istep == 2 .AND. idyn_timestep == ndyn_substeps_var(jg)) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_60_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing ndyn_substeps_var=real(ndyn_substeps_var(jg),wp)'

    !$ser data ndyn_substeps_var=real(ndyn_substeps_var(jg),wp)

    PRINT *, 'Serializing exner=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,1)'

    !$ser data ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,1)

    PRINT *, 'Serializing exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)'

    !$ser accdata exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = kstart_moist(jg), nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx
              p_nh%diag%exner_dyn_incr(jc,jk,jb) = p_nh%prog(nnew)%exner(jc,jk,jb) - &
               (p_nh%diag%exner_dyn_incr(jc,jk,jb) + ndyn_substeps_var(jg)*dtime*p_nh%diag%ddt_exner_phy(jc,jk,jb))
            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_60_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing ndyn_substeps_var=real(ndyn_substeps_var(jg),wp)'

    !$ser data ndyn_substeps_var=real(ndyn_substeps_var(jg),wp)

    PRINT *, 'Serializing exner=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,1)'

    !$ser data ddt_exner_phy=p_nh%diag%ddt_exner_phy(:,:,1)

    PRINT *, 'Serializing exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)'

    !$ser accdata exner_dyn_incr=p_nh%diag%exner_dyn_incr(:,:,1)

        ENDIF

        IF (istep == 2 .AND. l_child_vertnest) THEN
          ! Store values at nest interface levels
!DIR$ IVDEP
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR
          DO jc = i_startidx, i_endidx

            p_nh%diag%w_int(jc,jb,idyn_timestep) =  &
              0.5_wp*(p_nh%prog(nnow)%w(jc,nshift,jb) + p_nh%prog(nnew)%w(jc,nshift,jb))

            p_nh%diag%theta_v_ic_int(jc,jb,idyn_timestep) = p_nh%diag%theta_v_ic(jc,nshift,jb)

            p_nh%diag%rho_ic_int(jc,jb,idyn_timestep) =  p_nh%diag%rho_ic(jc,nshift,jb)

            p_nh%diag%mflx_ic_int(jc,jb,idyn_timestep) = p_nh%diag%rho_ic(jc,nshift,jb) * &
              (p_nh%metrics%vwind_expl_wgt(jc,jb)*p_nh%prog(nnow)%w(jc,nshift,jb) + &
              p_nh%metrics%vwind_impl_wgt(jc,jb)*p_nh%prog(nnew)%w(jc,nshift,jb))
          ENDDO
          !$ACC END PARALLEL
        ENDIF

      ENDDO
!$OMP END DO

      ! Boundary update in case of nesting
      IF (l_limited_area .OR. jg > 1) THEN

        rl_start = 1
        rl_end   = grf_bdywidth_c

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! non-MPI-parallelized (serial) case
          IF (istep == 1 .AND. my_process_is_mpi_all_seq() ) THEN

            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR COLLAPSE(2)
            DO jk = 1, nlev
#if __INTEL_COMPILER != 1400 || __INTEL_COMPILER_UPDATE != 3
!DIR$ IVDEP
#endif
              DO jc = i_startidx, i_endidx

                p_nh%prog(nnew)%rho(jc,jk,jb) = p_nh%prog(nnow)%rho(jc,jk,jb) + &
                  dtime*p_nh%diag%grf_tend_rho(jc,jk,jb)

                p_nh%prog(nnew)%theta_v(jc,jk,jb) = p_nh%prog(nnow)%theta_v(jc,jk,jb) + &
                  dtime*p_nh%diag%grf_tend_thv(jc,jk,jb)

                ! Diagnose exner from rho*theta
                p_nh%prog(nnew)%exner(jc,jk,jb) = EXP(rd_o_cvd*LOG(rd_o_p0ref* &
                  p_nh%prog(nnew)%rho(jc,jk,jb)*p_nh%prog(nnew)%theta_v(jc,jk,jb)))

                p_nh%prog(nnew)%w(jc,jk,jb) = p_nh%prog(nnow)%w(jc,jk,jb) + &
                  dtime*p_nh%diag%grf_tend_w(jc,jk,jb)

              ENDDO
            ENDDO
            !$ACC END PARALLEL

            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              p_nh%prog(nnew)%w(jc,nlevp1,jb) = p_nh%prog(nnow)%w(jc,nlevp1,jb) + &
                dtime*p_nh%diag%grf_tend_w(jc,nlevp1,jb)
            ENDDO
            !$ACC END PARALLEL

          ELSE IF (istep == 1 ) THEN

            ! In the MPI-parallelized case, only rho and w are updated here,
            ! and theta_v is preliminarily stored on exner in order to save
            ! halo communications



    !$ser savepoint mo_solve_nonhydro_stencil_61_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing grf_tend_rho=p_nh%diag%grf_tend_rho(:,:,1)'

    !$ser data grf_tend_rho=p_nh%diag%grf_tend_rho(:,:,1)

    PRINT *, 'Serializing theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing grf_tend_thv=p_nh%diag%grf_tend_thv(:,:,1)'

    !$ser data grf_tend_thv=p_nh%diag%grf_tend_thv(:,:,1)

    PRINT *, 'Serializing w_now=p_nh%prog(nnow)%w(:,:,1)'

    !$ser data w_now=p_nh%prog(nnow)%w(:,:,1)

    PRINT *, 'Serializing grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)'

    !$ser data grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)

    PRINT *, 'Serializing rho_new=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser accdata rho_new=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing exner_new=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner_new=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing w_new=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w_new=p_nh%prog(nnew)%w(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR COLLAPSE(2)
            DO jk = 1, nlev
#if __INTEL_COMPILER != 1400 || __INTEL_COMPILER_UPDATE != 3
!DIR$ IVDEP
#endif
              DO jc = i_startidx, i_endidx

                p_nh%prog(nnew)%rho(jc,jk,jb) = p_nh%prog(nnow)%rho(jc,jk,jb) + &
                  dtime*p_nh%diag%grf_tend_rho(jc,jk,jb)

                ! *** Storing theta_v on exner is done to save MPI communications ***
                ! DO NOT TOUCH THIS!
                p_nh%prog(nnew)%exner(jc,jk,jb) = p_nh%prog(nnow)%theta_v(jc,jk,jb) + &
                  dtime*p_nh%diag%grf_tend_thv(jc,jk,jb)

                p_nh%prog(nnew)%w(jc,jk,jb) = p_nh%prog(nnow)%w(jc,jk,jb) + &
                  dtime*p_nh%diag%grf_tend_w(jc,jk,jb)

              ENDDO
            ENDDO
            !$ACC END PARALLEL


    !$ser savepoint mo_solve_nonhydro_stencil_61_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing grf_tend_rho=p_nh%diag%grf_tend_rho(:,:,1)'

    !$ser data grf_tend_rho=p_nh%diag%grf_tend_rho(:,:,1)

    PRINT *, 'Serializing theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing grf_tend_thv=p_nh%diag%grf_tend_thv(:,:,1)'

    !$ser data grf_tend_thv=p_nh%diag%grf_tend_thv(:,:,1)

    PRINT *, 'Serializing w_now=p_nh%prog(nnow)%w(:,:,1)'

    !$ser data w_now=p_nh%prog(nnow)%w(:,:,1)

    PRINT *, 'Serializing grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)'

    !$ser data grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)

    PRINT *, 'Serializing rho_new=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser accdata rho_new=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing exner_new=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner_new=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing w_new=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w_new=p_nh%prog(nnew)%w(:,:,1)


    !$ser savepoint mo_solve_nonhydro_stencil_62_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing w_now=p_nh%prog(nnow)%w(:,:,1)'

    !$ser data w_now=p_nh%prog(nnow)%w(:,:,1)

    PRINT *, 'Serializing grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)'

    !$ser data grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)

    PRINT *, 'Serializing w_new=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w_new=p_nh%prog(nnew)%w(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR
            DO jc = i_startidx, i_endidx
              p_nh%prog(nnew)%w(jc,nlevp1,jb) = p_nh%prog(nnow)%w(jc,nlevp1,jb) + &
                dtime*p_nh%diag%grf_tend_w(jc,nlevp1,jb)
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_62_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing dtime=dtime'

    !$ser data dtime=dtime

    PRINT *, 'Serializing w_now=p_nh%prog(nnow)%w(:,:,1)'

    !$ser data w_now=p_nh%prog(nnow)%w(:,:,1)

    PRINT *, 'Serializing grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)'

    !$ser data grf_tend_w=p_nh%diag%grf_tend_w(:,:,1)

    PRINT *, 'Serializing w_new=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w_new=p_nh%prog(nnew)%w(:,:,1)

          ENDIF

          ! compute dw/dz for divergence damping term
          IF (lhdiff_rcf .AND. istep == 1 .AND. divdamp_type >= 3) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_56_63_e3bed1b9-20a7-47cc-82cf-51f7160635d2_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing z_dwdz_dd=z_dwdz_dd(:,:,1)'

    !$ser accdata z_dwdz_dd=z_dwdz_dd(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR TILE(32, 4)
            DO jk = kstart_dd3d(jg), nlev
!DIR$ IVDEP
              DO jc = i_startidx, i_endidx
                z_dwdz_dd(jc,jk,jb) = p_nh%metrics%inv_ddqz_z_full(jc,jk,jb) *          &
                  ( (p_nh%prog(nnew)%w(jc,jk,jb)-p_nh%prog(nnew)%w(jc,jk+1,jb)) -       &
                  (p_nh%diag%w_concorr_c(jc,jk,jb)-p_nh%diag%w_concorr_c(jc,jk+1,jb)) )
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_56_63_e3bed1b9-20a7-47cc-82cf-51f7160635d2_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)'

    !$ser data inv_ddqz_z_full=p_nh%metrics%inv_ddqz_z_full(:,:,1)

    PRINT *, 'Serializing w=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing z_dwdz_dd=z_dwdz_dd(:,:,1)'

    !$ser accdata z_dwdz_dd=z_dwdz_dd(:,:,1)

          ENDIF

          ! Preparations for tracer advection
          !
          ! Note that the vertical mass flux at nest boundary points is required in case that 
          ! vertical tracer transport precedes horizontal tracer transport.
          IF (lprep_adv .AND. istep == 2) THEN
            IF (lclean_mflx) THEN


    !$ser savepoint mo_solve_nonhydro_stencil_64_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)
              !$ACC KERNELS IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
              prep_adv%mass_flx_ic(i_startidx:i_endidx,:,jb) = 0._wp
              !$ACC END KERNELS

    !$ser savepoint mo_solve_nonhydro_stencil_64_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)
            ENDIF


    !$ser savepoint mo_solve_nonhydro_stencil_65_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing r_nsubsteps=r_nsubsteps'

    !$ser data r_nsubsteps=r_nsubsteps

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing w_now=p_nh%prog(nnow)%w(:,:,1)'

    !$ser data w_now=p_nh%prog(nnow)%w(:,:,1)

    PRINT *, 'Serializing w_new=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w_new=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)
            !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
            !$ACC LOOP GANG VECTOR COLLAPSE(2)
            DO jk = jk_start, nlev
!DIR$ IVDEP
!$NEC ivdep
              DO jc = i_startidx, i_endidx
                prep_adv%mass_flx_ic(jc,jk,jb) = prep_adv%mass_flx_ic(jc,jk,jb) + r_nsubsteps*p_nh%diag%rho_ic(jc,jk,jb)* &
                  (p_nh%metrics%vwind_expl_wgt(jc,jb)*p_nh%prog(nnow)%w(jc,jk,jb) +                                       &
                   p_nh%metrics%vwind_impl_wgt(jc,jb)*p_nh%prog(nnew)%w(jc,jk,jb) - p_nh%diag%w_concorr_c(jc,jk,jb) )
              ENDDO
            ENDDO
            !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_65_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing r_nsubsteps=r_nsubsteps'

    !$ser data r_nsubsteps=r_nsubsteps

    PRINT *, 'Serializing rho_ic=p_nh%diag%rho_ic(:,:,1)'

    !$ser accdata rho_ic=p_nh%diag%rho_ic(:,:,1)

    PRINT *, 'Serializing vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)'

    !$ser data vwind_expl_wgt=p_nh%metrics%vwind_expl_wgt(:,1)

    PRINT *, 'Serializing vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)'

    !$ser data vwind_impl_wgt=p_nh%metrics%vwind_impl_wgt(:,1)

    PRINT *, 'Serializing w_now=p_nh%prog(nnow)%w(:,:,1)'

    !$ser data w_now=p_nh%prog(nnow)%w(:,:,1)

    PRINT *, 'Serializing w_new=p_nh%prog(nnew)%w(:,:,1)'

    !$ser accdata w_new=p_nh%prog(nnew)%w(:,:,1)

    PRINT *, 'Serializing w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)'

    !$ser accdata w_concorr_c=p_nh%diag%w_concorr_c(:,:,1)

    PRINT *, 'Serializing mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)'

    !$ser accdata mass_flx_ic=prep_adv%mass_flx_ic(:,:,1)

            IF (l_vert_nested) THEN
              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
              !$ACC LOOP GANG VECTOR
              DO jc = i_startidx, i_endidx
                prep_adv%mass_flx_ic(jc,1,jb) = prep_adv%mass_flx_ic(jc,1,jb) + &
                  r_nsubsteps * (p_nh%diag%mflx_ic_ubc(jc,jb,1)                 &
                  + dt_linintp_ubc * p_nh%diag%mflx_ic_ubc(jc,jb,2))
              ENDDO
              !$ACC END PARALLEL
            ENDIF
          ENDIF

        ENDDO
!$OMP END DO

      ENDIF

!$OMP END PARALLEL


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
#ifdef __MIXED_PRECISION
            CALL sync_patch_array_mult_mp(SYNC_C,p_patch,1,1,p_nh%prog(nnew)%w,f3din1_sp=z_dwdz_dd, &
                 &                        opt_varname="w_nnew and z_dwdz_dd")
#else
            CALL sync_patch_array_mult(SYNC_C,p_patch,2,p_nh%prog(nnew)%w,z_dwdz_dd, &
                 &                     opt_varname="w_nnew and z_dwdz_dd")
#endif
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

! OpenMP directives are commented for the NEC because the overhead is too large
#if !defined( __SX__ ) 
!$OMP PARALLEL PRIVATE(rl_start,rl_end,i_startblk,i_endblk)
#endif
      IF (l_limited_area .OR. jg > 1) THEN

        ! Index list over halo points lying in the boundary interpolation zone
        ! Note: this list typically contains at most 10 grid points


    !$ser savepoint mo_solve_nonhydro_stencil_66_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rd_o_cvd=rd_o_cvd'

    !$ser data rd_o_cvd=rd_o_cvd

    PRINT *, 'Serializing rd_o_p0ref=rd_o_p0ref'

    !$ser data rd_o_p0ref=rd_o_p0ref

    PRINT *, 'Serializing bdy_halo_c=p_nh%metrics%mask_prog_halo_c_dsl_low_refin(:,1)'

    !$ser data bdy_halo_c=p_nh%metrics%mask_prog_halo_c_dsl_low_refin(:,1)

    PRINT *, 'Serializing rho=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnew)%theta_v(:,:,1)

    PRINT *, 'Serializing exner=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnew)%exner(:,:,1)
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
        !$ACC LOOP GANG
#ifndef __SX__
!$OMP DO PRIVATE(jb,ic,jk,jc) ICON_OMP_DEFAULT_SCHEDULE
#endif
        DO ic = 1, p_nh%metrics%bdy_halo_c_dim

          jb = p_nh%metrics%bdy_halo_c_blk(ic)
          jc = p_nh%metrics%bdy_halo_c_idx(ic)
!DIR$ IVDEP
          !$ACC LOOP VECTOR
          DO jk = 1, nlev
            p_nh%prog(nnew)%theta_v(jc,jk,jb) = p_nh%prog(nnew)%exner(jc,jk,jb)

            ! Diagnose exner from rho*theta
            p_nh%prog(nnew)%exner(jc,jk,jb) = EXP(rd_o_cvd*LOG(rd_o_p0ref* &
              p_nh%prog(nnew)%rho(jc,jk,jb)*p_nh%prog(nnew)%theta_v(jc,jk,jb)))

          ENDDO
        ENDDO
        !$ACC END PARALLEL

    rl_start = min_rlcell_int - 1
    rl_end   = min_rlcell

    CALL get_indices_c(p_patch, 1, 1, 1, &
                       i_startidx, i_endidx, rl_start, rl_end)


    !$ser savepoint mo_solve_nonhydro_stencil_66_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rd_o_cvd=rd_o_cvd'

    !$ser data rd_o_cvd=rd_o_cvd

    PRINT *, 'Serializing rd_o_p0ref=rd_o_p0ref'

    !$ser data rd_o_p0ref=rd_o_p0ref

    PRINT *, 'Serializing bdy_halo_c=p_nh%metrics%mask_prog_halo_c_dsl_low_refin(:,1)'

    !$ser data bdy_halo_c=p_nh%metrics%mask_prog_halo_c_dsl_low_refin(:,1)

    PRINT *, 'Serializing rho=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnew)%theta_v(:,:,1)

    PRINT *, 'Serializing exner=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnew)%exner(:,:,1)

#ifndef __SX__
!$OMP END DO
#endif

        rl_start = 1
        rl_end   = grf_bdywidth_c

        i_startblk = p_patch%cells%start_block(rl_start)
        i_endblk   = p_patch%cells%end_block(rl_end)

#ifndef __SX__
!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc) ICON_OMP_DEFAULT_SCHEDULE
#endif
        DO jb = i_startblk, i_endblk

          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)


    !$ser savepoint mo_solve_nonhydro_stencil_67_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rd_o_cvd=rd_o_cvd'

    !$ser data rd_o_cvd=rd_o_cvd

    PRINT *, 'Serializing rd_o_p0ref=rd_o_p0ref'

    !$ser data rd_o_p0ref=rd_o_p0ref

    PRINT *, 'Serializing rho=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnew)%theta_v(:,:,1)

    PRINT *, 'Serializing exner=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnew)%exner(:,:,1)
          !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)
          !$ACC LOOP GANG VECTOR COLLAPSE(2)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO jc = i_startidx, i_endidx

              p_nh%prog(nnew)%theta_v(jc,jk,jb) = p_nh%prog(nnew)%exner(jc,jk,jb)

              ! Diagnose exner from rhotheta
              p_nh%prog(nnew)%exner(jc,jk,jb) = EXP(rd_o_cvd*LOG(rd_o_p0ref* &
                p_nh%prog(nnew)%rho(jc,jk,jb)*p_nh%prog(nnew)%theta_v(jc,jk,jb)))

            ENDDO
          ENDDO
          !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_67_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing rd_o_cvd=rd_o_cvd'

    !$ser data rd_o_cvd=rd_o_cvd

    PRINT *, 'Serializing rd_o_p0ref=rd_o_p0ref'

    !$ser data rd_o_p0ref=rd_o_p0ref

    PRINT *, 'Serializing rho=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser data rho=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing theta_v=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v=p_nh%prog(nnew)%theta_v(:,:,1)

    PRINT *, 'Serializing exner=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner=p_nh%prog(nnew)%exner(:,:,1)
        ENDDO
#ifndef __SX__
!$OMP END DO
#endif
      ENDIF

      rl_start = min_rlcell_int - 1
      rl_end   = min_rlcell

      i_startblk = p_patch%cells%start_block(rl_start)
      i_endblk   = p_patch%cells%end_block(rl_end)

#ifndef __SX__
!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,jc) ICON_OMP_DEFAULT_SCHEDULE
#endif
      DO jb = i_startblk, i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                         i_startidx, i_endidx, rl_start, rl_end)


    !$ser savepoint mo_solve_nonhydro_stencil_68_start istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing mask_prog_halo_c=p_nh%metrics%mask_prog_halo_c(:,1)'

    !$ser data mask_prog_halo_c=p_nh%metrics%mask_prog_halo_c(:,1)

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing exner_new=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner_new=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing exner_now=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser data exner_now=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing rho_new=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser accdata rho_new=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)

    PRINT *, 'Serializing cvd_o_rd=cvd_o_rd'

    !$ser data cvd_o_rd=cvd_o_rd
        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(PRESENT) ASYNC(1)

#ifdef __LOOP_EXCHANGE
        !$ACC LOOP GANG
        DO jc = i_startidx, i_endidx
          IF (p_nh%metrics%mask_prog_halo_c(jc,jb)) THEN
!DIR$ IVDEP
            !$ACC LOOP VECTOR
            DO jk = 1, nlev
#else
        !$ACC LOOP GANG VECTOR TILE(32, 4)
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx
            IF (p_nh%metrics%mask_prog_halo_c(jc,jb)) THEN
#endif
              p_nh%prog(nnew)%theta_v(jc,jk,jb) = p_nh%prog(nnow)%rho(jc,jk,jb)*p_nh%prog(nnow)%theta_v(jc,jk,jb) &
                *( (p_nh%prog(nnew)%exner(jc,jk,jb)/p_nh%prog(nnow)%exner(jc,jk,jb)-1.0_wp) * cvd_o_rd+1.0_wp   ) &
                / p_nh%prog(nnew)%rho(jc,jk,jb)

#ifdef __LOOP_EXCHANGE
            ENDDO
          ENDIF
#else
            ENDIF
          ENDDO
#endif
        ENDDO
        !$ACC END PARALLEL

    !$ser savepoint mo_solve_nonhydro_stencil_68_end istep=istep mo_solve_nonhydro_ctr=mo_solve_nonhydro_ctr

    PRINT *, 'Serializing mask_prog_halo_c=p_nh%metrics%mask_prog_halo_c(:,1)'

    !$ser data mask_prog_halo_c=p_nh%metrics%mask_prog_halo_c(:,1)

    PRINT *, 'Serializing rho_now=p_nh%prog(nnow)%rho(:,:,1)'

    !$ser data rho_now=p_nh%prog(nnow)%rho(:,:,1)

    PRINT *, 'Serializing theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)'

    !$ser data theta_v_now=p_nh%prog(nnow)%theta_v(:,:,1)

    PRINT *, 'Serializing exner_new=p_nh%prog(nnew)%exner(:,:,1)'

    !$ser accdata exner_new=p_nh%prog(nnew)%exner(:,:,1)

    PRINT *, 'Serializing exner_now=p_nh%prog(nnow)%exner(:,:,1)'

    !$ser data exner_now=p_nh%prog(nnow)%exner(:,:,1)

    PRINT *, 'Serializing rho_new=p_nh%prog(nnew)%rho(:,:,1)'

    !$ser accdata rho_new=p_nh%prog(nnew)%rho(:,:,1)

    PRINT *, 'Serializing theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)'

    !$ser accdata theta_v_new=p_nh%prog(nnew)%theta_v(:,:,1)

    PRINT *, 'Serializing cvd_o_rd=cvd_o_rd'

    !$ser data cvd_o_rd=cvd_o_rd


      ENDDO
#ifndef __SX__
!$OMP END DO NOWAIT
!$OMP END PARALLEL
#endif

    ENDIF  ! .NOT. my_process_is_mpi_all_seq()

    IF (ltimer) CALL timer_stop(timer_solve_nh)
    CALL message('DSL', 'all dycore kernels ran')

    !$ACC WAIT
    !$ACC END DATA

#if !defined (__LOOP_EXCHANGE) && !defined (__SX__)
    CALL btraj%destruct()
#endif

  END SUBROUTINE solve_nh

#ifdef _OPENACC

     SUBROUTINE h2d_solve_nonhydro( nnow, jstep, jg, idiv_method, grf_intmethod_e, lprep_adv, l_vert_nested, is_iau_active, &
                                    p_nh, prep_adv )

       INTEGER, INTENT(IN)       :: nnow, jstep, jg, idiv_method, grf_intmethod_e
       LOGICAL, INTENT(IN)       :: l_vert_nested, lprep_adv, is_iau_active

       TYPE(t_nh_state),            INTENT(INOUT) :: p_nh
       TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv

       REAL(wp), DIMENSION(:,:,:),   POINTER  :: exner_tmp, rho_tmp, theta_v_tmp, vn_tmp, w_tmp                 ! p_prog  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: vn_ie_ubc_tmp                                                 ! p_diag  WP 2D
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: w_ubc_tmp, mflx_ic_ubc_tmp, theta_v_ic_ubc_tmp, rho_ic_ubc_tmp ! p_diag  WP

       REAL(wp), DIMENSION(:,:,:),   POINTER  :: theta_v_ic_tmp, rho_ic_tmp                                     ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: mass_fl_e_tmp, exner_pr_tmp                                    ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: grf_bdy_mflx_tmp                                               ! p_diag  WP

       REAL(vp), DIMENSION(:,:,:),   POINTER  :: vt_tmp, vn_ie_tmp, w_concorr_c_tmp, ddt_exner_phy_tmp          ! p_diag  VP
       REAL(vp), DIMENSION(:,:,:),   POINTER  :: exner_dyn_incr_tmp                                             ! p_diag  VP 
       REAL(vp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_phy_tmp                                                 ! p_diag  VP

       REAL(vp), DIMENSION(:,:,:),   POINTER  :: rho_incr_tmp, exner_incr_tmp                                   ! p_diag  VP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: vn_traj_tmp, mass_flx_me_tmp, mass_flx_ic_tmp                  ! prep_adv WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: vn_ref_tmp, w_ref_tmp                                          ! p_ref   WP

       REAL(vp), DIMENSION(:,:,:,:), POINTER  :: ddt_vn_apc_pc_tmp
       REAL(vp), DIMENSION(:,:,:,:), POINTER  :: ddt_vn_cor_pc_tmp
       REAL(vp), DIMENSION(:,:,:,:), POINTER  :: ddt_w_adv_pc_tmp

       REAL(wp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_dyn_tmp, ddt_vn_dmp_tmp, ddt_vn_adv_tmp, ddt_vn_cor_tmp ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_pgr_tmp, ddt_vn_phd_tmp, ddt_vn_iau_tmp, ddt_vn_ray_tmp ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_grf_tmp                                                 ! p_diag  WP

! p_patch:
!            p_patch%cells:   edge_idx/blk
!            p_patch%edges:   cell_idx/blk, vertex_idx/blk, quad_idx/blk, 
!                             primal/dual_normal_cell, inv_primal/dual_edge_length, tangent_orientation, refin_ctrl 

!
! p_nh%metrics:  vertidx_gradp, pg_vertidx, pg_edgeidx, pg_edgeblk,
!                bdy_halo_c_blk, bdy_halo_c_idx, bdy_mflx_e_blk, bdy_mflx_e_idx,
!                coeff_gradp, d_exner_dz_ref_ic, d2dexdz2_fac1_mc, 
!                ddqz_z_half, ddxn_z_full, ddxt_z_full, ddqz_z_full_e,
!                exner_exfac, exner_ref_mc, hmask_dd3d, inv_ddqz_z_full,
!                mask_prog_halo_c, nudge_e_blk, nudge_e_idx, pg_exdist,
!                rayleigh_vn, rayleigh_w, rho_ref_mc, rho_ref_me,
!                scalfac_dd3d, theta_ref_ic, theta_ref_mc, theta_ref_me,
!                vwind_expl_wgt, vwind_impl_wgt, 
!                wgtfac_c, wgtfac_e, wgtfacq_c, wgtfacq1_c, zdiff_gradp


! p_nh%prog(nnow)          All present (above)

       exner_tmp           => p_nh%prog(nnow)%exner 
       rho_tmp             => p_nh%prog(nnow)%rho
       theta_v_tmp         => p_nh%prog(nnow)%theta_v 
       vn_tmp              => p_nh%prog(nnow)%vn
       w_tmp               => p_nh%prog(nnow)%w
       !$ACC UPDATE DEVICE(exner_tmp, rho_tmp, theta_v_tmp, vn_tmp, w_tmp)

! p_nh%diag:

       rho_ic_tmp          => p_nh%diag%rho_ic
       theta_v_ic_tmp      => p_nh%diag%theta_v_ic
       !$ACC UPDATE DEVICE(rho_ic_tmp, theta_v_ic_tmp)

       vt_tmp              => p_nh%diag%vt
       vn_ie_tmp           => p_nh%diag%vn_ie
       w_concorr_c_tmp     => p_nh%diag%w_concorr_c
       !$ACC UPDATE DEVICE(vt_tmp, vn_ie_tmp, w_concorr_c_tmp)

       mass_fl_e_tmp       => p_nh%diag%mass_fl_e
       exner_pr_tmp        => p_nh%diag%exner_pr
       exner_dyn_incr_tmp  => p_nh%diag%exner_dyn_incr
       !$ACC UPDATE DEVICE(mass_fl_e_tmp, exner_pr_tmp, exner_dyn_incr_tmp)

! WS: I do not think these are necessary, but adding for completeness
       ddt_vn_apc_pc_tmp   => p_nh%diag%ddt_vn_apc_pc
       ddt_w_adv_pc_tmp    => p_nh%diag%ddt_w_adv_pc
       !$ACC UPDATE DEVICE(ddt_vn_apc_pc_tmp, ddt_w_adv_pc_tmp)
       IF (p_nh%diag%ddt_vn_adv_is_associated .OR. p_nh%diag%ddt_vn_cor_is_associated) THEN
          ddt_vn_cor_pc_tmp   => p_nh%diag%ddt_vn_cor_pc
          !$ACC UPDATE DEVICE(ddt_vn_cor_pc_tmp)
       END IF

! MAG: For completeness
       ddt_vn_dyn_tmp      => p_nh%diag%ddt_vn_dyn
       !$ACC UPDATE DEVICE(ddt_vn_dyn_tmp) IF(p_nh%diag%ddt_vn_dyn_is_associated)
       ddt_vn_dmp_tmp      => p_nh%diag%ddt_vn_dmp
       !$ACC UPDATE DEVICE(ddt_vn_dmp_tmp) IF(p_nh%diag%ddt_vn_dmp_is_associated)
       ddt_vn_adv_tmp      => p_nh%diag%ddt_vn_adv
       !$ACC UPDATE DEVICE(ddt_vn_adv_tmp) IF(p_nh%diag%ddt_vn_adv_is_associated)
       ddt_vn_cor_tmp      => p_nh%diag%ddt_vn_cor
       !$ACC UPDATE DEVICE(ddt_vn_cor_tmp) IF(p_nh%diag%ddt_vn_cor_is_associated)
       ddt_vn_pgr_tmp      => p_nh%diag%ddt_vn_pgr
       !$ACC UPDATE DEVICE(ddt_vn_pgr_tmp) IF(p_nh%diag%ddt_vn_pgr_is_associated)
       ddt_vn_phd_tmp      => p_nh%diag%ddt_vn_phd
       !$ACC UPDATE DEVICE(ddt_vn_phd_tmp) IF(p_nh%diag%ddt_vn_phd_is_associated)
       ddt_vn_iau_tmp      => p_nh%diag%ddt_vn_iau
       !$ACC UPDATE DEVICE(ddt_vn_iau_tmp) IF(p_nh%diag%ddt_vn_iau_is_associated)
       ddt_vn_ray_tmp      => p_nh%diag%ddt_vn_ray
       !$ACC UPDATE DEVICE(ddt_vn_ray_tmp) IF(p_nh%diag%ddt_vn_ray_is_associated)
       ddt_vn_grf_tmp      => p_nh%diag%ddt_vn_grf
       !$ACC UPDATE DEVICE(ddt_vn_grf_tmp) IF(p_nh%diag%ddt_vn_grf_is_associated)

       mflx_ic_ubc_tmp     => p_nh%diag%mflx_ic_ubc
       vn_ie_ubc_tmp       => p_nh%diag%vn_ie_ubc
       theta_v_ic_ubc_tmp  => p_nh%diag%theta_v_ic_ubc
       rho_ic_ubc_tmp      => p_nh%diag%rho_ic_ubc
       w_ubc_tmp           => p_nh%diag%w_ubc
       !$ACC UPDATE DEVICE(mflx_ic_ubc_tmp, vn_ie_ubc_tmp, theta_v_ic_ubc_tmp, rho_ic_ubc_tmp, w_ubc_tmp) IF(l_vert_nested)

       ddt_exner_phy_tmp   => p_nh%diag%ddt_exner_phy
       ddt_vn_phy_tmp      => p_nh%diag%ddt_vn_phy
       !$ACC UPDATE DEVICE(ddt_exner_phy_tmp, ddt_vn_phy_tmp)

       rho_incr_tmp        => p_nh%diag%rho_incr
       exner_incr_tmp      => p_nh%diag%exner_incr
       !$ACC UPDATE DEVICE(rho_incr_tmp, exner_incr_tmp)

       grf_bdy_mflx_tmp   => p_nh%diag%grf_bdy_mflx
       !$ACC UPDATE DEVICE(grf_bdy_mflx_tmp) IF((jg > 1) .AND. (grf_intmethod_e >= 5) .AND. (idiv_method == 1) .AND. (jstep == 0))

! prep_adv:

       vn_traj_tmp       => prep_adv%vn_traj
       mass_flx_me_tmp   => prep_adv%mass_flx_me
       mass_flx_ic_tmp   => prep_adv%mass_flx_ic
       !$ACC UPDATE DEVICE(vn_traj_tmp, mass_flx_me_tmp, mass_flx_ic_tmp) IF(lprep_adv)

! p_nh%ref:

       vn_ref_tmp          => p_nh%ref%vn_ref
       w_ref_tmp           => p_nh%ref%w_ref
       !$ACC UPDATE DEVICE(vn_ref_tmp, w_ref_tmp)

     END SUBROUTINE h2d_solve_nonhydro

     SUBROUTINE d2h_solve_nonhydro( nnew, jstep, jg, idyn_timestep, grf_intmethod_e, idiv_method, lsave_mflx, &
          &                         l_child_vertnest, lprep_adv, p_nh, prep_adv )

       INTEGER, INTENT(IN)       :: nnew, jstep, jg, idyn_timestep, grf_intmethod_e, idiv_method
       LOGICAL, INTENT(IN)       :: lsave_mflx, l_child_vertnest, lprep_adv

       TYPE(t_nh_state),            INTENT(INOUT) :: p_nh
       TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv

       REAL(wp), DIMENSION(:,:,:),   POINTER  :: exner_tmp, rho_tmp, theta_v_tmp, vn_tmp, w_tmp                 ! p_prog  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: vn_ie_int_tmp                                                  ! p_diag  WP 2D
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: theta_v_ic_tmp, rho_ic_tmp, rho_ic_int_tmp, w_int_tmp          ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: theta_v_ic_int_tmp, grf_bdy_mflx_tmp                           ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: mass_fl_e_tmp,  mflx_ic_int_tmp, exner_pr_tmp                  ! p_diag  WP

       REAL(vp), DIMENSION(:,:,:),   POINTER  :: vt_tmp, vn_ie_tmp, w_concorr_c_tmp                             ! p_diag  VP
       REAL(vp), DIMENSION(:,:,:),   POINTER  :: mass_fl_e_sv_tmp                                               ! p_diag  VP
       REAL(vp), DIMENSION(:,:,:),   POINTER  :: exner_dyn_incr_tmp                                             ! p_diag  VP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: vn_traj_tmp, mass_flx_me_tmp, mass_flx_ic_tmp                  ! prep_adv WP
       REAL(vp), DIMENSION(:,:,:,:), POINTER  :: ddt_vn_apc_pc_tmp, ddt_vn_cor_pc_tmp, ddt_w_adv_pc_tmp

       REAL(wp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_dyn_tmp, ddt_vn_dmp_tmp, ddt_vn_adv_tmp, ddt_vn_cor_tmp ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_pgr_tmp, ddt_vn_phd_tmp, ddt_vn_iau_tmp, ddt_vn_ray_tmp ! p_diag  WP
       REAL(wp), DIMENSION(:,:,:),   POINTER  :: ddt_vn_grf_tmp                                                 ! p_diag  WP

! The following code is necessary if the Dycore is to be run in isolation on the GPU
! Update all device output on host: the prognostic variables have shifted from nnow to nnew; diagnostics pointers set above

       exner_tmp           => p_nh%prog(nnew)%exner
       rho_tmp             => p_nh%prog(nnew)%rho
       theta_v_tmp         => p_nh%prog(nnew)%theta_v
       vn_tmp              => p_nh%prog(nnew)%vn
       w_tmp               => p_nh%prog(nnew)%w
       !$ACC UPDATE HOST(exner_tmp, rho_tmp, theta_v_tmp, vn_tmp, w_tmp)

       vt_tmp              => p_nh%diag%vt
       vn_ie_tmp           => p_nh%diag%vn_ie
       rho_ic_tmp          => p_nh%diag%rho_ic
       theta_v_ic_tmp      => p_nh%diag%theta_v_ic
       exner_pr_tmp        => p_nh%diag%exner_pr
       !$ACC UPDATE HOST(vt_tmp, vn_ie_tmp, rho_ic_tmp, theta_v_ic_tmp, exner_pr_tmp)

       w_concorr_c_tmp     => p_nh%diag%w_concorr_c
       mass_fl_e_tmp       => p_nh%diag%mass_fl_e
       exner_dyn_incr_tmp  => p_nh%diag%exner_dyn_incr
       !$ACC UPDATE HOST(w_concorr_c_tmp, mass_fl_e_tmp, exner_dyn_incr_tmp)

       ddt_vn_apc_pc_tmp   => p_nh%diag%ddt_vn_apc_pc
       ddt_w_adv_pc_tmp    => p_nh%diag%ddt_w_adv_pc
       !$ACC UPDATE HOST(ddt_vn_apc_pc_tmp, ddt_w_adv_pc_tmp)
       IF (p_nh%diag%ddt_vn_adv_is_associated .OR. p_nh%diag%ddt_vn_cor_is_associated) THEN
          ddt_vn_cor_pc_tmp   => p_nh%diag%ddt_vn_cor_pc
          !$ACC UPDATE HOST(ddt_vn_cor_pc_tmp)
       END IF

! MAG: For completeness
       ddt_vn_dyn_tmp      => p_nh%diag%ddt_vn_dyn
       !$ACC UPDATE HOST(ddt_vn_dyn_tmp) IF(p_nh%diag%ddt_vn_dyn_is_associated)
       ddt_vn_dmp_tmp      => p_nh%diag%ddt_vn_dmp
       !$ACC UPDATE HOST(ddt_vn_dmp_tmp) IF(p_nh%diag%ddt_vn_dmp_is_associated)
       ddt_vn_adv_tmp      => p_nh%diag%ddt_vn_adv
       !$ACC UPDATE HOST(ddt_vn_adv_tmp) IF(p_nh%diag%ddt_vn_adv_is_associated)
       ddt_vn_cor_tmp      => p_nh%diag%ddt_vn_cor
       !$ACC UPDATE HOST(ddt_vn_cor_tmp) IF(p_nh%diag%ddt_vn_cor_is_associated)
       ddt_vn_pgr_tmp      => p_nh%diag%ddt_vn_pgr
       !$ACC UPDATE HOST(ddt_vn_pgr_tmp) IF(p_nh%diag%ddt_vn_pgr_is_associated)
       ddt_vn_phd_tmp      => p_nh%diag%ddt_vn_phd
       !$ACC UPDATE HOST(ddt_vn_phd_tmp) IF(p_nh%diag%ddt_vn_phd_is_associated)
       ddt_vn_iau_tmp      => p_nh%diag%ddt_vn_iau
       !$ACC UPDATE HOST(ddt_vn_iau_tmp) IF(p_nh%diag%ddt_vn_iau_is_associated)
       ddt_vn_ray_tmp      => p_nh%diag%ddt_vn_ray
       !$ACC UPDATE HOST(ddt_vn_ray_tmp) IF(p_nh%diag%ddt_vn_ray_is_associated)
       ddt_vn_grf_tmp      => p_nh%diag%ddt_vn_grf
       !$ACC UPDATE HOST(ddt_vn_grf_tmp) IF(p_nh%diag%ddt_vn_grf_is_associated)

       mass_fl_e_sv_tmp    => p_nh%diag%mass_fl_e_sv
       !$ACC UPDATE HOST(mass_fl_e_sv_tmp) IF(lsave_mflx)

       w_int_tmp           => p_nh%diag%w_int
       mflx_ic_int_tmp     => p_nh%diag%mflx_ic_int
       theta_v_ic_int_tmp  => p_nh%diag%theta_v_ic_int
       rho_ic_int_tmp      => p_nh%diag%rho_ic_int
       !$ACC UPDATE HOST(w_int_tmp, mflx_ic_int_tmp, theta_v_ic_int_tmp, rho_ic_int_tmp) IF(l_child_vertnest)

      vn_ie_int_tmp      => p_nh%diag%vn_ie_int
      !$ACC UPDATE HOST(vn_ie_int_tmp) IF(idyn_timestep == 1 .AND. l_child_vertnest)

      grf_bdy_mflx_tmp    => p_nh%diag%grf_bdy_mflx
      !$ACC UPDATE HOST(grf_bdy_mflx_tmp) IF((jg > 1) .AND. (grf_intmethod_e >= 5) .AND. (idiv_method == 1) .AND. (jstep == 0))

      vn_traj_tmp         => prep_adv%vn_traj
      mass_flx_me_tmp     => prep_adv%mass_flx_me
      mass_flx_ic_tmp     => prep_adv%mass_flx_ic
      !$ACC UPDATE HOST(vn_traj_tmp, mass_flx_me_tmp, mass_flx_ic_tmp) IF(lprep_adv)

     END SUBROUTINE d2h_solve_nonhydro

#endif

END MODULE mo_solve_nonhydro
