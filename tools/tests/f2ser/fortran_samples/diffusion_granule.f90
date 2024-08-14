! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

!>
!! mo_nh_diffusion_new
!!
!! Diffusion in the nonhydrostatic model
!!
!! @author Almut Gassmann, MPI-M
!!
!!
!! @par Revision History
!! Initial release by Almut Gassmann, MPI-M (2009-08.25)
!! Modification by William Sawyer, CSCS (2015-02-06)
!! - OpenACC implementation
!! Modification by William Sawyer, CSCS (2015-02-06)
!! - Turned into a granule
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

MODULE mo_nh_diffusion_new

#ifdef __SX__
! for strange reasons, this routine is faster without mixed precision on the NEC
#undef __MIXED_PRECISION
  USE mo_kind_base,                 ONLY: wp, vp => wp
#else
  USE mo_kind_base,                 ONLY: wp, vp
#endif
  USE mo_math_types,           ONLY: t_tangent_vectors   ! to maintain compatibility w/ p_patch
#if 0
  USE mo_math_types_base,           ONLY: t_tangent_vectors
#endif
  USE mo_model_domain_advanced,     ONLY: t_patch ! until GridManager available
  USE mo_model_domain,              ONLY: p_patch
  USE mo_intp_rbf_math,             ONLY: rbf_vec_interpol_vertex, rbf_vec_interpol_cell
  USE mo_interpolation_scalar_math, ONLY: cells2verts_scalar
  USE mo_interpolation_vector_math, ONLY: edges2cells_vector
  USE mo_loopindices_advanced,      ONLY: get_indices_e, get_indices_c
  USE mo_impl_constants_base    ,   ONLY: min_rledge, min_rlcell, min_rlvert, min_rledge_int, min_rlcell_int, min_rlvert_int
  USE mo_impl_constants_grf_base,   ONLY: grf_bdywidth_e, grf_bdywidth_c
  USE mo_math_types_base,           ONLY: t_geographical_coordinates
  USE mo_math_laplace_math,         ONLY: nabla4_vec
  USE mo_math_constants_base,       ONLY: dbl_eps
  USE mo_sync,                ONLY: SYNC_E, SYNC_C, SYNC_V, sync_patch_array, &
                                    sync_patch_array_mult, sync_patch_array_mult_mp
  USE mo_timer,               ONLY: timer_nh_hdiffusion, timer_start, timer_stop
  USE mo_exception_advanced,        ONLY: finish, message, message_text

#ifdef _OPENACC
  USE mo_mpi_advanced,              ONLY: i_am_accel_node
#endif

  IMPLICIT NONE

  PUBLIC :: t_diffusion, diffusion_alloc, diffusion_dealloc, diffusion_init, diffusion_run, diffusion_finalize
  PRIVATE

  TYPE :: t_diffusion
    LOGICAL  :: lphys             ! Is a run with physics?
    LOGICAL  :: ltimer            ! Is the timer on?
    LOGICAL  :: l_limited_area    ! Is a limited area run?
    LOGICAL  :: ltkeshs
    LOGICAL  :: lfeedback
    LOGICAL  :: l_zdiffu_t
    LOGICAL  :: lsmag_3d
    LOGICAL  :: lhdiff_rcf
    LOGICAL  :: lhdiff_w
    LOGICAL  :: lhdiff_temp
    LOGICAL  :: lvert_nest
    LOGICAL  :: p_test_run
    LOGICAL  :: ddt_vn_hdf_is_associated, ddt_vn_dyn_is_associated
    INTEGER  :: nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, nrdmax, ndyn_substeps
    INTEGER  :: nshift, nshift_total

    INTEGER  :: hdiff_order
    INTEGER  :: discr_vn, discr_t
    INTEGER  :: itype_sher
    INTEGER  :: itype_comm

    REAL(wp) :: grav
    REAL(vp) :: cvd_o_rd
    REAL(wp) :: nudge_max_coeff
    REAL(wp) :: denom_diffu_v
    REAL(wp) :: k4, k4w
    REAL(wp) :: hdiff_smag_z, hdiff_smag_z2, hdiff_smag_z3, hdiff_smag_z4
    REAL(wp) :: hdiff_smag_fac, hdiff_smag_fac2, hdiff_smag_fac3, hdiff_smag_fac4
    REAL(wp) :: hdiff_efdt_ratio

    REAL(wp), POINTER     :: vct_a(:)                ! vertical coordinate part A

    REAL(wp), POINTER     :: c_lin_e(:,:,:)          ! p_int
    REAL(wp), POINTER     :: e_bln_c_s(:,:,:)        !   :
    REAL(wp), POINTER     :: e_bln_c_u(:,:,:)        !   :
    REAL(wp), POINTER     :: e_bln_c_v(:,:,:)        !   :
    REAL(wp), POINTER     :: cells_aw_verts(:,:,:)
    REAL(wp), POINTER     :: geofac_div(:,:,:)
    REAL(wp), POINTER     :: geofac_rot(:,:,:)
    REAL(wp), POINTER     :: geofac_n2s(:,:,:)
    REAL(wp), POINTER     :: geofac_grg(:,:,:,:)
    REAL(wp), POINTER     :: nudgecoeff_e(:,:)
    INTEGER,  POINTER     :: rbf_vec_idx_v(:,:,:)
    INTEGER,  POINTER     :: rbf_vec_blk_v(:,:,:)
    REAL(wp), POINTER     :: rbf_vec_coeff_v(:,:,:,:)

    REAL(wp), POINTER     :: enhfac_diffu(:)         ! p_nh_metrics
    REAL(wp), POINTER     :: zd_intcoef(:,:)         !    :
    REAL(wp), POINTER     :: zd_geofac(:,:)          !    :
    REAL(wp), POINTER     :: zd_diffcoef(:)          !    :
    REAL(vp), POINTER     :: ddqz_z_full_e(:,:,:)
    REAL(vp), POINTER     :: theta_ref_mc(:,:,:)
    REAL(vp), POINTER     :: wgtfac_c(:,:,:)
    REAL(vp), POINTER     :: wgtfac_e(:,:,:)
    REAL(vp), POINTER     :: wgtfacq_e(:,:,:)
    REAL(vp), POINTER     :: wgtfacq1_e(:,:,:)
    INTEGER,  POINTER     :: zd_indlist(:,:)
    INTEGER,  POINTER     :: zd_blklist(:,:)
    INTEGER,  POINTER     :: zd_vertidx(:,:)
    INTEGER               :: zd_listdim

  END TYPE t_diffusion

  TYPE (t_diffusion), ALLOCATABLE :: diff_inst(:)

  TYPE (t_patch),     ALLOCATABLE :: p_patch_diff(:)

#ifndef __SX__
#define __ENABLE_DDT_VN_XYZ__
#endif

  CONTAINS

    SUBROUTINE  diffusion_alloc( n_dom )
      INTEGER,  INTENT(IN)            :: n_dom
      ALLOCATE( diff_inst( n_dom ) )
      ALLOCATE( p_patch_diff( n_dom ) )
      !$ACC ENTER DATA CREATE(diff_inst,p_patch_diff)
    END SUBROUTINE diffusion_alloc

    SUBROUTINE  diffusion_dealloc( )
      !$ACC EXIT DATA DELETE(diff_inst,p_patch_diff)
      IF (ALLOCATED(diff_inst)) DEALLOCATE( diff_inst )
      IF (ALLOCATED(p_patch_diff)) DEALLOCATE( p_patch_diff )
    END SUBROUTINE diffusion_dealloc


  !>
  !! init_diffusion
  !!
  !! Prepares the horizontal diffusion of velocity and temperature
  !!
  !! @par Revision History
  !! Initial release by William Sawyer, CSCS (2022-11-25)
  !!
    SUBROUTINE  diffusion_init(cvd_o_rd, grav,                                                     &
                               jg, nproma, nlev, nblks_e, nblks_v, nblks_c, nshift, nshift_total,  &
                               nrdmax, ndyn_substeps, nudge_max_coeff, denom_diffu_v,              &
                               hdiff_smag_z, hdiff_smag_z2, hdiff_smag_z3, hdiff_smag_z4,          &
                               hdiff_smag_fac, hdiff_smag_fac2, hdiff_smag_fac3, hdiff_smag_fac4,  &
                               hdiff_order, hdiff_efdt_ratio,                                      &
                               k4, k4w, itype_comm, itype_sher, itype_vn_diffu, itype_t_diffu,     &
                               p_test_run, lphys, lhdiff_rcf, lhdiff_w, lhdiff_temp, l_limited_area,&
                               lfeedback, l_zdiffu_t, ltkeshs, lsmag_3d, lvert_nest, ltimer,       &
                               ddt_vn_hdf_is_associated, ddt_vn_dyn_is_associated,                 &
                               vct_a, c_lin_e, e_bln_c_s, e_bln_c_u, e_bln_c_v, cells_aw_verts,    &   ! p_int
                               geofac_div, geofac_rot, geofac_n2s, geofac_grg, nudgecoeff_e,       &   ! p_int
                               rbf_vec_idx_v, rbf_vec_blk_v, rbf_vec_coeff_v,                      &   ! p_int
                               enhfac_diffu, zd_intcoef, zd_geofac, zd_diffcoef,                   &   ! p_nh_metrics
                               wgtfac_c, wgtfac_e, wgtfacq_e, wgtfacq1_e,                          &   ! p_nh_metrics
                               ddqz_z_full_e, theta_ref_mc,                                        &   ! p_nh_metrics
                               zd_indlist, zd_blklist, zd_vertidx, zd_listdim,                     &   ! p_nh_metrics
                               edges_start_block, edges_end_block, edges_start_index, edges_end_index,&! p_patch%edges
                               edges_vertex_idx, edges_vertex_blk, edges_cell_idx, edges_cell_blk, &   ! p_patch%edges
                               edges_tangent_orientation,                                          &   ! p_patch%edges
                               edges_primal_normal_vert, edges_dual_normal_vert,                   &   ! p_patch%edges
                               edges_primal_normal_cell, edges_dual_normal_cell,                   &   ! p_patch%edges
                               edges_inv_vert_vert_length, edges_inv_primal_edge_length,           &   ! p_patch%edges
                               edges_inv_dual_edge_length, edges_area_edge,                        &   ! p_patch%edges
                               cells_start_block, cells_end_block, cells_start_index, cells_end_index,&! p_patch%cells
                               cells_neighbor_idx, cells_neighbor_blk,                             &   ! p_patch%cells
                               cells_edge_idx, cells_edge_blk, cells_area,                         &   ! p_patch%cells
                               verts_start_block, verts_end_block, verts_start_index, verts_end_index )! p_patch%verts
    REAL(wp), INTENT(IN)            :: cvd_o_rd, grav    ! Physical constants from central location
    INTEGER,  INTENT(IN)            :: jg
    INTEGER,  INTENT(IN)            :: nproma, nlev, nblks_e, nblks_v, nblks_c, nshift, nshift_total
    INTEGER,  INTENT(IN)            :: nrdmax                 ! = nrdmax(jg)
    INTEGER,  INTENT(IN)            :: ndyn_substeps
    INTEGER,  INTENT(IN)            :: hdiff_order, itype_comm, itype_sher, itype_vn_diffu, itype_t_diffu
    REAL(wp), INTENT(IN)            :: hdiff_smag_z, hdiff_smag_z2, hdiff_smag_z3, hdiff_smag_z4
    REAL(wp), INTENT(IN)            :: hdiff_smag_fac, hdiff_smag_fac2, hdiff_smag_fac3, hdiff_smag_fac4
    REAL(wp), INTENT(IN)            :: hdiff_efdt_ratio
    REAL(wp), INTENT(IN)            :: k4, k4w
    REAL(wp), INTENT(IN)            :: nudge_max_coeff
    REAL(wp), INTENT(IN)            :: denom_diffu_v
    LOGICAL,  INTENT(IN)            :: p_test_run
    LOGICAL,  INTENT(IN)            :: lphys      !< is a run with physics
    LOGICAL,  INTENT(IN)            :: lhdiff_rcf
    LOGICAL,  INTENT(IN)            :: lhdiff_w
    LOGICAL,  INTENT(IN)            :: lhdiff_temp
    LOGICAL,  INTENT(IN)            :: l_zdiffu_t
    LOGICAL,  INTENT(IN)            :: l_limited_area
    LOGICAL,  INTENT(IN)            :: lfeedback               ! = lfeedback(jg)
    LOGICAL,  INTENT(IN)            :: ltkeshs
    LOGICAL,  INTENT(IN)            :: lsmag_3d
    LOGICAL,  INTENT(IN)            :: lvert_nest
    LOGICAL,  INTENT(IN)            :: ltimer
    LOGICAL,  INTENT(IN)            :: ddt_vn_hdf_is_associated
    LOGICAL,  INTENT(IN)            :: ddt_vn_dyn_is_associated

    REAL(wp), TARGET, INTENT(IN)    :: vct_a(:)                ! param. A of the vertical coordinte

    REAL(wp), TARGET, INTENT(IN)    :: c_lin_e(:,:,:)          ! p_int
    REAL(wp), TARGET, INTENT(IN)    :: e_bln_c_s(:,:,:)        !   :
    REAL(wp), TARGET, INTENT(IN)    :: e_bln_c_u(:,:,:)        !   :
    REAL(wp), TARGET, INTENT(IN)    :: e_bln_c_v(:,:,:)        !   :
    REAL(wp), TARGET, INTENT(IN)    :: cells_aw_verts(:,:,:)   !   :
    REAL(wp), TARGET, INTENT(IN)    :: geofac_div(:,:,:)       !   :
    REAL(wp), TARGET, INTENT(IN)    :: geofac_rot(:,:,:)       !   :
    REAL(wp), TARGET, INTENT(IN)    :: geofac_n2s(:,:,:)       !   :
    REAL(wp), TARGET, INTENT(IN)    :: geofac_grg(:,:,:,:)     !   :
    REAL(wp), TARGET, INTENT(IN)    :: nudgecoeff_e(:,:)       !   :
    INTEGER,  TARGET, INTENT(IN)    :: rbf_vec_idx_v(:,:,:)
    INTEGER,  TARGET, INTENT(IN)    :: rbf_vec_blk_v(:,:,:)
    REAL(wp), TARGET, INTENT(IN)    :: rbf_vec_coeff_v(:,:,:,:)

    REAL(wp), TARGET, INTENT(IN)    :: enhfac_diffu(:)         ! p_nh_metrics
    REAL(wp), TARGET, INTENT(IN)    :: zd_intcoef(:,:)         !    :
    REAL(wp), TARGET, INTENT(IN)    :: zd_geofac(:,:)          !    :
    REAL(wp), TARGET, INTENT(IN)    :: zd_diffcoef(:)          !    :
    REAL(vp), TARGET, INTENT(IN)    :: wgtfac_c(:,:,:)         !    :
    REAL(vp), TARGET, INTENT(IN)    :: wgtfac_e(:,:,:)         !    :
    REAL(vp), TARGET, INTENT(IN)    :: wgtfacq_e(:,:,:)        !    :
    REAL(vp), TARGET, INTENT(IN)    :: wgtfacq1_e(:,:,:)       !    :
    REAL(vp), TARGET, INTENT(IN)    :: ddqz_z_full_e(:,:,:)    !    :
    REAL(vp), TARGET, INTENT(IN)    :: theta_ref_mc(:,:,:)     !    :
    INTEGER,  TARGET, INTENT(IN)    :: zd_indlist(:,:)         !    :
    INTEGER,  TARGET, INTENT(IN)    :: zd_blklist(:,:)         !    :
    INTEGER,  TARGET, INTENT(IN)    :: zd_vertidx(:,:)         !    :
    INTEGER,  INTENT(IN)            :: zd_listdim              !    :

    INTEGER,  TARGET, INTENT(IN)    :: edges_start_block(min_rledge:)    ! p_patch%edges
    INTEGER,  TARGET, INTENT(IN)    :: edges_end_block(min_rledge:)      !       :
    INTEGER,  TARGET, INTENT(IN)    :: edges_start_index(min_rledge:)    ! p_patch%edges
    INTEGER,  TARGET, INTENT(IN)    :: edges_end_index(min_rledge:)      !       :
    INTEGER,  TARGET, INTENT(IN)    :: edges_vertex_idx(:,:,:) !       :
    INTEGER,  TARGET, INTENT(IN)    :: edges_vertex_blk(:,:,:) !       :
    INTEGER,  TARGET, INTENT(IN)    :: edges_cell_idx(:,:,:)   !       :
    INTEGER,  TARGET, INTENT(IN)    :: edges_cell_blk(:,:,:)   !       :
    REAL(wp), TARGET, INTENT(IN)    :: edges_tangent_orientation(:,:)
    TYPE(t_tangent_vectors), TARGET, INTENT(IN)    :: edges_primal_normal_vert(:,:,:)
    TYPE(t_tangent_vectors), TARGET, INTENT(IN)    :: edges_dual_normal_vert(:,:,:)
    TYPE(t_tangent_vectors), TARGET, INTENT(IN)    :: edges_primal_normal_cell(:,:,:)
    TYPE(t_tangent_vectors), TARGET, INTENT(IN)    :: edges_dual_normal_cell(:,:,:)
    REAL(wp), TARGET, INTENT(IN)    :: edges_inv_vert_vert_length(:,:)
    REAL(wp), TARGET, INTENT(IN)    :: edges_inv_primal_edge_length(:,:)
    REAL(wp), TARGET, INTENT(IN)    :: edges_inv_dual_edge_length(:,:)
    REAL(wp), TARGET, INTENT(IN)    :: edges_area_edge(:,:)

    INTEGER,  TARGET, INTENT(IN)    :: cells_start_block(min_rlcell:)      ! p_patch%cells
    INTEGER,  TARGET, INTENT(IN)    :: cells_end_block(min_rlcell:)        !       :
    INTEGER,  TARGET, INTENT(IN)    :: cells_start_index(min_rlcell:)      ! p_patch%cells
    INTEGER,  TARGET, INTENT(IN)    :: cells_end_index(min_rlcell:)        !       :
    INTEGER,  TARGET, INTENT(IN)    :: cells_neighbor_idx(:,:,:) !       :
    INTEGER,  TARGET, INTENT(IN)    :: cells_neighbor_blk(:,:,:) !       :
    INTEGER,  TARGET, INTENT(IN)    :: cells_edge_idx(:,:,:)     !       :
    INTEGER,  TARGET, INTENT(IN)    :: cells_edge_blk(:,:,:)     !       :
    REAL(wp), TARGET, INTENT(IN)    :: cells_area(:,:)

    INTEGER,  TARGET, INTENT(IN)    :: verts_start_block(min_rlvert:)    ! p_patch%verts
    INTEGER,  TARGET, INTENT(IN)    :: verts_end_block(min_rlvert:)      !       :
    INTEGER,  TARGET, INTENT(IN)    :: verts_start_index(min_rlvert:)    ! p_patch%verts
    INTEGER,  TARGET, INTENT(IN)    :: verts_end_index(min_rlvert:)      !       :
    !--------------------------------------------------------------------------

    diff_inst(jg)%vct_a              => vct_a

    diff_inst(jg)%c_lin_e            => c_lin_e         ! p_int
    diff_inst(jg)%e_bln_c_s          => e_bln_c_s       !   :
    diff_inst(jg)%e_bln_c_u          => e_bln_c_u       !   :
    diff_inst(jg)%e_bln_c_v          => e_bln_c_v       !   :
    diff_inst(jg)%cells_aw_verts     => cells_aw_verts
    diff_inst(jg)%geofac_div         => geofac_div
    diff_inst(jg)%geofac_rot         => geofac_rot
    diff_inst(jg)%geofac_n2s         => geofac_n2s
    diff_inst(jg)%geofac_grg         => geofac_grg
    diff_inst(jg)%nudgecoeff_e       => nudgecoeff_e
    diff_inst(jg)%rbf_vec_idx_v      => rbf_vec_idx_v
    diff_inst(jg)%rbf_vec_blk_v      => rbf_vec_blk_v
    diff_inst(jg)%rbf_vec_coeff_v    => rbf_vec_coeff_v

    diff_inst(jg)%enhfac_diffu       => enhfac_diffu    ! p_nh_metrics
    diff_inst(jg)%zd_intcoef         => zd_intcoef      !      :
    diff_inst(jg)%zd_geofac          => zd_geofac
    diff_inst(jg)%zd_diffcoef        => zd_diffcoef
    diff_inst(jg)%wgtfac_c           => wgtfac_c
    diff_inst(jg)%wgtfac_e           => wgtfac_e
    diff_inst(jg)%wgtfacq_e          => wgtfacq_e
    diff_inst(jg)%wgtfacq1_e         => wgtfacq1_e
    diff_inst(jg)%ddqz_z_full_e      => ddqz_z_full_e
    diff_inst(jg)%theta_ref_mc       => theta_ref_mc
    diff_inst(jg)%zd_indlist         => zd_indlist
    diff_inst(jg)%zd_blklist         => zd_blklist
    diff_inst(jg)%zd_vertidx         => zd_vertidx
    diff_inst(jg)%zd_listdim         =  zd_listdim

    p_patch_diff(jg)%edges%start_block            => edges_start_block            ! p_patch%edges
    p_patch_diff(jg)%edges%end_block              => edges_end_block              !       :
    p_patch_diff(jg)%edges%start_index            => edges_start_index            ! p_patch%edges
    p_patch_diff(jg)%edges%end_index              => edges_end_index              !       :
    p_patch_diff(jg)%edges%vertex_idx             => edges_vertex_idx
    p_patch_diff(jg)%edges%vertex_blk             => edges_vertex_blk
    p_patch_diff(jg)%edges%cell_idx               => edges_cell_idx
    p_patch_diff(jg)%edges%cell_blk               => edges_cell_blk
    p_patch_diff(jg)%edges%tangent_orientation    => edges_tangent_orientation
    p_patch_diff(jg)%edges%primal_normal_vert     => edges_primal_normal_vert
    p_patch_diff(jg)%edges%dual_normal_vert       => edges_dual_normal_vert
    p_patch_diff(jg)%edges%primal_normal_cell     => edges_primal_normal_cell
    p_patch_diff(jg)%edges%dual_normal_cell       => edges_dual_normal_cell
    p_patch_diff(jg)%edges%inv_vert_vert_length   => edges_inv_vert_vert_length
    p_patch_diff(jg)%edges%inv_primal_edge_length => edges_inv_primal_edge_length
    p_patch_diff(jg)%edges%inv_dual_edge_length   => edges_inv_dual_edge_length
    p_patch_diff(jg)%edges%area_edge              => edges_area_edge

    p_patch_diff(jg)%cells%start_block            => cells_start_block            ! p_patch%cells
    p_patch_diff(jg)%cells%end_block              => cells_end_block              !       :
    p_patch_diff(jg)%cells%start_index            => cells_start_index            ! p_patch%cells
    p_patch_diff(jg)%cells%end_index              => cells_end_index              !       :
    p_patch_diff(jg)%cells%neighbor_idx           => cells_neighbor_idx
    p_patch_diff(jg)%cells%neighbor_blk           => cells_neighbor_blk
    p_patch_diff(jg)%cells%edge_idx               => cells_edge_idx
    p_patch_diff(jg)%cells%edge_blk               => cells_edge_blk
    p_patch_diff(jg)%cells%area                   => cells_area

    p_patch_diff(jg)%verts%start_block            => verts_start_block            ! p_patch%cells
    p_patch_diff(jg)%verts%end_block              => verts_end_block              !       :
    p_patch_diff(jg)%verts%start_index            => verts_start_index            ! p_patch%cells
    p_patch_diff(jg)%verts%end_index              => verts_end_index              !       :

    diff_inst(jg)%nrdmax         = nrdmax
    diff_inst(jg)%grav           = grav        ! from central location
    diff_inst(jg)%cvd_o_rd       = cvd_o_rd    !   "
    diff_inst(jg)%nudge_max_coeff= nudge_max_coeff
    diff_inst(jg)%denom_diffu_v  = denom_diffu_v
    diff_inst(jg)%k4             = k4
    diff_inst(jg)%k4w            = k4w

    ! number of vertical levels, blocks for edges, vertices and cells
    diff_inst(jg)%nproma         = nproma
    diff_inst(jg)%nlev           = nlev
    diff_inst(jg)%nlevp1         = nlev+1
    diff_inst(jg)%nblks_e        = nblks_e
    diff_inst(jg)%nblks_v        = nblks_v
    diff_inst(jg)%nblks_c        = nblks_c

    diff_inst(jg)%ndyn_substeps  = ndyn_substeps
    diff_inst(jg)%nshift         = nshift           ! p_patch%nshift
    diff_inst(jg)%nshift_total   = nshift_total     ! p_patch%nshift_total

    diff_inst(jg)%itype_sher     = itype_sher
    diff_inst(jg)%itype_comm     = itype_comm

    diff_inst(jg)%p_test_run     = p_test_run
    diff_inst(jg)%lphys          = lphys
    diff_inst(jg)%l_zdiffu_t     = l_zdiffu_t
    diff_inst(jg)%l_limited_area = l_limited_area
    diff_inst(jg)%lfeedback      = lfeedback
    diff_inst(jg)%ltkeshs        = ltkeshs
    diff_inst(jg)%lsmag_3d       = lsmag_3d
    diff_inst(jg)%lhdiff_w       = lhdiff_w
    diff_inst(jg)%lhdiff_rcf     = lhdiff_rcf
    diff_inst(jg)%lhdiff_temp    = lhdiff_temp
    diff_inst(jg)%ltimer         = ltimer
    diff_inst(jg)%lvert_nest     = lvert_nest
    diff_inst(jg)%ddt_vn_hdf_is_associated  = ddt_vn_hdf_is_associated
    diff_inst(jg)%ddt_vn_dyn_is_associated  = ddt_vn_dyn_is_associated

    diff_inst(jg)%hdiff_order    = hdiff_order

    diff_inst(jg)%discr_vn       = itype_vn_diffu
    diff_inst(jg)%discr_t        = itype_t_diffu

    diff_inst(jg)%hdiff_smag_z   = hdiff_smag_z
    diff_inst(jg)%hdiff_smag_z2  = hdiff_smag_z2
    diff_inst(jg)%hdiff_smag_z3  = hdiff_smag_z3
    diff_inst(jg)%hdiff_smag_z4  = hdiff_smag_z4
    diff_inst(jg)%hdiff_smag_fac = hdiff_smag_fac
    diff_inst(jg)%hdiff_smag_fac2= hdiff_smag_fac2
    diff_inst(jg)%hdiff_smag_fac3= hdiff_smag_fac3
    diff_inst(jg)%hdiff_smag_fac4= hdiff_smag_fac4
    diff_inst(jg)%hdiff_efdt_ratio = hdiff_efdt_ratio

    !$ACC ENTER DATA COPYIN(diff_inst(jg))

  END SUBROUTINE diffusion_init

  !>
  !! diffusion
  !!
  !! Computes the horizontal diffusion of velocity and temperature
  !!
  !! @par Revision History
  !! Initial release by Guenther Zaengl, DWD (2010-10-13), based on an earlier
  !! version initially developed by Almut Gassmann, MPI-M
  !!
  SUBROUTINE  diffusion_run(jg, dtime, linit,                              &
                            vn, w, theta_v, exner,                         & ! p_nh_prog
                            vt, theta_v_ic, div_ic, hdef_ic, dwdx, dwdy,   & ! p_nh_diag
                            ddt_vn_dyn, ddt_vn_hdf )                         ! p_nh_diag optional

    INTEGER,  INTENT(IN)     :: jg                ! patch ID
    REAL(wp), INTENT(IN)     :: dtime             !< time step
    LOGICAL,  INTENT(IN)     :: linit             !< initial call or runtime call
    REAL(wp), INTENT(INOUT)  :: vn(:,:,:)         ! orthogonal normal wind (nproma,nlev,nblks_e) [m/s]
    REAL(wp), INTENT(INOUT)  :: w(:,:,:)          ! orthogonal vertical wind (nproma,nlevp1,nblks_c) [m/s]
    REAL(wp), INTENT(INOUT)  :: theta_v(:,:,:)    ! virtual potential temperature (nproma,nlev,nblks_c) [K]
    REAL(wp), INTENT(INOUT)  :: exner(:,:,:)      ! Exner pressure (nproma,nlev,nblks_c) [-]
    REAL(wp), INTENT(INOUT)  :: theta_v_ic(:,:,:) ! theta_v at half levels (nproma,nlevp1,nblks_c) [K]
    REAL(vp), INTENT(IN)     :: vt(:,:,:)         ! tangential wind (nproma,nlev,nblks_e) [m/s]
    REAL(vp), INTENT(OUT)    :: div_ic(:,:,:)     ! divergence at half levels(nproma,nlevp1,nblks_c) [1/s]
    REAL(vp), INTENT(OUT)    :: hdef_ic(:,:,:)    ! horizontal wind field deformation (nproma,nlevp1,nblks_c) [1/s^2]
    REAL(vp), INTENT(OUT)    :: dwdx(:,:,:)       ! divergence at half levels(nproma,nlevp1,nblks_c) [1/s]
    REAL(vp), INTENT(OUT)    :: dwdy(:,:,:)       ! horizontal wind field deformation (nproma,nlevp1,nblks_c) [1/s^2]
    REAL(wp), INTENT(INOUT), OPTIONAL  :: ddt_vn_dyn(:,:,:) ! d vn / dt (sum of all contributions)
    REAL(wp), INTENT(INOUT), OPTIONAL  :: ddt_vn_hdf(:,:,:) ! d vn / dt (horizontal diffusion only)

    ! local variables - vp means variable precision depending on the __MIXED_PRECISION cpp flag
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_c) :: z_temp
    REAL(wp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_e) :: z_nabla2_e
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_c) :: z_nabla2_c
    REAL(wp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_e) :: z_nabla4_e
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev) :: z_nabla4_e2

    REAL(wp):: diff_multfac_vn(diff_inst(jg)%nlev), diff_multfac_w, diff_multfac_n2w(diff_inst(jg)%nlev)
    INTEGER :: i_startblk, i_endblk, i_startidx, i_endidx
    INTEGER :: rl_start, rl_end
    INTEGER :: jk, jb, jc, je, ic, ishift, nshift, jk1
    INTEGER :: nlev, nlevp1              !< number of full and half levels

    ! start index levels and diffusion coefficient for boundary diffusion
    INTEGER, PARAMETER  :: start_bdydiff_e = 5 ! refin_ctrl level at which boundary diffusion starts
    REAL(wp):: fac_bdydiff_v

    ! For Smagorinsky diffusion - vp means variable precision depending on the __MIXED_PRECISION cpp flag
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_e) :: kh_smag_e
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_e) :: kh_smag_ec
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_v) :: u_vert
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_v) :: v_vert
    REAL(wp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_c) :: u_cell
    REAL(wp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev,diff_inst(jg)%nblks_c) :: v_cell
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev) :: kh_c, div

    REAL(vp) :: dvt_norm, dvt_tang, vn_vert1, vn_vert2, vn_vert3, vn_vert4, vn_cell1, vn_cell2

    REAL(vp) :: smag_offset, nabv_tang, nabv_norm, rd_o_cvd, nudgezone_diff, bdy_diff, enh_diffu
    REAL(vp), DIMENSION(diff_inst(jg)%nlev) :: smag_limit, diff_multfac_smag, enh_smag_fac
    INTEGER  :: nblks_zdiffu, nproma_zdiffu, npromz_zdiffu, nlen_zdiffu

    REAL(wp) :: alin, dz32, df32, dz42, df42, bqdr, aqdr, zf, dzlin, dzqdr

    ! Additional variables for 3D Smagorinsky coefficient
    REAL(wp):: z_w_v(diff_inst(jg)%nproma,diff_inst(jg)%nlevp1,diff_inst(jg)%nblks_v)
    REAL(wp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlevp1) :: z_vn_ie, z_vt_ie
    REAL(wp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev) :: dvndz, dvtdz, dwdz, dthvdz, dwdn, dwdt, kh_smag3d_e

    ! Variables for provisional fix against runaway cooling in local topography depressions
    INTEGER  :: icount(diff_inst(jg)%nblks_c), iclist(2*diff_inst(jg)%nproma,diff_inst(jg)%nblks_c), iklist(2*diff_inst(jg)%nproma,diff_inst(jg)%nblks_c)
    REAL(wp) :: tdlist(2*diff_inst(jg)%nproma,diff_inst(jg)%nblks_c), tdiff, trefdiff, thresh_tdiff, z_theta, fac2d


    INTEGER,  DIMENSION(:,:,:), POINTER :: icidx, icblk, ieidx, ieblk, ividx, ivblk, iecidx, iecblk
    INTEGER,  DIMENSION(:,:),   POINTER :: icell, ilev, iblk !, iedge, iedblk
    REAL(wp), DIMENSION(:,:),   POINTER :: vcoef, zd_geofac !, blcoef
    LOGICAL :: ltemp_diffu
    INTEGER :: diffu_type

#ifdef _OPENACC
    REAL(vp), DIMENSION(diff_inst(jg)%nproma,diff_inst(jg)%nlev-1:diff_inst(jg)%nlev,diff_inst(jg)%nblks_c) :: enh_diffu_3d
#endif

    ! Variables for tendency diagnostics
    REAL(wp) :: z_d_vn_hdf
    REAL(wp) :: r_dtimensubsteps

    CHARACTER(*), PARAMETER :: routine = "diffusion_run"

    !--------------------------------------------------------------------------

    ividx => p_patch_diff(jg)%edges%vertex_idx
    ivblk => p_patch_diff(jg)%edges%vertex_blk

    iecidx => p_patch_diff(jg)%edges%cell_idx
    iecblk => p_patch_diff(jg)%edges%cell_blk

    icidx => p_patch_diff(jg)%cells%neighbor_idx
    icblk => p_patch_diff(jg)%cells%neighbor_blk

    ieidx => p_patch_diff(jg)%cells%edge_idx
    ieblk => p_patch_diff(jg)%cells%edge_blk

    ! prepare for tendency diagnostics
    IF (diff_inst(jg)%lhdiff_rcf) THEN
      r_dtimensubsteps = 1._wp/dtime                          ! without substepping, no averaging is necessary
    ELSE
      r_dtimensubsteps = 1._wp/(dtime*REAL(diff_inst(jg)%ndyn_substeps,wp)) ! with substepping the tendency is averaged over the substeps
    END IF

    ! number of vertical levels
    nlev   = diff_inst(jg)%nlev
    nlevp1 = nlev+1

        ! Normalized diffusion coefficient for boundary diffusion
    IF (diff_inst(jg)%lhdiff_rcf) THEN
      fac_bdydiff_v = SQRT(REAL(diff_inst(jg)%ndyn_substeps,wp))/diff_inst(jg)%denom_diffu_v
    ELSE
      fac_bdydiff_v = 1._wp/diff_inst(jg)%denom_diffu_v
    ENDIF

    ! scaling factor for enhanced diffusion in nudging zone (if present, i.e. for
    ! limited-area runs and one-way nesting)
    nudgezone_diff = 0.04_wp/(diff_inst(jg)%nudge_max_coeff + dbl_eps)

    ! scaling factor for enhanced near-boundary diffusion for
    ! two-way nesting (used with Smagorinsky diffusion only; not needed otherwise)
    bdy_diff = 0.015_wp/(diff_inst(jg)%nudge_max_coeff + dbl_eps)

    ! threshold temperature deviation from neighboring grid points
    ! that activates extra diffusion against runaway cooling
    thresh_tdiff = - 5._wp

    rd_o_cvd    = 1._wp/diff_inst(jg)%cvd_o_rd
    diffu_type  = diff_inst(jg)%hdiff_order


    IF (linit) THEN ! enhanced diffusion at all levels for initial velocity filtering call
      diff_multfac_vn(:) = diff_inst(jg)%k4/3._wp*diff_inst(jg)%hdiff_efdt_ratio
      smag_offset        =  0.0_vp
      diffu_type = 5 ! always combine nabla4 background diffusion with Smagorinsky diffusion for initial filtering call
      smag_limit(:) = 0.125_wp-4._wp*diff_multfac_vn(:)
    ELSE IF (diff_inst(jg)%lhdiff_rcf) THEN ! combination with divergence damping inside the dynamical core
      IF (diffu_type == 4) THEN
        diff_multfac_vn(:) = MIN(1._wp/128._wp,diff_inst(jg)%k4*REAL(diff_inst(jg)%ndyn_substeps,wp)/ &
                                 3._wp*diff_inst(jg)%enhfac_diffu(:))
      ELSE ! For Smagorinsky diffusion, the Smagorinsky coefficient rather than the background
           ! diffusion coefficient is enhanced near the model top (see below)
        diff_multfac_vn(:) = MIN(1._wp/128._wp,diff_inst(jg)%k4*REAL(diff_inst(jg)%ndyn_substeps,wp)/3._wp)
      ENDIF
      IF (diffu_type == 3) THEN
        smag_offset   = 0._vp
        smag_limit(:) = 0.125_vp
      ELSE
        smag_offset   = 0.25_wp*diff_inst(jg)%k4*REAL(diff_inst(jg)%ndyn_substeps,wp)
        smag_limit(:) = 0.125_wp-4._wp*diff_multfac_vn(:)
      ENDIF
    ELSE           ! enhanced diffusion near model top only
      IF (diffu_type == 4) THEN
        diff_multfac_vn(:) = diff_inst(jg)%k4/3._wp*diff_inst(jg)%enhfac_diffu(:)
      ELSE ! For Smagorinsky diffusion, the Smagorinsky coefficient rather than the background
           ! diffusion coefficient is enhanced near the model top (see below)
        diff_multfac_vn(:) = diff_inst(jg)%k4/3._wp
      ENDIF
      smag_offset        = 0.25_wp*diff_inst(jg)%k4
      smag_limit(:)      = 0.125_wp-4._wp*diff_multfac_vn(:)
      ! pure Smagorinsky diffusion does not work without divergence damping
      IF (diff_inst(jg)%hdiff_order == 3) diffu_type = 5
    ENDIF

    ! Multiplication factor for nabla4 diffusion on vertical wind speed
    diff_multfac_w = MIN(1._wp/48._wp,diff_inst(jg)%k4w*REAL(diff_inst(jg)%ndyn_substeps,wp))

    ! Factor for additional nabla2 diffusion in upper damping zone
    diff_multfac_n2w(:) = 0._wp
    IF (diff_inst(jg)%nrdmax > 1) THEN ! seems to be redundant, but the NEC issues invalid operations otherwise
      DO jk = 2, diff_inst(jg)%nrdmax
        jk1 = jk + diff_inst(jg)%nshift_total
        diff_multfac_n2w(jk) = 1._wp/12._wp*((diff_inst(jg)%vct_a(jk1)-diff_inst(jg)%vct_a(diff_inst(jg)%nshift_total+diff_inst(jg)%nrdmax+1))/ &
                               (diff_inst(jg)%vct_a(2)-diff_inst(jg)%vct_a(diff_inst(jg)%nshift_total+diff_inst(jg)%nrdmax+1)))**4
      ENDDO
    ENDIF

    IF (diffu_type == 3 .OR. diffu_type == 5) THEN

      ! temperature diffusion is used only in combination with Smagorinsky diffusion
      ltemp_diffu = diff_inst(jg)%lhdiff_temp

      ! The Smagorinsky diffusion factor enh_divdamp_fac is defined as a profile in height z
      ! above sea level with 4 height sections:
      !
      ! enh_smag_fac(z) = hdiff_smag_fac                                                    !                  z <= hdiff_smag_z
      ! enh_smag_fac(z) = hdiff_smag_fac  + (z-hdiff_smag_z )* alin                         ! hdiff_smag_z  <= z <= hdiff_smag_z2
      ! enh_smag_fac(z) = hdiff_smag_fac2 + (z-hdiff_smag_z2)*(aqdr+(z-hdiff_smag_z2)*bqdr) ! hdiff_smag_z2 <= z <= hdiff_smag_z4
      ! enh_smag_fac(z) = hdiff_smag_fac4                                                   ! hdiff_smag_z4 <= z
      !
      alin = (diff_inst(jg)%hdiff_smag_fac2-diff_inst(jg)%hdiff_smag_fac)/ &
           & (diff_inst(jg)%hdiff_smag_z2  -diff_inst(jg)%hdiff_smag_z)
      !
      df32 = diff_inst(jg)%hdiff_smag_fac3-diff_inst(jg)%hdiff_smag_fac2
      df42 = diff_inst(jg)%hdiff_smag_fac4-diff_inst(jg)%hdiff_smag_fac2
      !
      dz32 = diff_inst(jg)%hdiff_smag_z3-diff_inst(jg)%hdiff_smag_z2
      dz42 = diff_inst(jg)%hdiff_smag_z4-diff_inst(jg)%hdiff_smag_z2
      !
      bqdr = (df42*dz32-df32*dz42)/(dz32*dz42*(dz42-dz32))
      aqdr =  df32/dz32-bqdr*dz32
      !
      DO jk = 1, nlev
        jk1 = jk + diff_inst(jg)%nshift_total
        !
        zf = 0.5_wp*(diff_inst(jg)%vct_a(jk1)+diff_inst(jg)%vct_a(jk1+1))
        dzlin = MIN( diff_inst(jg)%hdiff_smag_z2-diff_inst(jg)%hdiff_smag_z , &
             &  MAX( 0._wp,                          zf-diff_inst(jg)%hdiff_smag_z ) )
        dzqdr = MIN( diff_inst(jg)%hdiff_smag_z4-diff_inst(jg)%hdiff_smag_z2, &
             &  MAX( 0._wp,                          zf-diff_inst(jg)%hdiff_smag_z2) )
        !
        enh_smag_fac(jk) = REAL(diff_inst(jg)%hdiff_smag_fac + dzlin*alin + dzqdr*(aqdr+dzqdr*bqdr),vp)
        !
      ENDDO

      ! Smagorinsky coefficient is also enhanced in the six model levels beneath a vertical nest interface
      IF (diff_inst(jg)%lvert_nest .AND. (diff_inst(jg)%nshift > 0)) THEN
        enh_smag_fac(1) = MAX(0.333_vp, enh_smag_fac(1))
        enh_smag_fac(2) = MAX(0.25_vp, enh_smag_fac(2))
        enh_smag_fac(3) = MAX(0.20_vp, enh_smag_fac(3))
        enh_smag_fac(4) = MAX(0.16_vp, enh_smag_fac(4))
        enh_smag_fac(5) = MAX(0.12_vp, enh_smag_fac(5))
        enh_smag_fac(6) = MAX(0.08_vp, enh_smag_fac(6))
      ENDIF

      ! empirically determined scaling factor
      diff_multfac_smag(:) = enh_smag_fac(:)*REAL(dtime,vp)

    ELSE
      ltemp_diffu = .FALSE.
    ENDIF

    !$ACC DATA CREATE(div, kh_c, kh_smag_e, kh_smag_ec, u_vert, v_vert, u_cell, v_cell, z_w_v, z_temp) &
    !$ACC   CREATE(z_nabla4_e, z_nabla4_e2, z_nabla2_e, z_nabla2_c, enh_diffu_3d, icount) &
    !$ACC   CREATE(z_vn_ie, z_vt_ie, dvndz, dvtdz, dwdz, dthvdz, dwdn, dwdt, kh_smag3d_e) &
    !$ACC   COPYIN(diff_multfac_vn, diff_multfac_n2w, diff_multfac_smag, smag_limit) &
    !$ACC   PRESENT(diff_inst, p_patch_diff) &
    !$ACC   PRESENT(ividx, ivblk, iecidx, iecblk, icidx, icblk, ieidx, ieblk) &
    !$ACC   IF(i_am_accel_node)

    !!! Following variables may be present in certain situations, but we don't want it to fail in the general case.
    !!! Should actually be in a separate data region with correct IF condition.
    !!! !$ACC               div_ic, dwdx, dwdy, hdef_ic,                     &

    ! The diffusion is an intrinsic part of the NH solver, thus it is added to the timer
    IF (diff_inst(jg)%ltimer) CALL timer_start(timer_nh_hdiffusion)

    IF (diffu_type == 4) THEN

      CALL nabla4_vec( vn, jg, p_patch_diff(jg), diff_inst(jg)%geofac_div, diff_inst(jg)%geofac_rot, &
                       z_nabla4_e, opt_rlstart=7,opt_nabla2=z_nabla2_e )
    ELSE IF ((diffu_type == 3 .OR. diffu_type == 5) &
             .AND. diff_inst(jg)%discr_vn == 1 .AND. .NOT. diff_inst(jg)%lsmag_3d) THEN

      IF (diff_inst(jg)%p_test_run) THEN
        !$ACC KERNELS PRESENT(u_vert, v_vert) ASYNC(1) IF(i_am_accel_node)
        u_vert = 0._vp
        v_vert = 0._vp
        !$ACC END KERNELS
      ENDIF

      !  RBF reconstruction of velocity at vertices
      CALL rbf_vec_interpol_vertex( vn, p_patch_diff(jg), &
                                    diff_inst(jg)%rbf_vec_idx_v, diff_inst(jg)%rbf_vec_blk_v, &
                                    diff_inst(jg)%rbf_vec_coeff_v, u_vert, v_vert,            &
                                    opt_rlend=min_rlvert_int, opt_acc_async=.TRUE. )
      rl_start = start_bdydiff_e
      rl_end   = min_rledge_int - 2

      IF (diff_inst(jg)%itype_comm == 1 .OR. diff_inst(jg)%itype_comm == 3) THEN
#ifdef __MIXED_PRECISION
        CALL sync_patch_array_mult_mp(SYNC_V,p_patch(jg),0,2,f3din1_sp=u_vert,f3din2_sp=v_vert, &
                                      opt_varname="diffusion: u_vert and v_vert")
#else
        CALL sync_patch_array_mult(SYNC_V,p_patch(jg),2,u_vert,v_vert,                          &
                                   opt_varname="diffusion: u_vert and v_vert")
#endif
      ENDIF

!$OMP PARALLEL PRIVATE(i_startblk,i_endblk)

      i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,vn_vert1,vn_vert2,vn_vert3,vn_vert4, &
!$OMP            dvt_norm,dvt_tang), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        ! Computation of wind field deformation

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO je = i_startidx, i_endidx
!DIR$ IVDEP
          DO jk = 1, nlev
#else
!$NEC outerloop_unroll(4)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
#endif

            vn_vert1 =  u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v1 + &
                        v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v2

            vn_vert2 =  u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v1 + &
                        v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v2

            dvt_tang =  p_patch_diff(jg)%edges%tangent_orientation(je,jb)* (   &
                        u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,2)%v1 + &
                        v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,2)%v2 - &
                        (u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,1)%v1 + &
                        v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,1)%v2) )

            vn_vert3 =  u_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,3)%v1 + &
                        v_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,3)%v2

            vn_vert4 =  u_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,4)%v1 + &
                        v_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,4)%v2

            dvt_norm =  u_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,4)%v1 + &
                        v_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,4)%v2 - &
                        (u_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,3)%v1 + &
                        v_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,3)%v2)
            ! Smagorinsky diffusion coefficient
            kh_smag_e(je,jk,jb) = diff_multfac_smag(jk)*SQRT(                           (  &
              (vn_vert4-vn_vert3)*p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb)- &
              dvt_tang*p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) )**2 + (         &
              (vn_vert2-vn_vert1)*p_patch_diff(jg)%edges%tangent_orientation(je,jb)*   &
              p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) +                                &
              dvt_norm*p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb))**2 )

            ! The factor of 4 comes from dividing by twice the "correct" length
            z_nabla2_e(je,jk,jb) = 4._wp * (                                      &
              (vn_vert4 + vn_vert3 - 2._wp*vn(je,jk,jb))  &
              *p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb)**2 +                     &
              (vn_vert2 + vn_vert1 - 2._wp*vn(je,jk,jb))  &
              *p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb)**2 )

#if defined (__LOOP_EXCHANGE) && !defined (_OPENACC)
          ENDDO
        ENDDO

        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
#endif
            kh_smag_ec(je,jk,jb) = kh_smag_e(je,jk,jb)
            ! Subtract part of the fourth-order background diffusion coefficient
            kh_smag_e(je,jk,jb) = MAX(0._vp,kh_smag_e(je,jk,jb) - smag_offset)
            ! Limit diffusion coefficient to the theoretical CFL stability threshold
            kh_smag_e(je,jk,jb) = MIN(kh_smag_e(je,jk,jb),smag_limit(jk))
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
      ENDDO ! block jb
!$OMP END DO NOWAIT
!$OMP END PARALLEL

    ELSE IF ((diffu_type == 3 .OR. diffu_type == 5) .AND. diff_inst(jg)%discr_vn == 1) THEN
      ! 3D Smagorinsky diffusion
      IF (diff_inst(jg)%p_test_run) THEN
        !$ACC KERNELS PRESENT(u_vert, v_vert, z_w_v) ASYNC(1) IF(i_am_accel_node)
        u_vert = 0._vp
        v_vert = 0._vp
        z_w_v  = 0._wp
        !$ACC END KERNELS
      ENDIF

      !  RBF reconstruction of velocity at vertices
      CALL rbf_vec_interpol_vertex( vn, p_patch_diff(jg),                                              &
                                    diff_inst(jg)%rbf_vec_idx_v, diff_inst(jg)%rbf_vec_blk_v, &
                                    diff_inst(jg)%rbf_vec_coeff_v, u_vert, v_vert,            &
                                    opt_rlend=min_rlvert_int, opt_acc_async=.TRUE. )

      rl_start = start_bdydiff_e
      rl_end   = min_rledge_int - 2

      IF (diff_inst(jg)%itype_comm == 1 .OR. diff_inst(jg)%itype_comm == 3) THEN
#ifdef __MIXED_PRECISION
        CALL sync_patch_array_mult_mp(SYNC_V,p_patch(jg),0,2,f3din1_sp=u_vert,f3din2_sp=v_vert, &
                                      opt_varname="diffusion: u_vert and v_vert 2")
#else
        CALL sync_patch_array_mult(SYNC_V,p_patch(jg),2,u_vert,v_vert,                          &
                                   opt_varname="diffusion: u_vert and v_vert 2")
#endif
      ENDIF
      CALL cells2verts_scalar(w, p_patch_diff(jg), diff_inst(jg)%cells_aw_verts, z_w_v, opt_rlend=min_rlvert_int)
      CALL sync_patch_array(SYNC_V,p_patch(jg),z_w_v,opt_varname="diffusion: z_w_v")
      CALL sync_patch_array(SYNC_C,p_patch(jg),theta_v_ic,opt_varname="diffusion: theta_v_ic")

      fac2d = 0.0625_wp ! Factor of the 2D deformation field which is used as minimum of the 3D def field

!$OMP PARALLEL PRIVATE(i_startblk,i_endblk)

      i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,vn_vert1,vn_vert2,vn_vert3,vn_vert4,dvt_norm,dvt_tang, &
!$OMP            z_vn_ie,z_vt_ie,dvndz,dvtdz,dwdz,dthvdz,dwdn,dwdt,kh_smag3d_e), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)


        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 2, nlev
          DO je = i_startidx, i_endidx
            z_vn_ie(je,jk) = diff_inst(jg)%wgtfac_e(je,jk,jb)*vn(je,jk,jb) +   &
             (1._wp - diff_inst(jg)%wgtfac_e(je,jk,jb))*vn(je,jk-1,jb)
            z_vt_ie(je,jk) = diff_inst(jg)%wgtfac_e(je,jk,jb)*vt(je,jk,jb) +   &
             (1._wp - diff_inst(jg)%wgtfac_e(je,jk,jb))*vt(je,jk-1,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) IF(i_am_accel_node)
        DO je = i_startidx, i_endidx
          z_vn_ie(je,1) =                                            &
            diff_inst(jg)%wgtfacq1_e(je,1,jb)*vn(je,1,jb) + &
            diff_inst(jg)%wgtfacq1_e(je,2,jb)*vn(je,2,jb) + &
            diff_inst(jg)%wgtfacq1_e(je,3,jb)*vn(je,3,jb)
          z_vn_ie(je,nlevp1) =                                           &
            diff_inst(jg)%wgtfacq_e(je,1,jb)*vn(je,nlev,jb)   + &
            diff_inst(jg)%wgtfacq_e(je,2,jb)*vn(je,nlev-1,jb) + &
            diff_inst(jg)%wgtfacq_e(je,3,jb)*vn(je,nlev-2,jb)
          z_vt_ie(je,1) =                                            &
            diff_inst(jg)%wgtfacq1_e(je,1,jb)*vt(je,1,jb) + &
            diff_inst(jg)%wgtfacq1_e(je,2,jb)*vt(je,2,jb) + &
            diff_inst(jg)%wgtfacq1_e(je,3,jb)*vt(je,3,jb)
          z_vt_ie(je,nlevp1) =                                           &
            diff_inst(jg)%wgtfacq_e(je,1,jb)*vt(je,nlev,jb)   + &
            diff_inst(jg)%wgtfacq_e(je,2,jb)*vt(je,nlev-1,jb) + &
            diff_inst(jg)%wgtfacq_e(je,3,jb)*vt(je,nlev-2,jb)
        ENDDO
        !$ACC END PARALLEL LOOP

        ! Computation of wind field deformation

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO je = i_startidx, i_endidx
!DIR$ IVDEP
          DO jk = 1, nlev
#else
!$NEC outerloop_unroll(4)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
#endif

            vn_vert1 =  u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v1 + &
                        v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v2

            vn_vert2 =  u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v1 + &
                        v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v2

            dvt_tang =  p_patch_diff(jg)%edges%tangent_orientation(je,jb)* (   &
                        u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,2)%v1 + &
                        v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,2)%v2 - &
                        (u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,1)%v1 + &
                        v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,1)%v2) )

            vn_vert3 =  u_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,3)%v1 + &
                        v_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,3)%v2

            vn_vert4 =  u_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,4)%v1 + &
                        v_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,4)%v2

            dvt_norm =  u_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,4)%v1 + &
                        v_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,4)%v2 - &
                        (u_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,3)%v1 + &
                        v_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%dual_normal_vert(je,jb,3)%v2)

            dvndz(je,jk) = (z_vn_ie(je,jk) - z_vn_ie(je,jk+1)) / diff_inst(jg)%ddqz_z_full_e(je,jk,jb)
            dvtdz(je,jk) = (z_vt_ie(je,jk) - z_vt_ie(je,jk+1)) / diff_inst(jg)%ddqz_z_full_e(je,jk,jb)

            dwdz (je,jk) =                                                                     &
              (diff_inst(jg)%c_lin_e(je,1,jb)*(w(iecidx(je,jb,1),jk,  iecblk(je,jb,1)) -     &
                                       w(iecidx(je,jb,1),jk+1,iecblk(je,jb,1)) ) +   &
               diff_inst(jg)%c_lin_e(je,2,jb)*(w(iecidx(je,jb,2),jk,  iecblk(je,jb,2)) -     &
                                       w(iecidx(je,jb,2),jk+1,iecblk(je,jb,2)) ) ) / &
               diff_inst(jg)%ddqz_z_full_e(je,jk,jb)

            dthvdz(je,jk) =                                                                             &
              (diff_inst(jg)%c_lin_e(je,1,jb)*(theta_v_ic(iecidx(je,jb,1),jk,  iecblk(je,jb,1)) -     &
                                       theta_v_ic(iecidx(je,jb,1),jk+1,iecblk(je,jb,1)) ) +   &
               diff_inst(jg)%c_lin_e(je,2,jb)*(theta_v_ic(iecidx(je,jb,2),jk,  iecblk(je,jb,2)) -     &
                                       theta_v_ic(iecidx(je,jb,2),jk+1,iecblk(je,jb,2)) ) ) / &
               diff_inst(jg)%ddqz_z_full_e(je,jk,jb)

            dwdn (je,jk) = p_patch_diff(jg)%edges%inv_dual_edge_length(je,jb)* (    &
              0.5_wp*(w(iecidx(je,jb,1),jk,  iecblk(je,jb,1)) +  &
                      w(iecidx(je,jb,1),jk+1,iecblk(je,jb,1))) - &
              0.5_wp*(w(iecidx(je,jb,2),jk,  iecblk(je,jb,2)) +  &
                      w(iecidx(je,jb,2),jk+1,iecblk(je,jb,2)))   )

            dwdt (je,jk) = p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) *                                   &
                           p_patch_diff(jg)%edges%tangent_orientation(je,jb) * (                                     &
              0.5_wp*(z_w_v(ividx(je,jb,1),jk,ivblk(je,jb,1))+z_w_v(ividx(je,jb,1),jk+1,ivblk(je,jb,1))) - &
              0.5_wp*(z_w_v(ividx(je,jb,2),jk,ivblk(je,jb,2))+z_w_v(ividx(je,jb,2),jk+1,ivblk(je,jb,2)))   )

            kh_smag3d_e(je,jk) = 2._wp*(                                                           &
              ( (vn_vert4-vn_vert3)*p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb) )**2 + &
              (dvt_tang*p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb))**2 + dwdz(je,jk)**2) + &
              0.5_wp *( (p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) *                             &
              p_patch_diff(jg)%edges%tangent_orientation(je,jb)*(vn_vert2-vn_vert1) +          &
              p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb)*dvt_norm )**2 +                     &
              (dvndz(je,jk) + dwdn(je,jk))**2 + (dvtdz(je,jk) + dwdt(je,jk))**2 ) -                &
              3._wp*diff_inst(jg)%grav * dthvdz(je,jk) / (                                                       &
              diff_inst(jg)%c_lin_e(je,1,jb)*theta_v(iecidx(je,jb,1),jk,iecblk(je,jb,1)) +       &
              diff_inst(jg)%c_lin_e(je,2,jb)*theta_v(iecidx(je,jb,2),jk,iecblk(je,jb,2)) )

            ! 2D Smagorinsky diffusion coefficient
            kh_smag_e(je,jk,jb) = diff_multfac_smag(jk)*SQRT( MAX(kh_smag3d_e(je,jk), fac2d*( &
              ((vn_vert4-vn_vert3)*p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb)-   &
              dvt_tang*p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) )**2 + (            &
              (vn_vert2-vn_vert1)*p_patch_diff(jg)%edges%tangent_orientation(je,jb)*      &
              p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) +                                   &
              dvt_norm*p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb) )**2 ) ) )

            ! The factor of 4 comes from dividing by twice the "correct" length
            z_nabla2_e(je,jk,jb) = 4._wp * (                                      &
              (vn_vert4 + vn_vert3 - 2._wp*vn(je,jk,jb))  &
              *p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb)**2 +                     &
              (vn_vert2 + vn_vert1 - 2._wp*vn(je,jk,jb))  &
              *p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb)**2 )

            kh_smag_ec(je,jk,jb) = kh_smag_e(je,jk,jb)
            ! Subtract part of the fourth-order background diffusion coefficient
            kh_smag_e(je,jk,jb) = MAX(0._vp,kh_smag_e(je,jk,jb) - smag_offset)
            ! Limit diffusion coefficient to the theoretical CFL stability threshold
            kh_smag_e(je,jk,jb) = MIN(kh_smag_e(je,jk,jb),smag_limit(jk))
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

      ENDDO
!$OMP END DO NOWAIT
!$OMP END PARALLEL

    ELSE IF ((diffu_type == 3 .OR. diffu_type == 5) .AND. diff_inst(jg)%discr_vn >= 2) THEN

       !  RBF reconstruction of velocity at vertices and cells
      CALL rbf_vec_interpol_vertex( vn, p_patch_diff(jg), &
                                    diff_inst(jg)%rbf_vec_idx_v, diff_inst(jg)%rbf_vec_blk_v, &
                                    diff_inst(jg)%rbf_vec_coeff_v, u_vert, v_vert,            &
                                    opt_rlend=min_rlvert_int-1, opt_acc_async=.TRUE. )

      ! DA: This wait ideally should be removed
      !$ACC WAIT

      IF (diff_inst(jg)%discr_vn == 2) THEN

        CALL rbf_vec_interpol_cell( vn, p_patch_diff(jg), &
                                    diff_inst(jg)%rbf_vec_idx_v, diff_inst(jg)%rbf_vec_blk_v, &
                                    diff_inst(jg)%rbf_vec_coeff_v, u_cell, v_cell,            &
                                    opt_rlend=min_rlcell_int-1 )
      ELSE

        CALL edges2cells_vector( vn, vt, p_patch_diff(jg), diff_inst(jg)%e_bln_c_u, diff_inst(jg)%e_bln_c_v, &
                                 u_cell, v_cell, opt_rlend=min_rlcell_int-1 )
      ENDIF

      IF (diff_inst(jg)%p_test_run) THEN
        !$ACC KERNELS IF(i_am_accel_node) ASYNC(1)
        z_nabla2_e = 0._wp
        !$ACC END KERNELS
      ENDIF

!$OMP PARALLEL PRIVATE(i_startblk,i_endblk,rl_start,rl_end)

      rl_start = start_bdydiff_e
      rl_end   = min_rledge_int - 1

      i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,vn_vert1,vn_vert2,vn_cell1,vn_cell2,&
!$OMP             dvt_norm,dvt_tang), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        ! Computation of wind field deformation

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO je = i_startidx, i_endidx
!DIR$ IVDEP
          DO jk = 1, nlev
#else
!$NEC outerloop_unroll(4)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
#endif

            vn_vert1 =        u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v1 + &
                              v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v2

            vn_vert2 =        u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v1 + &
                              v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v2

            dvt_tang =        p_patch_diff(jg)%edges%tangent_orientation(je,jb)* (   &
                              u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%dual_normal_vert(je,jb,2)%v1 + &
                              v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%dual_normal_vert(je,jb,2)%v2 - &
                             (u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%dual_normal_vert(je,jb,1)%v1 + &
                              v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%dual_normal_vert(je,jb,1)%v2) )

            vn_cell1 =        u_cell(iecidx(je,jb,1),jk,iecblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%primal_normal_cell(je,jb,1)%v1 + &
                              v_cell(iecidx(je,jb,1),jk,iecblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%primal_normal_cell(je,jb,1)%v2

            vn_cell2 =        u_cell(iecidx(je,jb,2),jk,iecblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%primal_normal_cell(je,jb,2)%v1 + &
                              v_cell(iecidx(je,jb,2),jk,iecblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%primal_normal_cell(je,jb,2)%v2

            dvt_norm =        u_cell(iecidx(je,jb,2),jk,iecblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%dual_normal_cell(je,jb,2)%v1 + &
                              v_cell(iecidx(je,jb,2),jk,iecblk(je,jb,2)) * &
                              p_patch_diff(jg)%edges%dual_normal_cell(je,jb,2)%v2 - &
                             (u_cell(iecidx(je,jb,1),jk,iecblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%dual_normal_cell(je,jb,1)%v1 + &
                              v_cell(iecidx(je,jb,1),jk,iecblk(je,jb,1)) * &
                              p_patch_diff(jg)%edges%dual_normal_cell(je,jb,1)%v2)


            ! Smagorinsky diffusion coefficient
            kh_smag_e(je,jk,jb) = diff_multfac_smag(jk)*SQRT(                           (  &
              (vn_cell2-vn_cell1)*p_patch_diff(jg)%edges%inv_dual_edge_length(je,jb)-               &
              dvt_tang*p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) )**2 + (         &
              (vn_vert2-vn_vert1)*p_patch_diff(jg)%edges%tangent_orientation(je,jb)*  &
              p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb) +                                &
              dvt_norm*p_patch_diff(jg)%edges%inv_dual_edge_length(je,jb))**2 )

            ! The factor of 4 comes from dividing by twice the "correct" length
            z_nabla2_e(je,jk,jb) = 4._wp * (                                      &
              (vn_cell2 + vn_cell1 - 2._wp*vn(je,jk,jb))                &
              *p_patch_diff(jg)%edges%inv_dual_edge_length(je,jb)**2 +                     &
              (vn_vert2 + vn_vert1 - 2._wp*vn(je,jk,jb))  &
              *p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb)**2 )

#ifndef _OPENACC
          ENDDO
        ENDDO

        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
#endif

            kh_smag_ec(je,jk,jb) = kh_smag_e(je,jk,jb)
            ! Subtract part of the fourth-order background diffusion coefficient
            kh_smag_e(je,jk,jb) = MAX(0._vp,kh_smag_e(je,jk,jb) - smag_offset)
            ! Limit diffusion coefficient to the theoretical CFL stability threshold
            kh_smag_e(je,jk,jb) = MIN(kh_smag_e(je,jk,jb),smag_limit(jk))
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

      ENDDO
!$OMP END DO NOWAIT
!$OMP END PARALLEL

    ENDIF

    ! Compute input quantities for turbulence scheme
    IF ((diffu_type == 3 .OR. diffu_type == 5) .AND.                               &
        (diff_inst(jg)%itype_sher >= 1 .OR. diff_inst(jg)%ltkeshs)) THEN

!$OMP PARALLEL PRIVATE(i_startblk,i_endblk)
      rl_start = grf_bdywidth_c+1
      rl_end   = min_rlcell_int

      i_startblk = p_patch_diff(jg)%cells%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%cells%end_block(rl_end)

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx,kh_c,div), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO jc = i_startidx, i_endidx
          DO jk = 1, nlev
#else
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx
#endif

            kh_c(jc,jk) = (kh_smag_ec(ieidx(jc,jb,1),jk,ieblk(jc,jb,1))*diff_inst(jg)%e_bln_c_s(jc,1,jb) + &
                           kh_smag_ec(ieidx(jc,jb,2),jk,ieblk(jc,jb,2))*diff_inst(jg)%e_bln_c_s(jc,2,jb) + &
                           kh_smag_ec(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))*diff_inst(jg)%e_bln_c_s(jc,3,jb))/ &
                          diff_multfac_smag(jk)

            div(jc,jk) = vn(ieidx(jc,jb,1),jk,ieblk(jc,jb,1))*diff_inst(jg)%geofac_div(jc,1,jb) + &
                         vn(ieidx(jc,jb,2),jk,ieblk(jc,jb,2))*diff_inst(jg)%geofac_div(jc,2,jb) + &
                         vn(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))*diff_inst(jg)%geofac_div(jc,3,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 2, nlev ! levels 1 and nlevp1 are unused
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx

            div_ic(jc,jk,jb) = diff_inst(jg)%wgtfac_c(jc,jk,jb)*div(jc,jk) + &
              (1._wp-diff_inst(jg)%wgtfac_c(jc,jk,jb))*div(jc,jk-1)

            hdef_ic(jc,jk,jb) = (diff_inst(jg)%wgtfac_c(jc,jk,jb)*kh_c(jc,jk) + &
              (1._wp-diff_inst(jg)%wgtfac_c(jc,jk,jb))*kh_c(jc,jk-1))**2
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
      ENDDO
!$OMP END DO
!$OMP END PARALLEL

    ENDIF

    IF (diffu_type == 5) THEN ! Add fourth-order background diffusion

      IF (diff_inst(jg)%discr_vn > 1) THEN
        CALL sync_patch_array(SYNC_E,p_patch(jg),z_nabla2_e,      &
                              opt_varname="diffusion: nabla2_e")
      END IF

      ! Interpolate nabla2(v) to vertices in order to compute nabla2(nabla2(v))

      IF (diff_inst(jg)%p_test_run) THEN
        !$ACC KERNELS IF(i_am_accel_node)
        u_vert = 0._wp
        v_vert = 0._wp
        !$ACC END KERNELS
      ENDIF

      CALL rbf_vec_interpol_vertex( z_nabla2_e, p_patch_diff(jg), &
                                    diff_inst(jg)%rbf_vec_idx_v, diff_inst(jg)%rbf_vec_blk_v, &
                                    diff_inst(jg)%rbf_vec_coeff_v, u_vert, v_vert,            &
                                    opt_rlstart=4, opt_rlend=min_rlvert_int, opt_acc_async=.TRUE. )
      rl_start = grf_bdywidth_e+1
      rl_end   = min_rledge_int

      IF (diff_inst(jg)%itype_comm == 1 .OR. diff_inst(jg)%itype_comm == 3) THEN
#ifdef __MIXED_PRECISION
        CALL sync_patch_array_mult_mp(SYNC_V,p_patch(jg),0,2,f3din1_sp=u_vert,f3din2_sp=v_vert, &
                                      opt_varname="diffusion: u_vert and v_vert 3")
#else
        CALL sync_patch_array_mult(SYNC_V,p_patch(jg),2,u_vert,v_vert,                          &
                                   opt_varname="diffusion: u_vert and v_vert 3")
#endif
      ENDIF

!$OMP PARALLEL PRIVATE(i_startblk,i_endblk)

      i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,nabv_tang,nabv_norm,z_nabla4_e2,z_d_vn_hdf), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

         ! Compute nabla4(v)
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO je = i_startidx, i_endidx
          DO jk = 1, nlev
#else
!$NEC outerloop_unroll(4)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
#endif

            nabv_tang = u_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v1 + &
                        v_vert(ividx(je,jb,1),jk,ivblk(je,jb,1)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,1)%v2 + &
                        u_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v1 + &
                        v_vert(ividx(je,jb,2),jk,ivblk(je,jb,2)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,2)%v2

            nabv_norm = u_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,3)%v1 + &
                        v_vert(ividx(je,jb,3),jk,ivblk(je,jb,3)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,3)%v2 + &
                        u_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,4)%v1 + &
                        v_vert(ividx(je,jb,4),jk,ivblk(je,jb,4)) * &
                        p_patch_diff(jg)%edges%primal_normal_vert(je,jb,4)%v2

            ! The factor of 4 comes from dividing by twice the "correct" length
            z_nabla4_e2(je,jk) = 4._wp * (                          &
              (nabv_norm - 2._wp*z_nabla2_e(je,jk,jb))              &
              *p_patch_diff(jg)%edges%inv_vert_vert_length(je,jb)**2 +          &
              (nabv_tang - 2._wp*z_nabla2_e(je,jk,jb))              &
              *p_patch_diff(jg)%edges%inv_primal_edge_length(je,jb)**2 )

          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

        ! Apply diffusion for the case of diffu_type = 5
        IF ( jg == 1 .AND. diff_inst(jg)%l_limited_area .OR. jg > 1 .AND. .NOT. diff_inst(jg)%lfeedback ) THEN
          !
          ! Domains with lateral boundary and nests without feedback
          !
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_d_vn_hdf =   p_patch_diff(jg)%edges%area_edge(je,jb)                              &
                &          * (  MAX(nudgezone_diff*diff_inst(jg)%nudgecoeff_e(je,jb),            &
                &                   REAL(kh_smag_e(je,jk,jb),wp)) * z_nabla2_e(je,jk,jb) &
                &             - p_patch_diff(jg)%edges%area_edge(je,jb)                           &
                &             * diff_multfac_vn(jk) * z_nabla4_e2(je,jk)   )
              !
              vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
                ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
              !
              IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
                ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP

        ELSE IF (jg > 1) THEN
          !
          ! Nests with feedback
          !
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_d_vn_hdf =   p_patch_diff(jg)%edges%area_edge(je,jb)                                                       &
                &          * (  kh_smag_e(je,jk,jb) * z_nabla2_e(je,jk,jb)                                        &
                &             - p_patch_diff(jg)%edges%area_edge(je,jb)                                                    &
                &             * MAX(diff_multfac_vn(jk),bdy_diff*diff_inst(jg)%nudgecoeff_e(je,jb)) * z_nabla4_e2(je,jk) )
              !
              vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
                ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
              !
              IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
                ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP

        ELSE

          !
          ! Global domains
          !
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_d_vn_hdf =   p_patch_diff(jg)%edges%area_edge(je,jb)                 &
                &          * (  kh_smag_e(je,jk,jb) * z_nabla2_e(je,jk,jb)  &
                &             - p_patch_diff(jg)%edges%area_edge(je,jb)              &
                &             * diff_multfac_vn(jk) * z_nabla4_e2(je,jk)   )

              !
              vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
                ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
              !
              IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
                ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDIF

      ENDDO
!$OMP END DO NOWAIT
!$OMP END PARALLEL

    ENDIF

    ! Apply diffusion for the cases of diffu_type = 3 or 4

!$OMP PARALLEL PRIVATE(i_startblk,i_endblk,rl_start,rl_end)

    rl_start = grf_bdywidth_e+1
    rl_end   = min_rledge_int

    i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
    i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

    IF (diffu_type == 3) THEN ! Only Smagorinsky diffusion
      IF ( jg == 1 .AND. diff_inst(jg)%l_limited_area .OR. jg > 1 .AND. .NOT. diff_inst(jg)%lfeedback ) THEN

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,z_d_vn_hdf) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk,i_endblk

          CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_d_vn_hdf =   p_patch_diff(jg)%edges%area_edge(je,jb)                                             &
                &          * MAX(nudgezone_diff*diff_inst(jg)%nudgecoeff_e(je,jb),REAL(kh_smag_e(je,jk,jb),wp)) &
                &          * z_nabla2_e(je,jk,jb)
              !
              vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
                ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
              !
              IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
                ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDDO
!$OMP END DO

      ELSE

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,z_d_vn_hdf) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = i_startblk,i_endblk

          CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
          DO jk = 1, nlev
!DIR$ IVDEP
            DO je = i_startidx, i_endidx
              !
              z_d_vn_hdf = p_patch_diff(jg)%edges%area_edge(je,jb) * kh_smag_e(je,jk,jb) * z_nabla2_e(je,jk,jb)
              !
              vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
              !
#ifdef __ENABLE_DDT_VN_XYZ__
              IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
                ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
              !
              IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
                ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
              END IF
#endif
              !
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDDO
!$OMP END DO

      ENDIF

    ELSE IF (diffu_type == 4) THEN

!$OMP DO PRIVATE(jb,i_startidx,i_endidx,jk,je,z_d_vn_hdf) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 1, nlev
!DIR$ IVDEP
          DO je = i_startidx, i_endidx
            !
            z_d_vn_hdf = - p_patch_diff(jg)%edges%area_edge(je,jb)*p_patch_diff(jg)%edges%area_edge(je,jb) &
              &          * diff_multfac_vn(jk) * z_nabla4_e(je,jk,jb)
            !
            vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
            !
#ifdef __ENABLE_DDT_VN_XYZ__
            IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
              ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
            END IF
            !
            IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
              ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
            END IF
#endif
            !
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
      ENDDO
!$OMP END DO

    ENDIF

    IF (diff_inst(jg)%l_limited_area .OR. jg > 1) THEN

      ! Lateral boundary diffusion for vn
      i_startblk = p_patch_diff(jg)%edges%start_block(start_bdydiff_e)
      i_endblk   = p_patch_diff(jg)%edges%end_block(grf_bdywidth_e)

!$OMP DO PRIVATE(je,jk,jb,i_startidx,i_endidx,z_d_vn_hdf) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, start_bdydiff_e, grf_bdywidth_e)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRIVATE(z_d_vn_hdf) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 1, nlev
!DIR$ IVDEP
          DO je = i_startidx, i_endidx
            !
            z_d_vn_hdf = p_patch_diff(jg)%edges%area_edge(je,jb) * fac_bdydiff_v * z_nabla2_e(je,jk,jb)
            !
            vn(je,jk,jb)            =  vn(je,jk,jb)         + z_d_vn_hdf
            !
#ifdef __ENABLE_DDT_VN_XYZ__
            IF ( diff_inst(jg)%ddt_vn_hdf_is_associated) THEN
              ddt_vn_hdf(je,jk,jb)  =  ddt_vn_hdf(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
            END IF
            !
            IF ( diff_inst(jg)%ddt_vn_dyn_is_associated) THEN
              ddt_vn_dyn(je,jk,jb)  =  ddt_vn_dyn(je,jk,jb) + z_d_vn_hdf * r_dtimensubsteps
            END IF
#endif
            !
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
      ENDDO
!$OMP END DO

    ENDIF ! vn boundary diffusion

    IF (diff_inst(jg)%lhdiff_rcf .AND. diff_inst(jg)%lhdiff_w) THEN ! add diffusion on vertical wind speed
                     ! remark: the surface level (nlevp1) is excluded because w is diagnostic there

      IF (diff_inst(jg)%l_limited_area .AND. jg == 1) THEN
        rl_start = grf_bdywidth_c+1
      ELSE
        rl_start = grf_bdywidth_c
      ENDIF
      rl_end   = min_rlcell_int-1

      i_startblk = p_patch_diff(jg)%cells%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%cells%end_block(rl_end)

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO jc = i_startidx, i_endidx
!DIR$ IVDEP
#ifdef _CRAYFTN
!DIR$ PREFERVECTOR
#endif
          DO jk = 1, nlev
#else
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx
#endif
            z_nabla2_c(jc,jk,jb) =  &
              w(jc,jk,jb)                        *diff_inst(jg)%geofac_n2s(jc,1,jb) + &
              w(icidx(jc,jb,1),jk,icblk(jc,jb,1))*diff_inst(jg)%geofac_n2s(jc,2,jb) + &
              w(icidx(jc,jb,2),jk,icblk(jc,jb,2))*diff_inst(jg)%geofac_n2s(jc,3,jb) + &
              w(icidx(jc,jb,3),jk,icblk(jc,jb,3))*diff_inst(jg)%geofac_n2s(jc,4,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

        IF (diff_inst(jg)%itype_sher == 2) THEN ! compute horizontal gradients of w
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
          DO jc = i_startidx, i_endidx
!DIR$ IVDEP
            DO jk = 2, nlev
#else
          DO jk = 2, nlev
            DO jc = i_startidx, i_endidx
#endif
             dwdx(jc,jk,jb) =  diff_inst(jg)%geofac_grg(jc,1,jb,1)*w(jc,jk,jb) + &
               diff_inst(jg)%geofac_grg(jc,2,jb,1)*w(icidx(jc,jb,1),jk,icblk(jc,jb,1))   + &
               diff_inst(jg)%geofac_grg(jc,3,jb,1)*w(icidx(jc,jb,2),jk,icblk(jc,jb,2))   + &
               diff_inst(jg)%geofac_grg(jc,4,jb,1)*w(icidx(jc,jb,3),jk,icblk(jc,jb,3))

             dwdy(jc,jk,jb) =  diff_inst(jg)%geofac_grg(jc,1,jb,2)*w(jc,jk,jb) + &
               diff_inst(jg)%geofac_grg(jc,2,jb,2)*w(icidx(jc,jb,1),jk,icblk(jc,jb,1))   + &
               diff_inst(jg)%geofac_grg(jc,3,jb,2)*w(icidx(jc,jb,2),jk,icblk(jc,jb,2))   + &
               diff_inst(jg)%geofac_grg(jc,4,jb,2)*w(icidx(jc,jb,3),jk,icblk(jc,jb,3))

            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDIF

      ENDDO
!$OMP END DO

      IF (diff_inst(jg)%l_limited_area .AND. jg == 1) THEN
        rl_start = 0
      ELSE
        rl_start = grf_bdywidth_c+1
      ENDIF
      rl_end   = min_rlcell_int

      i_startblk = p_patch_diff(jg)%cells%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%cells%end_block(rl_end)

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
        DO jc = i_startidx, i_endidx
!DIR$ IVDEP
          DO jk = 1, nlev
#else
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx
#endif
            w(jc,jk,jb) = w(jc,jk,jb) - diff_multfac_w * p_patch_diff(jg)%cells%area(jc,jb)**2 * &
             (z_nabla2_c(jc,jk,jb)                        *diff_inst(jg)%geofac_n2s(jc,1,jb) +                      &
              z_nabla2_c(icidx(jc,jb,1),jk,icblk(jc,jb,1))*diff_inst(jg)%geofac_n2s(jc,2,jb) +                      &
              z_nabla2_c(icidx(jc,jb,2),jk,icblk(jc,jb,2))*diff_inst(jg)%geofac_n2s(jc,3,jb) +                      &
              z_nabla2_c(icidx(jc,jb,3),jk,icblk(jc,jb,3))*diff_inst(jg)%geofac_n2s(jc,4,jb))
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

        ! Add nabla2 diffusion in upper damping layer (if present)
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 2, diff_inst(jg)%nrdmax
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            w(jc,jk,jb) = w(jc,jk,jb) +                         &
              diff_multfac_n2w(jk) * p_patch_diff(jg)%cells%area(jc,jb) * z_nabla2_c(jc,jk,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

      ENDDO
!$OMP END DO

    ENDIF ! w diffusion

!$OMP END PARALLEL

    IF (diff_inst(jg)%itype_comm == 1 .OR. diff_inst(jg)%itype_comm == 3) THEN
      CALL sync_patch_array(SYNC_E, p_patch(jg), vn,opt_varname="diffusion: vn sync")
    ENDIF

    IF (ltemp_diffu) THEN ! Smagorinsky temperature diffusion

      IF (diff_inst(jg)%l_zdiffu_t) THEN
        icell      => diff_inst(jg)%zd_indlist
        iblk       => diff_inst(jg)%zd_blklist
        ilev       => diff_inst(jg)%zd_vertidx
   !     iedge      => diff_inst(jg)%zd_edgeidx
   !     iedblk     => diff_inst(jg)%zd_edgeblk
        vcoef      => diff_inst(jg)%zd_intcoef
   !     blcoef     => diff_inst(jg)%zd_e2cell
        zd_geofac => diff_inst(jg)%zd_geofac

!!!        nproma_zdiffu = cpu_min_nproma(diff_inst(jg)%nproma,256)
#ifdef _OPENACC
        nproma_zdiffu = diff_inst(jg)%nproma
#else
        nproma_zdiffu = MIN(diff_inst(jg)%nproma,256)
#endif
        nblks_zdiffu = INT(diff_inst(jg)%zd_listdim/nproma_zdiffu)
        npromz_zdiffu = MOD(diff_inst(jg)%zd_listdim,nproma_zdiffu)
        IF (npromz_zdiffu > 0) THEN
          nblks_zdiffu = nblks_zdiffu + 1
        ELSE
            npromz_zdiffu = nproma_zdiffu
        ENDIF
      ENDIF

!$OMP PARALLEL PRIVATE(rl_start,rl_end,i_startblk,i_endblk)

      ! Enhance Smagorinsky diffusion coefficient in the presence of excessive grid-point cold pools
      ! This is restricted to the two lowest model levels
      !
      rl_start = grf_bdywidth_c
      rl_end   = min_rlcell_int-1

      i_startblk = p_patch_diff(jg)%cells%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%cells%end_block(rl_end)

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx,ic,tdiff,trefdiff), ICON_OMP_RUNTIME_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        ic = 0

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = nlev-1, nlev
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            ! Perturbation potential temperature difference between local point and average of the three neighbors
            tdiff = theta_v(jc,jk,jb) -                          &
              (theta_v(icidx(jc,jb,1),jk,icblk(jc,jb,1)) +       &
               theta_v(icidx(jc,jb,2),jk,icblk(jc,jb,2)) +       &
               theta_v(icidx(jc,jb,3),jk,icblk(jc,jb,3)) ) / 3._wp
            trefdiff = diff_inst(jg)%theta_ref_mc(jc,jk,jb) -                       &
              (diff_inst(jg)%theta_ref_mc(icidx(jc,jb,1),jk,icblk(jc,jb,1)) +       &
               diff_inst(jg)%theta_ref_mc(icidx(jc,jb,2),jk,icblk(jc,jb,2)) +       &
               diff_inst(jg)%theta_ref_mc(icidx(jc,jb,3),jk,icblk(jc,jb,3)) ) / 3._wp

            ! Enahnced horizontal diffusion is applied if the theta perturbation is either
            ! - at least 5 K colder than the average of the neighbor points on valley points (determined by trefdiff < 0.) or
            ! - at least 7.5 K colder than the average of the neighbor points otherwise
            IF (tdiff-trefdiff < thresh_tdiff .AND. trefdiff < 0._wp .OR. tdiff-trefdiff < 1.5_wp*thresh_tdiff) THEN
#ifndef _OPENACC
              ic = ic+1
              iclist(ic,jb) = jc
              iklist(ic,jb) = jk
              tdlist(ic,jb) = thresh_tdiff - tdiff + trefdiff
#else
      ! Enhance Smagorinsky coefficients at the three edges of the cells included in the list
! Attention: this operation is neither vectorizable nor OpenMP-parallelizable (race conditions!)
              enh_diffu_3d(jc,jk,jb) = (thresh_tdiff - tdiff + trefdiff)*5.e-4_vp
            ELSE
              enh_diffu_3d(jc,jk,jb) = -HUGE(0._vp)   ! In order that this is never taken as the MAX
#endif
            ENDIF
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
        icount(jb) = ic

      ENDDO
!$OMP END DO

      ! Enhance Smagorinsky coefficients at the three edges of the cells included in the list
      ! Attention: this operation is neither vectorizable nor OpenMP-parallelizable (race conditions!)

#ifndef _OPENACC
!$OMP MASTER
      DO jb = i_startblk,i_endblk

        IF (icount(jb) > 0) THEN
          DO ic = 1, icount(jb)
            jc = iclist(ic,jb)
            jk = iklist(ic,jb)
            enh_diffu = tdlist(ic,jb)*5.e-4_vp
            kh_smag_e(ieidx(jc,jb,1),jk,ieblk(jc,jb,1)) = MAX(enh_diffu,kh_smag_e(ieidx(jc,jb,1),jk,ieblk(jc,jb,1)))
            kh_smag_e(ieidx(jc,jb,2),jk,ieblk(jc,jb,2)) = MAX(enh_diffu,kh_smag_e(ieidx(jc,jb,2),jk,ieblk(jc,jb,2)))
            kh_smag_e(ieidx(jc,jb,3),jk,ieblk(jc,jb,3)) = MAX(enh_diffu,kh_smag_e(ieidx(jc,jb,3),jk,ieblk(jc,jb,3)))
          ENDDO
        ENDIF

      ENDDO

!$OMP END MASTER
!$OMP BARRIER

#else

      rl_start = grf_bdywidth_e+1
      rl_end   = min_rledge_int

      i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
      i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = nlev-1, nlev
          DO je = i_startidx, i_endidx
            kh_smag_e(je,jk,jb) = MAX(kh_smag_e(je,jk,jb), enh_diffu_3d(iecidx(je,jb,1),jk,iecblk(je,jb,1)), &
                 enh_diffu_3d(iecidx(je,jb,2),jk,iecblk(je,jb,2)) )
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
      ENDDO
#endif

      IF (diff_inst(jg)%discr_t == 1) THEN  ! use discretization K*nabla(theta)

        rl_start = grf_bdywidth_c+1
        rl_end   = min_rlcell_int

        i_startblk = p_patch_diff(jg)%cells%start_block(rl_start)
        i_endblk   = p_patch_diff(jg)%cells%end_block(rl_end)

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx), ICON_OMP_RUNTIME_SCHEDULE
        DO jb = i_startblk,i_endblk

          CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! interpolated diffusion coefficient times nabla2(theta)
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
          DO jc = i_startidx, i_endidx
!DIR$ IVDEP
#ifdef _CRAYFTN
!DIR$ PREFERVECTOR
#endif
            DO jk = 1, nlev
#else
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
#endif
              z_temp(jc,jk,jb) =  &
               (kh_smag_e(ieidx(jc,jb,1),jk,ieblk(jc,jb,1))*diff_inst(jg)%e_bln_c_s(jc,1,jb)          + &
                kh_smag_e(ieidx(jc,jb,2),jk,ieblk(jc,jb,2))*diff_inst(jg)%e_bln_c_s(jc,2,jb)          + &
                kh_smag_e(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))*diff_inst(jg)%e_bln_c_s(jc,3,jb))         * &
               (theta_v(jc,jk,jb)                        *diff_inst(jg)%geofac_n2s(jc,1,jb) + &
                theta_v(icidx(jc,jb,1),jk,icblk(jc,jb,1))*diff_inst(jg)%geofac_n2s(jc,2,jb) + &
                theta_v(icidx(jc,jb,2),jk,icblk(jc,jb,2))*diff_inst(jg)%geofac_n2s(jc,3,jb) + &
                theta_v(icidx(jc,jb,3),jk,icblk(jc,jb,3))*diff_inst(jg)%geofac_n2s(jc,4,jb))
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDDO
!$OMP END DO

      ELSE IF (diff_inst(jg)%discr_t == 2) THEN ! use conservative discretization div(k*grad(theta))

        rl_start = grf_bdywidth_e
        rl_end   = min_rledge_int - 1

        i_startblk = p_patch_diff(jg)%edges%start_block(rl_start)
        i_endblk   = p_patch_diff(jg)%edges%end_block(rl_end)

!$OMP DO PRIVATE(jk,je,jb,i_startidx,i_endidx), ICON_OMP_RUNTIME_SCHEDULE
        DO jb = i_startblk,i_endblk

          CALL get_indices_e(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! compute kh_smag_e * grad(theta) (stored in z_nabla2_e for memory efficiency)
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
          DO je = i_startidx, i_endidx
!DIR$ IVDEP
#ifdef _CRAYFTN
!DIR$ PREFERVECTOR
#endif
            DO jk = 1, nlev
#else
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
#endif
              z_nabla2_e(je,jk,jb) = kh_smag_e(je,jk,jb) *              &
                p_patch_diff(jg)%edges%inv_dual_edge_length(je,jb)*              &
               (theta_v(iecidx(je,jb,2),jk,iecblk(je,jb,2)) - &
                theta_v(iecidx(je,jb,1),jk,iecblk(je,jb,1)))
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDDO
!$OMP END DO

        rl_start = grf_bdywidth_c+1
        rl_end   = min_rlcell_int

        i_startblk = p_patch_diff(jg)%cells%start_block(rl_start)
        i_endblk   = p_patch_diff(jg)%cells%end_block(rl_end)

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx), ICON_OMP_RUNTIME_SCHEDULE
        DO jb = i_startblk,i_endblk

          CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                             i_startidx, i_endidx, rl_start, rl_end)

          ! now compute the divergence of the quantity above
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
#ifdef __LOOP_EXCHANGE
          DO jc = i_startidx, i_endidx
            DO jk = 1, nlev
#else
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
#endif
              z_temp(jc,jk,jb) =                                                         &
                z_nabla2_e(ieidx(jc,jb,1),jk,ieblk(jc,jb,1))*diff_inst(jg)%geofac_div(jc,1,jb) + &
                z_nabla2_e(ieidx(jc,jb,2),jk,ieblk(jc,jb,2))*diff_inst(jg)%geofac_div(jc,2,jb) + &
                z_nabla2_e(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))*diff_inst(jg)%geofac_div(jc,3,jb)
            ENDDO
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDDO
!$OMP END DO

      ENDIF

      IF (diff_inst(jg)%l_zdiffu_t) THEN ! Compute temperature diffusion truly horizontally over steep slopes
                           ! A conservative discretization is not possible here
!$OMP DO PRIVATE(jb,jc,ic,nlen_zdiffu,ishift) ICON_OMP_DEFAULT_SCHEDULE
        DO jb = 1, nblks_zdiffu
          IF (jb == nblks_zdiffu) THEN
            nlen_zdiffu = npromz_zdiffu
          ELSE
            nlen_zdiffu = nproma_zdiffu
          ENDIF
          ishift = (jb-1)*nproma_zdiffu
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) PRESENT(icell, ilev, iblk, vcoef, zd_geofac) &
          !$ACC   GANG VECTOR ASYNC(1) IF(i_am_accel_node)
!$NEC ivdep
!DIR$ IVDEP
          DO jc = 1, nlen_zdiffu
            ic = ishift+jc
            z_temp(icell(1,ic),ilev(1,ic),iblk(1,ic)) =                                          &
              z_temp(icell(1,ic),ilev(1,ic),iblk(1,ic)) + diff_inst(jg)%zd_diffcoef(ic)*          &
!              MAX(diff_inst(jg)%zd_diffcoef(ic),        &
!              kh_smag_e(iedge(1,ic),ilev(1,ic),iedblk(1,ic))* blcoef(1,ic)  +                    &
!              kh_smag_e(iedge(2,ic),ilev(1,ic),iedblk(2,ic))* blcoef(2,ic)  +                    &
!              kh_smag_e(iedge(3,ic),ilev(1,ic),iedblk(3,ic))* blcoef(3,ic) ) *                   &
             (zd_geofac(1,ic)*theta_v(icell(1,ic),ilev(1,ic),iblk(1,ic)) +            &
              zd_geofac(2,ic)*(vcoef(1,ic)*theta_v(icell(2,ic),ilev(2,ic),iblk(2,ic))+&
              (1._wp-vcoef(1,ic))* theta_v(icell(2,ic),ilev(2,ic)+1,iblk(2,ic)))  +    &
              zd_geofac(3,ic)*(vcoef(2,ic)*theta_v(icell(3,ic),ilev(3,ic),iblk(3,ic))+&
              (1._wp-vcoef(2,ic))*theta_v(icell(3,ic),ilev(3,ic)+1,iblk(3,ic)))  +     &
              zd_geofac(4,ic)*(vcoef(3,ic)*theta_v(icell(4,ic),ilev(4,ic),iblk(4,ic))+&
              (1._wp-vcoef(3,ic))* theta_v(icell(4,ic),ilev(4,ic)+1,iblk(4,ic)))  )
          ENDDO
          !$ACC END PARALLEL LOOP
        ENDDO
!$OMP END DO

      ENDIF

!$OMP DO PRIVATE(jk,jc,jb,i_startidx,i_endidx,z_theta) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_c(p_patch_diff(jg), jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1) IF(i_am_accel_node)
        DO jk = 1, nlev
!DIR$ IVDEP
          DO jc = i_startidx, i_endidx
            z_theta = theta_v(jc,jk,jb)

            theta_v(jc,jk,jb) = theta_v(jc,jk,jb) + &
              p_patch_diff(jg)%cells%area(jc,jb)*z_temp(jc,jk,jb)

            exner(jc,jk,jb) = exner(jc,jk,jb) *      &
              (1._wp+rd_o_cvd*(theta_v(jc,jk,jb)/z_theta-1._wp))

          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP

      ENDDO
!$OMP END DO NOWAIT
!$OMP END PARALLEL

      ! This could be further optimized, but applications without physics are quite rare;
      IF ( .NOT. diff_inst(jg)%lhdiff_rcf .OR. linit .OR. .NOT. diff_inst(jg)%lphys ) THEN
        CALL sync_patch_array_mult(SYNC_C,p_patch(jg),2,theta_v,exner,  &
                                   opt_varname="diffusion: theta and exner")
      ENDIF

    ENDIF ! temperature diffusion

    IF ( .NOT. diff_inst(jg)%lhdiff_rcf .OR. linit .OR. .NOT. diff_inst(jg)%lphys ) THEN
      IF (diff_inst(jg)%lhdiff_w) THEN
        CALL sync_patch_array(SYNC_C,p_patch(jg),w,"diffusion: w")
      END IF
    ENDIF

    IF (diff_inst(jg)%ltimer) CALL timer_stop(timer_nh_hdiffusion)

    !$ACC END DATA

    !$ACC WAIT

  END SUBROUTINE diffusion_run

  !>
  !! finalize_diffusion
  !!
  !! Prepares the horizontal diffusion of velocity and temperature
  !!
  !! @par Revision History
  !! Initial release by William Sawyer, CSCS (2022-11-25)
  !!
  SUBROUTINE diffusion_finalize(jg)
    INTEGER, INTENT(IN) :: jg
    ! Currently nothing to do here
  END SUBROUTINE diffusion_finalize

END MODULE mo_nh_diffusion_new
