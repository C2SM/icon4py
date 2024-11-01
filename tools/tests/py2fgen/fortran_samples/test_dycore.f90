! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

module random_utils
contains
   subroutine fill_random_1d(array, low, high)
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(inout) :: array(:)
      real(c_double), intent(in) :: low, high
      integer :: i
      real(c_double) :: rnd

      do i = 1, size(array)
         call random_number(rnd)
         array(i) = low + rnd*(high - low)
      end do
   end subroutine fill_random_1d

   subroutine fill_random_2d(array, low, high)
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(inout) :: array(:, :)
      real(c_double), intent(in) :: low, high
      integer :: i, j
      real(c_double) :: rnd

      do i = 1, size(array, 1)
         do j = 1, size(array, 2)
            call random_number(rnd)
            array(i, j) = low + rnd*(high - low)
         end do
      end do
   end subroutine fill_random_2d

   subroutine fill_random_2d_int(array, low, high)
     use, intrinsic :: iso_c_binding, only: c_int
     implicit none

     integer(c_int), intent(inout) :: array(:, :)
     integer(c_int), intent(in) :: low, high
     integer :: i, j
     real(c_int) :: rnd

     do i = 1, size(array, 1)
        do j = 1, size(array, 2)
           call random_number(rnd)
           array(i, j) = low + rnd*(high - low)
        end do
     end do
   end subroutine fill_random_2d_int

   subroutine fill_random_3d_int(array, low, high)
      use, intrinsic :: iso_c_binding, only: c_int
      implicit none

      integer(c_int), intent(inout) :: array(:, :, :)
      integer(c_int), intent(in) :: low, high
      integer :: i, j, k
      real :: rnd

      do i = 1, size(array, 1)
         do j = 1, size(array, 2)
            do k = 1, size(array, 3)
               call random_number(rnd)
               array(i, j, k) = int(low + rnd * (high - low))
            end do
         end do
      end do
   end subroutine fill_random_3d_int

   subroutine fill_random_3d(array, low, high)
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(inout) :: array(:, :, :)
      real(c_double), intent(in) :: low, high
      integer :: i, j, k
      real :: rnd

      do i = 1, size(array, 1)
         do j = 1, size(array, 2)
            do k = 1, size(array, 3)
               call random_number(rnd)
               array(i, j, k) = low + rnd * (high - low)
            end do
         end do
      end do
   end subroutine fill_random_3d

   subroutine fill_random_2d_bool(array)
      use, intrinsic :: iso_c_binding, only: c_int
      implicit none

      logical(c_int), intent(inout) :: array(:, :)
      integer :: i, j
      real :: rnd

      do i = 1, size(array, 1)
         do j = 1, size(array, 2)
            call random_number(rnd)
            if (rnd < 0.5) then
               array(i, j) = .false.
            else
               array(i, j) = .true.
            endif
         end do
      end do
   end subroutine fill_random_2d_bool

   subroutine fill_random_1d_bool(array)
      use, intrinsic :: iso_c_binding, only: c_int
      implicit none

      logical(c_int), intent(inout) :: array(:)
      integer :: i
      real :: rnd

      do i = 1, size(array, 1)
        call random_number(rnd)
        if (rnd < 0.5) then
           array(i) = .false.
        else
           array(i) = .true.
        endif
      end do
   end subroutine fill_random_1d_bool


end module random_utils

program solve_nh_simulation
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   use random_utils, only: fill_random_1d, fill_random_2d, fill_random_2d_int, fill_random_2d_bool, fill_random_1d_bool, &
                            fill_random_3d_int, fill_random_3d
   use dycore_plugin
   implicit none

   integer(c_int) :: rc
   integer(c_int) :: n

   ! Constants and parameters
   integer(c_int), parameter :: num_cells = 20896
   integer(c_int), parameter :: num_edges = 31558
   integer(c_int), parameter :: num_verts = 10663
   integer(c_int), parameter :: num_levels = 65
   integer(c_int), parameter :: num_c2ec2o = 4
   integer(c_int), parameter :: num_v2e = 6
   integer(c_int), parameter :: num_c2e = 3
   integer(c_int), parameter :: num_e2c2v = 4
   integer(c_int), parameter :: num_e2c = 2
   integer(c_int), parameter :: num_e2c2eo = 3  !todo: check
   integer(c_int), parameter :: nnow = 1
   integer(c_int), parameter :: nnew = 2
   real(c_double), parameter :: mean_cell_area = 24907282236.708576

   real(c_double), parameter :: dtime = 10.0
   real(c_double), parameter :: rayleigh_damping_height = 12500.0
   real(c_double), parameter :: flat_height = 16000.0
   integer(c_int), parameter :: idyn_timestep = 0
   integer(c_int), parameter :: nflat_gradp = 59
   real(c_double), parameter :: ndyn_substeps = 2.0

   integer(c_int), parameter :: itime_scheme = 4    ! itime scheme can only be 4
   integer(c_int), parameter :: iadv_rhotheta = 2
   integer(c_int), parameter :: igradp_method = 3
   integer(c_int), parameter :: rayleigh_type = 1
   real(c_double), parameter :: rayleigh_coeff = 0.1
   integer(c_int), parameter :: divdamp_order = 24  ! divdamp order can only be 24
   logical(c_int), parameter :: is_iau_active = .false.
   real(c_double), parameter :: iau_wgt_dyn = 0.5
   real(c_double), parameter :: divdamp_fac_o2 = 0.5
   integer(c_int), parameter :: divdamp_type = 1
   real(c_double), parameter :: divdamp_trans_start = 1000.0
   real(c_double), parameter :: divdamp_trans_end = 2000.0
   logical(c_int), parameter :: l_vert_nested = .false.  ! vertical nesting support is not implemented
   real(c_double), parameter :: divdamp_fac = 1.0
   real(c_double), parameter :: divdamp_fac2 = 2.0
   real(c_double), parameter :: divdamp_fac3 = 3.0
   real(c_double), parameter :: divdamp_fac4 = 4.0
   real(c_double), parameter :: divdamp_z = 1.0
   real(c_double), parameter :: divdamp_z2 = 2.0
   real(c_double), parameter :: divdamp_z3 = 3.0
   real(c_double), parameter :: divdamp_z4 = 4.0
   real(c_double), parameter :: htop_moist_proc = 1000.0
   logical(c_int), parameter :: limited_area = .true.
   logical(c_int), parameter :: lprep_adv = .false.
   logical(c_int), parameter :: clean_mflx = .true.
   logical(c_int), parameter :: recompute = .false.
   logical(c_int), parameter :: linit = .false.
   integer(c_int), parameter :: global_root = 4
   integer(c_int), parameter :: global_level = 9
   real(c_double), parameter :: lowest_layer_thickness = 20.0
   real(c_double), parameter :: model_top_height = 23000.0
   real(c_double), parameter :: stretch_factor = 0.65
   real(c_double), parameter :: rhotheta_offctr = -0.1
   real(c_double), parameter :: veladv_offctr = 0.25
   real(c_double), parameter :: max_nudging_coeff = 0.075
   integer(c_int), parameter :: vertical_size = num_levels


   ! Connectivity arrays
   integer(c_int), dimension(:, :), allocatable :: c2e, e2c, e2v, v2e, v2c, c2v
   integer(c_int), dimension(:, :), allocatable :: e2c2v, c2e2c, e2c2e

   integer(c_int), dimension(:), allocatable :: cell_starts, cell_ends
   integer(c_int), dimension(:), allocatable :: vertex_starts, vertex_ends
   integer(c_int), dimension(:), allocatable :: edge_starts, edge_ends

   ! Declaring arrays
    real(c_double), dimension(:), allocatable :: vct_a, vct_b
    real(c_double), dimension(:), allocatable :: rayleigh_w
    real(c_double), dimension(:), allocatable :: tangent_orientation
    real(c_double), dimension(:), allocatable :: inverse_primal_edge_lengths
    real(c_double), dimension(:), allocatable :: inv_dual_edge_length
    real(c_double), dimension(:), allocatable :: inv_vert_vert_length
    real(c_double), dimension(:), allocatable :: edge_areas
    real(c_double), dimension(:), allocatable :: f_e
    real(c_double), dimension(:), allocatable :: cell_areas
    real(c_double), dimension(:), allocatable :: vwind_expl_wgt
    real(c_double), dimension(:), allocatable :: vwind_impl_wgt
    real(c_double), dimension(:), allocatable :: scalfac_dd3d
    real(c_double), dimension(:), allocatable :: nudgecoeff_e
    real(c_double), dimension(:), allocatable :: hmask_dd3d
    logical(c_int), dimension(:), allocatable :: bdy_halo_c
    logical(c_int), dimension(:), allocatable :: mask_prog_halo_c
    logical(c_int), dimension(:), allocatable :: c_owner_mask

    real(c_double), dimension(:, :), allocatable :: theta_ref_mc
    real(c_double), dimension(:, :), allocatable :: exner_pr
    real(c_double), dimension(:, :), allocatable :: exner_dyn_incr
    real(c_double), dimension(:, :), allocatable :: wgtfac_c
    real(c_double), dimension(:, :), allocatable :: e_bln_c_s
    real(c_double), dimension(:, :), allocatable :: geofac_div
    real(c_double), dimension(:, :), allocatable :: geofac_grg_x
    real(c_double), dimension(:, :), allocatable :: geofac_grg_y
    real(c_double), dimension(:, :), allocatable :: geofac_n2s
    real(c_double), dimension(:, :), allocatable :: rbf_coeff_1
    real(c_double), dimension(:, :), allocatable :: rbf_coeff_2
    real(c_double), dimension(:, :), allocatable :: w_now
    real(c_double), dimension(:, :), allocatable :: w_new
    real(c_double), dimension(:, :), allocatable :: vn_now
    real(c_double), dimension(:, :), allocatable :: vn_new
    real(c_double), dimension(:, :), allocatable :: exner_now
    real(c_double), dimension(:, :), allocatable :: exner_new
    real(c_double), dimension(:, :), allocatable :: theta_v_now
    real(c_double), dimension(:, :), allocatable :: theta_v_new
    real(c_double), dimension(:, :), allocatable :: rho_now
    real(c_double), dimension(:, :), allocatable :: rho_new
    real(c_double), dimension(:, :), allocatable :: dual_normal_cell_x
    real(c_double), dimension(:, :), allocatable :: dual_normal_cell_y
    real(c_double), dimension(:, :), allocatable :: dual_normal_vert_x
    real(c_double), dimension(:, :), allocatable :: dual_normal_vert_y
    real(c_double), dimension(:, :), allocatable :: primal_normal_cell_x
    real(c_double), dimension(:, :), allocatable :: primal_normal_cell_y
    real(c_double), dimension(:, :), allocatable :: primal_normal_vert_x
    real(c_double), dimension(:, :), allocatable :: primal_normal_vert_y
    real(c_double), dimension(:, :), allocatable :: exner_exfac
    real(c_double), dimension(:, :), allocatable :: exner_ref_mc
    real(c_double), dimension(:, :), allocatable :: wgtfacq_c_dsl
    real(c_double), dimension(:, :), allocatable :: inv_ddqz_z_full
    real(c_double), dimension(:, :), allocatable :: d_exner_dz_ref_ic
    real(c_double), dimension(:, :), allocatable :: ddqz_z_half
    real(c_double), dimension(:, :), allocatable :: theta_ref_ic
    real(c_double), dimension(:, :), allocatable :: d2dexdz2_fac1_mc
    real(c_double), dimension(:, :), allocatable :: d2dexdz2_fac2_mc
    real(c_double), dimension(:, :), allocatable :: rho_ref_me
    real(c_double), dimension(:, :), allocatable :: theta_ref_me
    real(c_double), dimension(:, :), allocatable :: ddxn_z_full
    real(c_double), dimension(:, :), allocatable :: pg_exdist
    real(c_double), dimension(:, :), allocatable :: ddqz_z_full_e
    real(c_double), dimension(:, :), allocatable :: ddxt_z_full
    real(c_double), dimension(:, :), allocatable :: wgtfac_e
    real(c_double), dimension(:, :), allocatable :: wgtfacq_e
    real(c_double), dimension(:, :), allocatable :: coeff1_dwdz
    real(c_double), dimension(:, :), allocatable :: coeff2_dwdz
    real(c_double), dimension(:, :), allocatable :: grf_tend_rho
    real(c_double), dimension(:, :), allocatable :: grf_tend_thv
    real(c_double), dimension(:, :), allocatable :: grf_tend_w
    real(c_double), dimension(:, :), allocatable :: mass_fl_e
    real(c_double), dimension(:, :), allocatable :: ddt_vn_phy
    real(c_double), dimension(:, :), allocatable :: grf_tend_vn
    real(c_double), dimension(:, :), allocatable :: vn_ie
    real(c_double), dimension(:, :), allocatable :: vt
    real(c_double), dimension(:, :), allocatable :: mass_flx_me
    real(c_double), dimension(:, :), allocatable :: mass_flx_ic
    real(c_double), dimension(:, :), allocatable :: vn_traj
    real(c_double), dimension(:, :), allocatable :: ddt_vn_apc_ntl1
    real(c_double), dimension(:, :), allocatable :: ddt_vn_apc_ntl2
    real(c_double), dimension(:, :), allocatable :: ddt_w_adv_ntl1
    real(c_double), dimension(:, :), allocatable :: ddt_w_adv_ntl2
    real(c_double), dimension(:, :), allocatable :: c_lin_e
    real(c_double), dimension(:, :), allocatable :: pos_on_tplane_e_1
    real(c_double), dimension(:, :), allocatable :: pos_on_tplane_e_2
    real(c_double), dimension(:, :), allocatable :: rbf_vec_coeff_e
    real(c_double), dimension(:, :), allocatable :: w_concorr_c
    real(c_double), dimension(:, :), allocatable :: theta_v_ic
    real(c_double), dimension(:, :), allocatable :: rho_ref_mc
    real(c_double), dimension(:, :), allocatable :: rho_ic
    real(c_double), dimension(:, :), allocatable :: e_flx_avg
    real(c_double), dimension(:, :), allocatable :: ddt_exner_phy
    logical(c_int), dimension(:, :), allocatable :: ipeidx_dsl
    real(c_double), dimension(:, :), allocatable :: coeff_gradekin
    real(c_double), dimension(:, :), allocatable :: geofac_grdiv
    real(c_double), dimension(:, :), allocatable :: geofac_rot
    real(c_double), dimension(:, :), allocatable :: c_intp
    integer(c_int), dimension(:, :, :), allocatable :: vertoffset_gradp
    real(c_double), dimension(:, :, :), allocatable :: zdiff_gradp
    real(c_double), dimension(:), allocatable :: cell_center_lat
    real(c_double), dimension(:), allocatable :: cell_center_lon
    real(c_double), dimension(:), allocatable :: edge_center_lat
    real(c_double), dimension(:), allocatable :: edge_center_lon
    real(c_double), dimension(:), allocatable :: primal_normal_x
    real(c_double), dimension(:), allocatable :: primal_normal_y

   !$acc enter data create (vct_a, vct_b, rayleigh_w, tangent_orientation, inverse_primal_edge_lengths, &
   !$acc inv_dual_edge_length, inv_vert_vert_length, edge_areas, f_e, cell_areas, vwind_expl_wgt, &
   !$acc vwind_impl_wgt, scalfac_dd3d, nudgecoeff_e, &
   !$acc hmask_dd3d, bdy_halo_c, mask_prog_halo_c, c_owner_mask, & ! L 191

   !$acc theta_ref_mc, exner_pr, exner_dyn_incr, wgtfac_c, e_bln_c_s, &
   !$acc geofac_div, geofac_grg_x, geofac_grg_y, geofac_n2s, rbf_coeff_1, rbf_coeff_2, &
   !$acc w_now, w_new, vn_now, vn_new, exner_now, exner_new, theta_v_now, theta_v_new, & ! L 213
   !$acc rho_now, rho_new, &

   !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, dual_normal_vert_y, &
   !$acc primal_normal_cell_x, primal_normal_cell_y, primal_normal_vert_x, primal_normal_vert_y, &
   !$acc exner_exfac, exner_ref_mc, wgtfacq_c_dsl, inv_ddqz_z_full, d_exner_dz_ref_ic, & ! L 226
   !$acc ddqz_z_half, theta_ref_ic, d2dexdz2_fac1_mc, d2dexdz2_fac2_mc, rho_ref_me, theta_ref_me, &
   !$acc ddxn_z_full, pg_exdist, ddqz_z_full_e, ddxt_z_full, wgtfac_e, wgtfacq_e, coeff1_dwdz, &
   !$acc coeff2_dwdz, grf_tend_rho, grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, grf_tend_vn, & ! L 246

   !$acc vn_ie, vt, mass_flx_me, mass_flx_ic, vn_traj, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, ddt_w_adv_ntl1, &
   !$acc ddt_w_adv_ntl2, c_lin_e, pos_on_tplane_e_1, pos_on_tplane_e_2, rbf_vec_coeff_e, w_concorr_c, &
   !$acc theta_v_ic, rho_ref_mc, rho_ic, e_flx_avg, ddt_exner_phy, ipeidx_dsl, coeff_gradekin, &
   !$acc geofac_grdiv, geofac_rot, c_intp, vertoffset_gradp, zdiff_gradp, &
   !$acc c2e, e2c, e2v, v2e, v2c, c2v, e2c2v, c2e2c, e2c2e, cell_starts, cell_ends, &
   !$acc vertex_starts, vertex_ends, edge_starts, edge_ends)

   ! Allocate arrays
   allocate(vct_a(num_levels))
   allocate(vct_b(num_levels))
   allocate(rayleigh_w(num_levels))
   allocate(tangent_orientation(num_edges))
   allocate(nudgecoeff_e(num_edges))
   allocate(hmask_dd3d(num_edges))
   allocate(inverse_primal_edge_lengths(num_edges))
   allocate(inv_dual_edge_length(num_edges))
   allocate(inv_vert_vert_length(num_edges))
   allocate(edge_areas(num_edges))
   allocate(f_e(num_edges))
   allocate(cell_areas(num_cells))
   allocate(vwind_expl_wgt(num_cells))
   allocate(vwind_impl_wgt(num_cells))
   allocate(scalfac_dd3d(num_levels))

   allocate(theta_ref_mc(num_cells, num_levels))
   allocate(exner_pr(num_cells, num_levels))
   allocate(exner_dyn_incr(num_cells, num_levels))
   allocate(wgtfac_c(num_cells, num_levels + 1))
   allocate(e_bln_c_s(num_cells, num_c2e))
   allocate(geofac_div(num_cells, num_c2e))
   allocate(geofac_grg_x(num_cells, num_c2ec2o))
   allocate(geofac_grg_y(num_cells, num_c2ec2o))
   allocate(geofac_n2s(num_cells, num_c2ec2o))
   allocate(rbf_coeff_1(num_verts, num_v2e))
   allocate(rbf_coeff_2(num_verts, num_v2e))
   allocate(w_now(num_cells, num_levels + 1))
   allocate(w_new(num_cells, num_levels + 1))
   allocate(vn_now(num_edges, num_levels))
   allocate(vn_new(num_edges, num_levels))
   allocate(exner_now(num_cells, num_levels))
   allocate(exner_new(num_cells, num_levels))
   allocate(theta_v_now(num_cells, num_levels))
   allocate(theta_v_new(num_cells, num_levels))
   allocate(rho_now(num_cells, num_levels))
   allocate(rho_new(num_cells, num_levels))
   allocate(dual_normal_cell_x(num_edges, num_e2c))
   allocate(dual_normal_cell_y(num_edges, num_e2c))
   allocate(dual_normal_vert_x(num_edges, num_e2c2v))
   allocate(dual_normal_vert_y(num_edges, num_e2c2v))
   allocate(primal_normal_cell_x(num_edges, num_e2c))
   allocate(primal_normal_cell_y(num_edges, num_e2c))
   allocate(primal_normal_vert_x(num_edges, num_e2c2v))
   allocate(primal_normal_vert_y(num_edges, num_e2c2v))
   allocate(exner_exfac(num_cells, num_levels))
   allocate(exner_ref_mc(num_cells, num_levels))
   allocate(wgtfacq_c_dsl(num_cells, num_levels))
   allocate(inv_ddqz_z_full(num_cells, num_levels))
   allocate(d_exner_dz_ref_ic(num_cells, num_levels))
   allocate(ddqz_z_half(num_cells, num_levels))
   allocate(theta_ref_ic(num_cells, num_levels))
   allocate(d2dexdz2_fac1_mc(num_cells, num_levels))
   allocate(d2dexdz2_fac2_mc(num_cells, num_levels))
   allocate(rho_ref_me(num_edges, num_levels))
   allocate(theta_ref_me(num_edges, num_levels))
   allocate(ddxn_z_full(num_edges, num_levels))
   allocate(pg_exdist(num_edges, num_levels))
   allocate(ddqz_z_full_e(num_edges, num_levels))
   allocate(ddxt_z_full(num_edges, num_levels))
   allocate(wgtfac_e(num_edges, num_levels))
   allocate(wgtfacq_e(num_edges, num_levels))
   allocate(coeff1_dwdz(num_cells, num_levels))
   allocate(coeff2_dwdz(num_cells, num_levels))
   allocate(grf_tend_rho(num_cells, num_levels))
   allocate(grf_tend_thv(num_cells, num_levels))
   allocate(grf_tend_w(num_cells, num_levels + 1))
   allocate(mass_fl_e(num_edges, num_levels + 1))
   allocate(ddt_vn_phy(num_edges, num_levels))
   allocate(grf_tend_vn(num_edges, num_levels))
   allocate(vn_ie(num_edges, num_levels + 1))
   allocate(vt(num_edges, num_levels))
   allocate(mass_flx_me(num_edges, num_levels))
   allocate(mass_flx_ic(num_cells, num_levels))
   allocate(vn_traj(num_edges, num_levels))
   allocate(ddt_vn_apc_ntl1(num_edges, num_levels))
   allocate(ddt_vn_apc_ntl2(num_edges, num_levels))
   allocate(ddt_w_adv_ntl1(num_cells, num_levels))
   allocate(ddt_w_adv_ntl2(num_cells, num_levels))
   allocate(c_lin_e(num_edges, num_e2c))
   allocate(pos_on_tplane_e_1(num_edges, num_e2c))
   allocate(pos_on_tplane_e_2(num_edges, num_e2c))
   allocate(rbf_vec_coeff_e(num_edges, num_e2c2v))
   allocate(ipeidx_dsl(num_edges, num_levels))
   allocate(mask_prog_halo_c(num_cells))
   allocate(c_owner_mask(num_cells))
   allocate(bdy_halo_c(num_cells))
   allocate(coeff_gradekin(num_edges, num_e2c))
   allocate(geofac_grdiv(num_edges, num_e2c2eo))
   allocate(e_flx_avg(num_edges, num_e2c2eo))
   allocate(geofac_rot(num_verts, num_v2e))
   allocate(c_intp(num_verts, num_e2c))
   allocate(w_concorr_c(num_cells, num_levels + 1))
   allocate(theta_v_ic(num_cells, num_levels + 1))
   allocate(rho_ref_mc(num_cells, num_levels))
   allocate(rho_ic(num_cells, num_levels + 1))
   allocate(ddt_exner_phy(num_cells, num_levels))

   ! 3d arrays
   allocate(vertoffset_gradp(num_edges, num_e2c, num_levels))
   allocate(zdiff_gradp(num_edges, num_e2c, num_levels))

   ! Connectivity arrays allocation
   allocate(c2e(num_cells, num_c2e))
   allocate(e2c(num_edges, num_e2c))
   allocate(e2v(num_edges, num_e2c))
   allocate(v2e(num_verts, num_v2e))
   allocate(v2c(num_verts, num_v2e))
   allocate(c2v(num_cells, num_c2e))
   allocate(e2c2v(num_edges, num_e2c2v))
   allocate(c2e2c(num_cells, num_c2e))
   allocate(e2c2e(num_edges, num_e2c2v))
   allocate(cell_starts(14))
   allocate(cell_ends(14))
   allocate(vertex_starts(13))
   allocate(vertex_ends(13))
   allocate(edge_starts(24))
   allocate(edge_ends(24))

   allocate(cell_center_lat(num_cells))
   allocate(cell_center_lon(num_cells))
   allocate(edge_center_lat(num_edges))
   allocate(edge_center_lon(num_edges))
   allocate(primal_normal_x(num_edges))
   allocate(primal_normal_y(num_edges))

    ! Initialize the arrays with the values from the Python arrays

    ! cell_starts initialization
    cell_starts = (/ 20896, 20896, 20896, 20896, -1, -1, -1, -1, 4104, 0, 850, 1688, 2511, 3316 /)

    ! cell_ends initialization
    cell_ends = (/ 20896, 20896, 20896, 20896, 20896, 20896, 20896, 20896, 20896, 850, 1688, 2511, 3316, 4104 /)

    ! edge_starts initialization
    edge_starts = (/ 31558, 31558, 31558, 31558, 31558, -1, -1, -1, -1, -1, -1, -1, -1, 6176, 0, 428, 1278, 1700, &
                    2538, 2954, 3777, 4184, 4989, 5387 /)

    ! edge_ends initialization
    edge_ends = (/ 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, 31558, &
                    428, 1278, 1700, 2538, 2954, 3777, 4184, 4989, 5387, 6176 /)

    ! vertex_starts initialization
    vertex_starts = (/ 10663, 10663, 10663, -1, -1, -1, -1, 2071, 0, 428, 850, 1266, 1673 /)

    ! vertex_ends initialization
    vertex_ends = (/ 10663, 10663, 10663, 10663, 10663, 10663, 10663, 10663, 428, 850, 1266, 1673, 2071 /)

    ! The vct_a array must be set to the same values as the ones in ICON.
    ! It represents the reference heights of vertical levels in meters, and many key vertical indices are derived from it.
    ! Accurate computation of bounds relies on using the same vct_a values as those in ICON.

   vct_a = (/ 23000.0, 20267.579776144084, 18808.316862872744, 17645.20947843258, &
          16649.573524156993, 15767.598849006221, 14970.17804229092, 14239.283693028447, &
          13562.75820630252, 12931.905058984285, 12340.22824884565, 11782.711681133735, &
          11255.378878851721, 10755.009592797565, 10278.949589989745, 9824.978499468381, &
          9391.215299185755, 8976.0490382992, 8578.086969013575, 8196.11499008041, &
          7829.066987285794, 7476.0007272129105, 7136.078660578203, 6808.552460051288, &
          6492.750437928688, 6188.067212417723, 5893.955149722682, 5609.917223280037, &
          5335.501014932097, 5070.293644636082, 4813.9174616536775, 4566.0263653241045, &
          4326.302650484456, 4094.4542934937413, 3870.212611174545, 3653.3302379273473, &
          3443.5793766239703, 3240.750287267658, 3044.649984289428, 2855.101119099911, &
          2671.9410294241347, 2495.0209412555105, 2324.2053131841913, 2159.371316580089, &
          2000.4084488403732, 1847.2182808658658, 1699.7143443769428, 1557.8221699649202, &
          1421.479493379662, 1290.63665617212, 1165.2572384824416, 1045.3189781024735, &
          930.8150535208842, 821.7558437251436, 718.1713313259359, 620.1144009054101, &
          527.6654250683475, 440.93877255014786, 360.09231087410603, 285.34182080238656, &
          216.98400030452174, 155.43579225710877, 101.30847966961008, 55.56948426298202, &
          20.00000000000001, 0.0 /)

   vct_b = vct_a

   ! Fill arrays with random numbers
   call fill_random_1d(rayleigh_w, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(tangent_orientation, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(nudgecoeff_e, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(hmask_dd3d, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inverse_primal_edge_lengths, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inv_dual_edge_length, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inv_vert_vert_length, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(edge_areas, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(f_e, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(cell_areas, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(vwind_expl_wgt, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(vwind_impl_wgt, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(scalfac_dd3d, 0.0_c_double, 1.0_c_double)
   call fill_random_1d_bool(mask_prog_halo_c)
   call fill_random_1d_bool(c_owner_mask)
   call fill_random_1d_bool(bdy_halo_c)


   call fill_random_2d(theta_ref_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner_pr, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner_dyn_incr, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(wgtfac_c, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(e_bln_c_s, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_div, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grg_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grg_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_n2s, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_coeff_1, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_coeff_2, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(w_now, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(w_new, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(vn_now, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(vn_new, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner_now, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner_new, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(theta_v_now, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(theta_v_new, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rho_now, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rho_new, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dual_normal_cell_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dual_normal_cell_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dual_normal_vert_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dual_normal_vert_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(primal_normal_cell_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(primal_normal_cell_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(primal_normal_vert_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(primal_normal_vert_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner_exfac, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner_ref_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(wgtfacq_c_dsl, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(inv_ddqz_z_full, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(d_exner_dz_ref_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddqz_z_half, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(theta_ref_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(d2dexdz2_fac1_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(d2dexdz2_fac2_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rho_ref_me, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(theta_ref_me, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddxn_z_full, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(pg_exdist, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddqz_z_full_e, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddxt_z_full, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(wgtfac_e, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(wgtfacq_e, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(coeff1_dwdz, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(coeff2_dwdz, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(grf_tend_rho, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(grf_tend_thv, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(grf_tend_w, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(mass_fl_e, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddt_vn_phy, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(grf_tend_vn, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(vn_ie, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(vt, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(mass_flx_me, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(mass_flx_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(vn_traj, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddt_vn_apc_ntl1, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddt_vn_apc_ntl2, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddt_w_adv_ntl1, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddt_w_adv_ntl2, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(c_lin_e, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(pos_on_tplane_e_1, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(pos_on_tplane_e_2, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_vec_coeff_e, 0.0_c_double, 1.0_c_double)
   call fill_random_2d_bool(ipeidx_dsl)
   call fill_random_2d(coeff_gradekin, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grdiv, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(e_flx_avg, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_rot, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(c_intp, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(w_concorr_c, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(theta_v_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rho_ref_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rho_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(ddt_exner_phy, 0.0_c_double, 1.0_c_double)

   ! For 3D arrays
   call fill_random_3d(zdiff_gradp, 0.0_c_double, 1.0_c_double)
   call fill_random_3d_int(vertoffset_gradp, 0, 1)

   ! Fill connectivities with random values
   call fill_random_2d_int(c2e, 0, num_edges)
   call fill_random_2d_int(e2c, 0, num_cells)
   call fill_random_2d_int(e2v, 0, num_verts)
   call fill_random_2d_int(v2e, 0, num_edges)
   call fill_random_2d_int(v2c, 0, num_cells)
   call fill_random_2d_int(c2v, 0, num_verts)
   call fill_random_2d_int(e2c2v, 0, num_verts)
   call fill_random_2d_int(c2e2c, 0, num_cells)
   call fill_random_2d_int(e2c2e, 0, num_edges)

   !$acc data copyin (vct_a, vct_b, rayleigh_w, tangent_orientation, inverse_primal_edge_lengths, &
   !$acc inv_dual_edge_length, inv_vert_vert_length, edge_areas, f_e, cell_areas, vwind_expl_wgt, &
   !$acc vwind_impl_wgt, scalfac_dd3d, nudgecoeff_e, &
   !$acc hmask_dd3d, bdy_halo_c, mask_prog_halo_c, c_owner_mask, & ! L 191

   !$acc theta_ref_mc, exner_pr, exner_dyn_incr, wgtfac_c, e_bln_c_s, &
   !$acc geofac_div, geofac_grg_x, geofac_grg_y, geofac_n2s, rbf_coeff_1, rbf_coeff_2, &
   !$acc w_now, w_new, vn_now, vn_new, exner_now, exner_new, theta_v_now, theta_v_new, & ! L 213
   !$acc rho_now, rho_new, &

   !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, dual_normal_vert_y, &
   !$acc primal_normal_cell_x, primal_normal_cell_y, primal_normal_vert_x, primal_normal_vert_y, &
   !$acc exner_exfac, exner_ref_mc, wgtfacq_c_dsl, inv_ddqz_z_full, d_exner_dz_ref_ic, & ! L 226
   !$acc ddqz_z_half, theta_ref_ic, d2dexdz2_fac1_mc, d2dexdz2_fac2_mc, rho_ref_me, theta_ref_me, &
   !$acc ddxn_z_full, pg_exdist, ddqz_z_full_e, ddxt_z_full, wgtfac_e, wgtfacq_e, coeff1_dwdz, &
   !$acc coeff2_dwdz, grf_tend_rho, grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, grf_tend_vn, & ! L 246

   !$acc vn_ie, vt, mass_flx_me, mass_flx_ic, vn_traj, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, ddt_w_adv_ntl1, &
   !$acc ddt_w_adv_ntl2, c_lin_e, pos_on_tplane_e_1, pos_on_tplane_e_2, rbf_vec_coeff_e, w_concorr_c, &
   !$acc theta_v_ic, rho_ref_mc, rho_ic, e_flx_avg, ddt_exner_phy, ipeidx_dsl, coeff_gradekin, &
   !$acc geofac_grdiv, geofac_rot, c_intp, vertoffset_gradp, zdiff_gradp, &
   !$acc c2e, e2c, e2v, v2e, v2c, c2v, e2c2v, c2e2c, e2c2e, cell_starts, cell_ends, &
   !$acc vertex_starts, vertex_ends, edge_starts, edge_ends)

   call grid_init( &
    cell_starts, cell_ends, &
    vertex_starts, vertex_ends, &
    edge_starts, edge_ends, &
    c2e, e2c, c2e2c, e2c2e, &
    e2v, v2e, v2c, c2v, e2c2v, &
    global_root, global_level, &
    num_verts, num_cells, num_edges, &
    vertical_size, limited_area, rc)

   if (rc /= 0) then
       print *, "Error in solve_nh_init"
       call exit(1)
   end if

   ! Call solve_nh_init
   call solve_nh_init( &
        vct_a=vct_a, &
        vct_b=vct_b, &
        cell_areas=cell_areas, &
        primal_normal_cell_x=primal_normal_cell_x, &
        primal_normal_cell_y=primal_normal_cell_y, &
        dual_normal_cell_x=dual_normal_cell_x, &
        dual_normal_cell_y=dual_normal_cell_y, &
        edge_areas=edge_areas, &
        tangent_orientation=tangent_orientation, &
        inverse_primal_edge_lengths=inverse_primal_edge_lengths, &
        inverse_dual_edge_lengths=inv_dual_edge_length, &
        inverse_vertex_vertex_lengths=inv_vert_vert_length, &
        primal_normal_vert_x=primal_normal_vert_x, &
        primal_normal_vert_y=primal_normal_vert_y, &
        dual_normal_vert_x=dual_normal_vert_x, &
        dual_normal_vert_y=dual_normal_vert_y, &
        f_e=f_e, &
        c_lin_e=c_lin_e, &
        c_intp=c_intp, &
        e_flx_avg=e_flx_avg, &
        geofac_grdiv=geofac_grdiv, &
        geofac_rot=geofac_rot, &
        pos_on_tplane_e_1=pos_on_tplane_e_1, &
        pos_on_tplane_e_2=pos_on_tplane_e_2, &
        rbf_vec_coeff_e=rbf_vec_coeff_e, &
        e_bln_c_s=e_bln_c_s, &
        rbf_coeff_1=rbf_coeff_1, &
        rbf_coeff_2=rbf_coeff_2, &
        geofac_div=geofac_div, &
        geofac_n2s=geofac_n2s, &
        geofac_grg_x=geofac_grg_x, &
        geofac_grg_y=geofac_grg_y, &
        nudgecoeff_e=nudgecoeff_e, &
        bdy_halo_c=bdy_halo_c, &
        mask_prog_halo_c=mask_prog_halo_c, &
        rayleigh_w=rayleigh_w, &
        exner_exfac=exner_exfac, &
        exner_ref_mc=exner_ref_mc, &
        wgtfac_c=wgtfac_c, &
        wgtfacq_c=wgtfacq_c_dsl, &
        inv_ddqz_z_full=inv_ddqz_z_full, &
        rho_ref_mc=rho_ref_mc, &
        theta_ref_mc=theta_ref_mc, &
        vwind_expl_wgt=vwind_expl_wgt, &
        d_exner_dz_ref_ic=d_exner_dz_ref_ic, &
        ddqz_z_half=ddqz_z_half, &
        theta_ref_ic=theta_ref_ic, &
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc, &
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc, &
        rho_ref_me=rho_ref_me, &
        theta_ref_me=theta_ref_me, &
        ddxn_z_full=ddxn_z_full, &
        zdiff_gradp=zdiff_gradp, &
        vertoffset_gradp=vertoffset_gradp, &
        ipeidx_dsl=ipeidx_dsl, &
        pg_exdist=pg_exdist, &
        ddqz_z_full_e=ddqz_z_full_e, &
        ddxt_z_full=ddxt_z_full, &
        wgtfac_e=wgtfac_e, &
        wgtfacq_e=wgtfacq_e, &
        vwind_impl_wgt=vwind_impl_wgt, &
        hmask_dd3d=hmask_dd3d, &
        scalfac_dd3d=scalfac_dd3d, &
        coeff1_dwdz=coeff1_dwdz, &
        coeff2_dwdz=coeff2_dwdz, &
        coeff_gradekin=coeff_gradekin, &
        c_owner_mask=c_owner_mask, &
        cell_center_lat=cell_center_lat, &
        cell_center_lon=cell_center_lon, &
        edge_center_lat=edge_center_lat, &
        edge_center_lon=edge_center_lon, &
        primal_normal_x=primal_normal_x, &
        primal_normal_y=primal_normal_y, &
        rayleigh_damping_height=rayleigh_damping_height, &
        itime_scheme=itime_scheme, &
        iadv_rhotheta=iadv_rhotheta, &
        igradp_method=igradp_method, &
        ndyn_substeps=ndyn_substeps, &
        rayleigh_type=rayleigh_type, &
        rayleigh_coeff=rayleigh_coeff, &
        divdamp_order=divdamp_order, &
        is_iau_active=is_iau_active, &
        iau_wgt_dyn=iau_wgt_dyn, &
        divdamp_type=divdamp_type, &
        divdamp_trans_start=divdamp_trans_start, &
        divdamp_trans_end=divdamp_trans_end, &
        l_vert_nested=l_vert_nested, &
        rhotheta_offctr=rhotheta_offctr, &
        veladv_offctr=veladv_offctr, &
        max_nudging_coeff=max_nudging_coeff, &
        divdamp_fac=divdamp_fac, &
        divdamp_fac2=divdamp_fac2, &
        divdamp_fac3=divdamp_fac3, &
        divdamp_fac4=divdamp_fac4, &
        divdamp_z=divdamp_z, &
        divdamp_z2=divdamp_z2, &
        divdamp_z3=divdamp_z3, &
        divdamp_z4=divdamp_z4, &
        lowest_layer_thickness=lowest_layer_thickness, &
        model_top_height=model_top_height, &
        stretch_factor=stretch_factor, &
        nflat_gradp=nflat_gradp, &
        num_levels=num_levels, &
        rc=rc)

   if (rc /= 0) then
       print *, "Error in solve_nh_init"
       call exit(1)
   end if



  call solve_nh_run(rho_now, rho_new, exner_now, exner_new, w_now, w_new, &
                    theta_v_now, theta_v_new, vn_now, vn_new, &
                    w_concorr_c, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, &
                    ddt_w_adv_ntl1, ddt_w_adv_ntl2, theta_v_ic, rho_ic, &
                    exner_pr, exner_dyn_incr, ddt_exner_phy, grf_tend_rho, &
                    grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, &
                    grf_tend_vn, vn_ie, vt, mass_flx_me, mass_flx_ic, &
                    vn_traj, dtime, lprep_adv, clean_mflx, recompute, linit, &
                    divdamp_fac_o2, ndyn_substeps, idyn_timestep, nnow, nnew, rc)

  if (rc /= 0) then
      print *, "Error in solve_nh_run"
      call exit(1)
  end if


   !$acc update host (vct_a, vct_b, rayleigh_w, tangent_orientation, inverse_primal_edge_lengths, &
   !$acc inv_dual_edge_length, inv_vert_vert_length, edge_areas, f_e, cell_areas, vwind_expl_wgt, &
   !$acc vwind_impl_wgt, scalfac_dd3d, nudgecoeff_e, &
   !$acc hmask_dd3d, bdy_halo_c, mask_prog_halo_c, c_owner_mask, & ! L 191

   !$acc theta_ref_mc, exner_pr, exner_dyn_incr, wgtfac_c, e_bln_c_s, &
   !$acc geofac_div, geofac_grg_x, geofac_grg_y, geofac_n2s, rbf_coeff_1, rbf_coeff_2, &
   !$acc w_now, w_new, vn_now, vn_new, exner_now, exner_new, theta_v_now, theta_v_new, & ! L 213
   !$acc rho_now, rho_new, &

   !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, dual_normal_vert_y, &
   !$acc primal_normal_cell_x, primal_normal_cell_y, primal_normal_vert_x, primal_normal_vert_y, &
   !$acc exner_exfac, exner_ref_mc, wgtfacq_c_dsl, inv_ddqz_z_full, d_exner_dz_ref_ic, & ! L 226
   !$acc ddqz_z_half, theta_ref_ic, d2dexdz2_fac1_mc, d2dexdz2_fac2_mc, rho_ref_me, theta_ref_me, &
   !$acc ddxn_z_full, pg_exdist, ddqz_z_full_e, ddxt_z_full, wgtfac_e, wgtfacq_e, coeff1_dwdz, &
   !$acc coeff2_dwdz, grf_tend_rho, grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, grf_tend_vn, & ! L 246

   !$acc vn_ie, vt, mass_flx_me, mass_flx_ic, vn_traj, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, ddt_w_adv_ntl1, &
   !$acc ddt_w_adv_ntl2, c_lin_e, pos_on_tplane_e_1, pos_on_tplane_e_2, rbf_vec_coeff_e, w_concorr_c, &
   !$acc theta_v_ic, rho_ref_mc, rho_ic, e_flx_avg, ddt_exner_phy, ipeidx_dsl, coeff_gradekin, &
   !$acc geofac_grdiv, geofac_rot, c_intp, vertoffset_gradp, zdiff_gradp, &
   !$acc c2e, e2c, e2v, v2e, v2c, c2v, e2c2v, c2e2c, e2c2e, cell_starts, cell_ends, &
   !$acc vertex_starts, vertex_ends, edge_starts, edge_ends)

   print *, "passed"

  !$acc end data
  !$acc exit data delete (vct_a, vct_b, rayleigh_w, tangent_orientation, inverse_primal_edge_lengths, &
  !$acc inv_dual_edge_length, inv_vert_vert_length, edge_areas, f_e, cell_areas, vwind_expl_wgt, &
  !$acc vwind_impl_wgt, scalfac_dd3d, nudgecoeff_e, &
  !$acc hmask_dd3d, bdy_halo_c, mask_prog_halo_c, c_owner_mask, & ! L 191

  !$acc theta_ref_mc, exner_pr, exner_dyn_incr, wgtfac_c, e_bln_c_s, &
  !$acc geofac_div, geofac_grg_x, geofac_grg_y, geofac_n2s, rbf_coeff_1, rbf_coeff_2, &
  !$acc w_now, w_new, vn_now, vn_new, exner_now, exner_new, theta_v_now, theta_v_new, & ! L 213
  !$acc rho_now, rho_new, &

  !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, dual_normal_vert_y, &
  !$acc primal_normal_cell_x, primal_normal_cell_y, primal_normal_vert_x, primal_normal_vert_y, &
  !$acc exner_exfac, exner_ref_mc, wgtfacq_c_dsl, inv_ddqz_z_full, d_exner_dz_ref_ic, & ! L 226
  !$acc ddqz_z_half, theta_ref_ic, d2dexdz2_fac1_mc, d2dexdz2_fac2_mc, rho_ref_me, theta_ref_me, &
  !$acc ddxn_z_full, pg_exdist, ddqz_z_full_e, ddxt_z_full, wgtfac_e, wgtfacq_e, coeff1_dwdz, &
  !$acc coeff2_dwdz, grf_tend_rho, grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, grf_tend_vn, & ! L 246

  !$acc vn_ie, vt, mass_flx_me, mass_flx_ic, vn_traj, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, ddt_w_adv_ntl1, &
  !$acc ddt_w_adv_ntl2, c_lin_e, pos_on_tplane_e_1, pos_on_tplane_e_2, rbf_vec_coeff_e, w_concorr_c, &
  !$acc theta_v_ic, rho_ref_mc, rho_ic, e_flx_avg, ddt_exner_phy, ipeidx_dsl, coeff_gradekin, &
  !$acc geofac_grdiv, geofac_rot, c_intp, vertoffset_gradp, zdiff_gradp, &
  !$acc c2e, e2c, e2v, v2e, v2c, c2v, e2c2v, c2e2c, e2c2e, cell_starts, cell_ends, &
  !$acc vertex_starts, vertex_ends, edge_starts, edge_ends)


end program solve_nh_simulation
