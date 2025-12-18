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
      use mo_kind, only: wp
      implicit none

      real(wp), intent(inout) :: array(:)
      real(wp), intent(in) :: low, high
      integer :: i
      real(wp) :: rnd

      do i = 1, size(array)
         call random_number(rnd)
         array(i) = low + rnd*(high - low)
      end do
   end subroutine fill_random_1d

   subroutine fill_random_2d(array, low, high)
      use mo_kind, only: wp
      implicit none

      real(wp), intent(inout) :: array(:, :)
      real(wp), intent(in) :: low, high
      integer :: i, j
      real(wp) :: rnd

      do i = 1, size(array, 1)
         do j = 1, size(array, 2)
            call random_number(rnd)
            array(i, j) = low + rnd*(high - low)
         end do
      end do
   end subroutine fill_random_2d

    subroutine fill_random_3d_int(array, low, high)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none

    integer(c_int), intent(inout) :: array(:, :, :)
    integer(c_int), intent(in) :: low, high
    integer :: i, j, k
    real :: rnd  ! real number between 0 and 1

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
    use mo_kind, only: wp
    implicit none

    real(wp), intent(inout) :: array(:, :, :)
    real(wp), intent(in) :: low, high
    integer :: i, j, k
    real :: rnd  ! real number between 0 and 1

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
    real :: rnd  ! real number between 0 and 1

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
end module random_utils

program diffusion_simulation
   use, intrinsic :: iso_c_binding, only: c_int
   use mo_kind, only: wp
   use random_utils, only: fill_random_1d, fill_random_2d, fill_random_2d_bool, fill_random_3d_int, fill_random_3d
   use diffusion_plugin
   implicit none

   integer(c_int) :: rc
   integer(c_int) :: n

   ! Constants and types
   integer(c_int), parameter :: num_cells = 20896
   integer(c_int), parameter :: num_edges = 31558
   integer(c_int), parameter :: num_vertices = 10663
   integer(c_int), parameter :: num_levels = 65
   integer(c_int), parameter :: num_c2ec2o = 4
   integer(c_int), parameter :: num_v2e = 6
   integer(c_int), parameter :: num_c2e = 3
   integer(c_int), parameter :: num_e2c2v = 4
   integer(c_int), parameter :: num_c2e2c = 3
   integer(c_int), parameter :: num_e2c = 2
   real(wp), parameter :: mean_cell_area = 24907282236.708576
   integer(c_int), parameter :: ndyn_substeps = 2
   real(wp), parameter :: dtime = 10.0
   real(wp), parameter :: rayleigh_damping_height = 12500.0
   integer(c_int), parameter :: nflatlev = 30
   integer(c_int), parameter :: nflat_gradp = 59
   integer(c_int), parameter :: diffusion_type = 5 ! Assuming DiffusionType.SMAGORINSKY_4TH_ORDER is represented by 5
   logical(c_int), parameter :: hdiff_w = .true.
   logical(c_int), parameter :: hdiff_vn = .true.
   logical(c_int), parameter :: zdiffu_t = .true. ! this runs stencil 15 which uses the boolean mask
   integer(c_int), parameter :: type_t_diffu = 2
   integer(c_int), parameter :: type_vn_diffu = 1
   real(wp), parameter :: hdiff_efdt_ratio = 24.0
   real(wp), parameter :: smagorinski_scaling_factor = 0.025
   logical(c_int), parameter :: hdiff_temp = .true.
   logical(c_int), parameter :: linit = .false.
   real(wp), parameter :: denom_diffu_v = 150.0
   real(wp), parameter :: thslp_zdiffu = 0.02
   real(wp), parameter :: thhgtd_zdiffu = 125.0
   integer(c_int), parameter :: itype_sher = 2
   real(wp), parameter :: nudge_max_coeff = 0.075

   ! Declaring arrays for diffusion_init and diffusion_run
   real(wp), dimension(:), allocatable :: vct_a
   real(wp), dimension(:), allocatable :: nudgecoeff_e
   real(wp), dimension(:), allocatable :: tangent_orientation
   real(wp), dimension(:), allocatable :: inverse_primal_edge_lengths
   real(wp), dimension(:), allocatable :: inv_dual_edge_length
   real(wp), dimension(:), allocatable :: inv_vert_vert_length
   real(wp), dimension(:), allocatable :: edge_areas
   real(wp), dimension(:), allocatable :: f_e
   real(wp), dimension(:), allocatable :: cell_areas

   real(wp), dimension(:, :), allocatable :: theta_ref_mc
   real(wp), dimension(:, :), allocatable :: wgtfac_c
   real(wp), dimension(:, :), allocatable :: e_bln_c_s
   real(wp), dimension(:, :), allocatable :: geofac_div
   real(wp), dimension(:, :), allocatable :: geofac_grg_x
   real(wp), dimension(:, :), allocatable :: geofac_grg_y
   real(wp), dimension(:, :), allocatable :: geofac_n2s
   real(wp), dimension(:, :), allocatable :: rbf_coeff_1
   real(wp), dimension(:, :), allocatable :: rbf_coeff_2
   real(wp), dimension(:, :), allocatable :: dwdx
   real(wp), dimension(:, :), allocatable :: dwdy
   real(wp), dimension(:, :), allocatable :: hdef_ic
   real(wp), dimension(:, :), allocatable :: div_ic
   real(wp), dimension(:, :), allocatable :: w
   real(wp), dimension(:, :), allocatable :: vn
   real(wp), dimension(:, :), allocatable :: exner
   real(wp), dimension(:, :), allocatable :: theta_v
   real(wp), dimension(:, :), allocatable :: rho
   real(wp), dimension(:, :), allocatable :: dual_normal_cell_x
   real(wp), dimension(:, :), allocatable :: dual_normal_cell_y
   real(wp), dimension(:, :), allocatable :: dual_normal_vert_x
   real(wp), dimension(:, :), allocatable :: dual_normal_vert_y
   real(wp), dimension(:, :), allocatable :: primal_normal_cell_x
   real(wp), dimension(:, :), allocatable :: primal_normal_cell_y
   real(wp), dimension(:, :), allocatable :: primal_normal_vert_x
   real(wp), dimension(:, :), allocatable :: primal_normal_vert_y
   real(wp), dimension(:, :), allocatable :: zd_diffcoef
   logical(c_int), dimension(:, :), allocatable :: mask_hdiff

   integer(c_int), dimension(:, :, :), allocatable :: zd_vertoffset
   real(wp), dimension(:, :, :), allocatable :: zd_intcoef

    !$acc enter data create(vct_a, theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, &
    !$acc geofac_grg_x, geofac_grg_y, geofac_n2s, nudgecoeff_e, rbf_coeff_1, &
    !$acc rbf_coeff_2, dwdx, dwdy, hdef_ic, div_ic, w, vn, exner, theta_v, rho, &
    !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, &
    !$acc dual_normal_vert_y, primal_normal_cell_x, primal_normal_cell_y, &
    !$acc primal_normal_vert_x, primal_normal_vert_y, tangent_orientation, &
    !$acc inverse_primal_edge_lengths, inv_dual_edge_length, inv_vert_vert_length, &
    !$acc edge_areas, cell_areas, f_e, zd_diffcoef, zd_vertoffset, zd_intcoef, &
    !$acc mask_hdiff)

   ! allocating arrays
   allocate(zd_diffcoef(num_cells, num_levels))
   allocate(zd_vertoffset(num_cells, num_c2e2c, num_levels))
   allocate(zd_intcoef(num_cells, num_c2e2c, num_levels))
   allocate(mask_hdiff(num_cells, num_levels))
   allocate (vct_a(num_levels))
   allocate (theta_ref_mc(num_cells, num_levels))
   allocate (wgtfac_c(num_cells, num_levels + 1))
   allocate (e_bln_c_s(num_cells, num_c2e))
   allocate (geofac_div(num_cells, num_c2e))
   allocate (geofac_grg_x(num_cells, num_c2ec2o))
   allocate (geofac_grg_y(num_cells, num_c2ec2o))
   allocate (geofac_n2s(num_cells, num_c2ec2o))
   allocate (nudgecoeff_e(num_edges))
   allocate (rbf_coeff_1(num_vertices, num_v2e))
   allocate (rbf_coeff_2(num_vertices, num_v2e))
   allocate (dwdx(num_cells, num_levels))
   allocate (dwdy(num_cells, num_levels))
   allocate (hdef_ic(num_cells, num_levels + 1))
   allocate (div_ic(num_cells, num_levels + 1))
   allocate (w(num_cells, num_levels + 1))
   allocate (vn(num_edges, num_levels))
   allocate (exner(num_cells, num_levels))
   allocate (theta_v(num_cells, num_levels))
   allocate (rho(num_cells, num_levels))
   allocate (dual_normal_cell_x(num_edges, num_e2c))
   allocate (dual_normal_cell_y(num_edges, num_e2c))
   allocate (dual_normal_vert_x(num_edges, num_e2c2v))
   allocate (dual_normal_vert_y(num_edges, num_e2c2v))
   allocate (primal_normal_cell_x(num_edges, num_e2c))
   allocate (primal_normal_cell_y(num_edges, num_e2c))
   allocate (primal_normal_vert_x(num_edges, num_e2c))
   allocate (primal_normal_vert_y(num_edges, num_e2c))
   allocate (tangent_orientation(num_edges))
   allocate (inverse_primal_edge_lengths(num_edges))
   allocate (inv_dual_edge_length(num_edges))
   allocate (inv_vert_vert_length(num_edges))
   allocate (edge_areas(num_edges))
   allocate (f_e(num_edges))
   allocate (cell_areas(num_cells))

   ! Fill arrays with random numbers
   ! For 1D arrays
   call fill_random_1d(vct_a, 0.0_wp, 75000.0_wp) ! needs to be above 12500 damping height restriction
   call fill_random_1d(nudgecoeff_e, 0.0_wp, 1.0_wp)
   call fill_random_1d(tangent_orientation, 0.0_wp, 1.0_wp)
   call fill_random_1d(inverse_primal_edge_lengths, 0.0_wp, 1.0_wp)
   call fill_random_1d(inv_dual_edge_length, 0.0_wp, 1.0_wp)
   call fill_random_1d(inv_vert_vert_length, 0.0_wp, 1.0_wp)
   call fill_random_1d(edge_areas, 0.0_wp, 1.0_wp)
   call fill_random_1d(f_e, 0.0_wp, 1.0_wp)
   call fill_random_1d(cell_areas, 0.0_wp, 1.0_wp)

   ! For 2D arrays
   call fill_random_2d(theta_ref_mc, 0.0_wp, 1.0_wp)
   call fill_random_2d(wgtfac_c, 0.0_wp, 1.0_wp)
   call fill_random_2d(geofac_grg_x, 0.0_wp, 1.0_wp)
   call fill_random_2d(geofac_grg_y, 0.0_wp, 1.0_wp)
   call fill_random_2d(geofac_n2s, 0.0_wp, 1.0_wp)
   call fill_random_2d(rbf_coeff_1, 0.0_wp, 1.0_wp)
   call fill_random_2d(rbf_coeff_2, 0.0_wp, 1.0_wp)
   call fill_random_2d(dwdx, 0.0_wp, 1.0_wp)
   call fill_random_2d(dwdy, 0.0_wp, 1.0_wp)
   call fill_random_2d(hdef_ic, 0.0_wp, 1.0_wp)
   call fill_random_2d(div_ic, 0.0_wp, 1.0_wp)
   call fill_random_2d(w, 0.0_wp, 1.0_wp)
   call fill_random_2d(vn, 0.0_wp, 1.0_wp)
   call fill_random_2d(exner, 0.0_wp, 1.0_wp)
   call fill_random_2d(theta_v, 0.0_wp, 1.0_wp)
   call fill_random_2d(rho, 0.0_wp, 1.0_wp)
   call fill_random_2d(zd_diffcoef, 0.0_wp, 1.0_wp)
   call fill_random_2d_bool(mask_hdiff)

   call fill_random_2d(e_bln_c_s, 0.0_wp, 1.0_wp)
   call fill_random_2d(geofac_div, 0.0_wp, 1.0_wp)
   call fill_random_2d(dual_normal_vert_x, 0.0_wp, 1.0_wp)
   call fill_random_2d(dual_normal_vert_y, 0.0_wp, 1.0_wp)
   call fill_random_2d(primal_normal_vert_x, 0.0_wp, 1.0_wp)
   call fill_random_2d(primal_normal_vert_y, 0.0_wp, 1.0_wp)
   call fill_random_2d(dual_normal_cell_x, 0.0_wp, 1.0_wp)
   call fill_random_2d(dual_normal_cell_y, 0.0_wp, 1.0_wp)
   call fill_random_2d(primal_normal_cell_x, 0.0_wp, 1.0_wp)
   call fill_random_2d(primal_normal_cell_y, 0.0_wp, 1.0_wp)

   ! For 3D arrays
   call fill_random_3d(zd_intcoef, 0.0_wp, 1.0_wp)
   call fill_random_3d_int(zd_vertoffset, 0, 1)


    !$acc data copyin(vct_a, theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, &
    !$acc geofac_grg_x, geofac_grg_y, geofac_n2s, nudgecoeff_e, rbf_coeff_1, &
    !$acc rbf_coeff_2, dwdx, dwdy, hdef_ic, div_ic, w, vn, exner, theta_v, rho, &
    !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, &
    !$acc dual_normal_vert_y, primal_normal_cell_x, primal_normal_cell_y, &
    !$acc primal_normal_vert_x, primal_normal_vert_y, tangent_orientation, &
    !$acc inverse_primal_edge_lengths, inv_dual_edge_length, inv_vert_vert_length, &
    !$acc edge_areas, cell_areas, f_e, zd_diffcoef, zd_vertoffset, zd_intcoef, &
    !$acc mask_hdiff)

   ! Call diffusion_init
   call diffusion_init(vct_a, &
                      theta_ref_mc, &
                      wgtfac_c, &
                      e_bln_c_s, &
                      geofac_div, &
                      geofac_grg_x, &
                      geofac_grg_y, &
                      geofac_n2s, &
                      nudgecoeff_e, &
                      rbf_coeff_1, &
                      rbf_coeff_2, &
                      mask_hdiff, &
                      zd_diffcoef, &
                      zd_vertoffset, &
                      zd_intcoef, &
                      num_levels, &
                      mean_cell_area, &
                      ndyn_substeps, &
                      rayleigh_damping_height, &
                      nflatlev, &
                      nflat_gradp, &
                      diffusion_type, &
                      hdiff_w, &
                      hdiff_vn, &
                      zdiffu_t, &
                      type_t_diffu, &
                      type_vn_diffu, &
                      hdiff_efdt_ratio, &
                      smagorinski_scaling_factor, &
                      hdiff_temp, &
                      thslp_zdiffu, &
                      thhgtd_zdiffu, &
                      denom_diffu_v, &
                      nudge_max_coeff, &
                      itype_sher, &
                      tangent_orientation, &
                      inverse_primal_edge_lengths, &
                      inv_dual_edge_length, &
                      inv_vert_vert_length, &
                      edge_areas, &
                      f_e, &
                      cell_areas, &
                      primal_normal_vert_x, &
                      primal_normal_vert_y, &
                      dual_normal_vert_x, &
                      dual_normal_vert_y, &
                      primal_normal_cell_x, &
                      primal_normal_cell_y, &
                      dual_normal_cell_x, &
                      dual_normal_cell_y, &
                      rc)

   print *, "Python exit code = ", rc
   if (rc /= 0) then
       call exit(1)
   end if

   ! initial run
   call diffusion_run(w, vn, exner, theta_v, rho, hdef_ic, div_ic, dwdx, dwdy, dtime, linit, rc)
   print *, "Initial diffusion run done"

   ! Call diffusion_run
   call profile_enable(rc)
   call diffusion_run(w, vn, exner, theta_v, rho, hdef_ic, div_ic, dwdx, dwdy, dtime, linit, rc)
   call profile_disable(rc)

   print *, "Python exit code = ", rc
   if (rc /= 0) then
       call exit(1)
   end if

    !$acc update host(vct_a, theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, &
    !$acc geofac_grg_x, geofac_grg_y, geofac_n2s, nudgecoeff_e, rbf_coeff_1, &
    !$acc rbf_coeff_2, dwdx, dwdy, hdef_ic, div_ic, w, vn, exner, theta_v, rho, &
    !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, &
    !$acc dual_normal_vert_y, primal_normal_cell_x, primal_normal_cell_y, &
    !$acc primal_normal_vert_x, primal_normal_vert_y, tangent_orientation, &
    !$acc inverse_primal_edge_lengths, inv_dual_edge_length, inv_vert_vert_length, &
    !$acc edge_areas, cell_areas, f_e, zd_diffcoef, zd_vertoffset, zd_intcoef, &
    !$acc mask_hdiff)

   print *, "passed: could run diffusion"

    !$acc end data
    !$acc exit data delete(vct_a, theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, &
    !$acc geofac_grg_x, geofac_grg_y, geofac_n2s, nudgecoeff_e, rbf_coeff_1, &
    !$acc rbf_coeff_2, dwdx, dwdy, hdef_ic, div_ic, w, vn, exner, theta_v, rho, &
    !$acc dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, &
    !$acc dual_normal_vert_y, primal_normal_cell_x, primal_normal_cell_y, &
    !$acc primal_normal_vert_x, primal_normal_vert_y, tangent_orientation, &
    !$acc inverse_primal_edge_lengths, inv_dual_edge_length, inv_vert_vert_length, &
    !$acc edge_areas, cell_areas, f_e, zd_diffcoef, zd_vertoffset, zd_intcoef, &
    !$acc mask_hdiff)
end program diffusion_simulation
