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
end module random_utils

program diffusion_simulation
   use, intrinsic :: iso_c_binding
   use random_utils, only: fill_random_1d, fill_random_2d
   use diffusion_plugin
   implicit none

   ! Constants and types
   integer(c_int), parameter :: num_cells = 20480
   integer(c_int), parameter :: num_edges = 30720
   integer(c_int), parameter :: num_vertices = 10242
   integer(c_int), parameter :: num_levels = 60
   integer(c_int), parameter :: num_c2ec2o = 4
   integer(c_int), parameter :: num_v2e = 6
   integer(c_int), parameter :: num_ce = num_edges*2
   integer(c_int), parameter :: num_ec = num_edges*2
   integer(c_int), parameter :: num_ecv = num_edges*4
   real(c_double), parameter :: mean_cell_area = 24907282236.708576
   integer(c_int), parameter :: ndyn_substeps = 2
   real(c_double), parameter :: dtime = 2.0
   real(c_double), parameter :: rayleigh_damping_height = 50000
   integer(c_int), parameter :: nflatlev = 30
   integer(c_int), parameter :: nflat_gradp = 59
   integer(c_int), parameter :: diffusion_type = 5 ! Assuming DiffusionType.SMAGORINSKY_4TH_ORDER is represented by 5
   logical(c_int), parameter :: hdiff_w = .true.
   logical(c_int), parameter :: hdiff_vn = .true.
   logical(c_int), parameter :: zdiffu_t = .false.
   integer(c_int), parameter :: type_t_diffu = 2
   integer(c_int), parameter :: type_vn_diffu = 1
   real(c_double), parameter :: hdiff_efdt_ratio = 24.0
   real(c_double), parameter :: smagorinski_scaling_factor = 0.025
   logical(c_int), parameter :: hdiff_temp = .true.

   ! Declaring arrays for diffusion_init and diffusion_run
   real(c_double), dimension(:), allocatable :: vct_a
   real(c_double), dimension(:, :), allocatable :: theta_ref_mc
   real(c_double), dimension(:, :), allocatable :: wgtfac_c
   real(c_double), dimension(:), allocatable :: e_bln_c_s
   real(c_double), dimension(:), allocatable :: geofac_div
   real(c_double), dimension(:, :), allocatable :: geofac_grg_x
   real(c_double), dimension(:, :), allocatable :: geofac_grg_y
   real(c_double), dimension(:, :), allocatable :: geofac_n2s
   real(c_double), dimension(:), allocatable :: nudgecoeff_e
   real(c_double), dimension(:, :), allocatable :: rbf_coeff_1
   real(c_double), dimension(:, :), allocatable :: rbf_coeff_2
   real(c_double), dimension(:, :), allocatable :: dwdx
   real(c_double), dimension(:, :), allocatable :: dwdy
   real(c_double), dimension(:, :), allocatable :: hdef_ic
   real(c_double), dimension(:, :), allocatable :: div_ic
   real(c_double), dimension(:, :), allocatable :: w
   real(c_double), dimension(:, :), allocatable :: vn
   real(c_double), dimension(:, :), allocatable :: exner
   real(c_double), dimension(:, :), allocatable :: theta_v
   real(c_double), dimension(:, :), allocatable :: rho
   real(c_double), dimension(:), allocatable :: dual_normal_cell_x
   real(c_double), dimension(:), allocatable :: dual_normal_cell_y
   real(c_double), dimension(:), allocatable :: dual_normal_vert_x
   real(c_double), dimension(:), allocatable :: dual_normal_vert_y
   real(c_double), dimension(:), allocatable :: primal_normal_cell_x
   real(c_double), dimension(:), allocatable :: primal_normal_cell_y
   real(c_double), dimension(:), allocatable :: primal_normal_vert_x
   real(c_double), dimension(:), allocatable :: primal_normal_vert_y
   real(c_double), dimension(:), allocatable :: tangent_orientation
   real(c_double), dimension(:), allocatable :: inverse_primal_edge_lengths
   real(c_double), dimension(:), allocatable :: inv_dual_edge_length
   real(c_double), dimension(:), allocatable :: inv_vert_vert_length
   real(c_double), dimension(:), allocatable :: edge_areas
   real(c_double), dimension(:), allocatable :: f_e
   real(c_double), dimension(:), allocatable :: cell_areas

   ! allocating arrays
   allocate (vct_a(num_levels))
   allocate (theta_ref_mc(num_cells, num_levels))
   allocate (wgtfac_c(num_cells, num_levels + 1))
   allocate (e_bln_c_s(num_ce))
   allocate (geofac_div(num_ce))
   allocate (geofac_grg_x(num_cells, 4))
   allocate (geofac_grg_y(num_cells, 4))
   allocate (geofac_n2s(num_cells, 4))
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
   allocate (dual_normal_cell_x(num_ec))
   allocate (dual_normal_cell_y(num_ec))
   allocate (dual_normal_vert_x(num_ecv))
   allocate (dual_normal_vert_y(num_ecv))
   allocate (primal_normal_cell_x(num_ec))
   allocate (primal_normal_cell_y(num_ec))
   allocate (primal_normal_vert_x(num_ecv))
   allocate (primal_normal_vert_y(num_ecv))
   allocate (tangent_orientation(num_edges))
   allocate (inverse_primal_edge_lengths(num_edges))
   allocate (inv_dual_edge_length(num_edges))
   allocate (inv_vert_vert_length(num_edges))
   allocate (edge_areas(num_edges))
   allocate (f_e(num_edges))
   allocate (cell_areas(num_cells))

   ! Fill arrays with random numbers
   ! For 1D arrays
   call fill_random_1d(vct_a, 0.0_c_double, 75000.0_c_double) ! needs to be above 50000 damping height restriction
   call fill_random_1d(e_bln_c_s, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(geofac_div, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(nudgecoeff_e, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(dual_normal_cell_x, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(dual_normal_cell_y, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(primal_normal_cell_x, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(primal_normal_cell_y, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(tangent_orientation, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inverse_primal_edge_lengths, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inv_dual_edge_length, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inv_vert_vert_length, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(edge_areas, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(f_e, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(cell_areas, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(dual_normal_vert_x, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(dual_normal_vert_y, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(primal_normal_vert_x, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(primal_normal_vert_y, 0.0_c_double, 1.0_c_double)

   ! For 2D arrays
   call fill_random_2d(theta_ref_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(wgtfac_c, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grg_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grg_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_n2s, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_coeff_1, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_coeff_2, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dwdx, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dwdy, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(hdef_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(div_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(w, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(vn, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(exner, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(theta_v, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rho, 0.0_c_double, 1.0_c_double)

   ! Call diffusion_init
   call diffusion_init(vct_a, theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, &
                       geofac_grg_x, geofac_grg_y, geofac_n2s, nudgecoeff_e, rbf_coeff_1, &
                       rbf_coeff_2, num_levels, mean_cell_area, ndyn_substeps, &
                       rayleigh_damping_height, nflatlev, nflat_gradp, diffusion_type, &
                       hdiff_w, hdiff_vn, zdiffu_t, type_t_diffu, type_vn_diffu, &
                       hdiff_efdt_ratio, smagorinski_scaling_factor, hdiff_temp, &
                       tangent_orientation, inverse_primal_edge_lengths, inv_dual_edge_length, &
                       inv_vert_vert_length, edge_areas, f_e, cell_areas, primal_normal_vert_x, &
                       primal_normal_vert_y, dual_normal_vert_x, dual_normal_vert_y, &
                       primal_normal_cell_x, primal_normal_cell_y, dual_normal_cell_x, &
                       dual_normal_cell_y)

   ! Call diffusion_run
   call diffusion_run(w, vn, exner, theta_v, rho, hdef_ic, div_ic, dwdx, dwdy, dtime)

   print *, "passed: could run diffusion"

end program diffusion_simulation
