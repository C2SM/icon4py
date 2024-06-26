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
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(inout) :: array(:, :, :)
      real(c_double), intent(in) :: low, high
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

program solve_nh_simulation
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   use random_utils, only: fill_random_1d, fill_random_2d, fill_random_2d_bool, fill_random_3d_int, fill_random_3d
   use solve_nh_plugin
   implicit none

   integer(c_int) :: rc
   integer(c_int) :: n

   ! Constants and parameters
   integer(c_int), parameter :: num_cells = 20896
   integer(c_int), parameter :: num_edges = 31558
   integer(c_int), parameter :: num_verts = 10663
   integer(c_int), parameter :: num_levels = 60
   integer(c_int), parameter :: num_c2ec2o = 4
   integer(c_int), parameter :: num_v2e = 6
   integer(c_int), parameter :: num_c2e = 3
   integer(c_int), parameter :: num_e2c2v = 4
   integer(c_int), parameter :: num_c2e2c = 3
   integer(c_int), parameter :: num_e2c = 2
   real(c_double), parameter :: mean_cell_area = 24907282236.708576

   integer(c_int), parameter :: nrdmax = 50
   real(c_double), parameter :: dtime = 10.0
   real(c_double), parameter :: rayleigh_damping_height = 12500.0
   integer(c_int), parameter :: nflatlev = 30
   integer(c_int), parameter :: nflat_gradp = 59
   real(c_double), parameter :: ndyn_substeps = 2.0

   integer(c_int), parameter :: itime_scheme = 4
   integer(c_int), parameter :: iadv_rhotheta = 2
   integer(c_int), parameter :: igradp_method = 3
   integer(c_int), parameter :: rayleigh_type = 1
   real(c_double), parameter :: rayleigh_coeff = 0.1
   integer(c_int), parameter :: divdamp_order = 24
   logical(c_int), parameter :: is_iau_active = .false.
   real(c_double), parameter :: iau_wgt_dyn = 0.5
   real(c_double), parameter :: divdamp_fac_o2 = 0.5
   integer(c_int), parameter :: divdamp_type = 1
   real(c_double), parameter :: divdamp_trans_start = 1000.0
   real(c_double), parameter :: divdamp_trans_end = 2000.0
   logical(c_int), parameter :: l_vert_nested = .false.
   real(c_double), parameter :: rhotheta_offctr = 1.0
   real(c_double), parameter :: veladv_offctr = 1.0
   real(c_double), parameter :: max_nudging_coeff = 0.1
   real(c_double), parameter :: divdamp_fac = 1.0
   real(c_double), parameter :: divdamp_fac2 = 2.0
   real(c_double), parameter :: divdamp_fac3 = 3.0
   real(c_double), parameter :: divdamp_fac4 = 4.0
   real(c_double), parameter :: divdamp_z = 1.0
   real(c_double), parameter :: divdamp_z2 = 2.0
   real(c_double), parameter :: divdamp_z3 = 3.0
   real(c_double), parameter :: divdamp_z4 = 4.0
   real(c_double), parameter :: htop_moist_proc = 1000.0
   integer(c_int), parameter :: comm_id = 0
   logical(c_int), parameter :: limited_area = .true.

   ! Declaring arrays
   real(c_double), dimension(:), allocatable :: vct_a, vct_b, rayleigh_w, tangent_orientation, inverse_primal_edge_lengths, inv_dual_edge_length, inv_vert_vert_length, edge_areas, f_e, cell_areas, vwind_expl_wgt, vwind_impl_wgt, scalfac_dd3d
   real(c_double), dimension(:, :), allocatable :: theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, geofac_grg_x, geofac_grg_y, geofac_n2s, rbf_coeff_1, rbf_coeff_2, dwdx, dwdy, hdef_ic, div_ic, w_now, w_new, vn_now, vn_new, exner_now, exner_new, theta_v_now, theta_v_new, rho_now, rho_new, dual_normal_cell_x, dual_normal_cell_y, dual_normal_vert_x, dual_normal_vert_y, primal_normal_cell_x, primal_normal_cell_y, primal_normal_vert_x, primal_normal_vert_y, zd_diffcoef, exner_exfac, exner_ref_mc, wgtfacq_c_dsl, inv_ddqz_z_full, d_exner_dz_ref_ic, ddqz_z_half, theta_ref_ic, d2dexdz2_fac1_mc, d2dexdz2_fac2_mc, rho_ref_me, theta_ref_me, ddxn_z_full, pg_exdist, ddqz_z_full_e, ddxt_z_full, wgtfac_e, wgtfacq_e, coeff1_dwdz, coeff2_dwdz, grf_tend_rho, grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, grf_tend_vn, vn_ie, vt, mass_flx_me, mass_flx_ic, vn_traj, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, ddt_w_adv_ntl1, ddt_w_adv_ntl2, c_lin_e, pos_on_tplane_e_1, pos_on_tplane_e_2, rbf_vec_coeff_e
   integer(c_int), dimension(:, :, :), allocatable :: zd_vertoffset, vertoffset_gradp
   logical(c_int), dimension(:, :), allocatable :: ipeidx_dsl, mask_prog_halo_c, bdy_halo_c
   real(c_double), dimension(:, :), allocatable :: coeff_gradekin
   real(c_double), dimension(:, :), allocatable :: geofac_grdiv, geofac_rot, c_intp

   ! Allocate arrays
   allocate(vct_a(num_levels))
   allocate(vct_b(num_levels))
   allocate(rayleigh_w(num_levels))
   allocate(tangent_orientation(num_edges))
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
   allocate(wgtfac_c(num_cells, num_levels + 1))
   allocate(e_bln_c_s(num_cells, num_c2e))
   allocate(geofac_div(num_cells, num_c2e))
   allocate(geofac_grg_x(num_cells, num_c2ec2o))
   allocate(geofac_grg_y(num_cells, num_c2ec2o))
   allocate(geofac_n2s(num_cells, num_c2ec2o))
   allocate(rbf_coeff_1(num_verts, num_v2e))
   allocate(rbf_coeff_2(num_verts, num_v2e))
   allocate(dwdx(num_cells, num_levels))
   allocate(dwdy(num_cells, num_levels))
   allocate(hdef_ic(num_cells, num_levels + 1))
   allocate(div_ic(num_cells, num_levels + 1))
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
   allocate(zd_diffcoef(num_cells, num_levels))
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
   allocate(zd_vertoffset(num_cells, num_c2e2c, num_levels))
   allocate(vertoffset_gradp(num_edges, num_e2c, num_levels))
   allocate(ipeidx_dsl(num_edges, num_levels))
   allocate(mask_prog_halo_c(num_cells))
   allocate(bdy_halo_c(num_cells))
   allocate(coeff_gradekin(num_edges, num_e2c))
   allocate(geofac_grdiv(num_edges, num_e2c2v))
   allocate(geofac_rot(num_verts, num_v2e))
   allocate(c_intp(num_verts, num_e2c))

   ! Fill arrays with random numbers
   call fill_random_1d(vct_a, 0.0_c_double, 75000.0_c_double)
   call fill_random_1d(vct_b, 0.0_c_double, 75000.0_c_double)
   call fill_random_1d(rayleigh_w, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(tangent_orientation, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inverse_primal_edge_lengths, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inv_dual_edge_length, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(inv_vert_vert_length, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(edge_areas, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(f_e, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(cell_areas, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(vwind_expl_wgt, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(vwind_impl_wgt, 0.0_c_double, 1.0_c_double)
   call fill_random_1d(scalfac_dd3d, 0.0_c_double, 1.0_c_double)

   call fill_random_2d(theta_ref_mc, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(wgtfac_c, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(e_bln_c_s, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_div, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grg_x, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grg_y, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_n2s, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_coeff_1, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(rbf_coeff_2, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dwdx, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(dwdy, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(hdef_ic, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(div_ic, 0.0_c_double, 1.0_c_double)
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
   call fill_random_2d(zd_diffcoef, 0.0_c_double, 1.0_c_double)
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
   call fill_random_3d_int(zd_vertoffset, 0, 1)
   call fill_random_3d_int(vertoffset_gradp, 0, 1)
   call fill_random_2d_bool(ipeidx_dsl)
   call fill_random_2d_bool(mask_prog_halo_c)
   call fill_random_2d_bool(bdy_halo_c)
   call fill_random_2d(coeff_gradekin, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_grdiv, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(geofac_rot, 0.0_c_double, 1.0_c_double)
   call fill_random_2d(c_intp, 0.0_c_double, 1.0_c_double)

   ! Call solve_nh_init
   call solve_nh_init(vct_a, vct_b, nrdmax, nflat_gradp, nflatlev, num_cells, &
                      num_edges, num_verts, num_levels, mean_cell_area, &
                      cell_areas, primal_normal_cell_x, primal_normal_cell_y, &
                      dual_normal_cell_x, dual_normal_cell_y, edge_areas, &
                      tangent_orientation, inverse_primal_edge_lengths, &
                      inv_dual_edge_length, inv_vert_vert_length, &
                      primal_normal_vert_x, primal_normal_vert_y, &
                      dual_normal_vert_x, dual_normal_vert_y, f_e, &
                      c_lin_e, c_intp, e_flx_avg, geofac_grdiv, geofac_rot, &
                      pos_on_tplane_e_1, pos_on_tplane_e_2, rbf_vec_coeff_e, &
                      e_bln_c_s, rbf_coeff_1, rbf_coeff_2, geofac_div, &
                      geofac_n2s, geofac_grg_x, geofac_grg_y, nudgecoeff_e, &
                      bdy_halo_c, mask_prog_halo_c, rayleigh_w, exner_exfac, &
                      exner_ref_mc, wgtfac_c, wgtfacq_c_dsl, inv_ddqz_z_full, &
                      rho_ref_mc, theta_ref_mc, vwind_expl_wgt, &
                      d_exner_dz_ref_ic, ddqz_z_half, theta_ref_ic, &
                      d2dexdz2_fac1_mc, d2dexdz2_fac2_mc, rho_ref_me, &
                      theta_ref_me, ddxn_z_full, zdiff_gradp, &
                      vertoffset_gradp, ipeidx_dsl, pg_exdist, &
                      ddqz_z_full_e, ddxt_z_full, wgtfac_e, wgtfacq_e, &
                      vwind_impl_wgt, hmask_dd3d, scalfac_dd3d, &
                      coeff1_dwdz, coeff2_dwdz, coeff_gradekin, &
                      c_owner_mask, rayleigh_damping_height, itime_scheme, &
                      iadv_rhotheta, igradp_method, ndyn_substeps, &
                      rayleigh_type, rayleigh_coeff, divdamp_order, &
                      is_iau_active, iau_wgt_dyn, divdamp_type, &
                      divdamp_trans_start, divdamp_trans_end, l_vert_nested, &
                      rhotheta_offctr, veladv_offctr, max_nudging_coeff, &
                      divdamp_fac, divdamp_fac2, divdamp_fac3, divdamp_fac4, &
                      divdamp_z, divdamp_z2, divdamp_z3, divdamp_z4, &
                      htop_moist_proc, comm_id, limited_area, rc)

   if (rc /= 0) then
       print *, "Error in solve_nh_init"
       call exit(1)
   end if

   ! Main computation loop
   do n = 1, 60
      ! Call solve_nh_run
      call solve_nh_run(rho_now, rho_new, exner_now, exner_new, w_now, w_new, &
                        theta_v_now, theta_v_new, vn_now, vn_new, &
                        w_concorr_c, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2, &
                        ddt_w_adv_ntl1, ddt_w_adv_ntl2, theta_v_ic, rho_ic, &
                        exner_pr, exner_dyn_incr, ddt_exner_phy, grf_tend_rho, &
                        grf_tend_thv, grf_tend_w, mass_fl_e, ddt_vn_phy, &
                        grf_tend_vn, vn_ie, vt, mass_flx_me, mass_flx_ic, &
                        vn_traj, dtime, .false., .true., .false., .false., &
                        divdamp_fac_o2, ndyn_substeps, rc)

      if (rc /= 0) then
          print *, "Error in solve_nh_run"
          call exit(1)
      end if
   end do

   print *, "Simulation completed successfully"
end program solve_nh_simulation
