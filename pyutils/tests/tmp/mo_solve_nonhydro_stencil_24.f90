
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_24
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_24( &
         vn_nnow, &
         ddt_vn_adv_ntl1, &
         ddt_vn_phy, &
         z_theta_v_e, &
         z_gradh_exner, &
         vn_nnew, &
         dtime, &
         cpd, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn_nnow
         real(c_double), dimension(*), target :: ddt_vn_adv_ntl1
         real(c_double), dimension(*), target :: ddt_vn_phy
         real(c_double), dimension(*), target :: z_theta_v_e
         real(c_double), dimension(*), target :: z_gradh_exner
         real(c_double), dimension(*), target :: vn_nnew
         real(c_double), value, target :: dtime
         real(c_double), value, target :: cpd
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_24( &
         vn_nnow, &
         ddt_vn_adv_ntl1, &
         ddt_vn_phy, &
         z_theta_v_e, &
         z_gradh_exner, &
         vn_nnew, &
         dtime, &
         cpd, &
         vn_nnew_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         vn_nnew_rel_tol, &
         vn_nnew_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn_nnow
         real(c_double), dimension(*), target :: ddt_vn_adv_ntl1
         real(c_double), dimension(*), target :: ddt_vn_phy
         real(c_double), dimension(*), target :: z_theta_v_e
         real(c_double), dimension(*), target :: z_gradh_exner
         real(c_double), dimension(*), target :: vn_nnew
         real(c_double), value, target :: dtime
         real(c_double), value, target :: cpd
         real(c_double), dimension(*), target :: vn_nnew_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: vn_nnew_rel_tol
         real(c_double), value, target :: vn_nnew_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_24( &
         mesh, &
         k_size, &
         stream, &
         vn_nnew_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: vn_nnew_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_24() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_24( &
      vn_nnow, &
      ddt_vn_adv_ntl1, &
      ddt_vn_phy, &
      z_theta_v_e, &
      z_gradh_exner, &
      vn_nnew, &
      dtime, &
      cpd, &
      vn_nnew_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      vn_nnew_rel_tol, &
      vn_nnew_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: vn_nnow
      real(c_double), dimension(:, :), target :: ddt_vn_adv_ntl1
      real(c_double), dimension(:, :), target :: ddt_vn_phy
      real(c_double), dimension(:, :), target :: z_theta_v_e
      real(c_double), dimension(:, :), target :: z_gradh_exner
      real(c_double), dimension(:, :), target :: vn_nnew
      real(c_double), value, target :: dtime
      real(c_double), value, target :: cpd
      real(c_double), dimension(:, :), target :: vn_nnew_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: vn_nnew_rel_tol
      real(c_double), value, target, optional :: vn_nnew_abs_tol

      real(c_double) :: vn_nnew_rel_err_tol
      real(c_double) :: vn_nnew_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(vn_nnew_rel_tol)) then
         vn_nnew_rel_err_tol = vn_nnew_rel_tol
      else
         vn_nnew_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(vn_nnew_abs_tol)) then
         vn_nnew_abs_err_tol = vn_nnew_abs_tol
      else
         vn_nnew_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC vn_nnow, &
      !$ACC ddt_vn_adv_ntl1, &
      !$ACC ddt_vn_phy, &
      !$ACC z_theta_v_e, &
      !$ACC z_gradh_exner, &
      !$ACC vn_nnew, &
      !$ACC vn_nnew_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_24 &
         ( &
         vn_nnow, &
         ddt_vn_adv_ntl1, &
         ddt_vn_phy, &
         z_theta_v_e, &
         z_gradh_exner, &
         vn_nnew, &
         dtime, &
         cpd, &
         vn_nnew_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         vn_nnew_rel_err_tol, &
         vn_nnew_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_24 &
         ( &
         vn_nnow, &
         ddt_vn_adv_ntl1, &
         ddt_vn_phy, &
         z_theta_v_e, &
         z_gradh_exner, &
         vn_nnew, &
         dtime, &
         cpd, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_24( &
      mesh, &
      k_size, &
      stream, &
      vn_nnew_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: vn_nnew_kmax

      integer(c_int) :: vn_nnew_kvert_max

      if (present(vn_nnew_kmax)) then
         vn_nnew_kvert_max = vn_nnew_kmax
      else
         vn_nnew_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_24 &
         ( &
         mesh, &
         k_size, &
         stream, &
         vn_nnew_kvert_max &
         )
   end subroutine

end module