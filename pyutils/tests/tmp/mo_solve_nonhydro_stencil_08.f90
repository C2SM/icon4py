
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_08
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_08( &
         wgtfac_c, &
         rho, &
         rho_ref_mc, &
         theta_v, &
         theta_ref_mc, &
         rho_ic, &
         z_rth_pr_1, &
         z_rth_pr_2, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: wgtfac_c
         real(c_double), dimension(*), target :: rho
         real(c_double), dimension(*), target :: rho_ref_mc
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: theta_ref_mc
         real(c_double), dimension(*), target :: rho_ic
         real(c_double), dimension(*), target :: z_rth_pr_1
         real(c_double), dimension(*), target :: z_rth_pr_2
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_08( &
         wgtfac_c, &
         rho, &
         rho_ref_mc, &
         theta_v, &
         theta_ref_mc, &
         rho_ic, &
         z_rth_pr_1, &
         z_rth_pr_2, &
         rho_ic_before, &
         z_rth_pr_1_before, &
         z_rth_pr_2_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         rho_ic_rel_tol, &
         rho_ic_abs_tol, &
         z_rth_pr_1_rel_tol, &
         z_rth_pr_1_abs_tol, &
         z_rth_pr_2_rel_tol, &
         z_rth_pr_2_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: wgtfac_c
         real(c_double), dimension(*), target :: rho
         real(c_double), dimension(*), target :: rho_ref_mc
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: theta_ref_mc
         real(c_double), dimension(*), target :: rho_ic
         real(c_double), dimension(*), target :: z_rth_pr_1
         real(c_double), dimension(*), target :: z_rth_pr_2
         real(c_double), dimension(*), target :: rho_ic_before
         real(c_double), dimension(*), target :: z_rth_pr_1_before
         real(c_double), dimension(*), target :: z_rth_pr_2_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: rho_ic_rel_tol
         real(c_double), value, target :: rho_ic_abs_tol
         real(c_double), value, target :: z_rth_pr_1_rel_tol
         real(c_double), value, target :: z_rth_pr_1_abs_tol
         real(c_double), value, target :: z_rth_pr_2_rel_tol
         real(c_double), value, target :: z_rth_pr_2_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_08( &
         mesh, &
         k_size, &
         stream, &
         rho_ic_kmax, &
         z_rth_pr_1_kmax, &
         z_rth_pr_2_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: rho_ic_kmax
         integer(c_int), value, target :: z_rth_pr_1_kmax
         integer(c_int), value, target :: z_rth_pr_2_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_08() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_08( &
      wgtfac_c, &
      rho, &
      rho_ref_mc, &
      theta_v, &
      theta_ref_mc, &
      rho_ic, &
      z_rth_pr_1, &
      z_rth_pr_2, &
      rho_ic_before, &
      z_rth_pr_1_before, &
      z_rth_pr_2_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      rho_ic_rel_tol, &
      rho_ic_abs_tol, &
      z_rth_pr_1_rel_tol, &
      z_rth_pr_1_abs_tol, &
      z_rth_pr_2_rel_tol, &
      z_rth_pr_2_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: wgtfac_c
      real(c_double), dimension(:, :), target :: rho
      real(c_double), dimension(:, :), target :: rho_ref_mc
      real(c_double), dimension(:, :), target :: theta_v
      real(c_double), dimension(:, :), target :: theta_ref_mc
      real(c_double), dimension(:, :), target :: rho_ic
      real(c_double), dimension(:, :), target :: z_rth_pr_1
      real(c_double), dimension(:, :), target :: z_rth_pr_2
      real(c_double), dimension(:, :), target :: rho_ic_before
      real(c_double), dimension(:, :), target :: z_rth_pr_1_before
      real(c_double), dimension(:, :), target :: z_rth_pr_2_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: rho_ic_rel_tol
      real(c_double), value, target, optional :: rho_ic_abs_tol
      real(c_double), value, target, optional :: z_rth_pr_1_rel_tol
      real(c_double), value, target, optional :: z_rth_pr_1_abs_tol
      real(c_double), value, target, optional :: z_rth_pr_2_rel_tol
      real(c_double), value, target, optional :: z_rth_pr_2_abs_tol

      real(c_double) :: rho_ic_rel_err_tol
      real(c_double) :: rho_ic_abs_err_tol
      real(c_double) :: z_rth_pr_1_rel_err_tol
      real(c_double) :: z_rth_pr_1_abs_err_tol
      real(c_double) :: z_rth_pr_2_rel_err_tol
      real(c_double) :: z_rth_pr_2_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(rho_ic_rel_tol)) then
         rho_ic_rel_err_tol = rho_ic_rel_tol
      else
         rho_ic_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(rho_ic_abs_tol)) then
         rho_ic_abs_err_tol = rho_ic_abs_tol
      else
         rho_ic_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_rth_pr_1_rel_tol)) then
         z_rth_pr_1_rel_err_tol = z_rth_pr_1_rel_tol
      else
         z_rth_pr_1_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_rth_pr_1_abs_tol)) then
         z_rth_pr_1_abs_err_tol = z_rth_pr_1_abs_tol
      else
         z_rth_pr_1_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_rth_pr_2_rel_tol)) then
         z_rth_pr_2_rel_err_tol = z_rth_pr_2_rel_tol
      else
         z_rth_pr_2_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_rth_pr_2_abs_tol)) then
         z_rth_pr_2_abs_err_tol = z_rth_pr_2_abs_tol
      else
         z_rth_pr_2_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC wgtfac_c, &
      !$ACC rho, &
      !$ACC rho_ref_mc, &
      !$ACC theta_v, &
      !$ACC theta_ref_mc, &
      !$ACC rho_ic, &
      !$ACC z_rth_pr_1, &
      !$ACC z_rth_pr_2, &
      !$ACC rho_ic_before, &
      !$ACC z_rth_pr_1_before, &
      !$ACC z_rth_pr_2_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_08 &
         ( &
         wgtfac_c, &
         rho, &
         rho_ref_mc, &
         theta_v, &
         theta_ref_mc, &
         rho_ic, &
         z_rth_pr_1, &
         z_rth_pr_2, &
         rho_ic_before, &
         z_rth_pr_1_before, &
         z_rth_pr_2_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         rho_ic_rel_err_tol, &
         rho_ic_abs_err_tol, &
         z_rth_pr_1_rel_err_tol, &
         z_rth_pr_1_abs_err_tol, &
         z_rth_pr_2_rel_err_tol, &
         z_rth_pr_2_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_08 &
         ( &
         wgtfac_c, &
         rho, &
         rho_ref_mc, &
         theta_v, &
         theta_ref_mc, &
         rho_ic, &
         z_rth_pr_1, &
         z_rth_pr_2, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_08( &
      mesh, &
      k_size, &
      stream, &
      rho_ic_kmax, &
      z_rth_pr_1_kmax, &
      z_rth_pr_2_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: rho_ic_kmax
      integer(c_int), value, target, optional :: z_rth_pr_1_kmax
      integer(c_int), value, target, optional :: z_rth_pr_2_kmax

      integer(c_int) :: rho_ic_kvert_max
      integer(c_int) :: z_rth_pr_1_kvert_max
      integer(c_int) :: z_rth_pr_2_kvert_max

      if (present(rho_ic_kmax)) then
         rho_ic_kvert_max = rho_ic_kmax
      else
         rho_ic_kvert_max = k_size
      end if
      if (present(z_rth_pr_1_kmax)) then
         z_rth_pr_1_kvert_max = z_rth_pr_1_kmax
      else
         z_rth_pr_1_kvert_max = k_size
      end if
      if (present(z_rth_pr_2_kmax)) then
         z_rth_pr_2_kvert_max = z_rth_pr_2_kmax
      else
         z_rth_pr_2_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_08 &
         ( &
         mesh, &
         k_size, &
         stream, &
         rho_ic_kvert_max, &
         z_rth_pr_1_kvert_max, &
         z_rth_pr_2_kvert_max &
         )
   end subroutine

end module