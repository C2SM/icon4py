
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_44
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_44( &
         z_beta, &
         exner_nnow, &
         rho_nnow, &
         theta_v_nnow, &
         inv_ddqz_z_full, &
         z_alpha, &
         vwind_impl_wgt, &
         theta_v_ic, &
         rho_ic, &
         dtime, &
         rd, &
         cvd, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_beta
         real(c_double), dimension(*), target :: exner_nnow
         real(c_double), dimension(*), target :: rho_nnow
         real(c_double), dimension(*), target :: theta_v_nnow
         real(c_double), dimension(*), target :: inv_ddqz_z_full
         real(c_double), dimension(*), target :: z_alpha
         real(c_double), dimension(*), target :: vwind_impl_wgt
         real(c_double), dimension(*), target :: theta_v_ic
         real(c_double), dimension(*), target :: rho_ic
         real(c_double), value, target :: dtime
         real(c_double), value, target :: rd
         real(c_double), value, target :: cvd
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_44( &
         z_beta, &
         exner_nnow, &
         rho_nnow, &
         theta_v_nnow, &
         inv_ddqz_z_full, &
         z_alpha, &
         vwind_impl_wgt, &
         theta_v_ic, &
         rho_ic, &
         dtime, &
         rd, &
         cvd, &
         z_beta_before, &
         z_alpha_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         z_beta_rel_tol, &
         z_beta_abs_tol, &
         z_alpha_rel_tol, &
         z_alpha_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_beta
         real(c_double), dimension(*), target :: exner_nnow
         real(c_double), dimension(*), target :: rho_nnow
         real(c_double), dimension(*), target :: theta_v_nnow
         real(c_double), dimension(*), target :: inv_ddqz_z_full
         real(c_double), dimension(*), target :: z_alpha
         real(c_double), dimension(*), target :: vwind_impl_wgt
         real(c_double), dimension(*), target :: theta_v_ic
         real(c_double), dimension(*), target :: rho_ic
         real(c_double), value, target :: dtime
         real(c_double), value, target :: rd
         real(c_double), value, target :: cvd
         real(c_double), dimension(*), target :: z_beta_before
         real(c_double), dimension(*), target :: z_alpha_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: z_beta_rel_tol
         real(c_double), value, target :: z_beta_abs_tol
         real(c_double), value, target :: z_alpha_rel_tol
         real(c_double), value, target :: z_alpha_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_44( &
         mesh, &
         k_size, &
         stream, &
         z_beta_kmax, &
         z_alpha_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: z_beta_kmax
         integer(c_int), value, target :: z_alpha_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_44() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_44( &
      z_beta, &
      exner_nnow, &
      rho_nnow, &
      theta_v_nnow, &
      inv_ddqz_z_full, &
      z_alpha, &
      vwind_impl_wgt, &
      theta_v_ic, &
      rho_ic, &
      dtime, &
      rd, &
      cvd, &
      z_beta_before, &
      z_alpha_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      z_beta_rel_tol, &
      z_beta_abs_tol, &
      z_alpha_rel_tol, &
      z_alpha_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: z_beta
      real(c_double), dimension(:, :), target :: exner_nnow
      real(c_double), dimension(:, :), target :: rho_nnow
      real(c_double), dimension(:, :), target :: theta_v_nnow
      real(c_double), dimension(:, :), target :: inv_ddqz_z_full
      real(c_double), dimension(:, :), target :: z_alpha
      real(c_double), dimension(:), target :: vwind_impl_wgt
      real(c_double), dimension(:, :), target :: theta_v_ic
      real(c_double), dimension(:, :), target :: rho_ic
      real(c_double), value, target :: dtime
      real(c_double), value, target :: rd
      real(c_double), value, target :: cvd
      real(c_double), dimension(:, :), target :: z_beta_before
      real(c_double), dimension(:, :), target :: z_alpha_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: z_beta_rel_tol
      real(c_double), value, target, optional :: z_beta_abs_tol
      real(c_double), value, target, optional :: z_alpha_rel_tol
      real(c_double), value, target, optional :: z_alpha_abs_tol

      real(c_double) :: z_beta_rel_err_tol
      real(c_double) :: z_beta_abs_err_tol
      real(c_double) :: z_alpha_rel_err_tol
      real(c_double) :: z_alpha_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(z_beta_rel_tol)) then
         z_beta_rel_err_tol = z_beta_rel_tol
      else
         z_beta_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_beta_abs_tol)) then
         z_beta_abs_err_tol = z_beta_abs_tol
      else
         z_beta_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_alpha_rel_tol)) then
         z_alpha_rel_err_tol = z_alpha_rel_tol
      else
         z_alpha_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_alpha_abs_tol)) then
         z_alpha_abs_err_tol = z_alpha_abs_tol
      else
         z_alpha_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC z_beta, &
      !$ACC exner_nnow, &
      !$ACC rho_nnow, &
      !$ACC theta_v_nnow, &
      !$ACC inv_ddqz_z_full, &
      !$ACC z_alpha, &
      !$ACC vwind_impl_wgt, &
      !$ACC theta_v_ic, &
      !$ACC rho_ic, &
      !$ACC z_beta_before, &
      !$ACC z_alpha_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_44 &
         ( &
         z_beta, &
         exner_nnow, &
         rho_nnow, &
         theta_v_nnow, &
         inv_ddqz_z_full, &
         z_alpha, &
         vwind_impl_wgt, &
         theta_v_ic, &
         rho_ic, &
         dtime, &
         rd, &
         cvd, &
         z_beta_before, &
         z_alpha_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         z_beta_rel_err_tol, &
         z_beta_abs_err_tol, &
         z_alpha_rel_err_tol, &
         z_alpha_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_44 &
         ( &
         z_beta, &
         exner_nnow, &
         rho_nnow, &
         theta_v_nnow, &
         inv_ddqz_z_full, &
         z_alpha, &
         vwind_impl_wgt, &
         theta_v_ic, &
         rho_ic, &
         dtime, &
         rd, &
         cvd, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_44( &
      mesh, &
      k_size, &
      stream, &
      z_beta_kmax, &
      z_alpha_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: z_beta_kmax
      integer(c_int), value, target, optional :: z_alpha_kmax

      integer(c_int) :: z_beta_kvert_max
      integer(c_int) :: z_alpha_kvert_max

      if (present(z_beta_kmax)) then
         z_beta_kvert_max = z_beta_kmax
      else
         z_beta_kvert_max = k_size
      end if
      if (present(z_alpha_kmax)) then
         z_alpha_kvert_max = z_alpha_kmax
      else
         z_alpha_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_44 &
         ( &
         mesh, &
         k_size, &
         stream, &
         z_beta_kvert_max, &
         z_alpha_kvert_max &
         )
   end subroutine

end module