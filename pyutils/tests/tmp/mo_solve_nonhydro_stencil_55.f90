
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_55
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_55( &
         z_rho_expl, &
         vwind_impl_wgt, &
         inv_ddqz_z_full, &
         rho_ic, &
         w, &
         z_exner_expl, &
         exner_ref_mc, &
         z_alpha, &
         z_beta, &
         rho_now, &
         theta_v_now, &
         exner_now, &
         rho_new, &
         exner_new, &
         theta_v_new, &
         dtime, &
         cvd_o_rd, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_rho_expl
         real(c_double), dimension(*), target :: vwind_impl_wgt
         real(c_double), dimension(*), target :: inv_ddqz_z_full
         real(c_double), dimension(*), target :: rho_ic
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: z_exner_expl
         real(c_double), dimension(*), target :: exner_ref_mc
         real(c_double), dimension(*), target :: z_alpha
         real(c_double), dimension(*), target :: z_beta
         real(c_double), dimension(*), target :: rho_now
         real(c_double), dimension(*), target :: theta_v_now
         real(c_double), dimension(*), target :: exner_now
         real(c_double), dimension(*), target :: rho_new
         real(c_double), dimension(*), target :: exner_new
         real(c_double), dimension(*), target :: theta_v_new
         real(c_double), value, target :: dtime
         real(c_double), value, target :: cvd_o_rd
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_55( &
         z_rho_expl, &
         vwind_impl_wgt, &
         inv_ddqz_z_full, &
         rho_ic, &
         w, &
         z_exner_expl, &
         exner_ref_mc, &
         z_alpha, &
         z_beta, &
         rho_now, &
         theta_v_now, &
         exner_now, &
         rho_new, &
         exner_new, &
         theta_v_new, &
         dtime, &
         cvd_o_rd, &
         rho_new_before, &
         exner_new_before, &
         theta_v_new_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         rho_new_rel_tol, &
         rho_new_abs_tol, &
         exner_new_rel_tol, &
         exner_new_abs_tol, &
         theta_v_new_rel_tol, &
         theta_v_new_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_rho_expl
         real(c_double), dimension(*), target :: vwind_impl_wgt
         real(c_double), dimension(*), target :: inv_ddqz_z_full
         real(c_double), dimension(*), target :: rho_ic
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: z_exner_expl
         real(c_double), dimension(*), target :: exner_ref_mc
         real(c_double), dimension(*), target :: z_alpha
         real(c_double), dimension(*), target :: z_beta
         real(c_double), dimension(*), target :: rho_now
         real(c_double), dimension(*), target :: theta_v_now
         real(c_double), dimension(*), target :: exner_now
         real(c_double), dimension(*), target :: rho_new
         real(c_double), dimension(*), target :: exner_new
         real(c_double), dimension(*), target :: theta_v_new
         real(c_double), value, target :: dtime
         real(c_double), value, target :: cvd_o_rd
         real(c_double), dimension(*), target :: rho_new_before
         real(c_double), dimension(*), target :: exner_new_before
         real(c_double), dimension(*), target :: theta_v_new_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: rho_new_rel_tol
         real(c_double), value, target :: rho_new_abs_tol
         real(c_double), value, target :: exner_new_rel_tol
         real(c_double), value, target :: exner_new_abs_tol
         real(c_double), value, target :: theta_v_new_rel_tol
         real(c_double), value, target :: theta_v_new_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_55( &
         mesh, &
         k_size, &
         stream, &
         rho_new_kmax, &
         exner_new_kmax, &
         theta_v_new_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: rho_new_kmax
         integer(c_int), value, target :: exner_new_kmax
         integer(c_int), value, target :: theta_v_new_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_55() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_55( &
      z_rho_expl, &
      vwind_impl_wgt, &
      inv_ddqz_z_full, &
      rho_ic, &
      w, &
      z_exner_expl, &
      exner_ref_mc, &
      z_alpha, &
      z_beta, &
      rho_now, &
      theta_v_now, &
      exner_now, &
      rho_new, &
      exner_new, &
      theta_v_new, &
      dtime, &
      cvd_o_rd, &
      rho_new_before, &
      exner_new_before, &
      theta_v_new_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      rho_new_rel_tol, &
      rho_new_abs_tol, &
      exner_new_rel_tol, &
      exner_new_abs_tol, &
      theta_v_new_rel_tol, &
      theta_v_new_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: z_rho_expl
      real(c_double), dimension(:), target :: vwind_impl_wgt
      real(c_double), dimension(:, :), target :: inv_ddqz_z_full
      real(c_double), dimension(:, :), target :: rho_ic
      real(c_double), dimension(:, :), target :: w
      real(c_double), dimension(:, :), target :: z_exner_expl
      real(c_double), dimension(:, :), target :: exner_ref_mc
      real(c_double), dimension(:, :), target :: z_alpha
      real(c_double), dimension(:, :), target :: z_beta
      real(c_double), dimension(:, :), target :: rho_now
      real(c_double), dimension(:, :), target :: theta_v_now
      real(c_double), dimension(:, :), target :: exner_now
      real(c_double), dimension(:, :), target :: rho_new
      real(c_double), dimension(:, :), target :: exner_new
      real(c_double), dimension(:, :), target :: theta_v_new
      real(c_double), value, target :: dtime
      real(c_double), value, target :: cvd_o_rd
      real(c_double), dimension(:, :), target :: rho_new_before
      real(c_double), dimension(:, :), target :: exner_new_before
      real(c_double), dimension(:, :), target :: theta_v_new_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: rho_new_rel_tol
      real(c_double), value, target, optional :: rho_new_abs_tol
      real(c_double), value, target, optional :: exner_new_rel_tol
      real(c_double), value, target, optional :: exner_new_abs_tol
      real(c_double), value, target, optional :: theta_v_new_rel_tol
      real(c_double), value, target, optional :: theta_v_new_abs_tol

      real(c_double) :: rho_new_rel_err_tol
      real(c_double) :: rho_new_abs_err_tol
      real(c_double) :: exner_new_rel_err_tol
      real(c_double) :: exner_new_abs_err_tol
      real(c_double) :: theta_v_new_rel_err_tol
      real(c_double) :: theta_v_new_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(rho_new_rel_tol)) then
         rho_new_rel_err_tol = rho_new_rel_tol
      else
         rho_new_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(rho_new_abs_tol)) then
         rho_new_abs_err_tol = rho_new_abs_tol
      else
         rho_new_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(exner_new_rel_tol)) then
         exner_new_rel_err_tol = exner_new_rel_tol
      else
         exner_new_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(exner_new_abs_tol)) then
         exner_new_abs_err_tol = exner_new_abs_tol
      else
         exner_new_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(theta_v_new_rel_tol)) then
         theta_v_new_rel_err_tol = theta_v_new_rel_tol
      else
         theta_v_new_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(theta_v_new_abs_tol)) then
         theta_v_new_abs_err_tol = theta_v_new_abs_tol
      else
         theta_v_new_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC z_rho_expl, &
      !$ACC vwind_impl_wgt, &
      !$ACC inv_ddqz_z_full, &
      !$ACC rho_ic, &
      !$ACC w, &
      !$ACC z_exner_expl, &
      !$ACC exner_ref_mc, &
      !$ACC z_alpha, &
      !$ACC z_beta, &
      !$ACC rho_now, &
      !$ACC theta_v_now, &
      !$ACC exner_now, &
      !$ACC rho_new, &
      !$ACC exner_new, &
      !$ACC theta_v_new, &
      !$ACC rho_new_before, &
      !$ACC exner_new_before, &
      !$ACC theta_v_new_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_55 &
         ( &
         z_rho_expl, &
         vwind_impl_wgt, &
         inv_ddqz_z_full, &
         rho_ic, &
         w, &
         z_exner_expl, &
         exner_ref_mc, &
         z_alpha, &
         z_beta, &
         rho_now, &
         theta_v_now, &
         exner_now, &
         rho_new, &
         exner_new, &
         theta_v_new, &
         dtime, &
         cvd_o_rd, &
         rho_new_before, &
         exner_new_before, &
         theta_v_new_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         rho_new_rel_err_tol, &
         rho_new_abs_err_tol, &
         exner_new_rel_err_tol, &
         exner_new_abs_err_tol, &
         theta_v_new_rel_err_tol, &
         theta_v_new_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_55 &
         ( &
         z_rho_expl, &
         vwind_impl_wgt, &
         inv_ddqz_z_full, &
         rho_ic, &
         w, &
         z_exner_expl, &
         exner_ref_mc, &
         z_alpha, &
         z_beta, &
         rho_now, &
         theta_v_now, &
         exner_now, &
         rho_new, &
         exner_new, &
         theta_v_new, &
         dtime, &
         cvd_o_rd, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_55( &
      mesh, &
      k_size, &
      stream, &
      rho_new_kmax, &
      exner_new_kmax, &
      theta_v_new_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: rho_new_kmax
      integer(c_int), value, target, optional :: exner_new_kmax
      integer(c_int), value, target, optional :: theta_v_new_kmax

      integer(c_int) :: rho_new_kvert_max
      integer(c_int) :: exner_new_kvert_max
      integer(c_int) :: theta_v_new_kvert_max

      if (present(rho_new_kmax)) then
         rho_new_kvert_max = rho_new_kmax
      else
         rho_new_kvert_max = k_size
      end if
      if (present(exner_new_kmax)) then
         exner_new_kvert_max = exner_new_kmax
      else
         exner_new_kvert_max = k_size
      end if
      if (present(theta_v_new_kmax)) then
         theta_v_new_kvert_max = theta_v_new_kmax
      else
         theta_v_new_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_55 &
         ( &
         mesh, &
         k_size, &
         stream, &
         rho_new_kvert_max, &
         exner_new_kvert_max, &
         theta_v_new_kvert_max &
         )
   end subroutine

end module