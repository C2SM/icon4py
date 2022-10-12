
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_67
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_67( &
         rho, &
         theta_v, &
         exner, &
         rd_o_cvd, &
         rd_o_p0ref, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: rho
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: exner
         real(c_double), value, target :: rd_o_cvd
         real(c_double), value, target :: rd_o_p0ref
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_67( &
         rho, &
         theta_v, &
         exner, &
         rd_o_cvd, &
         rd_o_p0ref, &
         theta_v_before, &
         exner_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         theta_v_rel_tol, &
         theta_v_abs_tol, &
         exner_rel_tol, &
         exner_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: rho
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: exner
         real(c_double), value, target :: rd_o_cvd
         real(c_double), value, target :: rd_o_p0ref
         real(c_double), dimension(*), target :: theta_v_before
         real(c_double), dimension(*), target :: exner_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: theta_v_rel_tol
         real(c_double), value, target :: theta_v_abs_tol
         real(c_double), value, target :: exner_rel_tol
         real(c_double), value, target :: exner_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_67( &
         mesh, &
         k_size, &
         stream, &
         theta_v_kmax, &
         exner_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: theta_v_kmax
         integer(c_int), value, target :: exner_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_67() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_67( &
      rho, &
      theta_v, &
      exner, &
      rd_o_cvd, &
      rd_o_p0ref, &
      theta_v_before, &
      exner_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      theta_v_rel_tol, &
      theta_v_abs_tol, &
      exner_rel_tol, &
      exner_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: rho
      real(c_double), dimension(:, :), target :: theta_v
      real(c_double), dimension(:, :), target :: exner
      real(c_double), value, target :: rd_o_cvd
      real(c_double), value, target :: rd_o_p0ref
      real(c_double), dimension(:, :), target :: theta_v_before
      real(c_double), dimension(:, :), target :: exner_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: theta_v_rel_tol
      real(c_double), value, target, optional :: theta_v_abs_tol
      real(c_double), value, target, optional :: exner_rel_tol
      real(c_double), value, target, optional :: exner_abs_tol

      real(c_double) :: theta_v_rel_err_tol
      real(c_double) :: theta_v_abs_err_tol
      real(c_double) :: exner_rel_err_tol
      real(c_double) :: exner_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(theta_v_rel_tol)) then
         theta_v_rel_err_tol = theta_v_rel_tol
      else
         theta_v_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(theta_v_abs_tol)) then
         theta_v_abs_err_tol = theta_v_abs_tol
      else
         theta_v_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(exner_rel_tol)) then
         exner_rel_err_tol = exner_rel_tol
      else
         exner_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(exner_abs_tol)) then
         exner_abs_err_tol = exner_abs_tol
      else
         exner_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC rho, &
      !$ACC theta_v, &
      !$ACC exner, &
      !$ACC theta_v_before, &
      !$ACC exner_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_67 &
         ( &
         rho, &
         theta_v, &
         exner, &
         rd_o_cvd, &
         rd_o_p0ref, &
         theta_v_before, &
         exner_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         theta_v_rel_err_tol, &
         theta_v_abs_err_tol, &
         exner_rel_err_tol, &
         exner_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_67 &
         ( &
         rho, &
         theta_v, &
         exner, &
         rd_o_cvd, &
         rd_o_p0ref, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_67( &
      mesh, &
      k_size, &
      stream, &
      theta_v_kmax, &
      exner_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: theta_v_kmax
      integer(c_int), value, target, optional :: exner_kmax

      integer(c_int) :: theta_v_kvert_max
      integer(c_int) :: exner_kvert_max

      if (present(theta_v_kmax)) then
         theta_v_kvert_max = theta_v_kmax
      else
         theta_v_kvert_max = k_size
      end if
      if (present(exner_kmax)) then
         exner_kvert_max = exner_kmax
      else
         exner_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_67 &
         ( &
         mesh, &
         k_size, &
         stream, &
         theta_v_kvert_max, &
         exner_kvert_max &
         )
   end subroutine

end module