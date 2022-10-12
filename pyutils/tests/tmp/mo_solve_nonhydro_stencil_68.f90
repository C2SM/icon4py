
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_68
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_68( &
         mask_prog_halo_c, &
         rho_now, &
         theta_v_now, &
         exner_new, &
         exner_now, &
         rho_new, &
         theta_v_new, &
         cvd_o_rd, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         logical(c_int), dimension(*), target :: mask_prog_halo_c
         real(c_double), dimension(*), target :: rho_now
         real(c_double), dimension(*), target :: theta_v_now
         real(c_double), dimension(*), target :: exner_new
         real(c_double), dimension(*), target :: exner_now
         real(c_double), dimension(*), target :: rho_new
         real(c_double), dimension(*), target :: theta_v_new
         real(c_double), value, target :: cvd_o_rd
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_68( &
         mask_prog_halo_c, &
         rho_now, &
         theta_v_now, &
         exner_new, &
         exner_now, &
         rho_new, &
         theta_v_new, &
         cvd_o_rd, &
         theta_v_new_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         theta_v_new_rel_tol, &
         theta_v_new_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         logical(c_int), dimension(*), target :: mask_prog_halo_c
         real(c_double), dimension(*), target :: rho_now
         real(c_double), dimension(*), target :: theta_v_now
         real(c_double), dimension(*), target :: exner_new
         real(c_double), dimension(*), target :: exner_now
         real(c_double), dimension(*), target :: rho_new
         real(c_double), dimension(*), target :: theta_v_new
         real(c_double), value, target :: cvd_o_rd
         real(c_double), dimension(*), target :: theta_v_new_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: theta_v_new_rel_tol
         real(c_double), value, target :: theta_v_new_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_68( &
         mesh, &
         k_size, &
         stream, &
         theta_v_new_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: theta_v_new_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_68() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_68( &
      mask_prog_halo_c, &
      rho_now, &
      theta_v_now, &
      exner_new, &
      exner_now, &
      rho_new, &
      theta_v_new, &
      cvd_o_rd, &
      theta_v_new_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      theta_v_new_rel_tol, &
      theta_v_new_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      logical(c_int), dimension(:), target :: mask_prog_halo_c
      real(c_double), dimension(:, :), target :: rho_now
      real(c_double), dimension(:, :), target :: theta_v_now
      real(c_double), dimension(:, :), target :: exner_new
      real(c_double), dimension(:, :), target :: exner_now
      real(c_double), dimension(:, :), target :: rho_new
      real(c_double), dimension(:, :), target :: theta_v_new
      real(c_double), value, target :: cvd_o_rd
      real(c_double), dimension(:, :), target :: theta_v_new_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: theta_v_new_rel_tol
      real(c_double), value, target, optional :: theta_v_new_abs_tol

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
      !$ACC mask_prog_halo_c, &
      !$ACC rho_now, &
      !$ACC theta_v_now, &
      !$ACC exner_new, &
      !$ACC exner_now, &
      !$ACC rho_new, &
      !$ACC theta_v_new, &
      !$ACC theta_v_new_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_68 &
         ( &
         mask_prog_halo_c, &
         rho_now, &
         theta_v_now, &
         exner_new, &
         exner_now, &
         rho_new, &
         theta_v_new, &
         cvd_o_rd, &
         theta_v_new_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         theta_v_new_rel_err_tol, &
         theta_v_new_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_68 &
         ( &
         mask_prog_halo_c, &
         rho_now, &
         theta_v_now, &
         exner_new, &
         exner_now, &
         rho_new, &
         theta_v_new, &
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
      wrap_setup_mo_solve_nonhydro_stencil_68( &
      mesh, &
      k_size, &
      stream, &
      theta_v_new_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: theta_v_new_kmax

      integer(c_int) :: theta_v_new_kvert_max

      if (present(theta_v_new_kmax)) then
         theta_v_new_kvert_max = theta_v_new_kmax
      else
         theta_v_new_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_68 &
         ( &
         mesh, &
         k_size, &
         stream, &
         theta_v_new_kvert_max &
         )
   end subroutine

end module