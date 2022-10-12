
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_31
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_31( &
         e_flx_avg, &
         vn, &
         z_vn_avg, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: e_flx_avg
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: z_vn_avg
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_31( &
         e_flx_avg, &
         vn, &
         z_vn_avg, &
         z_vn_avg_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         z_vn_avg_rel_tol, &
         z_vn_avg_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: e_flx_avg
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: z_vn_avg
         real(c_double), dimension(*), target :: z_vn_avg_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: z_vn_avg_rel_tol
         real(c_double), value, target :: z_vn_avg_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_31( &
         mesh, &
         k_size, &
         stream, &
         z_vn_avg_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: z_vn_avg_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_31() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_31( &
      e_flx_avg, &
      vn, &
      z_vn_avg, &
      z_vn_avg_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      z_vn_avg_rel_tol, &
      z_vn_avg_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: e_flx_avg
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:, :), target :: z_vn_avg
      real(c_double), dimension(:, :), target :: z_vn_avg_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: z_vn_avg_rel_tol
      real(c_double), value, target, optional :: z_vn_avg_abs_tol

      real(c_double) :: z_vn_avg_rel_err_tol
      real(c_double) :: z_vn_avg_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(z_vn_avg_rel_tol)) then
         z_vn_avg_rel_err_tol = z_vn_avg_rel_tol
      else
         z_vn_avg_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_vn_avg_abs_tol)) then
         z_vn_avg_abs_err_tol = z_vn_avg_abs_tol
      else
         z_vn_avg_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC e_flx_avg, &
      !$ACC vn, &
      !$ACC z_vn_avg, &
      !$ACC z_vn_avg_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_31 &
         ( &
         e_flx_avg, &
         vn, &
         z_vn_avg, &
         z_vn_avg_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         z_vn_avg_rel_err_tol, &
         z_vn_avg_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_31 &
         ( &
         e_flx_avg, &
         vn, &
         z_vn_avg, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_31( &
      mesh, &
      k_size, &
      stream, &
      z_vn_avg_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: z_vn_avg_kmax

      integer(c_int) :: z_vn_avg_kvert_max

      if (present(z_vn_avg_kmax)) then
         z_vn_avg_kvert_max = z_vn_avg_kmax
      else
         z_vn_avg_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_31 &
         ( &
         mesh, &
         k_size, &
         stream, &
         z_vn_avg_kvert_max &
         )
   end subroutine

end module