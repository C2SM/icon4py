
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_62
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_62( &
         w_now, &
         grf_tend_w, &
         w_new, &
         dtime, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: w_now
         real(c_double), dimension(*), target :: grf_tend_w
         real(c_double), dimension(*), target :: w_new
         real(c_double), value, target :: dtime
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_62( &
         w_now, &
         grf_tend_w, &
         w_new, &
         dtime, &
         w_new_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         w_new_rel_tol, &
         w_new_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: w_now
         real(c_double), dimension(*), target :: grf_tend_w
         real(c_double), dimension(*), target :: w_new
         real(c_double), value, target :: dtime
         real(c_double), dimension(*), target :: w_new_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: w_new_rel_tol
         real(c_double), value, target :: w_new_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_62( &
         mesh, &
         k_size, &
         stream, &
         w_new_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: w_new_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_62() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_62( &
      w_now, &
      grf_tend_w, &
      w_new, &
      dtime, &
      w_new_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      w_new_rel_tol, &
      w_new_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: w_now
      real(c_double), dimension(:, :), target :: grf_tend_w
      real(c_double), dimension(:, :), target :: w_new
      real(c_double), value, target :: dtime
      real(c_double), dimension(:, :), target :: w_new_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: w_new_rel_tol
      real(c_double), value, target, optional :: w_new_abs_tol

      real(c_double) :: w_new_rel_err_tol
      real(c_double) :: w_new_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(w_new_rel_tol)) then
         w_new_rel_err_tol = w_new_rel_tol
      else
         w_new_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(w_new_abs_tol)) then
         w_new_abs_err_tol = w_new_abs_tol
      else
         w_new_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC w_now, &
      !$ACC grf_tend_w, &
      !$ACC w_new, &
      !$ACC w_new_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_62 &
         ( &
         w_now, &
         grf_tend_w, &
         w_new, &
         dtime, &
         w_new_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         w_new_rel_err_tol, &
         w_new_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_62 &
         ( &
         w_now, &
         grf_tend_w, &
         w_new, &
         dtime, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_62( &
      mesh, &
      k_size, &
      stream, &
      w_new_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: w_new_kmax

      integer(c_int) :: w_new_kvert_max

      if (present(w_new_kmax)) then
         w_new_kvert_max = w_new_kmax
      else
         w_new_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_62 &
         ( &
         mesh, &
         k_size, &
         stream, &
         w_new_kvert_max &
         )
   end subroutine

end module