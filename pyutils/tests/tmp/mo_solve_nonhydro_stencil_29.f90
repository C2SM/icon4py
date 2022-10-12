
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_29
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_29( &
         grf_tend_vn, &
         vn_now, &
         vn_new, &
         dtime, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: grf_tend_vn
         real(c_double), dimension(*), target :: vn_now
         real(c_double), dimension(*), target :: vn_new
         real(c_double), value, target :: dtime
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_29( &
         grf_tend_vn, &
         vn_now, &
         vn_new, &
         dtime, &
         vn_new_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         vn_new_rel_tol, &
         vn_new_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: grf_tend_vn
         real(c_double), dimension(*), target :: vn_now
         real(c_double), dimension(*), target :: vn_new
         real(c_double), value, target :: dtime
         real(c_double), dimension(*), target :: vn_new_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: vn_new_rel_tol
         real(c_double), value, target :: vn_new_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_29( &
         mesh, &
         k_size, &
         stream, &
         vn_new_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: vn_new_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_29() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_29( &
      grf_tend_vn, &
      vn_now, &
      vn_new, &
      dtime, &
      vn_new_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      vn_new_rel_tol, &
      vn_new_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: grf_tend_vn
      real(c_double), dimension(:, :), target :: vn_now
      real(c_double), dimension(:, :), target :: vn_new
      real(c_double), value, target :: dtime
      real(c_double), dimension(:, :), target :: vn_new_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: vn_new_rel_tol
      real(c_double), value, target, optional :: vn_new_abs_tol

      real(c_double) :: vn_new_rel_err_tol
      real(c_double) :: vn_new_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(vn_new_rel_tol)) then
         vn_new_rel_err_tol = vn_new_rel_tol
      else
         vn_new_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(vn_new_abs_tol)) then
         vn_new_abs_err_tol = vn_new_abs_tol
      else
         vn_new_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC grf_tend_vn, &
      !$ACC vn_now, &
      !$ACC vn_new, &
      !$ACC vn_new_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_29 &
         ( &
         grf_tend_vn, &
         vn_now, &
         vn_new, &
         dtime, &
         vn_new_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         vn_new_rel_err_tol, &
         vn_new_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_29 &
         ( &
         grf_tend_vn, &
         vn_now, &
         vn_new, &
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
      wrap_setup_mo_solve_nonhydro_stencil_29( &
      mesh, &
      k_size, &
      stream, &
      vn_new_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: vn_new_kmax

      integer(c_int) :: vn_new_kvert_max

      if (present(vn_new_kmax)) then
         vn_new_kvert_max = vn_new_kmax
      else
         vn_new_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_29 &
         ( &
         mesh, &
         k_size, &
         stream, &
         vn_new_kvert_max &
         )
   end subroutine

end module