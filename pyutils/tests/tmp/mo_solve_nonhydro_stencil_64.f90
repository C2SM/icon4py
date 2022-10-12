
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_64
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_64( &
         mass_flx_ic, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: mass_flx_ic
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_64( &
         mass_flx_ic, &
         mass_flx_ic_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         mass_flx_ic_rel_tol, &
         mass_flx_ic_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: mass_flx_ic
         real(c_double), dimension(*), target :: mass_flx_ic_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: mass_flx_ic_rel_tol
         real(c_double), value, target :: mass_flx_ic_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_64( &
         mesh, &
         k_size, &
         stream, &
         mass_flx_ic_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: mass_flx_ic_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_64() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_64( &
      mass_flx_ic, &
      mass_flx_ic_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      mass_flx_ic_rel_tol, &
      mass_flx_ic_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: mass_flx_ic
      real(c_double), dimension(:, :), target :: mass_flx_ic_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: mass_flx_ic_rel_tol
      real(c_double), value, target, optional :: mass_flx_ic_abs_tol

      real(c_double) :: mass_flx_ic_rel_err_tol
      real(c_double) :: mass_flx_ic_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(mass_flx_ic_rel_tol)) then
         mass_flx_ic_rel_err_tol = mass_flx_ic_rel_tol
      else
         mass_flx_ic_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(mass_flx_ic_abs_tol)) then
         mass_flx_ic_abs_err_tol = mass_flx_ic_abs_tol
      else
         mass_flx_ic_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC mass_flx_ic, &
      !$ACC mass_flx_ic_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_64 &
         ( &
         mass_flx_ic, &
         mass_flx_ic_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         mass_flx_ic_rel_err_tol, &
         mass_flx_ic_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_64 &
         ( &
         mass_flx_ic, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_64( &
      mesh, &
      k_size, &
      stream, &
      mass_flx_ic_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: mass_flx_ic_kmax

      integer(c_int) :: mass_flx_ic_kvert_max

      if (present(mass_flx_ic_kmax)) then
         mass_flx_ic_kvert_max = mass_flx_ic_kmax
      else
         mass_flx_ic_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_64 &
         ( &
         mesh, &
         k_size, &
         stream, &
         mass_flx_ic_kvert_max &
         )
   end subroutine

end module