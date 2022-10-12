
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_60
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_60( &
         exner, &
         ddt_exner_phy, &
         exner_dyn_incr, &
         ndyn_substeps_var, &
         dtime, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: exner
         real(c_double), dimension(*), target :: ddt_exner_phy
         real(c_double), dimension(*), target :: exner_dyn_incr
         real(c_double), value, target :: ndyn_substeps_var
         real(c_double), value, target :: dtime
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_60( &
         exner, &
         ddt_exner_phy, &
         exner_dyn_incr, &
         ndyn_substeps_var, &
         dtime, &
         exner_dyn_incr_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         exner_dyn_incr_rel_tol, &
         exner_dyn_incr_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: exner
         real(c_double), dimension(*), target :: ddt_exner_phy
         real(c_double), dimension(*), target :: exner_dyn_incr
         real(c_double), value, target :: ndyn_substeps_var
         real(c_double), value, target :: dtime
         real(c_double), dimension(*), target :: exner_dyn_incr_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: exner_dyn_incr_rel_tol
         real(c_double), value, target :: exner_dyn_incr_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_60( &
         mesh, &
         k_size, &
         stream, &
         exner_dyn_incr_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: exner_dyn_incr_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_60() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_60( &
      exner, &
      ddt_exner_phy, &
      exner_dyn_incr, &
      ndyn_substeps_var, &
      dtime, &
      exner_dyn_incr_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      exner_dyn_incr_rel_tol, &
      exner_dyn_incr_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: exner
      real(c_double), dimension(:, :), target :: ddt_exner_phy
      real(c_double), dimension(:, :), target :: exner_dyn_incr
      real(c_double), value, target :: ndyn_substeps_var
      real(c_double), value, target :: dtime
      real(c_double), dimension(:, :), target :: exner_dyn_incr_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: exner_dyn_incr_rel_tol
      real(c_double), value, target, optional :: exner_dyn_incr_abs_tol

      real(c_double) :: exner_dyn_incr_rel_err_tol
      real(c_double) :: exner_dyn_incr_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(exner_dyn_incr_rel_tol)) then
         exner_dyn_incr_rel_err_tol = exner_dyn_incr_rel_tol
      else
         exner_dyn_incr_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(exner_dyn_incr_abs_tol)) then
         exner_dyn_incr_abs_err_tol = exner_dyn_incr_abs_tol
      else
         exner_dyn_incr_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC exner, &
      !$ACC ddt_exner_phy, &
      !$ACC exner_dyn_incr, &
      !$ACC exner_dyn_incr_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_60 &
         ( &
         exner, &
         ddt_exner_phy, &
         exner_dyn_incr, &
         ndyn_substeps_var, &
         dtime, &
         exner_dyn_incr_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         exner_dyn_incr_rel_err_tol, &
         exner_dyn_incr_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_60 &
         ( &
         exner, &
         ddt_exner_phy, &
         exner_dyn_incr, &
         ndyn_substeps_var, &
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
      wrap_setup_mo_solve_nonhydro_stencil_60( &
      mesh, &
      k_size, &
      stream, &
      exner_dyn_incr_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: exner_dyn_incr_kmax

      integer(c_int) :: exner_dyn_incr_kvert_max

      if (present(exner_dyn_incr_kmax)) then
         exner_dyn_incr_kvert_max = exner_dyn_incr_kmax
      else
         exner_dyn_incr_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_60 &
         ( &
         mesh, &
         k_size, &
         stream, &
         exner_dyn_incr_kvert_max &
         )
   end subroutine

end module