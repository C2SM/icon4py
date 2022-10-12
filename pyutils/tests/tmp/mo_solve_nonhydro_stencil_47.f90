
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_47
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_47( &
         w_nnew, &
         z_contr_w_fl_l, &
         w_concorr_c, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: w_nnew
         real(c_double), dimension(*), target :: z_contr_w_fl_l
         real(c_double), dimension(*), target :: w_concorr_c
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_47( &
         w_nnew, &
         z_contr_w_fl_l, &
         w_concorr_c, &
         w_nnew_before, &
         z_contr_w_fl_l_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         w_nnew_rel_tol, &
         w_nnew_abs_tol, &
         z_contr_w_fl_l_rel_tol, &
         z_contr_w_fl_l_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: w_nnew
         real(c_double), dimension(*), target :: z_contr_w_fl_l
         real(c_double), dimension(*), target :: w_concorr_c
         real(c_double), dimension(*), target :: w_nnew_before
         real(c_double), dimension(*), target :: z_contr_w_fl_l_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: w_nnew_rel_tol
         real(c_double), value, target :: w_nnew_abs_tol
         real(c_double), value, target :: z_contr_w_fl_l_rel_tol
         real(c_double), value, target :: z_contr_w_fl_l_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_47( &
         mesh, &
         k_size, &
         stream, &
         w_nnew_kmax, &
         z_contr_w_fl_l_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: w_nnew_kmax
         integer(c_int), value, target :: z_contr_w_fl_l_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_47() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_47( &
      w_nnew, &
      z_contr_w_fl_l, &
      w_concorr_c, &
      w_nnew_before, &
      z_contr_w_fl_l_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      w_nnew_rel_tol, &
      w_nnew_abs_tol, &
      z_contr_w_fl_l_rel_tol, &
      z_contr_w_fl_l_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: w_nnew
      real(c_double), dimension(:, :), target :: z_contr_w_fl_l
      real(c_double), dimension(:, :), target :: w_concorr_c
      real(c_double), dimension(:, :), target :: w_nnew_before
      real(c_double), dimension(:, :), target :: z_contr_w_fl_l_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: w_nnew_rel_tol
      real(c_double), value, target, optional :: w_nnew_abs_tol
      real(c_double), value, target, optional :: z_contr_w_fl_l_rel_tol
      real(c_double), value, target, optional :: z_contr_w_fl_l_abs_tol

      real(c_double) :: w_nnew_rel_err_tol
      real(c_double) :: w_nnew_abs_err_tol
      real(c_double) :: z_contr_w_fl_l_rel_err_tol
      real(c_double) :: z_contr_w_fl_l_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(w_nnew_rel_tol)) then
         w_nnew_rel_err_tol = w_nnew_rel_tol
      else
         w_nnew_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(w_nnew_abs_tol)) then
         w_nnew_abs_err_tol = w_nnew_abs_tol
      else
         w_nnew_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_contr_w_fl_l_rel_tol)) then
         z_contr_w_fl_l_rel_err_tol = z_contr_w_fl_l_rel_tol
      else
         z_contr_w_fl_l_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_contr_w_fl_l_abs_tol)) then
         z_contr_w_fl_l_abs_err_tol = z_contr_w_fl_l_abs_tol
      else
         z_contr_w_fl_l_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC w_nnew, &
      !$ACC z_contr_w_fl_l, &
      !$ACC w_concorr_c, &
      !$ACC w_nnew_before, &
      !$ACC z_contr_w_fl_l_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_47 &
         ( &
         w_nnew, &
         z_contr_w_fl_l, &
         w_concorr_c, &
         w_nnew_before, &
         z_contr_w_fl_l_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         w_nnew_rel_err_tol, &
         w_nnew_abs_err_tol, &
         z_contr_w_fl_l_rel_err_tol, &
         z_contr_w_fl_l_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_47 &
         ( &
         w_nnew, &
         z_contr_w_fl_l, &
         w_concorr_c, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_47( &
      mesh, &
      k_size, &
      stream, &
      w_nnew_kmax, &
      z_contr_w_fl_l_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: w_nnew_kmax
      integer(c_int), value, target, optional :: z_contr_w_fl_l_kmax

      integer(c_int) :: w_nnew_kvert_max
      integer(c_int) :: z_contr_w_fl_l_kvert_max

      if (present(w_nnew_kmax)) then
         w_nnew_kvert_max = w_nnew_kmax
      else
         w_nnew_kvert_max = k_size
      end if
      if (present(z_contr_w_fl_l_kmax)) then
         z_contr_w_fl_l_kvert_max = z_contr_w_fl_l_kmax
      else
         z_contr_w_fl_l_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_47 &
         ( &
         mesh, &
         k_size, &
         stream, &
         w_nnew_kvert_max, &
         z_contr_w_fl_l_kvert_max &
         )
   end subroutine

end module