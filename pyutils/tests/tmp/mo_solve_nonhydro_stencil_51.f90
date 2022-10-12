
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_51
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_51( &
         z_q, &
         w_nnew, &
         vwind_impl_wgt, &
         theta_v_ic, &
         ddqz_z_half, &
         z_beta, &
         z_alpha, &
         z_w_expl, &
         z_exner_expl, &
         dtime, &
         cpd, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_q
         real(c_double), dimension(*), target :: w_nnew
         real(c_double), dimension(*), target :: vwind_impl_wgt
         real(c_double), dimension(*), target :: theta_v_ic
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: z_beta
         real(c_double), dimension(*), target :: z_alpha
         real(c_double), dimension(*), target :: z_w_expl
         real(c_double), dimension(*), target :: z_exner_expl
         real(c_double), value, target :: dtime
         real(c_double), value, target :: cpd
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_51( &
         z_q, &
         w_nnew, &
         vwind_impl_wgt, &
         theta_v_ic, &
         ddqz_z_half, &
         z_beta, &
         z_alpha, &
         z_w_expl, &
         z_exner_expl, &
         dtime, &
         cpd, &
         z_q_before, &
         w_nnew_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         z_q_rel_tol, &
         z_q_abs_tol, &
         w_nnew_rel_tol, &
         w_nnew_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_q
         real(c_double), dimension(*), target :: w_nnew
         real(c_double), dimension(*), target :: vwind_impl_wgt
         real(c_double), dimension(*), target :: theta_v_ic
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: z_beta
         real(c_double), dimension(*), target :: z_alpha
         real(c_double), dimension(*), target :: z_w_expl
         real(c_double), dimension(*), target :: z_exner_expl
         real(c_double), value, target :: dtime
         real(c_double), value, target :: cpd
         real(c_double), dimension(*), target :: z_q_before
         real(c_double), dimension(*), target :: w_nnew_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: z_q_rel_tol
         real(c_double), value, target :: z_q_abs_tol
         real(c_double), value, target :: w_nnew_rel_tol
         real(c_double), value, target :: w_nnew_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_51( &
         mesh, &
         k_size, &
         stream, &
         z_q_kmax, &
         w_nnew_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: z_q_kmax
         integer(c_int), value, target :: w_nnew_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_51() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_51( &
      z_q, &
      w_nnew, &
      vwind_impl_wgt, &
      theta_v_ic, &
      ddqz_z_half, &
      z_beta, &
      z_alpha, &
      z_w_expl, &
      z_exner_expl, &
      dtime, &
      cpd, &
      z_q_before, &
      w_nnew_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      z_q_rel_tol, &
      z_q_abs_tol, &
      w_nnew_rel_tol, &
      w_nnew_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: z_q
      real(c_double), dimension(:, :), target :: w_nnew
      real(c_double), dimension(:), target :: vwind_impl_wgt
      real(c_double), dimension(:, :), target :: theta_v_ic
      real(c_double), dimension(:, :), target :: ddqz_z_half
      real(c_double), dimension(:, :), target :: z_beta
      real(c_double), dimension(:, :), target :: z_alpha
      real(c_double), dimension(:, :), target :: z_w_expl
      real(c_double), dimension(:, :), target :: z_exner_expl
      real(c_double), value, target :: dtime
      real(c_double), value, target :: cpd
      real(c_double), dimension(:, :), target :: z_q_before
      real(c_double), dimension(:, :), target :: w_nnew_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: z_q_rel_tol
      real(c_double), value, target, optional :: z_q_abs_tol
      real(c_double), value, target, optional :: w_nnew_rel_tol
      real(c_double), value, target, optional :: w_nnew_abs_tol

      real(c_double) :: z_q_rel_err_tol
      real(c_double) :: z_q_abs_err_tol
      real(c_double) :: w_nnew_rel_err_tol
      real(c_double) :: w_nnew_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(z_q_rel_tol)) then
         z_q_rel_err_tol = z_q_rel_tol
      else
         z_q_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_q_abs_tol)) then
         z_q_abs_err_tol = z_q_abs_tol
      else
         z_q_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
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

      !$ACC host_data use_device( &
      !$ACC z_q, &
      !$ACC w_nnew, &
      !$ACC vwind_impl_wgt, &
      !$ACC theta_v_ic, &
      !$ACC ddqz_z_half, &
      !$ACC z_beta, &
      !$ACC z_alpha, &
      !$ACC z_w_expl, &
      !$ACC z_exner_expl, &
      !$ACC z_q_before, &
      !$ACC w_nnew_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_51 &
         ( &
         z_q, &
         w_nnew, &
         vwind_impl_wgt, &
         theta_v_ic, &
         ddqz_z_half, &
         z_beta, &
         z_alpha, &
         z_w_expl, &
         z_exner_expl, &
         dtime, &
         cpd, &
         z_q_before, &
         w_nnew_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         z_q_rel_err_tol, &
         z_q_abs_err_tol, &
         w_nnew_rel_err_tol, &
         w_nnew_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_51 &
         ( &
         z_q, &
         w_nnew, &
         vwind_impl_wgt, &
         theta_v_ic, &
         ddqz_z_half, &
         z_beta, &
         z_alpha, &
         z_w_expl, &
         z_exner_expl, &
         dtime, &
         cpd, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_51( &
      mesh, &
      k_size, &
      stream, &
      z_q_kmax, &
      w_nnew_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: z_q_kmax
      integer(c_int), value, target, optional :: w_nnew_kmax

      integer(c_int) :: z_q_kvert_max
      integer(c_int) :: w_nnew_kvert_max

      if (present(z_q_kmax)) then
         z_q_kvert_max = z_q_kmax
      else
         z_q_kvert_max = k_size
      end if
      if (present(w_nnew_kmax)) then
         w_nnew_kvert_max = w_nnew_kmax
      else
         w_nnew_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_51 &
         ( &
         mesh, &
         k_size, &
         stream, &
         z_q_kvert_max, &
         w_nnew_kvert_max &
         )
   end subroutine

end module