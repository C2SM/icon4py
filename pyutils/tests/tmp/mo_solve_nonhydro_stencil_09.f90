
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_09
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_09( &
         wgtfac_c, &
         z_rth_pr_2, &
         theta_v, &
         vwind_expl_wgt, &
         exner_pr, &
         d_exner_dz_ref_ic, &
         ddqz_z_half, &
         z_theta_v_pr_ic, &
         theta_v_ic, &
         z_th_ddz_exner_c, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: wgtfac_c
         real(c_double), dimension(*), target :: z_rth_pr_2
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: vwind_expl_wgt
         real(c_double), dimension(*), target :: exner_pr
         real(c_double), dimension(*), target :: d_exner_dz_ref_ic
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: z_theta_v_pr_ic
         real(c_double), dimension(*), target :: theta_v_ic
         real(c_double), dimension(*), target :: z_th_ddz_exner_c
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_09( &
         wgtfac_c, &
         z_rth_pr_2, &
         theta_v, &
         vwind_expl_wgt, &
         exner_pr, &
         d_exner_dz_ref_ic, &
         ddqz_z_half, &
         z_theta_v_pr_ic, &
         theta_v_ic, &
         z_th_ddz_exner_c, &
         z_theta_v_pr_ic_before, &
         theta_v_ic_before, &
         z_th_ddz_exner_c_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         z_theta_v_pr_ic_rel_tol, &
         z_theta_v_pr_ic_abs_tol, &
         theta_v_ic_rel_tol, &
         theta_v_ic_abs_tol, &
         z_th_ddz_exner_c_rel_tol, &
         z_th_ddz_exner_c_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: wgtfac_c
         real(c_double), dimension(*), target :: z_rth_pr_2
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: vwind_expl_wgt
         real(c_double), dimension(*), target :: exner_pr
         real(c_double), dimension(*), target :: d_exner_dz_ref_ic
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: z_theta_v_pr_ic
         real(c_double), dimension(*), target :: theta_v_ic
         real(c_double), dimension(*), target :: z_th_ddz_exner_c
         real(c_double), dimension(*), target :: z_theta_v_pr_ic_before
         real(c_double), dimension(*), target :: theta_v_ic_before
         real(c_double), dimension(*), target :: z_th_ddz_exner_c_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: z_theta_v_pr_ic_rel_tol
         real(c_double), value, target :: z_theta_v_pr_ic_abs_tol
         real(c_double), value, target :: theta_v_ic_rel_tol
         real(c_double), value, target :: theta_v_ic_abs_tol
         real(c_double), value, target :: z_th_ddz_exner_c_rel_tol
         real(c_double), value, target :: z_th_ddz_exner_c_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_09( &
         mesh, &
         k_size, &
         stream, &
         z_theta_v_pr_ic_kmax, &
         theta_v_ic_kmax, &
         z_th_ddz_exner_c_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: z_theta_v_pr_ic_kmax
         integer(c_int), value, target :: theta_v_ic_kmax
         integer(c_int), value, target :: z_th_ddz_exner_c_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_09() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_09( &
      wgtfac_c, &
      z_rth_pr_2, &
      theta_v, &
      vwind_expl_wgt, &
      exner_pr, &
      d_exner_dz_ref_ic, &
      ddqz_z_half, &
      z_theta_v_pr_ic, &
      theta_v_ic, &
      z_th_ddz_exner_c, &
      z_theta_v_pr_ic_before, &
      theta_v_ic_before, &
      z_th_ddz_exner_c_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      z_theta_v_pr_ic_rel_tol, &
      z_theta_v_pr_ic_abs_tol, &
      theta_v_ic_rel_tol, &
      theta_v_ic_abs_tol, &
      z_th_ddz_exner_c_rel_tol, &
      z_th_ddz_exner_c_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: wgtfac_c
      real(c_double), dimension(:, :), target :: z_rth_pr_2
      real(c_double), dimension(:, :), target :: theta_v
      real(c_double), dimension(:), target :: vwind_expl_wgt
      real(c_double), dimension(:, :), target :: exner_pr
      real(c_double), dimension(:, :), target :: d_exner_dz_ref_ic
      real(c_double), dimension(:, :), target :: ddqz_z_half
      real(c_double), dimension(:, :), target :: z_theta_v_pr_ic
      real(c_double), dimension(:, :), target :: theta_v_ic
      real(c_double), dimension(:, :), target :: z_th_ddz_exner_c
      real(c_double), dimension(:, :), target :: z_theta_v_pr_ic_before
      real(c_double), dimension(:, :), target :: theta_v_ic_before
      real(c_double), dimension(:, :), target :: z_th_ddz_exner_c_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: z_theta_v_pr_ic_rel_tol
      real(c_double), value, target, optional :: z_theta_v_pr_ic_abs_tol
      real(c_double), value, target, optional :: theta_v_ic_rel_tol
      real(c_double), value, target, optional :: theta_v_ic_abs_tol
      real(c_double), value, target, optional :: z_th_ddz_exner_c_rel_tol
      real(c_double), value, target, optional :: z_th_ddz_exner_c_abs_tol

      real(c_double) :: z_theta_v_pr_ic_rel_err_tol
      real(c_double) :: z_theta_v_pr_ic_abs_err_tol
      real(c_double) :: theta_v_ic_rel_err_tol
      real(c_double) :: theta_v_ic_abs_err_tol
      real(c_double) :: z_th_ddz_exner_c_rel_err_tol
      real(c_double) :: z_th_ddz_exner_c_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(z_theta_v_pr_ic_rel_tol)) then
         z_theta_v_pr_ic_rel_err_tol = z_theta_v_pr_ic_rel_tol
      else
         z_theta_v_pr_ic_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_theta_v_pr_ic_abs_tol)) then
         z_theta_v_pr_ic_abs_err_tol = z_theta_v_pr_ic_abs_tol
      else
         z_theta_v_pr_ic_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(theta_v_ic_rel_tol)) then
         theta_v_ic_rel_err_tol = theta_v_ic_rel_tol
      else
         theta_v_ic_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(theta_v_ic_abs_tol)) then
         theta_v_ic_abs_err_tol = theta_v_ic_abs_tol
      else
         theta_v_ic_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_th_ddz_exner_c_rel_tol)) then
         z_th_ddz_exner_c_rel_err_tol = z_th_ddz_exner_c_rel_tol
      else
         z_th_ddz_exner_c_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_th_ddz_exner_c_abs_tol)) then
         z_th_ddz_exner_c_abs_err_tol = z_th_ddz_exner_c_abs_tol
      else
         z_th_ddz_exner_c_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC wgtfac_c, &
      !$ACC z_rth_pr_2, &
      !$ACC theta_v, &
      !$ACC vwind_expl_wgt, &
      !$ACC exner_pr, &
      !$ACC d_exner_dz_ref_ic, &
      !$ACC ddqz_z_half, &
      !$ACC z_theta_v_pr_ic, &
      !$ACC theta_v_ic, &
      !$ACC z_th_ddz_exner_c, &
      !$ACC z_theta_v_pr_ic_before, &
      !$ACC theta_v_ic_before, &
      !$ACC z_th_ddz_exner_c_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_09 &
         ( &
         wgtfac_c, &
         z_rth_pr_2, &
         theta_v, &
         vwind_expl_wgt, &
         exner_pr, &
         d_exner_dz_ref_ic, &
         ddqz_z_half, &
         z_theta_v_pr_ic, &
         theta_v_ic, &
         z_th_ddz_exner_c, &
         z_theta_v_pr_ic_before, &
         theta_v_ic_before, &
         z_th_ddz_exner_c_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         z_theta_v_pr_ic_rel_err_tol, &
         z_theta_v_pr_ic_abs_err_tol, &
         theta_v_ic_rel_err_tol, &
         theta_v_ic_abs_err_tol, &
         z_th_ddz_exner_c_rel_err_tol, &
         z_th_ddz_exner_c_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_09 &
         ( &
         wgtfac_c, &
         z_rth_pr_2, &
         theta_v, &
         vwind_expl_wgt, &
         exner_pr, &
         d_exner_dz_ref_ic, &
         ddqz_z_half, &
         z_theta_v_pr_ic, &
         theta_v_ic, &
         z_th_ddz_exner_c, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_09( &
      mesh, &
      k_size, &
      stream, &
      z_theta_v_pr_ic_kmax, &
      theta_v_ic_kmax, &
      z_th_ddz_exner_c_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: z_theta_v_pr_ic_kmax
      integer(c_int), value, target, optional :: theta_v_ic_kmax
      integer(c_int), value, target, optional :: z_th_ddz_exner_c_kmax

      integer(c_int) :: z_theta_v_pr_ic_kvert_max
      integer(c_int) :: theta_v_ic_kvert_max
      integer(c_int) :: z_th_ddz_exner_c_kvert_max

      if (present(z_theta_v_pr_ic_kmax)) then
         z_theta_v_pr_ic_kvert_max = z_theta_v_pr_ic_kmax
      else
         z_theta_v_pr_ic_kvert_max = k_size
      end if
      if (present(theta_v_ic_kmax)) then
         theta_v_ic_kvert_max = theta_v_ic_kmax
      else
         theta_v_ic_kvert_max = k_size
      end if
      if (present(z_th_ddz_exner_c_kmax)) then
         z_th_ddz_exner_c_kvert_max = z_th_ddz_exner_c_kmax
      else
         z_th_ddz_exner_c_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_09 &
         ( &
         mesh, &
         k_size, &
         stream, &
         z_theta_v_pr_ic_kvert_max, &
         theta_v_ic_kvert_max, &
         z_th_ddz_exner_c_kvert_max &
         )
   end subroutine

end module