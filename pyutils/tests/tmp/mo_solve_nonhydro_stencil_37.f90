
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_solve_nonhydro_stencil_37
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_solve_nonhydro_stencil_37( &
         vn, &
         vt, &
         vn_ie, &
         z_vt_ie, &
         z_kin_hor_e, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: vt
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: z_vt_ie
         real(c_double), dimension(*), target :: z_kin_hor_e
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_solve_nonhydro_stencil_37( &
         vn, &
         vt, &
         vn_ie, &
         z_vt_ie, &
         z_kin_hor_e, &
         vn_ie_before, &
         z_vt_ie_before, &
         z_kin_hor_e_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         vn_ie_rel_tol, &
         vn_ie_abs_tol, &
         z_vt_ie_rel_tol, &
         z_vt_ie_abs_tol, &
         z_kin_hor_e_rel_tol, &
         z_kin_hor_e_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: vt
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: z_vt_ie
         real(c_double), dimension(*), target :: z_kin_hor_e
         real(c_double), dimension(*), target :: vn_ie_before
         real(c_double), dimension(*), target :: z_vt_ie_before
         real(c_double), dimension(*), target :: z_kin_hor_e_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: vn_ie_rel_tol
         real(c_double), value, target :: vn_ie_abs_tol
         real(c_double), value, target :: z_vt_ie_rel_tol
         real(c_double), value, target :: z_vt_ie_abs_tol
         real(c_double), value, target :: z_kin_hor_e_rel_tol
         real(c_double), value, target :: z_kin_hor_e_abs_tol

      end subroutine

      subroutine &
         setup_mo_solve_nonhydro_stencil_37( &
         mesh, &
         k_size, &
         stream, &
         vn_ie_kmax, &
         z_vt_ie_kmax, &
         z_kin_hor_e_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: vn_ie_kmax
         integer(c_int), value, target :: z_vt_ie_kmax
         integer(c_int), value, target :: z_kin_hor_e_kmax

      end subroutine

      subroutine &
         free_mo_solve_nonhydro_stencil_37() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_solve_nonhydro_stencil_37( &
      vn, &
      vt, &
      vn_ie, &
      z_vt_ie, &
      z_kin_hor_e, &
      vn_ie_before, &
      z_vt_ie_before, &
      z_kin_hor_e_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      vn_ie_rel_tol, &
      vn_ie_abs_tol, &
      z_vt_ie_rel_tol, &
      z_vt_ie_abs_tol, &
      z_kin_hor_e_rel_tol, &
      z_kin_hor_e_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:, :), target :: vt
      real(c_double), dimension(:, :), target :: vn_ie
      real(c_double), dimension(:, :), target :: z_vt_ie
      real(c_double), dimension(:, :), target :: z_kin_hor_e
      real(c_double), dimension(:, :), target :: vn_ie_before
      real(c_double), dimension(:, :), target :: z_vt_ie_before
      real(c_double), dimension(:, :), target :: z_kin_hor_e_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: vn_ie_rel_tol
      real(c_double), value, target, optional :: vn_ie_abs_tol
      real(c_double), value, target, optional :: z_vt_ie_rel_tol
      real(c_double), value, target, optional :: z_vt_ie_abs_tol
      real(c_double), value, target, optional :: z_kin_hor_e_rel_tol
      real(c_double), value, target, optional :: z_kin_hor_e_abs_tol

      real(c_double) :: vn_ie_rel_err_tol
      real(c_double) :: vn_ie_abs_err_tol
      real(c_double) :: z_vt_ie_rel_err_tol
      real(c_double) :: z_vt_ie_abs_err_tol
      real(c_double) :: z_kin_hor_e_rel_err_tol
      real(c_double) :: z_kin_hor_e_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(vn_ie_rel_tol)) then
         vn_ie_rel_err_tol = vn_ie_rel_tol
      else
         vn_ie_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(vn_ie_abs_tol)) then
         vn_ie_abs_err_tol = vn_ie_abs_tol
      else
         vn_ie_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_vt_ie_rel_tol)) then
         z_vt_ie_rel_err_tol = z_vt_ie_rel_tol
      else
         z_vt_ie_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_vt_ie_abs_tol)) then
         z_vt_ie_abs_err_tol = z_vt_ie_abs_tol
      else
         z_vt_ie_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_kin_hor_e_rel_tol)) then
         z_kin_hor_e_rel_err_tol = z_kin_hor_e_rel_tol
      else
         z_kin_hor_e_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_kin_hor_e_abs_tol)) then
         z_kin_hor_e_abs_err_tol = z_kin_hor_e_abs_tol
      else
         z_kin_hor_e_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC vn, &
      !$ACC vt, &
      !$ACC vn_ie, &
      !$ACC z_vt_ie, &
      !$ACC z_kin_hor_e, &
      !$ACC vn_ie_before, &
      !$ACC z_vt_ie_before, &
      !$ACC z_kin_hor_e_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_solve_nonhydro_stencil_37 &
         ( &
         vn, &
         vt, &
         vn_ie, &
         z_vt_ie, &
         z_kin_hor_e, &
         vn_ie_before, &
         z_vt_ie_before, &
         z_kin_hor_e_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         vn_ie_rel_err_tol, &
         vn_ie_abs_err_tol, &
         z_vt_ie_rel_err_tol, &
         z_vt_ie_abs_err_tol, &
         z_kin_hor_e_rel_err_tol, &
         z_kin_hor_e_abs_err_tol &
         )
#else
      call run_mo_solve_nonhydro_stencil_37 &
         ( &
         vn, &
         vt, &
         vn_ie, &
         z_vt_ie, &
         z_kin_hor_e, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_solve_nonhydro_stencil_37( &
      mesh, &
      k_size, &
      stream, &
      vn_ie_kmax, &
      z_vt_ie_kmax, &
      z_kin_hor_e_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: vn_ie_kmax
      integer(c_int), value, target, optional :: z_vt_ie_kmax
      integer(c_int), value, target, optional :: z_kin_hor_e_kmax

      integer(c_int) :: vn_ie_kvert_max
      integer(c_int) :: z_vt_ie_kvert_max
      integer(c_int) :: z_kin_hor_e_kvert_max

      if (present(vn_ie_kmax)) then
         vn_ie_kvert_max = vn_ie_kmax
      else
         vn_ie_kvert_max = k_size
      end if
      if (present(z_vt_ie_kmax)) then
         z_vt_ie_kvert_max = z_vt_ie_kmax
      else
         z_vt_ie_kvert_max = k_size
      end if
      if (present(z_kin_hor_e_kmax)) then
         z_kin_hor_e_kvert_max = z_kin_hor_e_kmax
      else
         z_kin_hor_e_kvert_max = k_size
      end if

      call setup_mo_solve_nonhydro_stencil_37 &
         ( &
         mesh, &
         k_size, &
         stream, &
         vn_ie_kvert_max, &
         z_vt_ie_kvert_max, &
         z_kin_hor_e_kvert_max &
         )
   end subroutine

end module