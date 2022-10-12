
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_01
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_01( &
         vn, &
         rbf_vec_coeff_e, &
         vt, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: rbf_vec_coeff_e
         real(c_double), dimension(*), target :: vt
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_01( &
         vn, &
         rbf_vec_coeff_e, &
         vt, &
         vt_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         vt_rel_tol, &
         vt_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: rbf_vec_coeff_e
         real(c_double), dimension(*), target :: vt
         real(c_double), dimension(*), target :: vt_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: vt_rel_tol
         real(c_double), value, target :: vt_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_01( &
         mesh, &
         k_size, &
         stream, &
         vt_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: vt_kmax

      end subroutine

      subroutine &
         free_mo_velocity_advection_stencil_01() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_01( &
      vn, &
      rbf_vec_coeff_e, &
      vt, &
      vt_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      vt_rel_tol, &
      vt_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:, :), target :: rbf_vec_coeff_e
      real(c_double), dimension(:, :), target :: vt
      real(c_double), dimension(:, :), target :: vt_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: vt_rel_tol
      real(c_double), value, target, optional :: vt_abs_tol

      real(c_double) :: vt_rel_err_tol
      real(c_double) :: vt_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(vt_rel_tol)) then
         vt_rel_err_tol = vt_rel_tol
      else
         vt_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(vt_abs_tol)) then
         vt_abs_err_tol = vt_abs_tol
      else
         vt_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC vn, &
      !$ACC rbf_vec_coeff_e, &
      !$ACC vt, &
      !$ACC vt_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_01 &
         ( &
         vn, &
         rbf_vec_coeff_e, &
         vt, &
         vt_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         vt_rel_err_tol, &
         vt_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_01 &
         ( &
         vn, &
         rbf_vec_coeff_e, &
         vt, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_velocity_advection_stencil_01( &
      mesh, &
      k_size, &
      stream, &
      vt_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: vt_kmax

      integer(c_int) :: vt_kvert_max

      if (present(vt_kmax)) then
         vt_kvert_max = vt_kmax
      else
         vt_kvert_max = k_size
      end if

      call setup_mo_velocity_advection_stencil_01 &
         ( &
         mesh, &
         k_size, &
         stream, &
         vt_kvert_max &
         )
   end subroutine

end module