
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_06
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_06( &
         wgtfacq_e, &
         vn, &
         vn_ie, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: wgtfacq_e
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: vn_ie
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_06( &
         wgtfacq_e, &
         vn, &
         vn_ie, &
         vn_ie_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         vn_ie_rel_tol, &
         vn_ie_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: wgtfacq_e
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: vn_ie_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: vn_ie_rel_tol
         real(c_double), value, target :: vn_ie_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_06( &
         mesh, &
         k_size, &
         stream, &
         vn_ie_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: vn_ie_kmax

      end subroutine

      subroutine &
         free_mo_velocity_advection_stencil_06() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_06( &
      wgtfacq_e, &
      vn, &
      vn_ie, &
      vn_ie_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      vn_ie_rel_tol, &
      vn_ie_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: wgtfacq_e
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:, :), target :: vn_ie
      real(c_double), dimension(:, :), target :: vn_ie_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: vn_ie_rel_tol
      real(c_double), value, target, optional :: vn_ie_abs_tol

      real(c_double) :: vn_ie_rel_err_tol
      real(c_double) :: vn_ie_abs_err_tol

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

      !$ACC host_data use_device( &
      !$ACC wgtfacq_e, &
      !$ACC vn, &
      !$ACC vn_ie, &
      !$ACC vn_ie_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_06 &
         ( &
         wgtfacq_e, &
         vn, &
         vn_ie, &
         vn_ie_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         vn_ie_rel_err_tol, &
         vn_ie_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_06 &
         ( &
         wgtfacq_e, &
         vn, &
         vn_ie, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_velocity_advection_stencil_06( &
      mesh, &
      k_size, &
      stream, &
      vn_ie_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: vn_ie_kmax

      integer(c_int) :: vn_ie_kvert_max

      if (present(vn_ie_kmax)) then
         vn_ie_kvert_max = vn_ie_kmax
      else
         vn_ie_kvert_max = k_size
      end if

      call setup_mo_velocity_advection_stencil_06 &
         ( &
         mesh, &
         k_size, &
         stream, &
         vn_ie_kvert_max &
         )
   end subroutine

end module