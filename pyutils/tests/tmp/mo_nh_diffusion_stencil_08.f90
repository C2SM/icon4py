
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_nh_diffusion_stencil_08
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_nh_diffusion_stencil_08( &
         w, &
         geofac_grg_x, &
         geofac_grg_y, &
         dwdx, &
         dwdy, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: geofac_grg_x
         real(c_double), dimension(*), target :: geofac_grg_y
         real(c_double), dimension(*), target :: dwdx
         real(c_double), dimension(*), target :: dwdy
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_nh_diffusion_stencil_08( &
         w, &
         geofac_grg_x, &
         geofac_grg_y, &
         dwdx, &
         dwdy, &
         dwdx_before, &
         dwdy_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         dwdx_rel_tol, &
         dwdx_abs_tol, &
         dwdy_rel_tol, &
         dwdy_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: geofac_grg_x
         real(c_double), dimension(*), target :: geofac_grg_y
         real(c_double), dimension(*), target :: dwdx
         real(c_double), dimension(*), target :: dwdy
         real(c_double), dimension(*), target :: dwdx_before
         real(c_double), dimension(*), target :: dwdy_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: dwdx_rel_tol
         real(c_double), value, target :: dwdx_abs_tol
         real(c_double), value, target :: dwdy_rel_tol
         real(c_double), value, target :: dwdy_abs_tol

      end subroutine

      subroutine &
         setup_mo_nh_diffusion_stencil_08( &
         mesh, &
         k_size, &
         stream, &
         dwdx_kmax, &
         dwdy_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: dwdx_kmax
         integer(c_int), value, target :: dwdy_kmax

      end subroutine

      subroutine &
         free_mo_nh_diffusion_stencil_08() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_nh_diffusion_stencil_08( &
      w, &
      geofac_grg_x, &
      geofac_grg_y, &
      dwdx, &
      dwdy, &
      dwdx_before, &
      dwdy_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      dwdx_rel_tol, &
      dwdx_abs_tol, &
      dwdy_rel_tol, &
      dwdy_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: w
      real(c_double), dimension(:, :), target :: geofac_grg_x
      real(c_double), dimension(:, :), target :: geofac_grg_y
      real(c_double), dimension(:, :), target :: dwdx
      real(c_double), dimension(:, :), target :: dwdy
      real(c_double), dimension(:, :), target :: dwdx_before
      real(c_double), dimension(:, :), target :: dwdy_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: dwdx_rel_tol
      real(c_double), value, target, optional :: dwdx_abs_tol
      real(c_double), value, target, optional :: dwdy_rel_tol
      real(c_double), value, target, optional :: dwdy_abs_tol

      real(c_double) :: dwdx_rel_err_tol
      real(c_double) :: dwdx_abs_err_tol
      real(c_double) :: dwdy_rel_err_tol
      real(c_double) :: dwdy_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(dwdx_rel_tol)) then
         dwdx_rel_err_tol = dwdx_rel_tol
      else
         dwdx_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(dwdx_abs_tol)) then
         dwdx_abs_err_tol = dwdx_abs_tol
      else
         dwdx_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(dwdy_rel_tol)) then
         dwdy_rel_err_tol = dwdy_rel_tol
      else
         dwdy_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(dwdy_abs_tol)) then
         dwdy_abs_err_tol = dwdy_abs_tol
      else
         dwdy_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC w, &
      !$ACC geofac_grg_x, &
      !$ACC geofac_grg_y, &
      !$ACC dwdx, &
      !$ACC dwdy, &
      !$ACC dwdx_before, &
      !$ACC dwdy_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_nh_diffusion_stencil_08 &
         ( &
         w, &
         geofac_grg_x, &
         geofac_grg_y, &
         dwdx, &
         dwdy, &
         dwdx_before, &
         dwdy_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         dwdx_rel_err_tol, &
         dwdx_abs_err_tol, &
         dwdy_rel_err_tol, &
         dwdy_abs_err_tol &
         )
#else
      call run_mo_nh_diffusion_stencil_08 &
         ( &
         w, &
         geofac_grg_x, &
         geofac_grg_y, &
         dwdx, &
         dwdy, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_nh_diffusion_stencil_08( &
      mesh, &
      k_size, &
      stream, &
      dwdx_kmax, &
      dwdy_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: dwdx_kmax
      integer(c_int), value, target, optional :: dwdy_kmax

      integer(c_int) :: dwdx_kvert_max
      integer(c_int) :: dwdy_kvert_max

      if (present(dwdx_kmax)) then
         dwdx_kvert_max = dwdx_kmax
      else
         dwdx_kvert_max = k_size
      end if
      if (present(dwdy_kmax)) then
         dwdy_kvert_max = dwdy_kmax
      else
         dwdy_kvert_max = k_size
      end if

      call setup_mo_nh_diffusion_stencil_08 &
         ( &
         mesh, &
         k_size, &
         stream, &
         dwdx_kvert_max, &
         dwdy_kvert_max &
         )
   end subroutine

end module