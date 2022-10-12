
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_14
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_14( &
         ddqz_z_half, &
         z_w_con_c, &
         cfl_clipping, &
         pre_levelmask, &
         vcfl, &
         cfl_w_limit, &
         dtime, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: z_w_con_c
         real(c_double), dimension(*), target :: cfl_clipping
         real(c_double), dimension(*), target :: pre_levelmask
         real(c_double), dimension(*), target :: vcfl
         real(c_double), value, target :: cfl_w_limit
         real(c_double), value, target :: dtime
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_14( &
         ddqz_z_half, &
         z_w_con_c, &
         cfl_clipping, &
         pre_levelmask, &
         vcfl, &
         cfl_w_limit, &
         dtime, &
         z_w_con_c_before, &
         cfl_clipping_before, &
         pre_levelmask_before, &
         vcfl_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         z_w_con_c_rel_tol, &
         z_w_con_c_abs_tol, &
         cfl_clipping_rel_tol, &
         cfl_clipping_abs_tol, &
         pre_levelmask_rel_tol, &
         pre_levelmask_abs_tol, &
         vcfl_rel_tol, &
         vcfl_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: z_w_con_c
         real(c_double), dimension(*), target :: cfl_clipping
         real(c_double), dimension(*), target :: pre_levelmask
         real(c_double), dimension(*), target :: vcfl
         real(c_double), value, target :: cfl_w_limit
         real(c_double), value, target :: dtime
         real(c_double), dimension(*), target :: z_w_con_c_before
         real(c_double), dimension(*), target :: cfl_clipping_before
         real(c_double), dimension(*), target :: pre_levelmask_before
         real(c_double), dimension(*), target :: vcfl_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: z_w_con_c_rel_tol
         real(c_double), value, target :: z_w_con_c_abs_tol
         real(c_double), value, target :: cfl_clipping_rel_tol
         real(c_double), value, target :: cfl_clipping_abs_tol
         real(c_double), value, target :: pre_levelmask_rel_tol
         real(c_double), value, target :: pre_levelmask_abs_tol
         real(c_double), value, target :: vcfl_rel_tol
         real(c_double), value, target :: vcfl_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_14( &
         mesh, &
         k_size, &
         stream, &
         z_w_con_c_kmax, &
         cfl_clipping_kmax, &
         pre_levelmask_kmax, &
         vcfl_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: z_w_con_c_kmax
         integer(c_int), value, target :: cfl_clipping_kmax
         integer(c_int), value, target :: pre_levelmask_kmax
         integer(c_int), value, target :: vcfl_kmax

      end subroutine

      subroutine &
         free_mo_velocity_advection_stencil_14() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_14( &
      ddqz_z_half, &
      z_w_con_c, &
      cfl_clipping, &
      pre_levelmask, &
      vcfl, &
      cfl_w_limit, &
      dtime, &
      z_w_con_c_before, &
      cfl_clipping_before, &
      pre_levelmask_before, &
      vcfl_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      z_w_con_c_rel_tol, &
      z_w_con_c_abs_tol, &
      cfl_clipping_rel_tol, &
      cfl_clipping_abs_tol, &
      pre_levelmask_rel_tol, &
      pre_levelmask_abs_tol, &
      vcfl_rel_tol, &
      vcfl_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: ddqz_z_half
      real(c_double), dimension(:, :), target :: z_w_con_c
      real(c_double), dimension(:, :), target :: cfl_clipping
      real(c_double), dimension(:, :), target :: pre_levelmask
      real(c_double), dimension(:, :), target :: vcfl
      real(c_double), value, target :: cfl_w_limit
      real(c_double), value, target :: dtime
      real(c_double), dimension(:, :), target :: z_w_con_c_before
      real(c_double), dimension(:, :), target :: cfl_clipping_before
      real(c_double), dimension(:, :), target :: pre_levelmask_before
      real(c_double), dimension(:, :), target :: vcfl_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: z_w_con_c_rel_tol
      real(c_double), value, target, optional :: z_w_con_c_abs_tol
      real(c_double), value, target, optional :: cfl_clipping_rel_tol
      real(c_double), value, target, optional :: cfl_clipping_abs_tol
      real(c_double), value, target, optional :: pre_levelmask_rel_tol
      real(c_double), value, target, optional :: pre_levelmask_abs_tol
      real(c_double), value, target, optional :: vcfl_rel_tol
      real(c_double), value, target, optional :: vcfl_abs_tol

      real(c_double) :: z_w_con_c_rel_err_tol
      real(c_double) :: z_w_con_c_abs_err_tol
      real(c_double) :: cfl_clipping_rel_err_tol
      real(c_double) :: cfl_clipping_abs_err_tol
      real(c_double) :: pre_levelmask_rel_err_tol
      real(c_double) :: pre_levelmask_abs_err_tol
      real(c_double) :: vcfl_rel_err_tol
      real(c_double) :: vcfl_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(z_w_con_c_rel_tol)) then
         z_w_con_c_rel_err_tol = z_w_con_c_rel_tol
      else
         z_w_con_c_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_w_con_c_abs_tol)) then
         z_w_con_c_abs_err_tol = z_w_con_c_abs_tol
      else
         z_w_con_c_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(cfl_clipping_rel_tol)) then
         cfl_clipping_rel_err_tol = cfl_clipping_rel_tol
      else
         cfl_clipping_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(cfl_clipping_abs_tol)) then
         cfl_clipping_abs_err_tol = cfl_clipping_abs_tol
      else
         cfl_clipping_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(pre_levelmask_rel_tol)) then
         pre_levelmask_rel_err_tol = pre_levelmask_rel_tol
      else
         pre_levelmask_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(pre_levelmask_abs_tol)) then
         pre_levelmask_abs_err_tol = pre_levelmask_abs_tol
      else
         pre_levelmask_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(vcfl_rel_tol)) then
         vcfl_rel_err_tol = vcfl_rel_tol
      else
         vcfl_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(vcfl_abs_tol)) then
         vcfl_abs_err_tol = vcfl_abs_tol
      else
         vcfl_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC ddqz_z_half, &
      !$ACC z_w_con_c, &
      !$ACC cfl_clipping, &
      !$ACC pre_levelmask, &
      !$ACC vcfl, &
      !$ACC z_w_con_c_before, &
      !$ACC cfl_clipping_before, &
      !$ACC pre_levelmask_before, &
      !$ACC vcfl_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_14 &
         ( &
         ddqz_z_half, &
         z_w_con_c, &
         cfl_clipping, &
         pre_levelmask, &
         vcfl, &
         cfl_w_limit, &
         dtime, &
         z_w_con_c_before, &
         cfl_clipping_before, &
         pre_levelmask_before, &
         vcfl_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         z_w_con_c_rel_err_tol, &
         z_w_con_c_abs_err_tol, &
         cfl_clipping_rel_err_tol, &
         cfl_clipping_abs_err_tol, &
         pre_levelmask_rel_err_tol, &
         pre_levelmask_abs_err_tol, &
         vcfl_rel_err_tol, &
         vcfl_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_14 &
         ( &
         ddqz_z_half, &
         z_w_con_c, &
         cfl_clipping, &
         pre_levelmask, &
         vcfl, &
         cfl_w_limit, &
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
      wrap_setup_mo_velocity_advection_stencil_14( &
      mesh, &
      k_size, &
      stream, &
      z_w_con_c_kmax, &
      cfl_clipping_kmax, &
      pre_levelmask_kmax, &
      vcfl_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: z_w_con_c_kmax
      integer(c_int), value, target, optional :: cfl_clipping_kmax
      integer(c_int), value, target, optional :: pre_levelmask_kmax
      integer(c_int), value, target, optional :: vcfl_kmax

      integer(c_int) :: z_w_con_c_kvert_max
      integer(c_int) :: cfl_clipping_kvert_max
      integer(c_int) :: pre_levelmask_kvert_max
      integer(c_int) :: vcfl_kvert_max

      if (present(z_w_con_c_kmax)) then
         z_w_con_c_kvert_max = z_w_con_c_kmax
      else
         z_w_con_c_kvert_max = k_size
      end if
      if (present(cfl_clipping_kmax)) then
         cfl_clipping_kvert_max = cfl_clipping_kmax
      else
         cfl_clipping_kvert_max = k_size
      end if
      if (present(pre_levelmask_kmax)) then
         pre_levelmask_kvert_max = pre_levelmask_kmax
      else
         pre_levelmask_kvert_max = k_size
      end if
      if (present(vcfl_kmax)) then
         vcfl_kvert_max = vcfl_kmax
      else
         vcfl_kvert_max = k_size
      end if

      call setup_mo_velocity_advection_stencil_14 &
         ( &
         mesh, &
         k_size, &
         stream, &
         z_w_con_c_kvert_max, &
         cfl_clipping_kvert_max, &
         pre_levelmask_kvert_max, &
         vcfl_kvert_max &
         )
   end subroutine

end module