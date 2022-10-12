
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_18
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_18( &
         levelmask, &
         cfl_clipping, &
         owner_mask, &
         z_w_con_c, &
         ddqz_z_half, &
         area, &
         geofac_n2s, &
         w, &
         ddt_w_adv, &
         scalfac_exdiff, &
         cfl_w_limit, &
         dtime, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         logical(c_int), dimension(*), target :: levelmask
         logical(c_int), dimension(*), target :: cfl_clipping
         logical(c_int), dimension(*), target :: owner_mask
         real(c_double), dimension(*), target :: z_w_con_c
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: area
         real(c_double), dimension(*), target :: geofac_n2s
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: ddt_w_adv
         real(c_double), value, target :: scalfac_exdiff
         real(c_double), value, target :: cfl_w_limit
         real(c_double), value, target :: dtime
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_18( &
         levelmask, &
         cfl_clipping, &
         owner_mask, &
         z_w_con_c, &
         ddqz_z_half, &
         area, &
         geofac_n2s, &
         w, &
         ddt_w_adv, &
         scalfac_exdiff, &
         cfl_w_limit, &
         dtime, &
         ddt_w_adv_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         ddt_w_adv_rel_tol, &
         ddt_w_adv_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         logical(c_int), dimension(*), target :: levelmask
         logical(c_int), dimension(*), target :: cfl_clipping
         logical(c_int), dimension(*), target :: owner_mask
         real(c_double), dimension(*), target :: z_w_con_c
         real(c_double), dimension(*), target :: ddqz_z_half
         real(c_double), dimension(*), target :: area
         real(c_double), dimension(*), target :: geofac_n2s
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: ddt_w_adv
         real(c_double), value, target :: scalfac_exdiff
         real(c_double), value, target :: cfl_w_limit
         real(c_double), value, target :: dtime
         real(c_double), dimension(*), target :: ddt_w_adv_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: ddt_w_adv_rel_tol
         real(c_double), value, target :: ddt_w_adv_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_18( &
         mesh, &
         k_size, &
         stream, &
         ddt_w_adv_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: ddt_w_adv_kmax

      end subroutine

      subroutine &
         free_mo_velocity_advection_stencil_18() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_18( &
      levelmask, &
      cfl_clipping, &
      owner_mask, &
      z_w_con_c, &
      ddqz_z_half, &
      area, &
      geofac_n2s, &
      w, &
      ddt_w_adv, &
      scalfac_exdiff, &
      cfl_w_limit, &
      dtime, &
      ddt_w_adv_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      ddt_w_adv_rel_tol, &
      ddt_w_adv_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      logical(c_int), dimension(:), target :: levelmask
      logical(c_int), dimension(:, :), target :: cfl_clipping
      logical(c_int), dimension(:), target :: owner_mask
      real(c_double), dimension(:, :), target :: z_w_con_c
      real(c_double), dimension(:, :), target :: ddqz_z_half
      real(c_double), dimension(:), target :: area
      real(c_double), dimension(:, :), target :: geofac_n2s
      real(c_double), dimension(:, :), target :: w
      real(c_double), dimension(:, :), target :: ddt_w_adv
      real(c_double), value, target :: scalfac_exdiff
      real(c_double), value, target :: cfl_w_limit
      real(c_double), value, target :: dtime
      real(c_double), dimension(:, :), target :: ddt_w_adv_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: ddt_w_adv_rel_tol
      real(c_double), value, target, optional :: ddt_w_adv_abs_tol

      real(c_double) :: ddt_w_adv_rel_err_tol
      real(c_double) :: ddt_w_adv_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(ddt_w_adv_rel_tol)) then
         ddt_w_adv_rel_err_tol = ddt_w_adv_rel_tol
      else
         ddt_w_adv_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(ddt_w_adv_abs_tol)) then
         ddt_w_adv_abs_err_tol = ddt_w_adv_abs_tol
      else
         ddt_w_adv_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC levelmask, &
      !$ACC cfl_clipping, &
      !$ACC owner_mask, &
      !$ACC z_w_con_c, &
      !$ACC ddqz_z_half, &
      !$ACC area, &
      !$ACC geofac_n2s, &
      !$ACC w, &
      !$ACC ddt_w_adv, &
      !$ACC ddt_w_adv_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_18 &
         ( &
         levelmask, &
         cfl_clipping, &
         owner_mask, &
         z_w_con_c, &
         ddqz_z_half, &
         area, &
         geofac_n2s, &
         w, &
         ddt_w_adv, &
         scalfac_exdiff, &
         cfl_w_limit, &
         dtime, &
         ddt_w_adv_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         ddt_w_adv_rel_err_tol, &
         ddt_w_adv_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_18 &
         ( &
         levelmask, &
         cfl_clipping, &
         owner_mask, &
         z_w_con_c, &
         ddqz_z_half, &
         area, &
         geofac_n2s, &
         w, &
         ddt_w_adv, &
         scalfac_exdiff, &
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
      wrap_setup_mo_velocity_advection_stencil_18( &
      mesh, &
      k_size, &
      stream, &
      ddt_w_adv_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: ddt_w_adv_kmax

      integer(c_int) :: ddt_w_adv_kvert_max

      if (present(ddt_w_adv_kmax)) then
         ddt_w_adv_kvert_max = ddt_w_adv_kmax
      else
         ddt_w_adv_kvert_max = k_size
      end if

      call setup_mo_velocity_advection_stencil_18 &
         ( &
         mesh, &
         k_size, &
         stream, &
         ddt_w_adv_kvert_max &
         )
   end subroutine

end module