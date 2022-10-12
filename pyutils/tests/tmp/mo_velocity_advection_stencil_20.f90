
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_20
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_20( &
         levelmask, &
         c_lin_e, &
         z_w_con_c_full, &
         ddqz_z_full_e, &
         area_edge, &
         tangent_orientation, &
         inv_primal_edge_length, &
         zeta, &
         geofac_grdiv, &
         vn, &
         ddt_vn_adv, &
         cfl_w_limit, &
         scalfac_exdiff, &
         d_time, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         logical(c_int), dimension(*), target :: levelmask
         real(c_double), dimension(*), target :: c_lin_e
         real(c_double), dimension(*), target :: z_w_con_c_full
         real(c_double), dimension(*), target :: ddqz_z_full_e
         real(c_double), dimension(*), target :: area_edge
         real(c_double), dimension(*), target :: tangent_orientation
         real(c_double), dimension(*), target :: inv_primal_edge_length
         real(c_double), dimension(*), target :: zeta
         real(c_double), dimension(*), target :: geofac_grdiv
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: ddt_vn_adv
         real(c_double), value, target :: cfl_w_limit
         real(c_double), value, target :: scalfac_exdiff
         real(c_double), value, target :: d_time
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_20( &
         levelmask, &
         c_lin_e, &
         z_w_con_c_full, &
         ddqz_z_full_e, &
         area_edge, &
         tangent_orientation, &
         inv_primal_edge_length, &
         zeta, &
         geofac_grdiv, &
         vn, &
         ddt_vn_adv, &
         cfl_w_limit, &
         scalfac_exdiff, &
         d_time, &
         ddt_vn_adv_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         ddt_vn_adv_rel_tol, &
         ddt_vn_adv_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         logical(c_int), dimension(*), target :: levelmask
         real(c_double), dimension(*), target :: c_lin_e
         real(c_double), dimension(*), target :: z_w_con_c_full
         real(c_double), dimension(*), target :: ddqz_z_full_e
         real(c_double), dimension(*), target :: area_edge
         real(c_double), dimension(*), target :: tangent_orientation
         real(c_double), dimension(*), target :: inv_primal_edge_length
         real(c_double), dimension(*), target :: zeta
         real(c_double), dimension(*), target :: geofac_grdiv
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: ddt_vn_adv
         real(c_double), value, target :: cfl_w_limit
         real(c_double), value, target :: scalfac_exdiff
         real(c_double), value, target :: d_time
         real(c_double), dimension(*), target :: ddt_vn_adv_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: ddt_vn_adv_rel_tol
         real(c_double), value, target :: ddt_vn_adv_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_20( &
         mesh, &
         k_size, &
         stream, &
         ddt_vn_adv_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: ddt_vn_adv_kmax

      end subroutine

      subroutine &
         free_mo_velocity_advection_stencil_20() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_20( &
      levelmask, &
      c_lin_e, &
      z_w_con_c_full, &
      ddqz_z_full_e, &
      area_edge, &
      tangent_orientation, &
      inv_primal_edge_length, &
      zeta, &
      geofac_grdiv, &
      vn, &
      ddt_vn_adv, &
      cfl_w_limit, &
      scalfac_exdiff, &
      d_time, &
      ddt_vn_adv_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      ddt_vn_adv_rel_tol, &
      ddt_vn_adv_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      logical(c_int), dimension(:), target :: levelmask
      real(c_double), dimension(:, :), target :: c_lin_e
      real(c_double), dimension(:, :), target :: z_w_con_c_full
      real(c_double), dimension(:, :), target :: ddqz_z_full_e
      real(c_double), dimension(:), target :: area_edge
      real(c_double), dimension(:), target :: tangent_orientation
      real(c_double), dimension(:), target :: inv_primal_edge_length
      real(c_double), dimension(:, :), target :: zeta
      real(c_double), dimension(:, :), target :: geofac_grdiv
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:, :), target :: ddt_vn_adv
      real(c_double), value, target :: cfl_w_limit
      real(c_double), value, target :: scalfac_exdiff
      real(c_double), value, target :: d_time
      real(c_double), dimension(:, :), target :: ddt_vn_adv_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: ddt_vn_adv_rel_tol
      real(c_double), value, target, optional :: ddt_vn_adv_abs_tol

      real(c_double) :: ddt_vn_adv_rel_err_tol
      real(c_double) :: ddt_vn_adv_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(ddt_vn_adv_rel_tol)) then
         ddt_vn_adv_rel_err_tol = ddt_vn_adv_rel_tol
      else
         ddt_vn_adv_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(ddt_vn_adv_abs_tol)) then
         ddt_vn_adv_abs_err_tol = ddt_vn_adv_abs_tol
      else
         ddt_vn_adv_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC levelmask, &
      !$ACC c_lin_e, &
      !$ACC z_w_con_c_full, &
      !$ACC ddqz_z_full_e, &
      !$ACC area_edge, &
      !$ACC tangent_orientation, &
      !$ACC inv_primal_edge_length, &
      !$ACC zeta, &
      !$ACC geofac_grdiv, &
      !$ACC vn, &
      !$ACC ddt_vn_adv, &
      !$ACC ddt_vn_adv_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_20 &
         ( &
         levelmask, &
         c_lin_e, &
         z_w_con_c_full, &
         ddqz_z_full_e, &
         area_edge, &
         tangent_orientation, &
         inv_primal_edge_length, &
         zeta, &
         geofac_grdiv, &
         vn, &
         ddt_vn_adv, &
         cfl_w_limit, &
         scalfac_exdiff, &
         d_time, &
         ddt_vn_adv_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         ddt_vn_adv_rel_err_tol, &
         ddt_vn_adv_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_20 &
         ( &
         levelmask, &
         c_lin_e, &
         z_w_con_c_full, &
         ddqz_z_full_e, &
         area_edge, &
         tangent_orientation, &
         inv_primal_edge_length, &
         zeta, &
         geofac_grdiv, &
         vn, &
         ddt_vn_adv, &
         cfl_w_limit, &
         scalfac_exdiff, &
         d_time, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_velocity_advection_stencil_20( &
      mesh, &
      k_size, &
      stream, &
      ddt_vn_adv_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: ddt_vn_adv_kmax

      integer(c_int) :: ddt_vn_adv_kvert_max

      if (present(ddt_vn_adv_kmax)) then
         ddt_vn_adv_kvert_max = ddt_vn_adv_kmax
      else
         ddt_vn_adv_kvert_max = k_size
      end if

      call setup_mo_velocity_advection_stencil_20 &
         ( &
         mesh, &
         k_size, &
         stream, &
         ddt_vn_adv_kvert_max &
         )
   end subroutine

end module