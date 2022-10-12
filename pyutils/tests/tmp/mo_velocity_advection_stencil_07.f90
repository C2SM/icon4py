
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_07
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_07( &
         vn_ie, &
         inv_dual_edge_length, &
         w, &
         z_vt_ie, &
         inv_primal_edge_length, &
         tangent_orientation, &
         z_w_v, &
         z_v_grad_w, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: inv_dual_edge_length
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: z_vt_ie
         real(c_double), dimension(*), target :: inv_primal_edge_length
         real(c_double), dimension(*), target :: tangent_orientation
         real(c_double), dimension(*), target :: z_w_v
         real(c_double), dimension(*), target :: z_v_grad_w
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_07( &
         vn_ie, &
         inv_dual_edge_length, &
         w, &
         z_vt_ie, &
         inv_primal_edge_length, &
         tangent_orientation, &
         z_w_v, &
         z_v_grad_w, &
         z_v_grad_w_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         z_v_grad_w_rel_tol, &
         z_v_grad_w_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: inv_dual_edge_length
         real(c_double), dimension(*), target :: w
         real(c_double), dimension(*), target :: z_vt_ie
         real(c_double), dimension(*), target :: inv_primal_edge_length
         real(c_double), dimension(*), target :: tangent_orientation
         real(c_double), dimension(*), target :: z_w_v
         real(c_double), dimension(*), target :: z_v_grad_w
         real(c_double), dimension(*), target :: z_v_grad_w_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: z_v_grad_w_rel_tol
         real(c_double), value, target :: z_v_grad_w_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_07( &
         mesh, &
         k_size, &
         stream, &
         z_v_grad_w_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: z_v_grad_w_kmax

      end subroutine

      subroutine &
         free_mo_velocity_advection_stencil_07() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_07( &
      vn_ie, &
      inv_dual_edge_length, &
      w, &
      z_vt_ie, &
      inv_primal_edge_length, &
      tangent_orientation, &
      z_w_v, &
      z_v_grad_w, &
      z_v_grad_w_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      z_v_grad_w_rel_tol, &
      z_v_grad_w_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: vn_ie
      real(c_double), dimension(:), target :: inv_dual_edge_length
      real(c_double), dimension(:, :), target :: w
      real(c_double), dimension(:, :), target :: z_vt_ie
      real(c_double), dimension(:), target :: inv_primal_edge_length
      real(c_double), dimension(:), target :: tangent_orientation
      real(c_double), dimension(:, :), target :: z_w_v
      real(c_double), dimension(:, :), target :: z_v_grad_w
      real(c_double), dimension(:, :), target :: z_v_grad_w_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: z_v_grad_w_rel_tol
      real(c_double), value, target, optional :: z_v_grad_w_abs_tol

      real(c_double) :: z_v_grad_w_rel_err_tol
      real(c_double) :: z_v_grad_w_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(z_v_grad_w_rel_tol)) then
         z_v_grad_w_rel_err_tol = z_v_grad_w_rel_tol
      else
         z_v_grad_w_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_v_grad_w_abs_tol)) then
         z_v_grad_w_abs_err_tol = z_v_grad_w_abs_tol
      else
         z_v_grad_w_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC vn_ie, &
      !$ACC inv_dual_edge_length, &
      !$ACC w, &
      !$ACC z_vt_ie, &
      !$ACC inv_primal_edge_length, &
      !$ACC tangent_orientation, &
      !$ACC z_w_v, &
      !$ACC z_v_grad_w, &
      !$ACC z_v_grad_w_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_07 &
         ( &
         vn_ie, &
         inv_dual_edge_length, &
         w, &
         z_vt_ie, &
         inv_primal_edge_length, &
         tangent_orientation, &
         z_w_v, &
         z_v_grad_w, &
         z_v_grad_w_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         z_v_grad_w_rel_err_tol, &
         z_v_grad_w_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_07 &
         ( &
         vn_ie, &
         inv_dual_edge_length, &
         w, &
         z_vt_ie, &
         inv_primal_edge_length, &
         tangent_orientation, &
         z_w_v, &
         z_v_grad_w, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_velocity_advection_stencil_07( &
      mesh, &
      k_size, &
      stream, &
      z_v_grad_w_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: z_v_grad_w_kmax

      integer(c_int) :: z_v_grad_w_kvert_max

      if (present(z_v_grad_w_kmax)) then
         z_v_grad_w_kvert_max = z_v_grad_w_kmax
      else
         z_v_grad_w_kvert_max = k_size
      end if

      call setup_mo_velocity_advection_stencil_07 &
         ( &
         mesh, &
         k_size, &
         stream, &
         z_v_grad_w_kvert_max &
         )
   end subroutine

end module