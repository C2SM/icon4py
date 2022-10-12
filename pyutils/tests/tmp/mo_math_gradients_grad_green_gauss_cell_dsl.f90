
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_math_gradients_grad_green_gauss_cell_dsl
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_math_gradients_grad_green_gauss_cell_dsl( &
         p_grad_1_u, &
         p_grad_1_v, &
         p_grad_2_u, &
         p_grad_2_v, &
         p_ccpr1, &
         p_ccpr2, &
         geofac_grg_x, &
         geofac_grg_y, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: p_grad_1_u
         real(c_double), dimension(*), target :: p_grad_1_v
         real(c_double), dimension(*), target :: p_grad_2_u
         real(c_double), dimension(*), target :: p_grad_2_v
         real(c_double), dimension(*), target :: p_ccpr1
         real(c_double), dimension(*), target :: p_ccpr2
         real(c_double), dimension(*), target :: geofac_grg_x
         real(c_double), dimension(*), target :: geofac_grg_y
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_math_gradients_grad_green_gauss_cell_dsl( &
         p_grad_1_u, &
         p_grad_1_v, &
         p_grad_2_u, &
         p_grad_2_v, &
         p_ccpr1, &
         p_ccpr2, &
         geofac_grg_x, &
         geofac_grg_y, &
         p_grad_1_u_before, &
         p_grad_1_v_before, &
         p_grad_2_u_before, &
         p_grad_2_v_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         p_grad_1_u_rel_tol, &
         p_grad_1_u_abs_tol, &
         p_grad_1_v_rel_tol, &
         p_grad_1_v_abs_tol, &
         p_grad_2_u_rel_tol, &
         p_grad_2_u_abs_tol, &
         p_grad_2_v_rel_tol, &
         p_grad_2_v_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: p_grad_1_u
         real(c_double), dimension(*), target :: p_grad_1_v
         real(c_double), dimension(*), target :: p_grad_2_u
         real(c_double), dimension(*), target :: p_grad_2_v
         real(c_double), dimension(*), target :: p_ccpr1
         real(c_double), dimension(*), target :: p_ccpr2
         real(c_double), dimension(*), target :: geofac_grg_x
         real(c_double), dimension(*), target :: geofac_grg_y
         real(c_double), dimension(*), target :: p_grad_1_u_before
         real(c_double), dimension(*), target :: p_grad_1_v_before
         real(c_double), dimension(*), target :: p_grad_2_u_before
         real(c_double), dimension(*), target :: p_grad_2_v_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: p_grad_1_u_rel_tol
         real(c_double), value, target :: p_grad_1_u_abs_tol
         real(c_double), value, target :: p_grad_1_v_rel_tol
         real(c_double), value, target :: p_grad_1_v_abs_tol
         real(c_double), value, target :: p_grad_2_u_rel_tol
         real(c_double), value, target :: p_grad_2_u_abs_tol
         real(c_double), value, target :: p_grad_2_v_rel_tol
         real(c_double), value, target :: p_grad_2_v_abs_tol

      end subroutine

      subroutine &
         setup_mo_math_gradients_grad_green_gauss_cell_dsl( &
         mesh, &
         k_size, &
         stream, &
         p_grad_1_u_kmax, &
         p_grad_1_v_kmax, &
         p_grad_2_u_kmax, &
         p_grad_2_v_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: p_grad_1_u_kmax
         integer(c_int), value, target :: p_grad_1_v_kmax
         integer(c_int), value, target :: p_grad_2_u_kmax
         integer(c_int), value, target :: p_grad_2_v_kmax

      end subroutine

      subroutine &
         free_mo_math_gradients_grad_green_gauss_cell_dsl() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_math_gradients_grad_green_gauss_cell_dsl( &
      p_grad_1_u, &
      p_grad_1_v, &
      p_grad_2_u, &
      p_grad_2_v, &
      p_ccpr1, &
      p_ccpr2, &
      geofac_grg_x, &
      geofac_grg_y, &
      p_grad_1_u_before, &
      p_grad_1_v_before, &
      p_grad_2_u_before, &
      p_grad_2_v_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      p_grad_1_u_rel_tol, &
      p_grad_1_u_abs_tol, &
      p_grad_1_v_rel_tol, &
      p_grad_1_v_abs_tol, &
      p_grad_2_u_rel_tol, &
      p_grad_2_u_abs_tol, &
      p_grad_2_v_rel_tol, &
      p_grad_2_v_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: p_grad_1_u
      real(c_double), dimension(:, :), target :: p_grad_1_v
      real(c_double), dimension(:, :), target :: p_grad_2_u
      real(c_double), dimension(:, :), target :: p_grad_2_v
      real(c_double), dimension(:, :), target :: p_ccpr1
      real(c_double), dimension(:, :), target :: p_ccpr2
      real(c_double), dimension(:, :), target :: geofac_grg_x
      real(c_double), dimension(:, :), target :: geofac_grg_y
      real(c_double), dimension(:, :), target :: p_grad_1_u_before
      real(c_double), dimension(:, :), target :: p_grad_1_v_before
      real(c_double), dimension(:, :), target :: p_grad_2_u_before
      real(c_double), dimension(:, :), target :: p_grad_2_v_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: p_grad_1_u_rel_tol
      real(c_double), value, target, optional :: p_grad_1_u_abs_tol
      real(c_double), value, target, optional :: p_grad_1_v_rel_tol
      real(c_double), value, target, optional :: p_grad_1_v_abs_tol
      real(c_double), value, target, optional :: p_grad_2_u_rel_tol
      real(c_double), value, target, optional :: p_grad_2_u_abs_tol
      real(c_double), value, target, optional :: p_grad_2_v_rel_tol
      real(c_double), value, target, optional :: p_grad_2_v_abs_tol

      real(c_double) :: p_grad_1_u_rel_err_tol
      real(c_double) :: p_grad_1_u_abs_err_tol
      real(c_double) :: p_grad_1_v_rel_err_tol
      real(c_double) :: p_grad_1_v_abs_err_tol
      real(c_double) :: p_grad_2_u_rel_err_tol
      real(c_double) :: p_grad_2_u_abs_err_tol
      real(c_double) :: p_grad_2_v_rel_err_tol
      real(c_double) :: p_grad_2_v_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(p_grad_1_u_rel_tol)) then
         p_grad_1_u_rel_err_tol = p_grad_1_u_rel_tol
      else
         p_grad_1_u_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_grad_1_u_abs_tol)) then
         p_grad_1_u_abs_err_tol = p_grad_1_u_abs_tol
      else
         p_grad_1_u_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_grad_1_v_rel_tol)) then
         p_grad_1_v_rel_err_tol = p_grad_1_v_rel_tol
      else
         p_grad_1_v_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_grad_1_v_abs_tol)) then
         p_grad_1_v_abs_err_tol = p_grad_1_v_abs_tol
      else
         p_grad_1_v_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_grad_2_u_rel_tol)) then
         p_grad_2_u_rel_err_tol = p_grad_2_u_rel_tol
      else
         p_grad_2_u_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_grad_2_u_abs_tol)) then
         p_grad_2_u_abs_err_tol = p_grad_2_u_abs_tol
      else
         p_grad_2_u_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_grad_2_v_rel_tol)) then
         p_grad_2_v_rel_err_tol = p_grad_2_v_rel_tol
      else
         p_grad_2_v_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_grad_2_v_abs_tol)) then
         p_grad_2_v_abs_err_tol = p_grad_2_v_abs_tol
      else
         p_grad_2_v_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC p_grad_1_u, &
      !$ACC p_grad_1_v, &
      !$ACC p_grad_2_u, &
      !$ACC p_grad_2_v, &
      !$ACC p_ccpr1, &
      !$ACC p_ccpr2, &
      !$ACC geofac_grg_x, &
      !$ACC geofac_grg_y, &
      !$ACC p_grad_1_u_before, &
      !$ACC p_grad_1_v_before, &
      !$ACC p_grad_2_u_before, &
      !$ACC p_grad_2_v_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_math_gradients_grad_green_gauss_cell_dsl &
         ( &
         p_grad_1_u, &
         p_grad_1_v, &
         p_grad_2_u, &
         p_grad_2_v, &
         p_ccpr1, &
         p_ccpr2, &
         geofac_grg_x, &
         geofac_grg_y, &
         p_grad_1_u_before, &
         p_grad_1_v_before, &
         p_grad_2_u_before, &
         p_grad_2_v_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         p_grad_1_u_rel_err_tol, &
         p_grad_1_u_abs_err_tol, &
         p_grad_1_v_rel_err_tol, &
         p_grad_1_v_abs_err_tol, &
         p_grad_2_u_rel_err_tol, &
         p_grad_2_u_abs_err_tol, &
         p_grad_2_v_rel_err_tol, &
         p_grad_2_v_abs_err_tol &
         )
#else
      call run_mo_math_gradients_grad_green_gauss_cell_dsl &
         ( &
         p_grad_1_u, &
         p_grad_1_v, &
         p_grad_2_u, &
         p_grad_2_v, &
         p_ccpr1, &
         p_ccpr2, &
         geofac_grg_x, &
         geofac_grg_y, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_math_gradients_grad_green_gauss_cell_dsl( &
      mesh, &
      k_size, &
      stream, &
      p_grad_1_u_kmax, &
      p_grad_1_v_kmax, &
      p_grad_2_u_kmax, &
      p_grad_2_v_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: p_grad_1_u_kmax
      integer(c_int), value, target, optional :: p_grad_1_v_kmax
      integer(c_int), value, target, optional :: p_grad_2_u_kmax
      integer(c_int), value, target, optional :: p_grad_2_v_kmax

      integer(c_int) :: p_grad_1_u_kvert_max
      integer(c_int) :: p_grad_1_v_kvert_max
      integer(c_int) :: p_grad_2_u_kvert_max
      integer(c_int) :: p_grad_2_v_kvert_max

      if (present(p_grad_1_u_kmax)) then
         p_grad_1_u_kvert_max = p_grad_1_u_kmax
      else
         p_grad_1_u_kvert_max = k_size
      end if
      if (present(p_grad_1_v_kmax)) then
         p_grad_1_v_kvert_max = p_grad_1_v_kmax
      else
         p_grad_1_v_kvert_max = k_size
      end if
      if (present(p_grad_2_u_kmax)) then
         p_grad_2_u_kvert_max = p_grad_2_u_kmax
      else
         p_grad_2_u_kvert_max = k_size
      end if
      if (present(p_grad_2_v_kmax)) then
         p_grad_2_v_kvert_max = p_grad_2_v_kmax
      else
         p_grad_2_v_kvert_max = k_size
      end if

      call setup_mo_math_gradients_grad_green_gauss_cell_dsl &
         ( &
         mesh, &
         k_size, &
         stream, &
         p_grad_1_u_kvert_max, &
         p_grad_1_v_kvert_max, &
         p_grad_2_u_kvert_max, &
         p_grad_2_v_kvert_max &
         )
   end subroutine

end module