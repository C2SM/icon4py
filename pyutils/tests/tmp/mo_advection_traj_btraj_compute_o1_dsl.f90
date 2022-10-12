
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_advection_traj_btraj_compute_o1_dsl
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_advection_traj_btraj_compute_o1_dsl( &
         p_vn, &
         p_vt, &
         cell_idx, &
         cell_blk, &
         pos_on_tplane_e_1, &
         pos_on_tplane_e_2, &
         primal_normal_cell_1, &
         dual_normal_cell_1, &
         primal_normal_cell_2, &
         dual_normal_cell_2, &
         p_cell_idx, &
         p_cell_blk, &
         p_distv_bary_1, &
         p_distv_bary_2, &
         p_dthalf, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: p_vn
         real(c_double), dimension(*), target :: p_vt
         integer(c_int), dimension(*), target :: cell_idx
         integer(c_int), dimension(*), target :: cell_blk
         real(c_double), dimension(*), target :: pos_on_tplane_e_1
         real(c_double), dimension(*), target :: pos_on_tplane_e_2
         real(c_double), dimension(*), target :: primal_normal_cell_1
         real(c_double), dimension(*), target :: dual_normal_cell_1
         real(c_double), dimension(*), target :: primal_normal_cell_2
         real(c_double), dimension(*), target :: dual_normal_cell_2
         integer(c_int), dimension(*), target :: p_cell_idx
         integer(c_int), dimension(*), target :: p_cell_blk
         real(c_double), dimension(*), target :: p_distv_bary_1
         real(c_double), dimension(*), target :: p_distv_bary_2
         real(c_double), value, target :: p_dthalf
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_advection_traj_btraj_compute_o1_dsl( &
         p_vn, &
         p_vt, &
         cell_idx, &
         cell_blk, &
         pos_on_tplane_e_1, &
         pos_on_tplane_e_2, &
         primal_normal_cell_1, &
         dual_normal_cell_1, &
         primal_normal_cell_2, &
         dual_normal_cell_2, &
         p_cell_idx, &
         p_cell_blk, &
         p_distv_bary_1, &
         p_distv_bary_2, &
         p_dthalf, &
         p_cell_idx_before, &
         p_cell_blk_before, &
         p_distv_bary_1_before, &
         p_distv_bary_2_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         p_cell_idx_rel_tol, &
         p_cell_idx_abs_tol, &
         p_cell_blk_rel_tol, &
         p_cell_blk_abs_tol, &
         p_distv_bary_1_rel_tol, &
         p_distv_bary_1_abs_tol, &
         p_distv_bary_2_rel_tol, &
         p_distv_bary_2_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: p_vn
         real(c_double), dimension(*), target :: p_vt
         integer(c_int), dimension(*), target :: cell_idx
         integer(c_int), dimension(*), target :: cell_blk
         real(c_double), dimension(*), target :: pos_on_tplane_e_1
         real(c_double), dimension(*), target :: pos_on_tplane_e_2
         real(c_double), dimension(*), target :: primal_normal_cell_1
         real(c_double), dimension(*), target :: dual_normal_cell_1
         real(c_double), dimension(*), target :: primal_normal_cell_2
         real(c_double), dimension(*), target :: dual_normal_cell_2
         integer(c_int), dimension(*), target :: p_cell_idx
         integer(c_int), dimension(*), target :: p_cell_blk
         real(c_double), dimension(*), target :: p_distv_bary_1
         real(c_double), dimension(*), target :: p_distv_bary_2
         real(c_double), value, target :: p_dthalf
         integer(c_int), dimension(*), target :: p_cell_idx_before
         integer(c_int), dimension(*), target :: p_cell_blk_before
         real(c_double), dimension(*), target :: p_distv_bary_1_before
         real(c_double), dimension(*), target :: p_distv_bary_2_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: p_cell_idx_rel_tol
         real(c_double), value, target :: p_cell_idx_abs_tol
         real(c_double), value, target :: p_cell_blk_rel_tol
         real(c_double), value, target :: p_cell_blk_abs_tol
         real(c_double), value, target :: p_distv_bary_1_rel_tol
         real(c_double), value, target :: p_distv_bary_1_abs_tol
         real(c_double), value, target :: p_distv_bary_2_rel_tol
         real(c_double), value, target :: p_distv_bary_2_abs_tol

      end subroutine

      subroutine &
         setup_mo_advection_traj_btraj_compute_o1_dsl( &
         mesh, &
         k_size, &
         stream, &
         p_cell_idx_kmax, &
         p_cell_blk_kmax, &
         p_distv_bary_1_kmax, &
         p_distv_bary_2_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: p_cell_idx_kmax
         integer(c_int), value, target :: p_cell_blk_kmax
         integer(c_int), value, target :: p_distv_bary_1_kmax
         integer(c_int), value, target :: p_distv_bary_2_kmax

      end subroutine

      subroutine &
         free_mo_advection_traj_btraj_compute_o1_dsl() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_advection_traj_btraj_compute_o1_dsl( &
      p_vn, &
      p_vt, &
      cell_idx, &
      cell_blk, &
      pos_on_tplane_e_1, &
      pos_on_tplane_e_2, &
      primal_normal_cell_1, &
      dual_normal_cell_1, &
      primal_normal_cell_2, &
      dual_normal_cell_2, &
      p_cell_idx, &
      p_cell_blk, &
      p_distv_bary_1, &
      p_distv_bary_2, &
      p_dthalf, &
      p_cell_idx_before, &
      p_cell_blk_before, &
      p_distv_bary_1_before, &
      p_distv_bary_2_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      p_cell_idx_rel_tol, &
      p_cell_idx_abs_tol, &
      p_cell_blk_rel_tol, &
      p_cell_blk_abs_tol, &
      p_distv_bary_1_rel_tol, &
      p_distv_bary_1_abs_tol, &
      p_distv_bary_2_rel_tol, &
      p_distv_bary_2_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: p_vn
      real(c_double), dimension(:, :), target :: p_vt
      integer(c_int), dimension(:, :), target :: cell_idx
      integer(c_int), dimension(:, :), target :: cell_blk
      real(c_double), dimension(:, :), target :: pos_on_tplane_e_1
      real(c_double), dimension(:, :), target :: pos_on_tplane_e_2
      real(c_double), dimension(:, :), target :: primal_normal_cell_1
      real(c_double), dimension(:, :), target :: dual_normal_cell_1
      real(c_double), dimension(:, :), target :: primal_normal_cell_2
      real(c_double), dimension(:, :), target :: dual_normal_cell_2
      integer(c_int), dimension(:, :), target :: p_cell_idx
      integer(c_int), dimension(:, :), target :: p_cell_blk
      real(c_double), dimension(:, :), target :: p_distv_bary_1
      real(c_double), dimension(:, :), target :: p_distv_bary_2
      real(c_double), value, target :: p_dthalf
      integer(c_int), dimension(:, :), target :: p_cell_idx_before
      integer(c_int), dimension(:, :), target :: p_cell_blk_before
      real(c_double), dimension(:, :), target :: p_distv_bary_1_before
      real(c_double), dimension(:, :), target :: p_distv_bary_2_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: p_cell_idx_rel_tol
      real(c_double), value, target, optional :: p_cell_idx_abs_tol
      real(c_double), value, target, optional :: p_cell_blk_rel_tol
      real(c_double), value, target, optional :: p_cell_blk_abs_tol
      real(c_double), value, target, optional :: p_distv_bary_1_rel_tol
      real(c_double), value, target, optional :: p_distv_bary_1_abs_tol
      real(c_double), value, target, optional :: p_distv_bary_2_rel_tol
      real(c_double), value, target, optional :: p_distv_bary_2_abs_tol

      real(c_double) :: p_cell_idx_rel_err_tol
      real(c_double) :: p_cell_idx_abs_err_tol
      real(c_double) :: p_cell_blk_rel_err_tol
      real(c_double) :: p_cell_blk_abs_err_tol
      real(c_double) :: p_distv_bary_1_rel_err_tol
      real(c_double) :: p_distv_bary_1_abs_err_tol
      real(c_double) :: p_distv_bary_2_rel_err_tol
      real(c_double) :: p_distv_bary_2_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(p_cell_idx_rel_tol)) then
         p_cell_idx_rel_err_tol = p_cell_idx_rel_tol
      else
         p_cell_idx_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_cell_idx_abs_tol)) then
         p_cell_idx_abs_err_tol = p_cell_idx_abs_tol
      else
         p_cell_idx_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_cell_blk_rel_tol)) then
         p_cell_blk_rel_err_tol = p_cell_blk_rel_tol
      else
         p_cell_blk_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_cell_blk_abs_tol)) then
         p_cell_blk_abs_err_tol = p_cell_blk_abs_tol
      else
         p_cell_blk_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_distv_bary_1_rel_tol)) then
         p_distv_bary_1_rel_err_tol = p_distv_bary_1_rel_tol
      else
         p_distv_bary_1_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_distv_bary_1_abs_tol)) then
         p_distv_bary_1_abs_err_tol = p_distv_bary_1_abs_tol
      else
         p_distv_bary_1_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_distv_bary_2_rel_tol)) then
         p_distv_bary_2_rel_err_tol = p_distv_bary_2_rel_tol
      else
         p_distv_bary_2_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_distv_bary_2_abs_tol)) then
         p_distv_bary_2_abs_err_tol = p_distv_bary_2_abs_tol
      else
         p_distv_bary_2_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC p_vn, &
      !$ACC p_vt, &
      !$ACC cell_idx, &
      !$ACC cell_blk, &
      !$ACC pos_on_tplane_e_1, &
      !$ACC pos_on_tplane_e_2, &
      !$ACC primal_normal_cell_1, &
      !$ACC dual_normal_cell_1, &
      !$ACC primal_normal_cell_2, &
      !$ACC dual_normal_cell_2, &
      !$ACC p_cell_idx, &
      !$ACC p_cell_blk, &
      !$ACC p_distv_bary_1, &
      !$ACC p_distv_bary_2, &
      !$ACC p_cell_idx_before, &
      !$ACC p_cell_blk_before, &
      !$ACC p_distv_bary_1_before, &
      !$ACC p_distv_bary_2_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_advection_traj_btraj_compute_o1_dsl &
         ( &
         p_vn, &
         p_vt, &
         cell_idx, &
         cell_blk, &
         pos_on_tplane_e_1, &
         pos_on_tplane_e_2, &
         primal_normal_cell_1, &
         dual_normal_cell_1, &
         primal_normal_cell_2, &
         dual_normal_cell_2, &
         p_cell_idx, &
         p_cell_blk, &
         p_distv_bary_1, &
         p_distv_bary_2, &
         p_dthalf, &
         p_cell_idx_before, &
         p_cell_blk_before, &
         p_distv_bary_1_before, &
         p_distv_bary_2_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         p_cell_idx_rel_err_tol, &
         p_cell_idx_abs_err_tol, &
         p_cell_blk_rel_err_tol, &
         p_cell_blk_abs_err_tol, &
         p_distv_bary_1_rel_err_tol, &
         p_distv_bary_1_abs_err_tol, &
         p_distv_bary_2_rel_err_tol, &
         p_distv_bary_2_abs_err_tol &
         )
#else
      call run_mo_advection_traj_btraj_compute_o1_dsl &
         ( &
         p_vn, &
         p_vt, &
         cell_idx, &
         cell_blk, &
         pos_on_tplane_e_1, &
         pos_on_tplane_e_2, &
         primal_normal_cell_1, &
         dual_normal_cell_1, &
         primal_normal_cell_2, &
         dual_normal_cell_2, &
         p_cell_idx, &
         p_cell_blk, &
         p_distv_bary_1, &
         p_distv_bary_2, &
         p_dthalf, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_advection_traj_btraj_compute_o1_dsl( &
      mesh, &
      k_size, &
      stream, &
      p_cell_idx_kmax, &
      p_cell_blk_kmax, &
      p_distv_bary_1_kmax, &
      p_distv_bary_2_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: p_cell_idx_kmax
      integer(c_int), value, target, optional :: p_cell_blk_kmax
      integer(c_int), value, target, optional :: p_distv_bary_1_kmax
      integer(c_int), value, target, optional :: p_distv_bary_2_kmax

      integer(c_int) :: p_cell_idx_kvert_max
      integer(c_int) :: p_cell_blk_kvert_max
      integer(c_int) :: p_distv_bary_1_kvert_max
      integer(c_int) :: p_distv_bary_2_kvert_max

      if (present(p_cell_idx_kmax)) then
         p_cell_idx_kvert_max = p_cell_idx_kmax
      else
         p_cell_idx_kvert_max = k_size
      end if
      if (present(p_cell_blk_kmax)) then
         p_cell_blk_kvert_max = p_cell_blk_kmax
      else
         p_cell_blk_kvert_max = k_size
      end if
      if (present(p_distv_bary_1_kmax)) then
         p_distv_bary_1_kvert_max = p_distv_bary_1_kmax
      else
         p_distv_bary_1_kvert_max = k_size
      end if
      if (present(p_distv_bary_2_kmax)) then
         p_distv_bary_2_kvert_max = p_distv_bary_2_kmax
      else
         p_distv_bary_2_kvert_max = k_size
      end if

      call setup_mo_advection_traj_btraj_compute_o1_dsl &
         ( &
         mesh, &
         k_size, &
         stream, &
         p_cell_idx_kvert_max, &
         p_cell_blk_kvert_max, &
         p_distv_bary_1_kvert_max, &
         p_distv_bary_2_kvert_max &
         )
   end subroutine

end module