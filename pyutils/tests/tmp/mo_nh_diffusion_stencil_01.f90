
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_nh_diffusion_stencil_01
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_nh_diffusion_stencil_01( &
         diff_multfac_smag, &
         tangent_orientation, &
         inv_primal_edge_length, &
         inv_vert_vert_length, &
         u_vert, &
         v_vert, &
         primal_normal_vert_x, &
         primal_normal_vert_y, &
         dual_normal_vert_x, &
         dual_normal_vert_y, &
         vn, &
         smag_limit, &
         kh_smag_e, &
         kh_smag_ec, &
         z_nabla2_e, &
         smag_offset, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: diff_multfac_smag
         real(c_double), dimension(*), target :: tangent_orientation
         real(c_double), dimension(*), target :: inv_primal_edge_length
         real(c_double), dimension(*), target :: inv_vert_vert_length
         real(c_double), dimension(*), target :: u_vert
         real(c_double), dimension(*), target :: v_vert
         real(c_double), dimension(*), target :: primal_normal_vert_x
         real(c_double), dimension(*), target :: primal_normal_vert_y
         real(c_double), dimension(*), target :: dual_normal_vert_x
         real(c_double), dimension(*), target :: dual_normal_vert_y
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: smag_limit
         real(c_double), dimension(*), target :: kh_smag_e
         real(c_double), dimension(*), target :: kh_smag_ec
         real(c_double), dimension(*), target :: z_nabla2_e
         real(c_double), value, target :: smag_offset
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_nh_diffusion_stencil_01( &
         diff_multfac_smag, &
         tangent_orientation, &
         inv_primal_edge_length, &
         inv_vert_vert_length, &
         u_vert, &
         v_vert, &
         primal_normal_vert_x, &
         primal_normal_vert_y, &
         dual_normal_vert_x, &
         dual_normal_vert_y, &
         vn, &
         smag_limit, &
         kh_smag_e, &
         kh_smag_ec, &
         z_nabla2_e, &
         smag_offset, &
         kh_smag_e_before, &
         kh_smag_ec_before, &
         z_nabla2_e_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         kh_smag_e_rel_tol, &
         kh_smag_e_abs_tol, &
         kh_smag_ec_rel_tol, &
         kh_smag_ec_abs_tol, &
         z_nabla2_e_rel_tol, &
         z_nabla2_e_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: diff_multfac_smag
         real(c_double), dimension(*), target :: tangent_orientation
         real(c_double), dimension(*), target :: inv_primal_edge_length
         real(c_double), dimension(*), target :: inv_vert_vert_length
         real(c_double), dimension(*), target :: u_vert
         real(c_double), dimension(*), target :: v_vert
         real(c_double), dimension(*), target :: primal_normal_vert_x
         real(c_double), dimension(*), target :: primal_normal_vert_y
         real(c_double), dimension(*), target :: dual_normal_vert_x
         real(c_double), dimension(*), target :: dual_normal_vert_y
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: smag_limit
         real(c_double), dimension(*), target :: kh_smag_e
         real(c_double), dimension(*), target :: kh_smag_ec
         real(c_double), dimension(*), target :: z_nabla2_e
         real(c_double), value, target :: smag_offset
         real(c_double), dimension(*), target :: kh_smag_e_before
         real(c_double), dimension(*), target :: kh_smag_ec_before
         real(c_double), dimension(*), target :: z_nabla2_e_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: kh_smag_e_rel_tol
         real(c_double), value, target :: kh_smag_e_abs_tol
         real(c_double), value, target :: kh_smag_ec_rel_tol
         real(c_double), value, target :: kh_smag_ec_abs_tol
         real(c_double), value, target :: z_nabla2_e_rel_tol
         real(c_double), value, target :: z_nabla2_e_abs_tol

      end subroutine

      subroutine &
         setup_mo_nh_diffusion_stencil_01( &
         mesh, &
         k_size, &
         stream, &
         kh_smag_e_kmax, &
         kh_smag_ec_kmax, &
         z_nabla2_e_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: kh_smag_e_kmax
         integer(c_int), value, target :: kh_smag_ec_kmax
         integer(c_int), value, target :: z_nabla2_e_kmax

      end subroutine

      subroutine &
         free_mo_nh_diffusion_stencil_01() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_nh_diffusion_stencil_01( &
      diff_multfac_smag, &
      tangent_orientation, &
      inv_primal_edge_length, &
      inv_vert_vert_length, &
      u_vert, &
      v_vert, &
      primal_normal_vert_x, &
      primal_normal_vert_y, &
      dual_normal_vert_x, &
      dual_normal_vert_y, &
      vn, &
      smag_limit, &
      kh_smag_e, &
      kh_smag_ec, &
      z_nabla2_e, &
      smag_offset, &
      kh_smag_e_before, &
      kh_smag_ec_before, &
      z_nabla2_e_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      kh_smag_e_rel_tol, &
      kh_smag_e_abs_tol, &
      kh_smag_ec_rel_tol, &
      kh_smag_ec_abs_tol, &
      z_nabla2_e_rel_tol, &
      z_nabla2_e_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:), target :: diff_multfac_smag
      real(c_double), dimension(:), target :: tangent_orientation
      real(c_double), dimension(:), target :: inv_primal_edge_length
      real(c_double), dimension(:), target :: inv_vert_vert_length
      real(c_double), dimension(:, :), target :: u_vert
      real(c_double), dimension(:, :), target :: v_vert
      real(c_double), dimension(:, :), target :: primal_normal_vert_x
      real(c_double), dimension(:, :), target :: primal_normal_vert_y
      real(c_double), dimension(:, :), target :: dual_normal_vert_x
      real(c_double), dimension(:, :), target :: dual_normal_vert_y
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:), target :: smag_limit
      real(c_double), dimension(:, :), target :: kh_smag_e
      real(c_double), dimension(:, :), target :: kh_smag_ec
      real(c_double), dimension(:, :), target :: z_nabla2_e
      real(c_double), value, target :: smag_offset
      real(c_double), dimension(:, :), target :: kh_smag_e_before
      real(c_double), dimension(:, :), target :: kh_smag_ec_before
      real(c_double), dimension(:, :), target :: z_nabla2_e_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: kh_smag_e_rel_tol
      real(c_double), value, target, optional :: kh_smag_e_abs_tol
      real(c_double), value, target, optional :: kh_smag_ec_rel_tol
      real(c_double), value, target, optional :: kh_smag_ec_abs_tol
      real(c_double), value, target, optional :: z_nabla2_e_rel_tol
      real(c_double), value, target, optional :: z_nabla2_e_abs_tol

      real(c_double) :: kh_smag_e_rel_err_tol
      real(c_double) :: kh_smag_e_abs_err_tol
      real(c_double) :: kh_smag_ec_rel_err_tol
      real(c_double) :: kh_smag_ec_abs_err_tol
      real(c_double) :: z_nabla2_e_rel_err_tol
      real(c_double) :: z_nabla2_e_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(kh_smag_e_rel_tol)) then
         kh_smag_e_rel_err_tol = kh_smag_e_rel_tol
      else
         kh_smag_e_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(kh_smag_e_abs_tol)) then
         kh_smag_e_abs_err_tol = kh_smag_e_abs_tol
      else
         kh_smag_e_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(kh_smag_ec_rel_tol)) then
         kh_smag_ec_rel_err_tol = kh_smag_ec_rel_tol
      else
         kh_smag_ec_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(kh_smag_ec_abs_tol)) then
         kh_smag_ec_abs_err_tol = kh_smag_ec_abs_tol
      else
         kh_smag_ec_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(z_nabla2_e_rel_tol)) then
         z_nabla2_e_rel_err_tol = z_nabla2_e_rel_tol
      else
         z_nabla2_e_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(z_nabla2_e_abs_tol)) then
         z_nabla2_e_abs_err_tol = z_nabla2_e_abs_tol
      else
         z_nabla2_e_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC diff_multfac_smag, &
      !$ACC tangent_orientation, &
      !$ACC inv_primal_edge_length, &
      !$ACC inv_vert_vert_length, &
      !$ACC u_vert, &
      !$ACC v_vert, &
      !$ACC primal_normal_vert_x, &
      !$ACC primal_normal_vert_y, &
      !$ACC dual_normal_vert_x, &
      !$ACC dual_normal_vert_y, &
      !$ACC vn, &
      !$ACC smag_limit, &
      !$ACC kh_smag_e, &
      !$ACC kh_smag_ec, &
      !$ACC z_nabla2_e, &
      !$ACC kh_smag_e_before, &
      !$ACC kh_smag_ec_before, &
      !$ACC z_nabla2_e_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_nh_diffusion_stencil_01 &
         ( &
         diff_multfac_smag, &
         tangent_orientation, &
         inv_primal_edge_length, &
         inv_vert_vert_length, &
         u_vert, &
         v_vert, &
         primal_normal_vert_x, &
         primal_normal_vert_y, &
         dual_normal_vert_x, &
         dual_normal_vert_y, &
         vn, &
         smag_limit, &
         kh_smag_e, &
         kh_smag_ec, &
         z_nabla2_e, &
         smag_offset, &
         kh_smag_e_before, &
         kh_smag_ec_before, &
         z_nabla2_e_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         kh_smag_e_rel_err_tol, &
         kh_smag_e_abs_err_tol, &
         kh_smag_ec_rel_err_tol, &
         kh_smag_ec_abs_err_tol, &
         z_nabla2_e_rel_err_tol, &
         z_nabla2_e_abs_err_tol &
         )
#else
      call run_mo_nh_diffusion_stencil_01 &
         ( &
         diff_multfac_smag, &
         tangent_orientation, &
         inv_primal_edge_length, &
         inv_vert_vert_length, &
         u_vert, &
         v_vert, &
         primal_normal_vert_x, &
         primal_normal_vert_y, &
         dual_normal_vert_x, &
         dual_normal_vert_y, &
         vn, &
         smag_limit, &
         kh_smag_e, &
         kh_smag_ec, &
         z_nabla2_e, &
         smag_offset, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_nh_diffusion_stencil_01( &
      mesh, &
      k_size, &
      stream, &
      kh_smag_e_kmax, &
      kh_smag_ec_kmax, &
      z_nabla2_e_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: kh_smag_e_kmax
      integer(c_int), value, target, optional :: kh_smag_ec_kmax
      integer(c_int), value, target, optional :: z_nabla2_e_kmax

      integer(c_int) :: kh_smag_e_kvert_max
      integer(c_int) :: kh_smag_ec_kvert_max
      integer(c_int) :: z_nabla2_e_kvert_max

      if (present(kh_smag_e_kmax)) then
         kh_smag_e_kvert_max = kh_smag_e_kmax
      else
         kh_smag_e_kvert_max = k_size
      end if
      if (present(kh_smag_ec_kmax)) then
         kh_smag_ec_kvert_max = kh_smag_ec_kmax
      else
         kh_smag_ec_kvert_max = k_size
      end if
      if (present(z_nabla2_e_kmax)) then
         z_nabla2_e_kvert_max = z_nabla2_e_kmax
      else
         z_nabla2_e_kvert_max = k_size
      end if

      call setup_mo_nh_diffusion_stencil_01 &
         ( &
         mesh, &
         k_size, &
         stream, &
         kh_smag_e_kvert_max, &
         kh_smag_ec_kvert_max, &
         z_nabla2_e_kvert_max &
         )
   end subroutine

end module