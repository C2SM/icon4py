
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_nh_diffusion_stencil_11
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_nh_diffusion_stencil_11( &
         theta_v, &
         theta_ref_mc, &
         enh_diffu_3d, &
         thresh_tdiff, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: theta_ref_mc
         real(c_double), dimension(*), target :: enh_diffu_3d
         real(c_double), value, target :: thresh_tdiff
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_nh_diffusion_stencil_11( &
         theta_v, &
         theta_ref_mc, &
         enh_diffu_3d, &
         thresh_tdiff, &
         enh_diffu_3d_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         enh_diffu_3d_rel_tol, &
         enh_diffu_3d_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: theta_v
         real(c_double), dimension(*), target :: theta_ref_mc
         real(c_double), dimension(*), target :: enh_diffu_3d
         real(c_double), value, target :: thresh_tdiff
         real(c_double), dimension(*), target :: enh_diffu_3d_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: enh_diffu_3d_rel_tol
         real(c_double), value, target :: enh_diffu_3d_abs_tol

      end subroutine

      subroutine &
         setup_mo_nh_diffusion_stencil_11( &
         mesh, &
         k_size, &
         stream, &
         enh_diffu_3d_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: enh_diffu_3d_kmax

      end subroutine

      subroutine &
         free_mo_nh_diffusion_stencil_11() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_nh_diffusion_stencil_11( &
      theta_v, &
      theta_ref_mc, &
      enh_diffu_3d, &
      thresh_tdiff, &
      enh_diffu_3d_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      enh_diffu_3d_rel_tol, &
      enh_diffu_3d_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: theta_v
      real(c_double), dimension(:, :), target :: theta_ref_mc
      real(c_double), dimension(:, :), target :: enh_diffu_3d
      real(c_double), value, target :: thresh_tdiff
      real(c_double), dimension(:, :), target :: enh_diffu_3d_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: enh_diffu_3d_rel_tol
      real(c_double), value, target, optional :: enh_diffu_3d_abs_tol

      real(c_double) :: enh_diffu_3d_rel_err_tol
      real(c_double) :: enh_diffu_3d_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(enh_diffu_3d_rel_tol)) then
         enh_diffu_3d_rel_err_tol = enh_diffu_3d_rel_tol
      else
         enh_diffu_3d_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(enh_diffu_3d_abs_tol)) then
         enh_diffu_3d_abs_err_tol = enh_diffu_3d_abs_tol
      else
         enh_diffu_3d_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC theta_v, &
      !$ACC theta_ref_mc, &
      !$ACC enh_diffu_3d, &
      !$ACC enh_diffu_3d_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_nh_diffusion_stencil_11 &
         ( &
         theta_v, &
         theta_ref_mc, &
         enh_diffu_3d, &
         thresh_tdiff, &
         enh_diffu_3d_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         enh_diffu_3d_rel_err_tol, &
         enh_diffu_3d_abs_err_tol &
         )
#else
      call run_mo_nh_diffusion_stencil_11 &
         ( &
         theta_v, &
         theta_ref_mc, &
         enh_diffu_3d, &
         thresh_tdiff, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_nh_diffusion_stencil_11( &
      mesh, &
      k_size, &
      stream, &
      enh_diffu_3d_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: enh_diffu_3d_kmax

      integer(c_int) :: enh_diffu_3d_kvert_max

      if (present(enh_diffu_3d_kmax)) then
         enh_diffu_3d_kvert_max = enh_diffu_3d_kmax
      else
         enh_diffu_3d_kvert_max = k_size
      end if

      call setup_mo_nh_diffusion_stencil_11 &
         ( &
         mesh, &
         k_size, &
         stream, &
         enh_diffu_3d_kvert_max &
         )
   end subroutine

end module