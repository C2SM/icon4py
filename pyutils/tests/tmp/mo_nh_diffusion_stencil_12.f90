
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_nh_diffusion_stencil_12
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_nh_diffusion_stencil_12( &
         kh_smag_e, &
         enh_diffu_3d, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: kh_smag_e
         real(c_double), dimension(*), target :: enh_diffu_3d
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_nh_diffusion_stencil_12( &
         kh_smag_e, &
         enh_diffu_3d, &
         kh_smag_e_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         kh_smag_e_rel_tol, &
         kh_smag_e_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: kh_smag_e
         real(c_double), dimension(*), target :: enh_diffu_3d
         real(c_double), dimension(*), target :: kh_smag_e_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: kh_smag_e_rel_tol
         real(c_double), value, target :: kh_smag_e_abs_tol

      end subroutine

      subroutine &
         setup_mo_nh_diffusion_stencil_12( &
         mesh, &
         k_size, &
         stream, &
         kh_smag_e_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: kh_smag_e_kmax

      end subroutine

      subroutine &
         free_mo_nh_diffusion_stencil_12() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_nh_diffusion_stencil_12( &
      kh_smag_e, &
      enh_diffu_3d, &
      kh_smag_e_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      kh_smag_e_rel_tol, &
      kh_smag_e_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: kh_smag_e
      real(c_double), dimension(:, :), target :: enh_diffu_3d
      real(c_double), dimension(:, :), target :: kh_smag_e_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: kh_smag_e_rel_tol
      real(c_double), value, target, optional :: kh_smag_e_abs_tol

      real(c_double) :: kh_smag_e_rel_err_tol
      real(c_double) :: kh_smag_e_abs_err_tol

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

      !$ACC host_data use_device( &
      !$ACC kh_smag_e, &
      !$ACC enh_diffu_3d, &
      !$ACC kh_smag_e_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_nh_diffusion_stencil_12 &
         ( &
         kh_smag_e, &
         enh_diffu_3d, &
         kh_smag_e_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         kh_smag_e_rel_err_tol, &
         kh_smag_e_abs_err_tol &
         )
#else
      call run_mo_nh_diffusion_stencil_12 &
         ( &
         kh_smag_e, &
         enh_diffu_3d, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_nh_diffusion_stencil_12( &
      mesh, &
      k_size, &
      stream, &
      kh_smag_e_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: kh_smag_e_kmax

      integer(c_int) :: kh_smag_e_kvert_max

      if (present(kh_smag_e_kmax)) then
         kh_smag_e_kvert_max = kh_smag_e_kmax
      else
         kh_smag_e_kvert_max = k_size
      end if

      call setup_mo_nh_diffusion_stencil_12 &
         ( &
         mesh, &
         k_size, &
         stream, &
         kh_smag_e_kvert_max &
         )
   end subroutine

end module