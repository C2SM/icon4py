
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_nh_diffusion_stencil_02
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_nh_diffusion_stencil_02( &
         kh_smag_ec, &
         vn, &
         e_bln_c_s, &
         geofac_div, &
         diff_multfac_smag, &
         kh_c, &
         div, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: kh_smag_ec
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: e_bln_c_s
         real(c_double), dimension(*), target :: geofac_div
         real(c_double), dimension(*), target :: diff_multfac_smag
         real(c_double), dimension(*), target :: kh_c
         real(c_double), dimension(*), target :: div
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_nh_diffusion_stencil_02( &
         kh_smag_ec, &
         vn, &
         e_bln_c_s, &
         geofac_div, &
         diff_multfac_smag, &
         kh_c, &
         div, &
         kh_c_before, &
         div_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         kh_c_rel_tol, &
         kh_c_abs_tol, &
         div_rel_tol, &
         div_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: kh_smag_ec
         real(c_double), dimension(*), target :: vn
         real(c_double), dimension(*), target :: e_bln_c_s
         real(c_double), dimension(*), target :: geofac_div
         real(c_double), dimension(*), target :: diff_multfac_smag
         real(c_double), dimension(*), target :: kh_c
         real(c_double), dimension(*), target :: div
         real(c_double), dimension(*), target :: kh_c_before
         real(c_double), dimension(*), target :: div_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: kh_c_rel_tol
         real(c_double), value, target :: kh_c_abs_tol
         real(c_double), value, target :: div_rel_tol
         real(c_double), value, target :: div_abs_tol

      end subroutine

      subroutine &
         setup_mo_nh_diffusion_stencil_02( &
         mesh, &
         k_size, &
         stream, &
         kh_c_kmax, &
         div_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: kh_c_kmax
         integer(c_int), value, target :: div_kmax

      end subroutine

      subroutine &
         free_mo_nh_diffusion_stencil_02() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_nh_diffusion_stencil_02( &
      kh_smag_ec, &
      vn, &
      e_bln_c_s, &
      geofac_div, &
      diff_multfac_smag, &
      kh_c, &
      div, &
      kh_c_before, &
      div_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      kh_c_rel_tol, &
      kh_c_abs_tol, &
      div_rel_tol, &
      div_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: kh_smag_ec
      real(c_double), dimension(:, :), target :: vn
      real(c_double), dimension(:, :), target :: e_bln_c_s
      real(c_double), dimension(:, :), target :: geofac_div
      real(c_double), dimension(:), target :: diff_multfac_smag
      real(c_double), dimension(:, :), target :: kh_c
      real(c_double), dimension(:, :), target :: div
      real(c_double), dimension(:, :), target :: kh_c_before
      real(c_double), dimension(:, :), target :: div_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: kh_c_rel_tol
      real(c_double), value, target, optional :: kh_c_abs_tol
      real(c_double), value, target, optional :: div_rel_tol
      real(c_double), value, target, optional :: div_abs_tol

      real(c_double) :: kh_c_rel_err_tol
      real(c_double) :: kh_c_abs_err_tol
      real(c_double) :: div_rel_err_tol
      real(c_double) :: div_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(kh_c_rel_tol)) then
         kh_c_rel_err_tol = kh_c_rel_tol
      else
         kh_c_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(kh_c_abs_tol)) then
         kh_c_abs_err_tol = kh_c_abs_tol
      else
         kh_c_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(div_rel_tol)) then
         div_rel_err_tol = div_rel_tol
      else
         div_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(div_abs_tol)) then
         div_abs_err_tol = div_abs_tol
      else
         div_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC kh_smag_ec, &
      !$ACC vn, &
      !$ACC e_bln_c_s, &
      !$ACC geofac_div, &
      !$ACC diff_multfac_smag, &
      !$ACC kh_c, &
      !$ACC div, &
      !$ACC kh_c_before, &
      !$ACC div_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_nh_diffusion_stencil_02 &
         ( &
         kh_smag_ec, &
         vn, &
         e_bln_c_s, &
         geofac_div, &
         diff_multfac_smag, &
         kh_c, &
         div, &
         kh_c_before, &
         div_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         kh_c_rel_err_tol, &
         kh_c_abs_err_tol, &
         div_rel_err_tol, &
         div_abs_err_tol &
         )
#else
      call run_mo_nh_diffusion_stencil_02 &
         ( &
         kh_smag_ec, &
         vn, &
         e_bln_c_s, &
         geofac_div, &
         diff_multfac_smag, &
         kh_c, &
         div, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_nh_diffusion_stencil_02( &
      mesh, &
      k_size, &
      stream, &
      kh_c_kmax, &
      div_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: kh_c_kmax
      integer(c_int), value, target, optional :: div_kmax

      integer(c_int) :: kh_c_kvert_max
      integer(c_int) :: div_kvert_max

      if (present(kh_c_kmax)) then
         kh_c_kvert_max = kh_c_kmax
      else
         kh_c_kvert_max = k_size
      end if
      if (present(div_kmax)) then
         div_kvert_max = div_kmax
      else
         div_kvert_max = k_size
      end if

      call setup_mo_nh_diffusion_stencil_02 &
         ( &
         mesh, &
         k_size, &
         stream, &
         kh_c_kvert_max, &
         div_kvert_max &
         )
   end subroutine

end module