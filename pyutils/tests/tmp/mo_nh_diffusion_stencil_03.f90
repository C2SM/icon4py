
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_nh_diffusion_stencil_03
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_nh_diffusion_stencil_03( &
         div, &
         kh_c, &
         wgtfac_c, &
         div_ic, &
         hdef_ic, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: div
         real(c_double), dimension(*), target :: kh_c
         real(c_double), dimension(*), target :: wgtfac_c
         real(c_double), dimension(*), target :: div_ic
         real(c_double), dimension(*), target :: hdef_ic
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_nh_diffusion_stencil_03( &
         div, &
         kh_c, &
         wgtfac_c, &
         div_ic, &
         hdef_ic, &
         div_ic_before, &
         hdef_ic_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         div_ic_rel_tol, &
         div_ic_abs_tol, &
         hdef_ic_rel_tol, &
         hdef_ic_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: div
         real(c_double), dimension(*), target :: kh_c
         real(c_double), dimension(*), target :: wgtfac_c
         real(c_double), dimension(*), target :: div_ic
         real(c_double), dimension(*), target :: hdef_ic
         real(c_double), dimension(*), target :: div_ic_before
         real(c_double), dimension(*), target :: hdef_ic_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: div_ic_rel_tol
         real(c_double), value, target :: div_ic_abs_tol
         real(c_double), value, target :: hdef_ic_rel_tol
         real(c_double), value, target :: hdef_ic_abs_tol

      end subroutine

      subroutine &
         setup_mo_nh_diffusion_stencil_03( &
         mesh, &
         k_size, &
         stream, &
         div_ic_kmax, &
         hdef_ic_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: div_ic_kmax
         integer(c_int), value, target :: hdef_ic_kmax

      end subroutine

      subroutine &
         free_mo_nh_diffusion_stencil_03() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_nh_diffusion_stencil_03( &
      div, &
      kh_c, &
      wgtfac_c, &
      div_ic, &
      hdef_ic, &
      div_ic_before, &
      hdef_ic_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      div_ic_rel_tol, &
      div_ic_abs_tol, &
      hdef_ic_rel_tol, &
      hdef_ic_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: div
      real(c_double), dimension(:, :), target :: kh_c
      real(c_double), dimension(:, :), target :: wgtfac_c
      real(c_double), dimension(:, :), target :: div_ic
      real(c_double), dimension(:, :), target :: hdef_ic
      real(c_double), dimension(:, :), target :: div_ic_before
      real(c_double), dimension(:, :), target :: hdef_ic_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: div_ic_rel_tol
      real(c_double), value, target, optional :: div_ic_abs_tol
      real(c_double), value, target, optional :: hdef_ic_rel_tol
      real(c_double), value, target, optional :: hdef_ic_abs_tol

      real(c_double) :: div_ic_rel_err_tol
      real(c_double) :: div_ic_abs_err_tol
      real(c_double) :: hdef_ic_rel_err_tol
      real(c_double) :: hdef_ic_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(div_ic_rel_tol)) then
         div_ic_rel_err_tol = div_ic_rel_tol
      else
         div_ic_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(div_ic_abs_tol)) then
         div_ic_abs_err_tol = div_ic_abs_tol
      else
         div_ic_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(hdef_ic_rel_tol)) then
         hdef_ic_rel_err_tol = hdef_ic_rel_tol
      else
         hdef_ic_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(hdef_ic_abs_tol)) then
         hdef_ic_abs_err_tol = hdef_ic_abs_tol
      else
         hdef_ic_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC div, &
      !$ACC kh_c, &
      !$ACC wgtfac_c, &
      !$ACC div_ic, &
      !$ACC hdef_ic, &
      !$ACC div_ic_before, &
      !$ACC hdef_ic_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_nh_diffusion_stencil_03 &
         ( &
         div, &
         kh_c, &
         wgtfac_c, &
         div_ic, &
         hdef_ic, &
         div_ic_before, &
         hdef_ic_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         div_ic_rel_err_tol, &
         div_ic_abs_err_tol, &
         hdef_ic_rel_err_tol, &
         hdef_ic_abs_err_tol &
         )
#else
      call run_mo_nh_diffusion_stencil_03 &
         ( &
         div, &
         kh_c, &
         wgtfac_c, &
         div_ic, &
         hdef_ic, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_nh_diffusion_stencil_03( &
      mesh, &
      k_size, &
      stream, &
      div_ic_kmax, &
      hdef_ic_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: div_ic_kmax
      integer(c_int), value, target, optional :: hdef_ic_kmax

      integer(c_int) :: div_ic_kvert_max
      integer(c_int) :: hdef_ic_kvert_max

      if (present(div_ic_kmax)) then
         div_ic_kvert_max = div_ic_kmax
      else
         div_ic_kvert_max = k_size
      end if
      if (present(hdef_ic_kmax)) then
         hdef_ic_kvert_max = hdef_ic_kmax
      else
         hdef_ic_kvert_max = k_size
      end if

      call setup_mo_nh_diffusion_stencil_03 &
         ( &
         mesh, &
         k_size, &
         stream, &
         div_ic_kvert_max, &
         hdef_ic_kvert_max &
         )
   end subroutine

end module