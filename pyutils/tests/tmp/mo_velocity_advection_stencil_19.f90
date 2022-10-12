
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_velocity_advection_stencil_19
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_velocity_advection_stencil_19( &
         z_kin_hor_e, &
         coeff_gradekin, &
         z_ekinh, &
         zeta, &
         vt, &
         f_e, &
         c_lin_e, &
         z_w_con_c_full, &
         vn_ie, &
         ddqz_z_full_e, &
         ddt_vn_adv, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: z_kin_hor_e
         real(c_double), dimension(*), target :: coeff_gradekin
         real(c_double), dimension(*), target :: z_ekinh
         real(c_double), dimension(*), target :: zeta
         real(c_double), dimension(*), target :: vt
         real(c_double), dimension(*), target :: f_e
         real(c_double), dimension(*), target :: c_lin_e
         real(c_double), dimension(*), target :: z_w_con_c_full
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: ddqz_z_full_e
         real(c_double), dimension(*), target :: ddt_vn_adv
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_velocity_advection_stencil_19( &
         z_kin_hor_e, &
         coeff_gradekin, &
         z_ekinh, &
         zeta, &
         vt, &
         f_e, &
         c_lin_e, &
         z_w_con_c_full, &
         vn_ie, &
         ddqz_z_full_e, &
         ddt_vn_adv, &
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
         real(c_double), dimension(*), target :: z_kin_hor_e
         real(c_double), dimension(*), target :: coeff_gradekin
         real(c_double), dimension(*), target :: z_ekinh
         real(c_double), dimension(*), target :: zeta
         real(c_double), dimension(*), target :: vt
         real(c_double), dimension(*), target :: f_e
         real(c_double), dimension(*), target :: c_lin_e
         real(c_double), dimension(*), target :: z_w_con_c_full
         real(c_double), dimension(*), target :: vn_ie
         real(c_double), dimension(*), target :: ddqz_z_full_e
         real(c_double), dimension(*), target :: ddt_vn_adv
         real(c_double), dimension(*), target :: ddt_vn_adv_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: ddt_vn_adv_rel_tol
         real(c_double), value, target :: ddt_vn_adv_abs_tol

      end subroutine

      subroutine &
         setup_mo_velocity_advection_stencil_19( &
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
         free_mo_velocity_advection_stencil_19() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_velocity_advection_stencil_19( &
      z_kin_hor_e, &
      coeff_gradekin, &
      z_ekinh, &
      zeta, &
      vt, &
      f_e, &
      c_lin_e, &
      z_w_con_c_full, &
      vn_ie, &
      ddqz_z_full_e, &
      ddt_vn_adv, &
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
      real(c_double), dimension(:, :), target :: z_kin_hor_e
      real(c_double), dimension(:, :), target :: coeff_gradekin
      real(c_double), dimension(:, :), target :: z_ekinh
      real(c_double), dimension(:, :), target :: zeta
      real(c_double), dimension(:, :), target :: vt
      real(c_double), dimension(:), target :: f_e
      real(c_double), dimension(:, :), target :: c_lin_e
      real(c_double), dimension(:, :), target :: z_w_con_c_full
      real(c_double), dimension(:, :), target :: vn_ie
      real(c_double), dimension(:, :), target :: ddqz_z_full_e
      real(c_double), dimension(:, :), target :: ddt_vn_adv
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
      !$ACC z_kin_hor_e, &
      !$ACC coeff_gradekin, &
      !$ACC z_ekinh, &
      !$ACC zeta, &
      !$ACC vt, &
      !$ACC f_e, &
      !$ACC c_lin_e, &
      !$ACC z_w_con_c_full, &
      !$ACC vn_ie, &
      !$ACC ddqz_z_full_e, &
      !$ACC ddt_vn_adv, &
      !$ACC ddt_vn_adv_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_velocity_advection_stencil_19 &
         ( &
         z_kin_hor_e, &
         coeff_gradekin, &
         z_ekinh, &
         zeta, &
         vt, &
         f_e, &
         c_lin_e, &
         z_w_con_c_full, &
         vn_ie, &
         ddqz_z_full_e, &
         ddt_vn_adv, &
         ddt_vn_adv_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         ddt_vn_adv_rel_err_tol, &
         ddt_vn_adv_abs_err_tol &
         )
#else
      call run_mo_velocity_advection_stencil_19 &
         ( &
         z_kin_hor_e, &
         coeff_gradekin, &
         z_ekinh, &
         zeta, &
         vt, &
         f_e, &
         c_lin_e, &
         z_w_con_c_full, &
         vn_ie, &
         ddqz_z_full_e, &
         ddt_vn_adv, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_velocity_advection_stencil_19( &
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

      call setup_mo_velocity_advection_stencil_19 &
         ( &
         mesh, &
         k_size, &
         stream, &
         ddt_vn_adv_kvert_max &
         )
   end subroutine

end module