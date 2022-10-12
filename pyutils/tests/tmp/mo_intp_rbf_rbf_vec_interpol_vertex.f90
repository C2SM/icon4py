
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_intp_rbf_rbf_vec_interpol_vertex
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_intp_rbf_rbf_vec_interpol_vertex( &
         p_e_in, &
         ptr_coeff_1, &
         ptr_coeff_2, &
         p_u_out, &
         p_v_out, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: p_e_in
         real(c_double), dimension(*), target :: ptr_coeff_1
         real(c_double), dimension(*), target :: ptr_coeff_2
         real(c_double), dimension(*), target :: p_u_out
         real(c_double), dimension(*), target :: p_v_out
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_intp_rbf_rbf_vec_interpol_vertex( &
         p_e_in, &
         ptr_coeff_1, &
         ptr_coeff_2, &
         p_u_out, &
         p_v_out, &
         p_u_out_before, &
         p_v_out_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         p_u_out_rel_tol, &
         p_u_out_abs_tol, &
         p_v_out_rel_tol, &
         p_v_out_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: p_e_in
         real(c_double), dimension(*), target :: ptr_coeff_1
         real(c_double), dimension(*), target :: ptr_coeff_2
         real(c_double), dimension(*), target :: p_u_out
         real(c_double), dimension(*), target :: p_v_out
         real(c_double), dimension(*), target :: p_u_out_before
         real(c_double), dimension(*), target :: p_v_out_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: p_u_out_rel_tol
         real(c_double), value, target :: p_u_out_abs_tol
         real(c_double), value, target :: p_v_out_rel_tol
         real(c_double), value, target :: p_v_out_abs_tol

      end subroutine

      subroutine &
         setup_mo_intp_rbf_rbf_vec_interpol_vertex( &
         mesh, &
         k_size, &
         stream, &
         p_u_out_kmax, &
         p_v_out_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: p_u_out_kmax
         integer(c_int), value, target :: p_v_out_kmax

      end subroutine

      subroutine &
         free_mo_intp_rbf_rbf_vec_interpol_vertex() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_intp_rbf_rbf_vec_interpol_vertex( &
      p_e_in, &
      ptr_coeff_1, &
      ptr_coeff_2, &
      p_u_out, &
      p_v_out, &
      p_u_out_before, &
      p_v_out_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      p_u_out_rel_tol, &
      p_u_out_abs_tol, &
      p_v_out_rel_tol, &
      p_v_out_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: p_e_in
      real(c_double), dimension(:, :), target :: ptr_coeff_1
      real(c_double), dimension(:, :), target :: ptr_coeff_2
      real(c_double), dimension(:, :), target :: p_u_out
      real(c_double), dimension(:, :), target :: p_v_out
      real(c_double), dimension(:, :), target :: p_u_out_before
      real(c_double), dimension(:, :), target :: p_v_out_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: p_u_out_rel_tol
      real(c_double), value, target, optional :: p_u_out_abs_tol
      real(c_double), value, target, optional :: p_v_out_rel_tol
      real(c_double), value, target, optional :: p_v_out_abs_tol

      real(c_double) :: p_u_out_rel_err_tol
      real(c_double) :: p_u_out_abs_err_tol
      real(c_double) :: p_v_out_rel_err_tol
      real(c_double) :: p_v_out_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(p_u_out_rel_tol)) then
         p_u_out_rel_err_tol = p_u_out_rel_tol
      else
         p_u_out_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_u_out_abs_tol)) then
         p_u_out_abs_err_tol = p_u_out_abs_tol
      else
         p_u_out_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if
      if (present(p_v_out_rel_tol)) then
         p_v_out_rel_err_tol = p_v_out_rel_tol
      else
         p_v_out_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(p_v_out_abs_tol)) then
         p_v_out_abs_err_tol = p_v_out_abs_tol
      else
         p_v_out_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC p_e_in, &
      !$ACC ptr_coeff_1, &
      !$ACC ptr_coeff_2, &
      !$ACC p_u_out, &
      !$ACC p_v_out, &
      !$ACC p_u_out_before, &
      !$ACC p_v_out_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_intp_rbf_rbf_vec_interpol_vertex &
         ( &
         p_e_in, &
         ptr_coeff_1, &
         ptr_coeff_2, &
         p_u_out, &
         p_v_out, &
         p_u_out_before, &
         p_v_out_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         p_u_out_rel_err_tol, &
         p_u_out_abs_err_tol, &
         p_v_out_rel_err_tol, &
         p_v_out_abs_err_tol &
         )
#else
      call run_mo_intp_rbf_rbf_vec_interpol_vertex &
         ( &
         p_e_in, &
         ptr_coeff_1, &
         ptr_coeff_2, &
         p_u_out, &
         p_v_out, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_intp_rbf_rbf_vec_interpol_vertex( &
      mesh, &
      k_size, &
      stream, &
      p_u_out_kmax, &
      p_v_out_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: p_u_out_kmax
      integer(c_int), value, target, optional :: p_v_out_kmax

      integer(c_int) :: p_u_out_kvert_max
      integer(c_int) :: p_v_out_kvert_max

      if (present(p_u_out_kmax)) then
         p_u_out_kvert_max = p_u_out_kmax
      else
         p_u_out_kvert_max = k_size
      end if
      if (present(p_v_out_kmax)) then
         p_v_out_kvert_max = p_v_out_kmax
      else
         p_v_out_kvert_max = k_size
      end if

      call setup_mo_intp_rbf_rbf_vec_interpol_vertex &
         ( &
         mesh, &
         k_size, &
         stream, &
         p_u_out_kvert_max, &
         p_v_out_kvert_max &
         )
   end subroutine

end module