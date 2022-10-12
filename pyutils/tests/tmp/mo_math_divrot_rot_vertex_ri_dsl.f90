
#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
module mo_math_divrot_rot_vertex_ri_dsl
   use, intrinsic :: iso_c_binding
   implicit none
   interface
      subroutine &
         run_mo_math_divrot_rot_vertex_ri_dsl( &
         vec_e, &
         geofac_rot, &
         rot_vec, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vec_e
         real(c_double), dimension(*), target :: geofac_rot
         real(c_double), dimension(*), target :: rot_vec
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
      end subroutine

      subroutine &
         run_and_verify_mo_math_divrot_rot_vertex_ri_dsl( &
         vec_e, &
         geofac_rot, &
         rot_vec, &
         rot_vec_before, &
         vertical_lower, &
         vertical_upper, &
         horizontal_lower, &
         horizontal_upper, &
         rot_vec_rel_tol, &
         rot_vec_abs_tol &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         real(c_double), dimension(*), target :: vec_e
         real(c_double), dimension(*), target :: geofac_rot
         real(c_double), dimension(*), target :: rot_vec
         real(c_double), dimension(*), target :: rot_vec_before
         integer(c_int), value, target :: vertical_lower
         integer(c_int), value, target :: vertical_upper
         integer(c_int), value, target :: horizontal_lower
         integer(c_int), value, target :: horizontal_upper
         real(c_double), value, target :: rot_vec_rel_tol
         real(c_double), value, target :: rot_vec_abs_tol

      end subroutine

      subroutine &
         setup_mo_math_divrot_rot_vertex_ri_dsl( &
         mesh, &
         k_size, &
         stream, &
         rot_vec_kmax &
         ) bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
         type(c_ptr), value, target :: mesh
         integer(c_int), value, target :: k_size
         integer(kind=acc_handle_kind), value, target :: stream
         integer(c_int), value, target :: rot_vec_kmax

      end subroutine

      subroutine &
         free_mo_math_divrot_rot_vertex_ri_dsl() bind(c)
         use, intrinsic :: iso_c_binding
         use openacc
      end subroutine
   end interface
contains

   subroutine &
      wrap_run_mo_math_divrot_rot_vertex_ri_dsl( &
      vec_e, &
      geofac_rot, &
      rot_vec, &
      rot_vec_before, &
      vertical_lower, &
      vertical_upper, &
      horizontal_lower, &
      horizontal_upper, &
      rot_vec_rel_tol, &
      rot_vec_abs_tol &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      real(c_double), dimension(:, :), target :: vec_e
      real(c_double), dimension(:, :), target :: geofac_rot
      real(c_double), dimension(:, :), target :: rot_vec
      real(c_double), dimension(:, :), target :: rot_vec_before
      integer(c_int), value, target :: vertical_lower
      integer(c_int), value, target :: vertical_upper
      integer(c_int), value, target :: horizontal_lower
      integer(c_int), value, target :: horizontal_upper
      real(c_double), value, target, optional :: rot_vec_rel_tol
      real(c_double), value, target, optional :: rot_vec_abs_tol

      real(c_double) :: rot_vec_rel_err_tol
      real(c_double) :: rot_vec_abs_err_tol

      integer(c_int) :: vertical_start
      integer(c_int) :: vertical_end
      integer(c_int) :: horizontal_start
      integer(c_int) :: horizontal_end
      vertical_start = vertical_lower - 1
      vertical_end = vertical_upper
      horizontal_start = horizontal_lower - 1
      horizontal_end = horizontal_upper
      if (present(rot_vec_rel_tol)) then
         rot_vec_rel_err_tol = rot_vec_rel_tol
      else
         rot_vec_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
      end if

      if (present(rot_vec_abs_tol)) then
         rot_vec_abs_err_tol = rot_vec_abs_tol
      else
         rot_vec_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
      end if

      !$ACC host_data use_device( &
      !$ACC vec_e, &
      !$ACC geofac_rot, &
      !$ACC rot_vec, &
      !$ACC rot_vec_before &
      !$ACC )
#ifdef __DSL_VERIFY
      call run_and_verify_mo_math_divrot_rot_vertex_ri_dsl &
         ( &
         vec_e, &
         geofac_rot, &
         rot_vec, &
         rot_vec_before, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end, &
         rot_vec_rel_err_tol, &
         rot_vec_abs_err_tol &
         )
#else
      call run_mo_math_divrot_rot_vertex_ri_dsl &
         ( &
         vec_e, &
         geofac_rot, &
         rot_vec, &
         vertical_start, &
         vertical_end, &
         horizontal_start, &
         horizontal_end &
         )
#endif
      !$ACC end host_data
   end subroutine

   subroutine &
      wrap_setup_mo_math_divrot_rot_vertex_ri_dsl( &
      mesh, &
      k_size, &
      stream, &
      rot_vec_kmax &
      )
      use, intrinsic :: iso_c_binding
      use openacc
      type(c_ptr), value, target :: mesh
      integer(c_int), value, target :: k_size
      integer(kind=acc_handle_kind), value, target :: stream
      integer(c_int), value, target, optional :: rot_vec_kmax

      integer(c_int) :: rot_vec_kvert_max

      if (present(rot_vec_kmax)) then
         rot_vec_kvert_max = rot_vec_kmax
      else
         rot_vec_kvert_max = k_size
      end if

      call setup_mo_math_divrot_rot_vertex_ri_dsl &
         ( &
         mesh, &
         k_size, &
         stream, &
         rot_vec_kvert_max &
         )
   end subroutine

end module