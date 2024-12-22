module square_plugin
   use, intrinsic :: iso_c_binding
   implicit none

   public :: square

   interface

      function square_wrapper(inp, &
                              result, &
                              n_CE, &
                              n_K) bind(c, name="square_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr

         integer(c_int), value :: n_CE

         integer(c_int), value :: n_K

         integer(c_int) :: rc  ! Stores the return code

         real(c_double), dimension(*), target :: inp

         real(c_double), dimension(*), target :: result

      end function square_wrapper

   end interface

contains

   subroutine square(inp, &
                     result, &
                     rc)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_CE

      integer(c_int) :: n_K

      real(c_double), dimension(:, :), target :: inp

      real(c_double), dimension(:, :), target :: result

      integer(c_int) :: rc  ! Stores the return code

      !$ACC host_data use_device( &
      !$ACC inp, &
      !$ACC result &
      !$ACC )

      n_CE = SIZE(inp, 1)

      n_K = SIZE(inp, 2)

      rc = square_wrapper(inp, &
                          result, &
                          n_CE, &
                          n_K)

      !$acc end host_data

   end subroutine square

end module