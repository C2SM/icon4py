program call_identity_cffi_plugin
   use openacc
   use, intrinsic :: iso_c_binding
   use identity_plugin
   implicit none
   integer(c_int) :: cdim, kdim, i, rc
   real(c_double), dimension(:, :), allocatable :: input

   ! array dimensions
   cdim = 5
   kdim = 5

   ! allocate arrays (allocate in column-major order)
   allocate(input(cdim, kdim))

   ! initialise arrays
   input = 5.0d0

   ! print array shapes and values before computation
   print *, "Fortran Arrays before calling Python:"
   print *, "input = ", input
   print *

   !$acc enter data create(input)
   ! call cffi code
   call identity(input, rc)

   if (rc /= 0) then
       print *, "Python failed with exit code = ", rc
       call exit(1)
   end if

   !$acc update host(input)

   print *, "input after calling Python = ", input
   print *

   !$acc exit data delete(input)

   deallocate (input)

end program call_identity_cffi_plugin
