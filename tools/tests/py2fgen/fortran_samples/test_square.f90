program call_square_wrapper_cffi_plugin
   use, intrinsic :: iso_c_binding
   use square_plugin
   implicit none
   character(len=100) :: str_buffer
   integer(c_int) :: cdim, kdim, i, j
   logical :: computation_correct
   real(c_double), dimension(:, :), allocatable :: input, result

   ! array dimensions
   cdim = 18
   kdim = 10

   ! allocate arrays (allocate in column-major order)
   allocate (input(cdim, kdim))
   allocate (result(cdim, kdim))

   ! initialise arrays
   input = 5.0d0
   result = 0.0d0

   ! print array shapes and values before computation
   print *, "Fortran Arrays before calling Python:"
   write (str_buffer, '("Shape of input = ", I2, ",", I2)') size(input, 1), size(input, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of result = ", I2, ",", I2)') size(result, 1), size(result, 2)
   print *, trim(str_buffer)
   print *
   print *, "input = ", input
   print *
   print *, "result = ", result
   print *

   ! Call the appropriate cffi plugin
#ifdef USE_SQUARE_FROM_FUNCTION
   call square_from_function(input, result)
#else
   call square(input, result)
#endif

   ! print array shapes and values before computation
   print *, "Fortran arrays after calling Python:"
   write (str_buffer, '("Shape of input = ", I2, ",", I2)') size(input, 1), size(input, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of result = ", I2, ",", I2)') size(result, 1), size(result, 2)
   print *, trim(str_buffer)
   print *
   print *, "input = ", input
   print *
   print *, "result = ", result
   print *

   ! Assert each element of result is the square of the corresponding element in input
   computation_correct = .true.
   do i = 1, cdim
      do j = 1, kdim
         if (result(i, j) /= input(i, j)**2) then
            print *, "Error: result(", i, ",", j, ") =", result(i, j), &
               "is not the square of input(", i, ",", j, ") =", input(i, j)
            computation_correct = .false.
            exit
         end if
      end do
      if (.not. computation_correct) exit
   end do

   ! deallocate arrays
   deallocate (input, result)

   ! Check and print the result of the assertion
   if (computation_correct) then
      print *, "passed: result has expected values."
   else
      print *, "failed: result does not have the expected values."
   end if
end program call_square_wrapper_cffi_plugin
