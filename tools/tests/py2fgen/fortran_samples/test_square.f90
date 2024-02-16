program call_square_wrapper_cffi_plugin
    use, intrinsic :: iso_c_binding
#ifdef USE_SQUARE_FROM_FUNCTION
    use square_from_function_plugin
#else
    use square_plugin
#endif
    implicit none

    integer(c_int) :: cdim, kdim, i, j
    logical :: computation_correct
    real(c_double), dimension(:, :), allocatable :: input, result

    ! array dimensions
    cdim = 18
    kdim = 10

    ! allocate arrays
    allocate(input(cdim, kdim))
    allocate(result(cdim, kdim))

    ! initialise arrays
    input = 5.0d0
    result = 0.0d0

    ! debug info
    print *, "Arrays before:"
    print *, "input = ", input
    print *, "result = ", result

    ! Call the appropriate cffi plugin
#ifdef USE_SQUARE_FROM_FUNCTION
    call square_from_function_wrapper(input, result, cdim, kdim)
#else
    call square_wrapper(input, result, cdim, kdim)
#endif

    ! debug info
    print *, "Arrays after:"
    print *, "input = ", input
    print *, "result = ", result

    ! Assert each element of result is the square of the corresponding element in input
    computation_correct = .true.
    do i = 1, cdim
        do j = 1, kdim
            if (result(i, j) /= input(i, j)**2) then
                print *, "Error: result(", i, ",", j, ") =", result(i, j), &
                        "is not the square of input(", i, ",", j, ") =", input(i, j)
                computation_correct = .false.
                exit
            endif
        enddo
        if (.not. computation_correct) exit
    enddo

    ! deallocate arrays
    deallocate(input, result)

    ! Check and print the result of the assertion
    if (computation_correct) then
        print *, "passed: result has expected values."
    else
        print *, "failed: result does not have the expected values."
    end if
end program call_square_wrapper_cffi_plugin
