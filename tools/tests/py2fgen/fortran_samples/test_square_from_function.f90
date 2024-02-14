program call_square_wrapper_cffi_plugin
    use, intrinsic :: iso_c_binding
    use square_plugin
    implicit none

    integer(c_int) :: cdim, kdim, i, j
    real(c_double), dimension(6, 3) :: a, res
    logical :: all_squared_correctly

    ! Initialize input array
    a = reshape((/ 1.0d0, 1.0d0, 2.0d0, 3.0d0, 5.0d0, 8.0d0, &
                  1.0d0, 1.0d0, 2.0d0, 3.0d0, 5.0d0, 8.0d0, &
                  1.0d0, 1.0d0, 2.0d0, 3.0d0, 5.0d0, 8.0d0 /), shape(a))
    print *, "fortran input: field = ", a

    ! Zero-initialize result array
    res = 0.0d0
    cdim = 6
    kdim = 3

    ! Call the square_wrapper function
    call square_from_function_wrapper(a, res, cdim, kdim)
    print *, "fortran output: res =", res

    ! Assert each element of res is the square of the corresponding element in a
    all_squared_correctly = .true.
    do i = 1, cdim
        do j = 1, kdim
            if (res(i, j) /= a(i, j)**2) then
                print *, "Error: res(", i, ",", j, ") =", res(i, j), "is not the square of a(", i, ",", j, ") =", a(i, j)
                all_squared_correctly = .false.
                exit
            endif
        enddo
        if (.not. all_squared_correctly) exit
    enddo

    if (all_squared_correctly) then
        print *, "All elements squared correctly."
    else
        print *, "Some elements were not squared correctly."
        stop 1
    endif

end program call_square_wrapper_cffi_plugin
