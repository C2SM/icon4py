module identity_plugin
    use, intrinsic :: iso_c_binding
    implicit none

    public :: identity

    interface

        function get_device_ptr_from_fortran_array(host_array) bind(C, name="get_device_ptr_from_fortran_array") result(ptr)
            use, intrinsic :: iso_c_binding
            ! Adjusted to assume a contiguous block of memory as a 1D array from Fortran's perspective
            real(c_double), dimension(*), intent(in), target :: host_array
            type(c_ptr) :: ptr
        end function get_device_ptr_from_fortran_array

        function identity_wrapper(device_ptr, &
                                  n_Cell, &
                                  n_K) bind(c, name="identity_wrapper") result(rc)
            import :: c_int, c_double, c_ptr

            type(c_ptr), value :: device_ptr
            integer(c_int), value :: n_Cell
            integer(c_int), value :: n_K
            integer(c_int) :: rc  ! Stores the return code

        end function identity_wrapper
    end interface

contains

    subroutine identity(inp, &
                        rc)
        use, intrinsic :: iso_c_binding
        type(c_ptr) :: device_ptr

        integer(c_int) :: n_Cell
        integer(c_int) :: n_K
        real(c_double), dimension(:, :), target :: inp
        integer(c_int) :: rc  ! Stores the return code

        n_Cell = SIZE(inp, 1)
        n_K = SIZE(inp, 2)

        ! Obtain device pointer for the inp array
        device_ptr = get_device_ptr_from_fortran_array(inp)

        ! Pass the device pointer to the C function
        rc = identity_wrapper(device_ptr, &
                              n_Cell, &
                              n_K)
    end subroutine identity

end module
