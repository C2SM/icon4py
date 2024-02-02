! run_python_compute_airmass.f90
program run_python_compute_airmass
    use, intrinsic :: iso_c_binding
    use compute_airmass_wrapper_plugin
    implicit none

    ! Define dimensions and variables
    integer(c_int) :: horizontal_start, horizontal_end, vertical_start, vertical_end
    integer(c_int) :: cdim, kdim
    real(c_double), allocatable :: rho_in(:,:)
    real(c_double), allocatable :: ddqz_z_full_in(:,:)
    real(c_double), allocatable :: deepatmo_t1mc_in(:)
    real(c_double), allocatable :: airmass_out(:,:)

    ! Initialize arrays
    cdim = 6
    kdim = 3
    allocate(rho_in(cdim, kdim))
    allocate(ddqz_z_full_in(cdim, kdim))
    allocate(deepatmo_t1mc_in(kdim))
    allocate(airmass_out(cdim, kdim))

    ! Initialize bounds
    horizontal_start = 1
    horizontal_end = cdim
    vertical_start = 1
    vertical_end = kdim

    ! Initialize rho_in, ddqz_z_full_in, deepatmo_t1mc_in
    rho_in = reshape((/ 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0 /), [kdim, cdim])
    ddqz_z_full_in = rho_in
    deepatmo_t1mc_in = (/ 1.0, 2.0, 3.0 /)

    ! Call the subroutine
    call compute_airmass_wrapper(rho_in, ddqz_z_full_in, deepatmo_t1mc_in, airmass_out, &
                         horizontal_start, horizontal_end, vertical_start, vertical_end)

    ! Print the results
    print *, "fortran output: airmass_out =", airmass_out

    ! Deallocate arrays
    deallocate(rho_in)
    deallocate(ddqz_z_full_in)
    deallocate(deepatmo_t1mc_in)
    deallocate(airmass_out)

end program run_python_compute_airmass
