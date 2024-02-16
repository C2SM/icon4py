program call_multi_return_cffi_plugin
    use, intrinsic :: iso_c_binding
    use multi_return_plugin
    implicit none

    integer(c_int) :: edim, kdim, i, j
    logical :: computation_correct
    real(c_double) :: r_nsubsteps
    real(c_double), dimension(:,:), allocatable :: z_vn_avg, mass_fl_e, vn_traj, mass_flx_me

    ! array dimensions
    edim = 3
    kdim = 4

    ! allocate arrays
    allocate(z_vn_avg(edim, kdim))
    allocate(mass_fl_e(edim, kdim))
    allocate(vn_traj(edim, kdim))
    allocate(mass_flx_me(edim, kdim))

    ! initialize arrays and variables
    z_vn_avg = 1.0d0
    mass_fl_e = 2.0d0
    vn_traj = 3.0d0
    mass_flx_me = 4.0d0
    r_nsubsteps = 9.0d0

    ! debug info
    print *, "Arrays before:"
    print *, "z_vn_avg = ", z_vn_avg
    print *, "mass_fl_e = ", mass_fl_e
    print *, "vn_traj = ", vn_traj
    print *, "mass_flx_me = ", mass_flx_me

    ! call the cffi plugin
    call multi_return_wrapper(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps, edim, kdim)

    ! debug info
    print *, "Arrays after:"
    print *, "z_vn_avg = ", z_vn_avg
    print *, "mass_fl_e = ", mass_fl_e
    print *, "vn_traj = ", vn_traj
    print *, "mass_flx_me = ", mass_flx_me

    ! Assert vn_traj == 12 and mass_flx_me == 22
    computation_correct = .true.
    do i = 1, edim
        do j = 1, kdim
            if (vn_traj(i, j) /= 12.0d0 .or. mass_flx_me(i, j) /= 22.0d0) then
                computation_correct = .false.
                exit
            end if
        end do
        if (.not. computation_correct) exit
    end do

    ! deallocate arrays
    deallocate(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

    ! Check and print the result of the assertion
    if (computation_correct) then
        print *, "passed: vn_traj and mass_flx_me have expected values."
    else
        print *, "failed: vn_traj or mass_flx_me does not have the expected values."
        stop 1
    end if
end program call_multi_return_cffi_plugin
