program call_accumulate_prep_adv_fields_cffi_plugin
    use, intrinsic :: iso_c_binding
    use accumulate_prep_adv_fields_plugin
    implicit none

    integer(c_int) :: n_edge, n_k
    integer(c_int) :: horizontal_start, horizontal_end, vertical_start, vertical_end
    real(c_double) :: r_nsubsteps
    real(c_double), dimension(:,:), allocatable :: z_vn_avg, mass_fl_e, vn_traj, mass_flx_me
    real(c_double), dimension(:,:), allocatable :: vn_traj_wp, mass_flx_me_wp
    integer :: i, j
    logical :: computation_correct

    ! Example dimensions for arrays
    n_edge = 3
    n_k = 5

    ! Allocate arrays
    allocate(z_vn_avg(n_edge, n_k))
    allocate(mass_fl_e(n_edge, n_k))
    allocate(vn_traj(n_edge, n_k))
    allocate(mass_flx_me(n_edge, n_k))
    allocate(vn_traj_wp(n_edge, n_k))
    allocate(mass_flx_me_wp(n_edge, n_k))

    ! Initialize arrays and variables
    z_vn_avg = 1.0d0
    mass_fl_e = 2.0d0
    vn_traj = 3.0d0
    mass_flx_me = 4.0d0
    r_nsubsteps = 9.0d0
    horizontal_start = 0
    horizontal_end = n_edge
    vertical_start = 0
    vertical_end = n_k

    print *, "Arrays before"
    print *, "z_vn_avg = ", z_vn_avg
    print *, "mass_fl_e = ", mass_fl_e
    print *, "vn_traj = ", vn_traj
    print *, "mass_flx_me = ", mass_flx_me

    ! Call the subroutine
    call accumulate_prep_adv_fields_wrapper(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps, &
                                            horizontal_start, horizontal_end, vertical_start, vertical_end, &
                                            n_edge, n_k)

    ! Compute expected results
    vn_traj_wp = vn_traj + r_nsubsteps * z_vn_avg
    mass_flx_me_wp = mass_flx_me + r_nsubsteps * mass_fl_e

    ! print arrays for verification
    print *, "Arrays after"
    print *, "z_vn_avg = ", z_vn_avg
    print *, "mass_fl_e = ", mass_fl_e
    print *, "vn_traj = ", vn_traj
    print *, "mass_flx_me = ", mass_flx_me

    ! Deallocate arrays
    deallocate(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, vn_traj_wp, mass_flx_me_wp)
end program call_accumulate_prep_adv_fields_cffi_plugin
