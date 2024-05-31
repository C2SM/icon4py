program call_multi_return_cffi_plugin
   use, intrinsic :: iso_c_binding
   use multi_return_from_function_plugin
   implicit none

   integer(c_int) :: cdim, edim, kdim, cedim, i, j, horizontal_start, horizontal_end, vertical_start, vertical_end, rc, n
   logical :: computation_correct
   character(len=100) :: str_buffer
   real(c_double) :: r_nsubsteps
   real(c_double), dimension(:, :), allocatable :: z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, z_nabla2_e
   real(c_double), dimension(:), allocatable :: geofac_div

   ! array dimensions
   cdim = 18
   edim = 27
   kdim = 10
   cedim = cdim * edim

   !$ACC enter data create(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! allocate arrays (allocate in column-major order)
   allocate(geofac_div(cedim))
   allocate(z_nabla2_e(edim, kdim))
   allocate (z_vn_avg(edim, kdim))
   allocate (mass_fl_e(edim, kdim))
   allocate (vn_traj(edim, kdim))
   allocate (mass_flx_me(edim, kdim))

   ! initialize arrays and variables
   geofac_div = 3.5d0
   z_vn_avg = 1.0d0
   mass_fl_e = 2.0d0
   vn_traj = 3.0d0
   mass_flx_me = 4.0d0
   r_nsubsteps = 9.0d0
   horizontal_start = 0
   horizontal_end = edim
   vertical_start = 0
   vertical_end = kdim

   !$ACC data copyin(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! print array shapes and values before computation
   print *, "Arrays before computation:"
   write (str_buffer, '("Shape of z_vn_avg = ", I2, ",", I2)') size(z_vn_avg, 1), size(z_vn_avg, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of mass_fl_e = ", I2, ",", I2)') size(mass_fl_e, 1), size(mass_fl_e, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of vn_traj = ", I2, ",", I2)') size(vn_traj, 1), size(vn_traj, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of mass_flx_me = ", I2, ",", I2)') size(mass_flx_me, 1), size(mass_flx_me, 2)
   print *, trim(str_buffer)
   print *


   ! call once just so that we compile the code once (profiling becomes easier later)
   call multi_return_from_function(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps, &
                     horizontal_start, horizontal_end, vertical_start, vertical_end, rc)

   ! call the cffi plugin
   call profile_enable(rc)
   do n = 1, 1000
       call multi_return_from_function(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps, &
                     horizontal_start, horizontal_end, vertical_start, vertical_end, rc)

       ! print array shapes and values
       print *, "Arrays after computation:"
       print *, "First value of vn_traj:", vn_traj(1, 1)
       print *, "First value of mass_flx_me:", mass_flx_me(1, 1)
       write (str_buffer, '("Shape of z_vn_avg = ", I2, ",", I2)') size(z_vn_avg, 1), size(z_vn_avg, 2)
       print *, trim(str_buffer)
       write (str_buffer, '("Shape of mass_fl_e = ", I2, ",", I2)') size(mass_fl_e, 1), size(mass_fl_e, 2)
       print *, trim(str_buffer)
       write (str_buffer, '("Shape of vn_traj = ", I2, ",", I2)') size(vn_traj, 1), size(vn_traj, 2)
       print *, trim(str_buffer)
       write (str_buffer, '("Shape of mass_flx_me = ", I2, ",", I2)') size(mass_flx_me, 1), size(mass_flx_me, 2)
       print *, trim(str_buffer)
       print *, "passed"

   end do
   call profile_disable(rc)
   print *, "Python exit code = ", rc
   if (rc /= 0) then
       call exit(1)
   end if

   !$ACC update host(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)


   !$ACC end data
   !$ACC exit data delete(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! deallocate arrays
   deallocate (z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

end program call_multi_return_cffi_plugin
