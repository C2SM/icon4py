program call_multi_return_cffi_plugin
   use, intrinsic :: iso_c_binding
   use multi_return_plugin
   implicit none

   integer(c_int) :: edim, kdim, i, j, horizontal_start, horizontal_end, vertical_start, vertical_end, rc
   logical :: computation_correct
   character(len=100) :: str_buffer
   real(c_double) :: r_nsubsteps
   real(c_double), dimension(:, :), allocatable :: z_vn_avg, mass_fl_e, vn_traj, mass_flx_me

   ! array dimensions
   edim = 27
   kdim = 10

   !$ACC enter data create(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! allocate arrays (allocate in column-major order)
   allocate (z_vn_avg(edim, kdim))
   allocate (mass_fl_e(edim, kdim))
   allocate (vn_traj(edim, kdim))
   allocate (mass_flx_me(edim, kdim))

   ! initialize arrays and variables
   z_vn_avg = 1.0d0
   mass_fl_e = 2.0d0
   vn_traj = 3.0d0
   mass_flx_me = 4.0d0
   r_nsubsteps = 9.0d0
   horizontal_start = 0
   horizontal_end = edim
   vertical_start = 0
   vertical_end = kdim

   !$ACC data copyin(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps)

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

   ! call the cffi plugin
   call multi_return(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps, &
                     horizontal_start, horizontal_end, vertical_start, vertical_end, rc)
   print *, "Python exit code = ", rc
   if (rc /= 0) then
       call exit(1)
   end if

   !$ACC update host(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! print array shapes and values before computation
   print *, "Arrays after computation:"
   write (str_buffer, '("Shape of z_vn_avg = ", I2, ",", I2)') size(z_vn_avg, 1), size(z_vn_avg, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of mass_fl_e = ", I2, ",", I2)') size(mass_fl_e, 1), size(mass_fl_e, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of vn_traj = ", I2, ",", I2)') size(vn_traj, 1), size(vn_traj, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of mass_flx_me = ", I2, ",", I2)') size(mass_flx_me, 1), size(mass_flx_me, 2)
   print *, trim(str_buffer)
   print *

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

   !$ACC end data
   !$ACC exit data delete(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! deallocate arrays
   deallocate (z_vn_avg, mass_fl_e, vn_traj, mass_flx_me)

   ! Check and print the result of the assertion
   if (computation_correct) then
      print *, "passed: vn_traj and mass_flx_me have expected values."
   else
      print *, "failed: vn_traj or mass_flx_me does not have the expected values."
      stop 1
   end if
end program call_multi_return_cffi_plugin
