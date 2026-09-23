! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

! Passes double-precision arrays into Python functions that compute in single precision.
program test_boundary_casts
   use, intrinsic :: iso_c_binding
   use cast_plugin
   implicit none
   integer, parameter :: n = 7
   integer :: i
   integer(c_int) :: rc
   logical :: ok
   real(c_double), dimension(:), allocatable :: inout_field, in_field, original, expected

   allocate (inout_field(n), in_field(n), original(n), expected(n))
   do i = 1, n
      ! not exactly representable in single, so a missing conversion shows up as a mismatch
      original(i) = 0.1d0 + 0.2d0*real(i - 1, c_double)
   end do
   ok = .true.

   ! 1. converted in, computed in single, converted back out
   inout_field = original
   call scale_inout(inout_field, 3.0d0, rc)
   call check_rc(rc)
   expected = real(real(original, c_float)*3.0_c_float, c_double)
   call check("INOUT converted back", inout_field, expected)

   ! 2. same array, new values: the copy must be refreshed, not served from a cache
   inout_field = 2.0d0*original
   call scale_inout(inout_field, 3.0d0, rc)
   call check_rc(rc)
   expected = real(real(2.0d0*original, c_float)*3.0_c_float, c_double)
   call check("INOUT refreshed on second call", inout_field, expected)

   ! 3. an IN argument comes back untouched, not truncated to single
   in_field = original
   call scale_in(in_field, 3.0d0, rc)
   call check_rc(rc)
   call check("IN not written back", in_field, original)

   deallocate (inout_field, in_field, original, expected)

   if (ok) then
      print *, "passed: all boundary conversions as expected."
   else
      print *, "failed: see mismatches above."
   end if

contains

   subroutine check_rc(rc)
      integer(c_int), intent(in) :: rc
      if (rc /= 1) then
         print *, "Python failed with exit code = ", rc
         call exit(1)
      end if
   end subroutine check_rc

   subroutine check(label, actual, want)
      character(len=*), intent(in) :: label
      real(c_double), intent(in) :: actual(:), want(:)
      if (any(actual /= want)) then
         print *, "mismatch in ", label
         print *, "  got      ", actual
         print *, "  expected ", want
         ok = .false.
      end if
   end subroutine check

end program test_boundary_casts
