! ICON
!
! ---------------------------------------------------------------
! Copyright (C) 2004-2025, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
! Contact information: icon-model.org
!
! See AUTHORS.TXT for a list of authors
! See LICENSES/ for license information
! SPDX-License-Identifier: BSD-3-Clause
! ---------------------------------------------------------------

! Module determines kind type parameters for different floating
! point precisions and integer ranges using either numerical
! requirements or a standard intrinsic module for interoperability
! with C.
!
! Example for numerical characteristics:
!
! @f{tabular}{{r@{\hspace*{3em}}c@{\hspace*{3em}}c}
!                     &4 byte REAL     &8 byte REAL        \\\
!        CRAY:        &-               &precision =   13   \\\
!                     &                &exponent  = 2465   \\\
!        IEEE:        &precision = 6   &precision =   15   \\\
!                     &exponent  = 37  &exponent  =  307
! @f}
! \\medskip
!
!  Most likely this are the only possible models.

MODULE mo_kind
  USE, INTRINSIC :: iso_c_binding, ONLY: c_float, c_double
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: sp, dp, qp, wp, xwp, vp, rp, i1, i2, i4, i8
  PUBLIC :: check_numerical_requirements
  !--------------------------------------------------------------------
  !
  ! Floating point section
  ! ----------------------
  !
  ! portable floating point kind type parameters: sp, dp
  INTEGER, PARAMETER :: sp = c_float                   !< single precision
  INTEGER, PARAMETER :: dp = c_double                  !< double precision

#ifdef __HAVE_QUAD_PRECISION
  INTEGER, PARAMETER :: pq = 30
  INTEGER, PARAMETER :: qp = SELECTED_REAL_KIND(pq)    !< quad precision
#else
  INTEGER, PARAMETER :: qp = -1                        !< quad precision
#endif
  !
#ifdef __SINGLE_PRECISION
  INTEGER, PARAMETER :: wp = sp                        !< selected working precision
  INTEGER, PARAMETER :: xwp = dp                       !< not working precision - {sp,dp} not wp
  INTEGER, PARAMETER :: vp = sp                        !< selected variable precision
#else
  INTEGER, PARAMETER :: wp = dp                        !< selected working precision
  INTEGER, PARAMETER :: xwp = sp                       !< not working precision - {sp,dp} not wp
#ifdef __MIXED_PRECISION
  INTEGER, PARAMETER :: vp = sp                        !< selected variable precision
#else
  INTEGER, PARAMETER :: vp = dp                        !< selected variable precision
#endif
#endif
#ifdef __SINGLE_PRECISION_ECRAD
  INTEGER, PARAMETER :: rp = sp
#else
  INTEGER, PARAMETER :: rp = wp
#endif


  !
  ! Integer section
  ! ---------------
  !
  INTEGER, PARAMETER :: pi1 =  2
  INTEGER, PARAMETER :: pi2 =  4
  INTEGER, PARAMETER :: pi4 =  9
  INTEGER, PARAMETER :: pi8 = 14  ! could be larger, but SX cannot do some operations otherwise
  !
  INTEGER, PARAMETER :: i1 = SELECTED_INT_KIND(pi1)   !< at least 1 byte integer
  INTEGER, PARAMETER :: i2 = SELECTED_INT_KIND(pi2)   !< at least 2 byte integer
  INTEGER, PARAMETER :: i4 = SELECTED_INT_KIND(pi4)   !< at least 4 byte integer
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(pi8)   !< at least 8 byte integer
  !
  !
  ! The following variable is made available internally only. configure needs to detect
  ! the addressing mode and according to this mo_kind has to be updated by an preprocessor
  ! directive and #include <config.h>. This needs some changes.
  !
  INTEGER, PARAMETER :: wi = i4                       !< selected working precission
  !
  !
  !--------------------------------------------------------------------
CONTAINS
  !
  ! check numerical requirements for real(sp) and real(dp)
  !
  ! return true if and only if all checks are passed
  !
  FUNCTION check_numerical_requirements() RESULT(is_okay)
    REAL(sp), PARAMETER :: one_sp = 1.0
    REAL(dp), PARAMETER :: one_dp = 1.0
    LOGICAL :: is_okay
    is_okay = PRECISION(one_sp) >= 6  .AND. RANGE(one_sp) >= 37 .AND. &
         &    PRECISION(one_dp) >= 12 .AND. RANGE(one_dp) >= 307
  END FUNCTION check_numerical_requirements

END MODULE mo_kind
