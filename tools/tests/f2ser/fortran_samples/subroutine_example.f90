! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

MODULE example_subroutines
    USE ISO_C_BINDING, ONLY: C_DOUBLE
    IMPLICIT NONE

    PUBLIC :: mysubroutine, foo_type
    PRIVATE

    TYPE, BIND(C) :: foo_type
        REAL(C_DOUBLE) :: p1
        REAL(C_DOUBLE) :: p2
    END TYPE foo_type

    CONTAINS

    SUBROUTINE mysubroutine(a, b, c)
        REAL, INTENT(in) :: a
        REAL, INTENT(inout) :: b
        REAL, INTENT(out) :: c
        c = a + b
        b = 2.0 * b
    end SUBROUTINE mysubroutine

END MODULE example_subroutines





