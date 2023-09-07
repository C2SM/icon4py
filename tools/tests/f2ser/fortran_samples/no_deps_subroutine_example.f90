MODULE no_deps_example_subroutines
    IMPLICIT NONE

    PUBLIC :: no_deps_init, no_deps_run
    PRIVATE

    USE foo

    CONTAINS

    SUBROUTINE no_deps_init(a, b, c)
        REAL, INTENT(in) :: a
        REAL, INTENT(inout) :: b
        REAL, INTENT(out) :: c
        c = a + b
        b = 2.0 * b
    END SUBROUTINE no_deps_init

    SUBROUTINE no_deps_run(a, b, c)
        REAL, INTENT(in) :: a
        REAL, INTENT(inout) :: b
        REAL, INTENT(out) :: c
        c = a + b
        b = 2.0 * b
    END SUBROUTINE no_deps_run

END MODULE no_deps_example_subroutines
