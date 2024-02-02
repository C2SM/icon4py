
    module square_wrapper_plugin
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
        subroutine square_wrapper(field_ptr, nx, ny, result_ptr) bind(c, name='square_wrapper')
        import :: c_double, c_int
              integer(c_int), value,  intent(in):: nx
              integer(c_int), value,  intent(in):: ny
              real(c_double), intent(in):: field_ptr(nx, ny)
              real(c_double), intent(in):: result_ptr(nx, ny)   ! TODO: need to add size params for arrays in subroutine.
        end subroutine square_wrapper
    end interface
    end module


!    module square_wrapper_plugin
!    use, intrinsic :: iso_c_binding
!    implicit none
!
!    public
!    interface
!        subroutine square_wrapper(in, nx, ny, out) bind(c, name='square_wrapper')
!            use iso_c_binding
!            integer(c_int), value, intent(in)::nx, ny
!            real(c_double), intent(in):: in(nx, ny)
!            real(c_double), intent(out) :: out(nx, ny)
!        end subroutine square_wrapper
!
!    end interface
!
!end module square_wrapper_plugin
