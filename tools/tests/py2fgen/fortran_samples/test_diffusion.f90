program call_diffusion_test_case_cffi_plugin
    use, intrinsic :: iso_c_binding
    use diffusion_test_case_plugin
    implicit none

    ! call the cffi plugin
    call run_diffusion_test_case()

end program call_diffusion_test_case_cffi_plugin
