#include <openacc.h>
#include <stdio.h>
#include <stdint.h> // Include for uintptr_t

uintptr_t get_device_ptr_from_fortran_array(double *host_array) {
    void* device_ptr = acc_deviceptr(host_array);
    printf("Host pointer = %p\n", (void*)host_array);
    printf("Device pointer = %p\n", device_ptr);

    // Cast the device pointer to uintptr_t before returning
    return (uintptr_t)device_ptr;
}
