#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_14(double *z_nabla2_e, double *geofac_div,
                                    double *z_temp, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_14(const double *z_temp_dsl,
                                       const double *z_temp,
                                       const double z_temp_rel_tol,
                                       const double z_temp_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_14(
    double *z_nabla2_e, double *geofac_div, double *z_temp,
    double *z_temp_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_temp_rel_tol, const double z_temp_abs_tol);
void setup_mo_nh_diffusion_stencil_14(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int z_temp_k_size);
void free_mo_nh_diffusion_stencil_14();
}
