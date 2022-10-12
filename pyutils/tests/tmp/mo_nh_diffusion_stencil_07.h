#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_07(double *w, double *geofac_n2s,
                                    double *z_nabla2_c, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_07(const double *z_nabla2_c_dsl,
                                       const double *z_nabla2_c,
                                       const double z_nabla2_c_rel_tol,
                                       const double z_nabla2_c_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_07(
    double *w, double *geofac_n2s, double *z_nabla2_c,
    double *z_nabla2_c_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_nabla2_c_rel_tol, const double z_nabla2_c_abs_tol);
void setup_mo_nh_diffusion_stencil_07(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int z_nabla2_c_k_size);
void free_mo_nh_diffusion_stencil_07();
}
