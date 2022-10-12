#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_10(double *w, double *diff_multfac_n2w,
                                    double *cell_area, double *z_nabla2_c,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_10(const double *w_dsl, const double *w,
                                       const double w_rel_tol,
                                       const double w_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_10(
    double *w, double *diff_multfac_n2w, double *cell_area, double *z_nabla2_c,
    double *w_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd, const double w_rel_tol,
    const double w_abs_tol);
void setup_mo_nh_diffusion_stencil_10(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream, const int w_k_size);
void free_mo_nh_diffusion_stencil_10();
}
