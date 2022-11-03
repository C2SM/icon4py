#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_06(double *z_nabla2_e, double *area_edge,
                                    double *vn, double fac_bdydiff_v,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_06(const double *vn_dsl, const double *vn,
                                       const double vn_rel_tol,
                                       const double vn_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_06(
    double *z_nabla2_e, double *area_edge, double *vn, double fac_bdydiff_v,
    double *vn_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd, const double vn_rel_tol,
    const double vn_abs_tol);
void setup_mo_nh_diffusion_stencil_06(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream, const int vn_k_size);
void free_mo_nh_diffusion_stencil_06();
}
