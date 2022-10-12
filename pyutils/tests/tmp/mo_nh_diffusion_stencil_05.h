#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_05(
    double *area_edge, double *kh_smag_e, double *z_nabla2_e,
    double *z_nabla4_e2, double *diff_multfac_vn, double *nudgecoeff_e,
    double *vn, double nudgezone_diff, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_05(const double *vn_dsl, const double *vn,
                                       const double vn_rel_tol,
                                       const double vn_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_05(
    double *area_edge, double *kh_smag_e, double *z_nabla2_e,
    double *z_nabla4_e2, double *diff_multfac_vn, double *nudgecoeff_e,
    double *vn, double nudgezone_diff, double *vn_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_rel_tol, const double vn_abs_tol);
void setup_mo_nh_diffusion_stencil_05(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream, const int vn_k_size);
void free_mo_nh_diffusion_stencil_05();
}
