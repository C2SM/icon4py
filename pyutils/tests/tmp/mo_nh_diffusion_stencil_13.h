#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_13(
    double *kh_smag_e, double *inv_dual_edge_length, double *theta_v,
    double *z_nabla2_e, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_13(const double *z_nabla2_e_dsl,
                                       const double *z_nabla2_e,
                                       const double z_nabla2_e_rel_tol,
                                       const double z_nabla2_e_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_13(
    double *kh_smag_e, double *inv_dual_edge_length, double *theta_v,
    double *z_nabla2_e, double *z_nabla2_e_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_nabla2_e_rel_tol, const double z_nabla2_e_abs_tol);
void setup_mo_nh_diffusion_stencil_13(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int z_nabla2_e_k_size);
void free_mo_nh_diffusion_stencil_13();
}
