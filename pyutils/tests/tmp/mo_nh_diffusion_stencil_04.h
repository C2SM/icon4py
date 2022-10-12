#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_04(
    double *u_vert, double *v_vert, double *primal_normal_vert_v1,
    double *primal_normal_vert_v2, double *z_nabla2_e,
    double *inv_vert_vert_length, double *inv_primal_edge_length,
    double *z_nabla4_e2, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_04(const double *z_nabla4_e2_dsl,
                                       const double *z_nabla4_e2,
                                       const double z_nabla4_e2_rel_tol,
                                       const double z_nabla4_e2_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_04(
    double *u_vert, double *v_vert, double *primal_normal_vert_v1,
    double *primal_normal_vert_v2, double *z_nabla2_e,
    double *inv_vert_vert_length, double *inv_primal_edge_length,
    double *z_nabla4_e2, double *z_nabla4_e2_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_nabla4_e2_rel_tol, const double z_nabla4_e2_abs_tol);
void setup_mo_nh_diffusion_stencil_04(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int z_nabla4_e2_k_size);
void free_mo_nh_diffusion_stencil_04();
}
