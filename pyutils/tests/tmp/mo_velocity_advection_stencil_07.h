#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_07(
    double *vn_ie, double *inv_dual_edge_length, double *w, double *z_vt_ie,
    double *inv_primal_edge_length, double *tangent_orientation, double *z_w_v,
    double *z_v_grad_w, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_07(const double *z_v_grad_w_dsl,
                                             const double *z_v_grad_w,
                                             const double z_v_grad_w_rel_tol,
                                             const double z_v_grad_w_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_07(
    double *vn_ie, double *inv_dual_edge_length, double *w, double *z_vt_ie,
    double *inv_primal_edge_length, double *tangent_orientation, double *z_w_v,
    double *z_v_grad_w, double *z_v_grad_w_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_v_grad_w_rel_tol, const double z_v_grad_w_abs_tol);
void setup_mo_velocity_advection_stencil_07(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_v_grad_w_k_size);
void free_mo_velocity_advection_stencil_07();
}
