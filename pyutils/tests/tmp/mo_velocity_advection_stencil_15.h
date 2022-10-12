#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_15(
    double *z_w_con_c, double *z_w_con_c_full, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_15(
    const double *z_w_con_c_full_dsl, const double *z_w_con_c_full,
    const double z_w_con_c_full_rel_tol, const double z_w_con_c_full_abs_tol,
    const int iteration);
void run_and_verify_mo_velocity_advection_stencil_15(
    double *z_w_con_c, double *z_w_con_c_full, double *z_w_con_c_full_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_w_con_c_full_rel_tol,
    const double z_w_con_c_full_abs_tol);
void setup_mo_velocity_advection_stencil_15(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_w_con_c_full_k_size);
void free_mo_velocity_advection_stencil_15();
}
