#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_11(double *w, double *z_w_con_c,
                                          const int verticalStart,
                                          const int verticalEnd,
                                          const int horizontalStart,
                                          const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_11(const double *z_w_con_c_dsl,
                                             const double *z_w_con_c,
                                             const double z_w_con_c_rel_tol,
                                             const double z_w_con_c_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_11(
    double *w, double *z_w_con_c, double *z_w_con_c_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_w_con_c_rel_tol,
    const double z_w_con_c_abs_tol);
void setup_mo_velocity_advection_stencil_11(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_w_con_c_k_size);
void free_mo_velocity_advection_stencil_11();
}
