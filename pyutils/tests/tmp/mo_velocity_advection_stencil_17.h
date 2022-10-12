#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_17(double *e_bln_c_s, double *z_v_grad_w,
                                          double *ddt_w_adv,
                                          const int verticalStart,
                                          const int verticalEnd,
                                          const int horizontalStart,
                                          const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_17(const double *ddt_w_adv_dsl,
                                             const double *ddt_w_adv,
                                             const double ddt_w_adv_rel_tol,
                                             const double ddt_w_adv_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_17(
    double *e_bln_c_s, double *z_v_grad_w, double *ddt_w_adv,
    double *ddt_w_adv_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double ddt_w_adv_rel_tol, const double ddt_w_adv_abs_tol);
void setup_mo_velocity_advection_stencil_17(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int ddt_w_adv_k_size);
void free_mo_velocity_advection_stencil_17();
}
