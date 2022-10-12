#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_11_lower(double *z_theta_v_pr_ic,
                                            const int verticalStart,
                                            const int verticalEnd,
                                            const int horizontalStart,
                                            const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_11_lower(
    const double *z_theta_v_pr_ic_dsl, const double *z_theta_v_pr_ic,
    const double z_theta_v_pr_ic_rel_tol, const double z_theta_v_pr_ic_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_11_lower(
    double *z_theta_v_pr_ic, double *z_theta_v_pr_ic_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_theta_v_pr_ic_rel_tol,
    const double z_theta_v_pr_ic_abs_tol);
void setup_mo_solve_nonhydro_stencil_11_lower(dawn::GlobalGpuTriMesh *mesh,
                                              int k_size, cudaStream_t stream,
                                              const int z_theta_v_pr_ic_k_size);
void free_mo_solve_nonhydro_stencil_11_lower();
}
