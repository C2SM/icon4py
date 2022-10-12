#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_01(double *z_rth_pr_1, double *z_rth_pr_2,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_01(
    const double *z_rth_pr_1_dsl, const double *z_rth_pr_1,
    const double *z_rth_pr_2_dsl, const double *z_rth_pr_2,
    const double z_rth_pr_1_rel_tol, const double z_rth_pr_1_abs_tol,
    const double z_rth_pr_2_rel_tol, const double z_rth_pr_2_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_01(
    double *z_rth_pr_1, double *z_rth_pr_2, double *z_rth_pr_1_before,
    double *z_rth_pr_2_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_rth_pr_1_rel_tol, const double z_rth_pr_1_abs_tol,
    const double z_rth_pr_2_rel_tol, const double z_rth_pr_2_abs_tol);
void setup_mo_solve_nonhydro_stencil_01(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_rth_pr_1_k_size,
                                        const int z_rth_pr_2_k_size);
void free_mo_solve_nonhydro_stencil_01();
}
