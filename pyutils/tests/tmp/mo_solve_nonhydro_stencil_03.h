#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_03(double *z_exner_ex_pr,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_03(const double *z_exner_ex_pr_dsl,
                                         const double *z_exner_ex_pr,
                                         const double z_exner_ex_pr_rel_tol,
                                         const double z_exner_ex_pr_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_03(
    double *z_exner_ex_pr, double *z_exner_ex_pr_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_exner_ex_pr_rel_tol,
    const double z_exner_ex_pr_abs_tol);
void setup_mo_solve_nonhydro_stencil_03(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_exner_ex_pr_k_size);
void free_mo_solve_nonhydro_stencil_03();
}
