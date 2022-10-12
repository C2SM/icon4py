#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_02(
    double *exner_exfac, double *exner, double *exner_ref_mc, double *exner_pr,
    double *z_exner_ex_pr, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_02(
    const double *exner_pr_dsl, const double *exner_pr,
    const double *z_exner_ex_pr_dsl, const double *z_exner_ex_pr,
    const double exner_pr_rel_tol, const double exner_pr_abs_tol,
    const double z_exner_ex_pr_rel_tol, const double z_exner_ex_pr_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_02(
    double *exner_exfac, double *exner, double *exner_ref_mc, double *exner_pr,
    double *z_exner_ex_pr, double *exner_pr_before,
    double *z_exner_ex_pr_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double exner_pr_rel_tol, const double exner_pr_abs_tol,
    const double z_exner_ex_pr_rel_tol, const double z_exner_ex_pr_abs_tol);
void setup_mo_solve_nonhydro_stencil_02(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int exner_pr_k_size,
                                        const int z_exner_ex_pr_k_size);
void free_mo_solve_nonhydro_stencil_02();
}
