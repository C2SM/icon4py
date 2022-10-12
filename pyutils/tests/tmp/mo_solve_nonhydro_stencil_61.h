#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_61(
    double *rho_now, double *grf_tend_rho, double *theta_v_now,
    double *grf_tend_thv, double *w_now, double *grf_tend_w, double *rho_new,
    double *exner_new, double *w_new, double dtime, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_61(
    const double *rho_new_dsl, const double *rho_new,
    const double *exner_new_dsl, const double *exner_new,
    const double *w_new_dsl, const double *w_new, const double rho_new_rel_tol,
    const double rho_new_abs_tol, const double exner_new_rel_tol,
    const double exner_new_abs_tol, const double w_new_rel_tol,
    const double w_new_abs_tol, const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_61(
    double *rho_now, double *grf_tend_rho, double *theta_v_now,
    double *grf_tend_thv, double *w_now, double *grf_tend_w, double *rho_new,
    double *exner_new, double *w_new, double dtime, double *rho_new_before,
    double *exner_new_before, double *w_new_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double rho_new_rel_tol, const double rho_new_abs_tol,
    const double exner_new_rel_tol, const double exner_new_abs_tol,
    const double w_new_rel_tol, const double w_new_abs_tol);
void setup_mo_solve_nonhydro_stencil_61(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int rho_new_k_size,
                                        const int exner_new_k_size,
                                        const int w_new_k_size);
void free_mo_solve_nonhydro_stencil_61();
}
