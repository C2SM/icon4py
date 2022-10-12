#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_51(
    double *z_q, double *w_nnew, double *vwind_impl_wgt, double *theta_v_ic,
    double *ddqz_z_half, double *z_beta, double *z_alpha, double *z_w_expl,
    double *z_exner_expl, double dtime, double cpd, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_51(
    const double *z_q_dsl, const double *z_q, const double *w_nnew_dsl,
    const double *w_nnew, const double z_q_rel_tol, const double z_q_abs_tol,
    const double w_nnew_rel_tol, const double w_nnew_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_51(
    double *z_q, double *w_nnew, double *vwind_impl_wgt, double *theta_v_ic,
    double *ddqz_z_half, double *z_beta, double *z_alpha, double *z_w_expl,
    double *z_exner_expl, double dtime, double cpd, double *z_q_before,
    double *w_nnew_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_q_rel_tol, const double z_q_abs_tol,
    const double w_nnew_rel_tol, const double w_nnew_abs_tol);
void setup_mo_solve_nonhydro_stencil_51(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_q_k_size,
                                        const int w_nnew_k_size);
void free_mo_solve_nonhydro_stencil_51();
}
