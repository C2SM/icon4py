#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_55(
    double *z_rho_expl, double *vwind_impl_wgt, double *inv_ddqz_z_full,
    double *rho_ic, double *w, double *z_exner_expl, double *exner_ref_mc,
    double *z_alpha, double *z_beta, double *rho_now, double *theta_v_now,
    double *exner_now, double *rho_new, double *exner_new, double *theta_v_new,
    double dtime, double cvd_o_rd, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_55(
    const double *rho_new_dsl, const double *rho_new,
    const double *exner_new_dsl, const double *exner_new,
    const double *theta_v_new_dsl, const double *theta_v_new,
    const double rho_new_rel_tol, const double rho_new_abs_tol,
    const double exner_new_rel_tol, const double exner_new_abs_tol,
    const double theta_v_new_rel_tol, const double theta_v_new_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_55(
    double *z_rho_expl, double *vwind_impl_wgt, double *inv_ddqz_z_full,
    double *rho_ic, double *w, double *z_exner_expl, double *exner_ref_mc,
    double *z_alpha, double *z_beta, double *rho_now, double *theta_v_now,
    double *exner_now, double *rho_new, double *exner_new, double *theta_v_new,
    double dtime, double cvd_o_rd, double *rho_new_before,
    double *exner_new_before, double *theta_v_new_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double rho_new_rel_tol,
    const double rho_new_abs_tol, const double exner_new_rel_tol,
    const double exner_new_abs_tol, const double theta_v_new_rel_tol,
    const double theta_v_new_abs_tol);
void setup_mo_solve_nonhydro_stencil_55(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int rho_new_k_size,
                                        const int exner_new_k_size,
                                        const int theta_v_new_k_size);
void free_mo_solve_nonhydro_stencil_55();
}
