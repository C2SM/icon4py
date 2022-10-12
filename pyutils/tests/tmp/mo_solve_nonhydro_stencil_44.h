#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_44(
    double *z_beta, double *exner_nnow, double *rho_nnow, double *theta_v_nnow,
    double *inv_ddqz_z_full, double *z_alpha, double *vwind_impl_wgt,
    double *theta_v_ic, double *rho_ic, double dtime, double rd, double cvd,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_44(
    const double *z_beta_dsl, const double *z_beta, const double *z_alpha_dsl,
    const double *z_alpha, const double z_beta_rel_tol,
    const double z_beta_abs_tol, const double z_alpha_rel_tol,
    const double z_alpha_abs_tol, const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_44(
    double *z_beta, double *exner_nnow, double *rho_nnow, double *theta_v_nnow,
    double *inv_ddqz_z_full, double *z_alpha, double *vwind_impl_wgt,
    double *theta_v_ic, double *rho_ic, double dtime, double rd, double cvd,
    double *z_beta_before, double *z_alpha_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_beta_rel_tol, const double z_beta_abs_tol,
    const double z_alpha_rel_tol, const double z_alpha_abs_tol);
void setup_mo_solve_nonhydro_stencil_44(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_beta_k_size,
                                        const int z_alpha_k_size);
void free_mo_solve_nonhydro_stencil_44();
}
