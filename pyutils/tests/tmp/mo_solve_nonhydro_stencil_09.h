#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_09(
    double *wgtfac_c, double *z_rth_pr_2, double *theta_v,
    double *vwind_expl_wgt, double *exner_pr, double *d_exner_dz_ref_ic,
    double *ddqz_z_half, double *z_theta_v_pr_ic, double *theta_v_ic,
    double *z_th_ddz_exner_c, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_09(
    const double *z_theta_v_pr_ic_dsl, const double *z_theta_v_pr_ic,
    const double *theta_v_ic_dsl, const double *theta_v_ic,
    const double *z_th_ddz_exner_c_dsl, const double *z_th_ddz_exner_c,
    const double z_theta_v_pr_ic_rel_tol, const double z_theta_v_pr_ic_abs_tol,
    const double theta_v_ic_rel_tol, const double theta_v_ic_abs_tol,
    const double z_th_ddz_exner_c_rel_tol,
    const double z_th_ddz_exner_c_abs_tol, const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_09(
    double *wgtfac_c, double *z_rth_pr_2, double *theta_v,
    double *vwind_expl_wgt, double *exner_pr, double *d_exner_dz_ref_ic,
    double *ddqz_z_half, double *z_theta_v_pr_ic, double *theta_v_ic,
    double *z_th_ddz_exner_c, double *z_theta_v_pr_ic_before,
    double *theta_v_ic_before, double *z_th_ddz_exner_c_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_theta_v_pr_ic_rel_tol,
    const double z_theta_v_pr_ic_abs_tol, const double theta_v_ic_rel_tol,
    const double theta_v_ic_abs_tol, const double z_th_ddz_exner_c_rel_tol,
    const double z_th_ddz_exner_c_abs_tol);
void setup_mo_solve_nonhydro_stencil_09(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_theta_v_pr_ic_k_size,
                                        const int theta_v_ic_k_size,
                                        const int z_th_ddz_exner_c_k_size);
void free_mo_solve_nonhydro_stencil_09();
}
