#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_43(
    double *z_w_expl, double *w_nnow, double *ddt_w_adv_ntl1,
    double *z_th_ddz_exner_c, double *z_contr_w_fl_l, double *rho_ic,
    double *w_concorr_c, double *vwind_expl_wgt, double dtime, double cpd,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_43(
    const double *z_w_expl_dsl, const double *z_w_expl,
    const double *z_contr_w_fl_l_dsl, const double *z_contr_w_fl_l,
    const double z_w_expl_rel_tol, const double z_w_expl_abs_tol,
    const double z_contr_w_fl_l_rel_tol, const double z_contr_w_fl_l_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_43(
    double *z_w_expl, double *w_nnow, double *ddt_w_adv_ntl1,
    double *z_th_ddz_exner_c, double *z_contr_w_fl_l, double *rho_ic,
    double *w_concorr_c, double *vwind_expl_wgt, double dtime, double cpd,
    double *z_w_expl_before, double *z_contr_w_fl_l_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_w_expl_rel_tol,
    const double z_w_expl_abs_tol, const double z_contr_w_fl_l_rel_tol,
    const double z_contr_w_fl_l_abs_tol);
void setup_mo_solve_nonhydro_stencil_43(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_w_expl_k_size,
                                        const int z_contr_w_fl_l_k_size);
void free_mo_solve_nonhydro_stencil_43();
}
