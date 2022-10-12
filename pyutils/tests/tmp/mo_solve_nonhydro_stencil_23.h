#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_23(
    double *vn_nnow, double *ddt_vn_adv_ntl1, double *ddt_vn_adv_ntl2,
    double *ddt_vn_phy, double *z_theta_v_e, double *z_gradh_exner,
    double *vn_nnew, double dtime, double wgt_nnow_vel, double wgt_nnew_vel,
    double cpd, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_23(const double *vn_nnew_dsl,
                                         const double *vn_nnew,
                                         const double vn_nnew_rel_tol,
                                         const double vn_nnew_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_23(
    double *vn_nnow, double *ddt_vn_adv_ntl1, double *ddt_vn_adv_ntl2,
    double *ddt_vn_phy, double *z_theta_v_e, double *z_gradh_exner,
    double *vn_nnew, double dtime, double wgt_nnow_vel, double wgt_nnew_vel,
    double cpd, double *vn_nnew_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double vn_nnew_rel_tol, const double vn_nnew_abs_tol);
void setup_mo_solve_nonhydro_stencil_23(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_nnew_k_size);
void free_mo_solve_nonhydro_stencil_23();
}
