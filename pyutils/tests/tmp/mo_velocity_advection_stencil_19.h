#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_19(
    double *z_kin_hor_e, double *coeff_gradekin, double *z_ekinh, double *zeta,
    double *vt, double *f_e, double *c_lin_e, double *z_w_con_c_full,
    double *vn_ie, double *ddqz_z_full_e, double *ddt_vn_adv,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_19(const double *ddt_vn_adv_dsl,
                                             const double *ddt_vn_adv,
                                             const double ddt_vn_adv_rel_tol,
                                             const double ddt_vn_adv_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_19(
    double *z_kin_hor_e, double *coeff_gradekin, double *z_ekinh, double *zeta,
    double *vt, double *f_e, double *c_lin_e, double *z_w_con_c_full,
    double *vn_ie, double *ddqz_z_full_e, double *ddt_vn_adv,
    double *ddt_vn_adv_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double ddt_vn_adv_rel_tol, const double ddt_vn_adv_abs_tol);
void setup_mo_velocity_advection_stencil_19(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int ddt_vn_adv_k_size);
void free_mo_velocity_advection_stencil_19();
}
