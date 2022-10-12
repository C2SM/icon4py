#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_04(
    double *vn, double *ddxn_z_full, double *ddxt_z_full, double *vt,
    double *z_w_concorr_me, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_04(
    const double *z_w_concorr_me_dsl, const double *z_w_concorr_me,
    const double z_w_concorr_me_rel_tol, const double z_w_concorr_me_abs_tol,
    const int iteration);
void run_and_verify_mo_velocity_advection_stencil_04(
    double *vn, double *ddxn_z_full, double *ddxt_z_full, double *vt,
    double *z_w_concorr_me, double *z_w_concorr_me_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_w_concorr_me_rel_tol,
    const double z_w_concorr_me_abs_tol);
void setup_mo_velocity_advection_stencil_04(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_w_concorr_me_k_size);
void free_mo_velocity_advection_stencil_04();
}
