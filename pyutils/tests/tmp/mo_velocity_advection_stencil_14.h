#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_14(
    double *ddqz_z_half, double *z_w_con_c, double *cfl_clipping,
    double *pre_levelmask, double *vcfl, double cfl_w_limit, double dtime,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_14(
    const double *z_w_con_c_dsl, const double *z_w_con_c,
    const double *cfl_clipping_dsl, const double *cfl_clipping,
    const double *pre_levelmask_dsl, const double *pre_levelmask,
    const double *vcfl_dsl, const double *vcfl, const double z_w_con_c_rel_tol,
    const double z_w_con_c_abs_tol, const double cfl_clipping_rel_tol,
    const double cfl_clipping_abs_tol, const double pre_levelmask_rel_tol,
    const double pre_levelmask_abs_tol, const double vcfl_rel_tol,
    const double vcfl_abs_tol, const int iteration);
void run_and_verify_mo_velocity_advection_stencil_14(
    double *ddqz_z_half, double *z_w_con_c, double *cfl_clipping,
    double *pre_levelmask, double *vcfl, double cfl_w_limit, double dtime,
    double *z_w_con_c_before, double *cfl_clipping_before,
    double *pre_levelmask_before, double *vcfl_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_w_con_c_rel_tol, const double z_w_con_c_abs_tol,
    const double cfl_clipping_rel_tol, const double cfl_clipping_abs_tol,
    const double pre_levelmask_rel_tol, const double pre_levelmask_abs_tol,
    const double vcfl_rel_tol, const double vcfl_abs_tol);
void setup_mo_velocity_advection_stencil_14(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_w_con_c_k_size,
                                            const int cfl_clipping_k_size,
                                            const int pre_levelmask_k_size,
                                            const int vcfl_k_size);
void free_mo_velocity_advection_stencil_14();
}
