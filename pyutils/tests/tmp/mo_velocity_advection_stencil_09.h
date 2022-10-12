#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_09(
    double *z_w_concorr_me, double *e_bln_c_s, double *z_w_concorr_mc,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_09(
    const double *z_w_concorr_mc_dsl, const double *z_w_concorr_mc,
    const double z_w_concorr_mc_rel_tol, const double z_w_concorr_mc_abs_tol,
    const int iteration);
void run_and_verify_mo_velocity_advection_stencil_09(
    double *z_w_concorr_me, double *e_bln_c_s, double *z_w_concorr_mc,
    double *z_w_concorr_mc_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_w_concorr_mc_rel_tol, const double z_w_concorr_mc_abs_tol);
void setup_mo_velocity_advection_stencil_09(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_w_concorr_mc_k_size);
void free_mo_velocity_advection_stencil_09();
}
