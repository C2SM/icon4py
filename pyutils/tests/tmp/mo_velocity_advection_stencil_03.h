#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_03(
    double *wgtfac_e, double *vt, double *z_vt_ie, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_03(const double *z_vt_ie_dsl,
                                             const double *z_vt_ie,
                                             const double z_vt_ie_rel_tol,
                                             const double z_vt_ie_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_03(
    double *wgtfac_e, double *vt, double *z_vt_ie, double *z_vt_ie_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_vt_ie_rel_tol,
    const double z_vt_ie_abs_tol);
void setup_mo_velocity_advection_stencil_03(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_vt_ie_k_size);
void free_mo_velocity_advection_stencil_03();
}
