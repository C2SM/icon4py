#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_01(double *vn, double *rbf_vec_coeff_e,
                                          double *vt, const int verticalStart,
                                          const int verticalEnd,
                                          const int horizontalStart,
                                          const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_01(const double *vt_dsl,
                                             const double *vt,
                                             const double vt_rel_tol,
                                             const double vt_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_01(
    double *vn, double *rbf_vec_coeff_e, double *vt, double *vt_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vt_rel_tol, const double vt_abs_tol);
void setup_mo_velocity_advection_stencil_01(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int vt_k_size);
void free_mo_velocity_advection_stencil_01();
}
