#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_08(double *z_kin_hor_e,
                                          double *e_bln_c_s, double *z_ekinh,
                                          const int verticalStart,
                                          const int verticalEnd,
                                          const int horizontalStart,
                                          const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_08(const double *z_ekinh_dsl,
                                             const double *z_ekinh,
                                             const double z_ekinh_rel_tol,
                                             const double z_ekinh_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_08(
    double *z_kin_hor_e, double *e_bln_c_s, double *z_ekinh,
    double *z_ekinh_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_ekinh_rel_tol, const double z_ekinh_abs_tol);
void setup_mo_velocity_advection_stencil_08(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int z_ekinh_k_size);
void free_mo_velocity_advection_stencil_08();
}
