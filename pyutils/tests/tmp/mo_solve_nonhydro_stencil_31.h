#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_31(double *e_flx_avg, double *vn,
                                      double *z_vn_avg, const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_31(const double *z_vn_avg_dsl,
                                         const double *z_vn_avg,
                                         const double z_vn_avg_rel_tol,
                                         const double z_vn_avg_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_31(
    double *e_flx_avg, double *vn, double *z_vn_avg, double *z_vn_avg_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_vn_avg_rel_tol,
    const double z_vn_avg_abs_tol);
void setup_mo_solve_nonhydro_stencil_31(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_vn_avg_k_size);
void free_mo_solve_nonhydro_stencil_31();
}
