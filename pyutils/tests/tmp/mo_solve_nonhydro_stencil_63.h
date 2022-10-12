#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_63(double *inv_ddqz_z_full, double *w,
                                      double *w_concorr_c, double *z_dwdz_dd,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_63(const double *z_dwdz_dd_dsl,
                                         const double *z_dwdz_dd,
                                         const double z_dwdz_dd_rel_tol,
                                         const double z_dwdz_dd_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_63(
    double *inv_ddqz_z_full, double *w, double *w_concorr_c, double *z_dwdz_dd,
    double *z_dwdz_dd_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_dwdz_dd_rel_tol, const double z_dwdz_dd_abs_tol);
void setup_mo_solve_nonhydro_stencil_63(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_dwdz_dd_k_size);
void free_mo_solve_nonhydro_stencil_63();
}
