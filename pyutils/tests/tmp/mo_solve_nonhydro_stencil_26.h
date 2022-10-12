#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_26(double *z_graddiv_vn, double *vn,
                                      double scal_divdamp_o2,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_26(const double *vn_dsl, const double *vn,
                                         const double vn_rel_tol,
                                         const double vn_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_26(
    double *z_graddiv_vn, double *vn, double scal_divdamp_o2, double *vn_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_rel_tol, const double vn_abs_tol);
void setup_mo_solve_nonhydro_stencil_26(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_k_size);
void free_mo_solve_nonhydro_stencil_26();
}
