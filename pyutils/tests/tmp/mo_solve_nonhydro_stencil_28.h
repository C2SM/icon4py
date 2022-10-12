#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_28(
    double *vn_incr, double *vn, double iau_wgt_dyn, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_28(const double *vn_dsl, const double *vn,
                                         const double vn_rel_tol,
                                         const double vn_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_28(
    double *vn_incr, double *vn, double iau_wgt_dyn, double *vn_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_rel_tol, const double vn_abs_tol);
void setup_mo_solve_nonhydro_stencil_28(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_k_size);
void free_mo_solve_nonhydro_stencil_28();
}
