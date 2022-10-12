#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_53(double *z_q, double *w,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_53(const double *w_dsl, const double *w,
                                         const double w_rel_tol,
                                         const double w_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_53(
    double *z_q, double *w, double *w_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double w_rel_tol, const double w_abs_tol);
void setup_mo_solve_nonhydro_stencil_53(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int w_k_size);
void free_mo_solve_nonhydro_stencil_53();
}
