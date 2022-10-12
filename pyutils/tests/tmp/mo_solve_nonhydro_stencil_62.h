#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_62(double *w_now, double *grf_tend_w,
                                      double *w_new, double dtime,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_62(const double *w_new_dsl,
                                         const double *w_new,
                                         const double w_new_rel_tol,
                                         const double w_new_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_62(
    double *w_now, double *grf_tend_w, double *w_new, double dtime,
    double *w_new_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double w_new_rel_tol, const double w_new_abs_tol);
void setup_mo_solve_nonhydro_stencil_62(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int w_new_k_size);
void free_mo_solve_nonhydro_stencil_62();
}
