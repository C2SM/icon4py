#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_29(double *grf_tend_vn, double *vn_now,
                                      double *vn_new, double dtime,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_29(const double *vn_new_dsl,
                                         const double *vn_new,
                                         const double vn_new_rel_tol,
                                         const double vn_new_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_29(
    double *grf_tend_vn, double *vn_now, double *vn_new, double dtime,
    double *vn_new_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double vn_new_rel_tol, const double vn_new_abs_tol);
void setup_mo_solve_nonhydro_stencil_29(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_new_k_size);
void free_mo_solve_nonhydro_stencil_29();
}
