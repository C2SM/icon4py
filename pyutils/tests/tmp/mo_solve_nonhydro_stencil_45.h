#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_45(double *z_alpha, const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_45(const double *z_alpha_dsl,
                                         const double *z_alpha,
                                         const double z_alpha_rel_tol,
                                         const double z_alpha_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_45(
    double *z_alpha, double *z_alpha_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_alpha_rel_tol, const double z_alpha_abs_tol);
void setup_mo_solve_nonhydro_stencil_45(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_alpha_k_size);
void free_mo_solve_nonhydro_stencil_45();
}
