#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_06(
    double *z_exner_ic, double *inv_ddqz_z_full, double *z_dexner_dz_c_1,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_06(const double *z_dexner_dz_c_1_dsl,
                                         const double *z_dexner_dz_c_1,
                                         const double z_dexner_dz_c_1_rel_tol,
                                         const double z_dexner_dz_c_1_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_06(
    double *z_exner_ic, double *inv_ddqz_z_full, double *z_dexner_dz_c_1,
    double *z_dexner_dz_c_1_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_dexner_dz_c_1_rel_tol, const double z_dexner_dz_c_1_abs_tol);
void setup_mo_solve_nonhydro_stencil_06(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_dexner_dz_c_1_k_size);
void free_mo_solve_nonhydro_stencil_06();
}
