#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_19(
    double *inv_dual_edge_length, double *z_exner_ex_pr, double *ddxn_z_full,
    double *c_lin_e, double *z_dexner_dz_c_1, double *z_gradh_exner,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_19(const double *z_gradh_exner_dsl,
                                         const double *z_gradh_exner,
                                         const double z_gradh_exner_rel_tol,
                                         const double z_gradh_exner_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_19(
    double *inv_dual_edge_length, double *z_exner_ex_pr, double *ddxn_z_full,
    double *c_lin_e, double *z_dexner_dz_c_1, double *z_gradh_exner,
    double *z_gradh_exner_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_gradh_exner_rel_tol, const double z_gradh_exner_abs_tol);
void setup_mo_solve_nonhydro_stencil_19(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_gradh_exner_k_size);
void free_mo_solve_nonhydro_stencil_19();
}
