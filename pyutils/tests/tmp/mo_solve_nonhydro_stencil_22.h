#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_22(
    int *ipeidx_dsl, double *pg_exdist, double *z_hydro_corr,
    double *z_gradh_exner, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_22(const double *z_gradh_exner_dsl,
                                         const double *z_gradh_exner,
                                         const double z_gradh_exner_rel_tol,
                                         const double z_gradh_exner_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_22(
    int *ipeidx_dsl, double *pg_exdist, double *z_hydro_corr,
    double *z_gradh_exner, double *z_gradh_exner_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_gradh_exner_rel_tol,
    const double z_gradh_exner_abs_tol);
void setup_mo_solve_nonhydro_stencil_22(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_gradh_exner_k_size);
void free_mo_solve_nonhydro_stencil_22();
}
