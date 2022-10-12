#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_60(
    double *exner, double *ddt_exner_phy, double *exner_dyn_incr,
    double ndyn_substeps_var, double dtime, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_60(const double *exner_dyn_incr_dsl,
                                         const double *exner_dyn_incr,
                                         const double exner_dyn_incr_rel_tol,
                                         const double exner_dyn_incr_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_60(
    double *exner, double *ddt_exner_phy, double *exner_dyn_incr,
    double ndyn_substeps_var, double dtime, double *exner_dyn_incr_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double exner_dyn_incr_rel_tol,
    const double exner_dyn_incr_abs_tol);
void setup_mo_solve_nonhydro_stencil_60(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int exner_dyn_incr_k_size);
void free_mo_solve_nonhydro_stencil_60();
}
