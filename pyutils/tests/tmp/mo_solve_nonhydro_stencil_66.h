#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_66(
    int *bdy_halo_c, double *rho, double *theta_v, double *exner,
    double rd_o_cvd, double rd_o_p0ref, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_66(
    const double *theta_v_dsl, const double *theta_v, const double *exner_dsl,
    const double *exner, const double theta_v_rel_tol,
    const double theta_v_abs_tol, const double exner_rel_tol,
    const double exner_abs_tol, const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_66(
    int *bdy_halo_c, double *rho, double *theta_v, double *exner,
    double rd_o_cvd, double rd_o_p0ref, double *theta_v_before,
    double *exner_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double theta_v_rel_tol, const double theta_v_abs_tol,
    const double exner_rel_tol, const double exner_abs_tol);
void setup_mo_solve_nonhydro_stencil_66(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int theta_v_k_size,
                                        const int exner_k_size);
void free_mo_solve_nonhydro_stencil_66();
}
