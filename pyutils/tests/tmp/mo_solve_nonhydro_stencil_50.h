#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_50(
    double *z_rho_expl, double *z_exner_expl, double *rho_incr,
    double *exner_incr, double iau_wgt_dyn, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_50(
    const double *z_rho_expl_dsl, const double *z_rho_expl,
    const double *z_exner_expl_dsl, const double *z_exner_expl,
    const double z_rho_expl_rel_tol, const double z_rho_expl_abs_tol,
    const double z_exner_expl_rel_tol, const double z_exner_expl_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_50(
    double *z_rho_expl, double *z_exner_expl, double *rho_incr,
    double *exner_incr, double iau_wgt_dyn, double *z_rho_expl_before,
    double *z_exner_expl_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_rho_expl_rel_tol, const double z_rho_expl_abs_tol,
    const double z_exner_expl_rel_tol, const double z_exner_expl_abs_tol);
void setup_mo_solve_nonhydro_stencil_50(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_rho_expl_k_size,
                                        const int z_exner_expl_k_size);
void free_mo_solve_nonhydro_stencil_50();
}
