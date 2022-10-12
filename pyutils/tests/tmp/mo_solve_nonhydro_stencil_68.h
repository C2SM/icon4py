#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_68(
    int *mask_prog_halo_c, double *rho_now, double *theta_v_now,
    double *exner_new, double *exner_now, double *rho_new, double *theta_v_new,
    double cvd_o_rd, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_68(const double *theta_v_new_dsl,
                                         const double *theta_v_new,
                                         const double theta_v_new_rel_tol,
                                         const double theta_v_new_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_68(
    int *mask_prog_halo_c, double *rho_now, double *theta_v_now,
    double *exner_new, double *exner_now, double *rho_new, double *theta_v_new,
    double cvd_o_rd, double *theta_v_new_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double theta_v_new_rel_tol, const double theta_v_new_abs_tol);
void setup_mo_solve_nonhydro_stencil_68(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int theta_v_new_k_size);
void free_mo_solve_nonhydro_stencil_68();
}
