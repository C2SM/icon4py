#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_12(
    double *z_theta_v_pr_ic, double *d2dexdz2_fac1_mc, double *d2dexdz2_fac2_mc,
    double *z_rth_pr_2, double *z_dexner_dz_c_2, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_12(const double *z_dexner_dz_c_2_dsl,
                                         const double *z_dexner_dz_c_2,
                                         const double z_dexner_dz_c_2_rel_tol,
                                         const double z_dexner_dz_c_2_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_12(
    double *z_theta_v_pr_ic, double *d2dexdz2_fac1_mc, double *d2dexdz2_fac2_mc,
    double *z_rth_pr_2, double *z_dexner_dz_c_2, double *z_dexner_dz_c_2_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_dexner_dz_c_2_rel_tol,
    const double z_dexner_dz_c_2_abs_tol);
void setup_mo_solve_nonhydro_stencil_12(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_dexner_dz_c_2_k_size);
void free_mo_solve_nonhydro_stencil_12();
}
