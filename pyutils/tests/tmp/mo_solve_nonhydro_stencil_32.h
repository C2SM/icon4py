#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_32(
    double *z_rho_e, double *z_vn_avg, double *ddqz_z_full_e,
    double *z_theta_v_e, double *mass_fl_e, double *z_theta_v_fl_e,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_32(
    const double *mass_fl_e_dsl, const double *mass_fl_e,
    const double *z_theta_v_fl_e_dsl, const double *z_theta_v_fl_e,
    const double mass_fl_e_rel_tol, const double mass_fl_e_abs_tol,
    const double z_theta_v_fl_e_rel_tol, const double z_theta_v_fl_e_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_32(
    double *z_rho_e, double *z_vn_avg, double *ddqz_z_full_e,
    double *z_theta_v_e, double *mass_fl_e, double *z_theta_v_fl_e,
    double *mass_fl_e_before, double *z_theta_v_fl_e_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double mass_fl_e_rel_tol,
    const double mass_fl_e_abs_tol, const double z_theta_v_fl_e_rel_tol,
    const double z_theta_v_fl_e_abs_tol);
void setup_mo_solve_nonhydro_stencil_32(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int mass_fl_e_k_size,
                                        const int z_theta_v_fl_e_k_size);
void free_mo_solve_nonhydro_stencil_32();
}
