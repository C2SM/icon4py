#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_41(
    double *geofac_div, double *mass_fl_e, double *z_theta_v_fl_e,
    double *z_flxdiv_mass, double *z_flxdiv_theta, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_41(
    const double *z_flxdiv_mass_dsl, const double *z_flxdiv_mass,
    const double *z_flxdiv_theta_dsl, const double *z_flxdiv_theta,
    const double z_flxdiv_mass_rel_tol, const double z_flxdiv_mass_abs_tol,
    const double z_flxdiv_theta_rel_tol, const double z_flxdiv_theta_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_41(
    double *geofac_div, double *mass_fl_e, double *z_theta_v_fl_e,
    double *z_flxdiv_mass, double *z_flxdiv_theta, double *z_flxdiv_mass_before,
    double *z_flxdiv_theta_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_flxdiv_mass_rel_tol, const double z_flxdiv_mass_abs_tol,
    const double z_flxdiv_theta_rel_tol, const double z_flxdiv_theta_abs_tol);
void setup_mo_solve_nonhydro_stencil_41(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_flxdiv_mass_k_size,
                                        const int z_flxdiv_theta_k_size);
void free_mo_solve_nonhydro_stencil_41();
}
