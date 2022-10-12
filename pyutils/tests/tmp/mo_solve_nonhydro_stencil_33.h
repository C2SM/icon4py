#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_33(double *vn_traj, double *mass_flx_me,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_33(
    const double *vn_traj_dsl, const double *vn_traj,
    const double *mass_flx_me_dsl, const double *mass_flx_me,
    const double vn_traj_rel_tol, const double vn_traj_abs_tol,
    const double mass_flx_me_rel_tol, const double mass_flx_me_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_33(
    double *vn_traj, double *mass_flx_me, double *vn_traj_before,
    double *mass_flx_me_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double vn_traj_rel_tol, const double vn_traj_abs_tol,
    const double mass_flx_me_rel_tol, const double mass_flx_me_abs_tol);
void setup_mo_solve_nonhydro_stencil_33(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_traj_k_size,
                                        const int mass_flx_me_k_size);
void free_mo_solve_nonhydro_stencil_33();
}
