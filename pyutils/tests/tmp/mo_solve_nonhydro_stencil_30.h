#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_30(
    double *e_flx_avg, double *vn, double *geofac_grdiv,
    double *rbf_vec_coeff_e, double *z_vn_avg, double *z_graddiv_vn, double *vt,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_30(
    const double *z_vn_avg_dsl, const double *z_vn_avg,
    const double *z_graddiv_vn_dsl, const double *z_graddiv_vn,
    const double *vt_dsl, const double *vt, const double z_vn_avg_rel_tol,
    const double z_vn_avg_abs_tol, const double z_graddiv_vn_rel_tol,
    const double z_graddiv_vn_abs_tol, const double vt_rel_tol,
    const double vt_abs_tol, const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_30(
    double *e_flx_avg, double *vn, double *geofac_grdiv,
    double *rbf_vec_coeff_e, double *z_vn_avg, double *z_graddiv_vn, double *vt,
    double *z_vn_avg_before, double *z_graddiv_vn_before, double *vt_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_vn_avg_rel_tol,
    const double z_vn_avg_abs_tol, const double z_graddiv_vn_rel_tol,
    const double z_graddiv_vn_abs_tol, const double vt_rel_tol,
    const double vt_abs_tol);
void setup_mo_solve_nonhydro_stencil_30(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_vn_avg_k_size,
                                        const int z_graddiv_vn_k_size,
                                        const int vt_k_size);
void free_mo_solve_nonhydro_stencil_30();
}
