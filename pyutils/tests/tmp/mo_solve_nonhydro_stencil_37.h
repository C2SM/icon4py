#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_37(double *vn, double *vt, double *vn_ie,
                                      double *z_vt_ie, double *z_kin_hor_e,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_37(
    const double *vn_ie_dsl, const double *vn_ie, const double *z_vt_ie_dsl,
    const double *z_vt_ie, const double *z_kin_hor_e_dsl,
    const double *z_kin_hor_e, const double vn_ie_rel_tol,
    const double vn_ie_abs_tol, const double z_vt_ie_rel_tol,
    const double z_vt_ie_abs_tol, const double z_kin_hor_e_rel_tol,
    const double z_kin_hor_e_abs_tol, const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_37(
    double *vn, double *vt, double *vn_ie, double *z_vt_ie, double *z_kin_hor_e,
    double *vn_ie_before, double *z_vt_ie_before, double *z_kin_hor_e_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_ie_rel_tol,
    const double vn_ie_abs_tol, const double z_vt_ie_rel_tol,
    const double z_vt_ie_abs_tol, const double z_kin_hor_e_rel_tol,
    const double z_kin_hor_e_abs_tol);
void setup_mo_solve_nonhydro_stencil_37(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_ie_k_size,
                                        const int z_vt_ie_k_size,
                                        const int z_kin_hor_e_k_size);
void free_mo_solve_nonhydro_stencil_37();
}
