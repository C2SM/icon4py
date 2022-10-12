#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_38(double *vn, double *wgtfacq_e,
                                      double *vn_ie, const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_38(const double *vn_ie_dsl,
                                         const double *vn_ie,
                                         const double vn_ie_rel_tol,
                                         const double vn_ie_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_38(
    double *vn, double *wgtfacq_e, double *vn_ie, double *vn_ie_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_ie_rel_tol,
    const double vn_ie_abs_tol);
void setup_mo_solve_nonhydro_stencil_38(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_ie_k_size);
void free_mo_solve_nonhydro_stencil_38();
}
