#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_39(double *e_bln_c_s, double *z_w_concorr_me,
                                      double *wgtfac_c, double *w_concorr_c,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_39(const double *w_concorr_c_dsl,
                                         const double *w_concorr_c,
                                         const double w_concorr_c_rel_tol,
                                         const double w_concorr_c_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_39(
    double *e_bln_c_s, double *z_w_concorr_me, double *wgtfac_c,
    double *w_concorr_c, double *w_concorr_c_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double w_concorr_c_rel_tol, const double w_concorr_c_abs_tol);
void setup_mo_solve_nonhydro_stencil_39(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int w_concorr_c_k_size);
void free_mo_solve_nonhydro_stencil_39();
}
