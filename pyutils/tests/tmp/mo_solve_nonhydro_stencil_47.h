#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_47(double *w_nnew, double *z_contr_w_fl_l,
                                      double *w_concorr_c,
                                      const int verticalStart,
                                      const int verticalEnd,
                                      const int horizontalStart,
                                      const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_47(
    const double *w_nnew_dsl, const double *w_nnew,
    const double *z_contr_w_fl_l_dsl, const double *z_contr_w_fl_l,
    const double w_nnew_rel_tol, const double w_nnew_abs_tol,
    const double z_contr_w_fl_l_rel_tol, const double z_contr_w_fl_l_abs_tol,
    const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_47(
    double *w_nnew, double *z_contr_w_fl_l, double *w_concorr_c,
    double *w_nnew_before, double *z_contr_w_fl_l_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double w_nnew_rel_tol,
    const double w_nnew_abs_tol, const double z_contr_w_fl_l_rel_tol,
    const double z_contr_w_fl_l_abs_tol);
void setup_mo_solve_nonhydro_stencil_47(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int w_nnew_k_size,
                                        const int z_contr_w_fl_l_k_size);
void free_mo_solve_nonhydro_stencil_47();
}
