#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_02(double *kh_smag_ec, double *vn,
                                    double *e_bln_c_s, double *geofac_div,
                                    double *diff_multfac_smag, double *kh_c,
                                    double *div, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_02(
    const double *kh_c_dsl, const double *kh_c, const double *div_dsl,
    const double *div, const double kh_c_rel_tol, const double kh_c_abs_tol,
    const double div_rel_tol, const double div_abs_tol, const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_02(
    double *kh_smag_ec, double *vn, double *e_bln_c_s, double *geofac_div,
    double *diff_multfac_smag, double *kh_c, double *div, double *kh_c_before,
    double *div_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double kh_c_rel_tol, const double kh_c_abs_tol,
    const double div_rel_tol, const double div_abs_tol);
void setup_mo_nh_diffusion_stencil_02(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int kh_c_k_size,
                                      const int div_k_size);
void free_mo_nh_diffusion_stencil_02();
}
