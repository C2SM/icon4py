#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_03(double *div, double *kh_c, double *wgtfac_c,
                                    double *div_ic, double *hdef_ic,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_03(
    const double *div_ic_dsl, const double *div_ic, const double *hdef_ic_dsl,
    const double *hdef_ic, const double div_ic_rel_tol,
    const double div_ic_abs_tol, const double hdef_ic_rel_tol,
    const double hdef_ic_abs_tol, const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_03(
    double *div, double *kh_c, double *wgtfac_c, double *div_ic,
    double *hdef_ic, double *div_ic_before, double *hdef_ic_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double div_ic_rel_tol,
    const double div_ic_abs_tol, const double hdef_ic_rel_tol,
    const double hdef_ic_abs_tol);
void setup_mo_nh_diffusion_stencil_03(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int div_ic_k_size,
                                      const int hdef_ic_k_size);
void free_mo_nh_diffusion_stencil_03();
}
