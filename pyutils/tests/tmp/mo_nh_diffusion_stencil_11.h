#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_11(double *theta_v, double *theta_ref_mc,
                                    double *enh_diffu_3d, double thresh_tdiff,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_11(const double *enh_diffu_3d_dsl,
                                       const double *enh_diffu_3d,
                                       const double enh_diffu_3d_rel_tol,
                                       const double enh_diffu_3d_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_11(
    double *theta_v, double *theta_ref_mc, double *enh_diffu_3d,
    double thresh_tdiff, double *enh_diffu_3d_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double enh_diffu_3d_rel_tol, const double enh_diffu_3d_abs_tol);
void setup_mo_nh_diffusion_stencil_11(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int enh_diffu_3d_k_size);
void free_mo_nh_diffusion_stencil_11();
}
