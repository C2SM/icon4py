#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_12(double *kh_smag_e, double *enh_diffu_3d,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_12(const double *kh_smag_e_dsl,
                                       const double *kh_smag_e,
                                       const double kh_smag_e_rel_tol,
                                       const double kh_smag_e_abs_tol,
                                       const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_12(
    double *kh_smag_e, double *enh_diffu_3d, double *kh_smag_e_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double kh_smag_e_rel_tol,
    const double kh_smag_e_abs_tol);
void setup_mo_nh_diffusion_stencil_12(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int kh_smag_e_k_size);
void free_mo_nh_diffusion_stencil_12();
}
