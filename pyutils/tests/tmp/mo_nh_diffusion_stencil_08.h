#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_08(double *w, double *geofac_grg_x,
                                    double *geofac_grg_y, double *dwdx,
                                    double *dwdy, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_08(
    const double *dwdx_dsl, const double *dwdx, const double *dwdy_dsl,
    const double *dwdy, const double dwdx_rel_tol, const double dwdx_abs_tol,
    const double dwdy_rel_tol, const double dwdy_abs_tol, const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_08(
    double *w, double *geofac_grg_x, double *geofac_grg_y, double *dwdx,
    double *dwdy, double *dwdx_before, double *dwdy_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double dwdx_rel_tol,
    const double dwdx_abs_tol, const double dwdy_rel_tol,
    const double dwdy_abs_tol);
void setup_mo_nh_diffusion_stencil_08(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int dwdx_k_size,
                                      const int dwdy_k_size);
void free_mo_nh_diffusion_stencil_08();
}
