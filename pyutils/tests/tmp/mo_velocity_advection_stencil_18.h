#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_18(
    int *levelmask, int *cfl_clipping, int *owner_mask, double *z_w_con_c,
    double *ddqz_z_half, double *area, double *geofac_n2s, double *w,
    double *ddt_w_adv, double scalfac_exdiff, double cfl_w_limit, double dtime,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_18(const double *ddt_w_adv_dsl,
                                             const double *ddt_w_adv,
                                             const double ddt_w_adv_rel_tol,
                                             const double ddt_w_adv_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_18(
    int *levelmask, int *cfl_clipping, int *owner_mask, double *z_w_con_c,
    double *ddqz_z_half, double *area, double *geofac_n2s, double *w,
    double *ddt_w_adv, double scalfac_exdiff, double cfl_w_limit, double dtime,
    double *ddt_w_adv_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double ddt_w_adv_rel_tol, const double ddt_w_adv_abs_tol);
void setup_mo_velocity_advection_stencil_18(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int ddt_w_adv_k_size);
void free_mo_velocity_advection_stencil_18();
}
