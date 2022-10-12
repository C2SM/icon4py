#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_velocity_advection_stencil_20(
    int *levelmask, double *c_lin_e, double *z_w_con_c_full,
    double *ddqz_z_full_e, double *area_edge, double *tangent_orientation,
    double *inv_primal_edge_length, double *zeta, double *geofac_grdiv,
    double *vn, double *ddt_vn_adv, double cfl_w_limit, double scalfac_exdiff,
    double d_time, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_velocity_advection_stencil_20(const double *ddt_vn_adv_dsl,
                                             const double *ddt_vn_adv,
                                             const double ddt_vn_adv_rel_tol,
                                             const double ddt_vn_adv_abs_tol,
                                             const int iteration);
void run_and_verify_mo_velocity_advection_stencil_20(
    int *levelmask, double *c_lin_e, double *z_w_con_c_full,
    double *ddqz_z_full_e, double *area_edge, double *tangent_orientation,
    double *inv_primal_edge_length, double *zeta, double *geofac_grdiv,
    double *vn, double *ddt_vn_adv, double cfl_w_limit, double scalfac_exdiff,
    double d_time, double *ddt_vn_adv_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double ddt_vn_adv_rel_tol, const double ddt_vn_adv_abs_tol);
void setup_mo_velocity_advection_stencil_20(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int ddt_vn_adv_k_size);
void free_mo_velocity_advection_stencil_20();
}
