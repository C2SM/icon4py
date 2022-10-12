#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_math_gradients_grad_green_gauss_cell_dsl(
    double *p_grad_1_u, double *p_grad_1_v, double *p_grad_2_u,
    double *p_grad_2_v, double *p_ccpr1, double *p_ccpr2, double *geofac_grg_x,
    double *geofac_grg_y, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_math_gradients_grad_green_gauss_cell_dsl(
    const double *p_grad_1_u_dsl, const double *p_grad_1_u,
    const double *p_grad_1_v_dsl, const double *p_grad_1_v,
    const double *p_grad_2_u_dsl, const double *p_grad_2_u,
    const double *p_grad_2_v_dsl, const double *p_grad_2_v,
    const double p_grad_1_u_rel_tol, const double p_grad_1_u_abs_tol,
    const double p_grad_1_v_rel_tol, const double p_grad_1_v_abs_tol,
    const double p_grad_2_u_rel_tol, const double p_grad_2_u_abs_tol,
    const double p_grad_2_v_rel_tol, const double p_grad_2_v_abs_tol,
    const int iteration);
void run_and_verify_mo_math_gradients_grad_green_gauss_cell_dsl(
    double *p_grad_1_u, double *p_grad_1_v, double *p_grad_2_u,
    double *p_grad_2_v, double *p_ccpr1, double *p_ccpr2, double *geofac_grg_x,
    double *geofac_grg_y, double *p_grad_1_u_before, double *p_grad_1_v_before,
    double *p_grad_2_u_before, double *p_grad_2_v_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double p_grad_1_u_rel_tol,
    const double p_grad_1_u_abs_tol, const double p_grad_1_v_rel_tol,
    const double p_grad_1_v_abs_tol, const double p_grad_2_u_rel_tol,
    const double p_grad_2_u_abs_tol, const double p_grad_2_v_rel_tol,
    const double p_grad_2_v_abs_tol);
void setup_mo_math_gradients_grad_green_gauss_cell_dsl(
    dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
    const int p_grad_1_u_k_size, const int p_grad_1_v_k_size,
    const int p_grad_2_u_k_size, const int p_grad_2_v_k_size);
void free_mo_math_gradients_grad_green_gauss_cell_dsl();
}
