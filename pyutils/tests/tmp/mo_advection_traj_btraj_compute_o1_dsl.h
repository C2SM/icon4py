#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_advection_traj_btraj_compute_o1_dsl(
    double *p_vn, double *p_vt, int *cell_idx, int *cell_blk,
    double *pos_on_tplane_e_1, double *pos_on_tplane_e_2,
    double *primal_normal_cell_1, double *dual_normal_cell_1,
    double *primal_normal_cell_2, double *dual_normal_cell_2, int *p_cell_idx,
    int *p_cell_blk, double *p_distv_bary_1, double *p_distv_bary_2,
    double p_dthalf, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_advection_traj_btraj_compute_o1_dsl(
    const int *p_cell_idx_dsl, const int *p_cell_idx, const int *p_cell_blk_dsl,
    const int *p_cell_blk, const double *p_distv_bary_1_dsl,
    const double *p_distv_bary_1, const double *p_distv_bary_2_dsl,
    const double *p_distv_bary_2, const double p_cell_idx_rel_tol,
    const double p_cell_idx_abs_tol, const double p_cell_blk_rel_tol,
    const double p_cell_blk_abs_tol, const double p_distv_bary_1_rel_tol,
    const double p_distv_bary_1_abs_tol, const double p_distv_bary_2_rel_tol,
    const double p_distv_bary_2_abs_tol, const int iteration);
void run_and_verify_mo_advection_traj_btraj_compute_o1_dsl(
    double *p_vn, double *p_vt, int *cell_idx, int *cell_blk,
    double *pos_on_tplane_e_1, double *pos_on_tplane_e_2,
    double *primal_normal_cell_1, double *dual_normal_cell_1,
    double *primal_normal_cell_2, double *dual_normal_cell_2, int *p_cell_idx,
    int *p_cell_blk, double *p_distv_bary_1, double *p_distv_bary_2,
    double p_dthalf, int *p_cell_idx_before, int *p_cell_blk_before,
    double *p_distv_bary_1_before, double *p_distv_bary_2_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double p_cell_idx_rel_tol,
    const double p_cell_idx_abs_tol, const double p_cell_blk_rel_tol,
    const double p_cell_blk_abs_tol, const double p_distv_bary_1_rel_tol,
    const double p_distv_bary_1_abs_tol, const double p_distv_bary_2_rel_tol,
    const double p_distv_bary_2_abs_tol);
void setup_mo_advection_traj_btraj_compute_o1_dsl(
    dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
    const int p_cell_idx_k_size, const int p_cell_blk_k_size,
    const int p_distv_bary_1_k_size, const int p_distv_bary_2_k_size);
void free_mo_advection_traj_btraj_compute_o1_dsl();
}
