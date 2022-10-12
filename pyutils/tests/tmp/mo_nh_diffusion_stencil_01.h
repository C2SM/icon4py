#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_nh_diffusion_stencil_01(
    double *diff_multfac_smag, double *tangent_orientation,
    double *inv_primal_edge_length, double *inv_vert_vert_length,
    double *u_vert, double *v_vert, double *primal_normal_vert_x,
    double *primal_normal_vert_y, double *dual_normal_vert_x,
    double *dual_normal_vert_y, double *vn, double *smag_limit,
    double *kh_smag_e, double *kh_smag_ec, double *z_nabla2_e,
    double smag_offset, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_nh_diffusion_stencil_01(
    const double *kh_smag_e_dsl, const double *kh_smag_e,
    const double *kh_smag_ec_dsl, const double *kh_smag_ec,
    const double *z_nabla2_e_dsl, const double *z_nabla2_e,
    const double kh_smag_e_rel_tol, const double kh_smag_e_abs_tol,
    const double kh_smag_ec_rel_tol, const double kh_smag_ec_abs_tol,
    const double z_nabla2_e_rel_tol, const double z_nabla2_e_abs_tol,
    const int iteration);
void run_and_verify_mo_nh_diffusion_stencil_01(
    double *diff_multfac_smag, double *tangent_orientation,
    double *inv_primal_edge_length, double *inv_vert_vert_length,
    double *u_vert, double *v_vert, double *primal_normal_vert_x,
    double *primal_normal_vert_y, double *dual_normal_vert_x,
    double *dual_normal_vert_y, double *vn, double *smag_limit,
    double *kh_smag_e, double *kh_smag_ec, double *z_nabla2_e,
    double smag_offset, double *kh_smag_e_before, double *kh_smag_ec_before,
    double *z_nabla2_e_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double kh_smag_e_rel_tol, const double kh_smag_e_abs_tol,
    const double kh_smag_ec_rel_tol, const double kh_smag_ec_abs_tol,
    const double z_nabla2_e_rel_tol, const double z_nabla2_e_abs_tol);
void setup_mo_nh_diffusion_stencil_01(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int kh_smag_e_k_size,
                                      const int kh_smag_ec_k_size,
                                      const int z_nabla2_e_k_size);
void free_mo_nh_diffusion_stencil_01();
}
