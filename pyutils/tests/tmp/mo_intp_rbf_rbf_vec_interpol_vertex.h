#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_intp_rbf_rbf_vec_interpol_vertex(
    double *p_e_in, double *ptr_coeff_1, double *ptr_coeff_2, double *p_u_out,
    double *p_v_out, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_intp_rbf_rbf_vec_interpol_vertex(
    const double *p_u_out_dsl, const double *p_u_out, const double *p_v_out_dsl,
    const double *p_v_out, const double p_u_out_rel_tol,
    const double p_u_out_abs_tol, const double p_v_out_rel_tol,
    const double p_v_out_abs_tol, const int iteration);
void run_and_verify_mo_intp_rbf_rbf_vec_interpol_vertex(
    double *p_e_in, double *ptr_coeff_1, double *ptr_coeff_2, double *p_u_out,
    double *p_v_out, double *p_u_out_before, double *p_v_out_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double p_u_out_rel_tol,
    const double p_u_out_abs_tol, const double p_v_out_rel_tol,
    const double p_v_out_abs_tol);
void setup_mo_intp_rbf_rbf_vec_interpol_vertex(dawn::GlobalGpuTriMesh *mesh,
                                               int k_size, cudaStream_t stream,
                                               const int p_u_out_k_size,
                                               const int p_v_out_k_size);
void free_mo_intp_rbf_rbf_vec_interpol_vertex();
}
