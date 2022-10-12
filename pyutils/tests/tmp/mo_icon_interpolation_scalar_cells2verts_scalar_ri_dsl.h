#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    double *p_cell_in, double *c_intp, double *p_vert_out,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    const double *p_vert_out_dsl, const double *p_vert_out,
    const double p_vert_out_rel_tol, const double p_vert_out_abs_tol,
    const int iteration);
void run_and_verify_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    double *p_cell_in, double *c_intp, double *p_vert_out,
    double *p_vert_out_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double p_vert_out_rel_tol, const double p_vert_out_abs_tol);
void setup_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
    const int p_vert_out_k_size);
void free_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl();
}
