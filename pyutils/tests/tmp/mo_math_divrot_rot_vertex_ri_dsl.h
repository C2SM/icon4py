#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_math_divrot_rot_vertex_ri_dsl(
    double *vec_e, double *geofac_rot, double *rot_vec, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_math_divrot_rot_vertex_ri_dsl(const double *rot_vec_dsl,
                                             const double *rot_vec,
                                             const double rot_vec_rel_tol,
                                             const double rot_vec_abs_tol,
                                             const int iteration);
void run_and_verify_mo_math_divrot_rot_vertex_ri_dsl(
    double *vec_e, double *geofac_rot, double *rot_vec, double *rot_vec_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double rot_vec_rel_tol,
    const double rot_vec_abs_tol);
void setup_mo_math_divrot_rot_vertex_ri_dsl(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int rot_vec_k_size);
void free_mo_math_divrot_rot_vertex_ri_dsl();
}
