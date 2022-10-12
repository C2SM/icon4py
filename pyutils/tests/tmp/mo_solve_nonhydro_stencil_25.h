#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_25(
    double *geofac_grdiv, double *z_graddiv_vn, double *z_graddiv2_vn,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_25(const double *z_graddiv2_vn_dsl,
                                         const double *z_graddiv2_vn,
                                         const double z_graddiv2_vn_rel_tol,
                                         const double z_graddiv2_vn_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_25(
    double *geofac_grdiv, double *z_graddiv_vn, double *z_graddiv2_vn,
    double *z_graddiv2_vn_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_graddiv2_vn_rel_tol, const double z_graddiv2_vn_abs_tol);
void setup_mo_solve_nonhydro_stencil_25(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_graddiv2_vn_k_size);
void free_mo_solve_nonhydro_stencil_25();
}
