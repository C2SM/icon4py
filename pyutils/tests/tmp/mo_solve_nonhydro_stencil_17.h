#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_17(
    double *hmask_dd3d, double *scalfac_dd3d, double *inv_dual_edge_length,
    double *z_dwdz_dd, double *z_graddiv_vn, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_17(const double *z_graddiv_vn_dsl,
                                         const double *z_graddiv_vn,
                                         const double z_graddiv_vn_rel_tol,
                                         const double z_graddiv_vn_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_17(
    double *hmask_dd3d, double *scalfac_dd3d, double *inv_dual_edge_length,
    double *z_dwdz_dd, double *z_graddiv_vn, double *z_graddiv_vn_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_graddiv_vn_rel_tol,
    const double z_graddiv_vn_abs_tol);
void setup_mo_solve_nonhydro_stencil_17(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_graddiv_vn_k_size);
void free_mo_solve_nonhydro_stencil_17();
}
