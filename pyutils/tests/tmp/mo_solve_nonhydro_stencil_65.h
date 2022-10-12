#pragma once
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
extern "C" {
void run_mo_solve_nonhydro_stencil_65(
    double *rho_ic, double *vwind_expl_wgt, double *vwind_impl_wgt,
    double *w_now, double *w_new, double *w_concorr_c, double *mass_flx_ic,
    double r_nsubsteps, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd);
bool verify_mo_solve_nonhydro_stencil_65(const double *mass_flx_ic_dsl,
                                         const double *mass_flx_ic,
                                         const double mass_flx_ic_rel_tol,
                                         const double mass_flx_ic_abs_tol,
                                         const int iteration);
void run_and_verify_mo_solve_nonhydro_stencil_65(
    double *rho_ic, double *vwind_expl_wgt, double *vwind_impl_wgt,
    double *w_now, double *w_new, double *w_concorr_c, double *mass_flx_ic,
    double r_nsubsteps, double *mass_flx_ic_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double mass_flx_ic_rel_tol, const double mass_flx_ic_abs_tol);
void setup_mo_solve_nonhydro_stencil_65(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int mass_flx_ic_k_size);
void free_mo_solve_nonhydro_stencil_65();
}
