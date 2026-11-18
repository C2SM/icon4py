# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: SIM108, PLR0912 [if-else-block-instead-of-if-exp, too-many-branches] literal Fortran transcription

"""
Literal scalar transcription of the ICON Fortran AES bubble initial condition.

It serves as test reference for the vectorized implementation in
icon4py.model.common.initial_condition.analytical and was written from the Fortran only.

Sources (icon-exclaim/icon/src):
  testcases/mo_aes_bubble.f90            : init_aes_bubble, gaussian
  atm_dyn_iconam/mo_hydro_adjust.f90      : hydro_adjust_iterative (bottom-up branch only)
  configure_model/mo_aes_bubble_config.f90: defaults (see AES_BUBBLE_DEFAULTS)
  atm_phy_aes/mo_aes_thermo.f90           : sat_pres_water, specific_humidity
  shared/mo_physical_constants.f90 (+ physics_granules/common/mo_granules_phys.f90)

Layout: full-level arrays (ncells, nlev), half-level arrays (ncells, nlev+1).
Python level index k = Fortran jk - 1 (k = 0 is model top, like jk = 1).
Plain Python loops on purpose (no vectorisation), statement and expression order as in Fortran.
"""

import math

import numpy as np

from icon4py.model.common import constants as phy_const


# ---------------------------------------------------------------------------------------------
# Constants (Fortran name -> icon4py name)
# ---------------------------------------------------------------------------------------------
# mo_granules_phys / mo_physical_constants:
#   grav  = 9.80665_wp      -> phy_const.GRAV
#   rd    = 287.04_wp       -> phy_const.RD
#   rv    = 461.51_wp       -> phy_const.RV
#   cpd   = 1004.64_wp      -> phy_const.CPD
#   tmelt = 273.15_wp       -> phy_const.MELTING_TEMPERATURE
#   rdv   = rd/rv           -> phy_const.RD_O_RV
#   o_m_rdv = 1 - rd/rv     -> 1.0 - phy_const.RD_O_RV
#   vtmpc1 = rv/rd - 1      -> phy_const.RV_O_RD_MINUS_1
#   rd_o_cpd = rd/cpd       -> phy_const.RD_O_CPD
#   cvd_o_rd = cvd/rd       -> phy_const.CVD_O_RD
#   cpd/rd (literal in hydro_adjust, folded by compiler) -> phy_const.CPD_O_RD
#   p0ref = 100000._wp      -> phy_const.P0REF
# mo_aes_thermo.f90 local PARAMETERs (identical to mo_lookup_tables_constants via mo_granules_phys):
#   c1es  = 610.78_wp       -> phy_const.TETENS_P0
#   c3les = 17.269_wp       -> phy_const.TETENS_A_WATER
#   c4les = 35.86_wp        -> phy_const.TETENS_B_WATER
GRAV = phy_const.GRAV
RD = phy_const.RD
RV = phy_const.RV
P0REF = phy_const.P0REF
RD_O_CPD = phy_const.RD_O_CPD
CPD_O_RD = phy_const.CPD_O_RD
CVD_O_RD = phy_const.CVD_O_RD
VTMPC1 = phy_const.RV_O_RD_MINUS_1
RDV = phy_const.RD_O_RV
O_M_RDV = 1.0 - RD / RV  # mo_physical_constants: o_m_rdv = 1._wp-rd/rv

C1ES = 610.78
C3LES = 17.269
C4LES = 35.86
TMELT = 273.15
assert C1ES == phy_const.TETENS_P0
assert C3LES == phy_const.TETENS_A_WATER
assert C4LES == phy_const.TETENS_B_WATER
assert TMELT == phy_const.MELTING_TEMPERATURE

# mo_aes_bubble_config.f90: init_aes_bubble_config defaults
AES_BUBBLE_DEFAULTS = dict(
    psfc=101325.0,
    t_am=180.0,
    t0=303.15,
    gamma0=0.009,
    z0=3000.0,
    gamma1=0.00001,
    t_perturb=3.0,
    relhum_bg=0.7,
    relhum_mx=0.95,
    hw_x=12500.0,
    hw_z=500.0,
    x_center=0.0,  # OFFSET added to geometry centre x; NOT applied to y in the Fortran
    lgaussxy=False,
)


# ---------------------------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------------------------
def sat_pres_water(tk: float) -> float:
    """mo_aes_thermo.f90: c1es*EXP( c3les*(TK-tmelt)/(TK-c4les) ).

    NB: hydro_adjust_iterative actually uses mo_thdyn_functions::sat_pres_water, which with
    ipsat=1 (hard-coded PARAMETER) is b1*EXP(b2w*(temp-b3)/(temp-b4w)) with b1=c1es, b2w=c3les,
    b3=tmelt, b4w=c4les -> the same formula and the same constants.
    """
    return C1ES * math.exp(C3LES * (tk - TMELT) / (tk - C4LES))


def specific_humidity(pvapor: float, ptotal: float) -> float:
    """mo_aes_thermo.f90: rdv*pvapor/( ptotal - o_m_rdv*pvapor ), local rdv=rd/rv, o_m_rdv=1-rdv."""
    rdv = RD / RV
    o_m_rdv = 1.0 - rdv
    return rdv * pvapor / (ptotal - o_m_rdv * pvapor)


def gaussian(mu: float, sigma: float, delta: float, x: float) -> float:
    """mo_aes_bubble.f90: xx = (x-mu)/sigma; EXP(-.5*xx*xx)*delta."""
    xx = (x - mu) / sigma
    return math.exp(-0.5 * xx * xx) * delta


# ---------------------------------------------------------------------------------------------
# init_aes_bubble, everything before CALL hydro_adjust_iterative
# ---------------------------------------------------------------------------------------------
def reference_first_guess(
    *,
    z_mc,
    cell_x,
    cell_y,
    psfc,
    t_am,
    t0,
    gamma0,
    z0,
    gamma1,
    t_perturb,
    relhum_bg,
    relhum_mx,
    hw_x,
    hw_z,
    x_center,
    y_center,
    lgaussxy,
):
    """x_center / y_center are ABSOLUTE centres
    (Fortran: x_center = geometry%center%x(1) + config%x_center, y_center = geometry%center%x(2))."""
    z_mc = np.asarray(z_mc, dtype=np.float64)
    cell_x = np.asarray(cell_x, dtype=np.float64)
    cell_y = np.asarray(cell_y, dtype=np.float64)
    ncells, nlev = z_mc.shape

    temp = np.zeros((ncells, nlev), dtype=np.float64)
    rh = np.zeros((ncells, nlev), dtype=np.float64)
    exner = np.zeros((ncells, nlev), dtype=np.float64)
    theta_v = np.zeros((ncells, nlev), dtype=np.float64)
    rho = np.zeros((ncells, nlev), dtype=np.float64)
    qv_arr = np.zeros((ncells, nlev), dtype=np.float64)
    pres_arr = np.zeros((ncells, nlev), dtype=np.float64)  # ptr_nh_diag%pres (not returned)
    zeta_xy = np.zeros(ncells, dtype=np.float64)

    # Parameters for horizontal Gaussian profile
    sigma_x = hw_x / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sigma_z = hw_z / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    if lgaussxy:
        for jl in range(ncells):
            x = float(cell_x[jl])
            y = float(cell_y[jl])
            # NB: same sigma_x in y; y_center has no config offset
            zeta_xy[jl] = gaussian(x_center, sigma_x, 1.0, x) * gaussian(y_center, sigma_x, 1.0, y)
    else:
        for jl in range(ncells):
            x = float(cell_x[jl])
            zeta_xy[jl] = gaussian(x_center, sigma_x, 1.0, x)

    tt = np.zeros(ncells, dtype=np.float64)
    pres = np.zeros(ncells, dtype=np.float64)
    for jl in range(ncells):
        tt[jl] = t0 + t_perturb * zeta_xy[jl]
        pres[jl] = psfc

    # DO jk = nlev, 1, -1  (bottom to top)
    for k in range(nlev - 1, -1, -1):
        for jl in range(ncells):
            z = float(z_mc[jl, k])
            if k == nlev - 1:
                dz = z  # integration from z=0 (no topography offset)
            else:
                dz = z - float(z_mc[jl, k + 1])
            zeta = zeta_xy[jl] * gaussian(0.0, sigma_z, 1.0, z)
            if z <= z0:
                t = t0 - gamma0 * z
            else:
                t = t0 - gamma0 * z0 - gamma1 * (z - z0)
            t = max(t, t_am) + t_perturb * zeta  # MAX before adding perturbation
            temp[jl, k] = t
            sat_pres = sat_pres_water(temp[jl, k])
            wat_pres = sat_pres * ((relhum_mx - relhum_bg) * zeta + relhum_bg)
            # layer-mean temperature (dry, not virtual) for the hydrostatic step
            tt[jl] = 0.5 * (tt[jl] + temp[jl, k])
            # Fortran: pres*EXP(-grav/rd/tt*dz); unary minus binds weaker than * and /
            pres[jl] = pres[jl] * math.exp(-(GRAV / RD / tt[jl] * dz))
            tt[jl] = temp[jl, k]
            tv = tt[jl] / (1.0 - (wat_pres / pres[jl]) * O_M_RDV)
            qv = specific_humidity(wat_pres, pres[jl])
            rh[jl, k] = wat_pres / sat_pres
            exner[jl, k] = (pres[jl] / P0REF) ** RD_O_CPD
            pres_arr[jl, k] = pres[jl]
            rho[jl, k] = pres[jl] / RD / tv
            theta_v[jl, k] = tv / exner[jl, k]
            qv_arr[jl, k] = qv

    return {
        "temp": temp,
        "rh": rh,
        "exner": exner,
        "theta_v": theta_v,
        "rho": rho,
        "qv": qv_arr,
    }


# ---------------------------------------------------------------------------------------------
# hydro_adjust_iterative, bottom-up branch (opt_exner_lbc = exner(:,nlev), luse_exner_fg=.TRUE.)
# ---------------------------------------------------------------------------------------------
def reference_hydro_adjust_iterative(
    *,
    temp,
    rh,
    exner,
    theta_v,
    rho,
    qv,
    wgtfac_c,
    ddqz_z_half,
    exner_ref_mc,
    d_exner_dz_ref_ic,
    theta_ref_mc,
    max_iteration=25,
):
    """Returns NEW arrays {'exner','theta_v','rho','qv'}; inputs are not mutated.

    Input qv is ignored (first guess qv_new = 0); input theta_v / rho only enter the
    convergence diagnostics (not reproduced here; they do not affect the result).
    """
    temp = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    ncells, nlev = temp.shape

    exner_out = np.array(exner, dtype=np.float64, copy=True)
    theta_v_out = np.array(theta_v, dtype=np.float64, copy=True)
    rho_out = np.array(rho, dtype=np.float64, copy=True)
    qv_out = np.array(qv, dtype=np.float64, copy=True)

    exner_lbc = np.array(exner_out[:, nlev - 1], copy=True)  # opt_exner_lbc = exner(:,nlev,:)
    ptr_exner_fg = exner_out  # luse_exner_fg = .TRUE. -> pointer to exner itself

    exner_new = np.zeros((ncells, nlev), dtype=np.float64)
    theta_v_new = np.zeros((ncells, nlev), dtype=np.float64)
    rho_new = np.zeros((ncells, nlev), dtype=np.float64)
    qv_new = np.zeros((ncells, nlev), dtype=np.float64)
    z_theta_v_pr = np.zeros((ncells, nlev), dtype=np.float64)
    theta_v_ic = np.zeros((ncells, nlev + 1), dtype=np.float64)
    theta_v_pr_ic = np.zeros((ncells, nlev + 1), dtype=np.float64)

    # first guess for qv
    for k in range(nlev):
        for jc in range(ncells):
            qv_new[jc, k] = 0.0
    # lower boundary condition
    for jc in range(ncells):
        exner_new[jc, nlev - 1] = exner_lbc[jc]
    # first guess  (DO jk = 1, nlev-1)
    for k in range(nlev - 1):
        for jc in range(ncells):
            exner_new[jc, k] = ptr_exner_fg[jc, k]

    for _iter in range(max_iteration):
        # (1)+(2) virtual temperature, theta_v, perturbation  (uses qv_new and exner_new of
        # the PREVIOUS iteration)
        for k in range(nlev):
            for jc in range(ncells):
                tempv = temp[jc, k] * (1.0 + VTMPC1 * qv_new[jc, k])
                theta_v_new[jc, k] = tempv / exner_new[jc, k]
                z_theta_v_pr[jc, k] = theta_v_new[jc, k] - theta_ref_mc[jc, k]

        # cell -> face interpolation, Fortran half levels jk = 2..nlev -> python 1..nlev-1
        for kh in range(1, nlev):
            for jc in range(ncells):
                theta_v_ic[jc, kh] = (
                    wgtfac_c[jc, kh] * theta_v_new[jc, kh]
                    + (1.0 - wgtfac_c[jc, kh]) * theta_v_new[jc, kh - 1]
                )
                theta_v_pr_ic[jc, kh] = (
                    wgtfac_c[jc, kh] * z_theta_v_pr[jc, kh]
                    + (1.0 - wgtfac_c[jc, kh]) * z_theta_v_pr[jc, kh - 1]
                )

        # (3) bottom-up: DO jk = nlev-1,1,-1 ; half level jk+1 -> python index k+1 (= jk)
        for k in range(nlev - 2, -1, -1):
            for jc in range(ncells):
                exner_new[jc, k] = (
                    exner_new[jc, k + 1]
                    + (exner_ref_mc[jc, k] - exner_ref_mc[jc, k + 1])
                    - ddqz_z_half[jc, k + 1]
                    / theta_v_ic[jc, k + 1]
                    * theta_v_pr_ic[jc, k + 1]
                    * d_exner_dz_ref_ic[jc, k + 1]
                )

        # (4) qv from UPDATED exner
        for k in range(nlev):
            for jc in range(ncells):
                es = sat_pres_water(temp[jc, k])
                pres = P0REF * math.exp(CPD_O_RD * math.log(exner_new[jc, k]))
                qv_new[jc, k] = RDV * rh[jc, k] * es / (pres - O_M_RDV * rh[jc, k] * es)

        # (5) rho from UPDATED exner but theta_v_new of step (2) (old exner, old qv)
        for k in range(nlev):
            for jc in range(ncells):
                rho_new[jc, k] = exner_new[jc, k] ** CVD_O_RD * P0REF / (RD * theta_v_new[jc, k])

        # copy back
        for k in range(nlev):
            for jc in range(ncells):
                exner_out[jc, k] = exner_new[jc, k]
                theta_v_out[jc, k] = theta_v_new[jc, k]
                rho_out[jc, k] = rho_new[jc, k]
                qv_out[jc, k] = qv_new[jc, k]

    return {"exner": exner_out, "theta_v": theta_v_out, "rho": rho_out, "qv": qv_out}
