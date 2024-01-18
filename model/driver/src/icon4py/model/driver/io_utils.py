# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np


from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    PrepAdvection,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.decomposition.definitions import DecompositionInfo, ProcessProperties
from icon4py.model.common.decomposition.mpi_decomposition import ParallelLogger
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim, KDim, V2C2VDim, E2C2VDim, E2CDim, CEDim, C2E2C2EDim, C2E2CDim, C2EDim, C2E
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.states.diagnostic_state import DiagnosticState, DiagnosticMetricState
from icon4py.model.driver.serialbox_helpers import (
    construct_diagnostics_for_diffusion,
    construct_interpolation_state_for_diffusion,
    construct_metric_state_for_diffusion,
)
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.common.constants import GRAV, RD, EARTH_RADIUS, EARTH_ANGULAR_VELOCITY, MATH_PI, MATH_PI_2, RD_O_CPD, CPD_O_RD, P0REF, CVD_O_RD, GRAV_O_RD
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field
from icon4py.model.common.interpolation.stencils.mo_rbf_vec_interpol_cell import mo_rbf_vec_interpol_cell

from gt4py.next import as_field
from gt4py.next.common import Field

import sys
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached, run_gtfn_gpu

compiler_backend = run_gtfn
compiler_cached_backend = run_gtfn_cached
compiler_gpu_backend = run_gtfn_gpu
backend = compiler_cached_backend


SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."
INITIALIZATION_ERROR_MSG = "Only SB and JABW are implemented for model initialization."

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, Enum):
    SB = "serialbox"
    NC = "netcdf"

class InitializationType(str, Enum):
    SB = "serialbox"
    JABW = "jabw"


def mo_rbf_vec_interpol_cell_numpy(
    p_e_in: np.array,
    ptr_coeff_1: np.array,
    ptr_coeff_2: np.array,
    c2e2c2e: np.array,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> tuple[np.array, np.array]:
    expanded_ptr_coeff_1 = np.expand_dims(ptr_coeff_1[0:horizontal_end,:], axis=-1)
    expanded_ptr_coeff_2 = np.expand_dims(ptr_coeff_2[0:horizontal_end,:], axis=-1)
    mask = np.ones(c2e2c2e.shape[0], dtype=bool)
    mask[horizontal_end:] = False
    mask[0:horizontal_start] = False
    mask = np.repeat(np.expand_dims(mask, axis=-1), p_e_in.shape[1], axis=1)
    mask[:, vertical_end:] = False
    mask[:, 0:vertical_start] = False
    #print("debug:", horizontal_start, ' - ', horizontal_end, ' - ', vertical_start, ' - ', vertical_end)
    #print("debug:", ptr_coeff_1.shape, ' - ', ptr_coeff_2.shape)
    #print("debug:", expanded_ptr_coeff_1.shape, ' - ', expanded_ptr_coeff_2.shape)
    #print("debug:", p_e_in.shape, ' - ', mask.shape)
    #print("debug:", p_e_in[c2e2c2e].shape)
    p_u_out = np.where(mask, np.sum(p_e_in[c2e2c2e] * expanded_ptr_coeff_1, axis=1), 0.0)
    p_v_out = np.where(mask, np.sum(p_e_in[c2e2c2e] * expanded_ptr_coeff_2, axis=1), 0.0)
    return p_u_out, p_v_out

def mo_cells2edges_scalar_numpy(
    grid: IconGrid,
    cells2edges_interpolation_coeff: np.array,
    cell_scalar: np.array,
    mask: np.array,
):
    e2c = grid.connectivities[E2CDim]
    cells2edges_interpolation_coeff = np.expand_dims(cells2edges_interpolation_coeff, axis=-1)
    mask = np.repeat(np.expand_dims(mask, axis=-1), cell_scalar.shape[1], axis=1)
    #print()
    #print("reference: ", cells2edges_interpolation_coeff.shape)
    #print("reference: ", cell_scalar.shape, ' - ', mask.shape)
    #print("reference: ", cell_scalar[e2c].shape)
    edge_scalar = np.where(mask, np.sum(cell_scalar[e2c] * cells2edges_interpolation_coeff, axis=1), 0.0)
    return edge_scalar

def mo_u2vn_jabw_numpy(
    jw_u0: float,
    jw_up: float,
    latC: float,
    lonC: float,
    edge_lat: np.array,
    edge_lon: np.array,
    primal_normal_x: np.array,
    eta_v_e: np.array,
    mask: np.array,
):
    mask = np.repeat(np.expand_dims(mask, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lat = np.repeat(np.expand_dims(edge_lat, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lon = np.repeat(np.expand_dims(edge_lon, axis=-1), eta_v_e.shape[1], axis=1)
    primal_normal_x = np.repeat(np.expand_dims(primal_normal_x, axis=-1), eta_v_e.shape[1], axis=1)
    u = np.where(mask, jw_u0 * (np.cos(eta_v_e) ** 1.5) * (np.sin(2.0 * edge_lat) ** 2), 0.0)
    if (jw_up > 1.e-20):
        u = np.where(
            mask,
            u + jw_up * np.exp(-10.0 * np.arccos(np.sin(latC) * np.sin(edge_lat) + np.cos(latC) * np.cos(edge_lat) * np.cos(edge_lon - lonC)) ** 2),
            u
        )
    vn = u * primal_normal_x

    return vn


def mo_hydro_adjust(
    wgtfac_c: np.array,
    ddqz_z_half: np.array,
    exner_ref_mc: np.array,
    d_exner_dz_ref_ic: np.array,
    theta_ref_mc: np.array,
    theta_ref_ic: np.array,
    rho: np.array,
    exner: np.array,
    theta_v: np.array,
    num_levels: int,
):
    # virtual temperature
    temp_v = theta_v * exner

    for k in range(num_levels-2,-1,-1):
        fac1 = wgtfac_c[:,k+1] * (temp_v[:,k+1] - theta_ref_mc[:,k+1] * exner[:,k+1]) - (1.0 - wgtfac_c[:,k+1]) * theta_ref_mc[:,k] * exner[:,k+1]
        fac2 = (1.0 - wgtfac_c[:,k+1]) * temp_v[:,k] * exner[:,k+1]
        fac3 = exner_ref_mc[:,k+1] - exner_ref_mc[:,k] - exner[:,k+1]

        quadratic_a = (theta_ref_ic[:,k+1] * exner[:,k+1] + fac1)/ddqz_z_half[:,k+1]
        quadratic_b = -(quadratic_a * fac3 + fac2 / ddqz_z_half[:,k+1] + fac1 * d_exner_dz_ref_ic[:,k+1])
        quadratic_c = -(fac2 * fac3 / ddqz_z_half[:,k+1] + fac2 * d_exner_dz_ref_ic[:,k+1])

        exner[:,k] = (quadratic_b + np.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)) / (2.0 * quadratic_a)
        theta_v[:,k] = temp_v[:,k] / exner[:,k]
        rho[:,k] = exner[:,k]**CVD_O_RD * P0REF / (RD * theta_v[:,k])

    return rho, exner, theta_v


def mo_diagnose_temperature_numpy(
    theta_v: np.array,
    exner: np.array,
) -> np.array:
    temperature = theta_v * exner
    return temperature

def mo_diagnose_pressure_sfc_numpy(
    exner: np.array,
    temperature: np.array,
    ddqz_z_full: np.array,
    num_levels: int,
) -> np.array:
    pressure_sfc = (
        P0REF * np.exp(CPD_O_RD * np.log( exner[:,num_levels-3] ) +
                       GRAV_O_RD * (
                           ddqz_z_full[:,num_levels-1] / temperature[:,num_levels-1] +
                           ddqz_z_full[:,num_levels-2] / temperature[:,num_levels-2] +
                           0.5 * ddqz_z_full[:,num_levels-3] / temperature[:,num_levels-3]
                       )
                       )
    )
    return pressure_sfc

def mo_diagnose_pressure_numpy(
    pressure_sfc: np.array,
    temperature: np.array,
    ddqz_z_full: np.array,
    cell_size: int,
    num_levels: int,
) -> tuple[np.array, np.array]:
    pressure_ifc = np.zeros((cell_size, num_levels), dtype=float)
    pressure = np.zeros((cell_size, num_levels), dtype=float)
    pressure_ifc[:, num_levels - 1] = pressure_sfc * np.exp( -ddqz_z_full[:, num_levels - 1] / temperature[:, num_levels - 1] )
    pressure[:, num_levels - 1] = np.sqrt(pressure_ifc[:, num_levels-1] * pressure_sfc)
    for k in range(num_levels-2,-1,-1):
        pressure_ifc[:, k] = pressure_ifc[:, k+1] * np.exp(-ddqz_z_full[:, k] / temperature[:, k])
        pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k+1])
    return pressure, pressure_ifc


def read_icon_grid(
    fname_prefix: str, path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads
        from ppser serialized test data
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return (
            sb.IconSerialDataProvider(fname_prefix, str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid()
            .construct_icon_grid()
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)

def compute_time_discretization_implicit_parameters(
    time_discretization_veladv_offctr: float,
    time_discretization_rhotheta_offctr: float
) -> tuple[float, float, float, float]:
    # Weighting coefficients for velocity advection if tendency averaging is used
    # The off - centering specified here turned out to be beneficial to numerical
    # stability in extreme situations
    wgt_nnow_vel = 0.5 - time_discretization_veladv_offctr # default value for veladv_offctr is 0.25
    wgt_nnew_vel = 0.5 + time_discretization_veladv_offctr

    # Weighting coefficients for rho and theta at interface levels in the corrector step
    # This empirically determined weighting minimizes the vertical wind off - centering
    # needed for numerical stability of vertical sound wave propagation
    wgt_nnew_rth = 0.5 + time_discretization_rhotheta_offctr # default value for rhotheta_offctr is -0.1
    wgt_nnow_rth = 1.0 - wgt_nnew_rth

    return wgt_nnow_vel, wgt_nnew_vel, wgt_nnow_rth, wgt_nnew_rth

# TODO (Chia Rui): initialization of prognostic variables and topography of Jablonowski Williamson test
def model_initialization_jabw(
    icon_grid: IconGrid,
    cell_param: CellParams,
    edge_param: EdgeParams,
    time_discretization_veladv_offctr: float,
    time_discretization_rhotheta_offctr: float,
    path: Path,
    rank=0
):
    data_provider = sb.IconSerialDataProvider(
        "jabw", str(path.absolute()), False, mpi_rank=rank
    )
    cells2edges_coeff = data_provider.from_interpolation_savepoint().c_lin_e().asnumpy()
    grid_idx_edge_start_plus1 = icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1)
    grid_idx_edge_end = icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.end(EdgeDim))
    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().asnumpy()
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().asnumpy()
    ddqz_z_full = data_provider.from_metrics_savepoint().ddqz_z_full().asnumpy()
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().asnumpy()
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().asnumpy()
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().asnumpy()
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().asnumpy()
    geopot = data_provider.from_metrics_savepoint().geopot().asnumpy()

    cell_lat = cell_param.cell_center_lat.asnumpy()
    cell_lon = cell_param.cell_center_lon.asnumpy()
    edge_lat = edge_param.edge_center[0].asnumpy()
    edge_lon = edge_param.edge_center[1].asnumpy()
    primal_normal_x = edge_param.primal_normal[0].asnumpy()

    rbv_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()

    cell_size = cell_lat.size
    edge_size = edge_lat.size
    num_levels = icon_grid.num_levels

    p_sfc = 100000.0
    jw_up = 0.0
    jw_u0 = 35.0
    jw_temp0 = 288.
    # DEFINED PARAMETERS for jablonowski williamson:
    eta_0 = 0.252
    eta_t = 0.2    # tropopause
    gamma = 0.005  # temperature elapse rate (K/m)
    dtemp = 4.8e5  # empirical temperature difference (K)
    # for baroclinic wave test
    lonC = MATH_PI / 9.0 # longitude of the perturb centre
    latC = 2.0 * lonC    # latitude of the perturb centre
    ps_o_p0ref = p_sfc / P0REF

    w_numpy = np.zeros((cell_size, num_levels+1), dtype=float)
    exner_numpy = np.zeros((cell_size, num_levels), dtype=float)
    rho_numpy = np.zeros((cell_size, num_levels), dtype=float)
    temperature_numpy = np.zeros((cell_size, num_levels), dtype=float)
    pressure_numpy = np.zeros((cell_size, num_levels), dtype=float)
    theta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)
    eta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)

    mask_array_edge_start_plus1_to_edge_end = np.ones(edge_size, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0:grid_idx_edge_start_plus1] = False

    sin_lat = np.sin(cell_lat)
    cos_lat = np.cos(cell_lat)
    fac1 = 1.0 / 6.3 - 2.0 * (sin_lat ** 6) * (cos_lat ** 2 + 1.0/ 3.0)
    fac2 = (8.0/ 5.0 * (cos_lat ** 3) * (sin_lat ** 2 + 2.0/ 3.0) - 0.25 * MATH_PI) * EARTH_RADIUS * EARTH_ANGULAR_VELOCITY
    lapse_rate = RD * gamma / GRAV
    for k_index in range(num_levels-1,-1,-1):
        eta_old = np.full(cell_size,fill_value=1.0e-7, dtype=float)
        log.info(f'In Newton iteration, k = {k_index}')
        # Newton iteration to determine zeta
        for newton_iter_index in range(100):
            eta_v_numpy[:, k_index] = (eta_old - eta_0) * MATH_PI_2
            cos_etav = np.cos(eta_v_numpy[:, k_index])
            sin_etav = np.sin(eta_v_numpy[:, k_index])

            temperature_avg = jw_temp0 * (eta_old ** lapse_rate)
            geopot_avg = jw_temp0 * GRAV / gamma * (1.0 - eta_old ** lapse_rate)
            temperature_avg = np.where(
                eta_old < eta_t,
                temperature_avg + dtemp * ((eta_t - eta_old) ** 5),
                temperature_avg
            )
            geopot_avg = np.where(
                eta_old < eta_t,
                geopot_avg - RD * dtemp * ((np.log(eta_old / eta_t) + 137.0 / 60.0) * (eta_t ** 5) - 5.0 * (eta_t ** 4) * eta_old + 5.0 * (eta_t ** 3) * (eta_old ** 2) - 10.0 / 3.0 * (eta_t ** 2) * (eta_old ** 3) + 1.25 * eta_t * (eta_old ** 4) - 0.2 * (eta_old ** 5)),
                geopot_avg
            )

            geopot_jw = geopot_avg + jw_u0 * (cos_etav ** 1.5) * (fac1 * jw_u0 * (cos_etav ** 1.5) + fac2)
            temperature_jw = temperature_avg + 0.75 * eta_old * MATH_PI * jw_u0 / RD * sin_etav * np.sqrt(cos_etav) * (2.0 * jw_u0 * fac1 * (cos_etav ** 1.5) + fac2)
            newton_function = geopot_jw - geopot[:, k_index]
            newton_function_prime = -RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        # Final update for zeta_v
        eta_v_numpy[:, k_index] = (eta_old - eta_0) * MATH_PI_2
        # Use analytic expressions at all model level
        exner_numpy[:, k_index] = (eta_old * ps_o_p0ref) ** RD_O_CPD
        theta_v_numpy[:, k_index] = temperature_jw / exner_numpy[:, k_index]
        rho_numpy[:, k_index] = exner_numpy[:, k_index] ** CVD_O_RD * P0REF / RD / theta_v_numpy[:, k_index]
        # initialize diagnose pressure and temperature variables
        pressure_numpy[:, k_index] = P0REF * exner_numpy[:, k_index] ** CPD_O_RD
        temperature_numpy[:, k_index] = temperature_jw

    log.info(f'Newton iteration completed!')
    '''
    mo_cells2edges_scalar_interior.with_backend(backend)(
        cells2edges_coeff,
        eta_v,
        eta_v_e,
        grid_idx_edge_start_plus1,
        grid_idx_edge_end,
        0,
        num_levels,
    )
    '''
    #print(cells2edges_coeff.shape)
    #print(eta_v_numpy.shape)
    #print(mask_array_edge_start_plus1_to_edge_end.shape)
    eta_v_e_numpy = mo_cells2edges_scalar_numpy(
        icon_grid,
        cells2edges_coeff,
        eta_v_numpy,
        mask_array_edge_start_plus1_to_edge_end,
    )
    log.info(f'Cell-to-edge computation completed.')

    vn_numpy = mo_u2vn_jabw_numpy(
        jw_u0,
        jw_up,
        latC,
        lonC,
        edge_lat,
        edge_lon,
        primal_normal_x,
        eta_v_e_numpy,
        mask_array_edge_start_plus1_to_edge_end,
    )
    log.info(f'U2vn computation completed.')

    rho_numpy, exner_numpy, theta_v_numpy = mo_hydro_adjust(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho_numpy,
        exner_numpy,
        theta_v_numpy,
        num_levels,
    )
    log.info(f'Hydrostatis adjustment computation completed.')

    vn = as_field((EdgeDim, KDim), vn_numpy)
    w = as_field((CellDim, KDim), w_numpy)
    exner = as_field((CellDim, KDim), exner_numpy)
    rho = as_field((CellDim, KDim), rho_numpy)
    temperature = as_field((CellDim, KDim), temperature_numpy)
    pressure = as_field((CellDim, KDim), pressure_numpy)
    theta_v = as_field((CellDim, KDim), theta_v_numpy)
    pressure_ifc = as_field((CellDim, KDim), np.zeros((cell_size, num_levels), dtype=float))

    # set surface pressure to the prescribed value
    pressure_sfc = as_field((CellDim,), np.full(cell_size, fill_value=p_sfc, dtype=float))

    u = _allocate(CellDim, KDim, grid=icon_grid)
    v = _allocate(CellDim, KDim, grid=icon_grid)
    grid_idx_cell_start_plus1 = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1)
    grid_idx_cell_end = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))

    mo_rbf_vec_interpol_cell.with_backend(backend)(
        vn,
        rbv_vec_coeff_c1,
        rbv_vec_coeff_c2,
        u,
        v,
        grid_idx_cell_start_plus1,
        grid_idx_cell_end,
        0,
        num_levels,
        offset_provider={
            "C2E2C2E": icon_grid.get_offset_provider("C2E2C2E"),
        },
    )

    log.info(f'U, V computation completed.')

    # TODO (Chia Rui): check whether it is better to diagnose pressure and temperature again after hydrostatic adjustment
    #temperature_numpy = mo_diagnose_temperature_numpy(theta_v_numpy,exner_numpy)
    #pressure_sfc_numpy = mo_diagnose_pressure_sfc_numpy(exner_numpy,temperature_numpy,ddqz_z_full,num_levels)

    diagnostic_state = DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        pressure_sfc=pressure_sfc,
        u=u,
        v=v,
    )

    prognostic_state_now = PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_next = PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )

    diffusion_diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        div_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        dwdx=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        dwdy=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True)
    )
    solve_nonhydro_diagnostic_state = DiagnosticStateNonHydro(
        theta_v_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        exner_pr=_allocate(CellDim, KDim, grid=icon_grid),
        rho_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        ddt_exner_phy=_allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_rho=_allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_thv=_allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_w=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        mass_fl_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_phy=_allocate(EdgeDim, KDim, grid=icon_grid),
        grf_tend_vn=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_apc_ntl1=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_apc_ntl2=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_w_adv_ntl1=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        ddt_w_adv_ntl2=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        vt=_allocate(EdgeDim, KDim, grid=icon_grid),
        vn_ie=_allocate(EdgeDim, KDim, grid=icon_grid, is_halfdim=True),
        w_concorr_c=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=None,
    )

    z_fields = ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_beta=_allocate(CellDim, KDim, grid=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_q=_allocate(CellDim, KDim, grid=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, grid=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, grid=icon_grid),
    )

    prep_adv = PrepAdvection(
        vn_traj=_allocate(EdgeDim, KDim, grid=icon_grid),
        mass_flx_me=_allocate(EdgeDim, KDim, grid=icon_grid),
        mass_flx_ic=_allocate(CellDim, KDim, grid=icon_grid),
    )
    log.info(f'Initialization completed.')

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        z_fields,
        prep_adv,
        0.0,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next
    )


def model_initialization_serialbox(
    icon_grid: IconGrid, path: Path, rank=0
):
    """
        Read prognostic and diagnostic state from serialized data.

        Args:
            path: path to the serialized input data
            rank: mpi rank of the current compute node

        Returns: a tuple containing the data_provider, the initial diagnostic and prognostic state.
            The data_provider is returned such that further timesteps of diagnostics and prognostics
            can be read from within the dummy timeloop

        """

    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )

    diffusion_init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    solve_nonhydro_init_savepoint = data_provider.from_savepoint_nonhydro_init(
        istep=1, date=SIMULATION_START_DATE, jstep=0
    )
    velocity_init_savepoint = data_provider.from_savepoint_velocity_init(
        istep=1, vn_only=False, date=SIMULATION_START_DATE, jstep=0
    )
    prognostic_state_now = diffusion_init_savepoint.construct_prognostics()
    diffusion_diagnostic_state = construct_diagnostics_for_diffusion(diffusion_init_savepoint)
    solve_nonhydro_diagnostic_state = DiagnosticStateNonHydro(
        theta_v_ic=solve_nonhydro_init_savepoint.theta_v_ic(),
        exner_pr=solve_nonhydro_init_savepoint.exner_pr(),
        rho_ic=solve_nonhydro_init_savepoint.rho_ic(),
        ddt_exner_phy=solve_nonhydro_init_savepoint.ddt_exner_phy(),
        grf_tend_rho=solve_nonhydro_init_savepoint.grf_tend_rho(),
        grf_tend_thv=solve_nonhydro_init_savepoint.grf_tend_thv(),
        grf_tend_w=solve_nonhydro_init_savepoint.grf_tend_w(),
        mass_fl_e=solve_nonhydro_init_savepoint.mass_fl_e(),
        ddt_vn_phy=solve_nonhydro_init_savepoint.ddt_vn_phy(),
        grf_tend_vn=solve_nonhydro_init_savepoint.grf_tend_vn(),
        ddt_vn_apc_ntl1=velocity_init_savepoint.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=velocity_init_savepoint.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=velocity_init_savepoint.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=velocity_init_savepoint.ddt_w_adv_pc(2),
        vt=velocity_init_savepoint.vt(),
        vn_ie=velocity_init_savepoint.vn_ie(),
        w_concorr_c=velocity_init_savepoint.w_concorr_c(),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=None,
    )

    z_fields = ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_beta=_allocate(CellDim, KDim, grid=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_q=_allocate(CellDim, KDim, grid=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, grid=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, grid=icon_grid),
    )

    diagnostic_state = DiagnosticState(
        pressure=_allocate(CellDim, KDim, grid=icon_grid),
        pressure_ifc=_allocate(CellDim, KDim, grid=icon_grid),
        temperature=_allocate(CellDim, KDim, grid=icon_grid),
        pressure_sfc=_allocate(CellDim, grid=icon_grid),
        u=_allocate(CellDim, KDim, grid=icon_grid),
        v=_allocate(CellDim, KDim, grid=icon_grid),
    )

    prognostic_state_next = PrognosticState(
        w=solve_nonhydro_init_savepoint.w_new(),
        vn=solve_nonhydro_init_savepoint.vn_new(),
        theta_v=solve_nonhydro_init_savepoint.theta_v_new(),
        rho=solve_nonhydro_init_savepoint.rho_new(),
        exner=solve_nonhydro_init_savepoint.exner_new(),
    )

    prep_adv = PrepAdvection(
        vn_traj=solve_nonhydro_init_savepoint.vn_traj(),
        mass_flx_me=solve_nonhydro_init_savepoint.mass_flx_me(),
        mass_flx_ic=solve_nonhydro_init_savepoint.mass_flx_ic(),
    )

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        z_fields,
        prep_adv,
        solve_nonhydro_init_savepoint.divdamp_fac_o2(),
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )

def read_initial_state(
    icon_grid: IconGrid,
    cell_param: CellParams,
    edge_param: EdgeParams,
    time_discretization_veladv_offctr: float,
    time_discretization_rhotheta_offctr: float,
    path: Path,
    rank=0,
    initialization_type = InitializationType.SB
) -> tuple[
    DiffusionDiagnosticState,
    DiagnosticStateNonHydro,
    ZFields,
    PrepAdvection,
    float,
    DiagnosticState,
    PrognosticState,
    PrognosticState,
]:

    if initialization_type == InitializationType.SB:

        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            z_fields,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_serialbox(icon_grid, path, rank)

    elif initialization_type == InitializationType.JABW:

        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            z_fields,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_jabw(
            icon_grid,
            cell_param,
            edge_param,
            time_discretization_veladv_offctr,
            time_discretization_rhotheta_offctr,
            path,
            rank
        )

    else:
        raise NotImplementedError(INITIALIZATION_ERROR_MSG)

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        z_fields,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_geometry_fields(
    fname_prefix: str, path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> tuple[EdgeParams, CellParams, VerticalModelParams, Field[[CellDim], bool]]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            fname_prefix, str(path.absolute()), False, mpi_rank=rank
        ).from_savepoint_grid()
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vertical_geometry = VerticalModelParams(
            vct_a=sp.vct_a(),
            rayleigh_damping_height=12500,
            nflatlev=sp.nflatlev(),
            nflat_gradp=sp.nflat_gradp(),
        )
        return edge_geometry, cell_geometry, vertical_geometry, sp.c_owner_mask()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_decomp_info(
    fname_prefix: str,
    path: Path,
    procs_props: ProcessProperties,
    ser_type=SerializationType.SB,
) -> DecompositionInfo:
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            fname_prefix, str(path.absolute()), True, procs_props.rank
        )
        return sp.from_savepoint_grid().construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    fname_prefix: str, path: Path, rank=0, ser_type: SerializationType = SerializationType.SB, init_type: InitializationType = InitializationType.SB
) -> tuple[
    DiffusionMetricState, DiffusionInterpolationState, MetricStateNonHydro, InterpolationState, DiagnosticMetricState
]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        data_provider = sb.IconSerialDataProvider(
            fname_prefix, str(path.absolute()), False, mpi_rank=rank
        )
        icon_grid = (
            sb.IconSerialDataProvider(fname_prefix, str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid()
            .construct_icon_grid()
        )
        diffusion_interpolation_state = construct_interpolation_state_for_diffusion(
            data_provider.from_interpolation_savepoint()
        )
        diffusion_metric_state = construct_metric_state_for_diffusion(
            data_provider.from_metrics_savepoint()
        )
        interpolation_savepoint = data_provider.from_interpolation_savepoint()
        grg = interpolation_savepoint.geofac_grg()
        nonhydro_interpolation_state = InterpolationState(
            c_lin_e=interpolation_savepoint.c_lin_e(),
            c_intp=interpolation_savepoint.c_intp(),
            e_flx_avg=interpolation_savepoint.e_flx_avg(),
            geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
            geofac_rot=interpolation_savepoint.geofac_rot(),
            pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            e_bln_c_s=as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), CEDim),
            rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
            geofac_div=as_1D_sparse_field(interpolation_savepoint.geofac_div(), CEDim),
            geofac_n2s=interpolation_savepoint.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        nonhydro_metric_state = MetricStateNonHydro(
            bdy_halo_c=metrics_savepoint.bdy_halo_c(),
            mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
            rayleigh_w=metrics_savepoint.rayleigh_w(),
            exner_exfac=metrics_savepoint.exner_exfac(),
            exner_ref_mc=metrics_savepoint.exner_ref_mc(),
            wgtfac_c=metrics_savepoint.wgtfac_c(),
            wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
            inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
            rho_ref_mc=metrics_savepoint.rho_ref_mc(),
            theta_ref_mc=metrics_savepoint.theta_ref_mc(),
            vwind_expl_wgt=metrics_savepoint.vwind_expl_wgt(),
            d_exner_dz_ref_ic=metrics_savepoint.d_exner_dz_ref_ic(),
            ddqz_z_half=metrics_savepoint.ddqz_z_half(),
            theta_ref_ic=metrics_savepoint.theta_ref_ic(),
            d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
            d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
            rho_ref_me=metrics_savepoint.rho_ref_me(),
            theta_ref_me=metrics_savepoint.theta_ref_me(),
            ddxn_z_full=metrics_savepoint.ddxn_z_full(),
            zdiff_gradp=metrics_savepoint.zdiff_gradp(),
            vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
            ipeidx_dsl=metrics_savepoint.ipeidx_dsl(),
            pg_exdist=metrics_savepoint.pg_exdist(),
            ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
            ddxt_z_full=metrics_savepoint.ddxt_z_full(),
            wgtfac_e=metrics_savepoint.wgtfac_e(),
            wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels),
            vwind_impl_wgt=metrics_savepoint.vwind_impl_wgt(),
            hmask_dd3d=metrics_savepoint.hmask_dd3d(),
            scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
            coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
            coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
            coeff_gradekin=metrics_savepoint.coeff_gradekin(),
        )
        # TODO (Chia RUi): ddqz_z_full is not in serialized data for normal testing
        if init_type == InitializationType.SB:
            diagnostic_metric_state = DiagnosticMetricState(
                ddqz_z_full=_allocate(CellDim, KDim, grid=icon_grid, dtype=float),
                rbv_vec_coeff_c1=_allocate(CellDim, C2E2C2EDim, grid=icon_grid, dtype=float),
                rbv_vec_coeff_c2=_allocate(CellDim, C2E2C2EDim, grid=icon_grid, dtype=float),
                cell_center_lat=_allocate(CellDim, grid=icon_grid, dtype=float),
                cell_center_lon=_allocate(CellDim, grid=icon_grid, dtype=float),
                v_lat=_allocate(VertexDim, grid=icon_grid, dtype=float),
                v_lon=_allocate(VertexDim, grid=icon_grid, dtype=float),
                e_lat=_allocate(EdgeDim, grid=icon_grid, dtype=float),
                e_lon=_allocate(EdgeDim, grid=icon_grid, dtype=float),
                vct_a=_allocate(KDim,grid=icon_grid,is_halfdim=True,dtype=float),
            )
        elif init_type == InitializationType.JABW:
            diagnostic_metric_state = DiagnosticMetricState(
                ddqz_z_full=metrics_savepoint.ddqz_z_full(),
                rbf_vec_coeff_c1=data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1(),
                rbf_vec_coeff_c2=data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2(),
                cell_center_lat=data_provider.from_savepoint_grid().cell_center_lat(),
                cell_center_lon=data_provider.from_savepoint_grid().cell_center_lon(),
                v_lat=data_provider.from_savepoint_grid().v_lat(),
                v_lon=data_provider.from_savepoint_grid().v_lon(),
                e_lat=data_provider.from_savepoint_grid().edge_center_lat(),
                e_lon=data_provider.from_savepoint_grid().edge_center_lon(),
                vct_a=data_provider.from_savepoint_grid().vct_a(),
            )
        return (
            diffusion_metric_state,
            diffusion_interpolation_state,
            nonhydro_metric_state,
            nonhydro_interpolation_state,
            diagnostic_metric_state,
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(
    run_path: str, experiment_name: str, processor_procs: ProcessProperties = None
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation

    """
    run_dir = Path(run_path).absolute() if run_path else Path(__file__).absolute().parent
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(f"dummy_dycore_driver_{experiment_name}.log")
    logfile.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    console_handler.addFilter(ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)
