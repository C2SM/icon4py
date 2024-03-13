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
# type: ignore
import os
import time

from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import float64, int32
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.dimension import (
    C2E2CODim,
    CECDim,
    CEDim,
    CellDim,
    ECDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.grid_utils import _load_from_gridfile


DIFFUSION: Diffusion = Diffusion()

GRID_PATH = (
    "/home/sk/Dev/icon4py/testdata"  # todo(samkellerhals): we need a better way to set this path
)
GRID_FILENAME = "grid.nc"


def diffusion_init(
    vct_a: Field[[KDim], float64],
    theta_ref_mc: Field[[CellDim, KDim], float64],
    wgtfac_c: Field[[CellDim, KDim], float64],
    e_bln_c_s: Field[[CEDim], float64],
    geofac_div: Field[[CEDim], float64],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float64],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float64],
    geofac_n2s: Field[[CellDim, C2E2CODim], float64],
    nudgecoeff_e: Field[[EdgeDim], float64],
    rbf_coeff_1: Field[[VertexDim, V2EDim], float64],
    rbf_coeff_2: Field[[VertexDim, V2EDim], float64],
    mask_hdiff: Field[[CellDim, KDim], bool],
    zd_diffcoef: Field[[CellDim, KDim], float64],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_intcoef: Field[[CECDim, KDim], float64],
    num_levels: int32,
    mean_cell_area: float64,
    ndyn_substeps: int32,
    rayleigh_damping_height: float64,
    nflatlev: int32,
    nflat_gradp: int32,
    diffusion_type: int32,
    hdiff_w: bool,
    hdiff_vn: bool,
    zdiffu_t: bool,
    type_t_diffu: int32,
    type_vn_diffu: int32,
    hdiff_efdt_ratio: float64,
    smagorinski_scaling_factor: float64,
    hdiff_temp: bool,
    tangent_orientation: Field[[EdgeDim], float64],
    inverse_primal_edge_lengths: Field[[EdgeDim], float64],
    inv_dual_edge_length: Field[[EdgeDim], float64],
    inv_vert_vert_length: Field[[EdgeDim], float64],
    edge_areas: Field[[EdgeDim], float64],
    f_e: Field[[EdgeDim], float64],
    cell_areas: Field[[CellDim], float64],
    primal_normal_vert_x: Field[[ECVDim], float64],
    primal_normal_vert_y: Field[[ECVDim], float64],
    dual_normal_vert_x: Field[[ECVDim], float64],
    dual_normal_vert_y: Field[[ECVDim], float64],
    primal_normal_cell_x: Field[[ECDim], float64],
    primal_normal_cell_y: Field[[ECDim], float64],
    dual_normal_cell_x: Field[[ECDim], float64],
    dual_normal_cell_y: Field[[ECDim], float64],
):
    # grid
    if os.environ.get("GT4PY_GPU"):
        on_gpu = True
    else:
        on_gpu = False

    icon_grid = _load_from_gridfile(
        file_path=GRID_PATH, filename=GRID_FILENAME, num_levels=num_levels, on_gpu=on_gpu
    )

    # Edge geometry
    edge_params = EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        edge_areas=edge_areas,
        f_e=f_e,
    )

    # cell geometry
    cell_params = CellParams(area=cell_areas, mean_cell_area=mean_cell_area)

    # diffusion parameters
    config = DiffusionConfig(
        diffusion_type=diffusion_type,
        hdiff_w=hdiff_w,
        hdiff_vn=hdiff_vn,
        zdiffu_t=zdiffu_t,
        type_t_diffu=type_t_diffu,
        type_vn_diffu=type_vn_diffu,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        smagorinski_scaling_factor=smagorinski_scaling_factor,
        hdiff_temp=hdiff_temp,
        n_substeps=ndyn_substeps,
    )

    diffusion_params = DiffusionParams(config)

    # vertical parameters
    # todo: use xp
    vertical_params = VerticalModelParams(
        vct_a=vct_a,
        rayleigh_damping_height=rayleigh_damping_height,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
    )

    # metric state
    metric_state = DiffusionMetricState(
        mask_hdiff=mask_hdiff,
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        zd_intcoef=zd_intcoef,
        zd_vertoffset=zd_vertoffset,
        zd_diffcoef=zd_diffcoef,
    )

    # interpolation state
    # todo: cupy arrays instead of as_numpy? (geofac_n2s_c, geofac_n2s)
    interpolation_state = DiffusionInterpolationState(
        e_bln_c_s=e_bln_c_s,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=geofac_div,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    # initialisation
    print("Initialising diffusion...")

    start_time = time.time()

    DIFFUSION.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )

    end_time = time.time()

    print("Done running initialising diffusion.")
    print(f"Diffusion initialisation time: {end_time - start_time:.2f} seconds")


def diffusion_run(
    w: Field[[CellDim, KDim], float64],
    vn: Field[[EdgeDim, KDim], float64],
    exner: Field[[CellDim, KDim], float64],
    theta_v: Field[[CellDim, KDim], float64],
    rho: Field[[CellDim, KDim], float64],
    hdef_ic: Field[[CellDim, KDim], float64],
    div_ic: Field[[CellDim, KDim], float64],
    dwdx: Field[[CellDim, KDim], float64],
    dwdy: Field[[CellDim, KDim], float64],
    dtime: float64,
):
    # prognostic and diagnostic variables
    prognostic_state = PrognosticState(
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
    )

    diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
    )

    print("Running diffusion...")

    start_time = time.time()

    DIFFUSION.run(prognostic_state=prognostic_state, diagnostic_state=diagnostic_state, dtime=dtime)

    end_time = time.time()

    print("Done running diffusion.")
    print(f"Diffusion run time: {end_time - start_time:.2f} seconds")
