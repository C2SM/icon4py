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

"""
Wrapper module for diffusion granule.

Module contains a diffusion_init and diffusion_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""

from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import float64, int32
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    TurbulenceShearForcingType,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    ECVDim,
    EdgeDim,
    KDim,
    KHalfDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.settings import device
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.grid_utils import _load_from_gridfile
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, flatten_first_two_dims

from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.utils import get_grid_filename, get_icon_grid_loc


logger = setup_logger(__name__)

# global diffusion object
diffusion_granule: Diffusion = Diffusion()


def diffusion_init(
    vct_a: Field[[KHalfDim], float64],
    theta_ref_mc: Field[[CellDim, KDim], float64],
    wgtfac_c: Field[[CellDim, KHalfDim], float64],
    e_bln_c_s: Field[[CellDim, C2EDim], float64],
    geofac_div: Field[[CellDim, C2EDim], float64],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float64],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float64],
    geofac_n2s: Field[[CellDim, C2E2CODim], float64],
    nudgecoeff_e: Field[[EdgeDim], float64],
    rbf_coeff_1: Field[[VertexDim, V2EDim], float64],
    rbf_coeff_2: Field[[VertexDim, V2EDim], float64],
    mask_hdiff: Field[[CellDim, KDim], bool],
    zd_diffcoef: Field[[CellDim, KDim], float64],
    zd_vertoffset: Field[[CellDim, C2E2CDim, KDim], int32],
    zd_intcoef: Field[[CellDim, C2E2CDim, KDim], float64],
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
    primal_normal_vert_x: Field[[EdgeDim, E2C2VDim], float64],
    primal_normal_vert_y: Field[[EdgeDim, E2C2VDim], float64],
    dual_normal_vert_x: Field[[EdgeDim, E2C2VDim], float64],
    dual_normal_vert_y: Field[[EdgeDim, E2C2VDim], float64],
    primal_normal_cell_x: Field[[EdgeDim, E2CDim], float64],
    primal_normal_cell_y: Field[[EdgeDim, E2CDim], float64],
    dual_normal_cell_x: Field[[EdgeDim, E2CDim], float64],
    dual_normal_cell_y: Field[[EdgeDim, E2CDim], float64],
):
    logger.info(f"Using Device = {device}")

    # ICON grid
    if device.name == "GPU":
        on_gpu = True
    else:
        on_gpu = False

    icon_grid = _load_from_gridfile(
        file_path=get_icon_grid_loc(),
        filename=get_grid_filename(),
        num_levels=num_levels,
        on_gpu=on_gpu,
        limited_area=True,
    )

    # Edge geometry
    edge_params = EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=as_1D_sparse_field(primal_normal_vert_x, ECVDim),
        primal_normal_vert_y=as_1D_sparse_field(primal_normal_vert_y, ECVDim),
        dual_normal_vert_x=as_1D_sparse_field(dual_normal_vert_x, ECVDim),
        dual_normal_vert_y=as_1D_sparse_field(dual_normal_vert_y, ECVDim),
        primal_normal_cell_x=as_1D_sparse_field(primal_normal_cell_x, ECVDim),
        primal_normal_cell_y=as_1D_sparse_field(primal_normal_cell_y, ECVDim),
        dual_normal_cell_x=as_1D_sparse_field(dual_normal_cell_x, ECVDim),
        dual_normal_cell_y=as_1D_sparse_field(dual_normal_cell_y, ECVDim),
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
        thslp_zdiffu=0.02,
        thhgtd_zdiffu=125.0,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        shear_type=TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )

    diffusion_params = DiffusionParams(config)

    # vertical parameters
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
        zd_intcoef=flatten_first_two_dims(CECDim, KDim, field=zd_intcoef),
        zd_vertoffset=flatten_first_two_dims(CECDim, KDim, field=zd_vertoffset),
        zd_diffcoef=zd_diffcoef,
    )

    # interpolation state
    interpolation_state = DiffusionInterpolationState(
        e_bln_c_s=as_1D_sparse_field(e_bln_c_s, CEDim),
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=as_1D_sparse_field(geofac_div, CEDim),
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )


def diffusion_run(
    w: Field[[CellDim, KHalfDim], float64],
    vn: Field[[EdgeDim, KDim], float64],
    exner: Field[[CellDim, KDim], float64],
    theta_v: Field[[CellDim, KDim], float64],
    rho: Field[[CellDim, KDim], float64],
    hdef_ic: Field[[CellDim, KHalfDim], float64],
    div_ic: Field[[CellDim, KHalfDim], float64],
    dwdx: Field[[CellDim, KHalfDim], float64],
    dwdy: Field[[CellDim, KHalfDim], float64],
    dtime: float64,
    linit: bool,
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

    if linit:
        diffusion_granule.initial_run(
            diagnostic_state,
            prognostic_state,
            dtime,
        )
    else:
        diffusion_granule.run(
            prognostic_state=prognostic_state, diagnostic_state=diagnostic_state, dtime=dtime
        )
