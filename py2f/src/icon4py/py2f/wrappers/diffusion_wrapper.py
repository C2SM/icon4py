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

# flake8: noqa
from gt4py.next.common import Field

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.diffusion.diagnostic_state import DiagnosticState
from icon4py.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
)
from icon4py.diffusion.horizontal import CellParams, EdgeParams, HorizontalMeshSize
from icon4py.diffusion.icon_grid import IconGrid, VerticalModelParams, MeshConfig, \
    VerticalMeshConfig
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.diffusion.prognostic_state import PrognosticState
from icon4py.py2f.cffi_utils import CffiMethod


diffusion: Diffusion(run_program=True)


@CffiMethod.register
def diffusion_init(
    nproma: int,
    nlev:int,
    n_shift_total: int,
    n_dyn_substeps: int,
    hdiff_order: int,
    type_vn_diffu:int,
    type_t_diffu:int,
    hdiff_smag_fac: float,
    hdiff_efdt_ratio: float,
    lhdiff_temp: bool,
    lhdiff_rcf: bool,
    lhdiff_w:bool,
    l_zdiffu_t: bool,
    denom_diffu_v: float,
    lsmag_3d: bool,
    nudge_max_coeff: float,
    l_limited_area: bool,
    ltimer: bool,
    lfeedback: bool,
    lvert_nest: bool,
    type_sher: int,
    ltkeshs: bool,
    vct_a: Field[[KDim], float],
    nrdmax: float,
    theta_ref_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    mask_hdiff: Field[[CellDim, KDim], int],
    zd_indlist: Field[[CellDim, C2E2CDim, KDim], int],
    zd_vertidx: Field[[CellDim, C2E2CDim, KDim], int],
    zd_diffcoef: Field[[CellDim, KDim], float],
    zd_intcoef: Field[[CellDim, C2E2CDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    rbf_coeff_1: Field[[VertexDim, V2EDim], float],
    rbf_coeff_2: Field[[VertexDim, V2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    primal_edge_lengths: Field[[EdgeDim], float],
    inverse_primal_edge_lengths: Field[[EdgeDim], float],
    dual_edge_lengths: Field[[EdgeDim], float],
    inverse_dual_edge_lengths: Field[[EdgeDim], float],
    inverse_vertex_vertex_lengths: Field[[EdgeDim], float],
    primal_normal_vert_1: Field[[ECVDim], float],
    primal_normal_vert_2: Field[[ECVDim], float],
    dual_normal_vert_1: Field[[ECVDim], float],
    dual_normal_vert_2: Field[[ECVDim], float],
    edge_areas: Field[[EdgeDim], float],
    cell_areas: Field[[CellDim], float],
):
    """
    Instantiate and Initialize the diffusion object.

    should only accept simple fields as arguments for compatibility with the standalone
    Fortran ICON Diffusion component (aka Diffusion granule)

    """
    #TODO set up grid from input fields
    # TODO set up configuration from input fields
    config: DiffusionConfig = DiffusionConfig(diffusion_type=hdiff_order,
                                              type_t_diffu=type_t_diffu,
                                              type_vn_diffu=type_vn_diffu,
                                              smagorinski_scaling_factor=hdiff_smag_fac,
                                              hdiff_efdt_ratio=hdiff_efdt_ratio,
                                              max_nudging_coeff=nudge_max_coeff,
                                              velocity_boundary_diffusion_denominator=denom_diffu_v,
                                              hdiff_rcf=lhdiff_rcf,
                                              hdiff_w = lhdiff_w,
                                              hdiff_temp = lhdiff_temp,
                                              zdiffu_t = l_zdiffu_t,
                                              type_sher=type_sher,
                                              tkeshs=ltkeshs,
                                              n_substeps=n_dyn_substeps,
                                              smag_3d=lsmag_3d,

    )
    horizontal_size = HorizontalMeshSize(num_cells=nproma,
                                         num_vertices = nproma,
                                         num_edges=nproma)
    vertical_size = VerticalMeshConfig(num_lev=nlev, nshift=n_shift_total)
    mesh_config = MeshConfig(
        horizontal_config=horizontal_size,
        vertical_config=vertical_size,
        limited_area=l_limited_area
    )

    grid = IconGrid(config=mesh_config,

    )

    edges_params = EdgeParams(
        tangent_orientation=tangent_orientation,
        primal_edge_lengths=primal_edge_lengths,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        dual_normal_vert=(dual_normal_vert_1, dual_normal_vert_2),
        dual_edge_lengths=dual_edge_lengths,
        inverse_dual_edge_lengths=inverse_dual_edge_lengths,
        inverse_vertex_vertex_lengths=inverse_vertex_vertex_lengths,
        primal_normal_vert=(primal_normal_vert_1, primal_normal_vert_2),
        edge_areas=edge_areas,
    )
    cell_params = CellParams(cell_areas)
    vertical_params = VerticalModelParams(vct_a=vct_a, rayleigh_damping_height=nrdmax)
    derived_diffusion_params = DiffusionParams(config)
    metric_state = MetricState(
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        mask_hdiff=mask_hdiff,
        zd_vertidx=zd_vertidx,
        zd_diffcoef=zd_diffcoef,
        zd_intcoef=zd_intcoef,
        zd_indlist = zd_indlist,
    )
    interpolation_state = InterpolationState(
        e_bln_c_s,
        rbf_coeff_1,
        rbf_coeff_2,
        geofac_div,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        nudgecoeff_e,
    )

    diffusion.init(
        grid=grid,
        cell_params=cell_params,
        edges_params=edges_params,
        config=config,
        params=derived_diffusion_params,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
    )


@CffiMethod.register
def diffusion_run(
    dtime: float,
    linit: bool,
    vn: Field[[EdgeDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
):
    diagnostic_state = DiagnosticState(hdef_ic, div_ic, dwdx, dwdy)
    prognostic_state = PrognosticState(
        w=w,
        vn=vn,
        exner_pressure=exner,
        theta_v=theta_v,
    )
    if linit:
        diffusion.initial_step(
            diagnostic_state,
            prognostic_state,
            dtime,
        )
    else:
        diffusion.time_step(diagnostic_state, prognostic_state, dtime)


class DuplicateInitializationException(Exception):
    """Raised if the component is already initilalized."""

    pass
