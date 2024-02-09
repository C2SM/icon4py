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

# We use gt4py type annotations and thus need to ignore this in MyPy
# mypy: disable-error-code="valid-type"

"""
Wrapper module for diffusion granule.

Module contains a diffusion_init and diffusion_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""

import numpy as np
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
    DiffusionParams,
    DiffusionType,
)
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.base import GridConfig, HorizontalGridSize
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalGridSize, VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState

from icon4pytools.py2f.cffi_utils import CffiMethod, to_fields


# TODO (magdalena) Revise interface architecture with Fortran granules:
# The module variable to match the Fortran interface: where only fields are passed.
# We should rather instantiate the object init and return it.
diffusion: Diffusion()

nproma = 50000
field_sizes = {EdgeDim: nproma, CellDim: nproma, VertexDim: nproma}


@to_fields(dim_sizes=field_sizes)
@CffiMethod.register
def diffusion_init(
    nproma: int,
    nlev: int,
    n_shift_total: int,
    nrdmax: float,
    n_dyn_substeps: int,
    nudge_max_coeff: float,
    denom_diffu_v: float,
    hdiff_smag_fac: float,
    hdiff_order: int,
    hdiff_efdt_ratio: float,
    itype_sher: int,
    itype_vn_diffu: int,
    itype_t_diffu: int,
    lhdiff_rcf: bool,
    lhdiff_w: bool,
    lhdiff_temp: bool,
    l_limited_area: bool,
    l_zdiffu_t: bool,
    lsmag_3d: bool,
    vct_a: Field[[KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    zd_intcoef: Field[
        [CellDim, C2E2CDim, KDim], float
    ],  # special DSL field:   zd_intcoef_dsl in mo_vertical_grid.f90
    zd_diffcoef: Field[
        [CellDim, KDim], float
    ],  # special DSL field mask instead of list: zd_diffcoef_dsl in mo_vertical_grid.f90
    wgtfac_c: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    edges_vertex_idx: Field[[EdgeDim, E2VDim], int32],
    edges_cell_idx: Field[[EdgeDim, E2CDim], int32],
    edges_tangent_orientation: Field[[EdgeDim], float],
    edges_primal_normal_vert_1: Field[[ECVDim], float],  # shallow derived type in Fortran
    edges_primal_normal_vert_2: Field[[ECVDim], float],  # shallow derived type in Fortran
    edges_dual_normal_vert_1: Field[[ECVDim], float],  # shallow derived type in Fortran
    edges_dual_normal_vert_2: Field[[ECVDim], float],  # shallow derived type in Fortran
    edges_inv_vert_vert_lengths: Field[[EdgeDim], float],
    edges_inv_primal_edge_length: Field[[EdgeDim], float],
    edges_inv_dual_edge_length: Field[[EdgeDim], float],
    edges_area_edge: Field[[EdgeDim], float],
    cells_neighbor_idx: Field[[CellDim, C2E2CDim], int32],
    cells_edge_idx: Field[[CellDim, C2EDim], int32],
    cells_area: Field[[CellDim], float],
    # dsl specific additional args
    mean_cell_area: float,
    mask_hdiff: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[
        [CECDim], int32
    ],  # vertical offsets used in DSL for absolute indices zd_vertidx in mo_vertical_grid.f90
    rbf_coeff_1: Field[[VertexDim, V2EDim], float],  # -> used in rbf_vec_interpol_vertex
    rbf_coeff_2: Field[[VertexDim, V2EDim], float],  # -> used in rbf_vec_interpol_vertex
    verts_edge_idx: Field[[VertexDim, V2EDim], int32],  # -> mo_intp_rbf_rbf_vec_interpol_vertex
):
    """
    Instantiate and Initialize the diffusion object.

    should only accept simple fields as arguments for compatibility with the standalone
    Fortran ICON Diffusion component (aka Diffusion granule)

    """
    if diffusion.initialized:
        raise DuplicateInitializationException("Diffusion has already been already initialized")

    horizontal_size = HorizontalGridSize(num_cells=nproma, num_vertices=nproma, num_edges=nproma)
    vertical_size = VerticalGridSize(num_lev=nlev)
    mesh_config = GridConfig(
        horizontal_config=horizontal_size,
        vertical_config=vertical_size,
        limited_area=l_limited_area,
        n_shift_total=n_shift_total,
    )
    # we need the start, end indices in order for the grid to be functional those are not passed
    # to init in the Fortran diffusion granule since they are hidden away in the get_indices_[c,e,v]
    # diffusion_run does take p_patch in order to pass it on to other subroutines (interpolation, get_indices...

    c2e2c0 = np.column_stack(
        (
            (np.asarray(cells_neighbor_idx)),
            (np.asarray(range(np.asarray(cells_neighbor_idx).shape[0]))),
        )
    )
    e2c2v = np.asarray(edges_vertex_idx)
    e2v = e2c2v[:, 0:2]
    connectivities = {
        E2CDim: np.asarray(edges_cell_idx),
        E2C2VDim: e2c2v,
        E2VDim: e2v,
        C2EDim: np.asarray(cells_edge_idx),
        C2E2CDim: (np.asarray(cells_neighbor_idx)),
        C2E2CODim: c2e2c0,
        V2EDim: np.asarray(verts_edge_idx),
    }
    # TODO (Magdalena) we need start_index, end_index: those are not passed in the fortran granules,
    #  because they are used through get_indices only
    grid = IconGrid().with_config(mesh_config).with_connectivities(connectivities)
    edge_params = EdgeParams(
        tangent_orientation=edges_tangent_orientation,
        inverse_primal_edge_lengths=edges_inv_primal_edge_length,
        dual_normal_vert_x=edges_dual_normal_vert_1,
        dual_normal_vert_y=edges_dual_normal_vert_2,
        inverse_dual_edge_lengths=edges_inv_dual_edge_length,
        inverse_vertex_vertex_lengths=edges_inv_vert_vert_lengths,
        primal_normal_vert_x=edges_primal_normal_vert_1,
        primal_normal_vert_y=edges_primal_normal_vert_2,
        edge_areas=edges_area_edge,
    )
    cell_params = CellParams(area=cells_area, mean_cell_area=mean_cell_area)
    config: DiffusionConfig = DiffusionConfig(
        diffusion_type=DiffusionType(hdiff_order),
        hdiff_w=lhdiff_w,
        hdiff_temp=lhdiff_temp,
        type_vn_diffu=itype_vn_diffu,
        smag_3d=lsmag_3d,
        type_t_diffu=itype_t_diffu,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        hdiff_w_efdt_ratio=hdiff_efdt_ratio,
        smagorinski_scaling_factor=hdiff_smag_fac,
        n_substeps=n_dyn_substeps,
        zdiffu_t=l_zdiffu_t,
        hdiff_rcf=lhdiff_rcf,
        velocity_boundary_diffusion_denom=denom_diffu_v,
        max_nudging_coeff=nudge_max_coeff,
    )
    vertical_params = VerticalModelParams(vct_a=vct_a, rayleigh_damping_height=nrdmax)

    derived_diffusion_params = DiffusionParams(config)
    metric_state = DiffusionMetricState(
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        mask_hdiff=mask_hdiff,
        zd_vertoffset=zd_vertoffset,
        zd_diffcoef=zd_diffcoef,
        zd_intcoef=zd_intcoef,
    )
    interpolation_state = DiffusionInterpolationState(
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
        config=config,
        params=derived_diffusion_params,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
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
    rho: Field[[CellDim, KDim], float],
):
    diagnostic_state = DiffusionDiagnosticState(hdef_ic, div_ic, dwdx, dwdy)
    prognostic_state = PrognosticState(
        rho=rho,
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
    )
    if linit:
        diffusion.initial_run(
            diagnostic_state,
            prognostic_state,
            dtime,
        )
    else:
        diffusion.run(diagnostic_state, prognostic_state, dtime)


class DuplicateInitializationException(Exception):
    """Raised if the component is already initilalized."""

    pass
