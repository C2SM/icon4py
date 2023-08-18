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
import numpy as np

# flake8: noqa
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.common.dimension import (
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
from icon4py.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    DiffusionType,
)
from icon4py.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
    PrognosticState,
)
from icon4py.grid.horizontal import CellParams, EdgeParams, HorizontalGridSize
from icon4py.grid.icon_grid import GridConfig, IconGrid
from icon4py.grid.vertical import VerticalGridSize, VerticalModelParams
from icon4py.py2f.cffi_utils import CffiMethod, to_fields


diffusion: Diffusion()

nproma = 50000
field_sizes = {EdgeDim: nproma, CellDim: nproma, VertexDim: nproma}


@to_fields(dim_sizes=field_sizes)
@CffiMethod.register
def diffusion_init(
    cvd_o_rd: float,  # unused, calculated internally
    grav: float,  # unused, calculated internally
    jg: int,  # unused (no nesting)
    nproma: int,
    nlev: int,
    nblks_e: int,  # unused (no blocking)
    nblks_v: int,  # unused (no blocking)
    nblks_c: int,  # unused (no blocking)
    n_shift: int,
    n_shift_total: int,
    nrdmax: float,
    n_dyn_substeps: int,
    nudge_max_coeff: float,
    denom_diffu_v: float,
    hdiff_smag_z: float,  # calculated internally
    hdiff_smag_z2: float,  # calculated internally
    hdiff_smag_z3: float,  # calculated internally
    hdiff_smag_z4: float,  # calculated internally
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,  # calculated internally
    hdiff_smag_fac3: float,  # calculated internally
    hdiff_smag_fac4: float,  # calculated internally
    hdiff_order: int,
    hdiff_efdt_ratio: float,
    k4: float,
    k4w: float,  # calculated internally
    itype_comm: int,  # unused, single node
    itype_sher: int,
    itype_vn_diffu: int,
    itype_t_diffu: int,
    p_test_run: bool,  # unused
    lphys: bool,  # unused
    lhdiff_rcf: bool,
    lhdiff_w: bool,
    lhdiff_temp: bool,
    l_limited_area: bool,
    lfeedback: bool,  # unused
    l_zdiffu_t: bool,
    ltkeshs: bool,
    lsmag_3d: bool,
    lvert_nest: bool,  # unused no nesting
    ltimer: bool,  # unused
    ddt_vn_hdf_is_associated: bool,
    ddt_vn_dyn_is_associated: bool,  # unused,
    vct_a: Field[[KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],  # unused option
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    # cells_aw_verts: Field[[VertexDim, V2C2VDim], float] # unused, dimension unclear 9 cell type
    geofac_div: Field[[CellDim, C2EDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    enhfac_diffu: Field[[KDim], float],  #  unused, diffusion type  only
    zd_intcoef: Field[
        [CellDim, C2E2CDim, KDim], float
    ],  # special DSL field:   zd_intcoef_dsl in mo_vertical_grid.f90
    zd_geofac: Field[[CellDim, C2E2CODim], float],  # same as geofac_n2s
    zd_diffcoef: Field[
        [CellDim, KDim], float
    ],  # special DSL field mask instead of list: zd_diffcoef_dsl in mo_vertical_grid.f90
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],  #  unused
    wgtfacq_e: Field[[EdgeDim, KDim], float],  #  unused
    wgtfacq1_e: Field[[EdgeDim, KDim], float],  #  unused
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],  # unused
    theta_ref_mc: Field[[CellDim, KDim], float],
    zd_indlist: Field[
        [CellDim, C2E2CDim, KDim], int32
    ],  # unused, index list for  steep points, DSL uses mask_hdiff instead (see below)
    zd_blklist: Field[[CellDim, C2E2CDim, KDim], int32],  # unused,
    zd_vertidx: Field[
        [CellDim, C2E2CDim, KDim], int32
    ],  # DSL uses offsets instead of absolute indices zd_vertoffset_dsl in mo_vertical_grid.f90
    zd_listdim: int,
    ### edges_start_block: np.ndarray, # unused, we need start, end indices instead, see below
    ### edges_end_block:np.ndarray, # unused, we need start, end indices instead, see below
    edges_vertex_idx: Field[[EdgeDim, E2VDim], int32],
    edges_vertex_blk: Field[[EdgeDim, E2VDim], int32],  # unused, no blocking
    edges_cell_idx: Field[[EdgeDim, E2CDim], int32],
    edges_cell_blk: Field[[EdgeDim, E2CDim], int32],  # unused, no blocking
    edges_tangent_orientation: Field[[EdgeDim], float],
    edges_primal_normal_vert_1: Field[
        [ECVDim], float
    ],  # shallow derived type in Fortran
    edges_primal_normal_vert_2: Field[
        [ECVDim], float
    ],  # shallow derived type in Fortran
    edges_dual_normal_vert_1: Field[[ECVDim], float],  # shallow derived type in Fortran
    edges_dual_normal_vert_2: Field[[ECVDim], float],  # shallow derived type in Fortran
    edges_inv_vert_vert_lengths: Field[[EdgeDim], float],
    edges_inv_primal_edge_length: Field[[EdgeDim], float],
    edges_inv_dual_edge_length: Field[[EdgeDim], float],
    edges_area_edge: Field[[EdgeDim], float],
    ### cells_start_block: np.ndarray,  # unused, we need start, end indices instead, see below
    ### cells_end_block: np.ndarray,  # unused, we need start, end indices instead, see below
    cells_neighbor_idx: Field[[CellDim, C2E2CDim], int32],
    ### cells_neighbor_blk: Field[[CellDim, C2E2CDim], int32],  # unused, not blocking,
    cells_edge_idx: Field[[CellDim, C2EDim], int32],
    ### cells_edge_blk: np.ndarray,  # unused, not blocking,
    cells_area: Field[[CellDim], float],
    # dsl specific additional args
    mask_hdiff: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim], int32],
    rbf_coeff_1: Field[
        [VertexDim, V2EDim], float
    ],  # -> used in rbf_vec_interpol_vertex
    rbf_coeff_2: Field[
        [VertexDim, V2EDim], float
    ],  # -> used in rbf_vec_interpol_vertex
    verts_edge_idx: Field[
        [VertexDim, V2EDim], int32
    ],  # -> mo_intp_rbf_rbf_vec_interpol_vertex
    ### vertex_starts: np.ndarray, # start and endindices for horizontal domain, are through p_patch in Fortran granule and used implicitly in get_indices_
    ### vertex_ends: np.ndarray,
    ### edge_starts:np.ndarray,
    ### edge_ends:np.ndarray,
    ### cell_starts: np.ndarray,
    ### cell_ends:np.ndarray,
):
    """
    Instantiate and Initialize the diffusion object.

    Takes all the argument that are passed to the Fortran diffusion granule, see
    https://gitlab.dkrz.de/icon/icon-cscs/-/blob/diffusion_granule/src/atm_dyn_iconam/mo_nh_diffusion_new.f90
    It should only accept simple fields as arguments for compatibility with the standalone
    Fortran ICON Diffusion component (aka Diffusion granule).
    Open issues:
    - It does not yet support shallow derived types so these have to passed as simple fields.
    - How to handle DSL specific fields (which are restructured to match the need of the DSL)

    """
    if diffusion.initialized:
        raise DuplicateInitializationException(
            "Diffusion has already been already initialized"
        )

    diffusion_config: DiffusionConfig = DiffusionConfig(
        diffusion_type=DiffusionType(hdiff_order),
        type_t_diffu=itype_t_diffu,
        type_vn_diffu=itype_vn_diffu,
        smagorinski_scaling_factor=hdiff_smag_fac,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        max_nudging_coeff=nudge_max_coeff,
        velocity_boundary_diffusion_denom=denom_diffu_v,
        hdiff_rcf=lhdiff_rcf,
        hdiff_w=lhdiff_w,
        hdiff_temp=lhdiff_temp,
        zdiffu_t=l_zdiffu_t,
        type_sher=itype_sher,
        tkeshs=ltkeshs,
        n_substeps=n_dyn_substeps,
        smag_3d=lsmag_3d,
    )
    horizontal_size = HorizontalGridSize(
        num_cells=nproma, num_vertices=nproma, num_edges=nproma
    )
    vertical_size = VerticalGridSize(num_lev=nlev)
    mesh_config = GridConfig(
        horizontal_config=horizontal_size,
        vertical_config=vertical_size,
        limited_area=l_limited_area,
        n_shift_total=n_shift_total,
    )

    edges_params = EdgeParams(
        tangent_orientation=edges_tangent_orientation,
        primal_edge_lengths=None,
        inverse_primal_edge_lengths=edges_inv_primal_edge_length,
        dual_edge_lengths=None,
        inverse_dual_edge_lengths=edges_inv_dual_edge_length,
        inverse_vertex_vertex_lengths=edges_inv_vert_vert_lengths,
        dual_normal_vert_x=edges_dual_normal_vert_1,
        dual_normal_vert_y=edges_dual_normal_vert_2,
        primal_normal_vert_x=edges_primal_normal_vert_1,
        primal_normal_vert_y=edges_primal_normal_vert_2,
        edge_areas=edges_area_edge,
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
    grid = (
        IconGrid()
        .with_config(mesh_config)
        .with_connectivities(connectivities)
        .with_start_end_indices(VertexDim, vertex_starts, vertex_ends)
        .with_start_end_indices(EdgeDim, edge_starts, edge_ends)
        .with_start_end_indices(CellDim, cell_starts, cell_ends)
    )

    cell_params = CellParams(cells_area)
    vertical_params = VerticalModelParams(vct_a=vct_a, rayleigh_damping_height=nrdmax)
    derived_diffusion_params = DiffusionParams(diffusion_config)
    metric_state = DiffusionMetricState(
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        mask_hdiff=mask_hdiff,
        zd_diffcoef=zd_diffcoef,
        zd_intcoef=zd_intcoef,
        zd_vertoffset=zd_vertoffset,
    )
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

    diffusion.init(
        grid=grid,
        config=config,
        params=derived_diffusion_params,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edges_params,
        cell_params=cell_params,
    )


@to_fields(dim_sizes=field_sizes)
@CffiMethod.register
def diffusion_run(
    jg: int,  # -> unused, no nesting
    dtime: float,
    linit: bool,
    vn: Field[[EdgeDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    vt: Field[
        [EdgeDim, KDim], float
    ],  # -> unused, part of diagnostic in velocity advection
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    ddt_vn_dyn: Field[[EdgeDim, KDim], float],  # unused -> optional
    ddt_vn_hdf: Field[[EdgeDim, KDim], float],  # unused -> optional
):
    diagnostic_state = DiffusionDiagnosticState(hdef_ic, div_ic, dwdx, dwdy)
    prognostic_state = PrognosticState(
        w=w,
        vn=vn,
        exner_pressure=exner,
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
