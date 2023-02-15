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

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2CDim,
    E2VDim,
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
from icon4py.diffusion.horizontal import (
    CellParams,
    EdgeParams,
    HorizontalMeshSize,
)
from icon4py.diffusion.icon_grid import (
    IconGrid,
    MeshConfig,
    VerticalMeshConfig,
    VerticalModelParams,
)
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.diffusion.prognostic_state import PrognosticState
from icon4py.py2f.cffi_utils import CffiMethod


diffusion: Diffusion(run_program=True)


@CffiMethod.register
def diffusion_init(
    # cvd_o_rd, -> const
    # grav, jg -> unused
    nproma: int,
    nlev: int,
    # nblks_e, nblks_v, nblks_c -> unused
    n_shift: int,
    n_shift_total: int,
    nrdmax: float,
    n_dyn_substeps: int,
    nudge_max_coeff: float,
    denom_diffu_v: float,
    # hdiff_smag_z, hdiff_smag_z2, hdiff_smag_z3, hdiff_smag_z4, -> calculated internally
    hdiff_smag_fac: float,
    # hdiff_smag_fac2, hdiff_smag_fac3, hdiff_smag_fac4, -> calculated internally
    hdiff_order: int,
    hdiff_efdt_ratio: float,
    # k4, k4w -> calculated internally
    # itype_comm -> unused
    itype_sher: int,
    itype_vn_diffu: int,
    itype_t_diffu: int,
    # p_test_run -> unused, ??
    # lphys -> ?? see spreadsheet
    lhdiff_rcf: bool,
    lhdiff_w: bool,
    lhdiff_temp: bool,
    l_limited_area: bool,
    lfeedback: bool,
    l_zdiffu_t: bool,
    ltkeshs: bool,
    lsmag_3d: bool,
    # lvert_nest -> unused
    ltimer: bool,
    lvert_nest: bool,
    # ddt_vn_hdf_is_associated, ddt_vn_dyn_is_associated, -> unused, vector machine specific
    vct_a: Field[[KDim], float],
    # c_lin_e: [[EdgeDim, E2CDim], float], -> unused, needed for velocity advection
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    # cells_aw_verts -> unused, ??
    geofac_div: Field[[CellDim, C2EDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    # enhfac_diffu -> calculated internally, ??
    zd_intcoef: Field[[CellDim, C2E2CDim, KDim], float],
    # zd_geofac -> ?? same as geofac_2ns
    zd_diffcoef: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    # wgtfac_e, wgtfacq_e, wgtfacq1_e, -> unused
    # ddqz_z_full_e -> unused, velocity advection?
    theta_ref_mc: Field[[CellDim, KDim], float],
    zd_indlist: Field[[CellDim, C2E2CDim, KDim], int],
    # zd_blklist -> unused,
    zd_vertidx: Field[[CellDim, C2E2CDim, KDim], int],
    zd_listdim: int,
    # edges_start_block, edges_end_block, -> unused. ??
    edges_vertex_idx: Field[[EdgeDim, E2VDim], int],
    # edges_vertex_blk, -> unused
    edges_cell_idx: Field[[EdgeDim, E2CDim], int],
    # edges_cell_blk, -> unused
    edges_tangent_orientation: Field[[EdgeDim], float],
    edges_primal_normal_vert_1: Field[[ECVDim], float],
    edges_primal_normal_vert_2: Field[[ECVDim], float],
    edges_dual_normal_vert_1: Field[[ECVDim], float],
    edges_dual_normal_vert_2: Field[[ECVDim], float],
    edges_inv_vert_vert_lengths: Field[[EdgeDim], float],
    edges_inv_primal_edge_length: Field[[EdgeDim], float],
    edges_inv_dual_edge_length: Field[[EdgeDim], float],
    edges_area_edge: Field[[EdgeDim], float],
    # cells_start_block, cells_end_block, -> unused
    cells_neighbor_idx: Field[[CellDim, C2E2CDim], int],
    # cells_neighbor_blk -> unused
    cells_edge_idx: Field[[CellDim, C2EDim], int],
    # cells_edge_blk -> unused
    cells_area: Field[[CellDim], float],
    mask_hdiff: Field[[CellDim, KDim], int],  # dsl specific
    rbf_coeff_1: Field[
        [VertexDim, V2EDim], float
    ],  # -> used in rbf_vec_interpol_vertex
    rbf_coeff_2: Field[
        [VertexDim, V2EDim], float
    ],  # -> used in rbf_vec_interpol_vertex
    verts_edge_idx: Field[
        [VertexDim, V2EDim], int
    ],  # -> mo_intp_rbf_rbf_vec_interpol_vertex
):
    """
    Instantiate and Initialize the diffusion object.

    should only accept simple fields as arguments for compatibility with the standalone
    Fortran ICON Diffusion component (aka Diffusion granule)

    """
    # TODO set up grid from input fields
    # TODO set up configuration from input fields
    config: DiffusionConfig = DiffusionConfig(
        diffusion_type=hdiff_order,
        type_t_diffu=itype_t_diffu,
        type_vn_diffu=itype_vn_diffu,
        smagorinski_scaling_factor=hdiff_smag_fac,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        max_nudging_coeff=nudge_max_coeff,
        velocity_boundary_diffusion_denominator=denom_diffu_v,
        hdiff_rcf=lhdiff_rcf,
        hdiff_w=lhdiff_w,
        hdiff_temp=lhdiff_temp,
        zdiffu_t=l_zdiffu_t,
        type_sher=itype_sher,
        tkeshs=ltkeshs,
        n_substeps=n_dyn_substeps,
        smag_3d=lsmag_3d,
    )
    horizontal_size = HorizontalMeshSize(
        num_cells=nproma, num_vertices=nproma, num_edges=nproma
    )
    vertical_size = VerticalMeshConfig(
        num_lev=nlev, nshift=n_shift, nshift_total=n_shift_total
    )
    mesh_config = MeshConfig(
        horizontal_config=horizontal_size,
        vertical_config=vertical_size,
        limited_area=l_limited_area,
    )
    c2e2c = np.asarray(cells_neighbor_idx)
    c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
    connectivities = {
        E2CDim: np.asarray(edges_cell_idx),
        E2VDim: np.asarray(edges_vertex_idx),
        C2EDim: np.asarray(cells_edge_idx),
        C2E2CDim: c2e2c,
        C2E2CODim: c2e2c0,
        V2EDim: np.asarray(verts_edge_idx),
    }
    grid = IconGrid().with_config(mesh_config).with_connectivities(connectivities)

    edges_params = EdgeParams(
        tangent_orientation=edges_tangent_orientation,
        inverse_primal_edge_lengths=edges_inv_primal_edge_length,
        dual_normal_vert=(edges_dual_normal_vert_1, edges_dual_normal_vert_2),
        inverse_dual_edge_lengths=edges_inv_dual_edge_length,
        inverse_vertex_vertex_lengths=edges_inv_vert_vert_lengths,
        primal_normal_vert=(edges_primal_normal_vert_1, edges_primal_normal_vert_2),
        edge_areas=edges_area_edge,
    )
    cell_params = CellParams(cells_area)
    vertical_params = VerticalModelParams(vct_a=vct_a, rayleigh_damping_height=nrdmax)
    derived_diffusion_params = DiffusionParams(config)
    metric_state = MetricState(
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        mask_hdiff=mask_hdiff,
        zd_vertidx=zd_vertidx,
        zd_diffcoef=zd_diffcoef,
        zd_intcoef=zd_intcoef,
        zd_indlist=zd_indlist,
        zd_listdim=zd_listdim,
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
    jg: int,  # -> unused
    dtime: float,
    linit: bool,
    # p_patch -> passed as simple fields
    # p_int -> passed as simple fields
    vn: Field[[EdgeDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],  # -> unused, part of diagnostic in velocity advection
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    # ddt_vn_dyn, ddt_vn_hdf -> optional, unused
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
