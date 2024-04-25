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
import numpy as np
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
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.decomposition.definitions import DecompositionInfo, MultiNodeRun
from icon4py.model.common.decomposition.mpi_decomposition import get_multinode_properties
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CellDim,
    CellIndexDim,
    E2C2VDim,
    E2CDim,
    ECVDim,
    EdgeDim,
    EdgeIndexDim,
    KDim,
    KHalfDim,
    V2EDim,
    VertexDim,
    VertexIndexDim,
)
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.settings import device
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.grid_utils import _load_from_gridfile, construct_icon_grid
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, flatten_first_two_dims
from icon4py.model.common.test_utils.parallel_helpers import (
    check_comm_size,
)

from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.utils import get_grid_filename, get_icon_grid_loc


logger = setup_logger(__name__)

# global diffusion object
#processor_props = get_multinode_properties(MultiNodeRun())
#exchange = definitions.create_exchange(processor_props)
#diffusion_granule: Diffusion = Diffusion(exchange)
#diffusion_granule: Diffusion = Diffusion()
diffusion_granule: Diffusion


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
    limited_area: bool,
    num_cells: int32,
    num_edges: int32,
    num_verts: int32,
    cells_start_index: Field[[CellIndexDim], int32],
    cells_end_index: Field[[CellIndexDim], int32],
    edge_start_index: Field[[EdgeIndexDim], int32],
    edge_end_index: Field[[EdgeIndexDim], int32],
    vert_start_index: Field[[VertexIndexDim], int32],
    vert_end_index: Field[[VertexIndexDim], int32],
    c2e: Field[[CellDim, C2EDim], int32],
    c2e2c: Field[[CellDim, C2E2CDim], int32],
    v2e: Field[[VertexDim, V2EDim], int32],
    e2c2v: Field[[EdgeDim, E2C2VDim], int32],
    e2c: Field[[EdgeDim, E2CDim], int32],
    c_owner_mask: Field[[CellDim], bool],
    e_owner_mask: Field[[EdgeDim], bool],
    v_owner_mask: Field[[VertexDim], bool],
    c_glb_index: Field[[CellDim], int32],
    e_glb_index: Field[[EdgeDim], int32],
    v_glb_index: Field[[VertexDim], int32],
):
    logger.info(f"Using Device = {device}")

    # ICON grid
    if device.name == "GPU":
        on_gpu = True
    else:
        on_gpu = False

    processor_props = get_multinode_properties(MultiNodeRun())
    exchange = definitions.create_exchange(processor_props)
    diffusion_granule = Diffusion(exchange)

    icon_grid2 = _load_from_gridfile(
        file_path=get_icon_grid_loc(),
        filename=get_grid_filename(),
        num_levels=num_levels,
        on_gpu=on_gpu,
        limited_area=True,
    )

    print("icon_grid2:cell_start:%s", icon_grid2.start_indices[CellDim])
    print("icon_grid2:cell_end:%s", icon_grid2.end_indices[CellDim])
    print("icon_grid2:vert_start:%s", icon_grid2.start_indices[VertexDim])
    print("icon_grid2:vert_end:%s", icon_grid2.end_indices[VertexDim])
    print("icon_grid2:edge_start:%s", icon_grid2.start_indices[EdgeDim])
    print("icon_grid2:edge_end:%s", icon_grid2.end_indices[EdgeDim])
    print("icon_grid2:c2e:%s", icon_grid2.connectivities[C2EDim])
    print("icon_grid2:c2e2c:%s", icon_grid2.connectivities[C2E2CDim])
    print("icon_grid2:v2e:%s", icon_grid2.connectivities[V2EDim])
    print("icon_grid2:e2c2v:%s", icon_grid2.connectivities[E2C2VDim])
    print("icon_grid2:e2c:%s", icon_grid2.connectivities[E2CDim])

    cells_start_index = np.subtract(cells_start_index, 1)
    vert_start_index = np.subtract(vert_start_index, 1)
    # vert_end_index = np.subtract(vert_end_index, 1)
    edge_start_index = np.subtract(edge_start_index, 1)
    # edge_end_index = np.subtract(edge_end_index, 1)
    c2e = np.subtract(c2e, 1)
    c2e2c = np.subtract(c2e2c, 1)
    v2e = np.subtract(v2e, 1)
    e2c2v = np.subtract(e2c2v, 1)
    e2c = np.subtract(e2c, 1)

    icon_grid = construct_icon_grid(
        cells_start_index.ndarray,
        cells_end_index.ndarray,
        vert_start_index.ndarray,
        vert_end_index.ndarray,
        edge_start_index.ndarray,
        edge_end_index.ndarray,
        num_cells,
        num_edges,
        num_verts,
        num_levels,
        c2e.ndarray,
        c2e2c.ndarray,
        v2e.ndarray,
        e2c2v.ndarray,
        e2c.ndarray,
        True,
        on_gpu,
    )

    print("icon_grid:cell_start%s", icon_grid.start_indices[CellDim])
    print("icon_grid:cell_end:%s", icon_grid.end_indices[CellDim])
    print("icon_grid:vert_start:%s", icon_grid.start_indices[VertexDim])
    print("icon_grid:vert_end:%s", icon_grid.end_indices[VertexDim])
    print("icon_grid:edge_start:%s", icon_grid.start_indices[EdgeDim])
    print("icon_grid:edge_end:%s", icon_grid.end_indices[EdgeDim])
    print("icon_grid:c2e:%s", icon_grid.connectivities[C2EDim])
    print("icon_grid:c2e2c:%s", icon_grid.connectivities[C2E2CDim])
    print("icon_grid:v2e:%s", icon_grid.connectivities[V2EDim])
    print("icon_grid:e2c2v:%s", icon_grid.connectivities[E2C2VDim])
    print("icon_grid:e2c:%s", icon_grid.connectivities[E2CDim])

    decomposition_info = (
        DecompositionInfo(klevels=num_levels)
        .with_dimension(CellDim, c_glb_index.ndarray, c_owner_mask.ndarray)
        .with_dimension(EdgeDim, e_glb_index.ndarray, e_owner_mask.ndarray)
        .with_dimension(VertexDim, v_glb_index.ndarray, v_owner_mask.ndarray)
    )

    check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing dycore for experiment 'mch_ch_r04_b09_dsl"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels} "
        f"local cells = {decomposition_info.global_index(CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    owned_cells = decomposition_info.owner_mask(CellDim)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo edges {np.count_nonzero(np.invert(decomposition_info.owner_mask(EdgeDim)))}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )

    #exchange = definitions.create_exchange(processor_props, decomposition_info)
    exchange.second_part(decomposition_info)

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

    #diffusion_granule.set_exchange(exchange)

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
