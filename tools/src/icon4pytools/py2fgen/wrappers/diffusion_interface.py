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

from pathlib import Path

import numpy as np
from gt4py.next.iterator.embedded import np_as_located_field
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    DiffusionType,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.decomposition.definitions import SingleNodeProcessProperties
from icon4py.model.common.dimension import (
    C2E2CODim,
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
from icon4py.model.common.test_utils.datatest_utils import create_icon_serial_data_provider


# todo: add type hints
def run_diffusion_simulation(
    datapath,
    vct_a,
    theta_ref_mc,
    wgtfac_c,
    e_bln_c_s,
    geofac_div,
    geofac_grg_x,
    geofac_grg_y,
    geofac_n2s,
    nudgecoeff_e,
    rbf_coeff_1,
    rbf_coeff_2,
    dwdx,
    dwdy,
    hdef_ic,
    div_ic,
    w,
    vn,
    exner,
    theta_v,
    rho,
    dual_normal_cell_x,
    dual_normal_cell_y,
    dual_normal_vert_x,
    dual_normal_vert_y,
    primal_normal_cell_x,
    primal_normal_cell_y,
    primal_normal_vert_x,
    primal_normal_vert_y,
    tangent_orientation,
    inverse_primal_edge_lengths,
    inv_dual_edge_length,
    inv_vert_vert_length,
    edge_areas,
    f_e,
    cell_areas,
    mean_cell_area,
    ndyn_substeps,
    dtime,
    rayleigh_damping_height,
    nflatlev,
    nflat_gradp,
    diffusion_type,
    hdiff_w,
    hdiff_vn,
    zdiffu_t,
    type_t_diffu,
    type_vn_diffu,
    hdiff_efdt_ratio,
    smagorinski_scaling_factor,
    hdiff_temp,
):
    # grid
    # todo: how to instantiate grid without serialized data?
    processor_props = SingleNodeProcessProperties()
    data_provider = create_icon_serial_data_provider(Path(datapath), processor_props)
    grid_savepoint = data_provider.from_savepoint_grid()
    icon_grid = grid_savepoint.construct_icon_grid(on_gpu=False)

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
    vertical_params = VerticalModelParams(
        vct_a=vct_a,
        rayleigh_damping_height=rayleigh_damping_height,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
    )

    # metric state
    metric_state = DiffusionMetricState(
        mask_hdiff=None,
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        zd_intcoef=None,
        zd_vertoffset=None,
        zd_diffcoef=None,
    )

    # interpolation state
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
    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )

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

    # running diffusion
    print("Running diffusion...")
    diffusion.run(prognostic_state=prognostic_state, diagnostic_state=diagnostic_state, dtime=dtime)


if __name__ == "__main__":
    # serialised data path for instantiating grid
    datapath = "/home/sk/Dev/icon4py/testdata/ser_icondata/mpitask1/exclaim_ape_R02B04/ser_data"

    # grid parameters
    num_cells = 20480
    num_edges = 30720
    num_vertices = 10242
    num_levels = 60
    num_c2ec2o = 4
    num_v2e = 6
    num_ce = num_edges * 2
    num_ec = num_edges * 2
    num_ecv = num_edges * 4
    mean_cell_area = 24907282236.708576

    # other configuration parameters
    limited_area = False
    ndyn_substeps = 2
    dtime = 2.0
    rayleigh_damping_height = 50000
    nflatlev = 30
    nrdmax = 8
    nflat_gradp = 59

    # diffusion configuration
    diffusion_type = DiffusionType.SMAGORINSKY_4TH_ORDER
    hdiff_w = True
    hdiff_vn = True
    zdiffu_t = False
    type_t_diffu = 2
    type_vn_diffu = 1
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    hdiff_temp = True

    # input data - numpy
    rng = np.random.default_rng()

    vct_a = rng.uniform(
        low=0, high=75000, size=(num_levels,)
    )  # has to be from 0 to 75000, must have larger values than rayleigh damping height
    theta_ref_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    wgtfac_c = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    e_bln_c_s = rng.uniform(low=0, high=1, size=(num_ce,))
    geofac_div = rng.uniform(low=0, high=1, size=(num_ce,))
    geofac_grg_x = rng.uniform(low=0, high=1, size=(num_cells, 4))
    geofac_grg_y = rng.uniform(low=0, high=1, size=(num_cells, 4))
    geofac_n2s = rng.uniform(low=0, high=1, size=(num_cells, 4))
    nudgecoeff_e = np.zeros((num_edges,))
    rbf_coeff_1 = rng.uniform(low=0, high=1, size=(num_vertices, num_v2e))
    rbf_coeff_2 = rng.uniform(low=0, high=1, size=(num_vertices, num_v2e))
    dwdx = np.zeros((num_cells, num_levels))
    dwdy = np.zeros((num_cells, num_levels))
    hdef_ic = np.zeros((num_cells, num_levels + 1))
    div_ic = np.zeros((num_cells, num_levels + 1))
    w = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    vn = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    exner = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    theta_v = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    rho = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    dual_normal_cell_x = rng.uniform(low=0, high=1, size=(num_ec))
    dual_normal_cell_y = rng.uniform(low=0, high=1, size=(num_ec))
    dual_normal_vert_x = rng.uniform(low=0, high=1, size=(num_ecv))
    dual_normal_vert_y = rng.uniform(low=0, high=1, size=(num_ecv))
    primal_normal_cell_x = rng.uniform(low=0, high=1, size=(num_ec))
    primal_normal_cell_y = rng.uniform(low=0, high=1, size=(num_ec))
    primal_normal_vert_x = rng.uniform(low=0, high=1, size=(num_ecv))
    primal_normal_vert_y = rng.uniform(low=0, high=1, size=(num_ecv))
    tangent_orientation = rng.uniform(low=0, high=1, size=(num_edges))
    inverse_primal_edge_lengths = rng.uniform(low=0, high=1, size=(num_edges))
    inv_dual_edge_length = rng.uniform(low=0, high=1, size=(num_edges))
    inv_vert_vert_length = rng.uniform(low=0, high=1, size=(num_edges))
    edge_areas = rng.uniform(low=0, high=1, size=(num_edges))
    f_e = rng.uniform(low=0, high=1, size=(num_edges))
    cell_areas = rng.uniform(low=0, high=1, size=(num_cells))

    # input data - gt4py fields
    theta_ref_mc = np_as_located_field(CellDim, KDim)(theta_ref_mc)
    wgtfac_c = np_as_located_field(CellDim, KDim)(wgtfac_c)
    vct_a = np_as_located_field(KDim)(vct_a)
    e_bln_c_s = np_as_located_field(CEDim)(e_bln_c_s)
    geofac_div = np_as_located_field(CEDim)(geofac_div)
    geofac_grg_x = np_as_located_field(CellDim, C2E2CODim)(geofac_grg_x)
    geofac_grg_y = np_as_located_field(CellDim, C2E2CODim)(geofac_grg_y)
    geofac_n2s = np_as_located_field(CellDim, C2E2CODim)(geofac_n2s)
    nudgecoeff_e = np_as_located_field(EdgeDim)(nudgecoeff_e)
    rbf_coeff_1 = np_as_located_field(VertexDim, V2EDim)(rbf_coeff_1)
    rbf_coeff_2 = np_as_located_field(VertexDim, V2EDim)(rbf_coeff_2)
    dwdx = np_as_located_field(CellDim, KDim)(dwdx)
    dwdy = np_as_located_field(CellDim, KDim)(dwdy)
    hdef_ic = np_as_located_field(CellDim, KDim)(hdef_ic)
    div_ic = np_as_located_field(CellDim, KDim)(div_ic)
    w = np_as_located_field(CellDim, KDim)(w)
    vn = np_as_located_field(EdgeDim, KDim)(vn)
    exner = np_as_located_field(CellDim, KDim)(exner)
    theta_v = np_as_located_field(CellDim, KDim)(theta_v)
    rho = np_as_located_field(CellDim, KDim)(rho)
    dual_normal_cell_x = np_as_located_field(
        ECDim,
    )(dual_normal_cell_x)
    dual_normal_cell_y = np_as_located_field(
        ECDim,
    )(dual_normal_cell_y)
    dual_normal_vert_x = np_as_located_field(
        ECVDim,
    )(dual_normal_vert_x)
    dual_normal_vert_y = np_as_located_field(
        ECVDim,
    )(dual_normal_vert_y)
    primal_normal_cell_x = np_as_located_field(
        ECDim,
    )(primal_normal_cell_x)
    primal_normal_cell_y = np_as_located_field(
        ECDim,
    )(primal_normal_cell_y)
    primal_normal_vert_x = np_as_located_field(
        ECVDim,
    )(primal_normal_vert_x)
    primal_normal_vert_y = np_as_located_field(
        ECVDim,
    )(primal_normal_vert_y)
    tangent_orientation = np_as_located_field(
        EdgeDim,
    )(tangent_orientation)
    inverse_primal_edge_lengths = np_as_located_field(
        EdgeDim,
    )(inverse_primal_edge_lengths)
    inv_dual_edge_length = np_as_located_field(
        EdgeDim,
    )(inv_dual_edge_length)
    inv_vert_vert_length = np_as_located_field(
        EdgeDim,
    )(inv_vert_vert_length)
    edge_areas = np_as_located_field(
        EdgeDim,
    )(edge_areas)
    f_e = np_as_located_field(
        EdgeDim,
    )(f_e)
    cell_areas = np_as_located_field(
        CellDim,
    )(cell_areas)

    run_diffusion_simulation(
        datapath,
        vct_a,
        theta_ref_mc,
        wgtfac_c,
        e_bln_c_s,
        geofac_div,
        geofac_grg_x,
        geofac_grg_y,
        geofac_n2s,
        nudgecoeff_e,
        rbf_coeff_1,
        rbf_coeff_2,
        dwdx,
        dwdy,
        hdef_ic,
        div_ic,
        w,
        vn,
        exner,
        theta_v,
        rho,
        dual_normal_cell_x,
        dual_normal_cell_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        primal_normal_cell_x,
        primal_normal_cell_y,
        primal_normal_vert_x,
        primal_normal_vert_y,
        tangent_orientation,
        inverse_primal_edge_lengths,
        inv_dual_edge_length,
        inv_vert_vert_length,
        edge_areas,
        f_e,
        cell_areas,
        mean_cell_area,
        ndyn_substeps,
        dtime,
        rayleigh_damping_height,
        nflatlev,
        nflat_gradp,
        diffusion_type,
        hdiff_w,
        hdiff_vn,
        zdiffu_t,
        type_t_diffu,
        type_vn_diffu,
        hdiff_efdt_ratio,
        smagorinski_scaling_factor,
        hdiff_temp,
    )
