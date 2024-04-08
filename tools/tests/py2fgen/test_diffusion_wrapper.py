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
import pytest
from gt4py.next import as_field
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionType
from icon4py.model.common.config import Icon4PyConfig
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)

from icon4pytools.py2fgen.wrappers.diffusion import diffusion_init, diffusion_run


# Choose array backend
config = Icon4PyConfig()
xp = config.array_ns


# todo(samkellerhals): turn on and off using a marker/option
@pytest.mark.skip
def test_diffusion_wrapper_py():
    # grid parameters
    num_cells = 20480
    num_edges = 30720
    num_vertices = 10242
    num_levels = 60
    num_c2ec2o = 4
    num_v2e = 6
    num_c2e = 2
    num_e2c2v = 4
    num_c2e2c = 3
    num_e2c = 3
    mean_cell_area = 24907282236.708576

    # other configuration parameters
    ndyn_substeps = 2
    dtime = 2.0
    rayleigh_damping_height = 50000
    nflatlev = 30
    nflat_gradp = 59

    # diffusion configuration
    diffusion_type = DiffusionType.SMAGORINSKY_4TH_ORDER  # 5
    hdiff_w = True
    hdiff_vn = True
    zdiffu_t = True  # Setting this to true triggers running stencil 15 with the boolean masks.
    type_t_diffu = 2
    type_vn_diffu = 1
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    hdiff_temp = True

    # input data - numpy
    rng = xp.random.default_rng()

    vct_a = rng.uniform(
        low=0, high=75000, size=(num_levels,)
    )  # has to be from 0 to 75000, must have larger values than rayleigh damping height
    theta_ref_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    wgtfac_c = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    e_bln_c_s = rng.uniform(low=0, high=1, size=(num_cells, num_c2e))
    geofac_div = rng.uniform(low=0, high=1, size=(num_cells, num_c2e))
    geofac_grg_x = rng.uniform(low=0, high=1, size=(num_cells, num_c2ec2o))
    geofac_grg_y = rng.uniform(low=0, high=1, size=(num_cells, num_c2ec2o))
    geofac_n2s = rng.uniform(low=0, high=1, size=(num_cells, num_c2ec2o))
    nudgecoeff_e = xp.zeros((num_edges,))
    rbf_coeff_1 = rng.uniform(low=0, high=1, size=(num_vertices, num_v2e))
    rbf_coeff_2 = rng.uniform(low=0, high=1, size=(num_vertices, num_v2e))
    dwdx = xp.zeros((num_cells, num_levels))
    dwdy = xp.zeros((num_cells, num_levels))
    hdef_ic = xp.zeros((num_cells, num_levels + 1))
    div_ic = xp.zeros((num_cells, num_levels + 1))
    w = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    vn = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    exner = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    theta_v = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    rho = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    dual_normal_cell_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    dual_normal_cell_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    dual_normal_vert_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    dual_normal_vert_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    primal_normal_cell_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    primal_normal_cell_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    primal_normal_vert_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    primal_normal_vert_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    tangent_orientation = rng.uniform(low=0, high=1, size=(num_edges))
    inverse_primal_edge_lengths = rng.uniform(low=0, high=1, size=(num_edges))
    inv_dual_edge_length = rng.uniform(low=0, high=1, size=(num_edges))
    inv_vert_vert_length = rng.uniform(low=0, high=1, size=(num_edges))
    edge_areas = rng.uniform(low=0, high=1, size=(num_edges))
    f_e = rng.uniform(low=0, high=1, size=(num_edges))
    cell_areas = rng.uniform(low=0, high=1, size=(num_cells))
    zd_diffcoef = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    zd_vertoffset = xp.round(
        rng.uniform(low=0, high=1, size=(num_cells, num_c2e2c, num_levels))
    ).astype(xp.int32)
    zd_intcoef = rng.uniform(low=0, high=1, size=(num_cells, num_c2e2c, num_levels))

    # Create a boolean array based on a condition
    mask_hdiff = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    mask_hdiff = mask_hdiff < 0.5

    # input data - gt4py fields
    theta_ref_mc = as_field((CellDim, KDim), theta_ref_mc)
    wgtfac_c = as_field((CellDim, KDim), wgtfac_c)
    vct_a = as_field((KDim,), vct_a)
    e_bln_c_s = as_field((CellDim, C2EDim), e_bln_c_s)
    geofac_div = as_field((CellDim, C2EDim), geofac_div)
    geofac_grg_x = as_field((CellDim, C2E2CODim), geofac_grg_x)
    geofac_grg_y = as_field((CellDim, C2E2CODim), geofac_grg_y)
    geofac_n2s = as_field((CellDim, C2E2CODim), geofac_n2s)
    nudgecoeff_e = as_field((EdgeDim,), nudgecoeff_e)
    rbf_coeff_1 = as_field((VertexDim, V2EDim), rbf_coeff_1)
    rbf_coeff_2 = as_field((VertexDim, V2EDim), rbf_coeff_2)
    dwdx = as_field((CellDim, KDim), dwdx)
    dwdy = as_field((CellDim, KDim), dwdy)
    hdef_ic = as_field((CellDim, KDim), hdef_ic)
    div_ic = as_field((CellDim, KDim), div_ic)
    mask_hdiff = as_field((CellDim, KDim), mask_hdiff)
    zd_diffcoef = as_field((CellDim, KDim), zd_diffcoef)
    zd_vertoffset = as_field((CellDim, C2E2CDim, KDim), zd_vertoffset)
    zd_intcoef = as_field((CellDim, C2E2CDim, KDim), zd_intcoef)
    w = as_field((CellDim, KDim), w)
    vn = as_field((EdgeDim, KDim), vn)
    exner = as_field((CellDim, KDim), exner)
    theta_v = as_field((CellDim, KDim), theta_v)
    rho = as_field((CellDim, KDim), rho)
    dual_normal_cell_x = as_field((EdgeDim, E2CDim), dual_normal_cell_x)
    dual_normal_cell_y = as_field((EdgeDim, E2CDim), dual_normal_cell_y)
    dual_normal_vert_x = as_field((EdgeDim, E2C2VDim), dual_normal_vert_x)
    dual_normal_vert_y = as_field((EdgeDim, E2C2VDim), dual_normal_vert_y)
    primal_normal_cell_x = as_field((EdgeDim, E2CDim), primal_normal_cell_x)
    primal_normal_cell_y = as_field((EdgeDim, E2CDim), primal_normal_cell_y)
    primal_normal_vert_x = as_field((EdgeDim, E2C2VDim), primal_normal_vert_x)
    primal_normal_vert_y = as_field((EdgeDim, E2C2VDim), primal_normal_vert_y)
    tangent_orientation = as_field((EdgeDim,), tangent_orientation)
    inverse_primal_edge_lengths = as_field((EdgeDim,), inverse_primal_edge_lengths)
    inv_dual_edge_length = as_field((EdgeDim,), inv_dual_edge_length)
    inv_vert_vert_length = as_field((EdgeDim,), inv_vert_vert_length)
    edge_areas = as_field((EdgeDim,), edge_areas)
    f_e = as_field((EdgeDim,), f_e)
    cell_areas = as_field((CellDim,), cell_areas)

    diffusion_init(
        vct_a=vct_a,
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        e_bln_c_s=e_bln_c_s,
        geofac_div=geofac_div,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        geofac_n2s=geofac_n2s,
        nudgecoeff_e=nudgecoeff_e,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        num_levels=num_levels,
        mean_cell_area=mean_cell_area,
        ndyn_substeps=ndyn_substeps,
        rayleigh_damping_height=rayleigh_damping_height,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        diffusion_type=diffusion_type,
        hdiff_w=hdiff_w,
        hdiff_vn=hdiff_vn,
        zdiffu_t=zdiffu_t,
        type_t_diffu=type_t_diffu,
        type_vn_diffu=type_vn_diffu,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        smagorinski_scaling_factor=smagorinski_scaling_factor,
        hdiff_temp=hdiff_temp,
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        edge_areas=edge_areas,
        f_e=f_e,
        cell_areas=cell_areas,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        mask_hdiff=mask_hdiff,
        zd_diffcoef=zd_diffcoef,
        zd_vertoffset=zd_vertoffset,
        zd_intcoef=zd_intcoef,
    )

    diffusion_run(
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
        dtime=dtime,
        linit=False,
    )
