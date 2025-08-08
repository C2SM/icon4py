# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.driver.testcases import utils as testcases_utils


def load_channel_data(
    num_cells: int,
    num_edges: int,
    num_levels: int,
    full_level_heights: data_alloc.NDArray,
    wgtfac_c: data_alloc.NDArray,
    ddqz_z_half: data_alloc.NDArray,
    exner_ref_mc: data_alloc.NDArray,
    d_exner_dz_ref_ic: data_alloc.NDArray,
    theta_ref_mc: data_alloc.NDArray,
    theta_ref_ic: data_alloc.NDArray,
    geopot: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    mask: data_alloc.NDArray,
    backend: gtx_backend.Backend,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:

    xp = data_alloc.import_array_ns(backend)

    # Lee & Moser data: Re_tau = 5200
    data = xp.loadtxt("../python-scripts/data/LeeMoser_chan5200.mean", skiprows=72)
    #
    nh_t0 = 300.0
    nh_brunt_vais = 0.0
    channel_height = 100

    LM_y = data[:,0]
    LM_u = data[:,2] * 4.14872e-02 # <U> * u_tau (that's how it's normalized in the file)

    # Rescale to the ICON grid and mirror y to full channel height
    LM_y = LM_y * channel_height / 2
    LM_y = xp.concatenate((LM_y, channel_height - LM_y[::-1]), axis=0)
    LM_u = xp.concatenate((LM_u,                  LM_u[::-1]), axis=0)

    # Interpolate LM_u onto the ICON grid
    nh_u0 = xp.zeros((num_edges, num_levels), dtype=float)
    for j in range(num_levels):
        LM_j = xp.argmin(xp.abs(LM_y - full_level_heights[j]))
        nh_u0[:, j] = LM_u[LM_j] + xp.random.normal(loc=0, scale=0.05, size=num_edges)

    u = xp.where(mask, nh_u0, 0.0)
    vn_ndarray = u * primal_normal_x

    w_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=float)

    #---------------------------------------------------------------------------
    # The following is from the Gauss3D experiment

    theta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    exner_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    rho_ndarray = xp.zeros((num_cells, num_levels), dtype=float)

    # Vertical temperature profile
    for k_index in range(num_levels - 1, -1, -1):
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
        # profile of theta is explicitly given
        theta_v_ndarray[:, k_index] = nh_t0 * xp.exp(z_help)

    # Lower boundary condition for exner pressure
    if nh_brunt_vais != 0.0:
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_ndarray[:, num_levels - 1] = (
            phy_const.GRAV / nh_brunt_vais
        ) ** 2 / nh_t0 / phy_const.CPD * (xp.exp(-z_help) - 1.0) + 1.0
    else:
        exner_ndarray[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / phy_const.CPD / nh_t0

    # Compute hydrostatically balanced exner, by integrating the (discretized!)
    # 3rd equation of motion under the assumption thetav=const.
    rho_ndarray, exner_ndarray = testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho_ndarray,
        exner_ndarray,
        theta_v_ndarray,
        num_levels,
    )

    vn = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray, allocator=backend)
    w = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray, allocator=backend)
    exner = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray, allocator=backend)
    rho = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray, allocator=backend)
    theta_v = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray, allocator=backend)

    return vn, w, exner, rho, theta_v
