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
from gt4py.next import where
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import C2E, V2E, C2EDim, CellDim, EdgeDim, V2EDim, VertexDim


def compute_c_lin_e(
    edge_cell_length: np.array,
    inv_dual_edge_length: np.array,
    owner_mask: np.array,
    second_boundary_layer_start_index: np.int32,
) -> np.array:
    """
    Compute E2C average inverse distance.

    Args:
        edge_cell_length: numpy array, representing a Field[[EdgeDim, E2CDim], float]
        inv_dual_edge_length: inverse dual edge length, numpy array representing a Field[[EdgeDim], float]
        owner_mask: numpy array, representing a Field[[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        second_boundary_layer_start_index: start index of the 2nd boundary line: c_lin_e is not calculated for the first boundary layer

    Returns: c_lin_e: numpy array  representing Field[[EdgeDim, E2CDim], float]

    """
    c_lin_e_ = edge_cell_length[:, 1] * inv_dual_edge_length
    c_lin_e = np.transpose(np.vstack((c_lin_e_, (1.0 - c_lin_e_))))
    c_lin_e[0:second_boundary_layer_start_index, :] = 0.0
    mask = np.transpose(np.tile(owner_mask, (2, 1)))
    return np.where(mask, c_lin_e, 0.0)


@field_operator
def compute_geofac_div(
    primal_edge_length: Field[[EdgeDim], float],
    edge_orientation: Field[[CellDim, C2EDim], float],
    area: Field[[CellDim], float],
) -> Field[[CellDim, C2EDim], float]:
    """
    Compute geofac_div.

    Args:
        primal_edge_length:
        edge_orientation:
        area:
    """
    geofac_div_ = primal_edge_length(C2E) * edge_orientation / area
    return geofac_div_


@field_operator
def compute_geofac_rot(
    dual_edge_length: Field[[EdgeDim], float],
    edge_orientation: Field[[VertexDim, V2EDim], float],
    dual_area: Field[[VertexDim], float],
    owner_mask: Field[[VertexDim], bool],
) -> Field[[VertexDim, V2EDim], float]:
    """
    Compute geofac_rot.

    Args:
        dual_edge_length:
        edge_orientation:
        dual_area:
        owner_mask:
    """
    geofac_rot_ = where(owner_mask, dual_edge_length(V2E) * edge_orientation / dual_area, 0.0)
    return geofac_rot_


def compute_geofac_div_np(
    geofac_div: np.array,
    primal_edge_length: np.array,
    edge_orientation: np.array,
    area: np.array,
    C2E: np.array,
) -> np.array:
    for i in range(3):
        geofac_div[:, i] = primal_edge_length[C2E[:, i]] * edge_orientation[:, i] / area
    return geofac_div


def compute_geofac_rot_np(
    geofac_rot: np.array,
    dual_edge_length: np.array,
    edge_orientation: np.array,
    dual_area: np.array,
    V2E: np.array,
    owner_mask: np.array,
    lateral_boundary_verts: np.array,
) -> np.array:
    llb = lateral_boundary_verts[0]
    for i in range(6):
        geofac_rot[llb:, i] = np.where(
            owner_mask[llb:],
            dual_edge_length[V2E[llb:, i]] * edge_orientation[llb:, i] / dual_area[llb:],
            0.0,
        )
    return geofac_rot


def compute_geofac_n2s(
    geofac_n2s: np.array,
    dual_edge_length: np.array,
    geofac_div: np.array,
    C2E: np.array,
    E2C: np.array,
    C2E2C: np.array,
    lateral_boundary: np.array,
) -> np.array:
    """
    Compute geofac_n2s.

    Args:
        geofac_n2s:
        dual_edge_length:
        geofac_div:
        C2E:
        E2C:
        C2E2C:
        lateral_boundary:
    """
    llb = lateral_boundary[0]
    index = np.transpose(
        np.vstack(
            (
                np.arange(lateral_boundary[1]),
                np.arange(lateral_boundary[1]),
                np.arange(lateral_boundary[1]),
            )
        )
    )
    mask = E2C[C2E, 0] == index
    geofac_n2s[llb:, 0] = geofac_n2s[llb:, 0] - np.sum(
        mask[llb:] * (geofac_div / dual_edge_length[C2E])[llb:], axis=1
    )
    mask = E2C[C2E, 1] == index
    geofac_n2s[llb:, 0] = geofac_n2s[llb:, 0] + np.sum(
        mask[llb:] * (geofac_div / dual_edge_length[C2E])[llb:], axis=1
    )
    mask = E2C[C2E, 0] == C2E2C
    geofac_n2s[llb:, 1:] = (
        geofac_n2s[llb:, 1:] - mask[llb:, :] * (geofac_div / dual_edge_length[C2E])[llb:, :]
    )
    mask = E2C[C2E, 1] == C2E2C
    geofac_n2s[llb:, 1:] = (
        geofac_n2s[llb:, 1:] + mask[llb:, :] * (geofac_div / dual_edge_length[C2E])[llb:, :]
    )
    return geofac_n2s


def compute_primal_normal_ec(
    primal_normal_ec: np.array,
    primal_normal_cell_x: np.array,
    primal_normal_cell_y: np.array,
    owner_mask: np.array,
    C2E: np.array,
    E2C: np.array,
    lateral_boundary: np.array,
) -> np.array:
    llb = lateral_boundary[0]
    index = np.transpose(
        np.vstack(
            (
                np.arange(lateral_boundary[1]),
                np.arange(lateral_boundary[1]),
                np.arange(lateral_boundary[1]),
            )
        )
    )
    for i in range(2):
        mask = E2C[C2E, i] == index
        primal_normal_ec[llb:, :, 0] = primal_normal_ec[llb:, :, 0] + np.where(
            owner_mask, mask[llb:, :] * primal_normal_cell_x[C2E[llb:], i], 0.0
        )
        primal_normal_ec[llb:, :, 1] = primal_normal_ec[llb:, :, 1] + np.where(
            owner_mask, mask[llb:, :] * primal_normal_cell_y[C2E[llb:], i], 0.0
        )
    return primal_normal_ec


def compute_geofac_grg(
    geofac_grg: np.array,
    primal_normal_ec: np.array,
    geofac_div: np.array,
    c_lin_e: np.array,
    C2E: np.array,
    E2C: np.array,
    C2E2C: np.array,
    lateral_boundary: np.array,
) -> np.array:
    """
    Compute geofac_grg.

    Args:
        geofac_grg:
        primal_normal_ec:
        geofac_div:
        c_lin_e:
        C2E:
        E2C:
        C2E2C:
        lateral_boundary:
    """
    llb = lateral_boundary[0]
    index = np.transpose(
        np.vstack(
            (
                np.arange(lateral_boundary[1]),
                np.arange(lateral_boundary[1]),
                np.arange(lateral_boundary[1]),
            )
        )
    )
    for k in range(2):
        mask = E2C[C2E, k] == index
        for i in range(2):
            for j in range(3):
                geofac_grg[llb:, 0, i] = (
                    geofac_grg[llb:, 0, i]
                    + mask[llb:, j]
                    * (primal_normal_ec[:, :, i] * geofac_div * c_lin_e[C2E, k])[llb:, j]
                )
    for k in range(2):
        mask = E2C[C2E, k] == C2E2C
        for i in range(2):
            for j in range(3):
                geofac_grg[llb:, 1 + j, i] = (
                    geofac_grg[llb:, 1 + j, i]
                    + mask[llb:, j]
                    * (primal_normal_ec[:, :, i] * geofac_div * c_lin_e[C2E, k])[llb:, j]
                )
    return geofac_grg


def compute_geofac_grdiv(
    geofac_grdiv: np.array,
    geofac_div: np.array,
    inv_dual_edge_length: np.array,
    owner_mask: np.array,
    C2E: np.array,
    E2C: np.array,
    C2E2C: np.array,
    E2C2E: np.array,
    lateral_boundary: np.array,
) -> np.array:
    """
    Compute geofac_grdiv.

    Args:
        geofac_grdiv:
        geofac_div:
        inv_dual_edge_length:
        owner_mask:
        C2E:
        E2C:
        C2E2C:
        lateral_boundary:
    """
    llb = lateral_boundary[0]
    index = np.arange(llb, lateral_boundary[1])
    for j in range(3):
        mask = np.where(C2E[E2C[llb:, 1], j] == index, owner_mask[llb:], False)
        geofac_grdiv[llb:, 0] = np.where(mask, geofac_div[E2C[llb:, 1], j], geofac_grdiv[llb:, 0])
    for j in range(3):
        mask = np.where(C2E[E2C[llb:, 0], j] == index, owner_mask[llb:], False)
        geofac_grdiv[llb:, 0] = np.where(
            mask,
            (geofac_grdiv[llb:, 0] - geofac_div[E2C[llb:, 0], j]) * inv_dual_edge_length[llb:],
            geofac_grdiv[llb:, 0],
        )
    for j in range(2):
        for k in range(3):
            mask = C2E[E2C[llb:, 0], k] == E2C2E[llb:, 0 + j]
            geofac_grdiv[llb:, 1 + j] = np.where(
                mask,
                -geofac_div[E2C[llb:, 0], k] * inv_dual_edge_length[llb:],
                geofac_grdiv[llb:, 1 + j],
            )
            mask = C2E[E2C[llb:, 1], k] == E2C2E[llb:, 2 + j]
            geofac_grdiv[llb:, 3 + j] = np.where(
                mask,
                geofac_div[E2C[llb:, 1], k] * inv_dual_edge_length[llb:],
                geofac_grdiv[llb:, 3 + j],
            )
    return geofac_grdiv


def rotate_latlon(
    lat: np.array,
    lon: np.array,
    pollat: np.array,
    pollon: np.array,
) -> (np.array, np.array):

    rotlat = np.arcsin(
        np.sin(lat) * np.sin(pollat) + np.cos(lat) * np.cos(pollat) * np.cos(lon - pollon)
    )
    rotlon = np.arctan2(
        np.cos(lat) * np.sin(lon - pollon),
        (np.cos(lat) * np.sin(pollat) * np.cos(lon - pollon) - np.sin(lat) * np.cos(pollat)),
    )

    return (rotlat, rotlon)


def compute_c_bln_avg_torus(
    c_bln_avg: np.array,
    divavg_cntrwgt: np.array,
    owner_mask: np.array,
    C2E2C: np.array,
    lateral_boundary: np.array,
    lat: np.array,
    lon: np.array,
) -> np.array:
    """
    Compute c_bln_avg_torus.

    calculate_uniform_bilinear_cellavg_wgt
    Args:
        c_bln_avg:
        divavg_cntrwgt:
    """
    local_weight = divavg_cntrwgt
    neigbor_weight = (1.0 - local_weight) / 3.0
    c_bln_avg[:, 0] = local_weight
    c_bln_avg[:, 1:3] = neigbor_weight
    return c_bln_avg


def compute_c_bln_avg(
    c_bln_avg: np.array,
    divavg_cntrwgt: np.array,
    owner_mask: np.array,
    C2E2C: np.array,
    lateral_boundary: np.array,
    lat: np.array,
    lon: np.array,
) -> np.array:
    """
    Compute c_bln_avg.

    calculate_bilinear_cellavg_wgt
    Args:
        c_bln_avg:
        divavg_cntrwgt:
    """
    llb = lateral_boundary[0]
    wgt_loc = divavg_cntrwgt
    yloc = lat[llb:]
    xloc = lon[llb:]
    pollat = np.where(yloc >= 0.0, yloc - np.pi * 0.5, yloc + np.pi * 0.5)
    pollon = xloc
    (yloc, xloc) = rotate_latlon(yloc, xloc, pollat, pollon)
    x = np.zeros([3, lateral_boundary[1] - llb])
    y = np.zeros([3, lateral_boundary[1] - llb])
    wgt = np.zeros([3, lateral_boundary[1] - llb])

    for i in range(3):
        ytemp = lat[C2E2C[llb:, i]]
        xtemp = lon[C2E2C[llb:, i]]
        (ytemp, xtemp) = rotate_latlon(ytemp, xtemp, pollat, pollon)
        y[i] = ytemp - yloc
        x[i] = xtemp - xloc
        # This is needed when the date line is crossed
        x[i] = np.where(x[i] > 3.5, x[i] - np.pi * 2, x[i])
        x[i] = np.where(x[i] < -3.5, x[i] + np.pi * 2, x[i])

    # The weighting factors are based on the requirement that sum(w(i)*x(i)) = 0
    # and sum(w(i)*y(i)) = 0, which ensures that linear horizontal gradients
    # are not aliased into a checkerboard pattern between upward- and downward
    # directed cells. The third condition is sum(w(i)) = 1., and the weight
    # of the local point is 0.5 (see above). Analytical elimination yields...

    mask = np.logical_and(abs(x[1] - x[0]) > 1.0e-11, abs(y[2] - y[0]) > 1.0e-11)
    wgt[2] = np.where(
        mask,
        1.0
        / ((y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
        * (1.0 - wgt_loc)
        * (-y[0] + x[0] * (y[1] - y[0]) / (x[1] - x[0])),
        1.0
        / ((y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[0]) / (x[2] - x[0]))
        * (1.0 - wgt_loc)
        * (-y[0] + x[0] * (y[2] - y[0]) / (x[2] - x[0])),
    )
    wgt[1] = np.where(
        mask,
        (-(1.0 - wgt_loc) * x[0] - wgt[2] * (x[2] - x[0])) / (x[1] - x[0]),
        (-(1.0 - wgt_loc) * x[0] - wgt[1] * (x[1] - x[0])) / (x[2] - x[0]),
    )
    wgt[1], wgt[2] = np.where(mask, (wgt[1], wgt[2]), (wgt[2], wgt[1]))
    wgt[0] = 1.0 - wgt_loc - wgt[1] - wgt[2]

    # Store results in ptr_patch%cells%avg_wgt
    c_bln_avg[llb:, 0] = np.where(owner_mask[llb:], wgt_loc, c_bln_avg[llb:, 0])
    for i in range(3):
        c_bln_avg[llb:, i + 1] = np.where(owner_mask[llb:], wgt[i], c_bln_avg[llb:, i + 1])

    return c_bln_avg


def compute_mass_conservation_c_bln_avg(
    c_bln_avg: np.array,
    divavg_cntrwgt: np.array,
    owner_mask: np.array,
    C2E2C: np.array,
    lateral_boundary: np.array,
    lat: np.array,
    lon: np.array,
    cell_areas: np.array,
    niter: np.array,
) -> np.array:
    """
    Compute c_bln_avg.

    calculate_bilinear_cellavg_wgt
    Args:
        c_bln_avg:
        divavg_cntrwgt:

    in this routine halo cell exchanges (sync) are missing
    """
    llb = lateral_boundary[0]
    llb2 = lateral_boundary[2]
    index = np.arange(llb, lateral_boundary[1])

    inv_neighbor_id = -np.ones([lateral_boundary[1] - llb, 3], dtype=int)
    for i in range(3):
        for j in range(3):
            inv_neighbor_id[:, j] = np.where(
                np.logical_and(C2E2C[C2E2C[llb:, j], i] == index, C2E2C[llb:, j] >= 0),
                i,
                inv_neighbor_id[:, j],
            )

    relax_coeff = 0.46
    maxwgt_loc = divavg_cntrwgt + 0.003
    minwgt_loc = divavg_cntrwgt - 0.003
    for iteration in range(niter):
        wgt_loc_sum = c_bln_avg[llb:, 0] * cell_areas[llb:] + np.sum(
            c_bln_avg[C2E2C[llb:], inv_neighbor_id + 1] * cell_areas[C2E2C[llb:]], axis=1
        )
        resid = wgt_loc_sum[llb2 - llb :] / cell_areas[llb2:] - 1.0
        if iteration < niter - 1:
            c_bln_avg[llb2:, 0] = np.where(
                owner_mask[llb2:], c_bln_avg[llb2:, 0] - relax_coeff * resid, c_bln_avg[llb2:, 0]
            )
            for i in range(3):
                c_bln_avg[llb2:, i + 1] = np.where(
                    owner_mask[llb2:],
                    c_bln_avg[llb2:, i + 1] - relax_coeff * resid[C2E2C[llb2:, i] - llb2],
                    c_bln_avg[llb2:, i + 1],
                )
            wgt_loc_sum = np.sum(c_bln_avg[llb2:], axis=1) - 1.0
            for i in range(4):
                c_bln_avg[llb2:, i] = c_bln_avg[llb2:, i] - 0.25 * wgt_loc_sum
            c_bln_avg[llb2:, 0] = np.where(
                owner_mask[llb2:],
                np.where(c_bln_avg[llb2:, 0] > minwgt_loc, c_bln_avg[llb2:, 0], minwgt_loc),
                c_bln_avg[llb2:, 0],
            )
            c_bln_avg[llb2:, 0] = np.where(
                owner_mask[llb2:],
                np.where(c_bln_avg[llb2:, 0] < maxwgt_loc, c_bln_avg[llb2:, 0], maxwgt_loc),
                c_bln_avg[llb2:, 0],
            )
        else:
            c_bln_avg[llb2:, 0] = np.where(
                owner_mask[llb2:], c_bln_avg[llb2:, 0] - resid, c_bln_avg[llb2:, 0]
            )
    return c_bln_avg


def compute_e_flx_avg(
    e_flx_avg: np.array,
    c_bln_avg: np.array,
    geofac_div: np.array,
    owner_mask: np.array,
    primal_cart_normal: np.array,
    E2C: np.array,
    C2E: np.array,
    C2E2C: np.array,
    E2C2E: np.array,
    lateral_boundary_cells: np.array,
    lateral_boundary_edges: np.array,
) -> np.array:
    llb = lateral_boundary_cells[0]
    llb = 0
    index = np.arange(llb, lateral_boundary_cells[1])
    inv_neighbor_id = -np.ones([lateral_boundary_cells[1] - llb, 3], dtype=int)
    for i in range(3):
        for j in range(3):
            inv_neighbor_id[:, j] = np.where(
                np.logical_and(C2E2C[C2E2C[llb:, j], i] == index, C2E2C[llb:, j] >= 0),
                i,
                inv_neighbor_id[:, j],
            )

    llb = lateral_boundary_edges[2]
    index = np.arange(llb, lateral_boundary_edges[1])
    for j in range(3):
        for i in range(2):
            e_flx_avg[llb:, i + 1] = np.where(
                owner_mask[llb:],
                np.where(
                    C2E[E2C[llb:, 0], j] == index,
                    c_bln_avg[E2C[llb:, 1], inv_neighbor_id[E2C[llb:, 0], j] + 1]
                    * geofac_div[E2C[llb:, 0], np.mod(i + j + 1, 3)]
                    / geofac_div[E2C[llb:, 1], inv_neighbor_id[E2C[llb:, 0], j]],
                    e_flx_avg[llb:, i + 1],
                ),
                e_flx_avg[llb:, i + 1],
            )
            e_flx_avg[llb:, i + 3] = np.where(
                owner_mask[llb:],
                np.where(
                    C2E[E2C[llb:, 0], j] == index,
                    c_bln_avg[E2C[llb:, 0], 1 + j]
                    * geofac_div[E2C[llb:, 1], np.mod(inv_neighbor_id[E2C[llb:, 0], j] + i + 1, 3)]
                    / geofac_div[E2C[llb:, 0], j],
                    e_flx_avg[llb:, i + 3],
                ),
                e_flx_avg[llb:, i + 3],
            )

    iie = -np.ones([lateral_boundary_edges[1], 4], dtype=int)
    iie[:, 0] = np.where(E2C[E2C2E[:, 0], 0] == E2C[:, 0], 2, -1)
    iie[:, 0] = np.where(
        np.logical_and(E2C[E2C2E[:, 0], 1] == E2C[:, 0], iie[:, 0] != 2), 4, iie[:, 0]
    )

    iie[:, 1] = np.where(E2C[E2C2E[:, 1], 0] == E2C[:, 0], 1, -1)
    iie[:, 1] = np.where(
        np.logical_and(E2C[E2C2E[:, 1], 1] == E2C[:, 0], iie[:, 1] != 1), 3, iie[:, 1]
    )

    iie[:, 2] = np.where(E2C[E2C2E[:, 2], 0] == E2C[:, 1], 2, -1)
    iie[:, 2] = np.where(
        np.logical_and(E2C[E2C2E[:, 2], 1] == E2C[:, 1], iie[:, 2] != 2), 4, iie[:, 2]
    )

    iie[:, 3] = np.where(E2C[E2C2E[:, 3], 0] == E2C[:, 1], 1, -1)
    iie[:, 3] = np.where(
        np.logical_and(E2C[E2C2E[:, 3], 1] == E2C[:, 1], iie[:, 3] != 1), 3, iie[:, 3]
    )

    llb = lateral_boundary_edges[3]
    index = np.arange(llb, lateral_boundary_edges[1])
    for i in range(3):
        e_flx_avg[llb:, 0] = np.where(
            owner_mask[llb:],
            np.where(
                C2E[E2C[llb:, 0], i] == index,
                0.5
                * (
                    (
                        geofac_div[E2C[llb:, 0], i] * c_bln_avg[E2C[llb:, 0], 0]
                        + geofac_div[E2C[llb:, 1], inv_neighbor_id[E2C[llb:, 0], i]]
                        * c_bln_avg[E2C[llb:, 0], i + 1]
                        - e_flx_avg[E2C2E[llb:, 0], iie[llb:, 0]]
                        * geofac_div[E2C[llb:, 0], np.mod(i + 1, 3)]
                        - e_flx_avg[E2C2E[llb:, 1], iie[llb:, 1]]
                        * geofac_div[E2C[llb:, 0], np.mod(i + 2, 3)]
                    )
                    / geofac_div[E2C[llb:, 0], i]
                    + (
                        geofac_div[E2C[llb:, 1], inv_neighbor_id[E2C[llb:, 0], i]]
                        * c_bln_avg[E2C[llb:, 1], 0]
                        + geofac_div[E2C[llb:, 0], i]
                        * c_bln_avg[E2C[llb:, 1], inv_neighbor_id[E2C[llb:, 0], i] + 1]
                        - e_flx_avg[E2C2E[llb:, 2], iie[llb:, 2]]
                        * geofac_div[E2C[llb:, 1], np.mod(inv_neighbor_id[E2C[llb:, 0], i] + 1, 3)]
                        - e_flx_avg[E2C2E[llb:, 3], iie[llb:, 3]]
                        * geofac_div[E2C[llb:, 1], np.mod(inv_neighbor_id[E2C[llb:, 0], i] + 2, 3)]
                    )
                    / geofac_div[E2C[llb:, 1], inv_neighbor_id[E2C[llb:, 0], i]]
                ),
                e_flx_avg[llb:, 0],
            ),
            e_flx_avg[llb:, 0],
        )

    checksum = e_flx_avg[:, 0]
    for i in range(4):
        checksum = (
            checksum
            + np.sum(primal_cart_normal * primal_cart_normal[E2C2E[:, i], :], axis=1)
            * e_flx_avg[:, 1 + i]
        )

    for i in range(5):
        e_flx_avg[llb:, i] = np.where(
            owner_mask[llb:], e_flx_avg[llb:, i] / checksum[llb:], e_flx_avg[llb:, i]
        )

    return e_flx_avg


def compute_cells_aw_verts(
    cells_aw_verts: np.array,
    dual_area: np.array,
    edge_vert_length: np.array,
    edge_cell_length: np.array,
    owner_mask: np.array,
    V2E: np.array,
    E2V: np.array,
    V2C: np.array,
    E2C: np.array,
    lateral_boundary_verts: np.array,
) -> np.array:
    llb = lateral_boundary_verts[0]
    for i in range(2):
        for je in range(6):
            for jc in range(6):
                mask = np.where(
                    np.logical_and(V2E[llb:, je] >= 0, E2C[V2E[llb:, je], i] == V2C[llb:, jc]),
                    owner_mask[llb:],
                    False,
                )
                index = np.arange(llb, lateral_boundary_verts[1])
                idx_ve = np.where(E2V[V2E[llb:, je], 0] == index, 0, 1)
                cells_aw_verts[llb:, jc] = np.where(
                    mask,
                    cells_aw_verts[llb:, jc]
                    + 0.5
                    / dual_area[llb:]
                    * edge_vert_length[V2E[llb:, je], idx_ve]
                    * edge_cell_length[V2E[llb:, je], i],
                    cells_aw_verts[llb:, jc],
                )
    return cells_aw_verts


def compute_e_bln_c_s(
    e_bln_c_s: np.array,
    owner_mask: np.array,
    C2E: np.array,
    cells_lat: np.array,
    cells_lon: np.array,
    edges_lat: np.array,
    edges_lon: np.array,
    lateral_boundary_cells: np.array,
) -> np.array:
    llb = 0

    yloc = cells_lat[llb:]
    xloc = cells_lon[llb:]
    pollat = np.where(yloc >= 0.0, yloc - np.pi * 0.5, yloc + np.pi * 0.5)
    pollon = xloc
    (yloc, xloc) = rotate_latlon(yloc, xloc, pollat, pollon)
    x = np.zeros([3, lateral_boundary_cells[1] - llb])
    y = np.zeros([3, lateral_boundary_cells[1] - llb])
    wgt = np.zeros([3, lateral_boundary_cells[1] - llb])

    for i in range(3):
        ytemp = edges_lat[C2E[llb:, i]]
        xtemp = edges_lon[C2E[llb:, i]]
        (ytemp, xtemp) = rotate_latlon(ytemp, xtemp, pollat, pollon)
        y[i] = ytemp - yloc
        x[i] = xtemp - xloc
        # This is needed when the date line is crossed
        x[i] = np.where(x[i] > 3.5, x[i] - np.pi * 2, x[i])
        x[i] = np.where(x[i] < -3.5, x[i] + np.pi * 2, x[i])

    # The weighting factors are based on the requirement that sum(w(i)*x(i)) = 0
    # and sum(w(i)*y(i)) = 0, which ensures that linear horizontal gradients
    # are not aliased into a checkerboard pattern between upward- and downward
    # directed cells. The third condition is sum(w(i)) = 1. Analytical elimination yields...

    mask = np.logical_and(abs(x[1] - x[0]) > 1.0e-11, abs(y[2] - y[0]) > 1.0e-11)
    wgt[2] = np.where(
        mask,
        1.0
        / ((y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
        * (-y[0] + x[0] * (y[1] - y[0]) / (x[1] - x[0])),
        1.0
        / ((y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[0]) / (x[2] - x[0]))
        * (-y[0] + x[0] * (y[2] - y[0]) / (x[2] - x[0])),
    )
    wgt[1] = np.where(
        mask,
        (-x[0] - wgt[2] * (x[2] - x[0])) / (x[1] - x[0]),
        (-x[0] - wgt[1] * (x[1] - x[0])) / (x[2] - x[0]),
    )
    wgt[1], wgt[2] = np.where(mask, (wgt[1], wgt[2]), (wgt[2], wgt[1]))
    wgt[0] = 1.0 - wgt[1] - wgt[2]

    # Store results in ptr_patch%cells%e_bln_c_s
    for i in range(3):
        e_bln_c_s[llb:, i] = np.where(owner_mask[llb:], wgt[i], e_bln_c_s[llb:, i])
    return e_bln_c_s


def gnomonic_proj(
    lon_c: np.array,
    lat_c: np.array,
    lon: np.array,
    lat: np.array,
) -> (np.array, np.array):
    #
    # !LITERATURE:
    # Map Projections: A Working Manual, Snyder, 1987, p. 165
    #
    #
    # REAL(wp), INTENT(in) :: lat_c, lon_c  ! center on tangent plane
    # REAL(wp), INTENT(in) :: lat, lon      ! point to be projected
    # REAL(wp), INTENT(out):: x, y          ! coordinates of projected point

    # REAL(wp) :: zk   ! scale factor perpendicular to the radius from the
    # center of the map
    # REAL(wp) :: cosc ! cosine of the angular distance of the given point
    # (lat,lon) from the center of projection

    # -----------------------------------------------------------------------

    cosc = np.sin(lat_c) * np.sin(lat) + np.cos(lat_c) * np.cos(lat) * np.cos(lon - lon_c)
    zk = 1.0 / cosc

    x = zk * np.cos(lat) * np.sin(lon - lon_c)
    y = zk * (np.cos(lat_c) * np.sin(lat) - np.sin(lat_c) * np.cos(lat) * np.cos(lon - lon_c))

    return x, y


def compute_pos_on_tplane_e(
    pos_on_tplane_e: np.array,
    grid_sphere_radius: np.array,
    primal_normal_v1: np.array,
    primal_normal_v2: np.array,
    dual_normal_v1: np.array,
    dual_normal_v2: np.array,
    cells_lon: np.array,
    cells_lat: np.array,
    edges_lon: np.array,
    edges_lat: np.array,
    vertex_lon: np.array,
    vertex_lat: np.array,
    owner_mask: np.array,
    E2C: np.array,
    E2V: np.array,
    E2C2E: np.array,
    lateral_boundary_edges,
) -> np.array:
    llb = lateral_boundary_edges[0]
    #     get geographical coordinates of edge midpoint
    #     get line and block indices of neighbour cells
    #     get geographical coordinates of first cell center
    #     projection first cell center into local \lambda-\Phi-system
    #     get geographical coordinates of second cell center
    #     projection second cell center into local \lambda-\Phi-system
    xyloc_plane_n1 = np.zeros([2, lateral_boundary_edges[1] - llb])
    xyloc_plane_n2 = np.zeros([2, lateral_boundary_edges[1] - llb])
    xyloc_plane_n1[0], xyloc_plane_n1[1] = gnomonic_proj(
        edges_lon[llb:], edges_lat[llb:], cells_lon[E2C[llb:, 0]], cells_lat[E2C[llb:, 0]]
    )
    xyloc_plane_n2[0], xyloc_plane_n2[1] = gnomonic_proj(
        edges_lon[llb:], edges_lat[llb:], cells_lon[E2C[llb:, 1]], cells_lat[E2C[llb:, 1]]
    )

    xyloc_quad = np.zeros([4, 2, lateral_boundary_edges[1] - llb])
    xyloc_plane_quad = np.zeros([4, 2, lateral_boundary_edges[1] - llb])
    for ne in range(4):
        xyloc_quad[ne, 0] = edges_lon[E2C2E[llb:, ne]]
        xyloc_quad[ne, 1] = edges_lat[E2C2E[llb:, ne]]
        xyloc_plane_quad[ne, 0], xyloc_plane_quad[ne, 1] = gnomonic_proj(
            edges_lon[llb:], edges_lat[llb:], xyloc_quad[ne, 0], xyloc_quad[ne, 1]
        )

    xyloc_ve = np.zeros([2, 2, lateral_boundary_edges[1] - llb])
    xyloc_plane_ve = np.zeros([2, 2, lateral_boundary_edges[1] - llb])
    for nv in range(2):
        xyloc_ve[nv, 0] = vertex_lon[E2V[llb:, nv]]
        xyloc_ve[nv, 1] = vertex_lat[E2V[llb:, nv]]
        xyloc_plane_ve[nv, 0], xyloc_plane_ve[nv, 1] = gnomonic_proj(
            edges_lon[llb:], edges_lat[llb:], xyloc_ve[nv, 0], xyloc_ve[nv, 1]
        )

    pos_on_tplane_e[llb:, 0, 0] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (xyloc_plane_n1[0] * primal_normal_v1[llb:] + xyloc_plane_n1[1] * primal_normal_v2[llb:]),
        pos_on_tplane_e[llb:, 0, 0],
    )
    pos_on_tplane_e[llb:, 0, 1] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (xyloc_plane_n1[0] * dual_normal_v1[llb:] + xyloc_plane_n1[1] * dual_normal_v2[llb:]),
        pos_on_tplane_e[llb:, 0, 1],
    )
    pos_on_tplane_e[llb:, 1, 0] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (xyloc_plane_n2[0] * primal_normal_v1[llb:] + xyloc_plane_n2[1] * primal_normal_v2[llb:]),
        pos_on_tplane_e[llb:, 1, 0],
    )
    pos_on_tplane_e[llb:, 1, 1] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (xyloc_plane_n2[0] * dual_normal_v1[llb:] + xyloc_plane_n2[1] * dual_normal_v2[llb:]),
        pos_on_tplane_e[llb:, 1, 1],
    )

    for ne in range(4):
        pos_on_tplane_e[llb:, 2 + ne, 0] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_quad[ne, 0] * primal_normal_v1[llb:]
                + xyloc_plane_quad[ne, 1] * primal_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 2 + ne, 0],
        )
        pos_on_tplane_e[llb:, 2 + ne, 1] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_quad[ne, 0] * dual_normal_v1[llb:]
                + xyloc_plane_quad[ne, 1] * dual_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 2 + ne, 1],
        )

    for nv in range(2):
        pos_on_tplane_e[llb:, 6 + nv, 0] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_ve[nv, 0] * primal_normal_v1[llb:]
                + xyloc_plane_ve[nv, 1] * primal_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 6 + nv, 0],
        )
        pos_on_tplane_e[llb:, 6 + nv, 1] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_ve[nv, 0] * dual_normal_v1[llb:]
                + xyloc_plane_ve[nv, 1] * dual_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 6 + nv, 1],
        )

    return pos_on_tplane_e


def compute_pos_on_tplane_e_x_y(
    pos_on_tplane_e_x: np.array,
    pos_on_tplane_e_y: np.array,
    pos_on_tplane_e: np.array,
) -> (np.array, np.array):
    pos_on_tplane_e_x = np.reshape(
        pos_on_tplane_e[:, 0:2, 0], (np.size(pos_on_tplane_e[:, 0:2, 0]))
    )
    pos_on_tplane_e_y = np.reshape(
        pos_on_tplane_e[:, 0:2, 1], (np.size(pos_on_tplane_e[:, 0:2, 1]))
    )
    return pos_on_tplane_e_x, pos_on_tplane_e_y
