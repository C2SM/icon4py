# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
from gt4py.next import where

import icon4py.model.common.field_type_aliases as fa
import icon4py.model.common.math.projection as proj
import icon4py.model.common.type_alias as ta
from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import C2E, V2E


def compute_c_lin_e(
    edge_cell_length: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    owner_mask: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute E2C average inverse distance.

    Args:
        edge_cell_length: numpy array, representing a Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        inv_dual_edge_length: inverse dual edge length, numpy array representing a Field[gtx.Dims[EdgeDim], ta.wpfloat]
        owner_mask: numpy array, representing a Field[gtx.Dims[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        horizontal_start: start index of the 2nd boundary line: c_lin_e is not calculated for the first boundary layer

    Returns: c_lin_e: numpy array, representing Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]

    """
    c_lin_e_ = edge_cell_length[:, 1] * inv_dual_edge_length
    c_lin_e = np.transpose(np.vstack((c_lin_e_, (1.0 - c_lin_e_))))
    c_lin_e[0:horizontal_start, :] = 0.0
    mask = np.transpose(np.tile(owner_mask, (2, 1)))
    return np.where(mask, c_lin_e, 0.0)


@gtx.field_operator
def compute_geofac_div(
    primal_edge_length: fa.EdgeField[ta.wpfloat],
    edge_orientation: gtx.Field[[dims.CellDim, dims.C2EDim], ta.wpfloat],
    area: fa.CellField[ta.wpfloat],
) -> gtx.Field[[dims.CellDim, dims.C2EDim], ta.wpfloat]:
    """
    Compute geometrical factor for divergence.

    Args:
        primal_edge_length:
        edge_orientation:
        area:

    Returns:
    """
    geofac_div = primal_edge_length(C2E) * edge_orientation / area
    return geofac_div


@gtx.field_operator
def compute_geofac_rot(
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    edge_orientation: gtx.Field[[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    dual_area: fa.VertexField[ta.wpfloat],
    owner_mask: fa.VertexField[bool],
) -> gtx.Field[[dims.VertexDim, dims.V2EDim], ta.wpfloat]:
    """
    Compute geometrical factor for curl.

    Args:
        dual_edge_length:
        edge_orientation:
        dual_area:
        owner_mask:

    Returns:
    """
    geofac_rot = where(owner_mask, dual_edge_length(V2E) * edge_orientation / dual_area, 0.0)
    return geofac_rot


def compute_geofac_n2s(
    dual_edge_length: np.ndarray,
    geofac_div: np.ndarray,
    c2e: np.ndarray,
    e2c: np.ndarray,
    c2e2c: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute geometric factor for nabla2-scalar.

    Args:
        dual_edge_length: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        geofac_div: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        horizontal_start:

    Returns:
        geometric factor for nabla2-scalar, Field[CellDim, C2E2CODim]
    """
    llb = horizontal_start
    geofac_n2s = np.zeros([c2e.shape[0], 4])
    index = np.transpose(
        np.vstack(
            (
                np.arange(c2e.shape[0]),
                np.arange(c2e.shape[0]),
                np.arange(c2e.shape[0]),
            )
        )
    )
    mask = e2c[c2e, 0] == index
    geofac_n2s[llb:, 0] = geofac_n2s[llb:, 0] - np.sum(
        mask[llb:] * (geofac_div / dual_edge_length[c2e])[llb:], axis=1
    )
    mask = e2c[c2e, 1] == index
    geofac_n2s[llb:, 0] = geofac_n2s[llb:, 0] + np.sum(
        mask[llb:] * (geofac_div / dual_edge_length[c2e])[llb:], axis=1
    )
    mask = e2c[c2e, 0] == c2e2c
    geofac_n2s[llb:, 1:] = (
        geofac_n2s[llb:, 1:] - mask[llb:, :] * (geofac_div / dual_edge_length[c2e])[llb:, :]
    )
    mask = e2c[c2e, 1] == c2e2c
    geofac_n2s[llb:, 1:] = (
        geofac_n2s[llb:, 1:] + mask[llb:, :] * (geofac_div / dual_edge_length[c2e])[llb:, :]
    )
    return geofac_n2s


def compute_primal_normal_ec(
    primal_normal_cell_x: np.ndarray,
    primal_normal_cell_y: np.ndarray,
    owner_mask: np.ndarray,
    c2e: np.ndarray,
    e2c: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute primal_normal_ec.

    Args:
        primal_normal_cell_x: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        primal_normal_cell_y: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        horizontal_start:

    Returns:
        primal_normal_ec: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim, 2], ta.wpfloat]
    """
    llb = horizontal_start
    primal_normal_ec = np.zeros([c2e.shape[0], c2e.shape[1], 2])
    index = np.transpose(
        np.vstack(
            (
                np.arange(c2e.shape[0]),
                np.arange(c2e.shape[0]),
                np.arange(c2e.shape[0]),
            )
        )
    )
    for i in range(2):
        mask = e2c[c2e, i] == index
        primal_normal_ec[llb:, :, 0] = primal_normal_ec[llb:, :, 0] + np.where(
            owner_mask, mask[llb:, :] * primal_normal_cell_x[c2e[llb:], i], 0.0
        )
        primal_normal_ec[llb:, :, 1] = primal_normal_ec[llb:, :, 1] + np.where(
            owner_mask, mask[llb:, :] * primal_normal_cell_y[c2e[llb:], i], 0.0
        )
    return primal_normal_ec


def compute_geofac_grg(
    primal_normal_ec: np.ndarray,
    geofac_div: np.ndarray,
    c_lin_e: np.ndarray,
    c2e: np.ndarray,
    e2c: np.ndarray,
    c2e2c: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute geometrical factor for Green-Gauss gradient.

    Args:
        primal_normal_ec: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim, 2], ta.wpfloat]
        geofac_div: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        c_lin_e: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        horizontal_start:

    Returns:
        geofac_grg: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim + 1, 2], ta.wpfloat]
    """
    llb = horizontal_start
    num_cells = c2e.shape[0]
    geofac_grg = np.zeros([num_cells, c2e.shape[1] + 1, primal_normal_ec.shape[2]])
    index = np.transpose(
        np.vstack(
            (
                np.arange(num_cells),
                np.arange(num_cells),
                np.arange(num_cells),
            )
        )
    )
    for k in range(e2c.shape[1]):
        mask = e2c[c2e, k] == index
        for i in range(primal_normal_ec.shape[2]):
            for j in range(c2e.shape[1]):
                geofac_grg[llb:, 0, i] = (
                    geofac_grg[llb:, 0, i]
                    + mask[llb:, j]
                    * (primal_normal_ec[:, :, i] * geofac_div * c_lin_e[c2e, k])[llb:, j]
                )
    for k in range(e2c.shape[1]):
        mask = e2c[c2e, k] == c2e2c
        for i in range(primal_normal_ec.shape[2]):
            for j in range(c2e.shape[1]):
                geofac_grg[llb:, 1 + j, i] = (
                    geofac_grg[llb:, 1 + j, i]
                    + mask[llb:, j]
                    * (primal_normal_ec[:, :, i] * geofac_div * c_lin_e[c2e, k])[llb:, j]
                )
    return geofac_grg


def compute_geofac_grdiv(
    geofac_div: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    owner_mask: np.ndarray,
    c2e: np.ndarray,
    e2c: np.ndarray,
    e2c2e: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute geometrical factor for gradient of divergence (triangles only).

    Args:
        geofac_div: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        inv_dual_edge_length: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        e2c2e: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EDim], gtx.int32]
        horizontal_start:

    Returns:
        geofac_grdiv: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EODim], ta.wpfloat]
    """
    llb = horizontal_start
    num_edges = e2c.shape[0]
    geofac_grdiv = np.zeros([num_edges, 1 + 2 * e2c.shape[1]])
    index = np.arange(llb, num_edges)
    for j in range(c2e.shape[1]):
        mask = np.where(c2e[e2c[llb:, 1], j] == index, owner_mask[llb:], False)
        geofac_grdiv[llb:, 0] = np.where(mask, geofac_div[e2c[llb:, 1], j], geofac_grdiv[llb:, 0])
    for j in range(c2e.shape[1]):
        mask = np.where(c2e[e2c[llb:, 0], j] == index, owner_mask[llb:], False)
        geofac_grdiv[llb:, 0] = np.where(
            mask,
            (geofac_grdiv[llb:, 0] - geofac_div[e2c[llb:, 0], j]) * inv_dual_edge_length[llb:],
            geofac_grdiv[llb:, 0],
        )
    for j in range(e2c.shape[1]):
        for k in range(c2e.shape[1]):
            mask = c2e[e2c[llb:, 0], k] == e2c2e[llb:, j]
            geofac_grdiv[llb:, e2c.shape[1] - 1 + j] = np.where(
                mask,
                -geofac_div[e2c[llb:, 0], k] * inv_dual_edge_length[llb:],
                geofac_grdiv[llb:, e2c.shape[1] - 1 + j],
            )
            mask = c2e[e2c[llb:, 1], k] == e2c2e[llb:, e2c.shape[1] + j]
            geofac_grdiv[llb:, 2 * e2c.shape[1] - 1 + j] = np.where(
                mask,
                geofac_div[e2c[llb:, 1], k] * inv_dual_edge_length[llb:],
                geofac_grdiv[llb:, 2 * e2c.shape[1] - 1 + j],
            )
    return geofac_grdiv


def rotate_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
    pollat: np.ndarray,
    pollon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (Compute rotation of lattitude and longitude.)

    Rotates latitude and longitude for more accurate computation
    of bilinear interpolation

    Args:
        lat: scalar or numpy array
        lon: scalar or numpy array
        pollat: scalar or numpy array
        pollon: scalar or numpy array

    Returns:
        rotlat:
        rotlon:
    """
    rotlat = np.arcsin(
        np.sin(lat) * np.sin(pollat) + np.cos(lat) * np.cos(pollat) * np.cos(lon - pollon)
    )
    rotlon = np.arctan2(
        np.cos(lat) * np.sin(lon - pollon),
        (np.cos(lat) * np.sin(pollat) * np.cos(lon - pollon) - np.sin(lat) * np.cos(pollat)),
    )

    return (rotlat, rotlon)


def weighting_factors(
    ytemp: np.ndarray,
    xtemp: np.ndarray,
    yloc: np.ndarray,
    xloc: np.ndarray,
    wgt_loc: ta.wpfloat,
) -> np.ndarray:
    """
        Compute weighting factors.
        The weighting factors are based on the requirement that sum(w(i)*x(i)) = 0
        and sum(w(i)*y(i)) = 0, which ensures that linear horizontal gradients
        are not aliased into a checkerboard pattern between upward- and downward
        directed cells. The third condition is sum(w(i)) = 1., and the weight
        of the local point is 0.5 (see above). Analytical elimination yields...

    # TODO (Andreas J) computation different for Torus grids see mo_intp_coeffs.f90
    # The function weighting_factors does not exist in the Fortran code, the
    # Fortran is organised differently with code duplication

        Args:
            ytemp:  \\   numpy array of size [[3, flexible], ta.wpfloat]
            xtemp:  //
            yloc:   \\   numpy array of size [[flexible], ta.wpfloat]
            xloc:   //
            wgt_loc:

        Returns:
            wgt: numpy array of size [[3, flexible], ta.wpfloat]
    """
    pollat = np.where(yloc >= 0.0, yloc - np.pi * 0.5, yloc + np.pi * 0.5)
    pollon = xloc
    (yloc, xloc) = rotate_latlon(yloc, xloc, pollat, pollon)
    x = np.zeros([ytemp.shape[0], ytemp.shape[1]])
    y = np.zeros([ytemp.shape[0], ytemp.shape[1]])
    wgt = np.zeros([ytemp.shape[0], ytemp.shape[1]])

    for i in range(ytemp.shape[0]):
        (ytemp[i], xtemp[i]) = rotate_latlon(ytemp[i], xtemp[i], pollat, pollon)
        y[i] = ytemp[i] - yloc
        x[i] = xtemp[i] - xloc
        # This is needed when the date line is crossed
        x[i] = np.where(x[i] > 3.5, x[i] - np.pi * 2, x[i])
        x[i] = np.where(x[i] < -3.5, x[i] + np.pi * 2, x[i])

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

    return wgt


def compute_c_bln_avg(
    divavg_cntrwgt: ta.wpfloat,
    owner_mask: np.ndarray,
    c2e2c: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute bilinear cell average weight.

    Args:
        divavg_cntrwgt:
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, C2E2CDim], gtx.int32]
        lat: \\ numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        lon: //
        horizontal_start:

    Returns:
        c_bln_avg: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
    """
    llb = horizontal_start
    num_cells = c2e2c.shape[0]
    c_bln_avg = np.zeros([num_cells, 4])
    wgt_loc = divavg_cntrwgt
    yloc = np.zeros(num_cells)
    xloc = np.zeros(num_cells)
    yloc[llb:] = lat[llb:]
    xloc[llb:] = lon[llb:]
    ytemp = np.zeros([3, num_cells])
    xtemp = np.zeros([3, num_cells])

    for i in range(3):
        ytemp[i, llb:] = lat[c2e2c[llb:, i]]
        xtemp[i, llb:] = lon[c2e2c[llb:, i]]

    wgt = weighting_factors(
        ytemp[:, llb:],
        xtemp[:, llb:],
        yloc[llb:],
        xloc[llb:],
        wgt_loc,
    )

    c_bln_avg[llb:, 0] = np.where(owner_mask[llb:], wgt_loc, c_bln_avg[llb:, 0])
    for i in range(3):
        c_bln_avg[llb:, i + 1] = np.where(owner_mask[llb:], wgt[i], c_bln_avg[llb:, i + 1])

    return c_bln_avg


def compute_force_mass_conservation_to_c_bln_avg(
    c_bln_avg: np.ndarray,
    divavg_cntrwgt: ta.wpfloat,
    owner_mask: np.ndarray,
    c2e2c: np.ndarray,
    cell_areas: np.ndarray,
    horizontal_start: np.int32,
    horizontal_start_p3: np.int32,
    niter: np.ndarray = 1000,
) -> np.ndarray:
    """
    Compute the weighting coefficients for cell averaging with variable interpolation factors.

    The weighting factors are based on the requirement that sum(w(i)*x(i)) = 0

    and sum(w(i)*y(i)) = 0, which ensures that linear horizontal gradients are not aliased into a checkerboard pattern between upward- and downward directed cells. The third condition is sum(w(i)) = 1., and the weight of the local point is 0.5.

    force_mass_conservation_to_bilinear_cellavg_wgt
    Args:
        c_bln_avg: bilinear cellavg wgt, numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        divavg_cntrwgt:
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, C2E2CDim], gtx.int32]
        cell_areas: numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        horizontal_start:
        horizontal_start_p3:
        niter: number of iterations until convergence is assumed

    Returns:
        c_bln_avg: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
    """
    llb = horizontal_start
    llb2 = horizontal_start_p3
    num_cells = c2e2c.shape[0]
    index = np.arange(llb, num_cells)

    inv_neighbor_id = -np.ones([num_cells, 3], dtype=int)
    for i in range(3):
        for j in range(3):
            inv_neighbor_id[llb:, j] = np.where(
                np.logical_and(c2e2c[c2e2c[llb:, j], i] == index, c2e2c[llb:, j] >= 0),
                i,
                inv_neighbor_id[llb:, j],
            )

    relax_coeff = 0.46
    maxwgt_loc = divavg_cntrwgt + 0.003
    minwgt_loc = divavg_cntrwgt - 0.003
    # TODO: in this function halo cell exchanges (sync) are missing, here for inv_neighbor_id, but also within the iteration for several variables
    for iteration in range(niter):
        wgt_loc_sum = c_bln_avg[llb:, 0] * cell_areas[llb:] + np.sum(
            c_bln_avg[c2e2c[llb:], inv_neighbor_id[llb:] + 1] * cell_areas[c2e2c[llb:]], axis=1
        )
        resid = wgt_loc_sum[llb2 - llb :] / cell_areas[llb2:] - 1.0
        if iteration < niter - 1:
            c_bln_avg[llb2:, 0] = np.where(
                owner_mask[llb2:], c_bln_avg[llb2:, 0] - relax_coeff * resid, c_bln_avg[llb2:, 0]
            )
            for i in range(3):
                c_bln_avg[llb2:, i + 1] = np.where(
                    owner_mask[llb2:],
                    c_bln_avg[llb2:, i + 1] - relax_coeff * resid[c2e2c[llb2:, i] - llb2],
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
    c_bln_avg: np.ndarray,
    geofac_div: np.ndarray,
    owner_mask: np.ndarray,
    primal_cart_normal: np.ndarray,
    e2c: np.ndarray,
    c2e: np.ndarray,
    c2e2c: np.ndarray,
    e2c2e: np.ndarray,
    horizontal_start_p3: np.int32,
    horizontal_start_p4: np.int32,
) -> np.ndarray:
    """
    Compute edge flux average

    Args:
        c_bln_avg: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        geofac_div: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], bool]
        primal_cart_normal: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        e2c2e: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EDim], gtx.int32]
        horizontal_start_p3:
        horizontal_start_p4:

    Returns:
        e_flx_avg: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EODim], ta.wpfloat]
    """
    llb = 0
    e_flx_avg = np.zeros([e2c.shape[0], 5])
    index = np.arange(llb, c2e.shape[0])
    inv_neighbor_id = -np.ones([c2e.shape[0] - llb, 3], dtype=int)
    for i in range(c2e2c.shape[1]):
        for j in range(c2e2c.shape[1]):
            inv_neighbor_id[:, j] = np.where(
                np.logical_and(c2e2c[c2e2c[llb:, j], i] == index, c2e2c[llb:, j] >= 0),
                i,
                inv_neighbor_id[:, j],
            )

    llb = horizontal_start_p3
    index = np.arange(llb, e2c.shape[0])
    for j in range(c2e.shape[1]):
        for i in range(2):
            e_flx_avg[llb:, i + 1] = np.where(
                owner_mask[llb:],
                np.where(
                    c2e[e2c[llb:, 0], j] == index,
                    c_bln_avg[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], j] + 1]
                    * geofac_div[e2c[llb:, 0], np.mod(i + j + 1, 3)]
                    / geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], j]],
                    e_flx_avg[llb:, i + 1],
                ),
                e_flx_avg[llb:, i + 1],
            )
            e_flx_avg[llb:, i + 3] = np.where(
                owner_mask[llb:],
                np.where(
                    c2e[e2c[llb:, 0], j] == index,
                    c_bln_avg[e2c[llb:, 0], 1 + j]
                    * geofac_div[e2c[llb:, 1], np.mod(inv_neighbor_id[e2c[llb:, 0], j] + i + 1, 3)]
                    / geofac_div[e2c[llb:, 0], j],
                    e_flx_avg[llb:, i + 3],
                ),
                e_flx_avg[llb:, i + 3],
            )

    iie = -np.ones([e2c.shape[0], 4], dtype=int)
    iie[:, 0] = np.where(e2c[e2c2e[:, 0], 0] == e2c[:, 0], 2, -1)
    iie[:, 0] = np.where(
        np.logical_and(e2c[e2c2e[:, 0], 1] == e2c[:, 0], iie[:, 0] != 2), 4, iie[:, 0]
    )

    iie[:, 1] = np.where(e2c[e2c2e[:, 1], 0] == e2c[:, 0], 1, -1)
    iie[:, 1] = np.where(
        np.logical_and(e2c[e2c2e[:, 1], 1] == e2c[:, 0], iie[:, 1] != 1), 3, iie[:, 1]
    )

    iie[:, 2] = np.where(e2c[e2c2e[:, 2], 0] == e2c[:, 1], 2, -1)
    iie[:, 2] = np.where(
        np.logical_and(e2c[e2c2e[:, 2], 1] == e2c[:, 1], iie[:, 2] != 2), 4, iie[:, 2]
    )

    iie[:, 3] = np.where(e2c[e2c2e[:, 3], 0] == e2c[:, 1], 1, -1)
    iie[:, 3] = np.where(
        np.logical_and(e2c[e2c2e[:, 3], 1] == e2c[:, 1], iie[:, 3] != 1), 3, iie[:, 3]
    )

    llb = horizontal_start_p4
    index = np.arange(llb, e2c.shape[0])
    for i in range(c2e.shape[1]):
        e_flx_avg[llb:, 0] = np.where(
            owner_mask[llb:],
            np.where(
                c2e[e2c[llb:, 0], i] == index,
                0.5
                * (
                    (
                        geofac_div[e2c[llb:, 0], i] * c_bln_avg[e2c[llb:, 0], 0]
                        + geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i]]
                        * c_bln_avg[e2c[llb:, 0], i + 1]
                        - e_flx_avg[e2c2e[llb:, 0], iie[llb:, 0]]
                        * geofac_div[e2c[llb:, 0], np.mod(i + 1, 3)]
                        - e_flx_avg[e2c2e[llb:, 1], iie[llb:, 1]]
                        * geofac_div[e2c[llb:, 0], np.mod(i + 2, 3)]
                    )
                    / geofac_div[e2c[llb:, 0], i]
                    + (
                        geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i]]
                        * c_bln_avg[e2c[llb:, 1], 0]
                        + geofac_div[e2c[llb:, 0], i]
                        * c_bln_avg[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i] + 1]
                        - e_flx_avg[e2c2e[llb:, 2], iie[llb:, 2]]
                        * geofac_div[e2c[llb:, 1], np.mod(inv_neighbor_id[e2c[llb:, 0], i] + 1, 3)]
                        - e_flx_avg[e2c2e[llb:, 3], iie[llb:, 3]]
                        * geofac_div[e2c[llb:, 1], np.mod(inv_neighbor_id[e2c[llb:, 0], i] + 2, 3)]
                    )
                    / geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i]]
                ),
                e_flx_avg[llb:, 0],
            ),
            e_flx_avg[llb:, 0],
        )

    checksum = e_flx_avg[:, 0]
    for i in range(4):
        checksum = (
            checksum
            + np.sum(primal_cart_normal * primal_cart_normal[e2c2e[:, i], :], axis=1)
            * e_flx_avg[:, 1 + i]
        )

    for i in range(5):
        e_flx_avg[llb:, i] = np.where(
            owner_mask[llb:], e_flx_avg[llb:, i] / checksum[llb:], e_flx_avg[llb:, i]
        )

    return e_flx_avg


def compute_cells_aw_verts(
    dual_area: np.ndarray,
    edge_vert_length: np.ndarray,
    edge_cell_length: np.ndarray,
    owner_mask: np.ndarray,
    v2e: np.ndarray,
    e2v: np.ndarray,
    v2c: np.ndarray,
    e2c: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute cells_aw_verts.

    Args:
        dual_area: numpy array, representing a gtx.Field[gtx.Dims[VertexDim], ta.wpfloat]
        edge_vert_length: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        edge_cell_length: //
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[VertexDim], bool]
        v2e: numpy array, representing a gtx.Field[gtx.Dims[VertexDim, V2EDim], gtx.int32]
        e2v: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2VDim], gtx.int32]
        v2c: numpy array, representing a gtx.Field[gtx.Dims[VertexDim, V2CDim], gtx.int32]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        horizontal_start:

    Returns:
        aw_verts: numpy array, representing a gtx.Field[gtx.Dims[VertexDim, 6], ta.wpfloat]
    """
    llb = horizontal_start
    cells_aw_verts = np.zeros([v2e.shape[0], 6])
    for i in range(e2c.shape[1]):
        for je in range(v2e.shape[1]):
            for jc in range(v2c.shape[1]):
                mask = np.where(
                    np.logical_and(v2e[llb:, je] >= 0, e2c[v2e[llb:, je], i] == v2c[llb:, jc]),
                    owner_mask[llb:],
                    False,
                )
                index = np.arange(llb, v2e.shape[0])
                idx_ve = np.where(e2v[v2e[llb:, je], 0] == index, 0, 1)
                cells_aw_verts[llb:, jc] = np.where(
                    mask,
                    cells_aw_verts[llb:, jc]
                    + 0.5
                    / dual_area[llb:]
                    * edge_vert_length[v2e[llb:, je], idx_ve]
                    * edge_cell_length[v2e[llb:, je], i],
                    cells_aw_verts[llb:, jc],
                )
    return cells_aw_verts


def compute_e_bln_c_s(
    owner_mask: np.ndarray,
    c2e: np.ndarray,
    cells_lat: np.ndarray,
    cells_lon: np.ndarray,
    edges_lat: np.ndarray,
    edges_lon: np.ndarray,
) -> np.ndarray:
    """
    Compute e_bln_c_s.

    Args:
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        cells_lat: \\ numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        cells_lon: //
        edges_lat: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        edges_lon: //

    Returns:
        e_bln_c_s: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
    """
    llb = 0
    num_cells = c2e.shape[0]
    e_bln_c_s = np.zeros([num_cells, c2e.shape[1]])
    yloc = cells_lat[llb:]
    xloc = cells_lon[llb:]
    ytemp = np.zeros([c2e.shape[1], num_cells])
    xtemp = np.zeros([c2e.shape[1], num_cells])

    for i in range(ytemp.shape[0]):
        ytemp[i] = edges_lat[c2e[llb:, i]]
        xtemp[i] = edges_lon[c2e[llb:, i]]

    wgt = weighting_factors(
        ytemp,
        xtemp,
        yloc,
        xloc,
        0.0,
    )

    for i in range(wgt.shape[0]):
        e_bln_c_s[llb:, i] = np.where(owner_mask[llb:], wgt[i], e_bln_c_s[llb:, i])
    return e_bln_c_s


def compute_pos_on_tplane_e_x_y(
    grid_sphere_radius: ta.wpfloat,
    primal_normal_v1: np.ndarray,
    primal_normal_v2: np.ndarray,
    dual_normal_v1: np.ndarray,
    dual_normal_v2: np.ndarray,
    cells_lon: np.ndarray,
    cells_lat: np.ndarray,
    edges_lon: np.ndarray,
    edges_lat: np.ndarray,
    vertex_lon: np.ndarray,
    vertex_lat: np.ndarray,
    owner_mask: np.ndarray,
    e2c: np.ndarray,
    e2v: np.ndarray,
    e2c2e: np.ndarray,
    horizontal_start: np.int32,
) -> np.ndarray:
    """
    Compute pos_on_tplane_e_x_y.
    get geographical coordinates of edge midpoint
    get line and block indices of neighbour cells
    get geographical coordinates of first cell center
    projection first cell center into local \\lambda-\\Phi-system
    get geographical coordinates of second cell center
    projection second cell center into local \\lambda-\\Phi-system

    Args:
        grid_sphere_radius:
        primal_normal_v1: \\
        primal_normal_v2:  \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        dual_normal_v1:    //
        dual_normal_v2:   //
        cells_lon: \\ numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        cells_lat: //
        edges_lon: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        edges_lat: //
        vertex_lon: \\ numpy array, representing a gtx.Field[gtx.Dims[VertexDim], ta.wpfloat]
        vertex_lat: //
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], bool]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        e2v: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2VDim], gtx.int32]
        e2c2e: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EDim], gtx.int32]
        horizontal_start:

    Returns:
        pos_on_tplane_e_x: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        pos_on_tplane_e_y: //
    """
    llb = horizontal_start
    pos_on_tplane_e = np.zeros([e2c.shape[0], 8, 2])
    xyloc_plane_n1 = np.zeros([2, e2c.shape[0]])
    xyloc_plane_n2 = np.zeros([2, e2c.shape[0]])
    xyloc_plane_n1[0, llb:], xyloc_plane_n1[1, llb:] = proj.gnomonic_proj(
        edges_lon[llb:], edges_lat[llb:], cells_lon[e2c[llb:, 0]], cells_lat[e2c[llb:, 0]]
    )
    xyloc_plane_n2[0, llb:], xyloc_plane_n2[1, llb:] = proj.gnomonic_proj(
        edges_lon[llb:], edges_lat[llb:], cells_lon[e2c[llb:, 1]], cells_lat[e2c[llb:, 1]]
    )

    xyloc_quad = np.zeros([4, 2, e2c.shape[0]])
    xyloc_plane_quad = np.zeros([4, 2, e2c.shape[0]])
    for ne in range(4):
        xyloc_quad[ne, 0, llb:] = edges_lon[e2c2e[llb:, ne]]
        xyloc_quad[ne, 1, llb:] = edges_lat[e2c2e[llb:, ne]]
        xyloc_plane_quad[ne, 0, llb:], xyloc_plane_quad[ne, 1, llb:] = proj.gnomonic_proj(
            edges_lon[llb:], edges_lat[llb:], xyloc_quad[ne, 0, llb:], xyloc_quad[ne, 1, llb:]
        )

    xyloc_ve = np.zeros([2, 2, e2c.shape[0]])
    xyloc_plane_ve = np.zeros([2, 2, e2c.shape[0]])
    for nv in range(2):
        xyloc_ve[nv, 0, llb:] = vertex_lon[e2v[llb:, nv]]
        xyloc_ve[nv, 1, llb:] = vertex_lat[e2v[llb:, nv]]
        xyloc_plane_ve[nv, 0, llb:], xyloc_plane_ve[nv, 1, llb:] = proj.gnomonic_proj(
            edges_lon[llb:], edges_lat[llb:], xyloc_ve[nv, 0, llb:], xyloc_ve[nv, 1, llb:]
        )

    pos_on_tplane_e[llb:, 0, 0] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n1[0, llb:] * primal_normal_v1[llb:]
            + xyloc_plane_n1[1, llb:] * primal_normal_v2[llb:]
        ),
        pos_on_tplane_e[llb:, 0, 0],
    )
    pos_on_tplane_e[llb:, 0, 1] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n1[0, llb:] * dual_normal_v1[llb:]
            + xyloc_plane_n1[1, llb:] * dual_normal_v2[llb:]
        ),
        pos_on_tplane_e[llb:, 0, 1],
    )
    pos_on_tplane_e[llb:, 1, 0] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n2[0, llb:] * primal_normal_v1[llb:]
            + xyloc_plane_n2[1, llb:] * primal_normal_v2[llb:]
        ),
        pos_on_tplane_e[llb:, 1, 0],
    )
    pos_on_tplane_e[llb:, 1, 1] = np.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n2[0, llb:] * dual_normal_v1[llb:]
            + xyloc_plane_n2[1, llb:] * dual_normal_v2[llb:]
        ),
        pos_on_tplane_e[llb:, 1, 1],
    )

    for ne in range(4):
        pos_on_tplane_e[llb:, 2 + ne, 0] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_quad[ne, 0, llb:] * primal_normal_v1[llb:]
                + xyloc_plane_quad[ne, 1, llb:] * primal_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 2 + ne, 0],
        )
        pos_on_tplane_e[llb:, 2 + ne, 1] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_quad[ne, 0, llb:] * dual_normal_v1[llb:]
                + xyloc_plane_quad[ne, 1, llb:] * dual_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 2 + ne, 1],
        )

    for nv in range(2):
        pos_on_tplane_e[llb:, 6 + nv, 0] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_ve[nv, 0, llb:] * primal_normal_v1[llb:]
                + xyloc_plane_ve[nv, 1, llb:] * primal_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 6 + nv, 0],
        )
        pos_on_tplane_e[llb:, 6 + nv, 1] = np.where(
            owner_mask[llb:],
            grid_sphere_radius
            * (
                xyloc_plane_ve[nv, 0, llb:] * dual_normal_v1[llb:]
                + xyloc_plane_ve[nv, 1, llb:] * dual_normal_v2[llb:]
            ),
            pos_on_tplane_e[llb:, 6 + nv, 1],
        )

    pos_on_tplane_e_x = np.reshape(
        pos_on_tplane_e[:, 0:2, 0], (np.size(pos_on_tplane_e[:, 0:2, 0]))
    )
    pos_on_tplane_e_y = np.reshape(
        pos_on_tplane_e[:, 0:2, 1], (np.size(pos_on_tplane_e[:, 0:2, 1]))
    )
    return pos_on_tplane_e_x, pos_on_tplane_e_y
