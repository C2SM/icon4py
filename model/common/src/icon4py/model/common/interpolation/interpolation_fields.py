# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math

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
        edge_cell_length: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        inv_dual_edge_length: inverse dual edge length, numpy array representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        horizontal_start: start index of the 2nd boundary line: c_lin_e is not calculated for the first boundary layer

    Returns: c_lin_e: numpy array, representing gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]

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


def _compute_rbf_vec_idx_c(c2e2c, c2e, num_cells):
    rbf_vec_idx_c = np.zeros((9, num_cells)) # TODO: do not hard code 9
    rbf_vec_idx_c[0, :] = c2e[c2e2c[:, 0], 0]
    rbf_vec_idx_c[1, :] = c2e[c2e2c[:, 0], 1]
    rbf_vec_idx_c[2, :] = c2e[c2e2c[:, 0], 2]
    rbf_vec_idx_c[3, :] = c2e[c2e2c[:, 1], 0]
    rbf_vec_idx_c[4, :] = c2e[c2e2c[:, 1], 1]
    rbf_vec_idx_c[5, :] = c2e[c2e2c[:, 1], 2]
    rbf_vec_idx_c[6, :] = c2e[c2e2c[:, 2], 0]
    rbf_vec_idx_c[7, :] = c2e[c2e2c[:, 2], 1]
    rbf_vec_idx_c[8, :] = c2e[c2e2c[:, 2], 2]
    return rbf_vec_idx_c

def _gvec2cvec(p_gu, p_gv, p_long, p_lat):
    z_sln = math.sin(p_long)
    z_cln = math.cos(p_long)
    z_slt = math.sin(p_lat)
    z_clt = math.cos(p_lat)

    p_cu = z_sln * p_gu + z_slt * z_cln * p_gv
    p_cu = -1.0 * p_cu
    p_cv = z_cln * p_gu - z_slt * z_sln * p_gv
    p_cw = z_clt * p_gv
    return p_cu, p_cv, p_cw


def _compute_z_xn1_z_xn2(
    lon,
    lat,
    cartesian_center,
    owner_mask,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    upper_bound,
    num_cells
):
    z_nx1 = np.zeros((num_cells, 3))
    z_nx2 = np.zeros((num_cells, 3))

    for jc in range(lower_bound, upper_bound):
        #if not owner_mask[jc]: continue
        z_nx1[jc, 0], z_nx1[jc, 1], z_nx1[jc, 2] = _gvec2cvec(1., 0., lon[jc], lat[jc])
        z_norm = math.sqrt(np.dot(z_nx1[jc, :], z_nx1[jc, :]))
        z_nx1[jc, :] = 1. / z_norm * z_nx1[jc, :]
        z_nx2[jc, 0], z_nx2[jc, 1], z_nx2[jc, 2] = _gvec2cvec(0., 1., lon[jc], lat[jc])
        z_norm = math.sqrt(np.dot(z_nx2[jc, :], z_nx2[jc, :]))
        z_nx2[jc, :] = 1. / z_norm * z_nx2[jc, :]

    return z_nx1, z_nx2

def _compute_rbf_vec_scale_c(mean_characteristic_length):
    resol = mean_characteristic_length/1000.0
    rbf_vec_scale_c = 0.5 / (1. + 1.8 * math.log(2.5/resol) ** 3.75) if resol < 2.5 else 0.5
    rbf_vec_scale_c = rbf_vec_scale_c*(resol/0.125)**0.9 if resol <= 0.125 else rbf_vec_scale_c
    return rbf_vec_scale_c

def _compute_arc_length_v(p_x, p_y):
    z_lx = math.sqrt(np.dot(p_x, p_x))
    z_ly = math.sqrt(np.dot(p_y, p_y))

    z_cc = np.dot(p_x, p_y)/(z_lx*z_ly)

    if z_cc > 1.0: z_cc = 1.0
    if z_cc < -1.0: z_cc = -1.0

    p_arc = np.arccos(z_cc)

    return p_arc

def _compute_rhs1_rhs2(
    c2e,
    c2e2c,
    cartesian_center_c,
    cartesian_center_e,
    mean_characteristic_length,
    z_nx1,
    z_nx2,
    istencil,
    owner_mask,
    primal_cart_normal_x,
    jg,
    rbf_vec_kern_c,  # rbf_vec_kern_c from config
    lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
    num_cells
):
    rbf_vec_idx_c = _compute_rbf_vec_idx_c(c2e2c, c2e, num_cells)
    rbf_vec_dim_c = 9 # rbf_vec_dim_c from config
    z_rhs1 = np.zeros((num_cells, rbf_vec_dim_c))
    z_rhs2 = np.zeros((num_cells, rbf_vec_dim_c))
    z_rbfval = np.zeros((num_cells, rbf_vec_dim_c))
    z_nx3 = np.zeros((num_cells, 3))
    rbf_vec_scale_c = _compute_rbf_vec_scale_c(mean_characteristic_length)
    for je2 in range(rbf_vec_dim_c):
        for jc in range(lower_bound, num_cells):
            if not owner_mask[jc]: continue
            if je2 > istencil[jc]: continue
            cc_c = (cartesian_center_c[0][jc].ndarray,
                    cartesian_center_c[1][jc].ndarray,
                    cartesian_center_c[2][jc].ndarray)
            ile2 = rbf_vec_idx_c[je2, jc]
            cc_e2 = (cartesian_center_e[0][int(ile2)].ndarray,
             cartesian_center_e[1][int(ile2)].ndarray,
             cartesian_center_e[2][int(ile2)].ndarray)
            z_nx3[jc, 0] = primal_cart_normal_x[0][int(ile2)].ndarray
            z_nx3[jc, 1] = primal_cart_normal_x[1][int(ile2)].ndarray
            z_nx3[jc, 2] = primal_cart_normal_x[2][int(ile2)].ndarray
            z_dist = _compute_arc_length_v(cc_c, cc_e2)
            if rbf_vec_kern_c == 1:  # rbf_vec_kern_c from config
                z_rbfval[jc, je2] = _gaussi(z_dist, rbf_vec_scale_c)
            elif rbf_vec_kern_c == 3:  # rbf_vec_kern_c from config
                z_rbfval[jc, je2] = _inv_multiq(z_dist, rbf_vec_scale_c)

            z_rhs1[jc, je2] = z_rbfval[jc, je2] * np.dot(z_nx1[jc, :], z_nx3[jc, :])
            z_rhs2[jc, je2] = z_rbfval[jc, je2] * np.dot(z_nx2[jc, :], z_nx3[jc, :])

    return z_rhs1, z_rhs2

import cmath
def _compute_z_diag(
    k_dim,
    rbf_vec_dim_c,
    p_a,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    maxdim,
    num_cells
):
    # # non-vectorized version
    # p_diag = np.zeros((num_cells, rbf_vec_dim_c))
    # for jc in range(lower_bound, num_cells):
    #     for ji in range(int(k_dim[jc])):
    #         jj = ji
    #         z_sum = p_a[jc, ji, jj]
    #         for jk in reversed(range(ji-1)):
    #             z_sum = z_sum - p_a[jc, ji, jk] * p_a[jc, jj, jk]
    #             if z_sum < 0.:
    #                 a = 1
    #         p_diag[jc, ji] = math.sqrt(z_sum)
    #
    #         for jj in range(ji + 1, int(k_dim[jc])):
    #             z_sum = p_a[jc, ji, jj]
    #             for jk in reversed(range(ji-1)):
    #                 z_sum = z_sum - p_a[jc, ji, jk] * p_a[jc, jj, jk]
    #
    #             p_a[jc, jj, ji] = z_sum / p_diag[jc, ji]

    # vectorized version
    z_sum = np.zeros((num_cells))
    p_diag = np.zeros((num_cells, maxdim))
    for ji in range(maxdim):
        for jj in range(ji, maxdim):
            for jc in range(lower_bound, num_cells):
                if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                z_sum[jc] = p_a[jc, ji, jj]
            for jk in reversed(range(ji-1)):
                for jc in range(lower_bound, num_cells):
                    if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                    z_sum[jc] = z_sum[jc] - p_a[jc, ji, jk] * p_a[jc, jj, jk]
            if ji == jj:
                for jc in range(lower_bound, num_cells):
                    if (ji > k_dim[jc]): continue
                    p_diag[jc, ji] = cmath.sqrt(z_sum[jc])
            else:
                for jc in range(lower_bound, num_cells):
                    if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                    p_a[jc, jj, ji] = z_sum[jc] / p_diag[jc, ji]
    return p_diag

def _gaussi(p_x, p_scale):
    p_rbf_val = p_x / p_scale
    p_rbf_val = -1. * p_rbf_val * p_rbf_val
    p_rbf_val = np.exp(p_rbf_val)
    return p_rbf_val

def _inv_multiq(p_x, p_scale):
    p_rbf_val = p_x / p_scale
    p_rbf_val = p_rbf_val * p_rbf_val
    p_rbf_val = np.sqrt(1. + p_rbf_val)
    p_rbf_val = 1. / p_rbf_val
    return p_rbf_val

def _compute_z_rbfmat_istencil(
    c2e2c,
    c2e,
    cartesian_center_e,
    primal_cart_normal_x,
    mean_characteristic_length,
    owner_mask,
    rbf_vec_dim_c,
    rbf_vec_kern_c,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    upper_bound,
    num_cells
):
    jg = 1
    rbf_vec_idx_c = _compute_rbf_vec_idx_c(c2e2c, c2e, num_cells)
    rbf_vec_scale_c = _compute_rbf_vec_scale_c(mean_characteristic_length)
    rbf_vec_stencil_c = np.zeros((num_cells,))
    z_rbfmat = np.zeros((num_cells, rbf_vec_dim_c, rbf_vec_dim_c))
    z_nx1 = np.zeros((num_cells, 3))
    z_nx2 = np.zeros((num_cells, 3))
    istencil = np.zeros(num_cells)
    for je1 in list(range(rbf_vec_dim_c)):
        for je2 in range(je1+1):
            for jc in range(lower_bound, upper_bound):
                if jc == 403:
                    a = 1
                # if not owner_mask[jc]:
                #     istencil[jc] = 0
                #     continue
                rbf_vec_stencil_c[jc] = len(np.argwhere(rbf_vec_idx_c[:, jc] != 0))
                istencil[jc] = rbf_vec_stencil_c[jc]
                ile1 = rbf_vec_idx_c[je1, jc]
                ile2 = rbf_vec_idx_c[je2, jc]
                if (je1 > istencil[jc] or je2 > istencil[jc]):
                    continue
                cc_e1 = (cartesian_center_e[0][int(ile1)].ndarray,
                         cartesian_center_e[1][int(ile1)].ndarray,
                         cartesian_center_e[2][int(ile1)].ndarray)
                cc_e2 = (cartesian_center_e[0][int(ile2)].ndarray,
                         cartesian_center_e[1][int(ile2)].ndarray,
                         cartesian_center_e[2][int(ile2)].ndarray)
                z_nx1[jc, 0] = primal_cart_normal_x[0][int(ile1)].ndarray
                z_nx1[jc, 1] = primal_cart_normal_x[1][int(ile1)].ndarray
                z_nx1[jc, 2] = primal_cart_normal_x[2][int(ile1)].ndarray
                z_nx2[jc, 0] = primal_cart_normal_x[0][int(ile2)].ndarray
                z_nx2[jc, 1] = primal_cart_normal_x[1][int(ile2)].ndarray
                z_nx2[jc, 2] = primal_cart_normal_x[2][int(ile2)].ndarray
                z_nxprod = np.dot(z_nx1[jc, :], z_nx2[jc, :])
                z_dist = _compute_arc_length_v(cc_e1, cc_e2)
                if rbf_vec_kern_c == 1:
                    z_rbfmat[jc, je1, je2] = z_nxprod * _gaussi(z_dist, rbf_vec_scale_c) #[max(jg, 0)])
                elif rbf_vec_kern_c == 3:
                    z_rbfmat[jc, je1, je2] = z_nxprod * _inv_multiq(z_dist, rbf_vec_scale_c) #[max(jg, 0)])

                if je1 > je2:
                    z_rbfmat[jc, je2, je1] = z_rbfmat[jc, je1, je2]

    return z_rbfmat, istencil

def _compute_rbf_vec_coeff(
    k_dim,
    p_a,
    p_diag,
    p_b,
    maxdim,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    num_cells
):
    # non-vectorized version
    # p_x = np.zeros((maxdim, num_cells))
    # #z_sum = np.zeros(num_cells)
    # for jc in range(lower_bound, num_cells):
    #     for ji in range(int(k_dim[jc])):
    #         z_sum = p_b[jc, ji]
    #         # z_sum[lower_bound:] = p_b[lower_bound:, ji]
    #         for jj in reversed(range(ji-1)):
    #             # z_sum[lower_bound:] = z_sum[lower_bound:] - p_a[lower_bound:, ji, jj] * p_x[jj, lower_bound:]
    #             z_sum = z_sum - p_a[jc, ji, jj] * p_x[jj, jc]
    #
    #         if p_diag[jc, ji] == 0:
    #             a = 1
    #
    #         p_x[ji, jc] = z_sum / p_diag[jc, ji]
    #
    #     for ji in reversed(range(int(k_dim[jc]))):
    #         z_sum = p_x[ji, jc]
    #         # z_sum[lower_bound:] = p_x[ji, lower_bound:]
    #         # z_sum[lower_bound:] = z_sum[lower_bound:] - p_a[lower_bound:, ji + 1: maxdim, ji] * p_x[ji + 1: maxdim, lower_bound:]
    #         for jj in range(ji+1, int(k_dim[jc])):
    #             z_sum = z_sum - p_a[jc, jj, ji] * p_x[jj, jc]
    #
    #         if p_diag[jc, ji] == 0:
    #             a = 1
    #
    #         p_x[ji, jc] = z_sum / p_diag[jc, ji]

    # vectorized version
    p_x = np.zeros((maxdim, num_cells))
    z_sum = np.zeros(num_cells)
    for ji in range(maxdim):
        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            z_sum[jc] = p_b[jc, ji]
        for jj in reversed(range(ji-1)):
            for jc in range(lower_bound, num_cells):
                if ji > k_dim[jc]: continue
                z_sum[jc] = z_sum[jc] - p_a[jc, ji, jj] * p_x[jj, jc]

        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            p_x[ji, jc] = z_sum[jc] / p_diag[jc, ji]

    for ji in reversed(range(maxdim)):
        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            z_sum[jc] = p_x[ji, jc]
        for jj in range(ji+1, maxdim):
            for jc in range(lower_bound, num_cells):
                if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                z_sum[jc] = z_sum[jc] - p_a[jc, jj, ji] * p_x[jj, jc]

        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            p_x[ji, jc] = z_sum[jc] / p_diag[jc, ji]

    return p_x

def compute_rbf_vec_coeff(
    c2e2c,
    c2e,
    lon,
    lat,
    cartesian_center_e,
    cartesian_center_c,
    mean_cell_area,
    primal_cart_normal_x,
    owner_mask,
    rbf_vec_dim_c,
    rbf_vec_kern_c,
    maxdim,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    num_cells
):
    rbf_vec_coeff_c = np.zeros((maxdim, 3, num_cells))
    mean_characteristic_length = math.sqrt(mean_cell_area)
    jg = 0

    z_rbfmat, istencil = _compute_z_rbfmat_istencil(
        c2e2c,
        c2e,
        cartesian_center_e,
        primal_cart_normal_x,
        mean_characteristic_length,
        owner_mask,
        rbf_vec_dim_c,
        rbf_vec_kern_c,
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    z_nx1, z_nx2 = _compute_z_xn1_z_xn2(
        lon,
        lat,
        cartesian_center_c,
        owner_mask,
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    z_rhs1, z_rhs2 = _compute_rhs1_rhs2(
        c2e,
        c2e2c,
        cartesian_center_c,
        cartesian_center_e,
        mean_characteristic_length,
        z_nx1,
        z_nx2,
        istencil,
        owner_mask,
        primal_cart_normal_x,
        jg,
        rbf_vec_kern_c,  # rbf_vec_kern_c from config
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    z_diag = _compute_z_diag(
        istencil,
        rbf_vec_dim_c,
        z_rbfmat,
        lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
        maxdim,
        num_cells
    )

    rbf_vec_coeff_c[:, 0, :] = _compute_rbf_vec_coeff(
        istencil,
        z_rbfmat,
        z_diag,
        z_rhs1,
        rbf_vec_dim_c,
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    return rbf_vec_coeff_c[:, 0, :]
