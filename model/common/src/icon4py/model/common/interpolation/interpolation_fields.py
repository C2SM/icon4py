# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math
from types import ModuleType

import gt4py.next as gtx
import numpy as np
from gt4py.next import where

import icon4py.model.common.field_type_aliases as fa
import icon4py.model.common.math.projection as proj
import icon4py.model.common.type_alias as ta
from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import C2E, V2E
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_c_lin_e(
    edge_cell_length: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    edge_owner_mask: data_alloc.NDArray,
    horizontal_start: np.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute E2C average inverse distance.

    Args:
        edge_cell_length: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        inv_dual_edge_length: ndarray, inverse dual edge length, numpy array representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        edge_owner_mask: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        horizontal_start: start index from the field is computed: c_lin_e is not calculated for the first boundary layer
        array_ns: ModuleType to use for the computation, numpy or cupy, defaults to cupy
    Returns: c_lin_e: numpy array, representing gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]

    """
    c_lin_e_ = edge_cell_length[:, 1] * inv_dual_edge_length
    c_lin_e = array_ns.transpose(array_ns.vstack((c_lin_e_, (1.0 - c_lin_e_))))
    c_lin_e[0:horizontal_start, :] = 0.0
    mask = array_ns.transpose(array_ns.tile(edge_owner_mask, (2, 1)))
    return array_ns.where(mask, c_lin_e, 0.0)


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
    dual_edge_length: data_alloc.NDArray,
    geofac_div: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    horizontal_start: np.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute geometric factor for nabla2-scalar.

    Args:
        dual_edge_length: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        geofac_div: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        c2e: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e2c: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        horizontal_start: start index from where the field is computed
        array_ns: python module, numpy or cpu defaults to numpy

    Returns:
        geometric factor for nabla2-scalar, Field[CellDim, C2E2CODim]
    """
    num_cells = c2e.shape[0]
    geofac_n2s = array_ns.zeros([num_cells, 4])
    index = array_ns.transpose(
        array_ns.vstack(
            (
                array_ns.arange(num_cells),
                array_ns.arange(num_cells),
                array_ns.arange(num_cells),
            )
        )
    )
    mask = e2c[c2e, 0] == index
    geofac_n2s[horizontal_start:, 0] = geofac_n2s[horizontal_start:, 0] - array_ns.sum(
        mask[horizontal_start:] * (geofac_div / dual_edge_length[c2e])[horizontal_start:], axis=1
    )
    mask = e2c[c2e, 1] == index
    geofac_n2s[horizontal_start:, 0] = geofac_n2s[horizontal_start:, 0] + array_ns.sum(
        mask[horizontal_start:] * (geofac_div / dual_edge_length[c2e])[horizontal_start:], axis=1
    )
    mask = e2c[c2e, 0] == c2e2c
    geofac_n2s[horizontal_start:, 1:] = (
        geofac_n2s[horizontal_start:, 1:]
        - mask[horizontal_start:, :] * (geofac_div / dual_edge_length[c2e])[horizontal_start:, :]
    )
    mask = e2c[c2e, 1] == c2e2c
    geofac_n2s[horizontal_start:, 1:] = (
        geofac_n2s[horizontal_start:, 1:]
        + mask[horizontal_start:, :] * (geofac_div / dual_edge_length[c2e])[horizontal_start:, :]
    )
    return geofac_n2s


def _compute_primal_normal_ec(
    primal_normal_cell_x: data_alloc.NDArray,
    primal_normal_cell_y: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Compute primal_normal_ec.

    Args:
        primal_normal_cell_x: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        primal_normal_cell_y: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        owner_mask: ndarray, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        array_ns: module - the array interface implementation to compute on, defaults to numpy
    Returns:
        primal_normal_ec: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim, 2], ta.wpfloat]
    """

    owned = array_ns.stack((owner_mask, owner_mask, owner_mask)).T

    inv_neighbor_index = create_inverse_neighbor_index(e2c, c2e, array_ns)
    u_component = primal_normal_cell_x[c2e, inv_neighbor_index]
    v_component = primal_normal_cell_y[c2e, inv_neighbor_index]
    return (array_ns.where(owned, u_component, 0.0), array_ns.where(owned, v_component, 0.0))


def _compute_geofac_grg(
    primal_normal_ec_u: data_alloc.NDArray,
    primal_normal_ec_v: data_alloc.NDArray,
    geofac_div: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Compute geometrical factor for Green-Gauss gradient.

    Args:
        primal_normal_ec_u: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim, 2], ta.wpfloat]
        primal_normal_ec_v: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim, 2], ta.wpfloat]
        geofac_div: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        c_lin_e: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        c2e: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e2c: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        horizontal_start: start index from where the computation is done
        array_ns: module - the array interface implementation to compute on, defaults to numpy
    Returns:
        geofac_grg: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim + 1, 2], ta.wpfloat]
    """
    num_cells = c2e.shape[0]
    targ_local_size = c2e.shape[1] + 1
    target_shape = (num_cells, targ_local_size)
    geofac_grg_x = array_ns.zeros(target_shape)
    geofac_grg_y = array_ns.zeros(target_shape)

    inverse_neighbor = create_inverse_neighbor_index(e2c, c2e, array_ns)

    tmp = geofac_div * c_lin_e[c2e, inverse_neighbor]
    geofac_grg_x[horizontal_start:, 0] = np.sum(primal_normal_ec_u * tmp, axis=1)[horizontal_start:]
    geofac_grg_y[horizontal_start:, 0] = np.sum(primal_normal_ec_v * tmp, axis=1)[horizontal_start:]

    for k in range(e2c.shape[1]):
        mask = (e2c[c2e, k] == c2e2c)[horizontal_start:, :]
        geofac_grg_x[horizontal_start:, 1:] = (
            geofac_grg_x[horizontal_start:, 1:]
            + mask * (primal_normal_ec_u * geofac_div * c_lin_e[c2e, k])[horizontal_start:, :]
        )
        geofac_grg_y[horizontal_start:, 1:] = (
            geofac_grg_y[horizontal_start:, 1:]
            + mask * (primal_normal_ec_v * geofac_div * c_lin_e[c2e, k])[horizontal_start:, :]
        )

    return geofac_grg_x, geofac_grg_y


def compute_geofac_grg(
    primal_normal_cell_x: data_alloc.NDArray,
    primal_normal_cell_y: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    geofac_div: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    primal_normal_ec_u, primal_normal_ec_v = functools.partial(
        _compute_primal_normal_ec, array_ns=array_ns
    )(primal_normal_cell_x, primal_normal_cell_y, owner_mask, c2e, e2c)
    return functools.partial(_compute_geofac_grg, array_ns=array_ns)(
        primal_normal_ec_u,
        primal_normal_ec_v,
        geofac_div,
        c_lin_e,
        c2e,
        e2c,
        c2e2c,
        horizontal_start,
    )


def compute_geofac_grdiv(
    geofac_div: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    e2c2e: data_alloc.NDArray,
    horizontal_start: np.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute geometrical factor for gradient of divergence (triangles only).

    Args:
        geofac_div:  ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        inv_dual_edge_length: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        owner_mask:  ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], bool]
        c2e:  ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        e2c2e: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EDim], gtx.int32]
        horizontal_start:
        array_ns: module either used or array computations defaults to numpy

    Returns:
        geofac_grdiv:  ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EODim], ta.wpfloat]
    """
    num_edges = e2c.shape[0]
    geofac_grdiv = array_ns.zeros([num_edges, 1 + 2 * e2c.shape[1]])
    index = array_ns.arange(horizontal_start, num_edges)
    for j in range(c2e.shape[1]):
        mask = array_ns.where(
            c2e[e2c[horizontal_start:, 1], j] == index, owner_mask[horizontal_start:], False
        )
        geofac_grdiv[horizontal_start:, 0] = array_ns.where(
            mask, geofac_div[e2c[horizontal_start:, 1], j], geofac_grdiv[horizontal_start:, 0]
        )
    for j in range(c2e.shape[1]):
        mask = array_ns.where(
            c2e[e2c[horizontal_start:, 0], j] == index, owner_mask[horizontal_start:], False
        )
        geofac_grdiv[horizontal_start:, 0] = array_ns.where(
            mask,
            (geofac_grdiv[horizontal_start:, 0] - geofac_div[e2c[horizontal_start:, 0], j])
            * inv_dual_edge_length[horizontal_start:],
            geofac_grdiv[horizontal_start:, 0],
        )
    for j in range(e2c.shape[1]):
        for k in range(c2e.shape[1]):
            mask = c2e[e2c[horizontal_start:, 0], k] == e2c2e[horizontal_start:, j]
            geofac_grdiv[horizontal_start:, e2c.shape[1] - 1 + j] = array_ns.where(
                mask,
                -geofac_div[e2c[horizontal_start:, 0], k] * inv_dual_edge_length[horizontal_start:],
                geofac_grdiv[horizontal_start:, e2c.shape[1] - 1 + j],
            )
            mask = c2e[e2c[horizontal_start:, 1], k] == e2c2e[horizontal_start:, e2c.shape[1] + j]
            geofac_grdiv[horizontal_start:, 2 * e2c.shape[1] - 1 + j] = array_ns.where(
                mask,
                geofac_div[e2c[horizontal_start:, 1], k] * inv_dual_edge_length[horizontal_start:],
                geofac_grdiv[horizontal_start:, 2 * e2c.shape[1] - 1 + j],
            )
    return geofac_grdiv


def rotate_latlon(
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    pollat: data_alloc.NDArray,
    pollon: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    (Compute rotation of lattitude and longitude.)

    Rotates latitude and longitude for more accurate computation
    of bilinear interpolation

    Args:
        lat: scalar or numpy array
        lon: scalar or numpy array
        pollat: scalar or numpy array
        pollon: scalar or numpy array
        array_ns array namespace to be used, defaults to numpy

    Returns:
        rotlat:
        rotlon:
    """
    rotlat = array_ns.arcsin(
        array_ns.sin(lat) * array_ns.sin(pollat)
        + array_ns.cos(lat) * array_ns.cos(pollat) * array_ns.cos(lon - pollon)
    )
    rotlon = array_ns.arctan2(
        array_ns.cos(lat) * array_ns.sin(lon - pollon),
        (
            array_ns.cos(lat) * array_ns.sin(pollat) * array_ns.cos(lon - pollon)
            - array_ns.sin(lat) * array_ns.cos(pollat)
        ),
    )

    return (rotlat, rotlon)


def _weighting_factors(
    ytemp: data_alloc.NDArray,
    xtemp: data_alloc.NDArray,
    yloc: data_alloc.NDArray,
    xloc: data_alloc.NDArray,
    wgt_loc: ta.wpfloat,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
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
            array_ns: array namespace to be used defaults to numpy

        Returns:
            wgt: numpy array of size [[3, flexible], ta.wpfloat]
    """
    rotate = functools.partial(rotate_latlon, array_ns=array_ns)

    pollat = array_ns.where(yloc >= 0.0, yloc - math.pi * 0.5, yloc + math.pi * 0.5)
    pollon = xloc
    (yloc, xloc) = rotate(yloc, xloc, pollat, pollon)
    x = array_ns.zeros([ytemp.shape[0], ytemp.shape[1]])
    y = array_ns.zeros([ytemp.shape[0], ytemp.shape[1]])
    wgt = array_ns.zeros([ytemp.shape[0], ytemp.shape[1]])

    for i in range(ytemp.shape[0]):
        (ytemp[i], xtemp[i]) = rotate(ytemp[i], xtemp[i], pollat, pollon)
        y[i] = ytemp[i] - yloc
        x[i] = xtemp[i] - xloc
        # This is needed when the date line is crossed
        x[i] = array_ns.where(x[i] > 3.5, x[i] - math.pi * 2, x[i])
        x[i] = array_ns.where(x[i] < -3.5, x[i] + math.pi * 2, x[i])

    mask = array_ns.logical_and(abs(x[1] - x[0]) > 1.0e-11, abs(y[2] - y[0]) > 1.0e-11)
    wgt_1_no_mask = (
        1.0
        / ((y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[0]) / (x[2] - x[0]))
        * (1.0 - wgt_loc)
        * (-y[0] + x[0] * (y[2] - y[0]) / (x[2] - x[0]))
    )
    wgt[2] = array_ns.where(
        mask,
        1.0
        / ((y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
        * (1.0 - wgt_loc)
        * (-y[0] + x[0] * (y[1] - y[0]) / (x[1] - x[0])),
        (-(1.0 - wgt_loc) * x[0] - wgt_1_no_mask * (x[1] - x[0])) / (x[2] - x[0]),
    )
    wgt[1] = array_ns.where(
        mask,
        (-(1.0 - wgt_loc) * x[0] - wgt[2] * (x[2] - x[0])) / (x[1] - x[0]),
        wgt_1_no_mask,
    )
    wgt[0] = 1.0 - wgt[1] - wgt[2] if wgt_loc == 0.0 else 1.0 - wgt_loc - wgt[1] - wgt[2]
    return wgt


def _compute_c_bln_avg(
    c2e2c: data_alloc.NDArray,
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    divavg_cntrwgt: ta.wpfloat,
    horizontal_start: np.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
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
    num_cells = c2e2c.shape[0]
    ytemp = array_ns.zeros([c2e2c.shape[1], num_cells - horizontal_start])
    xtemp = array_ns.zeros([c2e2c.shape[1], num_cells - horizontal_start])

    for i in range(ytemp.shape[0]):
        ytemp[i] = lat[c2e2c[horizontal_start:, i]]
        xtemp[i] = lon[c2e2c[horizontal_start:, i]]

    wgt = _weighting_factors(
        ytemp,
        xtemp,
        lat[horizontal_start:],
        lon[horizontal_start:],
        divavg_cntrwgt,
        array_ns=array_ns,
    )
    c_bln_avg = array_ns.zeros((c2e2c.shape[0], c2e2c.shape[1] + 1))
    c_bln_avg[horizontal_start:, 0] = divavg_cntrwgt
    c_bln_avg[horizontal_start:, 1] = wgt[0]
    c_bln_avg[horizontal_start:, 2] = wgt[1]
    c_bln_avg[horizontal_start:, 3] = wgt[2]
    return c_bln_avg


def _force_mass_conservation_to_c_bln_avg(
    c2e2c0: data_alloc.NDArray,
    c_bln_avg: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    divavg_cntrwgt: ta.wpfloat,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
    niter: int = 1000,
) -> data_alloc.NDArray:
    """
    Iteratively enforce mass conservation to the input field c_bln_avg.

    Mass conservation is enforced by the following condition:
    The three point divergence calculated on any given grid point is used with a total factor of 1.

    Practically, the sum of the  bilinear cell weights  applied to a cell from all neighbors times its area should be exactly one.

    The weights are adjusted iteratively by the condition up to a max of niter iterations

    Args:
        c2e2c0: cell to cell connectivity
        c_bln_avg: input field: bilinear cell weight average
        cell_areas: area of cells
        cell_owner_mask:
        divavg_cntrwgt: configured central weight
        horizontal_start:
        niter: max number of iterations

    Returns:

    """

    def _compute_local_weights(
        c_bln_avg, cell_areas, c2e2c0, inverse_neighbor_idx
    ) -> data_alloc.NDArray:
        """
        Compute the total weight which each local point contributes to the sum.

        Args:
            c_bln_avg: ndarray representing a weight field of (CellDim, C2E2C0Dim)
            inverse_neighbor_index: Sequence of to access all weights of a local cell in a field of shape (CellDim, C2E2C0Dim)

        Returns: ndarray of CellDim, containing the sum of weigh contributions for each local cell index

        """
        weights = array_ns.sum(c_bln_avg[c2e2c0, inverse_neighbor_idx] * cell_areas[c2e2c0], axis=1)
        return weights

    def _compute_residual_to_mass_conservation(
        owner_mask: data_alloc.NDArray,
        local_weight: data_alloc.NDArray,
        cell_area: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        """The local_weight weighted by the area should be 1. We compute how far we are off that weight."""
        horizontal_size = local_weight.shape[0]
        assert horizontal_size == owner_mask.shape[0], "Fields do not have the same shape"
        assert horizontal_size == cell_area.shape[0], "Fields do not have the same shape"
        residual = array_ns.where(owner_mask, local_weight / cell_area - 1.0, 0.0)
        return residual

    def _apply_correction(
        c_bln_avg: data_alloc.NDArray,
        residual: data_alloc.NDArray,
        c2e2c0: data_alloc.NDArray,
        divavg_cntrwgt: float,
        horizontal_start: gtx.int32,
    ) -> data_alloc.NDArray:
        """Apply correction to local weigths based on the computed residuals."""
        maxwgt_loc = divavg_cntrwgt + 0.003
        minwgt_loc = divavg_cntrwgt - 0.003
        relax_coeff = 0.46
        c_bln_avg[horizontal_start:, :] = (
            c_bln_avg[horizontal_start:, :] - relax_coeff * residual[c2e2c0][horizontal_start:, :]
        )
        local_weight = array_ns.sum(c_bln_avg, axis=1) - 1.0

        c_bln_avg[horizontal_start:, :] = c_bln_avg[horizontal_start:, :] - (
            0.25 * local_weight[horizontal_start:, np.newaxis]
        )

        # avoid runaway condition:
        c_bln_avg[horizontal_start:, 0] = array_ns.maximum(
            c_bln_avg[horizontal_start:, 0], minwgt_loc
        )
        c_bln_avg[horizontal_start:, 0] = array_ns.minimum(
            c_bln_avg[horizontal_start:, 0], maxwgt_loc
        )
        return c_bln_avg

    def _enforce_mass_conservation(
        c_bln_avg: data_alloc.NDArray,
        residual: data_alloc.NDArray,
        owner_mask: data_alloc.NDArray,
        horizontal_start: gtx.int32,
    ) -> data_alloc.NDArray:
        """Enforce the mass conservation condition on the local cells by forcefully subtracting the
        residual from the central field contribution."""
        c_bln_avg[horizontal_start:, 0] = array_ns.where(
            owner_mask[horizontal_start:],
            c_bln_avg[horizontal_start:, 0] - residual[horizontal_start:],
            c_bln_avg[horizontal_start:, 0],
        )
        return c_bln_avg

    local_summed_weights = array_ns.zeros(c_bln_avg.shape[0])
    residual = array_ns.zeros(c_bln_avg.shape[0])
    inverse_neighbor_idx = create_inverse_neighbor_index(c2e2c0, c2e2c0, array_ns=array_ns)

    for iteration in range(niter):
        local_summed_weights[horizontal_start:] = _compute_local_weights(
            c_bln_avg, cell_areas, c2e2c0, inverse_neighbor_idx
        )[horizontal_start:]

        residual[horizontal_start:] = _compute_residual_to_mass_conservation(
            cell_owner_mask, local_summed_weights, cell_areas
        )[horizontal_start:]

        max_ = array_ns.max(residual)
        if iteration >= (niter - 1) or max_ < 1e-9:
            print(f"number of iterations: {iteration} - max residual={max_}")
            c_bln_avg = _enforce_mass_conservation(
                c_bln_avg, residual, cell_owner_mask, horizontal_start
            )
            return c_bln_avg

        c_bln_avg = _apply_correction(
            c_bln_avg=c_bln_avg,
            residual=residual,
            c2e2c0=c2e2c0,
            divavg_cntrwgt=divavg_cntrwgt,
            horizontal_start=horizontal_start,
        )

    return c_bln_avg


def compute_mass_conserving_bilinear_cell_average_weight(
    c2e2c0: data_alloc.NDArray,
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    divavg_cntrwgt: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_start_level_3: gtx.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    c_bln_avg = _compute_c_bln_avg(
        c2e2c0[:, 1:], lat, lon, divavg_cntrwgt, horizontal_start, array_ns
    )
    return _force_mass_conservation_to_c_bln_avg(
        c2e2c0,
        c_bln_avg,
        cell_areas,
        cell_owner_mask,
        divavg_cntrwgt,
        horizontal_start_level_3,
        array_ns,
    )


def create_inverse_neighbor_index(source_offset, inverse_offset, array_ns: ModuleType):
    """
    The inverse neighbor index determines the position of an central element c_1
    in the neighbor table of its neighbors:

    For example: for let e_1, e_2, e_3 be the neighboring edges of a cell: c2e(c_1) will
    map  c_1 -> (e_1, e_2,e_3) then in the inverse lookup table e2c the
    neighborhoods of e_1, e_2, e_3 will all contain c_1 in some position.
    Then inverse neighbor index tells what position that is. It essentially says
    "I am neighbor number x \in (0,1) of my neighboring edges"


    Args:
        source_offset:
        inverse_offset:

    Returns:
        ndarray of the same shape as target_offset

    """
    inv_neighbor_idx = -1 * array_ns.ones(inverse_offset.shape, dtype=int)

    for jc in range(inverse_offset.shape[0]):
        for i in range(inverse_offset.shape[1]):
            if inverse_offset[jc, i] >= 0:
                inv_neighbor_idx[jc, i] = array_ns.argwhere(
                    source_offset[inverse_offset[jc, i], :] == jc
                )[0, 0]

    return inv_neighbor_idx


# TODO (@halungge) this can be simplified using only
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
    dual_area: data_alloc.NDArray,
    edge_vert_length: data_alloc.NDArray,
    edge_cell_length: data_alloc.NDArray,
    v2e: data_alloc.NDArray,
    e2v: data_alloc.NDArray,
    v2c: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute cells_aw_verts.

    Args:
        dual_area: numpy array, representing a gtx.Field[gtx.Dims[VertexDim], ta.wpfloat]
        edge_vert_length: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat]
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
    cells_aw_verts = array_ns.zeros(v2e.shape)
    for jv in range(horizontal_start, cells_aw_verts.shape[0]):
        cells_aw_verts[jv, :] = 0.0
        for je in range(v2e.shape[1]):
            # INVALID_INDEX
            if v2e[jv, je] == gm.GridFile.INVALID_INDEX or (
                je > 0 and v2e[jv, je] == v2e[jv, je - 1]
            ):
                continue
            ile = v2e[jv, je]
            idx_ve = 0 if e2v[ile, 0] == jv else 1
            cell_offset_idx_0 = e2c[ile, 0]
            cell_offset_idx_1 = e2c[ile, 1]
            for jc in range(v2e.shape[1]):
                if v2c[jv, jc] == gm.GridFile.INVALID_INDEX or (
                    jc > 0 and v2c[jv, jc] == v2c[jv, jc - 1]
                ):
                    continue
                if cell_offset_idx_0 == v2c[jv, jc]:
                    cells_aw_verts[jv, jc] = (
                        cells_aw_verts[jv, jc]
                        + 0.5
                        / dual_area[jv]
                        * edge_vert_length[ile, idx_ve]
                        * edge_cell_length[ile, 0]
                    )
                elif cell_offset_idx_1 == v2c[jv, jc]:
                    cells_aw_verts[jv, jc] = (
                        cells_aw_verts[jv, jc]
                        + 0.5
                        / dual_area[jv]
                        * edge_vert_length[ile, idx_ve]
                        * edge_cell_length[ile, 1]
                    )

    return cells_aw_verts


def compute_e_bln_c_s(
    c2e: data_alloc.NDArray,
    cells_lat: data_alloc.NDArray,
    cells_lon: data_alloc.NDArray,
    edges_lat: data_alloc.NDArray,
    edges_lon: data_alloc.NDArray,
    weighting_factor: float,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
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
    e_bln_c_s = array_ns.zeros([num_cells, c2e.shape[1]])
    yloc = cells_lat[llb:]
    xloc = cells_lon[llb:]
    ytemp = array_ns.zeros([c2e.shape[1], num_cells])
    xtemp = array_ns.zeros([c2e.shape[1], num_cells])

    for i in range(ytemp.shape[0]):
        ytemp[i] = edges_lat[c2e[llb:, i]]
        xtemp[i] = edges_lon[c2e[llb:, i]]

    wgt = _weighting_factors(
        ytemp,
        xtemp,
        yloc,
        xloc,
        weighting_factor,
        array_ns=array_ns,
    )

    e_bln_c_s[:, 0] = wgt[0]
    e_bln_c_s[:, 1] = wgt[1]
    e_bln_c_s[:, 2] = wgt[2]
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
