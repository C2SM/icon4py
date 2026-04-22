# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_directional_derivative_on_cells(
    cell_field: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    lb_e: int,
    ub_e: int,
    num_edges: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """
    Compute directional derivative of a cell centered variable with respect to
    direction normal to triangle edge.
    """
    directional_derivative_on_cells = array_ns.zeros((num_edges,))
    directional_derivative_on_cells[lb_e:ub_e] = (
        cell_field[e2c[lb_e:ub_e, 1]] - cell_field[e2c[lb_e:ub_e, 0]]
    ) * inv_dual_edge_length[lb_e:ub_e]
    return directional_derivative_on_cells


def interpolate_edges_to_cell(
    edge_field: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    edge_cell_length: data_alloc.NDArray,
    primal_edge_length: data_alloc.NDArray,
    cell_area: data_alloc.NDArray,
    ub_c: int,
    num_cells: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """
    Compute interpolation of scalar fields from edge points to cell centers.
    """
    e_inn_c = array_ns.zeros((num_cells, 3))
    jc_indices = array_ns.arange(ub_c)[:, array_ns.newaxis]
    c2e_local = c2e[:ub_c]
    idx_ce = e2c[c2e_local][:, :, 0] != jc_indices
    e_inn_c[:ub_c] = (
        edge_cell_length[c2e_local, idx_ce]
        * primal_edge_length[c2e_local]
        / cell_area[:ub_c, array_ns.newaxis]
    )
    return array_ns.sum(edge_field[c2e] * e_inn_c, axis=1)
