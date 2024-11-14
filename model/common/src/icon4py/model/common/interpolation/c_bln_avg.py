# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np


def inverse_neighbor_index(c2e2c0: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute "inverse neighbor index" for cells:
    The inverse neighbor index of a neighbor cell c is the neighbor index that the cell has from
    the point of view of its neighbors.

    For each cell index (c2e2c0[:, 0] is the local index) the functions returns a index tuple into c2e2c0
    containing the index positions of this cell index in c2e2c0

    Not fast but short.
    """
    return [np.where(c2e2c0 == i) for i in c2e2c0[:, 0]]


def _weight_sum_on_local_cell(
    c_bln_avg: np.ndarray,
    c2e2c0: np.ndarray,
    cell_area: np.ndarray,
    inverse_neighbor_index: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Compute the total weight which each local point contributes to the sum.

    Args:
        c_bln_avg: ndarray representing a weight field of (CellDim, C2E2C0Dim)
        inverse_neighbor_index: Sequence of to access all weights of a local cell in a field of shape (CellDim, C2E2C0Dim)

    Returns: ndarray of CellDim, containing the sum of weigh contributions for each local cell index

    """
    num_cells = c2e2c0.shape[0]
    assert (
        len(inverse_neighbor_index) == num_cells and c_bln_avg.shape[0] == num_cells
    ), "Fields need to  have 'cell length' "
    valid_connectivities = [np.where(c2e2c0[i] != -1) for i in range(num_cells)]

    x = [
        np.sum(c_bln_avg[vals] * cell_area[np.sort(c2e2c0[i][valid_connectivities[i]])])
        for i, vals in enumerate(inverse_neighbor_index)
    ]
    return np.stack(x)
  

def _residual_to_mass_conservation(
    local_weight: np.ndarray, owner_mask: np.ndarray, cell_area: np.ndarray
):
    """The local_weight weighted by the area should be 1. We compute how far we are off that weight."""
    horizontal_size = local_weight.shape[0]
    assert horizontal_size == owner_mask.shape[0], "Fields do not have the same shape"
    assert horizontal_size == cell_area.shape[0], "Fields do not have the same shape"
    residual = np.where(owner_mask, local_weight / cell_area - 1.0, 0.0)
    return residual


def _apply_correction(
    c_bln_avg: np.ndarray,
    residual: np.ndarray,
    c2e2c0: np.ndarray,
    owner_mask: np.ndarray,
    divavg_cntrwgt: float,
    horizontal_start: gtx.int32,
):
    maxwgt_loc = divavg_cntrwgt + 0.003
    minwgt_loc = divavg_cntrwgt - 0.003
    relax_coeff = 0.46
    c_bln_avg[horizontal_start:, :] = (
        c_bln_avg[horizontal_start:, :] - relax_coeff * residual[c2e2c0][horizontal_start:, :]
    )
    local_weight = np.sum(c_bln_avg, axis=1) - 1.0

    c_bln_avg[horizontal_start:, :] = c_bln_avg[horizontal_start:, :] - (
        0.25 * local_weight[horizontal_start:, np.newaxis]
    )

    # avoid runaway condition:
    c_bln_avg[horizontal_start:, 0] = np.maximum(c_bln_avg[horizontal_start:, 0], minwgt_loc)
    c_bln_avg[horizontal_start:, 0] = np.minimum(c_bln_avg[horizontal_start:, 0], maxwgt_loc)
    return c_bln_avg


def force_mass_conservation(
    c_bln_avg: np.ndarray,
    residual: np.ndarray,
    owner_mask: np.ndarray,
    horizontal_start: gtx.int32,
) -> np.ndarray:
    """enforce the mass conservation condition on the local cells by forcefully subtracting the residual."""
    c_bln_avg[horizontal_start:, 0] = np.where(
        owner_mask[horizontal_start:],
        c_bln_avg[horizontal_start:, 0] - residual[horizontal_start:],
        c_bln_avg[horizontal_start:, 0],
    )
    return c_bln_avg


def compute_force_mass_conservation(
    c_bln_avg: np.ndarray,
    c2e2c0: np.ndarray,
    owner_mask: np.ndarray,
    cell_areas: np.ndarray,
    divavg_cntrwgt: float,
    horizontal_start: gtx.int32,
    niter: int = 5,
) -> np.ndarray:
    inverse_neighbors = inverse_neighbor_index(c2e2c0)
    for i in range(niter):
        local_weight = _weight_sum_on_local_cell(
            c_bln_avg=c_bln_avg,
            inverse_neighbor_index=inverse_neighbors,
            c2e2c0=c2e2c0,
            cell_area=cell_areas,
        )
        residual = _residual_to_mass_conservation(
            local_weight=local_weight,
            cell_area=cell_areas,
            owner_mask=owner_mask,
        )
        if i >= (niter - 1):
            return force_mass_conservation(c_bln_avg, residual, owner_mask, horizontal_start)
        else:
            c_bln_avg = _apply_correction(
                c_bln_avg=c_bln_avg,
                residual=residual,
                c2e2c0=c2e2c0,
                owner_mask=owner_mask,
                divavg_cntrwgt=divavg_cntrwgt,
                horizontal_start=horizontal_start,
            )
