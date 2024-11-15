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
    The inverse neighbor index of a cell c is the index of its neighbors pointing back to it, ie
    the neighbor index that the cell c has from the point of view of its neighbors.

    For each cell index the functions returns a index tuple into
    c2e2c0 containing the index positions of this cell index in c2e2c0.
    Due to the construction the indices are ordered.

    Not fast but short.

    >>> inv_nn = inverse_neighbor_index(c2e2c0)
    >>> i = 788
    >>> c2e2c0[inv_nn[i]]
    ... (i, i, i, i)


    TODO (@halungge) they do halo-exchange this in ICON
        (see mo_intp_coeffs.f90:force_mass_conservation_to_cellavg_wgt( ptr_patch, ptr_int, niter )
    """
    return [np.where(c2e2c0 == i) for i in c2e2c0[:, 0]]


def _compute_weight_on_local_cell(
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
  

def _compute_residual_to_mass_conservation(owner_mask: np.ndarray, local_weight: np.ndarray,
                                           cell_area: np.ndarray):
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


def _enforce_mass_conservation(
    c_bln_avg: np.ndarray,
    residual: np.ndarray,
    owner_mask: np.ndarray,
    horizontal_start: gtx.int32,
) -> np.ndarray:
    """Enforce the mass conservation condition on the local cells by forcefully subtracting the
    residual from the central field contribution."""
    c_bln_avg[horizontal_start:, 0] = np.where(
        owner_mask[horizontal_start:],
        c_bln_avg[horizontal_start:, 0] - residual[horizontal_start:],
        c_bln_avg[horizontal_start:, 0],
    )
    return c_bln_avg


def force_mass_conservation(
    c_bln_avg: np.ndarray,
    c2e2c0: np.ndarray,
    owner_mask: np.ndarray,
    cell_areas: np.ndarray,
    divavg_cntrwgt: float,
    horizontal_start: gtx.int32,
    niter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively enforce mass conservation to the input field c_bln_avg.

    Mass conservation is enforced by the following condition:
    The three point divergence calculated on any given grid point is used with a total factor of 1.

    Hence, the bilinear

    Args:
        c_bln_avg: input field
        c2e2c0:
        owner_mask: shape (num_cells, ) boolean mask marking local cells
        cell_areas: shape (num_cells, ), area of cells
        divavg_cntrwgt: float, configuration value
        horizontal_start: horizontal_start index
        niter: maximal number of iterations

    Returns:
        corrected input field that  fullfill mass conservation condition

    """
    inverse_neighbor_indices = inverse_neighbor_index(c2e2c0)
    #
    max_resid = np.zeros(niter)
    for i in range(niter):
        local_weight = _compute_weight_on_local_cell(
            c_bln_avg=c_bln_avg,
            inverse_neighbor_index=inverse_neighbor_indices,
            c2e2c0=c2e2c0,
            cell_area=cell_areas,
        )
        residual = _compute_residual_to_mass_conservation(owner_mask=owner_mask,
                                                          local_weight=local_weight,
                                                          cell_area=cell_areas)
        # TODO (@halungge) halo-echange residual
        max_residual = np.max(residual)
        max_resid[i] = max_residual
        if i >= (niter - 1) or max_residual < 1e-8:
            print(f"residuals in last 10 iterations {max_resid[-9:]}")
            print(f"number of iterations: {i} - max residual={max_residual}")
            unforced = np.copy(c_bln_avg)
            return (unforced, _enforce_mass_conservation(c_bln_avg, residual, owner_mask, horizontal_start))
        else:
            c_bln_avg = _apply_correction(
                c_bln_avg=c_bln_avg,
                residual=residual,
                c2e2c0=c2e2c0,
                divavg_cntrwgt=divavg_cntrwgt,
                horizontal_start=horizontal_start,
            )
        # TODO (@halungge) halo-echange c_bln_avg