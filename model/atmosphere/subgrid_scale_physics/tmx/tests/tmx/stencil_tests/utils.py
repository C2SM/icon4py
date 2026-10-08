# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
NumPy helpers shared by the tmx stencil tests.

A helper moves here once a second test module needs it; a helper that one module uses
stays next to its test.
"""

import numpy as np

from icon4py.model.common.constants import PhysicsConstants as phy


def diffusion_matrix_numpy(interface_coeff: np.ndarray, inv_air_mass: np.ndarray) -> np.ndarray:
    """
    Matrix of minus the flux divergence on a column of rows, built column by column from unit
    vectors. interface_coeff[:, j] couples rows j and j + 1; no flux crosses the column ends.
    """
    num_rows = inv_air_mass.shape[1]
    unit_vectors = np.broadcast_to(np.eye(num_rows), (inv_air_mass.shape[0], num_rows, num_rows))
    downward_flux = interface_coeff[:, :, np.newaxis] * (unit_vectors[:, :-1] - unit_vectors[:, 1:])
    downward_flux = np.pad(downward_flux, ((0, 0), (1, 1), (0, 0)))
    return inv_air_mass[:, :, np.newaxis] * (downward_flux[:, 1:] - downward_flux[:, :-1])


def tridiagonal_matrix_numpy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    matrix = np.zeros((*b.shape, b.shape[1]))
    rows = np.arange(b.shape[1])
    matrix[:, rows, rows] = b
    matrix[:, rows[1:], rows[:-1]] = a[:, 1:]
    matrix[:, rows[:-1], rows[1:]] = c[:, :-1]
    return matrix


def matrix_diagonals_on_rows(
    matrix: np.ndarray, shape: tuple[int, int], rows: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    b[:, rows] = np.diagonal(matrix, axis1=1, axis2=2)
    a[:, rows][:, 1:] = np.diagonal(matrix, offset=-1, axis1=1, axis2=2)
    c[:, rows][:, :-1] = np.diagonal(matrix, offset=1, axis1=1, axis2=2)
    return a, b, c


def implicit_diffusion_tendency_numpy(
    *,
    var: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    rhs: np.ndarray,
    dtime: float,
    rows: slice,
) -> np.ndarray:
    matrix = tridiagonal_matrix_numpy(a[:, rows], b[:, rows], c[:, rows])
    matrix += np.eye(matrix.shape[1]) / dtime
    new_var = np.linalg.solve(matrix, (var[:, rows] / dtime + rhs[:, rows])[..., np.newaxis])
    out = np.zeros_like(var)
    out[:, rows] = (new_var[..., 0] - var[:, rows]) / dtime
    return out


def moist_heat_capacity_numpy(
    qv: np.ndarray, q_liquid: np.ndarray, q_solid: np.ndarray
) -> np.ndarray:
    return (
        phy.cvd * (1.0 - qv - q_liquid - q_solid)
        + phy.cvv * qv
        + phy.cpl * q_liquid
        + phy.cpi * q_solid
    )


def internal_energy_per_area_numpy(
    *,
    temperature: np.ndarray,
    qv: np.ndarray,
    q_liquid: np.ndarray,
    q_solid: np.ndarray,
    rho: np.ndarray | float,
    dz: np.ndarray | float,
) -> np.ndarray:
    return (
        rho
        * dz
        * (
            moist_heat_capacity_numpy(qv, q_liquid, q_solid) * temperature
            - q_liquid * phy.lvc
            - q_solid * phy.lsc
        )
    )


def on_subdomain(
    values: np.ndarray,
    horizontal: slice,
    vertical: slice,
    *,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """
    An output as a program with this domain writes it: `values` on the domain, and outside it
    the output's initial value (`initial`, zero if not given).
    """
    out = np.zeros_like(values) if initial is None else initial.copy()
    out[horizontal, vertical] = values[horizontal, vertical]
    return out
