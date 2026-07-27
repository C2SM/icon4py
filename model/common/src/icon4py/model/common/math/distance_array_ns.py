# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side (NumPy/CuPy) distance operations.

Counterparts of the GT4Py field operators in ``distance.py`` that operate on host
arrays, e.g. during initial-condition and topography setup.
"""

from __future__ import annotations

from icon4py.model.common.utils import data_allocation as data_alloc


def horizontal_distance_to_point(
    *,
    x: data_alloc.NDArray,
    y: data_alloc.NDArray,
    point_x: float,
    point_y: float,
    domain_length: float | None = None,
    domain_height: float | None = None,
    wrap: bool = False,
) -> data_alloc.NDArray:
    """Horizontal distance from each point ``(x, y)`` to a fixed ``(point_x, point_y)``.

    With ``wrap=False`` (the default) this is the plain Euclidean distance on the
    plane. With ``wrap=True`` the distance is computed on a doubly-periodic torus
    using the minimum-image convention, where ``domain_length`` and ``domain_height``
    are the periodic extents in the x and y directions (both required in that case).

    The default reproduces ICON: its ``plane_torus_distance`` (``mo_grid_utilities.f90``)
    is fed coordinates normalized by the feature width while the wrap threshold uses the
    dimensional domain size, so the periodic branch is never taken and the effective
    distance is non-periodic. Idealized torus test cases (e.g. the Weisman-Klemp warm
    bubble and the Gaussian-hill topography) rely on this non-wrapping behaviour.
    """
    array_ns = data_alloc.array_namespace(x)
    dx = x - point_x
    dy = y - point_y
    if wrap:
        if domain_length is None or domain_height is None:
            raise ValueError("Periodic wrapping requires both 'domain_length' and 'domain_height'.")
        dx = dx - domain_length * array_ns.round(dx / domain_length)
        dy = dy - domain_height * array_ns.round(dy / domain_height)
    return array_ns.sqrt(dx * dx + dy * dy)


def diff_on_edges_torus_numpy(
    *,
    cc_cv_x: float,
    cc_cv_y: float,
    cc_cell_x: float,
    cc_cell_y: float,
    domain_length: float,
    domain_height: float,
) -> tuple[float, float]:
    if abs(cc_cell_x - cc_cv_x) <= 0.5 * domain_length:
        x1 = cc_cell_x
    elif cc_cv_x > cc_cell_x:
        x1 = cc_cell_x + domain_length
    else:
        x1 = cc_cell_x - domain_length

    if abs(cc_cell_y - cc_cv_y) <= 0.5 * domain_height:
        y1 = cc_cell_y
    elif cc_cv_y > cc_cell_y:
        y1 = cc_cell_y + domain_height
    else:
        y1 = cc_cell_y - domain_height
    return x1, y1
