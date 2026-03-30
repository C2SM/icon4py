# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Backward-compatibility re-export module.

Functions have been reorganized into the following submodules:
- ``coordinate_transformations``: geographical/cartesian and zonal/meridional conversions
- ``distance``: arc length, torus distance and difference operations
- ``gradient``: finite difference gradient operators
- ``vector_operations``: dot product, cross product, norms, normalization, inversion
- ``vertical_operations``: vertical level averaging and differencing

Please update imports to use the specific submodules directly.
"""

from icon4py.model.common.math.coordinate_transformations import (  # noqa: F401
    cartesian_coordinates_from_zonal_and_meridional_components_on_cells,
    cartesian_coordinates_from_zonal_and_meridional_components_on_edges,
    compute_cartesian_coordinates_from_zonal_and_meridional_components_on_cells,
    compute_cartesian_coordinates_from_zonal_and_meridional_components_on_edges,
    compute_zonal_and_meridional_components_on_edges,
    geographical_to_cartesian_on_cells,
    geographical_to_cartesian_on_edges,
    geographical_to_cartesian_on_vertices,
    zonal_and_meridional_components_on_cells,
    zonal_and_meridional_components_on_edges,
)
from icon4py.model.common.math.distance import (  # noqa: F401
    arc_length_on_edges,
    diff_on_edges_torus,
    distance_on_edges_torus,
)
from icon4py.model.common.math.gradient import (  # noqa: F401
    _grad_fd_tang,
    grad_fd_norm,
)
from icon4py.model.common.math.vector_operations import (  # noqa: F401
    compute_inverse_on_edges,
    cross_product_on_edges,
    dot_product_on_cells,
    dot_product_on_edges,
    dot_product_on_vertices,
    invert_edge_field,
    norm2_on_cells,
    norm2_on_edges,
    norm2_on_vertices,
    normalize_cartesian_vector_on_edges,
)
from icon4py.model.common.math.vertical_operations import (  # noqa: F401
    average_level_plus1_on_cells,
    average_level_plus1_on_edges,
    average_two_vertical_levels_downwards_on_cells,
    average_two_vertical_levels_downwards_on_edges,
    difference_level_plus1_on_cells,
)
