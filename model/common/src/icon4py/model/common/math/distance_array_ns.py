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


def minimum_image_offset(
    *,
    delta: data_alloc.NDArray,
    extent: float,
) -> data_alloc.NDArray:
    """Periodic offset mapping ``delta`` onto its minimum image in ``[-extent/2, extent/2]``.

    Subtracting the result from ``delta`` gives the shortest signed separation along a
    periodic direction of size ``extent``; subtracting it from a coordinate gives the
    periodic image of that coordinate closest to the point ``delta`` is measured from.
    """
    array_ns = data_alloc.array_namespace(delta)
    return extent * array_ns.round(delta / extent)


def minimum_image_separation(
    *,
    x: data_alloc.NDArray,
    y: data_alloc.NDArray,
    reference_x: data_alloc.NDArray | float,
    reference_y: data_alloc.NDArray | float,
    domain_extent_x: float,
    domain_extent_y: float,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Shortest signed separation from ``(reference_x, reference_y)`` to ``(x, y)`` on the torus.

    Equivalently the offset from the reference to the periodic image of ``(x, y)`` closest
    to it.

    The image coordinate is formed first and the reference subtracted from it, rather than
    shifting the raw difference. Algebraically the same, but it is the operation order of
    ICON's ``plane_torus_closest_coordinates`` followed by ``dv%x - v0`` for bit-identicity.
    """
    image_x = x - minimum_image_offset(delta=x - reference_x, extent=domain_extent_x)
    image_y = y - minimum_image_offset(delta=y - reference_y, extent=domain_extent_y)
    return (image_x - reference_x, image_y - reference_y)


def horizontal_distance_to_point(
    *,
    x: data_alloc.NDArray,
    y: data_alloc.NDArray,
    point_x: float,
    point_y: float,
    wrap: bool,
    domain_extent_x: float | None = None,
    domain_extent_y: float | None = None,
) -> data_alloc.NDArray:
    """Horizontal distance from each point ``(x, y)`` to a fixed ``(point_x, point_y)``.

    With ``wrap=False`` this is the plain Euclidean distance on the plane. With
    ``wrap=True`` the distance is computed on a doubly-periodic torus using the
    minimum-image convention, where ``domain_extent_x`` and ``domain_extent_y`` are the
    periodic extents in the x and y directions (both required in that case).

    ``wrap`` has no default on purpose: the choice is not obvious for a doubly-periodic
    torus. ICON's own ``plane_torus_distance`` (``mo_grid_utilities.f90``) is fed
    coordinates normalized by the feature width while its wrap threshold uses the
    dimensional domain size, so the periodic branch is never taken and the effective
    distance is non-periodic. Idealized torus test cases (e.g. the Weisman-Klemp warm
    bubble and the Gaussian-hill topography) reproduce ICON and hence pass ``wrap=False``.
    """
    array_ns = data_alloc.array_namespace(x)
    if wrap:
        if domain_extent_x is None or domain_extent_y is None:
            raise ValueError(
                "Periodic wrapping requires both 'domain_extent_x' and 'domain_extent_y'."
            )
        dx, dy = minimum_image_separation(
            x=x,
            y=y,
            reference_x=point_x,
            reference_y=point_y,
            domain_extent_x=domain_extent_x,
            domain_extent_y=domain_extent_y,
        )
    else:
        dx = x - point_x
        dy = y - point_y
    return array_ns.sqrt(dx * dx + dy * dy)


def cos_central_angle(
    *,
    lon_center: float | data_alloc.NDArray,
    lat_center: float | data_alloc.NDArray,
    lon: data_alloc.NDArray,
    lat: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Cosine of the central angle between ``(lon, lat)`` and ``(lon_center, lat_center)``.

    Angles are in radians. Exposed separately from ``central_angle`` so that callers
    needing the cosine itself, such as map projections, avoid a ``cos(arccos(...))``
    round trip, which loses precision for nearby points.
    """
    array_ns = data_alloc.array_namespace(lat)
    return array_ns.sin(lat_center) * array_ns.sin(lat) + array_ns.cos(lat_center) * array_ns.cos(
        lat
    ) * array_ns.cos(lon - lon_center)


def central_angle(
    *,
    lon_center: float | data_alloc.NDArray,
    lat_center: float | data_alloc.NDArray,
    lon: data_alloc.NDArray,
    lat: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Central angle [rad] between ``(lon, lat)`` and ``(lon_center, lat_center)``.

    Multiply by the sphere radius to obtain the great-circle distance. The cosine is
    clipped to ``[-1, 1]`` because round-off can push it marginally outside, which
    would turn ``arccos`` into NaN.
    """
    array_ns = data_alloc.array_namespace(lat)
    cosine = cos_central_angle(lon_center=lon_center, lat_center=lat_center, lon=lon, lat=lat)
    return array_ns.arccos(array_ns.clip(cosine, -1.0, 1.0))
