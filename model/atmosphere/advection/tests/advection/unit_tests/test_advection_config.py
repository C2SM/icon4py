# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.advection import advection
from icon4py.model.common.decomposition import definitions as decomposition


def _weno_config(
    horizontal_advection_type: advection.HorizontalAdvectionType,
) -> advection.AdvectionConfig:
    return advection.AdvectionConfig(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=advection.HorizontalAdvectionLimiter.NO_LIMITER,
        vertical_advection_type=advection.VerticalAdvectionType.NO_ADVECTION,
        vertical_advection_limiter=advection.VerticalAdvectionLimiter.NO_LIMITER,
    )


def _transport_dict(ihadv_tracer: int) -> dict:
    # from_fortran_dict reads max_dom-sized lists (fortran_config.list_to_value)
    return {
        "transport_nml": {
            "ihadv_tracer": [ihadv_tracer],
            "itype_hlimit": [0],
            "ivadv_tracer": [0],
            "itype_vlimit": [0],
        }
    }


@pytest.mark.parametrize(
    ("ihadv_tracer", "expected"),
    [
        (0, advection.HorizontalAdvectionType.NO_ADVECTION),
        (2, advection.HorizontalAdvectionType.LINEAR_2ND_ORDER),
        (102, advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO),
        (103, advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO),
    ],
)
def test_from_fortran_dict_maps_horizontal_advection_type(
    ihadv_tracer: int, expected: advection.HorizontalAdvectionType
) -> None:
    config = advection.AdvectionConfig.from_fortran_dict(_transport_dict(ihadv_tracer))
    assert config.horizontal_advection_type == expected


def test_linear_weno_requires_weno_linear_state() -> None:
    # the ValueError is raised before any of the None-passed states are accessed
    with pytest.raises(ValueError, match="requires 'weno_linear_state'"):
        advection.convert_config_to_horizontal_vertical_advection(
            config=_weno_config(advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO),
            grid=None,
            interpolation_state=None,
            least_squares_state=None,
            metric_state=None,
            edge_params=None,
            cell_params=None,
            backend=None,
            exchange=decomposition.single_node_exchange,
            weno_linear_state=None,
        )


def test_quadratic_weno_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Quadratic WENO"):
        advection.convert_config_to_horizontal_vertical_advection(
            config=_weno_config(advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO),
            grid=None,
            interpolation_state=None,
            least_squares_state=None,
            metric_state=None,
            edge_params=None,
            cell_params=None,
            backend=None,
            exchange=decomposition.single_node_exchange,
        )
