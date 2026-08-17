# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common.decomposition import definitions as decomposition


def _weno_config(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
) -> tracer_advection.AdvectionConfig:
    return tracer_advection.AdvectionConfig(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=tracer_advection.HorizontalAdvectionLimiter.NO_LIMITER,
        vertical_advection_type=tracer_advection.VerticalAdvectionType.NO_ADVECTION,
        vertical_advection_limiter=tracer_advection.VerticalAdvectionLimiter.NO_LIMITER,
    )


def _transport_dict(ihadv_tracer: int) -> dict:
    # from_fortran_dict reads max_dom-sized lists (fortran_config.list_to_value)
    return {
        "transport_nml": {
            "ihadv_tracer": [ihadv_tracer],
            "itype_hlimit": [0],
            "ivadv_tracer": [0],
            "itype_vlimit": [0],
            # beta_fct is a scalar in the namelist, not a max_dom-sized list
            "beta_fct": 1.005,
            "nadv_substeps": [3],
        }
    }


@pytest.mark.parametrize(
    ("ihadv_tracer", "expected"),
    [
        (0, tracer_advection.HorizontalAdvectionType.NO_ADVECTION),
        (2, tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER),
        (3, tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER),
        (20, tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_SUBCYCLED),
        (102, tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO),
        (103, tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO),
    ],
)
def test_from_fortran_dict_maps_horizontal_advection_type(
    ihadv_tracer: int, expected: tracer_advection.HorizontalAdvectionType
) -> None:
    config = tracer_advection.AdvectionConfig.from_fortran_dict(_transport_dict(ihadv_tracer))
    assert config.horizontal_advection_type == expected


@pytest.mark.parametrize(
    ("horizontal_advection_type", "expected"),
    [
        # Fortran passes opt_beta_fct only from the quadratic-reconstruction schemes;
        # the linear ones get hflx_limiter_mo's own default of 1
        (tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER, 1.0),
        (tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO, 1.0),
        (tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER, 1.005),
        (tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO, 1.005),
    ],
)
def test_monotonic_limiter_beta_fct_depends_on_the_scheme(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType, expected: float
) -> None:
    config = _weno_config(horizontal_advection_type)
    assert config.monotonic_limiter_boost_factor == 1.005
    assert tracer_advection._monotonic_limiter_beta_fct(config) == expected


@pytest.mark.parametrize("boost_factor", [0.999, 2.0, 2.5])
def test_monotonic_limiter_boost_factor_is_range_checked(boost_factor: float) -> None:
    with pytest.raises(ValueError, match="must be in \\[1, 2\\)"):
        tracer_advection.AdvectionConfig(
            horizontal_advection_type=tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            horizontal_advection_limiter=tracer_advection.HorizontalAdvectionLimiter.MONOTONIC,
            vertical_advection_type=tracer_advection.VerticalAdvectionType.NO_ADVECTION,
            vertical_advection_limiter=tracer_advection.VerticalAdvectionLimiter.NO_LIMITER,
            monotonic_limiter_boost_factor=boost_factor,
        )


def test_quadratic_requires_quadratic_state() -> None:
    # the ValueError is raised before any of the None-passed states are accessed
    with pytest.raises(ValueError, match="requires 'quadratic_state'"):
        tracer_advection.convert_config_to_horizontal_vertical_advection(
            config=_weno_config(tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER),
            grid=None,
            interpolation_state=None,
            least_squares_state=None,
            metric_state=None,
            edge_params=None,
            cell_params=None,
            backend=None,
            exchange=decomposition.single_node_exchange,
            quadratic_state=None,
        )


def test_linear_weno_requires_weno_linear_state() -> None:
    # the ValueError is raised before any of the None-passed states are accessed
    with pytest.raises(ValueError, match="requires 'weno_linear_state'"):
        tracer_advection.convert_config_to_horizontal_vertical_advection(
            config=_weno_config(tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO),
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


def test_quadratic_weno_requires_weno_quadratic_state() -> None:
    # the ValueError is raised before any of the None-passed states are accessed
    with pytest.raises(ValueError, match="requires 'weno_quadratic_state'"):
        tracer_advection.convert_config_to_horizontal_vertical_advection(
            config=_weno_config(tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO),
            grid=None,
            interpolation_state=None,
            least_squares_state=None,
            metric_state=None,
            edge_params=None,
            cell_params=None,
            backend=None,
            exchange=decomposition.single_node_exchange,
            weno_quadratic_state=None,
        )
