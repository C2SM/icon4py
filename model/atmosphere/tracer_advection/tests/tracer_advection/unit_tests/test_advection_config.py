# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import (
    tracer_advection,
    tracer_advection_states,
    weno_least_squares,
)
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


def _transport_dict(ihadv_tracer: int, itype_hlimit: int = 0) -> dict:
    # from_fortran_dict reads max_dom-sized lists (fortran_config.list_to_value)
    return {
        "transport_nml": {
            "ihadv_tracer": [ihadv_tracer],
            "itype_hlimit": [itype_hlimit],
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
        (132, tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO_HYBRID),
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
        (tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO_HYBRID, 1.005),
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


def _weno_quadratic_state_with_weights(
    l_weights_s: tuple[float, ...],
) -> tracer_advection_states.AdvectionWenoQuadraticState:
    # only the weights matter for the consistency check; the fields are never touched
    fields = {
        name: None
        for name in tracer_advection_states.AdvectionWenoQuadraticState.__dataclass_fields__
        if name != "l_weights_s"
    }
    return tracer_advection_states.AdvectionWenoQuadraticState(l_weights_s=l_weights_s, **fields)  # type: ignore [arg-type] # None placeholders


def test_weno_linear_weights_default_is_the_live_set() -> None:
    config = _weno_config(tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO)
    assert config.weno_linear_weights == tracer_advection.WenoLinearWeights.OPTIMIZED
    from_fortran = tracer_advection.AdvectionConfig.from_fortran_dict(_transport_dict(103))
    assert from_fortran.weno_linear_weights == tracer_advection.WenoLinearWeights.OPTIMIZED
    # the state default is the same set, so today's driver path passes the check unchanged
    state = tracer_advection_states.AdvectionWenoQuadraticState(
        **{
            name: None
            for name in tracer_advection_states.AdvectionWenoQuadraticState.__dataclass_fields__
            if name != "l_weights_s"
        }  # type: ignore [arg-type] # None placeholders
    )
    np.testing.assert_array_equal(state.l_weights_s, weno_least_squares.L_WEIGHTS_S)
    tracer_advection._check_weno_linear_weights(config, state)


@pytest.mark.parametrize("option", list(tracer_advection.WenoLinearWeights))
def test_weno_linear_weights_state_must_match_config(option) -> None:
    config = dataclasses.replace(
        _weno_config(tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO),
        weno_linear_weights=option,
    )
    matching = _weno_quadratic_state_with_weights(tuple(weno_least_squares.linear_weights(option)))
    tracer_advection._check_weno_linear_weights(config, matching)

    other = next(o for o in tracer_advection.WenoLinearWeights if o != option)
    mismatching = _weno_quadratic_state_with_weights(
        tuple(weno_least_squares.linear_weights(other))
    )
    with pytest.raises(ValueError, match="assembled with linear weights"):
        tracer_advection._check_weno_linear_weights(config, mismatching)


_HADV = tracer_advection.HorizontalAdvectionType
_HLIM = tracer_advection.HorizontalAdvectionLimiter


@pytest.mark.parametrize(
    ("ihadv_tracer", "itype_hlimit", "expected"),
    [
        # inside his schemes itype_hlimit=4 is his cell-local limiter ...
        (102, 4, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE),
        (103, 4, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE),
        (132, 4, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE),
        # ... elsewhere ICON's, and 3 is ICON's monotonic limiter everywhere
        (2, 4, _HLIM.POSITIVE_DEFINITE),
        (3, 4, _HLIM.POSITIVE_DEFINITE),
        (20, 4, _HLIM.POSITIVE_DEFINITE),
        (103, 3, _HLIM.MONOTONIC),
        (3, 3, _HLIM.MONOTONIC),
        (132, 0, _HLIM.NO_LIMITER),
    ],
)
def test_from_fortran_dict_maps_itype_hlimit_4_per_scheme(
    ihadv_tracer: int, itype_hlimit: int, expected: tracer_advection.HorizontalAdvectionLimiter
) -> None:
    config = tracer_advection.AdvectionConfig.from_fortran_dict(
        _transport_dict(ihadv_tracer, itype_hlimit)
    )
    assert config.horizontal_advection_limiter == expected


def test_cell_local_limiter_value_does_not_collide_with_icon() -> None:
    # ICON's itype_hlimit: 0, 3 (ifluxl_m), 4 (ifluxl_sm)
    assert _HLIM.CELL_LOCAL_POSITIVE_DEFINITE.value not in {0, 3, 4}


@pytest.mark.parametrize(
    "horizontal_advection_type",
    [_HADV.LINEAR_2ND_ORDER, _HADV.QUADRATIC_3RD_ORDER, _HADV.LINEAR_2ND_ORDER_SUBCYCLED],
)
def test_cell_local_limiter_is_refused_outside_jocksch_schemes(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
) -> None:
    with pytest.raises(ValueError, match="exists only inside Jocksch's schemes"):
        dataclasses.replace(
            _weno_config(horizontal_advection_type),
            horizontal_advection_limiter=_HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        )


@pytest.mark.parametrize(
    "horizontal_advection_type",
    [
        _HADV.LINEAR_2ND_ORDER_WENO,
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
    ],
)
def test_cell_local_limiter_is_accepted_inside_jocksch_schemes(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
) -> None:
    config = dataclasses.replace(
        _weno_config(horizontal_advection_type),
        horizontal_advection_limiter=_HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
    )
    assert config.horizontal_advection_limiter == _HLIM.CELL_LOCAL_POSITIVE_DEFINITE


def test_hybrid_requires_weno_hybrid_state() -> None:
    # the ValueError is raised before any of the None-passed states are accessed
    with pytest.raises(ValueError, match="requires 'weno_hybrid_state'"):
        tracer_advection.convert_config_to_horizontal_vertical_advection(
            config=_weno_config(_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID),
            grid=None,
            interpolation_state=None,
            least_squares_state=None,
            metric_state=None,
            edge_params=None,
            cell_params=None,
            backend=None,
            exchange=decomposition.single_node_exchange,
            weno_hybrid_state=None,
        )


def test_hybrid_selection_threshold_default_is_the_fortran_literal() -> None:
    config = _weno_config(_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID)
    assert config.weno_hybrid_selection_threshold == 5e-5
    from_fortran = tracer_advection.AdvectionConfig.from_fortran_dict(_transport_dict(132))
    assert from_fortran.weno_hybrid_selection_threshold == 5e-5
