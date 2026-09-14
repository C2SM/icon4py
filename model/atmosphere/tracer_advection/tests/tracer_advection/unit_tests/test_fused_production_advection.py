# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection, tracer_advection_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition


def _production_config(**overrides: Any) -> tracer_advection.AdvectionConfig:
    config = {
        "horizontal_advection_type": tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
        "horizontal_advection_limiter": tracer_advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
        "vertical_advection_type": tracer_advection.VerticalAdvectionType.PPM_3RD_ORDER,
        "vertical_advection_limiter": tracer_advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
    }
    config.update(overrides)
    return tracer_advection.AdvectionConfig(**cast(Any, config))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"horizontal_advection_type": tracer_advection.HorizontalAdvectionType.NO_ADVECTION},
        {"horizontal_advection_limiter": tracer_advection.HorizontalAdvectionLimiter.NO_LIMITER},
        {"vertical_advection_type": tracer_advection.VerticalAdvectionType.NO_ADVECTION},
        {"vertical_advection_limiter": tracer_advection.VerticalAdvectionLimiter.NO_LIMITER},
    ],
)
def test_production_config_requires_all_production_schemes(overrides: dict[str, Any]) -> None:
    assert tracer_advection._is_production_advection_config(_production_config(**overrides)) == (
        not overrides
    )


def test_convert_config_wires_production_programs_into_godunov(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not hasattr(tracer_advection, "FusedProductionAdvection")

    expected_advection = mock.sentinel.advection
    production_state = mock.sentinel.production_state
    initialize_production_state = mock.Mock(return_value=production_state)
    convert_horizontal_vertical = mock.Mock()
    godunov_advection = mock.Mock(return_value=expected_advection)
    allocator = mock.sentinel.allocator
    monkeypatch.setattr(
        tracer_advection.tracer_advection_states,
        "initialize_production_advection_state",
        initialize_production_state,
    )
    monkeypatch.setattr(
        tracer_advection,
        "convert_config_to_horizontal_vertical_advection",
        convert_horizontal_vertical,
    )
    monkeypatch.setattr(tracer_advection, "GodunovSplittingAdvection", godunov_advection)
    monkeypatch.setattr(model_backends, "get_allocator", mock.Mock(return_value=allocator))

    advection = tracer_advection.convert_config_to_advection(
        config=_production_config(),
        grid=mock.sentinel.grid,
        interpolation_state=mock.sentinel.interpolation_state,
        least_squares_state=mock.sentinel.least_squares_state,
        metric_state=mock.sentinel.metric_state,
        edge_params=mock.sentinel.edge_params,
        cell_params=mock.sentinel.cell_params,
        backend=mock.sentinel.backend,
        exchange=mock.sentinel.exchange,
        even_timestep=True,
    )

    assert advection is expected_advection
    convert_horizontal_vertical.assert_not_called()
    initialize_production_state.assert_called_once_with(
        grid=mock.sentinel.grid,
        allocator=allocator,
    )
    godunov_advection.assert_called_once_with(
        horizontal_advection=None,
        vertical_advection=None,
        grid=mock.sentinel.grid,
        metric_state=mock.sentinel.metric_state,
        backend=mock.sentinel.backend,
        exchange=mock.sentinel.exchange,
        even_timestep=True,
        production_state=production_state,
        production_interpolation_state=mock.sentinel.interpolation_state,
        production_least_squares_state=mock.sentinel.least_squares_state,
        production_edge_params=mock.sentinel.edge_params,
    )


def test_production_state_only_materializes_program_boundary_intermediates() -> None:
    assert [
        field.name for field in dataclasses.fields(tracer_advection_states.ProductionAdvectionState)
    ] == [
        "rhodz_ast2",
        "r_m",
        "p_tracer_after_vertical",
        "p_mflx_tracer_h_unlimited",
    ]


@pytest.mark.parametrize(
    "even_timestep, pre_program, post_program, limited_area",
    [
        (
            False,
            "_compute_odd_timestep_before_horizontal_limiter",
            "_compute_odd_timestep_after_horizontal_limiter",
            False,
        ),
        (
            True,
            "_compute_even_timestep_before_horizontal_limiter",
            "_compute_even_timestep_after_horizontal_limiter",
            True,
        ),
    ],
)
def test_godunov_production_wiring_exchanges_limiter_factor_between_programs(
    even_timestep: bool,
    pre_program: str,
    post_program: str,
    limited_area: bool,
) -> None:
    advection: Any = object.__new__(tracer_advection.GodunovSplittingAdvection)
    events: list[str] = []
    advection._grid = SimpleNamespace(limited_area=limited_area)
    advection._even_timestep = even_timestep

    production_state = SimpleNamespace(
        rhodz_ast2=mock.sentinel.rhodz_ast2,
        r_m=mock.sentinel.r_m,
        p_tracer_after_vertical=mock.sentinel.p_tracer_after_vertical,
        p_mflx_tracer_h_unlimited=mock.sentinel.p_mflx_tracer_h_unlimited,
    )
    advection._production_state = production_state
    advection._production_k = mock.sentinel.k

    for program_name in (
        "_compute_odd_timestep_before_horizontal_limiter",
        "_compute_odd_timestep_after_horizontal_limiter",
        "_compute_even_timestep_before_horizontal_limiter",
        "_compute_even_timestep_after_horizontal_limiter",
    ):
        setattr(
            advection,
            program_name,
            mock.Mock(side_effect=lambda *args, name=program_name, **kwargs: events.append(name)),
        )
    advection._apply_interpolated_tracer_time_tendency = mock.Mock(
        side_effect=lambda *args, **kwargs: events.append("limited-area-tendency")
    )

    def record_exchange(_dimension: Any, field: Any, *, stream: Any) -> None:
        assert _dimension is dims.CellDim
        assert stream is decomposition.DEFAULT_STREAM
        events.append(
            {
                mock.sentinel.mass_flx_ic: "mass-flux",
                mock.sentinel.r_m: "limiter-factor",
                mock.sentinel.p_tracer_new: "tracer",
            }[field]
        )

    advection._exchange = mock.Mock()
    advection._exchange.exchange.side_effect = record_exchange
    diagnostic_state = mock.Mock(
        airmass_now=mock.sentinel.airmass_now,
        airmass_new=mock.sentinel.airmass_new,
        hfl_tracer=mock.sentinel.hfl_tracer,
        vfl_tracer=mock.sentinel.vfl_tracer,
        grf_tend_tracer=mock.sentinel.grf_tend_tracer,
    )
    prep_adv = mock.Mock(
        mass_flx_ic=mock.sentinel.mass_flx_ic,
        mass_flx_me=mock.sentinel.mass_flx_me,
        vn_traj=mock.sentinel.vn_traj,
    )

    advection.run(
        diagnostic_state=diagnostic_state,
        prep_adv=prep_adv,
        p_tracer_now=mock.sentinel.p_tracer_now,
        p_tracer_new=mock.sentinel.p_tracer_new,
        dtime=mock.sentinel.dtime,
    )

    expected_events = ["mass-flux", pre_program, "limiter-factor", post_program]
    if limited_area:
        expected_events.append("limited-area-tendency")
    expected_events.append("tracer")
    assert events == expected_events
    if not even_timestep:
        assert (
            advection._compute_odd_timestep_before_horizontal_limiter.call_args.kwargs["rhodz_now"]
            is mock.sentinel.airmass_now
        )
        assert (
            advection._compute_odd_timestep_after_horizontal_limiter.call_args.kwargs["rhodz_now"]
            is mock.sentinel.airmass_now
        )
    assert advection._even_timestep is not even_timestep
