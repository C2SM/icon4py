# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from icon4py.model.common.components import states
from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.initial_condition.analytical import (
    gauss3d as gauss_ic,
    jablonowski_williamson as jw_ic,
    linear_horizontal_tracer_advection as lin_hor_adv_ic,
    linear_vertical_tracer_advection as lin_ver_adv_ic,
    weisman_klemp as wk_ic,
)
from icon4py.model.common.initial_condition.config import ConfigContext
from icon4py.model.common.math.stencils import generic_math_operations as gt4py_math_op
from icon4py.model.common.metrics import metrics_attributes


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import icon as icon_grid
    from icon4py.model.common.states import (
        prognostic_state as prognostics,
        static_fields,
        tracer_states,
    )


def apply(
    *,
    config: ConfigContext,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    dycore_diagnostics: states.DycoreDiagnostics | None,
    tracer_prep_adv_state: states.PrepAdvection | None,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    global_reductions: decomposition_defs.Reductions,
) -> None:
    """
    Fill the prognostic and tracer states by dispatching on the type of ``config``.

    The perturbed exner function of the dycore is initialized too, when its diagnostic
    state is given: diagnosed from the initial state, or, when restarting, read from the
    serialized data together with the advective tendencies of the previous time step.
    """
    match config.initial_condition:
        case jw_ic.JablonowskiWilliamsonConfig():
            jw_ic.jablonowski_williamson(
                config=config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                tracer_state_now=tracer_state_now,
                backend=backend,
                exchange=exchange,
                global_reductions=global_reductions,
            )
        case gauss_ic.Gauss3DConfig():
            gauss_ic.gauss3d(
                config=config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
            )
        case wk_ic.WeismanKlempConfig():
            wk_ic.weisman_klemp(
                config=config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                tracer_state_now=tracer_state_now,
                backend=backend,
                exchange=exchange,
            )
        case lin_hor_adv_ic.LinearHorizontalAdvectionConfig():
            if tracer_prep_adv_state is None:
                raise ValueError(
                    "'tracer_prep_adv_state' must not be None for 'linear_horizontal_advection' initial conditions."
                )
            lin_hor_adv_ic.linear_horizontal_advection(
                config=config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                tracer_state_now=tracer_state_now,
                tracer_prep_adv_state=tracer_prep_adv_state,
            )
        case lin_ver_adv_ic.LinearVerticalAdvectionConfig():
            if tracer_prep_adv_state is None:
                raise ValueError(
                    "'tracer_prep_adv_state' must not be None for 'linear_vertical_advection' initial conditions."
                )
            lin_ver_adv_ic.linear_vertical_advection(
                config=config,
                metrics=static_fields.metrics,
                prognostic_state_now=prognostic_state_now,
                tracer_state_now=tracer_state_now,
                tracer_prep_adv_state=tracer_prep_adv_state,
            )
        case from_file_ic.FromFileConfig():
            if config.is_restart:
                if dycore_diagnostics is None:
                    raise ValueError(
                        "restarting needs the diagnostic state of the dycore to initialize."
                    )
                from_file_ic.read_restart_from_file(
                    config=config,
                    grid=grid,
                    prognostic_state_now=prognostic_state_now,
                    dycore_diagnostics=dycore_diagnostics,
                    backend=backend,
                    exchange=exchange,
                )
                return
            from_file_ic.read_initial_condition_from_file(
                config=config,
                grid=grid,
                prognostic_state_now=prognostic_state_now,
                tracer_state_now=tracer_state_now,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise TypeError(
                f"Unknown initial conditions config type: {type(config.initial_condition)!r}"
            )

    if dycore_diagnostics is not None:
        # exner_pr, diagnosed from the initial state (compute_exner_pert in mo_nh_stepping.f90)
        gt4py_math_op.compute_difference_on_cell_k.with_backend(backend)(
            field_a=prognostic_state_now.exner,
            field_b=static_fields.metrics.get(metrics_attributes.EXNER_REF_MC),
            output_field=dycore_diagnostics.perturbed_exner_at_cells_on_model_levels.data,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
            offset_provider={},
        )
