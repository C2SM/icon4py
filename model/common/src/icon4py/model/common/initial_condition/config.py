# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import TYPE_CHECKING, Any

from icon4py.model.common import time
from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.initial_condition.analytical import (
    gauss3d as gauss_ic,
    jablonowski_williamson as jw_ic,
)
from icon4py.model.common.math.stencils import generic_math_operations as gt4py_math_op
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import fortran_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid
    from icon4py.model.common.states import (
        nonhydro_states,
        prognostic_state as prognostics,
        static_fields,
    )

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InitialConditionConfig:
    config: jw_ic.JablonowskiWilliamsonConfig | gauss_ic.Gauss3DConfig | from_file_ic.FromFileConfig

    @classmethod
    def from_fortran_dict(
        cls,
        *,
        atm_dict: dict[str, Any],
        input_dict: dict[str, Any],
        data_path: pathlib.Path,
        start_of_simulation: time.AbsoluteTime,
        start_of_timestepping: time.AbsoluteTime,
        dtime: time.RelativeTime,
    ) -> InitialConditionConfig:
        run_nml = atm_dict["run_nml"]
        if not run_nml["ltestcase"]:
            log.info("Reading initial condition from file")
            return cls(
                config=from_file_ic.FromFileConfig(
                    data_path=data_path / fortran_config.SER_DATA_SUBDIR,
                    start_of_simulation=start_of_simulation,
                    start_of_timestepping=start_of_timestepping,
                    dtime=dtime,
                    ntracer=fortran_config.list_to_value(run_nml["ntracer"]),
                ),
            )

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        test_name = testcase_nml.get("nh_test_name")
        config: (
            jw_ic.JablonowskiWilliamsonConfig | gauss_ic.Gauss3DConfig
        )  # mypy does not automatically catch type
        match test_name:
            case "jabw" | "jabw_s" | "APE_nwp" | "APE_aes":
                log.info("Analytical initial condition for Jablonowski-Williamson test case")
                config = fortran_config.config_dataclass_from_dict(
                    jw_ic.JablonowskiWilliamsonConfig, testcase_nml
                )
                # Only the APE cases rescale qv to a prescribed global moisture content.
                config.normalize_global_moisture = test_name in ("APE_nwp", "APE_aes")
                # Fortran resets jw_up to 0 only for jabw_s; other cases keep the default (1.0).
                if test_name == "jabw_s":
                    config.baroclinic_amplitude = 0.0
            case (
                "gauss3D" | "wk82"
            ):  # TODO (jcanton): wk82 is just a placeholder until next PR, it is not actually used
                log.info("Analytical initial condition for Gauss 3D test case")
                config = fortran_config.config_dataclass_from_dict(
                    gauss_ic.Gauss3DConfig, testcase_nml
                )
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(config=config)


def create(
    *,
    config: InitialConditionConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    solve_nonhydro_diagnostic_state: nonhydro_states.DiagnosticStateNonHydro | None,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    global_reductions: decomposition_defs.Reductions,
) -> None:
    """
    Fill the prognostic state by dispatching on the type of ``config.config``.

    The perturbed exner function of the dycore is initialized too, when its diagnostic
    state is given: diagnosed from the initial state, or, when restarting, read from the
    serialized data together with the advective tendencies of the previous time step.
    """
    match config.config:
        case jw_ic.JablonowskiWilliamsonConfig():
            jw_ic.jablonowski_williamson(
                config=config.config,
                vertical_config=vertical_config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
                global_reductions=global_reductions,
            )
        case gauss_ic.Gauss3DConfig():
            gauss_ic.gauss3d(
                config=config.config,
                vertical_config=vertical_config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
            )
        case from_file_ic.FromFileConfig():
            if config.config.is_restart:
                if solve_nonhydro_diagnostic_state is None:
                    raise ValueError(
                        "restarting needs the diagnostic state of the dycore to initialize."
                    )
                from_file_ic.read_restart_from_file(
                    config=config.config,
                    grid=grid,
                    prognostic_state_now=prognostic_state_now,
                    solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
                    backend=backend,
                    exchange=exchange,
                )
                return
            from_file_ic.read_initial_condition_from_file(
                config=config.config,
                grid=grid,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise TypeError(f"Unknown initial conditions config type: {type(config.config)!r}")

    if solve_nonhydro_diagnostic_state is not None:
        # exner_pr, diagnosed from the initial state (compute_exner_pert in mo_nh_stepping.f90)
        gt4py_math_op.compute_difference_on_cell_k.with_backend(backend)(
            field_a=prognostic_state_now.exner,
            field_b=static_fields.metrics.get(metrics_attributes.EXNER_REF_MC),
            output_field=solve_nonhydro_diagnostic_state.perturbed_exner_at_cells_on_model_levels,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
            offset_provider={},
        )
