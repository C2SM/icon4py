# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common.grid import geometry as grid_geometry
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)


class StaticFieldFactories(NamedTuple):
    """
    Factories of static fields for the driver and components.

    Attributes:
        geometry_field_source: grid geometry field factory that stores geometrical properties of a grid
        interpolation_field_source: interpolation field factory that stores pre-computed coefficients for interpolation employed in the model
        metrics_field_source: metric field factory that stores pre-computed coefficients for numerical operations employed in the model
    """

    geometry_field_source: grid_geometry.GridGeometry
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory
    metrics_field_source: metrics_factory.MetricsFieldsFactory


class DriverStates(NamedTuple):
    """
    Initialized states for the driver run.

    Attributes:
        prep_advection_prognostic: Fields collecting data for advection during the solve nonhydro timestep.
        solve_nonhydro_diagnostic: Initial state for solve_nonhydro diagnostic variables.
        diffusion_diagnostic: Initial state for diffusion diagnostic variables.
        prognostics: Initial state for prognostic variables (double buffered).
        diagnostic: Initial state for global diagnostic variables.
    """

    prep_advection_prognostic: dycore_states.PrepAdvection
    solve_nonhydro_diagnostic: dycore_states.DiagnosticStateNonHydro
    diffusion_diagnostic: diffusion_states.DiffusionDiagnosticState
    prognostics: common_utils.TimeStepPair[prognostics.PrognosticState]
    diagnostic: diagnostics.DiagnosticState
