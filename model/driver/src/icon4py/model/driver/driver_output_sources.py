# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The model state and the physics outputs as CF/UGRID-annotated DataArrays the IO module writes.

Three of the four sources of output variables: the prognostic state, the active tracers
and the diagnostic outputs of the physics processes, such as the precipitation fluxes.
The fourth, the standard diagnostics, is ``driver_diagnostics``; ``driver_io`` assembles
the variables of all four at a capture step.
"""

from collections.abc import Collection, Mapping
from typing import Final

import gt4py.next as gtx
import xarray as xr

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common.io import utils as io_utils
from icon4py.model.common.states import (
    data as state_data,
    prognostic_state as prognostics,
    tracer_states,
)


#: Prognostic output variables, selected by CF name from the
#: ``states.data.PROGNOSTIC_CF_ATTRIBUTES`` catalog (which also holds fields the driver does
#: not output, e.g. ``tangential_velocity``). The metadata and the state attribute
#: (``icon_var_name``) come from that catalog; this list only selects which entries to emit.
PROGNOSTIC_VARIABLES: Final[list[str]] = [
    "air_density",
    "exner_function",
    "virtual_potential_temperature",
    "upward_air_velocity",
    "normal_velocity",
]


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: Collection[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    selected = PROGNOSTIC_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for name in selected:
        try:
            metadata = state_data.PROGNOSTIC_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown prognostic output variable '{name}'. "
                f"Known variables are: {PROGNOSTIC_VARIABLES}."
            ) from err
        assert metadata.icon_var_name is not None, (
            f"prognostic output '{name}' must declare icon_var_name to be read from the state"
        )
        field = getattr(prognostic_state, metadata.icon_var_name)
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            to_host=True,
        )
    return state


def tracer_state_to_dataarrays(
    tracer_state: tracer_states.TracerState, variables: Collection[str]
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays of the requested active tracers.

    The variables are the keys of ``states.data.COMMON_TRACER_CF_ATTRIBUTES`` (``qv``,
    ``qc``, ...), which are also the names of the ``TracerState`` fields.
    """
    active: dict[str, gtx.Field] = {
        tracer.name: tracer.field for tracer in tracer_state.active_fields()
    }
    state: dict[str, xr.DataArray] = {}
    for name in variables:
        try:
            metadata = state_data.COMMON_TRACER_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown tracer output variable '{name}'. Known tracers are: "
                f"{', '.join(state_data.COMMON_TRACER_CF_ATTRIBUTES)}."
            ) from err
        if name not in active:
            raise ValueError(f"Tracer output variable '{name}' is not active in the tracer state.")
        state[name] = io_utils.to_data_array(active[name], metadata, to_host=True)
    return state


def physics_diagnostic_outputs(
    physics_diagnostics: physics_state.DiagnosticsStore | None,
) -> dict[str, physics_state.DiagnosticOutput]:
    """The diagnostic outputs of the physics processes offered as output variables.

    Keyed by the CF ``standard_name`` of the output's metadata, e.g. ``precipitation_flux``
    for the ``pflx`` output of muphys; surface (purely horizontal) outputs are offered like
    the others. Without physics there are none. The buffers hold the values of the last
    step the process ran, and zeros before that.
    """
    outputs: dict[str, physics_state.DiagnosticOutput] = {}
    if physics_diagnostics is None:
        return outputs
    for output in physics_diagnostics.outputs():
        name = output.metadata.standard_name
        if name in outputs:
            raise ValueError(
                f"Physics diagnostic output '{name}' is provided by more than one process: "
                f"{outputs[name].process_name}, {output.process_name}."
            )
        outputs[name] = output
    return outputs


def physics_diagnostics_to_dataarrays(
    outputs: Mapping[str, physics_state.DiagnosticOutput], variables: Collection[str]
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays of the requested physics diagnostics."""
    return {
        name: io_utils.to_data_array(outputs[name].field, outputs[name].metadata, to_host=True)
        for name in variables
    }
