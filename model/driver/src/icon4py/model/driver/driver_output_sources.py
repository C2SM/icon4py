# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""
The output variables of the model state and of the physics, as typed fields by standard name.

Three of the four sources of output variables: the prognostic state, the active tracers
and the diagnostic outputs of the physics processes, such as the precipitation fluxes.
The fourth, the standard diagnostics, is ``driver_diagnostics``; ``driver_io`` assembles
the variables of all four at a capture step.

A variable is named by the standard name of its quantity, which IO also reads its CF attributes from;
a field whose quantity has no standard name is not offered.
"""

from typing import Final

from icon4py.model.common import field, framework as fw, quantity
from icon4py.model.common.framework import states


# NOTE: temporary, replaced in parts 04 (recipes) and 05 (derived inputs): goes with
# `DiagnosticsComputer`, its only user.
def standard_name(quantity_: type[quantity.Quantity]) -> str:
    """The standard name of a quantity that is an output variable."""
    if quantity_.standard_name is None:
        raise ValueError(f"'{quantity_.__name__}' has no standard name: it cannot be output")
    return quantity_.standard_name


def standard_names(cls: type[fw.State]) -> dict[str, str]:
    """The standard name of each leaf of a `State` class that has one, by leaf name."""
    return {
        name: standard_name_
        for name, quantity_ in fw.leaf_quantities(cls).items()
        if (standard_name_ := quantity_.standard_name) is not None
    }


def by_standard_name(state: fw.State) -> dict[str, field.Field]:
    """The present leaves of a state that have a standard name, by that name."""
    return {
        name: field_
        for field_ in fw.fields(state).values()
        if (name := field_.quantity.standard_name) is not None
    }


#: The prognostic output variables.
PROGNOSTIC_VARIABLES: Final[tuple[str, ...]] = tuple(
    standard_names(states.PrognosticState).values()
)


def tracer_variables(tracer_config: states.TracerConfig | None) -> tuple[str, ...]:
    """The output variables of the active tracers."""
    if tracer_config is None:
        return ()
    names = standard_names(states.TracerState)
    return tuple(names[name] for name in tracer_config.active_names)
