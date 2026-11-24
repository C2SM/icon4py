# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.common import component, field, state

from tests.common.unit_tests.utils import Pressure, Salt


class Halve(component.Component["Halve.Input", "Halve.Output"]):
    class Input(state.State):
        pressure: field.Field[Pressure]

    class Output(state.State):
        pressure: field.Field[Pressure]

    def run(self, input: Input, output: Output) -> Output:
        np.asarray(output.pressure.ndarray)[...] = 0.5 * np.asarray(input.pressure.ndarray)
        return output


class Double:
    """A component in structure only: it does not inherit from `component.Component`."""

    Input = Halve.Input
    Output = Halve.Output

    def run(self, input: Halve.Input, output: Halve.Output) -> Halve.Output:
        np.asarray(output.pressure.ndarray)[...] = 2.0 * np.asarray(input.pressure.ndarray)
        return output


# What mypy checks: a quantity mismatch is a type error. Each ignore below is
# required (an unused one is reported), so type-checking this module is the test.
def static_checks(pressure: field.Field[Pressure], salt: field.Field[Salt]) -> None:
    wrong: field.Field[Pressure] = salt  # type: ignore[assignment]
    Halve.Input(pressure=salt)  # type: ignore[arg-type]
    Halve.Input(pressure=pressure)
    # a component that does not inherit from `component.Component` satisfies the protocol
    some_component: component.Component[Halve.Input, Halve.Output] = Double()
    del wrong, some_component
