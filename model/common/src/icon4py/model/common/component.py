# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Protocol

from icon4py.model.common import state


class Component[I: state.State, O: state.State](Protocol):
    """
    `run(input, output) -> output`, the only method a component implements.

    A component declares its two states, as class attributes (`class Input(state.State)`) or as
    instance attributes assigned at construction; a component that writes nothing declares
    `Output = state.Empty`. The protocol has no implementation: the constructor is each
    component's own.

    `component.run(input, output)` writes where the caller says, numpy `out=` style, and
    returns `output`: the caller owns the buffers. Whether `output` may alias `input` is
    the component's business; nothing checks it here.
    """

    Input: type[I]
    Output: type[O]

    def run(self, input: I, output: O) -> O: ...
