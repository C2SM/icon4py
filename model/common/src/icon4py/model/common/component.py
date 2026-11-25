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


def copy_to_output(input: state.State, output: state.State) -> None:
    """
    Copy the leaves `input` and `output` share into `output`, unless they share a buffer.

    For components that update in place: when the caller passes distinct buffers, `run`
    starts from a copy of `input`.
    """
    sources = state.fields(input)
    for name, target in state.fields(output).items():
        source = sources.get(name)
        if source is not None and target.ndarray is not source.ndarray:
            target.ndarray[...] = source.ndarray  # type: ignore[index]  # NDArrayObject declares no __setitem__
