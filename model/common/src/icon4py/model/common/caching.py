# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from typing import Any, Callable, Optional

import numpy as np
from gt4py import next as gtx
from gt4py.next.otf import arguments, workflow
from gt4py.next.ffront import signature
from gt4py.next.program_processors.runners.gtfn import extract_connectivity_args

from icon4py.model.common.settings import device


def handle_numpy_integer(value):
    return int(value)


def handle_default(value):
    return value  # Return the value unchanged


type_handlers = {
    np.integer: handle_numpy_integer,
}


def process_arg(value):
    handler = type_handlers.get(type(value), handle_default)
    return handler(value)


@dataclasses.dataclass
class CachedProgram:
    """Class to handle caching and compilation of GT4Py programs.

    This class is responsible for caching and compiling GT4Py programs
    with optional domain information. The compiled program and its
    connectivity arguments are stored for efficient execution.

    Attributes:
        program (gtx.ffront.decorator.Program): The GT4Py program to be cached and compiled.
        with_domain (bool): Flag to indicate if the program should be compiled with domain information. Defaults to True.
        _compiled_program (Optional[Callable]): The compiled GT4Py program.
        _conn_args (Any): Connectivity arguments extracted from the offset provider.
        _compiled_args (tuple): Arguments used during the compilation of the program.

    Properties:
        compiled_program (Callable): Returns the compiled GT4Py program.
        conn_args (Any): Returns the connectivity arguments.

    Note:
        This functionality will be provided by GT4Py in the future.
    """

    program: gtx.ffront.decorator.Program
    with_domain: bool = True
    _compiled_program: Optional[Callable] = None
    _conn_args: Any = None

    @property
    def compiled_program(self) -> Callable:
        return self._compiled_program

    @property
    def conn_args(self) -> Callable:
        return self._conn_args

    def compile_the_program(
        self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any
    ) -> Callable:
        return self.program.backend.jit(
            self.program.definition_stage, *args, **kwargs | {"offset_provider": offset_provider}
        )

    def __call__(self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any) -> None:
        if not self.compiled_program:
            self._compiled_program = self.compile_the_program(
                *args, offset_provider=offset_provider, **kwargs
            )
            self._conn_args = extract_connectivity_args(offset_provider, device)

        args, kwargs = signature.convert_to_positional(
            self.program.definition_stage, *args, **kwargs
        )

        # Convert numpy integers in args to int and handle gtx.common.Field
        args = tuple(process_arg(arg) for arg in args)

        # todo(samkellerhals): if we merge gt4py PR we can also pass connectivity args here conn_args=self.conn_args
        return self.compiled_program(*args, offset_provider=offset_provider)
