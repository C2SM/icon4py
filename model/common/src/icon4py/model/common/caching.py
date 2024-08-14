# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Callable, Optional

import numpy as np
from gt4py import next as gtx
from gt4py.next.common import Connectivity
from gt4py.next.otf import workflow
from gt4py.next.program_processors.runners.gtfn import extract_connectivity_args

from icon4py.model.common.settings import device


try:
    import cupy as cp
    from gt4py.next.embedded.nd_array_field import CuPyArrayField
except ImportError:
    cp: Optional = None  # type:ignore[no-redef]

from gt4py.next.embedded.nd_array_field import NumPyArrayField


def handle_numpy_integer(value):
    return int(value)


def handle_common_field(value, sizes):
    sizes.extend(value.shape)
    return value  # Return the value unmodified, but side-effect on sizes


def handle_default(value):
    return value  # Return the value unchanged


if cp:
    type_handlers = {
        np.integer: handle_numpy_integer,
        NumPyArrayField: handle_common_field,
        CuPyArrayField: handle_common_field,
    }
else:
    type_handlers = {
        np.integer: handle_numpy_integer,
        NumPyArrayField: handle_common_field,
    }


def process_arg(value, sizes):
    handler = type_handlers.get(type(value), handle_default)
    return handler(value, sizes) if handler == handle_common_field else handler(value)


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
    _compiled_args: tuple = dataclasses.field(default_factory=tuple)

    @property
    def compiled_program(self) -> Callable:
        return self._compiled_program

    @property
    def conn_args(self) -> Callable:
        return self._conn_args

    def compile_the_program(
        self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any
    ) -> Callable:
        backend = self.program.backend
        program_call = backend.transforms_prog(
            workflow.InputWithArgs(
                data=self.program.definition_stage,
                args=args,
                kwargs=kwargs | {"offset_provider": offset_provider},
            )
        )
        self._compiled_args = program_call.args
        return backend.executor.otf_workflow(program_call)

    def __call__(self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any) -> None:
        if not self.compiled_program:
            self._compiled_program = self.compile_the_program(
                *args, offset_provider=offset_provider, **kwargs
            )
            self._conn_args = extract_connectivity_args(offset_provider, device)

        kwargs_as_tuples = tuple(kwargs.values())
        program_args = list(args) + list(kwargs_as_tuples)
        sizes = []

        # Convert numpy integers in args to int and handle gtx.common.Field
        for i in range(len(program_args)):
            program_args[i] = process_arg(program_args[i], sizes)

        if not self.with_domain:
            program_args.extend(sizes)

        # todo(samkellerhals): if we merge gt4py PR we can also pass connectivity args here conn_args=self.conn_args
        return self.compiled_program(*program_args, offset_provider=offset_provider)

    def with_connectivities(self, connectivities: dict[str, Connectivity]) -> "CachedProgram":
        """Used ONLY in DaCe Orchestration for ahead-of-time compilation."""
        return self
