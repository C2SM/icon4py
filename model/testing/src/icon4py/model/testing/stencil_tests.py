# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py import eve
from gt4py._core.definitions import is_scalar_type
from gt4py.next import backend as gtx_backend, constructors
from gt4py.next.ffront.decorator import FieldOperator, Program

from icon4py.model.common.grid import base
from icon4py.model.common.utils import device_utils


def allocate_data(
    backend: gtx_backend.Backend | None,
    input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
) -> dict[str, gtx.Field | tuple[gtx.Field, ...]]:
    _allocate_field = constructors.as_field.partial(allocator=backend)  # type:ignore[attr-defined] # TODO(havogt): check why it doesn't understand the fluid_partial
    input_data = {
        k: tuple(_allocate_field(domain=field.domain, data=field.ndarray) for field in v)
        if isinstance(v, tuple)
        else _allocate_field(domain=v.domain, data=v.ndarray)
        if not is_scalar_type(v) and k != "domain"
        else v
        for k, v in input_data.items()
    }
    return input_data


@dataclasses.dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))


@dataclasses.dataclass
class _ConnectivityConceptFixer:
    """
    This works around a misuse of dimensions as an identifier for connectivities.
    Since GT4Py might change the way the mesh is represented, we could
    keep this for a while, otherwise we need to touch all StencilTests.
    """

    _grid: base.Grid

    def __getitem__(self, dim: gtx.Dimension | str) -> np.ndarray:
        if isinstance(dim, gtx.Dimension):
            dim = dim.value
        return self._grid.get_connectivity(dim).asnumpy()


class StandardStaticVariants(eve.StrEnum):
    NONE = "none"
    COMPILE_TIME_DOMAIN = "compile_time_domain"
    COMPILE_TIME_VERTICAL = "compile_time_vertical"


class StencilTest:
    """
    Base class to be used for testing stencils.

    Example (pseudo-code):

        >>> class TestMultiplyByTwo(StencilTest):  # doctest: +SKIP
        ...     PROGRAM = multiply_by_two  # noqa: F821
        ...     OUTPUTS = ("some_output",)
        ...     MARKERS = (pytest.mark.some_marker,)
        ...     STATIC_PARAMS = {"category_a": ["flag0"], "category_b": ["flag0", "flag1"]}
        ...
        ...     @pytest.fixture
        ...     def input_data(self):
        ...         return {"some_input": ..., "some_output": ...}
        ...
        ...     @staticmethod
        ...     def reference(some_input, **kwargs):
        ...         return dict(some_output=np.asarray(some_input) * 2)
    """

    PROGRAM: ClassVar[Program | FieldOperator]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    MARKERS: ClassVar[tuple | None] = None
    STATIC_PARAMS: ClassVar[dict[str, Sequence[str]] | None] = None

    reference: ClassVar[Callable[..., dict[str, np.ndarray | tuple[np.ndarray, ...]]]]

    @pytest.fixture
    def _configured_program(
        self,
        backend: gtx_backend.Backend | None,
        static_variant: Sequence[str],
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        grid: base.Grid,
    ) -> Callable[..., None]:
        unused_static_params = set(static_variant) - set(input_data.keys())
        if unused_static_params:
            raise ValueError(
                f"Parameter defined in 'STATIC_PARAMS' not in 'input_data': {unused_static_params}"
            )
        static_args = {name: [input_data[name]] for name in static_variant}
        program = self.PROGRAM.with_backend(backend)  # type: ignore[arg-type]  # TODO(havogt): gt4py should accept `None` in with_backend
        if backend is not None:
            if isinstance(program, FieldOperator):
                if len(static_args) > 0:
                    raise NotImplementedError(
                        "'FieldOperator's do not support static arguments yet."
                    )
            else:
                program.compile(
                    offset_provider=grid.connectivities,
                    enable_jit=False,
                    **static_args,  # type: ignore[arg-type]
                )

        test_func = device_utils.synchronized_function(program, backend=backend)
        return test_func

    @pytest.fixture
    def _properly_allocated_input_data(
        self,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        backend: gtx_backend.Backend | None,
    ) -> dict[str, gtx.Field | tuple[gtx.Field, ...]]:
        # TODO(havogt): this is a workaround,
        # because in the `input_data` fixture provided by the user
        # it does not allocate for the correct device.
        return allocate_data(backend, input_data)

    def test_stencil(
        self: StencilTest,
        benchmark: Any,  # should be `pytest_benchmark.fixture.BenchmarkFixture` but pytest_benchmark is not typed
        grid: base.Grid,
        backend: gtx_backend.Backend | None,
        _properly_allocated_input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        _configured_program: Callable[..., None],
    ) -> None:
        reference_outputs = self.reference(
            _ConnectivityConceptFixer(
                grid  # TODO(havogt): pass as keyword argument (needs fixes in some tests)
            ),
            **{
                k: v.asnumpy() if isinstance(v, gtx.Field) else v
                for k, v in _properly_allocated_input_data.items()
            },
        )

        _configured_program(**_properly_allocated_input_data, offset_provider=grid.connectivities)
        self._verify_stencil_test(
            input_data=_properly_allocated_input_data, reference_outputs=reference_outputs
        )

        if benchmark is not None and benchmark.enabled:
            benchmark(
                _configured_program,
                **_properly_allocated_input_data,
                offset_provider=grid.connectivities,
            )

    def _verify_stencil_test(
        self,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        reference_outputs: dict[str, np.ndarray | tuple[np.ndarray, ...]],
    ) -> None:
        for out in self.OUTPUTS:
            name, refslice, gtslice = (
                (out.name, out.refslice, out.gtslice)
                if isinstance(out, Output)
                else (out, (slice(None),), (slice(None),))
            )

            input_data_name = input_data[name]  # for mypy
            if isinstance(input_data_name, tuple):
                for i_out_field, out_field in enumerate(input_data_name):
                    np.testing.assert_allclose(
                        out_field.asnumpy()[gtslice],
                        reference_outputs[name][i_out_field][refslice],
                        equal_nan=True,
                        err_msg=f"Verification failed for '{name}[{i_out_field}]'",
                    )
            else:
                reference_outputs_name = reference_outputs[name]  # for mypy
                assert isinstance(reference_outputs_name, np.ndarray)
                np.testing.assert_allclose(
                    input_data_name.asnumpy()[gtslice],
                    reference_outputs_name[refslice],
                    equal_nan=True,
                    err_msg=f"Verification failed for '{name}'",
                )

    @staticmethod
    def static_variant(request: pytest.FixtureRequest) -> Sequence[str]:
        """
        Fixture for parametrization over the `STATIC_PARAMS` of the test class.

        Note: the actual `pytest.fixture()`  decoration happens inside `__init_subclass__`,
          when all information is available.
        """
        _, variant = request.param
        return () if variant is None else variant

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # decorate `static_variant` with parametrized fixtures, since the
        # parametrization is only available in the concrete subclass definition
        if cls.STATIC_PARAMS is None:
            # not parametrized, return an empty tuple
            cls.static_variant = staticmethod(pytest.fixture(lambda: ()))  # type: ignore[method-assign, assignment] # we override with a non-parametrized function
        else:
            cls.static_variant = staticmethod(  # type: ignore[method-assign]
                pytest.fixture(params=cls.STATIC_PARAMS.items(), scope="class", ids=lambda p: p[0])(
                    cls.static_variant
                )
            )

        # apply markers to the test function.
        # TODO(egparedes,havogt): use directly pytest.mark as class decorators
        if cls.MARKERS:
            for marker in cls.MARKERS:
                cls.test_stencil = marker(cls.test_stencil)  # type:ignore[method-assign] # we override with a decorated function
