# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py import eve
from gt4py.next import backend as gtx_backend
from gt4py.next.ffront.decorator import FieldOperator, Program

from icon4py.model.common import type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation, device_utils


if TYPE_CHECKING:
    import numpy.typing as npt


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


class DataAlloc(Protocol):
    """
    This protocol mimics the 'icon4py.model.common.utils.data_allocation',
    but with 'backend' bound in the respective functions.
    """

    @staticmethod
    def random_field(
        *dims: gtx.Dimension,
        low: float = -1.0,
        high: float = 1.0,
        dtype: npt.DTypeLike | None = None,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...

    @staticmethod
    def random_mask(
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = None,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...

    @staticmethod
    def zero_field(
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = ta.wpfloat,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...

    @staticmethod
    def constant_field(
        value: float,
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = ta.wpfloat,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...


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
    STATIC_PARAMS: ClassVar[dict[str, Sequence[str] | None] | None] = None

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
    def data_alloc(self, backend: gtx_backend.Backend | None, grid: base.Grid) -> DataAlloc:
        """
        Convenience fixture to provide data allocation function with backend and grid already bound.
        """

        class data_alloc_impl:
            def __getattr__(self, name: str) -> Callable[..., gtx.Field]:
                if not hasattr(DataAlloc, name):
                    raise AttributeError(
                        f"Data allocation function '{name}' not found. Maybe missing in the 'DataAlloc' protocol?"
                    )
                alloc_fun = getattr(data_allocation, name)
                return functools.partial(alloc_fun, grid, backend=backend)

        return data_alloc_impl()

    def test_stencil(
        self: StencilTest,
        benchmark: Any,  # should be `pytest_benchmark.fixture.BenchmarkFixture` but pytest_benchmark is not typed
        grid: base.Grid,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        _configured_program: Callable[..., None],
    ) -> None:
        reference_outputs = self.reference(
            _ConnectivityConceptFixer(
                grid  # TODO(havogt): pass as keyword argument (needs fixes in some tests)
            ),
            **{k: v.asnumpy() if isinstance(v, gtx.Field) else v for k, v in input_data.items()},
        )

        _configured_program(**input_data, offset_provider=grid.connectivities)
        self._verify_stencil_test(input_data=input_data, reference_outputs=reference_outputs)

        if benchmark is not None and benchmark.enabled:
            benchmark(
                _configured_program,
                **input_data,
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
