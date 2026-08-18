# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for the `StencilTest` machinery itself.

The conventions enforced by `stencil_tests` are what keep ~200 stencil test suites
consistent, so the enforcement is tested here rather than only implicitly through those
suites (where a hole in a check simply goes unnoticed).
"""

import inspect
import types

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation
from icon4py.model.testing import stencil_tests


# Deliberately also bound under the name a fixture receives the wrapper as: the check must
# reject a fixture reaching for the module, yet accept a parameter that shadows this name.
data_alloc = data_allocation


# -- helpers ---------------------------------------------------------------------------


def valid_reference():
    """A `reference` implementation following the convention."""

    def reference(grid, **kwargs):
        return {}

    return reference


def valid_input_data():
    """An `input_data` fixture function following the convention."""

    def input_data(data_alloc):
        return {}

    return input_data


def make_suite(**namespace):
    """A minimal valid `StencilTest` subclass, with `namespace` merged into its body."""
    body = {
        "reference": stencil_tests.static_reference(valid_reference()),
        "input_data": stencil_tests.input_data_fixture(valid_input_data()),
        **namespace,
    }
    # Not named `Test...`: these are created inside tests and must not be collected.
    return type("Suite", (stencil_tests.StencilTest,), body)


def cell_field(values):
    return gtx.as_field((dims.CellDim,), np.asarray(values, dtype=np.float64))


@pytest.fixture
def grid():
    return simple.simple_grid()


# -- static_reference ------------------------------------------------------------------


class TestStaticReference:
    def test_returns_marked_staticmethod(self):
        marked = stencil_tests.static_reference(valid_reference())

        assert isinstance(marked, staticmethod)
        assert getattr(marked, "__stencil_test_reference__", False)

    def test_accepts_an_existing_staticmethod(self):
        marked = stencil_tests.static_reference(staticmethod(valid_reference()))

        assert isinstance(marked, staticmethod)
        assert getattr(marked, "__stencil_test_reference__", False)

    def test_does_not_wrap_twice(self):
        once = stencil_tests.static_reference(valid_reference())

        assert stencil_tests.static_reference(once) is once

    def test_rejects_non_function(self):
        with pytest.raises(TypeError, match="must be a regular function or a staticmethod"):
            stencil_tests.static_reference(object())

    def test_rejects_wrong_name(self):
        def not_reference(grid, **kwargs): ...

        with pytest.raises(ValueError, match="must be named 'reference'"):
            stencil_tests.static_reference(not_reference)

    def test_rejects_wrong_first_parameter(self):
        def reference(connectivities, **kwargs): ...

        with pytest.raises(ValueError, match=r"must be 'reference\(grid, \.\.\.\)'"):
            stencil_tests.static_reference(reference)

    def test_rejects_empty_signature(self):
        """Regression: this used to raise `IndexError` while checking the convention."""

        def reference(): ...

        with pytest.raises(ValueError, match=r"must be 'reference\(grid, \.\.\.\)'"):
            stencil_tests.static_reference(reference)

    def test_rejects_positional_only_grid(self):
        """`test_and_benchmark` calls `reference(grid=...)`, so positional-only would fail late."""

        def reference(grid, /, **kwargs): ...

        with pytest.raises(ValueError, match="cannot be positional-only"):
            stencil_tests.static_reference(reference)

    def test_reports_the_actual_signature_readably(self):
        def reference(connectivities, other, **kwargs): ...

        with pytest.raises(ValueError) as excinfo:
            stencil_tests.static_reference(reference)

        # expected and actual are rendered the same way, not one joined and one a tuple repr
        assert "but got 'reference(connectivities, other, kwargs)'" in str(excinfo.value)


# -- input_data_fixture ----------------------------------------------------------------


class TestInputDataFixture:
    def test_returns_a_marked_class_scoped_static_fixture(self):
        fixture = stencil_tests.input_data_fixture(valid_input_data())

        # a staticmethod: pytest deprecated class-scoped fixtures defined as instance methods
        assert isinstance(fixture, staticmethod)
        assert getattr(fixture, "__stencil_test_input_fixture__", False)
        assert fixture.__func__._fixture_function_marker.scope == "class"

    def test_forwards_keyword_arguments_to_pytest_fixture(self):
        fixture = stencil_tests.input_data_fixture(params=[1, 2], scope="function")(
            valid_input_data()
        )
        marker = fixture.__func__._fixture_function_marker

        assert getattr(fixture, "__stencil_test_input_fixture__", False)
        assert marker.params == (1, 2)
        assert marker.scope == "function"  # an explicit scope wins over the default

    def test_accepts_an_existing_staticmethod(self):
        fixture = stencil_tests.input_data_fixture(staticmethod(valid_input_data()))

        assert isinstance(fixture, staticmethod)
        assert getattr(fixture, "__stencil_test_input_fixture__", False)

    def test_rejects_non_function(self):
        with pytest.raises(TypeError, match="must be a regular function"):
            stencil_tests.input_data_fixture(object())

    def test_rejects_wrong_name(self):
        def not_input_data(data_alloc): ...

        with pytest.raises(ValueError, match="must be named 'input_data'"):
            stencil_tests.input_data_fixture(not_input_data)

    @pytest.mark.parametrize(
        "func",
        [
            pytest.param(lambda: None, id="no_parameters"),
            pytest.param(lambda self, data_alloc: None, id="self_first"),
            pytest.param(lambda grid, data_alloc: None, id="grid_first"),
        ],
    )
    def test_rejects_wrong_leading_parameters(self, func):
        func.__name__ = "input_data"

        with pytest.raises(ValueError, match=r"must be 'input_data\(data_alloc, \.\.\.\)'"):
            stencil_tests.input_data_fixture(func)

    def test_rejects_direct_data_allocation_call(self):
        def input_data(data_alloc, grid):
            return {"a": data_allocation.zero_field(grid, dims.CellDim)}

        with pytest.raises(TypeError, match="should not call 'data_allocation' functions"):
            stencil_tests.input_data_fixture(input_data)

    def test_rejects_a_call_hidden_in_a_nested_function(self):
        """A scan of only the fixture's own code object would miss this."""

        def input_data(data_alloc, grid):
            def build():
                return data_allocation.zero_field(grid, dims.CellDim)

            return {"a": build()}

        with pytest.raises(TypeError, match="should not call 'data_allocation' functions"):
            stencil_tests.input_data_fixture(input_data)

    def test_rejects_a_call_hidden_in_a_comprehension(self):
        def input_data(data_alloc, grid):
            return {"a": [data_allocation.zero_field(grid, dims.CellDim) for _ in range(1)]}

        with pytest.raises(TypeError, match="should not call 'data_allocation' functions"):
            stencil_tests.input_data_fixture(input_data)

    def test_is_skipped_when_no_source_is_available(self):
        """Dynamically generated fixtures cannot be inspected; they must not blow up."""
        namespace: dict = {}
        exec("def input_data(data_alloc):\n    return {}", namespace)

        assert stencil_tests.input_data_fixture(namespace["input_data"]) is not None

    def test_rejects_a_directly_imported_constructor(self):
        """A `from ...data_allocation import zero_field` binds the function, not the module."""
        zero_field = data_allocation.zero_field

        def input_data(data_alloc, grid):
            return {"a": zero_field(grid, dims.CellDim)}

        with pytest.raises(TypeError, match="should not call 'data_allocation' functions"):
            stencil_tests.input_data_fixture(input_data)

    def test_rejects_positional_only_data_alloc(self):
        """pytest injects fixtures by keyword, so positional-only would fail at run time."""

        def input_data(data_alloc, /): ...

        with pytest.raises(ValueError, match="cannot be positional-only"):
            stencil_tests.input_data_fixture(input_data)

    def test_rejects_data_allocation_from_an_enclosing_scope(self):
        module = data_allocation

        def input_data(data_alloc, grid):
            return {"a": module.zero_field(grid, dims.CellDim)}

        with pytest.raises(TypeError, match="should not call 'data_allocation' functions"):
            stencil_tests.input_data_fixture(input_data)

    def test_accepts_a_parameter_shadowing_the_module_alias(self):
        """
        Regression: the `data_alloc` parameter must not be read as the `data_alloc` global.

        This module binds `data_alloc` to `data_allocation`, and every fixture receives the
        wrapper under that same name, so the check has to discount the fixture's own
        parameters.
        """

        def input_data(data_alloc):
            return {"a": data_alloc.zero_field(dims.CellDim)}

        assert stencil_tests.input_data_fixture(input_data) is not None


# -- StencilTest.__init_subclass__ -----------------------------------------------------


class TestSuiteConventions:
    def test_attaches_a_test_function_named_after_the_subclass(self):
        suite = make_suite()

        assert "test_Suite" in vars(suite)
        assert vars(suite)["test_Suite"] is stencil_tests.test_and_benchmark

    def test_rejects_missing_reference(self):
        with pytest.raises(TypeError, match="does not implement the required 'reference'"):

            class Suite(stencil_tests.StencilTest):
                input_data = stencil_tests.input_data_fixture(valid_input_data())

    def test_rejects_missing_input_data(self):
        with pytest.raises(TypeError, match="does not implement the required 'input_data'"):

            class Suite(stencil_tests.StencilTest):
                reference = stencil_tests.static_reference(valid_reference())

    def test_rejects_undecorated_reference(self):
        with pytest.raises(TypeError, match="must be decorated with '@static_reference'"):

            class Suite(stencil_tests.StencilTest):
                reference = staticmethod(valid_reference())
                input_data = stencil_tests.input_data_fixture(valid_input_data())

    def test_rejects_undecorated_input_data(self):
        with pytest.raises(TypeError, match="must be decorated with '@input_data_fixture'"):

            class Suite(stencil_tests.StencilTest):
                reference = stencil_tests.static_reference(valid_reference())
                input_data = pytest.fixture(valid_input_data())

    def test_accepts_members_inherited_from_another_suite(self):
        """Regression: the members were looked up in `cls.__dict__`, raising `KeyError`."""
        base = make_suite()

        derived = type("Derived", (base,), {})

        assert "test_Derived" in vars(derived)

    def test_static_variant_is_empty_without_static_params(self):
        suite = make_suite()

        fixture = inspect.getattr_static(suite, "static_variant").__func__
        assert fixture._fixture_function_marker.params is None
        assert fixture._get_wrapped_function()() == ()

    @pytest.mark.parametrize(
        ("param", "expected"),
        [(("domain", ("horizontal_start",)), ("horizontal_start",)), (("none", None), ())],
    )
    def test_static_variant_reads_the_variant_off_the_request(self, param, expected):
        suite = make_suite(STATIC_PARAMS={"domain": ("horizontal_start",)})
        variant_of = inspect.getattr_static(suite, "static_variant").__func__
        request = types.SimpleNamespace(param=param)

        assert variant_of._get_wrapped_function()(request) == expected

    def test_configured_program_rejects_static_params_absent_from_input_data(self):
        """`STATIC_PARAMS` names arguments of `input_data`; a typo should say so."""
        build = inspect.getattr_static(
            stencil_tests.StencilTest, "configured_program"
        )._get_wrapped_function()

        with pytest.raises(ValueError, match=r"not in 'input_data': \{'typo'\}"):
            build(
                stencil_tests.StencilTest(),
                backend_like=None,
                static_variant=("typo",),
                input_data={"real": None},
                grid=None,
            )

    def test_static_variant_is_parametrized_from_static_params(self):
        static_params = {"none": (), "domain": ("horizontal_start",)}

        suite = make_suite(STATIC_PARAMS=static_params)

        marker = inspect.getattr_static(suite, "static_variant").__func__._fixture_function_marker
        assert list(marker.params) == list(static_params.items())
        assert marker.scope == "class"


# -- Output ----------------------------------------------------------------------------


class TestOutput:
    def test_defaults_select_the_whole_field(self):
        """`verify_data` wraps a bare name in an `Output`, so the defaults must be no-ops."""
        out = stencil_tests.Output("field")

        assert out.refslice == (slice(None),)
        assert out.gtslice == (slice(None),)


# -- connectivities_asnumpy ------------------------------------------------------------


class StubGrid:
    """Minimal stand-in exposing only what the connectivities view uses."""

    def __init__(self, connectivities):
        self.connectivities = connectivities

    def get_connectivity(self, offset):
        return self.connectivities[offset if isinstance(offset, str) else offset.value]


class TestConnectivitiesAsNumpy:
    def test_lookup_by_name_and_by_field_offset_agree(self, grid):
        view = stencil_tests.connectivities_asnumpy(grid)

        assert isinstance(view[dims.E2C], np.ndarray)
        np.testing.assert_array_equal(view[dims.E2C], view["E2C"])
        np.testing.assert_array_equal(view[dims.E2C], grid.get_connectivity("E2C").asnumpy())

    def test_iteration_and_length_cover_the_neighbor_tables(self, grid):
        view = stencil_tests.connectivities_asnumpy(grid)

        expected = {
            key
            for key, connectivity in grid.connectivities.items()
            if gtx_common.is_neighbor_table(connectivity)
        }
        assert set(view) == expected
        assert len(view) == len(expected)

    def test_non_neighbor_table_entries_are_skipped(self, grid):
        stub = StubGrid({**dict(grid.connectivities), "Koff": dims.KDim})
        view = stencil_tests.connectivities_asnumpy(stub)

        assert "Koff" not in set(view)
        assert len(view) == len(set(view))
        with pytest.raises(KeyError, match="is not a neighbor table"):
            view["Koff"]

    def test_honours_the_mapping_contract_for_a_missing_key(self, grid):
        """`get` and `in` are built on `__getitem__`, so it has to raise `KeyError`."""
        view = stencil_tests.connectivities_asnumpy(grid)

        assert view.get("NoSuchOffset", "default") == "default"
        assert "NoSuchOffset" not in view
        assert "E2C" in view
        with pytest.raises(KeyError):
            view["NoSuchOffset"]


# -- DataAllocationWrapper -------------------------------------------------------------


class TestDataAllocationWrapper:
    @pytest.fixture
    def wrapper(self, grid):
        return stencil_tests.DataAllocationWrapper(grid=grid, allocator=None)

    @pytest.mark.parametrize(
        "construct",
        [
            pytest.param(lambda w: w.constant_field(1.0, dims.CellDim), id="constant_field"),
            pytest.param(lambda w: w.index_field(dims.CellDim), id="index_field"),
            pytest.param(lambda w: w.random_field(dims.CellDim), id="random_field"),
            pytest.param(lambda w: w.random_mask(dims.CellDim), id="random_mask"),
            pytest.param(lambda w: w.random_sign(dims.CellDim), id="random_sign"),
            pytest.param(lambda w: w.zero_field(dims.CellDim), id="zero_field"),
        ],
    )
    def test_every_constructor_binds_the_grid(self, wrapper, grid, construct):
        assert construct(wrapper).shape == (grid.num_cells,)

    def test_binds_the_grid(self, wrapper, grid):
        assert wrapper.zero_field(dims.CellDim).shape == (grid.num_cells,)
        assert wrapper.random_field(dims.EdgeDim, dims.KDim).shape == (
            grid.num_edges,
            grid.num_levels,
        )

    def test_forwards_keyword_arguments(self, wrapper, grid):
        field = wrapper.zero_field(dims.CellDim, dims.KDim, extend={dims.KDim: 1})

        assert field.shape == (grid.num_cells, grid.num_levels + 1)
        assert np.all(field.asnumpy() == 0.0)
        assert np.all(wrapper.constant_field(3.5, dims.CellDim).asnumpy() == 3.5)

    def test_random_field_respects_bounds(self, wrapper):
        values = wrapper.random_field(dims.CellDim, low=2.0, high=3.0).asnumpy()

        assert np.all((values >= 2.0) & (values < 3.0))

    def test_from_numpy_hands_host_data_to_the_allocator(self, wrapper, grid):
        """
        A fixture that must build its input with NumPy has to hand the array over.

        Writing into an already allocated field instead only works while its buffer is host
        memory; under a GPU backend it raises, or is silently discarded when the write goes
        through `asnumpy()`, which returns a copy.
        """
        values = np.arange(grid.num_cells * grid.num_levels, dtype=np.int32).reshape(
            grid.num_cells, grid.num_levels
        )

        field = wrapper.from_numpy(values, dims.CellDim, dims.KDim)

        assert field.domain.dims == (dims.CellDim, dims.KDim)
        np.testing.assert_array_equal(field.asnumpy(), values)

    def test_connectivity_field_returns_a_plain_field(self, wrapper, grid):
        """
        Regression: dropping `allocate_data` also dropped the conversion it performed.

        A raw `NeighborTable` cannot be passed as a program argument, so stencils that
        consume a connectivity as data need it re-allocated as an ordinary field.
        """
        field = wrapper.connectivity_field("E2C")

        assert isinstance(field, gtx.Field)
        assert not gtx_common.is_neighbor_table(field)
        np.testing.assert_array_equal(field.asnumpy(), grid.get_connectivity("E2C").asnumpy())

    def test_signatures_stay_in_sync_with_data_allocation(self):
        """The wrapper duplicates the wrapped signatures, so guard against drift."""
        for name in (
            "constant_field",
            "index_field",
            "random_field",
            "random_mask",
            "random_sign",
            "zero_field",
        ):
            wrapped = [
                param
                for param in inspect.signature(getattr(data_alloc, name)).parameters.values()
                if param.name not in ("grid", "allocator")
            ]
            method = [
                param
                for param in inspect.signature(
                    getattr(stencil_tests.DataAllocationWrapper, name)
                ).parameters.values()
                if param.name != "self"
            ]
            assert wrapped == method, f"'{name}' has drifted from 'data_allocation.{name}'"


# -- StencilTest.verify_data -----------------------------------------------------------


class TestVerifyData:
    def test_passes_when_the_output_matches(self):
        suite = make_suite(OUTPUTS=("out",))()

        suite.verify_data(
            input_data={"out": cell_field([1.0, 2.0, 3.0])},
            reference_outputs={"out": np.array([1.0, 2.0, 3.0])},
        )

    def test_fails_and_names_the_output(self):
        suite = make_suite(OUTPUTS=("out",))()

        with pytest.raises(AssertionError, match="Verification failed for 'out'"):
            suite.verify_data(
                input_data={"out": cell_field([1.0, 2.0, 3.0])},
                reference_outputs={"out": np.array([1.0, 2.0, 9.0])},
            )

    def test_honours_the_output_slices(self):
        out = stencil_tests.Output("out", refslice=(slice(1, None),), gtslice=(slice(1, None),))
        suite = make_suite(OUTPUTS=(out,))()

        # the excluded leading entry differs, the compared tail does not
        suite.verify_data(
            input_data={"out": cell_field([42.0, 2.0, 3.0])},
            reference_outputs={"out": np.array([-1.0, 2.0, 3.0])},
        )

    def test_applies_each_slice_to_its_own_side(self):
        """`gtslice` indexes the computed field and `refslice` the reference, not vice versa."""
        out = stencil_tests.Output("out", refslice=(slice(None, -1),), gtslice=(slice(1, None),))
        suite = make_suite(OUTPUTS=(out,))()

        # computed[1:] == [1, 2] == reference[:-1]; swapping the two slices compares
        # [9, 1] against [2, 9] instead, so this fails if they are applied to the wrong side
        suite.verify_data(
            input_data={"out": cell_field([9.0, 1.0, 2.0])},
            reference_outputs={"out": np.array([1.0, 2.0, 9.0])},
        )

    def test_verifies_every_element_of_a_tuple_output(self):
        suite = make_suite(OUTPUTS=("out",))()
        fields = (cell_field([1.0, 2.0]), cell_field([3.0, 4.0]))

        suite.verify_data(
            input_data={"out": fields},
            reference_outputs={"out": (np.array([1.0, 2.0]), np.array([3.0, 4.0]))},
        )

        with pytest.raises(AssertionError, match=r"Verification failed for 'out\[1\]'"):
            suite.verify_data(
                input_data={"out": fields},
                reference_outputs={"out": (np.array([1.0, 2.0]), np.array([3.0, 9.0]))},
            )

    def test_rejects_a_tuple_output_of_the_wrong_length(self):
        suite = make_suite(OUTPUTS=("out",))()

        with pytest.raises(ValueError):
            suite.verify_data(
                input_data={"out": (cell_field([1.0]), cell_field([2.0]))},
                reference_outputs={"out": (np.array([1.0]),)},
            )
