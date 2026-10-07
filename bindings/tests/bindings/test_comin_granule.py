# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the plugin's granule ('_granule.py') against py2fgen's bindings: on the same inputs,
'grid_init', 'diffusion_init' and 'diffusion_run' call the same functions of icon4py's API in the
same order with equal arguments as 'grid_wrapper.grid_init', 'diffusion_wrapper.diffusion_init'
and 'diffusion_run'. The constructors of the grid, the vertical grid, the geometry, the states and
the granule are replaced by recorders; what computes the arguments in between (the backend, the
allocator, the 'zd_*' fields, the RBF coefficients, DiffusionParams, the dummy fields) runs.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from gt4py import next as gtx

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.bindings import (
    common as wrapper_common,
    config as wrapper_config,
    debug_utils,
    diffusion_wrapper,
    grid_wrapper,
)
from icon4py.bindings.comin import _arguments, _config, _granule
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import vertical

from .test_comin_config import PY2FGEN_DIFFUSION, excerpt


@dataclasses.dataclass(frozen=True)
class Made:
    """What a replaced constructor returned: the n-th object of its kind."""

    label: str

    def is_single_rank(self) -> bool:  # as a ProcessProperties
        return False

    def run(self, **kwargs: Any) -> None:  # as the Diffusion granule
        CALLS.append(("Diffusion.run", (), kwargs))


CALLS: list[tuple[str, tuple, dict[str, Any]]] = []


def made(name: str, result: Callable[[str], Any] | None = None) -> Callable[..., Any]:
    def constructor(*args: Any, **kwargs: Any) -> Any:
        CALLS.append((name, args, kwargs))
        label = f"{name}#{sum(c[0] == name for c in CALLS)}"
        return Made(label) if result is None else result(label)

    return constructor


def recorded(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        CALLS.append((name, args, kwargs))
        return function(*args, **kwargs)

    return call


@pytest.fixture
def api(monkeypatch):
    """icon4py's API as both routes see it, with recorders."""
    CALLS.clear()

    def triple(label: str) -> tuple[Made, Made, Made]:  # construct_decomposition's result
        return Made(f"{label}.props"), Made(f"{label}.info"), Made(f"{label}.exchange")

    patches = [
        (
            wrapper_common,
            "select_backend",
            recorded("select_backend", wrapper_common.select_backend),
        ),
        (model_backends, "get_allocator", recorded("get_allocator", model_backends.get_allocator)),
        (wrapper_common, "construct_decomposition", made("construct_decomposition", triple)),
        (wrapper_common, "construct_icon_grid", made("construct_icon_grid")),
        (debug_utils, "print_grid_decomp_info", made("print_grid_decomp_info")),
        (vertical, "VerticalGridConfig", made("VerticalGridConfig")),
        (vertical, "VerticalGrid", made("VerticalGrid")),
        (grid_states, "EdgeParams", made("EdgeParams")),
        (grid_states, "CellParams", made("CellParams")),
        (data_alloc, "scattered_field", recorded("scattered_field", data_alloc.scattered_field)),
        (gtx, "as_field", recorded("as_field", gtx.as_field)),
        (gtx, "wait_for_compilation", made("wait_for_compilation")),
        (
            wrapper_common,
            "cached_dummy_field_factory",
            recorded("cached_dummy_field_factory", wrapper_common.cached_dummy_field_factory),
        ),
        (wrapper_config, "WAIT_FOR_COMPILATION", True),  # as the plugin set it for py2fgen's
    ]
    for module in (diffusion_wrapper, _granule):  # imported by name
        for name in ("DiffusionParams", "DiffusionMetricState", "DiffusionInterpolationState"):
            real = getattr(module, name)
            patches.append(
                (module, name, recorded(name, real) if name == "DiffusionParams" else made(name))
            )
        patches += [
            (module, "Diffusion", made("Diffusion")),
            (module, "PrognosticState", made("PrognosticState")),
            (module, "DiffusionDiagnosticState", made("DiffusionDiagnosticState")),
        ]
    for module, name, value in patches:
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(grid_wrapper, "grid_state", None)
    monkeypatch.setattr(diffusion_wrapper, "granule", None)
    yield CALLS
    CALLS.clear()


SIZES = {
    "Cell": 6,
    "Edge": 9,
    "Vertex": 5,
    "K": 4,
    dims.KHalfDim.value: 5,
    "C2E2CO": 4,
    "V2E": 6,
    "V2C": 6,
}
"""The toy grid: cells, edges, vertices, levels, half levels, and neighbours (3 unless given)."""


def inputs(params: dict[str, _arguments.Param], seed: int) -> dict[str, Any]:
    """Arrays of every form of the parameter table (NumPy; Fields where a Field), scalars."""
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {}
    for name, kind in params.items():
        if isinstance(kind, _arguments.ArrayParam):
            if kind.dims is not None:
                shape = tuple(SIZES.get(d.value, 3) for d in kind.dims)
            else:
                shape = (7,) * kind.rank
            array = np.asfortranarray(rng.integers(1, 4, shape)).astype(kind.dtype)
            if kind.dtype == np.float64:
                array = np.asfortranarray(rng.normal(size=shape))
            values[name] = array if kind.dims is None else gtx.as_field(list(kind.dims), array)
        elif kind is float:
            values[name] = float(rng.normal())
        elif kind is int:
            values[name] = int(rng.integers(1, 9))
        elif kind is bool:
            values[name] = True
    return values


def diffusion_inputs(lists: bool) -> dict[str, Any]:
    values = inputs(_granule.DIFFUSION_INIT, seed=2)
    nz = 3
    values.update(
        wgtfac_c=gtx.as_field(
            [dims.CellDim, dims.KHalfDim], np.linspace(0.0, 1.0, 30).reshape(6, 5)
        ),
        rbf_vec_coeff_v=np.asfortranarray(np.random.default_rng(3).normal(size=(6, 2, 5))),
        zd_cellidx=np.asfortranarray(np.array([[1, 3, 6]] * 4, dtype=np.int32)),
        zd_vertidx=np.asfortranarray(
            np.array([[2, 3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 4]], dtype=np.int32)
        ),
        zd_intcoef=np.asfortranarray(np.linspace(0.1, 0.9, 3 * nz).reshape(3, nz)),
        zd_diffcoef=np.linspace(0.2, 0.3, nz),
    )
    if not lists:
        values.update(zd_cellidx=None, zd_vertidx=None, zd_intcoef=None, zd_diffcoef=None)
    # what the plugin reads from ICON's namelist output besides the DiffusionConfig
    kinds = {p: _granule.DIFFUSION_INIT[p] for p in ("ndyn_substeps", "nudge_max_coeff")}
    read, errors = _config.arguments(excerpt(), "diffusion_init", kinds)
    assert errors == []
    values.update(read, backend=0)
    return values


def same(a: Any, b: Any) -> bool:
    """Equal arguments: Fields and arrays by dimensions, dtype and values; the rest by value."""
    if isinstance(a, gtx.Field) or isinstance(b, gtx.Field):
        return (
            isinstance(a, gtx.Field)
            and isinstance(b, gtx.Field)
            and tuple(a.domain) == tuple(b.domain)
            and a.dtype == b.dtype
            and np.array_equal(a.asnumpy(), b.asnumpy())
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if callable(a) and not isinstance(a, Made):  # an allocator, a cached factory
        return type(a) is type(b)
    return type(a) is type(b) and a == b


def compare(theirs: list, ours: list) -> None:
    assert [c[0] for c in ours] == [c[0] for c in theirs]
    for (name, args, kwargs), (_, our_args, our_kwargs) in zip(theirs, ours, strict=True):
        assert same(args, our_args), name
        assert kwargs.keys() == our_kwargs.keys(), name
        for key in kwargs:
            assert same(kwargs[key], our_kwargs[key]), f"{name}: {key}"


@pytest.mark.parametrize("lists", [True, False], ids=["zd_lists", "no_zd_lists"])
def test_the_same_calls_as_py2fgens_bindings(api, lists):
    grid = inputs(_granule.GRID_INIT, seed=1)
    grid["backend"] = 0  # BackendIntEnum.DEFAULT
    init = diffusion_inputs(lists)
    config, errors = _granule.diffusion_config(excerpt())
    assert errors == [] and config is not None
    run = inputs(_granule.DIFFUSION_RUN, seed=4)
    run.update(hdef_ic=None, dwdx=None, dtime=10.0, linit=True)
    api.clear()  # the inputs' Fields

    # py2fgen's bindings (the undecorated functions, with the 25 values of the configuration)
    grid_wrapper.grid_init.__wrapped__(**grid)
    py2fgen_init = {k: v for k, v in init.items()} | PY2FGEN_DIFFUSION
    diffusion_wrapper.diffusion_init.__wrapped__(**py2fgen_init)
    diffusion_wrapper.diffusion_run.__wrapped__(**run)
    theirs = list(api)
    api.clear()
    # the plugin's granule
    granule = _granule.Granule()
    granule.grid_init(**grid)
    granule.diffusion_init(**init, config=config)
    granule.diffusion_run(**run)
    ours = list(api)

    names = [c[0] for c in ours]
    assert names[:9] == [
        "select_backend",
        "get_allocator",
        "construct_decomposition",
        "construct_icon_grid",
        "print_grid_decomp_info",
        "VerticalGridConfig",
        "VerticalGrid",
        "EdgeParams",
        "CellParams",
    ]
    assert names[-3:] == ["PrognosticState", "DiffusionDiagnosticState", "Diffusion.run"]
    assert names.count("scattered_field") == (3 if lists else 0)
    compare(theirs, ours)
    # the configuration the granule got: py2fgen's from 23 of the 25 values, ours from the
    # namelist ('ndyn_substeps' and 'nudge_max_coeff' are compared with Diffusion's arguments)
    (params,) = [c for c in ours if c[0] == "DiffusionParams"]
    (their_params,) = [c for c in theirs if c[0] == "DiffusionParams"]
    assert params[1][0] == their_params[1][0] == config
    # the state
    assert granule.grid_state is not None and granule.diffusion == Made("Diffusion#1")
    assert granule.grid_state.exchange_runtime == Made("construct_decomposition#1.exchange")
    granule.release()
    assert (granule.grid_state, granule.diffusion, granule.dummy_field_factory) == (
        None,
        None,
        None,
    )


def test_the_order_of_the_calls_is_needed():
    granule = _granule.Granule()
    with pytest.raises(RuntimeError, match="'grid_init' must run before 'diffusion_init'"):
        granule.diffusion_init(**diffusion_inputs(False), config=None)
    with pytest.raises(RuntimeError, match="'diffusion_init' must run before 'diffusion_run'"):
        granule.diffusion_run(**inputs(_granule.DIFFUSION_RUN, seed=4))


def test_the_configuration_object_is_icon4pys():
    assert _granule.DIFFUSION_INIT["config"] is _granule.DiffusionConfig
    assert _config.LOUTSHS_FIELD == "loutshs"
    arguments = set(_config.ARGUMENTS["diffusion_init"])
    assert arguments == {"ndyn_substeps", "nudge_max_coeff", "backend"}
