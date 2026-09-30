# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the dual-channel check ('_dual.py'), the plugin's source table, the owner-mask
observation and the probe mode, against the fake ComIn and ICON of 'test_comin_marshal.py'.
"""

import math
import struct
import types

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import grid_wrapper, icon4py_export
from icon4py.bindings.comin import _dual, _marshal, plugin
from icon4py.model.common import dimension as dims
from icon4py.tools import py2fgen

from .test_comin_marshal import (  # fixtures: calls, logs, restore_package_logger, wrappers
    REAL,
    FakeComIn,
    FakeIcon,
    calls,
    fortran_buffer,
    logs,
    restore_package_logger,
    wrappers,
)


# ---- comparisons ------------------------------------------------------------------------------


def test_identical_with_padding():
    new = np.arange(24, dtype=np.int32).reshape(8, 3)
    result = _dual.compare_arrays(new, new.copy(order="F"), real=5)
    assert result.identical
    assert result.details == "int32 (8, 3); real 15: 0 differ; padding 9: 0 differ"


def test_differing_real_entry():
    new = np.linspace(1.0, 2.0, 12).reshape(6, 2)
    old = new.copy()
    old[3, 1] = np.nextafter(old[3, 1], np.inf)
    old[4, 0] = old[4, 0] * 2
    result = _dual.compare_arrays(new, old, real=5)
    assert not result.identical and (result.real_differ, result.padding_differ) == (2, 0)
    assert "real 10: 2 differ, first (3, 1), max abs" in result.details
    one_ulp = _dual.compare_arrays(new[3:4, 1], old[3:4, 1])
    assert one_ulp.details.endswith("max ulp 1; padding 0: 0 differ")


def test_padding_only_difference_is_identical():
    new = np.zeros(10, dtype=np.float64)
    old = new.copy()
    old[7:] = np.inf  # e.g. 1/0 beyond the local edges
    result = _dual.compare_arrays(new, old, real=7)
    assert result.identical and result.padding_differ == 3
    assert result.details == "float64 (10,); real 7: 0 differ; padding 3: 3 differ"
    assert not _dual.compare_arrays(new, old).identical  # no padding: all entries are real


def test_padding_along_another_axis():
    new = np.zeros((6, 2, 5))
    old = new.copy()
    old[:, :, 4] = 1.0
    assert _dual.compare_arrays(new, old, real=4, axis=2).identical
    assert not _dual.compare_arrays(new, old, real=5, axis=2).identical


def test_nan_payload_and_signed_zero():
    nan_a = struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0]
    nan_b = struct.unpack("<d", bytes.fromhex("020000000000f87f"))[0]
    new = np.array([nan_a, 0.0, 1.0])
    same_nan = np.array([nan_a, 0.0, 1.0])
    assert _dual.compare_arrays(new, same_nan).identical  # NaN == NaN when the bits are equal
    other_nan = np.array([nan_b, 0.0, 1.0])
    assert not _dual.compare_arrays(new, other_nan).identical
    negative_zero = np.array([nan_a, -0.0, 1.0])
    result = _dual.compare_arrays(new, negative_zero)
    assert not result.identical and "first (1,), max abs 0" in result.details


def test_bool_and_int_arrays():
    mask = np.array([True, False, True, True])
    flipped = mask.copy()
    flipped[2] = False
    result = _dual.compare_arrays(mask, flipped, real=3)
    assert not result.identical and result.details == (
        "bool (4,); real 3: 1 differ, first (2,); padding 1: 0 differ"
    )
    ints = np.array([1, 2, 3], dtype=np.int32)
    result = _dual.compare_arrays(ints, ints + np.array([0, 0, 5], dtype=np.int32))
    assert "max abs 5, max rel 0.625" in result.details


@pytest.mark.parametrize(
    "new, old, details",
    [
        (np.zeros(3, np.int32), np.zeros(3, np.int64), "int32 (3,) vs int64 (3,)"),
        (np.zeros(3), np.zeros(4), "float64 (3,) vs float64 (4,)"),
        (None, np.zeros(3), "present only on the reference"),
        (np.zeros(3), None, "present only on the new route"),
    ],
)
def test_type_shape_and_presence(new, old, details):
    result = _dual.compare_arrays(new, old)
    assert not result.identical and result.details == details


def test_absent_on_both_routes():
    assert _dual.compare_arrays(None, None).identical


def test_squeeze5():
    # ComIn's 3-D field (nproma, nlev, nblks, 1, 1) against py2fgen's (nproma, nlev)
    field = np.asfortranarray(np.arange(4 * 3, dtype=np.float64).reshape(4, 3, 1, 1, 1))
    view = _dual.squeeze5(field, rank=2, block_axis=2)
    assert view.shape == (4, 3) and np.shares_memory(view, field)
    assert _dual.compare_arrays(view, field[:, :, 0, 0, 0].copy()).identical
    # a 2-D field (nproma, nblks, 1, 1, 1)
    surface = np.zeros((4, 1, 1, 1, 1))
    assert _dual.squeeze5(surface, rank=1, block_axis=1).shape == (4,)
    # py2fgen-shaped argument variables (rank r, padded with 1)
    assert _dual.squeeze5(np.zeros((4, 3, 1, 1, 1)), rank=2).shape == (4, 3)
    with pytest.raises(ValueError, match="2 blocks"):
        _dual.squeeze5(np.zeros((4, 3, 2, 1, 1)), rank=2, block_axis=2)
    with pytest.raises(ValueError, match="Cannot squeeze"):
        _dual.squeeze5(np.zeros((4, 3, 1, 2, 1)), rank=2, block_axis=2)


def test_first_block_of_descriptive_data():
    edge_idx = np.asfortranarray(np.arange(18, dtype=np.int32).reshape(6, 1, 3))
    assert _dual.first_block(edge_idx, 1).shape == (6, 3)
    with pytest.raises(ValueError, match="nblks == 1"):
        _dual.first_block(np.zeros((6, 2, 3)), 1)


def test_scalars():
    assert _dual.compare_scalars(0.1, 0.1).identical
    assert "0x1.999999999999ap-4" in _dual.compare_scalars(0.1, 0.1).details
    assert not _dual.compare_scalars(0.0, -0.0).identical
    assert not _dual.compare_scalars(1, True).identical  # a bool is not an int here
    assert not _dual.compare_scalars(1, 1.0).identical
    assert _dual.compare_scalars(True, True).identical
    assert not _dual.compare(3, 4).identical


def test_pointers():
    a = _dual.Pointer(device=0x1000, host=0x2000, shape=(4, 3), present=True)
    assert _dual.compare(a, _dual.Pointer(0x1000, 0x2000, (4, 3), True)).identical
    for other in (
        _dual.Pointer(0x1008, 0x2000, (4, 3), True),
        _dual.Pointer(0x1000, 0x2000, (4, 4), True),
        _dual.Pointer(0x1000, 0x2000, (4, 3), False),
    ):
        assert not _dual.compare(a, other).identical
    assert not _dual.compare(a, np.zeros(3)).identical


def test_host_array_of_a_field():
    array = np.arange(5.0)
    field = gtx.as_field([dims.CellDim], array)
    assert np.array_equal(_dual.host_array(field), array)
    assert _dual.host_array(None) is None


# ---- the self-test perturbation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        np.linspace(0.5, 1.5, 6).reshape(3, 2, order="F"),
        np.array([True, False, True]),
        np.arange(4, dtype=np.int32),
        np.arange(4, dtype=np.float32),
    ],
)
def test_perturb_changes_one_element_of_a_copy(value):
    before = value.copy()
    perturbed = _dual.perturb(value)
    assert np.array_equal(value, before) and not np.shares_memory(value, perturbed)
    result = _dual.compare_arrays(perturbed, value, real=1)
    assert not result.identical and (result.real_differ, result.padding_differ) == (1, 0)
    if value.dtype.kind == "f":
        assert "max ulp 1" in result.details


def test_perturb_scalars_and_pointers():
    assert _dual.perturb(True) is False
    assert _dual.perturb(3) == 4
    assert _dual.perturb(1.0) == math.nextafter(1.0, math.inf)
    assert _dual.perturb(math.nan) == 0.0
    assert _dual.perturb(_dual.Pointer(0x10, None, (2,), True)).device == 0x18
    assert _dual.perturb(_dual.Pointer(None, 0x20, (2,), True)).host == 0x28
    assert _dual.perturb(None) is None


# ---- the checker ------------------------------------------------------------------------------


def items(new_c, old_c):
    return [
        _dual.Item("a", _dual.OLD, None, None),  # never compared
        _dual.Item("c", _dual.NEW, new_c, old_c, real=3),
        _dual.Item("o", _dual.OBSERVE, np.array([1, 2]), np.array([1, 3])),
        _dual.Item("s", _dual.NEW, 0.5, 0.5),
    ]


def test_checker_report_logs_everything():
    lines: list[str] = []
    checker = _dual.Checker(_dual.REPORT, frozenset(), lines.append)
    summary = checker.check("EP pass 1", items(np.zeros(4), np.ones(4)), first=True)
    checker.log_summary("EP pass 1", summary)
    assert lines == [
        "dual EP pass 1 NEW c vs old: differ (float64 (4,); real 3: 3 differ, first (0,),"
        " max abs 1, max rel 1, max ulp 4607182418800017408; padding 1: 1 differ)",
        "dual EP pass 1 OBSERVE o vs old: differ (int64 (2,); real 2: 1 differ, first (1,),"
        " max abs 1, max rel 0.333; padding 0: 0 differ)",
        "dual EP pass 1 NEW s vs old: identical (float 0.5 (0x1.0000000000000p-1, bits"
        " 000000000000e03f))",
        "dual EP pass 1: 3 checked (NEW 2, OBSERVE 1), 1 identical, 2 differ (NEW 1, OBSERVE 1);"
        " selftest 0",
    ]


def test_checker_later_calls_log_differences_only():
    lines: list[str] = []
    checker = _dual.Checker(_dual.REPORT, frozenset(), lines.append)
    checker.check("EP call 2", items(np.zeros(4), np.zeros(4)), first=False)
    assert [line.split(":")[0] for line in lines] == ["dual EP call 2 OBSERVE o vs old"]


def test_checker_strict_raises_for_new_only():
    lines: list[str] = []
    checker = _dual.Checker(_dual.STRICT, frozenset(), lines.append)
    only_observed = [i for i in items(np.zeros(4), np.zeros(4)) if i.name != "c"]
    assert checker.check("EP", only_observed, first=True).differ[_dual.OBSERVE] == 1
    with pytest.raises(_dual.DualCheckError, match="new route of 'c' differs from the old"):
        checker.check("EP", items(np.zeros(4), np.ones(4)), first=True)
    assert "NEW c vs old: differ" in lines[-1]  # logged before it raises


def test_checker_off():
    lines: list[str] = []
    checker = _dual.Checker(_dual.OFF, frozenset(), lines.append)
    summary = checker.check("EP", items(np.zeros(4), np.ones(4)), first=True)
    checker.log_summary("EP", summary)
    assert summary.total == 0 and lines == []


def test_checker_selftest_reports_exactly_the_perturbed_items():
    lines: list[str] = []
    new = np.zeros(4)
    checker = _dual.Checker(_dual.REPORT, frozenset({"c", "s"}), lines.append)
    summary = checker.check("EP", items(new, np.zeros(4)), first=True)
    assert np.array_equal(new, np.zeros(4))  # the new-route value itself is unchanged
    assert summary.selftest == 2 and summary.differ == {_dual.NEW: 2, _dual.OBSERVE: 1}
    assert [line.split(":")[0] for line in lines if "differ" in line] == [
        "dual EP NEW c vs old [selftest]",
        "dual EP OBSERVE o vs old",
        "dual EP NEW s vs old [selftest]",
    ]
    assert "real 3: 1 differ, first (0,)" in lines[0]


def test_checker_reports_a_failing_new_route():
    lines: list[str] = []

    def broken():
        raise KeyError("decomp_domain")

    checker = _dual.Checker(_dual.REPORT, frozenset(), lines.append)
    item = _dual.Item("o", _dual.OBSERVE, broken, np.zeros(2))
    summary = checker.check("EP", [item], first=True)
    assert summary.differ[_dual.OBSERVE] == 1
    assert lines == [
        "dual EP OBSERVE o vs old: differ (new route failed: KeyError('decomp_domain'))"
    ]


def test_checker_reference_label():
    lines: list[str] = []
    checker = _dual.Checker(_dual.REPORT, frozenset(), lines.append, reference=_dual.REF_PY2FGEN)
    checker.check("probe", [_dual.Item("o", _dual.OBSERVE, 1, 1)], first=True)
    assert lines == ["dual probe OBSERVE o vs py2fgen: identical (int 1)"]


@pytest.mark.parametrize(
    "environ, mode, selftest",
    [
        ({}, _dual.STRICT, frozenset()),
        ({_dual.MODE_ENV: "report", _dual.SELFTEST_ENV: "all"}, _dual.REPORT, {"x", "y"}),
        ({_dual.MODE_ENV: "OFF", _dual.SELFTEST_ENV: " y "}, _dual.OFF, {"y"}),
    ],
)
def test_switches(environ, mode, selftest):
    assert _dual.parse_mode(environ) == mode
    assert _dual.parse_selftest(environ, {"x", "y"}) == selftest


def test_bad_switches():
    with pytest.raises(ValueError, match="expected one of strict, report, off"):
        _dual.parse_mode({_dual.MODE_ENV: "loud"})
    with pytest.raises(ValueError, match="z is not a NEW or OBSERVE item"):
        _dual.parse_selftest({_dual.SELFTEST_ENV: "x,z"}, {"x"})
    with pytest.raises(ValueError, match="Unknown argument class"):
        _dual.Source("F", _dual.OLD)
    with pytest.raises(ValueError, match="Unknown provider"):
        _dual.Source("A", "LATER")


# ---- the source table of the real functions ---------------------------------------------------


def test_source_table_covers_the_106_arguments():
    assert plugin.check_sources(plugin.FUNCTIONS, plugin.SOURCES) == []
    rows = [
        (name, param, plugin.SOURCES[name][param])
        for name, fn in REAL.items()
        for param in fn.param_descriptors
    ]
    assert len(rows) == 106
    arrays = [
        r
        for r in rows
        if isinstance(REAL[r[0]].param_descriptors[r[1]], py2fgen.ArrayParamDescriptor)
    ]
    assert len(arrays) == 65
    counts = {k: sum(s.klass == k for *_, s in rows) for k in _dual.CLASSES}
    assert counts == {"A": 15, "B": 41, "C": 3, "D": 32, "E": 15}
    # classes of the arrays per function (A/B/C/E: 0/26/3/14, 6/7/0/0, 9/0/0/0)
    for name, expected in (
        ("grid_init", {"A": 0, "B": 26, "C": 3, "E": 14}),
        ("diffusion_init", {"A": 6, "B": 7, "C": 0, "E": 0}),
        ("diffusion_run", {"A": 9, "B": 0, "C": 0, "E": 0}),
    ):
        got = {k: sum(s.klass == k for n, _, s in arrays if n == name) for k in expected}
        assert got == expected, name
    observed = {(n, p) for n, p, s in rows if s.provider == _dual.OBSERVE}
    assert observed == {("grid_init", f"{x}_owner_mask") for x in "cev"}
    assert not any(s.provider == _dual.NEW for *_, s in rows)
    # scalars have no location; arrays without one are the index ranges, vct_a and the zd_* lists
    assert all(s.location is None for n, p, s in rows if (n, p, s) not in arrays)
    unlocated = {p for _, p, s in arrays if s.location is None}
    assert unlocated == {
        *(f"{e}_{b}" for e in ("cell", "vertex", "edge") for b in ("starts", "ends")),
        "vct_a", "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef",
    }  # fmt: skip


def test_source_table_errors():
    functions = {"grid_init": plugin.FUNCTIONS["grid_init"]}
    table = dict(plugin.SOURCES["grid_init"])
    del table["c2e"]
    table["nonsense"] = _dual.Source("B", _dual.OLD)
    table["vct_a"] = _dual.Source("B", _dual.NEW)
    table["f_e"] = _dual.Source("E", _dual.OBSERVE, "edge")
    errors = plugin.check_sources(functions, {"grid_init": table})
    assert errors == [
        "grid_init: no source for 'c2e'.",
        "grid_init: 'nonsense' is not an argument.",
        "grid_init: 'vct_a' is NEW, but this plugin passes only the old route to the granule"
        " (OLD) or observes a native one (OBSERVE).",
        "grid_init: 'f_e' is OBSERVE, but has no native route.",
    ]
    with pytest.raises(ValueError, match="bad source table"):
        plugin.Plugin(FakeComIn(), functions=functions, sources={"grid_init": table})


def test_register_logs_the_source_table(logs):
    plugin.Plugin(FakeComIn(), environ={}).register()
    assert (
        "dual source table: 106 arguments (65 arrays, 41 scalars; A 15, B 41, C 3, D 32, E 15);"
        " OLD 103, NEW 0, OBSERVE 3; mode strict; selftest none" in logs.messages
    )
    assert (
        "dual plan EP_ATM_TIMELOOP_BEFORE: 3 items (NEW 0, OBSERVE 3):"
        " c_owner_mask, e_owner_mask, v_owner_mask" in logs.messages
    )
    assert "dual plan EP_ATM_DIFFUSION_ENTER: 0 items (NEW 0, OBSERVE 0)" in logs.messages


def test_unknown_selftest_item_is_refused():
    with pytest.raises(ValueError, match="c2e is not a NEW or OBSERVE item"):
        plugin.Plugin(FakeComIn(), environ={_dual.SELFTEST_ENV: "c2e"})


# ---- the owner-mask observation, end to end with toy functions -------------------------------

NPROMA, NC, NE, NV = 8, 5, 7, 4
CALLS: list[dict] = []


@icon4py_export.export
def toy_grid_init(  # noqa: PLR0917 [too-many-positional-arguments]
    c2e: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    c_owner_mask: grid_wrapper.NumpyBoolArray1D,
    e_owner_mask: grid_wrapper.NumpyBoolArray1D,
    v_owner_mask: grid_wrapper.NumpyBoolArray1D,
    cell_areas: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    num_cells: gtx.int32,
) -> None:
    CALLS.append(dict(c_owner_mask=c_owner_mask.copy(), e_owner_mask=e_owner_mask.copy(), c2e=c2e))


@icon4py_export.export
def toy_diffusion_run(dtime: gtx.float64) -> None:
    pass


TOY_FUNCTIONS = {
    "grid_init": plugin.FunctionEntry(
        toy_grid_init, "EP_ATM_TIMELOOP_BEFORE", False, _marshal.PASS_KEY
    ),
    "diffusion_run": plugin.FunctionEntry(
        toy_diffusion_run, "EP_ATM_DIFFUSION_ENTER", True, _marshal.CALL_COUNT_KEY
    ),
}
TOY_SOURCES = {
    "grid_init": {
        **{p: plugin.SOURCES["grid_init"][p] for p in ("c2e", "cell_areas", "num_cells")},
        **{f"{x}_owner_mask": plugin.SOURCES["grid_init"][f"{x}_owner_mask"] for x in "cev"},
    },
    "diffusion_run": {"dtime": plugin.SOURCES["diffusion_run"]["dtime"]},
}

# ICON's decomposition on one PE: decomp_domain 0 = owned; edges and vertices on the boundary
# to a PE with a higher number are not owned although decomp_domain is 0; padding is -1
DECOMP = {
    "cells": np.array([0, 0, 0, 1, 2, -1, -1, -1], dtype=np.int32),
    "edges": np.array([0, 0, 0, 0, 0, 2, 2, -1], dtype=np.int32),
    "verts": np.array([0, 0, 2, 2, -1, -1, -1, -1], dtype=np.int32),
}
OWNER = {
    "c": np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32),
    "e": np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=np.int32),  # 2 boundary edges not owned
    "v": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),  # 1 boundary vertex not owned
}


class DomainComIn(FakeComIn):
    """FakeComIn with the descriptive data of domain 1 (host arrays, nblks == 1)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        edge_idx = np.asfortranarray(
            np.arange(1, NPROMA * 3 + 1, dtype=np.int32).reshape(NPROMA, 1, 3)
        )
        area = np.asfortranarray(np.linspace(1.0, 2.0, NPROMA).reshape(NPROMA, 1))
        self.domain = types.SimpleNamespace(
            cells=types.SimpleNamespace(
                decomp_domain=memoryview(DECOMP["cells"].reshape(NPROMA, 1)),
                ncells=NC,
                edge_idx=memoryview(edge_idx),
                area=memoryview(area),
            ),
            edges=types.SimpleNamespace(
                decomp_domain=memoryview(DECOMP["edges"].reshape(NPROMA, 1)), nedges=NE
            ),
            verts=types.SimpleNamespace(
                decomp_domain=memoryview(DECOMP["verts"].reshape(NPROMA, 1)), nverts=NV
            ),
        )

    def descrdata_get_domain(self, jg: int):
        assert jg == 1
        return self.domain


def toy_icon(comin: FakeComIn) -> FakeIcon:
    icon = FakeIcon(comin, TOY_FUNCTIONS)
    edge_idx = np.asarray(comin.domain.cells.edge_idx)[:, 0, :]
    arrays = {
        "c2e": fortran_buffer((NPROMA, 3), np.int32, fill=edge_idx),
        **{f"{x}_owner_mask": fortran_buffer((NPROMA,), np.int32, fill=OWNER[x]) for x in "cev"},
        "cell_areas": fortran_buffer((NPROMA,), np.float64, fill=1.5),
    }
    icon.expose("grid_init", arrays, dict(num_cells=NC))
    icon.expose("diffusion_run", {}, dict(dtime=10.0))
    return icon


def toy_plugin(comin: FakeComIn, **environ: str) -> plugin.Plugin:
    instance = plugin.Plugin(comin, functions=TOY_FUNCTIONS, environ=environ, sources=TOY_SOURCES)
    instance.register()
    return instance


def dual_lines(logs) -> list[str]:
    return [m for m in logs.messages if m.startswith("dual EP_")]


@pytest.fixture
def toy_calls():
    CALLS.clear()
    yield CALLS
    CALLS.clear()


def test_owner_masks_observed(toy_calls, wrappers, logs):
    comin = DomainComIn()
    icon = toy_icon(comin)
    toy_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    where = "dual EP_ATM_TIMELOOP_BEFORE pass 1"
    assert dual_lines(logs) == [
        f"{where} OBSERVE c_owner_mask vs old: identical (bool (8,); real 5: 0 differ;"
        " padding 3: 0 differ)",
        f"{where} OBSERVE e_owner_mask vs old: differ (bool (8,); real 7: 2 differ,"
        " first (2,); padding 1: 0 differ)",
        f"{where} OBSERVE v_owner_mask vs old: differ (bool (8,); real 4: 1 differ,"
        " first (1,); padding 4: 0 differ)",
        f"{where}: 3 checked (NEW 0, OBSERVE 3), 1 identical, 2 differ (NEW 0, OBSERVE 2);"
        " selftest 0",
    ]
    # observe only: strict mode does not stop, and the granule gets the old route
    assert len(toy_calls) == 1
    assert list(toy_calls[0]["e_owner_mask"]) == [bool(x) for x in OWNER["e"]]
    # pass 2: only the differing items, and the summary
    icon.new_pass()
    later = [m for m in dual_lines(logs) if "pass 2" in m]
    assert [m.split(":")[0] for m in later] == [
        "dual EP_ATM_TIMELOOP_BEFORE pass 2 OBSERVE e_owner_mask vs old",
        "dual EP_ATM_TIMELOOP_BEFORE pass 2 OBSERVE v_owner_mask vs old",
        "dual EP_ATM_TIMELOOP_BEFORE pass 2",
    ]
    # nothing is checked per call (no NEW or OBSERVE argument of diffusion_run)
    icon.diffusion_call(dtime=10.0, linit=True)
    assert not any("EP_ATM_DIFFUSION_ENTER" in m for m in dual_lines(logs))


@pytest.mark.parametrize("selftest, perturbed", [("all", "cev"), ("c_owner_mask", "c")])
def test_owner_masks_selftest(toy_calls, wrappers, logs, selftest, perturbed):
    comin = DomainComIn()
    icon = toy_icon(comin)
    toy_plugin(comin, **{_dual.MODE_ENV: "report", _dual.SELFTEST_ENV: selftest})
    icon.secondary_constructor()
    icon.new_pass()
    lines = dual_lines(logs)
    for x in "cev":
        (line,) = [m for m in lines if f" {x}_owner_mask " in m]
        assert ("[selftest]" in line) == (x in perturbed)
        assert "differ (" in line  # c only because of the perturbation
    assert lines[-1].endswith(
        f"3 checked (NEW 0, OBSERVE 3), 0 identical, 3 differ (NEW 0, OBSERVE 3);"
        f" selftest {len(perturbed)}"
    )
    (c_line,) = [m for m in lines if " c_owner_mask " in m]
    assert "real 5: 1 differ, first (0,)" in c_line
    # the granule's input is ICON's mask, not the perturbed copy
    assert list(toy_calls[0]["c_owner_mask"]) == [bool(x) for x in OWNER["c"]]
    assert any("must report" in r.message and r.levelname == "WARNING" for r in logs.records)


def test_dual_off(toy_calls, wrappers, logs):
    comin = DomainComIn()
    icon = toy_icon(comin)
    toy_plugin(comin, **{_dual.MODE_ENV: "off"})
    icon.secondary_constructor()
    icon.new_pass()
    assert dual_lines(logs) == [] and len(toy_calls) == 1


def test_probe_mode_keeps_the_native_routes(logs, monkeypatch):
    comin = DomainComIn()
    instance = toy_plugin(comin, **{plugin.PROBE_ENV: "1"})
    values = instance.probe_values
    assert list(values) == list(plugin.PROBE_ITEMS)
    assert values["c2e"].value.shape == (NPROMA, 3) and values["c2e"].real == NC
    assert np.array_equal(values["c2e"].value, np.asarray(comin.domain.cells.edge_idx)[:, 0, :])
    assert values["cell_areas"].value.dtype == np.float64 and values["cell_areas"].real == NC
    assert list(values["e_owner_mask"].value) == [x == 0 for x in DECOMP["edges"]]
    assert values["v_owner_mask"].real == NV
    assert any(m.startswith("ICON4PY_COMIN_PROBE=1: kept the native routes") for m in logs.messages)
    monkeypatch.setattr(plugin, "_instance", instance)
    assert plugin.instance() is instance


def test_probe_mode_switch():
    with pytest.raises(ValueError, match="expected 0 or 1"):
        plugin.Plugin(FakeComIn(), environ={plugin.PROBE_ENV: "yes"})
    assert plugin.Plugin(FakeComIn(), environ={plugin.PROBE_ENV: "0"}).probe_values == {}
