# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the dual-channel check ('_dual.py'), the plugin's source table, the arguments from
ComIn's descriptive data end to end and the probe mode,
against the fake ComIn and ICON of 'test_comin_marshal.py'.
"""

import dataclasses
import math
import re
import struct
import types

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import grid_wrapper, icon4py_export
from icon4py.bindings.comin import _config, _descrdata, _dual, _marshal, _views, plugin
from icon4py.model.common import dimension as dims
from icon4py.tools import py2fgen

from .test_comin_descrdata import mpi_or_skip
from .test_comin_marshal import (  # fixtures: calls, icon_run_dir, logs, restore_package_logger, wrappers
    REAL,
    FakeComIn,
    FakeIcon,
    calls,
    fortran_buffer,
    icon_run_dir,
    logs,
    restore_package_logger,
    set_mode,
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


def test_zero_entries_are_counted():
    """A difference against a zero (e.g. ICON's 0 where it does not compute a field) is named."""
    old = np.array([0.0, 2.0, 0.0, 4.0])
    result = _dual.compare_arrays(np.array([1.0, 2.0, 0.5, 0.0]), old)
    assert result.details.endswith(
        "real 4: 3 differ, first (0,), max abs 4, max rel inf, max ulp 4616189618054758400,"
        " reference zero at 2, new route zero at 1; padding 0: 0 differ"
    )


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


def test_communicators():
    new = _dual.Communicator(-2080374780, (0, 1), "CONGRUENT")
    old = _dual.Communicator(1140850688, (0, 1))
    result = _dual.compare(new, old)
    assert result.identical
    assert result.details == "MPI_Comm_compare CONGRUENT; handle 0x84000004 vs 0x44000000; 2 ranks"
    assert _dual.compare(_dual.Communicator(5, (0, 1), "IDENT"), old).identical
    for relation in ("SIMILAR", "UNEQUAL", None):
        assert not _dual.compare(_dual.Communicator(5, (0, 1), relation), old).identical
    other = _dual.compare(_dual.Communicator(5, (1, 0), "CONGRUENT"), old)
    assert not other.identical and other.details.endswith("; members (1, 0) vs (0, 1)")
    assert not _dual.compare(new, 1140850688).identical
    perturbed = _dual.perturb(new)
    assert perturbed.members == (1, 1) and perturbed.handle == new.handle
    assert not _dual.compare(perturbed, old).identical
    assert _dual.host_array(new) is new


def test_form():
    array = np.zeros((8, 3), dtype=np.int32, order="F")
    assert _dual.form(array) == "numpy int32 (8, 3) strides (1, 8)"
    assert _dual.form(np.ascontiguousarray(array)) == "numpy int32 (8, 3) strides (3, 1)"
    assert _dual.form(np.zeros((8, 1), order="C")) == "numpy float64 (8, 1) strides (1, 0)"
    field = gtx.as_field([dims.CellDim, dims.C2EDim], array)
    assert _dual.form(field).startswith("Field[Cell, C2E] over numpy int32 (8, 3) strides")
    assert _dual.form(None) == "None"


def test_checker_takes_a_field_as_a_value():
    """A Field is callable, but a value; only functions are new routes to call."""
    lines: list[str] = []
    field = gtx.as_field([dims.CellDim], np.arange(4.0))
    items = [_dual.Item("f", _dual.NEW, field, np.arange(4.0), form=_dual.form(field))]
    assert _dual.Checker(_dual.STRICT, frozenset(), lines.append).check("EP", items, True).total
    assert lines == [
        "dual EP NEW f vs old: identical (float64 (4,); real 4: 0 differ; padding 0: 0 differ)"
    ]


def test_checker_compares_the_form_before_a_perturbation():
    lines: list[str] = []
    checker = _dual.Checker(_dual.REPORT, frozenset({"b"}), lines.append)
    f_order = np.arange(6.0).reshape(3, 2, order="F")
    c_order = np.ascontiguousarray(f_order)
    items = [
        _dual.Item("a", _dual.NEW, c_order, f_order, form=_dual.form(f_order)),
        _dual.Item("b", _dual.NEW, f_order.copy(order="F"), f_order, form=_dual.form(f_order)),
        _dual.Item("c", _dual.NEW, c_order, f_order),  # no form required: the values only
    ]
    summary = checker.check("EP", items, first=True)
    assert summary.differ[_dual.NEW] == 2
    assert lines[0] == (
        "dual EP NEW a vs old: differ (form numpy float64 (3, 2) strides (2, 1) vs"
        " numpy float64 (3, 2) strides (1, 3))"
    )
    assert lines[1].startswith("dual EP NEW b vs old [selftest]: differ (float64 (3, 2); real 6:")
    assert lines[2].startswith("dual EP NEW c vs old: identical")


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
        " max abs 1, max rel 1, max ulp 4607182418800017408, new route zero at 3; padding 1: 1 differ)",
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

DESCRIPTIVE_NEW = (
    *(f"{e}_{b}" for e in ("cell", "vertex", "edge") for b in ("starts", "ends")),
    *("c2e", "e2c", "c2e2c", "e2c2e", "e2v", "v2e", "v2c", "e2c2v", "c2v"),
    *(f"{x}_glb_index" for x in "cev"),
    *(f"{x}_owner_mask" for x in "cev"),
    *("comm_id", "num_vertices", "num_cells", "num_edges", "vertical_size", "limited_area"),
    *("tangent_orientation", "inverse_primal_edge_lengths", "inv_dual_edge_length"),
    *("inv_vert_vert_length", "edge_areas", "f_e", "cell_center_lat", "cell_center_lon"),
    "cell_areas",
    *(f"{n}_{x}" for n in ("primal_normal_vert", "dual_normal_vert") for x in "xy"),
    *(f"{n}_{x}" for n in ("primal_normal_cell", "dual_normal_cell") for x in "xy"),
    *("edge_center_lat", "edge_center_lon", "primal_normal_x", "primal_normal_y", "vct_a"),
    "mean_cell_area",
)
"""The arguments that the plugin takes from ComIn's descriptive data (all of 'grid_init')."""
DESCRIPTIVE_NEW_INIT = (
    *("e_bln_c_s", "geofac_div", "geofac_grg_x", "geofac_grg_y", "geofac_n2s", "nudgecoeff_e"),
    "rbf_vec_coeff_v",
)
"""The arguments of 'diffusion_init' from ComIn's descriptive data (the interpolation coefficients)."""
STATIC = ("theta_ref_mc", "wgtfac_c", "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef")
"""The arguments of 'diffusion_init' that are ICON's variables (plugin.STATIC_VARIABLES)."""


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
    assert counts == {"A": 15, "B": 41, "C": 5, "D": 32, "E": 13}
    # classes of the arrays per function (A/B/C/E: 0/26/5/12, 6/7/0/0, 9/0/0/0)
    for name, expected in (
        ("grid_init", {"A": 0, "B": 26, "C": 5, "E": 12}),
        ("diffusion_init", {"A": 6, "B": 7, "C": 0, "E": 0}),
        ("diffusion_run", {"A": 9, "B": 0, "C": 0, "E": 0}),
    ):
        got = {k: sum(s.klass == k for n, _, s in arrays if n == name) for k in expected}
        assert got == expected, name
    # nothing is only observed
    assert not [(n, p) for n, p, s in rows if s.provider == _dual.OBSERVE]
    # NEW: the configuration scalars from ICON's namelist output, the arguments from ComIn's
    # descriptive data, ICON's variables that diffusion_init gets, and all 11 arguments of
    # diffusion_run (per call, at DIFFUSION_BEFORE); nothing is OLD
    assert not [(n, p) for n, p, s in rows if s.provider == _dual.OLD]
    new = {(n, p) for n, p, s in rows if s.provider == _dual.NEW}
    configuration = {(n, p) for n, p, s in rows if s.klass == "D"}
    assert configuration == {(n, p) for n, table in _config.ARGUMENTS.items() for p in table}
    descriptive = {(n, p) for n, table in _descrdata.ROUTES.items() for p in table}
    assert descriptive == {("grid_init", p) for p in DESCRIPTIVE_NEW} | {
        ("diffusion_init", p) for p in DESCRIPTIVE_NEW_INIT
    }
    per_call = {("diffusion_run", p) for p in REAL["diffusion_run"].param_descriptors}
    static = {(n, p) for n, table in plugin.STATIC_VARIABLES.items() for p in table}
    assert static == {("diffusion_init", p) for p in STATIC}
    assert {p: s.klass for n, p, s in rows if (n, p) in static} == dict.fromkeys(STATIC, "A")
    assert new == configuration | descriptive | per_call | static
    assert len(new) == 32 + len(DESCRIPTIVE_NEW) + len(DESCRIPTIVE_NEW_INIT) + 11 + 6
    # scalars have no location; arrays without one are the index ranges, vct_a and the zd_* lists
    assert all(s.location is None for n, p, s in rows if (n, p, s) not in arrays)
    unlocated = {p for _, p, s in arrays if s.location is None}
    assert unlocated == {
        *(f"{e}_{b}" for e in ("cell", "vertex", "edge") for b in ("starts", "ends")),
        "vct_a", "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef",
    }  # fmt: skip


NEW_ERROR = (
    "is NEW, but has no native route that this plugin passes to the granule (configuration"
    " scalars from ICON's namelist output, ComIn's descriptive data, ICON's variables of"
    " STATIC_VARIABLES, and the per-call arguments of a writing function)."
)


def without_static(monkeypatch, *names: str) -> None:
    """Drop arguments of 'diffusion_init' from plugin.STATIC_VARIABLES (no native route left)."""
    table = {p: v for p, v in plugin.STATIC_VARIABLES["diffusion_init"].items() if p not in names}
    monkeypatch.setattr(plugin, "STATIC_VARIABLES", {"diffusion_init": table})


def test_source_table_errors(monkeypatch):
    without_static(monkeypatch, "theta_ref_mc", "wgtfac_c")
    functions = {name: plugin.FUNCTIONS[name] for name in ("grid_init", "diffusion_init")}
    grid = dict(plugin.SOURCES["grid_init"])
    del grid["c2e"]
    grid["nonsense"] = _dual.Source("B", _dual.OLD)
    grid["e_owner_mask"] = _dual.Source("C", _dual.OBSERVE, "edge")  # has a route: accepted
    init = dict(plugin.SOURCES["diffusion_init"])
    init["theta_ref_mc"] = _dual.Source("A", _dual.NEW, "cell")  # no route (dropped above)
    init["wgtfac_c"] = _dual.Source("A", _dual.OBSERVE, "cell")
    init["zd_cellidx"] = _dual.Source("A", _dual.OBSERVE)  # ICON's variable: accepted
    init["hdiff_w"] = _dual.Source("D", _dual.OBSERVE)  # configuration: NEW or OLD
    errors = plugin.check_sources(functions, {"grid_init": grid, "diffusion_init": init})
    assert errors == [
        "grid_init: no source for 'c2e'.",
        "grid_init: 'nonsense' is not an argument.",
        f"diffusion_init: 'theta_ref_mc' {NEW_ERROR}",
        "diffusion_init: 'wgtfac_c' is OBSERVE, but has no native route.",
        "diffusion_init: 'hdiff_w' is OBSERVE, but has no native route.",
    ]
    with pytest.raises(ValueError, match="bad source table"):
        plugin.Plugin(FakeComIn(), functions=functions, sources={"grid_init": grid})


def test_per_call_source_errors(monkeypatch):
    functions = {"diffusion_run": plugin.FUNCTIONS["diffusion_run"]}
    table = dict(plugin.SOURCES["diffusion_run"])
    table["dtime"] = _dual.Source("B", _dual.OLD)  # NEW arrays need all per-call arguments NEW
    table["hdef_ic"] = _dual.Source("A", _dual.OBSERVE, "cell")  # arrays are not observed
    errors = plugin.check_sources(functions, {"diffusion_run": table})
    assert errors == [
        "diffusion_run: 'hdef_ic' is OBSERVE, but has no native route.",
        "diffusion_run: its per-call arguments are NEW only all together (the plugin then calls"
        " it at EP_ATM_DYCORE_DIFFUSION_BEFORE), but hdef_ic, dtime are not.",
    ]
    # all per-call arguments from the old route: valid (the plugin calls it at ENTER)
    old = {p: _dual.Source(s.klass, _dual.OLD, s.location) for p, s in table.items()}
    assert plugin.check_sources(functions, {"diffusion_run": old}) == []
    # a function that does not write has no per-call arguments
    without_static(monkeypatch, "wgtfac_c")
    functions = {"diffusion_init": plugin.FUNCTIONS["diffusion_init"]}
    init = dict(plugin.SOURCES["diffusion_init"], wgtfac_c=_dual.Source("A", _dual.NEW, "cell"))
    assert plugin.check_sources(functions, {"diffusion_init": init}) == [
        f"diffusion_init: 'wgtfac_c' {NEW_ERROR}"
    ]
    # ICON's variables are a route of a function that does not write only
    writing = {"diffusion_init": dataclasses.replace(functions["diffusion_init"], inout=True)}
    assert not plugin.static_route("diffusion_init", writing["diffusion_init"], "theta_ref_mc")
    assert plugin.static_route("diffusion_init", functions["diffusion_init"], "theta_ref_mc")


def test_sources_for_mode():
    for mode in (_config.SUBSTITUTE, _config.OFF):
        assert plugin.sources_for_mode(plugin.SOURCES, plugin.FUNCTIONS, mode) == plugin.SOURCES
    verify = plugin.sources_for_mode(plugin.SOURCES, plugin.FUNCTIONS, _config.VERIFY)
    assert {k: v for k, v in verify.items() if k != "diffusion_run"} == {
        k: v for k, v in plugin.SOURCES.items() if k != "diffusion_run"
    }
    providers = {p: s.provider for p, s in verify["diffusion_run"].items()}
    assert providers == {
        **dict.fromkeys(("w", "vn", "exner", "theta_v", "rho"), _dual.OLD),
        **dict.fromkeys(("hdef_ic", "div_ic", "dwdx", "dwdy"), _dual.OLD),
        "dtime": _dual.OBSERVE,
        "linit": _dual.OBSERVE,
    }
    assert plugin.check_sources(plugin.FUNCTIONS, verify) == []
    assert plugin.computes_before(
        plugin.FUNCTIONS["diffusion_run"], plugin.SOURCES["diffusion_run"]
    )
    assert not plugin.computes_before(plugin.FUNCTIONS["diffusion_run"], verify["diffusion_run"])
    assert not plugin.computes_before(plugin.FUNCTIONS["grid_init"], plugin.SOURCES["grid_init"])


def test_register_logs_the_source_table(logs):
    plugin.Plugin(FakeComIn(), environ={}).register()
    assert (
        "dual source table: 106 arguments (65 arrays, 41 scalars; A 15, B 41, C 5, D 32, E 13);"
        " OLD 0, NEW 106, OBSERVE 0; mode strict; selftest none" in logs.messages
    )
    assert (
        "SUBSTITUTE: every argument of grid_init, diffusion_init, diffusion_run from a native"
        " route (ICON's namelist output, ComIn's descriptive data, ICON's variables, the entry"
        " points); none from ICON's argument variables" in logs.messages
    )
    (plan,) = [m for m in logs.messages if m.startswith("dual plan EP_ATM_TIMELOOP_BEFORE")]
    assert plan.startswith(
        "dual plan EP_ATM_TIMELOOP_BEFORE: 95 items (NEW 95, OBSERVE 0):"
        " cell_starts, cell_ends, vertex_starts, vertex_ends, edge_starts, edge_ends, c2e, e2c,"
        " c2e2c, e2c2e, e2v, v2e, v2c, e2c2v, c2v, c_owner_mask, e_owner_mask, v_owner_mask,"
        " c_glb_index, e_glb_index, v_glb_index, tangent_orientation,"
        " inverse_primal_edge_lengths, inv_dual_edge_length, inv_vert_vert_length, edge_areas,"
        " f_e, cell_center_lat, cell_center_lon, cell_areas, primal_normal_vert_x,"
        " primal_normal_vert_y, dual_normal_vert_x, dual_normal_vert_y, primal_normal_cell_x,"
        " primal_normal_cell_y, dual_normal_cell_x, dual_normal_cell_y, edge_center_lat,"
        " edge_center_lon, primal_normal_x, primal_normal_y, vct_a, lowest_layer_thickness,"
        " model_top_height, stretch_factor, flat_height, rayleigh_damping_height,"
        " mean_cell_area, comm_id, num_vertices, num_cells, num_edges, vertical_size,"
        " limited_area, backend, theta_ref_mc, wgtfac_c, e_bln_c_s, geofac_div, geofac_grg_x,"
        " geofac_grg_y, geofac_n2s, nudgecoeff_e, rbf_vec_coeff_v, zd_cellidx, zd_vertidx,"
        " zd_intcoef, zd_diffcoef, ndyn_substeps, diffusion_type,"
    )
    assert plan.endswith(" itype_sher, iforcing, a_hshr, loutshs, backend")
    assert (
        "dual plan EP_ATM_DIFFUSION_ENTER: 11 items (NEW 11, OBSERVE 0): w, vn, exner, theta_v,"
        " rho, hdef_ic, div_ic, dwdx, dwdy, dtime, linit" in logs.messages
    )


def test_register_logs_the_source_table_in_verify(logs, icon_run_dir):
    set_mode(icon_run_dir, 2)
    plugin.Plugin(FakeComIn(), environ={}).register()
    assert (
        "dual source table: 106 arguments (65 arrays, 41 scalars; A 15, B 41, C 5, D 32, E 13);"
        " OLD 9, NEW 95, OBSERVE 2; mode strict; selftest none" in logs.messages
    )
    assert not any(" from a native route " in m for m in logs.messages)  # VERIFY: ICON's copies
    assert (
        "dual plan EP_ATM_DIFFUSION_ENTER: 2 items (NEW 0, OBSERVE 2): dtime, linit"
        in logs.messages
    )


def test_substitute_takes_no_argument_from_the_old_route(monkeypatch, logs, icon_run_dir):
    """The plugin's own source table: an OLD item stops the registration in SUBSTITUTE only."""
    old = _dual.Source("C", _dual.OLD, "edge")
    monkeypatch.setitem(plugin.SOURCES["grid_init"], "e_owner_mask", old)
    with pytest.raises(
        RuntimeError, match=r"come from ICON's argument variables: grid_init.e_owner"
    ):
        plugin.Plugin(FakeComIn(), environ={}).register()
    set_mode(icon_run_dir, 2)  # VERIFY: the per-call arrays are OLD anyway
    plugin.Plugin(FakeComIn(), environ={}).register()
    set_mode(icon_run_dir, 1)
    # a table given by the caller (tests of the old route) is not checked
    sources = {n: dict(t) for n, t in plugin.SOURCES.items()}
    plugin.Plugin(FakeComIn(), environ={}, sources=sources).register()
    assert plugin.old_arguments(sources) == ["grid_init.e_owner_mask"]


def test_unknown_selftest_item_is_refused():
    with pytest.raises(ValueError, match="theta_ref is not a NEW or OBSERVE item"):
        plugin.Plugin(FakeComIn(), environ={_dual.SELFTEST_ENV: "theta_ref"})


# ---- the arguments from the descriptive data, end to end with toy functions -----------------

NPROMA, NC, NE, NV = 8, 5, 7, 4
CALLS: list[dict] = []


@icon4py_export.export
def toy_grid_init(  # noqa: PLR0917 [too-many-positional-arguments]
    c2e: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    c_owner_mask: grid_wrapper.NumpyBoolArray1D,
    e_owner_mask: grid_wrapper.NumpyBoolArray1D,
    v_owner_mask: grid_wrapper.NumpyBoolArray1D,
    cell_areas: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    comm_id: gtx.int32,
    num_cells: gtx.int32,
) -> None:
    CALLS.append(
        dict(
            c_owner_mask=c_owner_mask.copy(),
            e_owner_mask=e_owner_mask.copy(),
            c2e=c2e,
            cell_areas=cell_areas,
            comm_id=comm_id,
            num_cells=num_cells,
        )
    )


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
    "grid_init": {p: plugin.SOURCES["grid_init"][p] for p in toy_grid_init.param_descriptors},
    "diffusion_run": {"dtime": _dual.Source("B", _dual.OLD)},
}

# ICON's decomposition on one PE: decomp_domain is the halo level (0 = owned), -1 on the
# padding; ICON's owner masks, which ICON passes as the arguments, are decomp_domain == 0
DECOMP = {
    "cells": np.array([0, 0, 0, 1, 2, -1, -1, -1], dtype=np.int32),
    "edges": np.array([0, 0, 0, 0, 0, 2, 2, -1], dtype=np.int32),
    "verts": np.array([0, 0, 2, 2, -1, -1, -1, -1], dtype=np.int32),
}
OWNER = {
    x: (DECOMP[kind] == 0).astype(np.int32)
    for x, kind in (("c", "cells"), ("e", "edges"), ("v", "verts"))
}


class DomainComIn(FakeComIn):
    """FakeComIn with the descriptive data of domain 1 (host arrays, nblks == 1) and an MPI
    host communicator (a duplicate of MPI_COMM_WORLD, so CONGRUENT to it)."""

    def __init__(self, nblks: int = 1, **kwargs):
        super().__init__(**kwargs)
        edge_idx = np.asfortranarray(
            np.arange(1, NPROMA * 3 + 1, dtype=np.int32).reshape(NPROMA, 1, 3)
        )
        area = np.asfortranarray(np.linspace(1.0, 2.0, NPROMA).reshape(NPROMA, 1))
        ns = types.SimpleNamespace
        self.domain = ns(
            id=1,
            nlev=3,
            cells=ns(
                decomp_domain=memoryview(DECOMP["cells"].reshape(NPROMA, 1)),
                ncells=NC,
                nblks=nblks,
                edge_idx=memoryview(edge_idx),
                area=memoryview(area),
            ),
            edges=ns(
                decomp_domain=memoryview(DECOMP["edges"].reshape(NPROMA, 1)), nedges=NE, nblks=1
            ),
            verts=ns(
                decomp_domain=memoryview(DECOMP["verts"].reshape(NPROMA, 1)), nverts=NV, nblks=1
            ),
        )
        self.host_comm: int | None = None

    def descrdata_get_global(self):
        return types.SimpleNamespace(
            has_device=self.has_device, lrestartrun=self.lrestartrun, n_dom=1, l_limited_area=True
        )

    def descrdata_get_domain(self, jg: int):
        assert jg == 1
        return self.domain

    def parallel_get_host_mpi_comm(self) -> int:
        if self.host_comm is None:
            self.host_comm = mpi_or_skip().COMM_WORLD.Dup().py2f()
        return self.host_comm


def toy_icon(comin: FakeComIn, **arrays: np.ndarray) -> FakeIcon:
    """ICON's side: the argument variables hold what py2fgen gets ('arrays' overrides)."""
    icon = FakeIcon(comin, TOY_FUNCTIONS)
    edge_idx = np.asarray(comin.domain.cells.edge_idx)[:, 0, :]
    buffers = {
        "c2e": fortran_buffer((NPROMA, 3), np.int32, fill=edge_idx),
        **{f"{x}_owner_mask": fortran_buffer((NPROMA,), np.int32, fill=OWNER[x]) for x in "cev"},
        "cell_areas": fortran_buffer(
            (NPROMA,), np.float64, fill=np.asarray(comin.domain.cells.area)[:, 0]
        ),
    }
    buffers.update({k: fortran_buffer(v.shape, v.dtype, fill=v) for k, v in arrays.items()})
    icon.expose("grid_init", buffers, dict(comm_id=mpi_or_skip().COMM_WORLD.py2f(), num_cells=NC))
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


def test_descriptive_data_end_to_end(toy_calls, wrappers, logs):
    comin = DomainComIn()
    icon = toy_icon(comin)
    toy_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    where = "dual EP_ATM_TIMELOOP_BEFORE pass 1"
    lines = dual_lines(logs)
    assert lines[0] == (
        f"{where} NEW c2e vs old: identical (int32 (8, 3); real 15: 0 differ; padding 9: 0 differ)"
    )
    assert lines[1:4] == [
        f"{where} NEW c_owner_mask vs old: identical (bool (8,); real 5: 0 differ;"
        " padding 3: 0 differ)",
        f"{where} NEW e_owner_mask vs old: identical (bool (8,); real 7: 0 differ;"
        " padding 1: 0 differ)",
        f"{where} NEW v_owner_mask vs old: identical (bool (8,); real 4: 0 differ;"
        " padding 4: 0 differ)",
    ]
    comm = [m for m in lines if " comm_id " in m]
    assert len(comm) == 1 and re.fullmatch(
        rf"{where} NEW comm_id vs old: identical \(MPI_Comm_compare CONGRUENT; handle 0x[0-9a-f]{{8}}"
        r" vs 0x[0-9a-f]{8}; 1 ranks\)",
        comm[0],
    )
    assert f"{where} NEW num_cells vs old: identical (int 5)" in lines
    assert (
        f"{where} NEW cell_areas vs old: identical (float64 (8,); real 5: 0 differ;"
        " padding 3: 0 differ)" in lines
    )
    n_new = sum(
        s.provider == _dual.NEW and _descrdata.has_route("grid_init", p)
        for p, s in TOY_SOURCES["grid_init"].items()
    )
    assert n_new == len(TOY_SOURCES["grid_init"])  # every argument of the toy grid_init
    assert lines[-1] == (
        f"{where}: {n_new} checked (NEW {n_new}, OBSERVE 0), {n_new} identical, 0 differ"
        " (NEW 0, OBSERVE 0); selftest 0"
    )
    assert any(m.startswith("descriptive data: n_dom 1, domain 1, nblks 1") for m in logs.messages)
    # the granule gets the plugin's copies (NEW), never ICON's descriptive data or the arguments
    (call,) = toy_calls
    c2e = call["c2e"]
    assert np.array_equal(c2e.ndarray, np.asarray(comin.domain.cells.edge_idx)[:, 0, :])
    assert c2e.ndarray.flags.f_contiguous and not np.shares_memory(
        c2e.ndarray, np.asarray(comin.domain.cells.edge_idx)
    )
    assert _views.data_ptr(c2e.ndarray) != _views.data_ptr(icon.live["c2e"])
    for x, kind in (("c", "cells"), ("e", "edges")):
        assert list(call[f"{x}_owner_mask"]) == [d == 0 for d in DECOMP[kind]]
    assert call["comm_id"] == comin.host_comm and call["num_cells"] == NC
    # pass 2: a fresh copy, only the differing items and the summary
    icon.new_pass()
    assert not np.shares_memory(toy_calls[1]["c2e"].ndarray, c2e.ndarray)
    later = [m for m in dual_lines(logs) if "pass 2" in m]
    assert [m.split(":")[0] for m in later] == ["dual EP_ATM_TIMELOOP_BEFORE pass 2"]
    # nothing is checked per call (no NEW or OBSERVE argument of diffusion_run)
    icon.diffusion_call(dtime=10.0, linit=True)
    assert not any("EP_ATM_DIFFUSION_ENTER" in m for m in dual_lines(logs))


def test_descriptive_data_differs(toy_calls, wrappers, logs):
    """ICON's argument differs from the descriptive data: strict stops, report goes on with it."""
    other = np.asarray(DomainComIn().domain.cells.edge_idx)[:, 0, :].copy()
    other[6, 1] = 99  # padding: reported, not a difference
    comin = DomainComIn()
    icon = toy_icon(comin, c2e=other)
    toy_plugin(comin, **{_dual.MODE_ENV: "report"})
    icon.secondary_constructor()
    icon.new_pass()
    (line,) = [m for m in dual_lines(logs) if " c2e " in m]
    assert line.endswith("identical (int32 (8, 3); real 15: 0 differ; padding 9: 1 differ)")
    other[2, 0] = 98  # a real entry
    comin = DomainComIn()
    icon = toy_icon(comin, c2e=other)
    toy_plugin(comin, **{_dual.MODE_ENV: "report"})
    icon.secondary_constructor()
    icon.new_pass()
    (line,) = [m for m in dual_lines(logs) if " c2e " in m and "real 15: 1 differ" in m]
    assert "first (2, 0), max abs" in line
    assert toy_calls[-1]["c2e"].ndarray[2, 0] != 98  # the granule got the descriptive data
    comin = DomainComIn()
    icon = toy_icon(comin, c2e=other)
    toy_plugin(comin)
    icon.secondary_constructor()
    with pytest.raises(_dual.DualCheckError, match="new route of 'c2e' differs"):
        icon.new_pass()


@pytest.mark.parametrize("selftest, perturbed", [("all", None), ("c_owner_mask", {"c_owner_mask"})])
def test_descriptive_data_selftest(toy_calls, wrappers, logs, selftest, perturbed):
    comin = DomainComIn()
    icon = toy_icon(comin)
    toy_plugin(comin, **{_dual.MODE_ENV: "report", _dual.SELFTEST_ENV: selftest})
    icon.secondary_constructor()
    icon.new_pass()
    lines = dual_lines(logs)
    checked = [p for p, s in TOY_SOURCES["grid_init"].items() if s.provider in _dual.CHECKED]
    expected = set(checked) if perturbed is None else perturbed
    for name in checked:
        (line,) = [m for m in lines if f" {name} vs " in m]
        assert ("[selftest]" in line) == (name in expected)
        assert ("differ (" in line) == (name in expected)
    assert lines[-1].endswith(f"selftest {len(expected)}")
    (c_line,) = [m for m in lines if " c_owner_mask " in m]
    assert "real 5: 1 differ, first (0,)" in c_line
    # the granule's input is the plugin's copy, unperturbed
    assert list(toy_calls[0]["c_owner_mask"]) == [bool(x) for x in OWNER["c"]]
    assert any("must report" in r.message and r.levelname == "WARNING" for r in logs.records)


def test_owner_masks_differ(toy_calls, wrappers, logs):
    """
    A host whose owner mask of the edges is not decomp_domain == 0: the dual check reports it
    (report) or stops (strict), and the granule gets decomp_domain == 0.
    """
    owner = OWNER["e"].copy()
    owner[3] = 0  # an edge on halo level 0 that the host does not own
    comin = DomainComIn()
    icon = toy_icon(comin, e_owner_mask=owner)
    toy_plugin(comin, **{_dual.MODE_ENV: "report"})
    icon.secondary_constructor()
    icon.new_pass()
    (line,) = [m for m in dual_lines(logs) if " e_owner_mask " in m]
    assert line.endswith("differ (bool (8,); real 7: 1 differ, first (3,); padding 1: 0 differ)")
    assert list(toy_calls[0]["e_owner_mask"]) == [x == 0 for x in DECOMP["edges"]]
    comin = DomainComIn()
    icon = toy_icon(comin, e_owner_mask=owner)
    toy_plugin(comin)
    icon.secondary_constructor()
    with pytest.raises(_dual.DualCheckError, match="new route of 'e_owner_mask' differs"):
        icon.new_pass()


def test_descriptive_data_needs_one_block(toy_calls, wrappers, logs):
    comin = DomainComIn(nblks=2)
    icon = toy_icon(comin)
    toy_plugin(comin)
    icon.secondary_constructor()
    with pytest.raises(RuntimeError, match=r"cells.nblks is 2, expected 1"):
        icon.new_pass()
    assert toy_calls == []


def test_dual_off(toy_calls, wrappers, logs):
    comin = DomainComIn()
    icon = toy_icon(comin)
    toy_plugin(comin, **{_dual.MODE_ENV: "off"})
    icon.secondary_constructor()
    icon.new_pass()
    assert dual_lines(logs) == [] and len(toy_calls) == 1
    # the NEW arguments still come from the descriptive data
    assert toy_calls[0]["comm_id"] == comin.host_comm


def test_probe_mode_keeps_the_native_routes(logs, monkeypatch):
    comin = DomainComIn()
    instance = toy_plugin(comin, **{plugin.PROBE_ENV: "1"})
    values = instance.probe_values
    assert list(values) == list(plugin.PROBE_ITEMS)
    assert values["c2e"].value.shape == (NPROMA, 3) and values["c2e"].real == NC
    assert np.array_equal(values["c2e"].value, np.asarray(comin.domain.cells.edge_idx)[:, 0, :])
    assert values["cell_areas"].value.dtype == np.float64 and values["cell_areas"].real == NC
    assert list(values["e_owner_mask"].value) == [x == 0 for x in DECOMP["edges"]]
    assert list(values["c_owner_mask"].value) == [x == 0 for x in DECOMP["cells"]]
    assert values["v_owner_mask"].real == NV
    assert any(m.startswith("ICON4PY_COMIN_PROBE=1: kept the native routes") for m in logs.messages)
    monkeypatch.setattr(plugin, "_instance", instance)
    assert plugin.instance() is instance


def test_probe_mode_switch():
    with pytest.raises(ValueError, match="expected 0 or 1"):
        plugin.Plugin(FakeComIn(), environ={plugin.PROBE_ENV: "yes"})
    assert plugin.Plugin(FakeComIn(), environ={plugin.PROBE_ENV: "0"}).probe_values == {}
