# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the icon4py ComIn plugin's comparison tables ('_verify.py'): a port of ICON's
'verify_field_wp' ('mo_nh_verify_utils.f90'). The expected lines are verbatim lines of ICON's
own tables, from the log of a diffusion VERIFY run of 'mch_icon-ch1_small' (two work PEs).
"""

import threading

import numpy as np
import pytest

from icon4py.bindings.comin import _verify


# ---- ICON's format -------------------------------------------------------------------------------

ICON_HEADER = [  # without the blank of ICON's 'message' (and the title's list-directed blank)
    "-------------------- domain --------------------",
    "               field  is_close   percent       rel MSE       abs MSE   max_rel_err  ( jc/e,"
    "  jk)   max_abs_err  ( jc/e,  jk)   max_abs_ref  max_abs_fiel   min_abs_ref  min_abs_fiel",
    "--------------------  --------  --------  ------------  ------------  ------------  "
    "------------  ------------  ------------  ------------  ------------  ------------  "
    "------------",
]
ICON_ROWS = {
    "vn": "                  vn  Yes          100.0    0.1328E-32    0.2234E-30    0.8971E-11  "
    "( 8500,  33)    0.7105E-14  ( 8500,  33)    0.4449E+02    0.4449E+02    0.0000E+00    "
    "0.0000E+00",
    "rho": "                 rho  Yes          100.0    0.0000E+00    0.0000E+00    0.0000E+00  "
    "( 5887,  80)    0.0000E+00  ( 5887,  80)    0.1160E+01    0.1160E+01    0.0000E+00    "
    "0.0000E+00",
    "padding": "                  vn  Yes          100.0           NaN    0.0000E+00    0.0000E+00"
    "  (    0,   0)    0.0000E+00  (    0,   0)    0.0000E+00    0.0000E+00    0.0000E+00    "
    "0.0000E+00",
}


def test_header_and_footer_as_icon_prints_them():
    assert _verify.header_lines("domain") == ICON_HEADER
    assert _verify.footer_line() == ICON_HEADER[2]


def _row(name: str, size: int, **values) -> _verify.Row:
    defaults = dict(
        close=size, abs_sq=0.0, field_sq=0.0, max_rel_err=0.0, location=(0, 0), max_abs_err=0.0
    )
    defaults |= dict(max_ref=0.0, max_field=0.0, min_ref=0.0, min_field=0.0)
    return _verify.Row(name=name, size=size, **(defaults | values))


def test_rows_as_icon_prints_them():
    size = 1000
    vn = _row(
        "vn",
        size,
        abs_sq=0.2234e-30 * size,
        field_sq=0.2234e-30 * size / 0.1328e-32,
        max_rel_err=0.8971e-11,
        location=(8500, 33),
        max_abs_err=0.7105e-14,
        max_ref=0.4449e02,
        max_field=0.4449e02,
    )
    assert _verify.format_row(vn) == ICON_ROWS["vn"]
    rho = _row("rho", size, field_sq=1.0, location=(5887, 80), max_ref=1.16, max_field=1.16)
    assert _verify.format_row(rho) == ICON_ROWS["rho"]
    # the padding: everything 0, so 'rel MSE' is 0 / 0
    assert _verify.format_row(_row("vn", 320)) == ICON_ROWS["padding"]
    no = _verify.format_row(_row("w", 10, close=9))
    assert no.split()[1:3] == ["No", "90.0"]


@pytest.mark.parametrize(
    "value, text",
    [
        (0.0, "  0.0000E+00"),
        (1.0, "  0.1000E+01"),
        (-2.5e-7, " -0.2500E-06"),
        (9.99995e-5, "  0.1000E-03"),  # the rounding carries into the exponent
        (1.5e-101, "  0.1500-100"),  # three exponent digits: no letter
        (2.0e120, "  0.2000+121"),
        (float("nan"), "         NaN"),
        (float("inf"), "    Infinity"),
        (-float("inf"), "   -Infinity"),
    ],
)
def test_fortran_e(value, text):
    assert _verify.fortran_e(value) == text


def test_fortran_f_a_i():
    assert _verify.fortran_f(100.0) == "   100.0"
    assert _verify.fortran_f(float("nan")) == "     NaN"
    assert _verify.fortran_f(float("inf")) == "Infinity"
    assert _verify.fortran_a("max_abs_field", 12) == "max_abs_fiel"
    assert _verify.fortran_a("jk", 3) == " jk"
    assert _verify.fortran_i(33, 4) == "  33"


# ---- the statistics -----------------------------------------------------------------------------


def _reference_row(name, ref, fld, start, end) -> dict:
    """The statistics of 'verify_field_wp' for one PE, written out loop by loop."""
    abs_err = np.abs(ref[start:end] - fld[start:end])
    nonzero = ref[start:end] != 0
    rel = np.zeros_like(abs_err)
    rel[nonzero] = abs_err[nonzero] / np.abs(ref[start:end][nonzero])
    close = abs_err <= _verify.ATOL + _verify.RTOL * np.abs(ref[start:end])
    return dict(
        close=int(close.sum()),
        size=abs_err.size,
        max_abs_err=float(abs_err.max(initial=0.0)),
        max_rel_err=float(rel.max(initial=0.0)),
        max_ref=float(np.abs(ref[start:end]).max(initial=0.0)),
        max_field=float(np.abs(fld[start:end]).max(initial=0.0)),
    )


def test_table_of_one_pe():
    rng = np.random.default_rng(1)
    nproma, levels, real = 12, 5, 9
    ref = np.asfortranarray(rng.normal(size=(nproma, levels)))
    ref[0, :] = 0.0  # no relative error where the reference is 0
    fld = ref * (1 + 1e-6 * rng.normal(size=ref.shape))
    fld[3, 2] += 1.0  # one entry far off
    fld[real:, :] = 0.0
    ref[real:, :] = 0.0
    pairs = [_verify.Pair("x", ref, fld, real)]
    (domain,) = _verify.table(pairs, "domain")
    want = _reference_row("x", ref, fld, 0, real)
    assert {k: getattr(domain, k) for k in want} == want
    assert not domain.is_close and domain.close == domain.size - 1
    assert domain.location == (4, 3)  # 1-based, as ICON: the largest relative error
    assert domain.abs_sq == pytest.approx(float(((ref - fld)[:real] ** 2).sum()), rel=1e-12)
    assert domain.field_sq == pytest.approx(float((fld[:real] ** 2).sum()), rel=1e-12)
    assert (domain.min_ref, domain.min_field) == (0.0, 0.0)  # ICON's minima start at 0
    (padding,) = _verify.table(pairs, "padding")
    assert (padding.size, padding.close, padding.location) == ((nproma - real) * levels, 15, (0, 0))
    assert np.isnan(padding.rel_mse) and padding.percent == 100.0
    assert _verify.format_row(padding).split()[3] == "NaN"


def test_location_of_ties_is_the_last_in_fortran_order():
    ref = np.asfortranarray(np.ones((4, 3)))
    (row,) = _verify.table([_verify.Pair("x", ref, ref.copy(), 4)], "domain")
    assert row.max_rel_err == 0.0 and row.location == (4, 3)
    ref[3, 2] = 0.0  # the last entry has no relative error
    (row,) = _verify.table([_verify.Pair("x", ref, ref.copy(), 4)], "domain")
    assert row.location == (3, 3)


def test_empty_range():
    ref = np.asfortranarray(np.ones((4, 3)))
    (row,) = _verify.table([_verify.Pair("x", ref, ref, 4)], "padding")
    assert row.size == 0 and np.isnan(row.percent) and row.is_close  # 0 of 0 close


class TwoPes:
    """An allreduce over two threads that stand for two work PEs."""

    def __init__(self) -> None:
        self._barrier = threading.Barrier(2)
        self._slots: list = [None, None]

    def allreduce(self, pe: int) -> _verify.Allreduce:
        ops = {"sum": np.add, "max": np.maximum, "min": np.minimum}

        def reduce(values: np.ndarray, op: str) -> np.ndarray:
            self._slots[pe] = values
            self._barrier.wait()
            result = ops[op](self._slots[0], self._slots[1])
            self._barrier.wait()
            return result

        return reduce


def test_table_over_two_pes():
    rng = np.random.default_rng(2)
    shape = (10, 4)
    refs = [np.asfortranarray(rng.normal(size=shape)) for _ in range(2)]
    flds = [r + 1e-9 * rng.normal(size=shape) for r in refs]
    flds[1][6, 1] += 1e-3  # the largest error, on PE 1
    reals = (8, 7)
    pes, results = TwoPes(), [None, None]

    def run(pe: int) -> None:
        pairs = [_verify.Pair("x", refs[pe], flds[pe], reals[pe])]
        results[pe] = [_verify.table(pairs, t, pes.allreduce(pe)) for t in _verify.TABLES]

    threads = [threading.Thread(target=run, args=(pe,)) for pe in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results[0] == results[1]  # every PE gets the global rows
    ((domain,), (padding,)) = results[0]
    one = [_reference_row("x", refs[pe], flds[pe], 0, reals[pe]) for pe in (0, 1)]
    assert domain.size == one[0]["size"] + one[1]["size"] and domain.close == domain.size - 1
    assert domain.max_abs_err == max(x["max_abs_err"] for x in one)
    assert domain.location == (7, 2)  # PE 1's entry; PE 0 reports (0, 0)
    assert padding.size == (shape[0] - 8 + shape[0] - 7) * shape[1]


def test_shapes_must_match():
    with pytest.raises(ValueError, match="expected the same 2-D shape"):
        _verify.table([_verify.Pair("x", np.zeros((3, 2)), np.zeros((3, 3)), 3)], "domain")
    with pytest.raises(ValueError, match="Unknown table"):
        _verify.table([], "scalars")


# ---- the plugin's buffers -----------------------------------------------------------------------


def test_copies_are_kept():
    copies = _verify.Copies()
    source = np.asfortranarray(np.arange(6.0).reshape(3, 2))
    buffer = copies.take("w", source)
    assert buffer is not source and np.array_equal(buffer, source) and buffer.flags.f_contiguous
    source += 1.0
    assert copies.take("w", source) is buffer and np.array_equal(buffer, source)
    assert copies.nbytes() == 48
    with pytest.raises(ValueError, match="'w': ICON's array has the shape"):
        copies.take("w", np.zeros((3, 3)))
    copies.release()
    assert copies.nbytes() == 0
