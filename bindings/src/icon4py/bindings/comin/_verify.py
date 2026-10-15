# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The plugin's verification of icon4py against ICON: ICON's comparison tables, ported.

In ICON's VERIFY mode ICON computes the horizontal diffusion itself and continues with its own
result; icon4py computes it too, on copies of ICON's input, and the two results are compared.
This module holds the plugin's side of it:
- 'Copies': the plugin's own buffers, kept for the whole run (on the device on the GPU, in
  py2fgen's layout), into which the plugin copies ICON's input before ICON computes, and on
  which icon4py computes;
- 'table': a port of ICON's 'verify_field_wp' ('mo_nh_verify_utils.f90') to NumPy and CuPy.
  Per field it compares the plugin's result ('field') with ICON's ('reference') over a range of
  the first axis and all of the second, with the same tolerance ('ATOL', 'RTOL', NumPy's
  'allclose' rule), the same statistics, the same reductions over ICON's work PEs (sums,
  maxima, minima) and the same columns; 'format_row' and 'header_lines' print them in ICON's
  format. ICON prints two tables per call: 'domain' (the real entries, from the first one to
  the end of the outermost halo row) and 'padding' (the rest of the first block, up to nproma).

What ICON's routine does and the port keeps: the location printed in both location columns is
that of the largest relative error; it is reduced over the PEs component by component with a
maximum, and where several entries share the largest error ICON's GPU loop keeps one of them
(the port takes the last one in Fortran order); the minima start at 0, so they print 0; a
relative error is taken only where the reference is not 0. What may differ: the two
mean-square columns, whose sums depend on the order of the reduction.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Final

import numpy as np


ATOL: Final = 1.0e-8
RTOL: Final = 1.0e-5
"""ICON's defaults in 'verify_field_wp': close iff |reference - field| <= ATOL + RTOL * |reference|."""
TABLES: Final = ("domain", "padding")

Allreduce = Callable[[np.ndarray, str], np.ndarray]
"""A reduction over ICON's work PEs: (local values, 'sum' | 'max' | 'min') -> global values."""


def local_allreduce(values: np.ndarray, op: str) -> np.ndarray:
    """The reduction over one process."""
    return values


def mpi_allreduce(comm: Any) -> Allreduce:
    """The reduction over the processes of an mpi4py communicator (ComIn's host communicator)."""
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]: only inside ICON

    ops = {"sum": MPI.SUM, "max": MPI.MAX, "min": MPI.MIN}

    def allreduce(values: np.ndarray, op: str) -> np.ndarray:
        send = np.ascontiguousarray(values)
        receive = np.empty_like(send)
        comm.Allreduce(send, receive, op=ops[op])
        return receive

    return allreduce


@dataclasses.dataclass(frozen=True)
class Pair:
    """One field: ICON's result and the plugin's, 2-D (nproma, levels), and its real entries."""

    name: str
    reference: Any
    field: Any
    real: int
    """The number of real entries along the first axis: the 'domain' table covers [0, real),
    the 'padding' table [real, nproma)."""


@dataclasses.dataclass(frozen=True)
class Row:
    """One row of a table: the global statistics of one field over one range."""

    name: str
    close: int
    """Entries within the tolerance."""
    size: int
    """Entries compared."""
    abs_sq: float
    """The sum of the squared differences."""
    field_sq: float
    """The sum of the squared values of 'field'."""
    max_rel_err: float
    location: tuple[int, int]
    """1-based (first index, second index) of the largest relative error; (0, 0) if none."""
    max_abs_err: float
    max_ref: float
    max_field: float
    min_ref: float
    min_field: float

    @property
    def is_close(self) -> bool:
        return self.close == self.size

    @property
    def percent(self) -> float:
        return _divide(100.0 * self.close, float(self.size))

    @property
    def rel_mse(self) -> float:
        """ICON's 'rel MSE' column: the squared differences over the squared values."""
        return _divide(self.abs_sq, self.field_sq)

    @property
    def abs_mse(self) -> float:
        """ICON's 'abs MSE' column: the squared differences over the number of entries."""
        return _divide(self.abs_sq, float(self.size))


def _divide(a: float, b: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _array_module(array: Any) -> ModuleType:
    if isinstance(array, np.ndarray):
        return np
    import cupy  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415 [import-outside-top-level]

    return cupy


class Copies:
    """
    The plugin's buffers for ICON's input: one per name, allocated at the first copy (of the
    source's array module, shape and dtype, in Fortran order: py2fgen's layout) and kept.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, Any] = {}

    def take(self, name: str, source: Any) -> Any:
        """Copy 'source' (a NumPy or CuPy array) into the buffer of 'name'; return the buffer."""
        xp = _array_module(source)
        buffer = self._buffers.get(name)
        if buffer is None:
            buffer = xp.empty(source.shape, dtype=source.dtype, order="F")
            self._buffers[name] = buffer
        elif (
            buffer.shape != source.shape
            or buffer.dtype != source.dtype
            or _array_module(buffer) is not xp
        ):
            raise ValueError(
                f"'{name}': ICON's array has the shape {tuple(source.shape)} and dtype"
                f" {source.dtype}, the plugin's buffer {tuple(buffer.shape)} and {buffer.dtype}."
            )
        buffer[...] = source
        return buffer

    def nbytes(self) -> int:
        return sum(int(b.nbytes) for b in self._buffers.values())

    def release(self) -> None:
        self._buffers.clear()


_MAXIMA, _MINIMA = 4, 2


def _local(reference: Any, field: Any, start: int, end: int) -> tuple[np.ndarray, ...]:
    """
    The local statistics of one field over the rows [start, end) (ICON's first loop): integer
    sums (close, size), float sums (squared differences, squared values), maxima (absolute
    error, relative error, |reference|, |field|), minima (|reference|, |field|, from 0).
    """
    levels = int(reference.shape[1])
    end = max(start, end)
    size = levels * (end - start)
    if size == 0:
        return (
            np.array([0, 0], dtype=np.int64),
            np.zeros(2),
            np.zeros(_MAXIMA),
            np.zeros(_MINIMA),
        )
    xp = _array_module(reference)
    ref, fld = reference[start:end, :], field[start:end, :]
    abs_ref = xp.abs(ref)
    abs_err = xp.abs(ref - fld)
    nonzero = ref != 0.0
    rel_err = xp.where(nonzero, abs_err / xp.where(nonzero, abs_ref, 1.0), 0.0)
    close = abs_err <= ATOL + RTOL * abs_ref
    values = [
        int(xp.count_nonzero(close)),
        float(xp.sum(abs_err * abs_err)),
        float(xp.sum(fld * fld)),
        float(abs_err.max()),
        float(rel_err.max()),
        float(abs_ref.max()),
        float(xp.abs(fld).max()),
        float(abs_ref.min()),
        float(xp.abs(fld).min()),
    ]
    return (
        np.array([values[0], size], dtype=np.int64),
        np.array(values[1:3]),
        np.maximum(np.array(values[3:7]), 0.0),
        np.minimum(np.array(values[7:9]), 0.0),
    )


def _location(reference: Any, field: Any, start: int, end: int, max_rel_err: float) -> list[int]:
    """
    ICON's second loop: the 1-based location of an entry whose relative error is the global
    largest one (the last such entry in Fortran order here), or (0, 0) if this PE has none.
    """
    rows = max(0, end - start)
    if rows == 0:
        return [0, 0]
    xp = _array_module(reference)
    ref, fld = reference[start:end, :], field[start:end, :]
    nonzero = ref != 0.0
    rel_err = xp.abs(ref - fld) / xp.where(nonzero, xp.abs(ref), 1.0)
    match = nonzero & (rel_err == max_rel_err)
    index = xp.arange(rows)[:, None] + rows * xp.arange(ref.shape[1])[None, :]
    last = int(xp.where(match, index, -1).max())
    if last < 0:
        return [0, 0]
    return [start + last % rows + 1, last // rows + 1]


def table(pairs: Sequence[Pair], which: str, allreduce: Allreduce = local_allreduce) -> list[Row]:
    """
    The rows of table 'which' ('domain' or 'padding') for 'pairs', reduced over ICON's work PEs
    with 'allreduce'. Every PE must call it with the same fields, in the same order (collective).
    """
    if which not in TABLES:
        raise ValueError(f"Unknown table {which!r}: expected one of {', '.join(TABLES)}.")
    ranges = []
    for pair in pairs:
        if pair.reference.shape != pair.field.shape or len(pair.reference.shape) != 2:
            raise ValueError(
                f"'{pair.name}': ICON's field has the shape {tuple(pair.reference.shape)}, the"
                f" plugin's {tuple(pair.field.shape)}; expected the same 2-D shape."
            )
        nproma = int(pair.reference.shape[0])
        ranges.append((0, pair.real) if which == "domain" else (pair.real, nproma))
    if not pairs:
        return []
    local = [_local(p.reference, p.field, *r) for p, r in zip(pairs, ranges, strict=True)]
    counts = allreduce(np.stack([x[0] for x in local]), "sum")
    sums = allreduce(np.stack([x[1] for x in local]), "sum")
    maxima = allreduce(np.stack([x[2] for x in local]), "max")
    minima = allreduce(np.stack([x[3] for x in local]), "min")
    locations = [
        _location(p.reference, p.field, *r, float(maxima[i, 1]))
        for i, (p, r) in enumerate(zip(pairs, ranges, strict=True))
    ]
    where = allreduce(np.array(locations, dtype=np.int64).reshape(-1, 2), "max")
    return [
        Row(
            name=p.name,
            close=int(counts[i, 0]),
            size=int(counts[i, 1]),
            abs_sq=float(sums[i, 0]),
            field_sq=float(sums[i, 1]),
            max_rel_err=float(maxima[i, 1]),
            location=(int(where[i, 0]), int(where[i, 1])),
            max_abs_err=float(maxima[i, 0]),
            max_ref=float(maxima[i, 2]),
            max_field=float(maxima[i, 3]),
            min_ref=float(minima[i, 0]),
            min_field=float(minima[i, 1]),
        )
        for i, p in enumerate(pairs)
    ]


# ---- ICON's format -------------------------------------------------------------------------------


def fortran_a(text: str, width: int) -> str:
    """Fortran's 'Aw' for output: right-justified, or the leftmost 'width' characters."""
    return text[:width] if len(text) >= width else text.rjust(width)


def _special(value: float, width: int) -> str | None:
    if math.isnan(value):
        return "NaN".rjust(width)
    if math.isinf(value):
        text = ("-" if value < 0 else "") + "Infinity"
        if len(text) > width:
            text = ("-" if value < 0 else "") + "Inf"
        return text.rjust(width)
    return None


def fortran_e(value: float, width: int = 12, digits: int = 4) -> str:
    """Fortran's 'Ew.d' for output: '0.dddd' and a two-digit exponent 'E+nn' (three: '+nnn')."""
    special = _special(value, width)
    if special is not None:
        return special
    sign = "-" if math.copysign(1.0, value) < 0 and value != 0.0 else ""
    if value == 0.0:
        mantissa, exponent = "0" * digits, 0
    else:
        text = f"{abs(value):.{digits - 1}e}"  # d.ddde+xx, correctly rounded
        significand, power = text.split("e")
        mantissa, exponent = significand.replace(".", ""), int(power) + 1
    if abs(exponent) <= 99:
        suffix = f"E{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    else:
        suffix = f"{'-' if exponent < 0 else '+'}{abs(exponent):03d}"
    return f"{sign}0.{mantissa}{suffix}".rjust(width)


def fortran_f(value: float, width: int = 8, decimals: int = 1) -> str:
    """Fortran's 'Fw.d' for output."""
    special = _special(value, width)
    if special is not None:
        return special
    return f"{value:{width}.{decimals}f}"


def fortran_i(value: int, width: int) -> str:
    """Fortran's 'Iw' for output."""
    return str(value).rjust(width)


_E12 = 12


def format_row(row: Row) -> str:
    """One row as 'verify_field_wp' writes it (without the leading blank of ICON's 'message')."""
    location = f"({fortran_i(row.location[0], 5)},{fortran_i(row.location[1], 4)})"
    columns = [
        fortran_a(row.name, 20),
        ("Yes" if row.is_close else "No").ljust(8),  # ICON's CHARACTER(LEN=8), as 'A8'
        fortran_f(row.percent),
        fortran_e(row.rel_mse),
        fortran_e(row.abs_mse),
        fortran_e(row.max_rel_err),
        location,
        fortran_e(row.max_abs_err),
        location,
        fortran_e(row.max_ref),
        fortran_e(row.max_field),
        fortran_e(row.min_ref),
        fortran_e(row.min_field),
    ]
    return "  ".join(columns)


_DASHES: Final = "  ".join(["-" * 20, "-" * 8, "-" * 8] + ["-" * _E12] * 10)


def header_lines(which: str) -> list[str]:
    """The lines before a table's rows ('print_verify_header'), without blank lines."""
    names = [
        fortran_a("field", 20),
        fortran_a("is_close", 8),
        fortran_a("percent", 8),
        fortran_a("rel MSE", _E12),
        fortran_a("abs MSE", _E12),
        fortran_a("max_rel_err", _E12),
        f"({fortran_a('jc/e', 5)}, {fortran_a('jk', 3)})",
        fortran_a("max_abs_err", _E12),
        f"({fortran_a('jc/e', 5)}, {fortran_a('jk', 3)})",
        fortran_a("max_abs_ref", _E12),
        fortran_a("max_abs_field", _E12),
        fortran_a("min_abs_ref", _E12),
        fortran_a("min_abs_field", _E12),
    ]
    return [f"-------------------- {which} --------------------", "  ".join(names), _DASHES]


def footer_line() -> str:
    """The line after a table's rows ('print_verify_footer')."""
    return _DASHES
