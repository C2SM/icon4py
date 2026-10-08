# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Bitwise comparisons of a value the plugin builds with a reference, for the py2fgen probe.

The py2fgen probe ('_probe.py') compares each input the plugin hands icon4py's granule, and
each object the plugin builds from them, with py2fgen's in the same run, and names each item
that differs. The comparisons are exact: the raw bytes of every element (so NaN payloads and
signed zeros count), 'float.hex' and the IEEE bits for scalars. Array entries beyond the local
number of cells, edges or vertices along the location axis are padding and reported
separately: the verdict is on the real entries. An array may also be required to have the same
form as the granule gets it ('form': Field or plain array, array module, dtype, shape, element
strides), and an MPI communicator is compared by MPI_Comm_compare and its members
('Communicator'), not by the value of its handle. 'perturb' gives a comparison-only copy with
one element changed, for the positive controls.
"""

import dataclasses
import math
import struct
from typing import Any, Final

import numpy as np


# ---- values -----------------------------------------------------------------------------------


SAME_COMMUNICATOR: Final = ("IDENT", "CONGRUENT")
"""MPI_Comm_compare's results for which two communicators have the same processes in the
same order (CONGRUENT: a different context, e.g. a split or a duplicate)."""


@dataclasses.dataclass(frozen=True)
class Communicator:
    """
    An MPI communicator argument (py2fgen passes its Fortran handle): the handle, its members
    (their ranks in MPI_COMM_WORLD, in the communicator's rank order) and, for the new route,
    MPI_Comm_compare's result against the reference. Two communicators are the same argument
    iff that result is IDENT or CONGRUENT and they have the same members; the handles may
    differ.
    """

    handle: int
    members: tuple[int, ...]
    relation: str | None = None


def host_array(value: Any) -> Any:
    """A NumPy copy or view of an argument value: gtx.Field, CuPy or NumPy array (else as is)."""
    if value is None or isinstance(value, (np.ndarray, Communicator, bool, int, float, str)):
        return value
    ndarray = getattr(value, "ndarray", None)  # a gtx.Field
    if ndarray is not None:
        value = ndarray
    if isinstance(value, np.ndarray):
        return value
    get = getattr(value, "get", None)  # a CuPy array
    if callable(get):
        return get()
    return np.asarray(value)


def form(value: Any) -> str:
    """
    The form of an array argument as the granule gets it: a Field (with its dimensions) or a
    plain array, the array module (numpy, cupy), the dtype, the shape and the strides in
    elements (0 for an axis of extent 1, whose stride does not matter).
    """
    if value is None:
        return "None"
    prefix = ""
    ndarray = getattr(value, "ndarray", None)  # a gtx.Field
    if ndarray is not None:
        dims = ", ".join(str(getattr(d, "value", d)) for d in value.domain.dims)
        prefix, value = f"Field[{dims}] over ", ndarray
    module = type(value).__module__.split(".")[0]
    shape = tuple(int(n) for n in value.shape)
    strides = tuple(
        int(s) // value.itemsize if n > 1 else 0 for s, n in zip(value.strides, shape, strict=True)
    )
    return f"{prefix}{module} {value.dtype} {shape} strides {strides}"


# ---- comparisons ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Comparison:
    identical: bool
    """The verdict: presence, type, shape and every real entry bitwise equal."""
    details: str
    real_differ: int = 0
    padding_differ: int = 0


def _bits(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.uint8).reshape((*contiguous.shape, contiguous.dtype.itemsize))


def _ordered(values: np.ndarray) -> list[int]:
    """IEEE floats as integers whose order and differences count ulps."""
    width = values.dtype.itemsize * 8
    ints = values.view(np.dtype(f"i{values.dtype.itemsize}")).tolist()
    return [i if i >= 0 else -(1 << (width - 1)) - i for i in ints]


def _magnitudes(new: np.ndarray, old: np.ndarray) -> str:
    """Largest absolute and relative difference (and ulp distance) of the given entries."""
    if new.dtype.kind == "f":
        with np.errstate(all="ignore"):
            a, b = new.astype(np.float64), old.astype(np.float64)
            diff = np.abs(a - b)
            rel = diff / np.abs(b)
        ulps = [abs(x - y) for x, y in zip(_ordered(new), _ordered(old))]
        abs_max = np.nanmax(diff) if not np.all(np.isnan(diff)) else math.nan
        rel_max = np.nanmax(rel) if not np.all(np.isnan(rel)) else math.nan
        zeros = "".join(
            f", {route} zero at {int(n)}"
            for route, n in (("reference", np.sum(b == 0)), ("new route", np.sum(a == 0)))
            if n
        )
        return f", max abs {abs_max:.3g}, max rel {rel_max:.3g}, max ulp {max(ulps)}{zeros}"
    if new.dtype.kind in "iu":
        diff = np.abs(new.astype(np.int64) - old.astype(np.int64))
        with np.errstate(all="ignore"):
            rel = diff / np.abs(old.astype(np.float64))
        return f", max abs {int(diff.max())}, max rel {float(np.nanmax(rel)):.3g}"
    return ""


def _describe(array: np.ndarray) -> str:
    return f"{array.dtype} {tuple(array.shape)}"


def compare_arrays(new: Any, old: Any, real: int | None = None, axis: int = 0) -> Comparison:
    """
    Bitwise comparison of two host arrays; entries along 'axis' at or beyond 'real' are
    padding. 'None' stands for an absent argument.
    """
    if new is None or old is None:
        if new is None and old is None:
            return Comparison(True, "absent on both routes")
        where = "reference" if new is None else "new route"
        return Comparison(False, f"present only on the {where}")
    new, old = np.asarray(new), np.asarray(old)
    if new.dtype != old.dtype or new.shape != old.shape:
        return Comparison(False, f"{_describe(new)} vs {_describe(old)}")
    differ = (_bits(new) != _bits(old)).any(axis=-1)
    if new.ndim == 0:
        n_real = None
    else:
        n_real = new.shape[axis] if real is None else max(0, min(real, new.shape[axis]))
    if n_real is None:
        real_mask = differ.reshape(1)
        pad_mask = np.zeros(0, dtype=bool)
        real_index: tuple[Any, ...] = ()
    else:
        real_index = (slice(None),) * axis + (slice(0, n_real),)
        pad_index = (slice(None),) * axis + (slice(n_real, None),)
        real_mask, pad_mask = differ[real_index], differ[pad_index]
    n_real_differ, n_pad_differ = int(real_mask.sum()), int(pad_mask.sum())
    text = f"{_describe(new)}; real {real_mask.size}: {n_real_differ} differ"
    if n_real_differ:
        first = tuple(int(i) for i in np.argwhere(real_mask)[0])
        new_real = new[real_index] if real_index else new.reshape(1)
        old_real = old[real_index] if real_index else old.reshape(1)
        text += f", first {first}" + _magnitudes(new_real[real_mask], old_real[real_mask])
    text += f"; padding {pad_mask.size}: {n_pad_differ} differ"
    return Comparison(n_real_differ == 0, text, n_real_differ, n_pad_differ)


def _scalar_bits(value: Any) -> str:
    if isinstance(value, str):
        return f"str {value!r}"
    if isinstance(value, bool):
        return f"bool {value}"
    if isinstance(value, int):
        return f"int {value}"
    if isinstance(value, float):
        return f"float {value!r} ({value.hex()}, bits {struct.pack('<d', value).hex()})"
    return f"{type(value).__name__} {value!r}"


def compare_scalars(new: Any, old: Any) -> Comparison:
    """Same Python type and the same bits (for floats: IEEE bits, so -0.0 != 0.0)."""
    same = type(new) is type(old) and _scalar_bits(new) == _scalar_bits(old)
    text = _scalar_bits(new) if same else f"{_scalar_bits(new)} vs {_scalar_bits(old)}"
    return Comparison(same, text)


def _handle(handle: int) -> str:
    return f"{handle & 0xFFFFFFFF:#010x}"


def compare_communicators(new: Communicator, old: Communicator) -> Comparison:
    """MPI_Comm_compare IDENT or CONGRUENT (the new route's verdict) and the same members."""
    same_members = new.members == old.members
    same = same_members and new.relation in SAME_COMMUNICATOR
    text = (
        f"MPI_Comm_compare {new.relation or 'unknown'}; handle {_handle(new.handle)} vs"
        f" {_handle(old.handle)}; {len(new.members)} ranks"
    )
    if not same_members:
        text += f"; members {new.members} vs {old.members}"
    return Comparison(same, text)


def compare(new: Any, old: Any, real: int | None = None, axis: int = 0) -> Comparison:
    if isinstance(new, Communicator) or isinstance(old, Communicator):
        if not (isinstance(new, Communicator) and isinstance(old, Communicator)):
            return Comparison(
                False, f"communicator vs value: {type(new).__name__}, {type(old).__name__}"
            )
        return compare_communicators(new, old)
    if isinstance(new, (bool, int, float, str)) or isinstance(old, (bool, int, float, str)):
        return compare_scalars(new, old)
    return compare_arrays(host_array(new), host_array(old), real, axis)


def perturb(value: Any) -> Any:
    """
    A comparison-only copy of 'value' with one element changed: the lowest bit of the first
    element (a real entry whenever there is one), i.e. 1 ulp, +-1 or a flipped bool; for
    scalars the next float, +1 or 'not'; for a string one more character; for a communicator
    its first member + 1 (no MPI call); for 'None' (an absent array) a one-element array, i.e.
    a present one. Never a view.
    """
    if isinstance(value, Communicator):
        first = value.members[0] + 1 if value.members else -1
        return dataclasses.replace(value, members=(first, *value.members[1:]))
    if isinstance(value, str):
        return value + "~"
    if isinstance(value, (bool, int, float)):
        return _perturb_scalar(value)
    if value is None:
        return np.zeros(1, dtype=np.uint8)
    copy = np.array(host_array(value), copy=True, order="C")
    if copy.size:
        flat = copy.reshape(-1).view(np.uint8)
        flat[0 if np.little_endian or copy.dtype.itemsize == 1 else copy.dtype.itemsize - 1] ^= 1
    return copy


def _perturb_scalar(value: bool | int | float) -> bool | int | float:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return 0.0 if math.isnan(value) else math.nextafter(value, math.inf)
