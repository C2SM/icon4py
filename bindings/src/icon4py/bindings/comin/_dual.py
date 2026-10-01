# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Dual-channel check: compare an argument from its new route with a reference, bit by bit.

While the plugin moves the arguments of the py2fgen-exported functions from ICON's argument
variables (the "old route") to ComIn's native data (descriptive data, ICON variables, the
namelist output), every moved item is also taken from a reference and compared, and the log
names each item that differs. The references are pluggable: the old ComIn route (the argument
variables and carriers), or py2fgen's own arguments (a recorder on py2fgen's call path).

Each argument has a provider ('Source.provider'):
- OLD: the granule gets the old route; nothing is compared.
- NEW: the granule gets the new route; the reference only feeds the check.
- OBSERVE: the new route is computed and compared; reported, never gated, never used.

Comparisons are exact: the raw bytes of every element (so NaN payloads and signed zeros
count), 'float.hex' and the IEEE bits for scalars, and the pointers, shape and presence for
ICON variables that the plugin passes through. Array entries beyond the local number of
cells, edges or vertices along the location axis are padding and reported separately: the
verdict is on the real entries. An item may also require the same form of the value as the
granule gets it ('form': Field or plain array, array module, dtype, shape, element strides),
and an MPI communicator is compared by MPI_Comm_compare and its members ('Communicator'),
not by the value of its handle.

Environment:
- ICON4PY_COMIN_DUAL=strict|report|off (default strict): 'strict' raises at the first differing
  NEW item (ComIn turns that into ICON's finish), 'report' logs every item and continues,
  'off' compares nothing.
- ICON4PY_COMIN_DUAL_SELFTEST=all|<item>[,<item>...] (default unset): perturb one real element
  of a comparison-only copy of the new-route value of the named NEW/OBSERVE items (all of
  them for 'all'), so that the check must report exactly those items as differing. The
  granule's input and ICON's memory are never touched: the positive control in real runs.

Log lines (a stable format, for automated checks of a run):
  dual plan <EP>: <n> items (NEW <n>, OBSERVE <n>)[: <names>]
  dual <EP> <pass|call> <k> <provider> <item> vs <reference>[ [selftest]]: identical|differ (<details>)
  dual <EP> <pass|call> <k>: <n> checked (NEW <n>, OBSERVE <n>), <n> identical, <n> differ
      (NEW <n>, OBSERVE <n>); selftest <n>
"""

import dataclasses
import functools
import math
import struct
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np


MODE_ENV: Final = "ICON4PY_COMIN_DUAL"
SELFTEST_ENV: Final = "ICON4PY_COMIN_DUAL_SELFTEST"
STRICT, REPORT, OFF = "strict", "report", "off"
MODES: Final = (STRICT, REPORT, OFF)

OLD, NEW, OBSERVE = "OLD", "NEW", "OBSERVE"
PROVIDERS: Final = (OLD, NEW, OBSERVE)
CHECKED: Final = (NEW, OBSERVE)
CLASSES: Final = ("A", "B", "C", "D", "E")
"""Argument classes (ICON variable, descriptive data, derived, configuration, not in ComIn 1.0)."""
LOCATIONS: Final = ("cell", "edge", "vertex")

REF_OLD: Final = "old"
"""Reference: the old ComIn route (ICON's argument variables and carriers)."""
REF_PY2FGEN: Final = "py2fgen"
"""Reference: py2fgen's own argument (the py2fgen probe)."""


@dataclasses.dataclass(frozen=True)
class Source:
    """Where one argument of an exported function comes from."""

    klass: str
    """Argument class: A (ICON variable), B (descriptive data), C (derived), D (configuration),
    E (not in ComIn 1.0)."""
    provider: str
    """OLD, NEW or OBSERVE."""
    location: str | None = None
    """For arrays: 'cell', 'edge' or 'vertex' if the entries along 'axis' beyond the local count
    of that location are padding; 'None' if every entry is real."""
    axis: int = 0

    def __post_init__(self) -> None:
        if self.klass not in CLASSES:
            raise ValueError(f"Unknown argument class {self.klass!r}.")
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {self.provider!r}.")
        if self.location is not None and self.location not in LOCATIONS:
            raise ValueError(f"Unknown location {self.location!r}.")


def parse_mode(environ: Mapping[str, str]) -> str:
    mode = environ.get(MODE_ENV, STRICT).strip().lower() or STRICT
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV}={mode!r}: expected one of {', '.join(MODES)}.")
    return mode


def parse_selftest(environ: Mapping[str, str], checked: Iterable[str]) -> frozenset[str]:
    """The items to perturb; 'checked' are the NEW and OBSERVE items. Unknown names raise."""
    raw = environ.get(SELFTEST_ENV, "").strip()
    checked = frozenset(checked)
    if not raw:
        return frozenset()
    if raw == "all":
        return checked
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    unknown = sorted(names - checked)
    if unknown:
        raise ValueError(
            f"{SELFTEST_ENV}={raw!r}: {', '.join(unknown)} is not a NEW or OBSERVE item"
            f" (those are: {', '.join(sorted(checked)) or 'none'})."
        )
    return names


# ---- values -----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pointer:
    """An ICON variable as the granule sees it: addresses, squeezed shape, presence."""

    device: int | None
    host: int | None
    shape: tuple[int, ...]
    present: bool


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
    if value is None or isinstance(value, (np.ndarray, Pointer, Communicator, bool, int, float)):
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


def first_block(array: Any, block_axis: int) -> np.ndarray:
    """Drop ICON's block axis (index 0); EXCLAIM runs with one block (nproma >= local size)."""
    array = np.asarray(array)
    if array.shape[block_axis] != 1:
        raise ValueError(
            f"{array.shape[block_axis]} blocks along axis {block_axis} of shape {array.shape}:"
            " the icon4py arguments are the first block, which is everything only for nblks == 1."
        )
    index: list[Any] = [slice(None)] * array.ndim
    index[block_axis] = 0
    return array[tuple(index)]


def squeeze5(array: Any, rank: int, block_axis: int | None = None) -> Any:
    """
    The py2fgen-shaped view of a 5-D ComIn field: drop the block axis (if any, see
    'first_block'), then the trailing padding axes, which must have extent 1.

    ComIn exposes ICON's 3-D fields as (nproma, nlev, nblks, 1, 1) and 2-D fields as
    (nproma, nblks, 1, 1, 1); py2fgen gets (nproma, nlev) and (nproma,).
    """
    if block_axis is not None:
        array = first_block(array, block_axis)
    extra = array.shape[rank:]
    if any(e != 1 for e in extra):
        raise ValueError(f"Cannot squeeze shape {array.shape} to rank {rank}.")
    return array[(slice(None),) * rank + (0,) * len(extra)]


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


def compare_pointers(new: Pointer, old: Pointer) -> Comparison:
    """Same device and host address, the same squeezed shape, the same presence."""

    def show(p: Pointer) -> str:
        device = "-" if p.device is None else f"{p.device:#x}"
        host = "-" if p.host is None else f"{p.host:#x}"
        return f"device {device}, host {host}, shape {p.shape}, present {p.present}"

    same = new == old
    return Comparison(same, show(new) if same else f"{show(new)} vs {show(old)}")


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
    if isinstance(new, Pointer) or isinstance(old, Pointer):
        if not (isinstance(new, Pointer) and isinstance(old, Pointer)):
            return Comparison(
                False, f"pointer vs value: {type(new).__name__}, {type(old).__name__}"
            )
        return compare_pointers(new, old)
    if isinstance(new, (bool, int, float)) or isinstance(old, (bool, int, float)):
        return compare_scalars(new, old)
    return compare_arrays(host_array(new), host_array(old), real, axis)


def perturb(value: Any) -> Any:
    """
    A comparison-only copy of 'value' with one element changed: the lowest bit of the first
    element (a real entry whenever there is one), i.e. 1 ulp, +-1 or a flipped bool; for
    scalars the next float, +1 or 'not'; for a pointer the address + 8; for a communicator its
    first member + 1 (no MPI call). Never a view.
    """
    if isinstance(value, Communicator):
        first = value.members[0] + 1 if value.members else -1
        return dataclasses.replace(value, members=(first, *value.members[1:]))
    if isinstance(value, Pointer):
        if value.device is not None:
            return dataclasses.replace(value, device=value.device + 8)
        return dataclasses.replace(value, host=(value.host or 0) + 8)
    if isinstance(value, (bool, int, float)):
        return _perturb_scalar(value)
    if value is None:
        return None
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


# ---- the check of one entry point ----------------------------------------------------------


LAZY: Final = (functools.partial, types.FunctionType, types.MethodType)
"""The types of a new route given as a function, which the check calls (a Field is callable,
but a value)."""


@dataclasses.dataclass(frozen=True)
class Item:
    """One comparison: 'new' may be a function (LAZY), so that a failing new route is reported."""

    name: str
    provider: str
    new: Any
    reference: Any
    real: int | None = None
    axis: int = 0
    form: str | None = None
    """If set: the reference's 'form', which the new value must have too (before a self-test
    perturbation, which changes one value of a host copy)."""


class DualCheckError(RuntimeError):
    pass


@dataclasses.dataclass
class Summary:
    checked: dict[str, int] = dataclasses.field(default_factory=lambda: {NEW: 0, OBSERVE: 0})
    differ: dict[str, int] = dataclasses.field(default_factory=lambda: {NEW: 0, OBSERVE: 0})
    selftest: int = 0

    @property
    def total(self) -> int:
        return sum(self.checked.values())


class Checker:
    """
    Runs the comparisons of one entry point call and logs them ('log(message)' prints one
    line on the current rank). A new Checker per entry point call; 'first' logs every item,
    later calls only the differing ones.
    """

    def __init__(
        self,
        mode: str,
        selftest: frozenset[str],
        log: Callable[[str], None],
        reference: str = REF_OLD,
    ) -> None:
        self.mode = mode
        self.selftest = selftest
        self._log = log
        self.reference = reference

    def check(self, where: str, items: Sequence[Item], first: bool) -> Summary:
        summary = Summary()
        if self.mode == OFF:
            return summary
        for item in items:
            if item.provider not in CHECKED:
                continue
            perturbed = item.name in self.selftest
            try:
                new = item.new() if isinstance(item.new, LAZY) else item.new
                new_form = None if item.form is None else form(new)
                if perturbed:
                    new = perturb(new)
                if new_form is not None and new_form != item.form:
                    result = Comparison(False, f"form {new_form} vs {item.form}")
                else:
                    result = compare(new, item.reference, item.real, item.axis)
            except Exception as error:  # a failing new route is a difference, not a crash
                result = Comparison(False, f"new route failed: {error!r}")
            summary.checked[item.provider] += 1
            summary.selftest += perturbed
            if not result.identical:
                summary.differ[item.provider] += 1
            if first or not result.identical:
                self._log(
                    f"dual {where} {item.provider} {item.name} vs {self.reference}"
                    f"{' [selftest]' if perturbed else ''}:"
                    f" {'identical' if result.identical else 'differ'} ({result.details})"
                )
            if self.mode == STRICT and item.provider == NEW and not result.identical:
                raise DualCheckError(
                    f"icon4py ComIn plugin: the new route of '{item.name}' differs from the"
                    f" {self.reference} route at {where} ({result.details});"
                    f" {MODE_ENV}={REPORT} logs every difference and continues."
                )
        return summary

    def log_summary(self, where: str, summary: Summary) -> None:
        if self.mode == OFF:
            return
        identical = summary.total - sum(summary.differ.values())
        self._log(
            f"dual {where}: {summary.total} checked (NEW {summary.checked[NEW]},"
            f" OBSERVE {summary.checked[OBSERVE]}), {identical} identical,"
            f" {sum(summary.differ.values())} differ (NEW {summary.differ[NEW]},"
            f" OBSERVE {summary.differ[OBSERVE]}); selftest {summary.selftest}"
        )
