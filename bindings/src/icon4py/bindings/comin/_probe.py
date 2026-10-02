# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The py2fgen probe: compare what this plugin would hand icon4py's granules with what py2fgen
hands them, argument by argument, in the same run.

ICON computes the diffusion through py2fgen ('icon4py_interface=0', 'icon4py_mode=1') and
loads this plugin with ICON4PY_COMIN_PROBE=1. py2fgen's glue runs the callables named in
PY2FGEN_EXTRA_CALLABLES before it imports the exported functions, so with

    PY2FGEN_EXTRA_CALLABLES=icon4py.bindings.comin._probe:install

it binds the recorders that 'install' puts in place of 'grid_init', 'diffusion_init' and
'diffusion_run': copies of the exported functions whose inner function records or compares
py2fgen's arguments, the very values the granule gets, and then calls the original function
unchanged. 'install' replaces nothing unless the plugin is registered in probe mode.

In probe mode the plugin computes nothing and writes nothing. It reads ICON's namelist output,
ComIn's descriptive data and ICON's variables (READ, on the device on the GPU) as it does when
it computes, and builds its arguments where it would hand them to the granule:
- 'grid_init', 'diffusion_init': ICON calls them in 'icon4py_init', before EP_ATM_TIMELOOP_BEFORE
  of the same pass. The recorder keeps py2fgen's arguments of each call (host copies of the
  arrays, with their form and address; the scalars). At EP_ATM_TIMELOOP_BEFORE, where the
  plugin calls these functions when it computes, it builds its arguments as for the granule
  and compares them with the recorded call of that pass ('Probe.compare_init').
- 'diffusion_run': at EP_ATM_DYCORE_DIFFUSION_BEFORE of domain 1 the plugin keeps what it would
  hand the granule there (ICON's variables as zero-copy views: address, form, presence; the
  time step; the initial-call flag; 'Probe.expect'). ICON then calls py2fgen's 'diffusion_run'
  and the recorder compares its arguments with them, before the granule runs.

The comparison of one argument ('compare'):
- an array: the presence, the form ('_dual.form': Field or plain array, array module, dtype,
  shape, element strides), the address of the first element where the plugin hands the granule
  ICON's own memory (class A of the source table), and the values where they were copied, bit
  by bit, real entries and padding reported separately ('_dual.compare_arrays'; the verdict is
  on the real entries). Bool arrays are compared as values: py2fgen views Fortran's logicals
  without normalising them (nvfortran writes .TRUE. as 255); the byte values are logged.
- a scalar: the same Python type and bits ('_dual.compare_scalars'); the communicator by
  MPI_Comm_compare and its members ('_descrdata.communicators').
The positive control: every comparison of an identical item is repeated with a perturbed copy
of the plugin's value ('perturb'), which must be reported as differing (a differing item is
reported already).

Log lines (one per item at the first comparison of a function, later the differing ones):
  probe <function> <pass|call> <k> <item> vs py2fgen: identical|differ (<details>)
  probe <function> <pass|call> <k> control <item>: not reported (<details>)
  probe <function> <pass|call> <k>: <n> compared, <n> identical, <n> differ[: <items>];
      control <n> of <n> reported; not probed <n>[: <items>]
  probe summary <function>: py2fgen calls <n>, compared <n> (<n> items each), ...
"""

import copy
import dataclasses
import functools
import importlib
import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np

from icon4py.bindings.comin import _descrdata, _dual, _views
from icon4py.tools import py2fgen


ENV: Final = "ICON4PY_COMIN_PROBE"
INSTALL: Final = f"{__name__}:install"
"""The value of PY2FGEN_EXTRA_CALLABLES that installs the recorders."""
ABSENT: Final = "None"
"""The form of an absent array ('_dual.form(None)')."""
ALL_RANKS: Final[Mapping[str, Any]] = {"all_ranks": True}

log = logging.getLogger(__name__)


def parse(environ: Mapping[str, str]) -> bool:
    """Whether ICON4PY_COMIN_PROBE switches the probe on (0 or 1)."""
    value = environ.get(ENV, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: expected 0 or 1.")
    return value == "1"


# ---- values ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Array:
    """An array argument as a granule gets it, reduced to what the probe compares."""

    form: str
    """'_dual.form': Field or plain array, array module, dtype, shape, element strides."""
    address: int | None
    """The address of the first element (on the device for a device array); 'None' if absent."""
    values: np.ndarray | None = None
    """A host copy (bools as values), where one was taken."""
    raw: tuple[int, ...] = ()
    """For a bool array with a copy: the byte values in its memory."""


def capture(value: Any, values: bool) -> Array:
    """An array argument (gtx.Field, CuPy or NumPy array, or 'None'); a host copy if 'values'."""
    if value is None:
        return Array(ABSENT, None)
    data = getattr(value, "ndarray", value)
    address = _views.data_ptr(data)
    if not values:
        return Array(_dual.form(value), address)
    host = np.array(_dual.host_array(value), copy=True)
    raw: tuple[int, ...] = ()
    if host.dtype == np.bool_:
        raw = tuple(int(b) for b in np.unique(np.ascontiguousarray(host).view(np.uint8)))
        host = host != 0
    return Array(_dual.form(value), address, host, raw)


def _address(value: Array) -> str:
    return "-" if value.address is None else f"{value.address:#x}"


def compare_arrays(
    new: Array, reference: Array, real: int | None, axis: int, same_memory: bool
) -> _dual.Comparison:
    """Presence, form, address (if 'same_memory') and copied values ('real': see '_dual')."""
    if ABSENT in (new.form, reference.form):
        if new.form == reference.form:
            return _dual.Comparison(True, "absent on both routes")
        where = "reference" if new.form == ABSENT else "new route"
        return _dual.Comparison(False, f"present only on the {where}")
    if new.form != reference.form:
        return _dual.Comparison(False, f"form {new.form} vs {reference.form}")
    parts = []
    if same_memory:
        if new.address != reference.address:
            return _dual.Comparison(
                False, f"address {_address(new)} vs {_address(reference)}; {new.form}"
            )
        parts.append(f"same address {_address(new)}")
    if new.values is None or reference.values is None:
        return _dual.Comparison(True, "; ".join([*parts, new.form]))
    result = _dual.compare_arrays(new.values, reference.values, real, axis)
    text = result.details
    if new.raw or reference.raw:
        text += f"; as values, bytes {list(new.raw)} vs {list(reference.raw)}"
    return dataclasses.replace(result, details="; ".join([*parts, text]))


def compare(
    new: Any, reference: Any, real: int | None = None, axis: int = 0, same_memory: bool = False
) -> _dual.Comparison:
    """One argument: arrays by 'compare_arrays', anything else by '_dual.compare'."""
    if isinstance(new, Array) and isinstance(reference, Array):
        return compare_arrays(new, reference, real, axis, same_memory)
    if isinstance(new, Array) or isinstance(reference, Array):
        return _dual.Comparison(
            False, f"array vs value: {type(new).__name__}, {type(reference).__name__}"
        )
    return _dual.compare(new, reference)


def perturb(value: Any) -> Any:
    """
    A perturbed copy for the positive control: one value changed ('_dual.perturb'); an array
    without a copy at another address, an absent one present, an empty one in another form.
    """
    if not isinstance(value, Array):
        return _dual.perturb(value)
    if value.form == ABSENT:
        return Array(_dual.form(np.zeros(1, dtype=np.uint8)), 0)
    if value.values is not None and value.values.size:
        return dataclasses.replace(value, values=_dual.perturb(value.values))
    if value.values is None and value.address is not None:
        return dataclasses.replace(value, address=value.address + 8)
    return dataclasses.replace(value, form=f"{value.form} (perturbed)")


# ---- the probe ---------------------------------------------------------------------------------


@dataclasses.dataclass
class Tally:
    """The comparisons of one function."""

    calls: int = 0
    """py2fgen's calls of the function (seen by the recorder)."""
    compared: int = 0
    """Comparisons made (passes or calls)."""
    items: int = 0
    identical: int = 0
    differ: dict[str, int] = dataclasses.field(default_factory=dict)
    controls: int = 0
    reported: int = 0
    not_probed: dict[str, str] = dataclasses.field(default_factory=dict)
    """Items or calls that were not compared, with the reason."""


@dataclasses.dataclass(frozen=True)
class Function:
    """What the probe needs to know of one exported function."""

    exported: Any
    """The py2fgen-exported function."""
    sources: Mapping[str, _dual.Source]
    """The plugin's source table of the function (class, location, axis)."""
    per_call: bool
    """Compared at py2fgen's call (else recorded there and compared by 'compare_init')."""
    entry_point: str
    """Where the plugin builds its arguments."""


class Probe:
    """
    The records of py2fgen's calls and the plugin's arguments, their comparisons and counts.
    'real(source)' is the local number of entries of the source's location (else 'None').
    """

    def __init__(
        self,
        functions: Mapping[str, Function],
        real: Callable[[_dual.Source], int | None],
    ) -> None:
        self._functions = functions
        self._real = real
        self._records: dict[str, list[dict[str, Any]]] = {name: [] for name in functions}
        self._expected: dict[str, tuple[int, dict[str, Any]]] = {}
        self.tallies: dict[str, Tally] = {name: Tally() for name in functions}
        self.installed: list[str] = []

    # --- py2fgen's side (the recorders)
    def install(self) -> None:
        """Replace each function in its module by a recorder (once)."""
        if self.installed:
            log.warning("py2fgen probe: installed already; nothing replaced", extra=ALL_RANKS)
            return
        for name, function in self._functions.items():
            module = importlib.import_module(function.exported.__module__)
            original = getattr(module, name)
            proxy = copy.copy(original)  # py2fgen's exported function: the copy converts the same
            proxy._fun = self._recorder(name, original._fun)
            setattr(module, name, proxy)
            self.installed.append(name)
        log.info(
            f"py2fgen probe: installed; {', '.join(self.installed)} replaced by recorders before"
            " py2fgen's glue imports them",
            extra=ALL_RANKS,
        )

    def _recorder(self, name: str, function: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(function)
        def recorder(**kwargs: Any) -> None:
            try:
                self.on_call(name, kwargs)
            except Exception as error:  # the probe must not change what py2fgen does
                self.tallies[name].not_probed[f"call {self.tallies[name].calls}"] = repr(error)
                log.error(f"py2fgen probe: {name}: {error!r}", extra=ALL_RANKS)
            function(**kwargs)

        return recorder

    def on_call(self, name: str, kwargs: Mapping[str, Any]) -> None:
        """py2fgen calls 'name' with 'kwargs' (its converted arguments)."""
        tally = self.tallies[name]
        tally.calls += 1
        function = self._functions[name]
        references = self._references(function, kwargs, values=not function.per_call)
        if not function.per_call:
            self._records[name].append(references)
            copies = sum(
                v.values.nbytes
                for v in references.values()
                if isinstance(v, Array) and v.values is not None
            )
            log.info(
                f"probe {name} call {tally.calls}: recorded py2fgen's {len(references)} arguments"
                f" (host copies {copies / 2**20:.2f} MiB); compared at {function.entry_point}",
                extra=ALL_RANKS,
            )
            return
        expected = self._expected.pop(name, None)
        if expected is None:
            tally.not_probed[f"call {tally.calls}"] = (
                f"the plugin kept no arguments at {function.entry_point}"
            )
            log.warning(
                f"probe {name} call {tally.calls}: the plugin kept no arguments at"
                f" {function.entry_point}; not probed",
                extra=ALL_RANKS,
            )
            return
        call, natives = expected
        if call != tally.calls:
            log.warning(
                f"probe {name}: py2fgen's call {tally.calls} is the plugin's call {call}",
                extra=ALL_RANKS,
            )
        self._compare(name, "call", call, natives, references)

    def _references(
        self, function: Function, kwargs: Mapping[str, Any], values: bool
    ) -> dict[str, Any]:
        descriptors = function.exported.param_descriptors
        return {
            param: capture(value, values)
            if isinstance(descriptors[param], py2fgen.ArrayParamDescriptor)
            else value
            for param, value in kwargs.items()
        }

    # --- the plugin's side
    def natives(self, name: str, arguments: Mapping[str, Any], values: bool) -> dict[str, Any]:
        """The plugin's arguments of 'name' in the form the probe compares."""
        return self._references(self._functions[name], arguments, values)

    def compare_init(self, name: str, k: int, natives: Mapping[str, Any]) -> None:
        """Pass 'k': compare the plugin's arguments with py2fgen's call 'k' of 'name'."""
        records = self._records[name]
        tally = self.tallies[name]
        if len(records) < k:
            tally.not_probed[f"pass {k}"] = f"py2fgen made {len(records)} call(s)"
            log.warning(
                f"probe {name} pass {k}: py2fgen made {len(records)} call(s) of {name}; not"
                f" probed (PY2FGEN_EXTRA_CALLABLES={INSTALL} installs the recorders)",
                extra=ALL_RANKS,
            )
            return
        self._compare(name, "pass", k, natives, records[k - 1])
        records[k - 1] = {}  # the host copies are no longer needed

    def expect(self, name: str, call: int, natives: Mapping[str, Any]) -> None:
        """The plugin's arguments of call 'call' of 'name', which py2fgen's next call must get."""
        if name in self._expected:
            previous = self._expected[name][0]
            self.tallies[name].not_probed[f"call {previous}"] = "py2fgen made no call"
            log.warning(
                f"probe {name} call {previous}: py2fgen made no call before the next one; not"
                " probed",
                extra=ALL_RANKS,
            )
        self._expected[name] = (call, dict(natives))

    def after(self, name: str) -> None:
        """After the call site: py2fgen must have taken the arguments kept for it."""
        expected = self._expected.pop(name, None)
        if expected is not None:
            self.tallies[name].not_probed[f"call {expected[0]}"] = "py2fgen made no call"
            log.warning(
                f"probe {name} call {expected[0]}: py2fgen made no call at this call site; not"
                " probed",
                extra=ALL_RANKS,
            )

    def _compare(
        self,
        name: str,
        counter: str,
        k: int,
        natives: Mapping[str, Any],
        references: Mapping[str, Any],
    ) -> None:
        function = self._functions[name]
        tally = self.tallies[name]
        first = tally.compared == 0
        tally.compared += 1
        where = f"{name} {counter} {k}"
        differ, missing, unreported = [], [], []
        compared = 0
        for param in function.exported.param_descriptors:
            if param not in natives or param not in references:
                side = "the plugin" if param not in natives else "py2fgen"
                missing.append(param)
                tally.not_probed[f"{counter} {k} {param}"] = f"no value from {side}"
                log.warning(
                    f"probe {where} {param}: no value from {side}; not probed", extra=ALL_RANKS
                )
                continue
            source = function.sources[param]
            new, reference = natives[param], references[param]
            failure = None
            if (name, param) in _descrdata.COMMUNICATORS:
                try:
                    new, reference = _descrdata.communicators(new, reference)
                except Exception as error:
                    failure = f"MPI_Comm_compare failed: {error!r}"
            copied = isinstance(reference, Array) and reference.values is not None
            real = self._real(source) if copied else None  # the values' real entries
            same_memory = source.klass == "A"
            if failure is not None:
                result = _dual.Comparison(False, failure)
            else:
                result = compare(new, reference, real, source.axis, same_memory)
            # the positive control: a perturbed copy of the plugin's value must be reported; an
            # item that differs is reported already (its perturbed copy may even match)
            control = (
                compare(perturb(new), reference, real, source.axis, same_memory)
                if result.identical
                else result
            )
            compared += 1
            tally.controls += 1
            if result.identical:
                tally.identical += 1
            else:
                differ.append(param)
                tally.differ[param] = tally.differ.get(param, 0) + 1
            if not control.identical:
                tally.reported += 1
            else:
                unreported.append(param)
                log.warning(
                    f"probe {where} control {param}: not reported ({control.details})",
                    extra=ALL_RANKS,
                )
            if first or not result.identical:
                log.info(
                    f"probe {where} {param} vs py2fgen:"
                    f" {'identical' if result.identical else 'differ'} ({result.details})",
                    extra=ALL_RANKS,
                )
        tally.items += compared
        log.info(
            f"probe {where}: {compared} compared, {compared - len(differ)} identical,"
            f" {len(differ)} differ{': ' + ' '.join(differ) if differ else ''}; control"
            f" {compared - len(unreported)} of {compared} reported; not probed {len(missing)}"
            f"{': ' + ' '.join(missing) if missing else ''}",
            extra=ALL_RANKS,
        )

    def summary(self) -> None:
        """Log the counts per function and in total (when the plugin is released)."""
        for name, tally in self.tallies.items():
            pending = name in self._expected
            if pending:
                self.after(name)
            per = tally.items // tally.compared if tally.compared else 0
            log.info(
                f"probe summary {name}: py2fgen calls {tally.calls}, compared {tally.compared}"
                f" ({per} items each), items {tally.items}, identical {tally.identical}, differ"
                f" {tally.items - tally.identical}"
                + (
                    f" ({', '.join(f'{k} {v}' for k, v in tally.differ.items())})"
                    if tally.differ
                    else ""
                )
                + f"; control {tally.reported} of {tally.controls} reported; not probed"
                f" {len(tally.not_probed)}"
                + (
                    f": {'; '.join(f'{k} ({v})' for k, v in tally.not_probed.items())}"
                    if tally.not_probed
                    else ""
                ),
                extra=ALL_RANKS,
            )
        items = sum(t.items for t in self.tallies.values())
        identical = sum(t.identical for t in self.tallies.values())
        reported = sum(t.reported for t in self.tallies.values())
        controls = sum(t.controls for t in self.tallies.values())
        not_probed = sum(len(t.not_probed) for t in self.tallies.values())
        log.info(
            f"probe summary: {items} items compared, {identical} identical,"
            f" {items - identical} differ; control {reported} of {controls} reported; not probed"
            f" {not_probed}",
            extra=ALL_RANKS,
        )


def install() -> None:
    """
    PY2FGEN_EXTRA_CALLABLES entry: replace the exported functions by the probe's recorders
    before py2fgen's glue imports them, if the plugin is registered in probe mode; else do
    nothing.
    """
    from icon4py.bindings.comin import plugin  # noqa: PLC0415 [import-outside-top-level]: cyclic

    instance = plugin.instance()
    probe = None if instance is None else instance.probe
    if probe is None:
        why = (
            "the icon4py ComIn plugin is not loaded"
            if instance is None
            else f"the icon4py ComIn plugin does not run in probe mode ({ENV}=1)"
        )
        # the plugin's log handler may not exist
        print(f"icon4py-comin: py2fgen probe: not installed: {why}; nothing replaced", flush=True)
        return
    probe.install()
