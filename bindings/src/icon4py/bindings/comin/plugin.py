# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
ComIn plugin that runs icon4py's diffusion for ICON (diffusion MVP).

ICON's ComIn backend of the icon4py interface exposes the arguments of 'grid_init',
'diffusion_init' and 'diffusion_run' as ComIn variables ('_marshal.py'). The plugin
- requests them in the secondary constructor after checking them against the functions'
  'param_descriptors', or stays idle if ICON does not delegate to ComIn;
- calls 'grid_init' and 'diffusion_init' at EP_ATM_TIMELOOP_BEFORE (every pass) and writes
  the pass number into their carriers;
- calls 'diffusion_run' at EP_ATM_DIFFUSION_ENTER and writes the call count into its carrier,
  which ICON checks after the entry point.
It calls the undecorated functions ('__wrapped__') with the arguments py2fgen would build.

Environment:
- ICON4PY_COMIN_CHECK=0 turns the in-job consistency checks off (default: on): after every call
  of a writing function, each Field argument must still alias ICON's array; the first call logs
  the value sums of some fields before and after.
- ICON4PY_COMIN_TIMING=1 logs the time of each phase of every call of a writing function, and
  their medians when the plugin is released (default: off). It adds a device synchronization
  before the call, so that GPU work still pending at the entry point is timed separately. It also
  logs the state of the process at some entry points, also when the plugin is idle: native
  threads and their CPU time, child processes, the main thread's CPU affinity and context
  switches, the scheduling flags of the primary CUDA context ('_diagnostics.py').
- ICON4PY_COMIN_PROFILE=<n> profiles call <n> of 'diffusion_run' with cProfile and logs the most
  expensive functions (default: off).
- ICON4PY_COMIN_TEST_SKIP_ACK=1 skips the acknowledgement of 'diffusion_run' (negative test).

Log records go to the logger of this package. When ICON loads the plugin ('register'), they are
printed on standard output: on every rank for per-rank lines, else on rank 0 only.
"""

import cProfile
import dataclasses
import functools
import hashlib
import importlib
import io
import logging
import operator
import os
import pathlib
import pstats
import statistics
import sys
import threading
import time
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, Final, TextIO

import numpy as np

import icon4py.bindings
from icon4py.bindings import config as wrapper_config, diffusion_wrapper, grid_wrapper
from icon4py.bindings.comin import _diagnostics, _marshal, _views


try:
    import cupy as cp  # type: ignore[import-not-found, import-untyped, unused-ignore]
except ImportError:
    cp = None

log = logging.getLogger(__name__)

CHECK_ENV: Final = "ICON4PY_COMIN_CHECK"
TIMING_ENV: Final = "ICON4PY_COMIN_TIMING"
PROFILE_ENV: Final = "ICON4PY_COMIN_PROFILE"
SKIP_ACK_ENV: Final = "ICON4PY_COMIN_TEST_SKIP_ACK"

LOG_FORMAT: Final = "icon4py-comin: %(level_prefix)s%(message)s [rank %(rank)d]"
ALL_RANKS: Final[Mapping[str, Any]] = {"all_ranks": True}
"""'extra' of a log record that every rank prints; the others are printed on rank 0 only."""

SECONDARY_CONSTRUCTOR: Final = "EP_SECONDARY_CONSTRUCTOR"
DESTRUCTOR: Final = "EP_DESTRUCTOR"
DOMAIN_ENTRY_POINTS: Final = frozenset({"EP_ATM_DIFFUSION_ENTER"})
"""Entry points ICON fires for a domain; there the plugin acts on domain 1 only."""
EXPECTED_ENTRY_POINTS: Final[Mapping[str, int]] = {
    "EP_FINISH": 68,
    "EP_DESTRUCTOR": 69,
    "EP_ATM_DIFFUSION_ENTER": 70,
    "EP_ATM_DIFFUSION_LEAVE": 71,
}
"""
Entry points whose numbers 'register' logs, with their values in the ComIn build this plugin
was written for (the diffusion entry points appended after EP_DESTRUCTOR).
"""
VALUE_LOG_FIELDS: Final = ("vn", "w", "theta_v", "exner")
"""Fields whose sums the first call of a writing function logs (information only)."""
PROFILE_LINES: Final = 40
"""Number of functions ICON4PY_COMIN_PROFILE logs, per sort order."""


class RankFilter(logging.Filter):
    """
    Add the rank and a level prefix to every record; on ranks other than 0, pass warnings and
    errors and the records logged with 'extra=ALL_RANKS' only.
    """

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.level_prefix = "" if record.levelno == logging.INFO else f"{record.levelname}: "
        return (
            self.rank == 0
            or record.levelno >= logging.WARNING
            or getattr(record, "all_ranks", False)
        )


def configure_logging(rank: int, stream: TextIO | None = None) -> logging.Handler:
    """
    Print this package's log records on 'stream' (default: standard output) of every rank.

    ComIn's own print functions write on ICON's standard-output rank only, but the per-rank lines
    (calls, checks, timings) are needed from every rank. The records do not propagate further,
    so that a handler of the root logger (py2fgen's, in the same interpreter) cannot print them
    a second time.
    """
    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.addFilter(RankFilter(rank))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    package_logger = logging.getLogger(__package__)
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    return handler


@dataclasses.dataclass(frozen=True)
class FunctionEntry:
    exported: Any
    """The py2fgen-exported function."""
    entry_point: str
    """The ComIn entry point at which the plugin calls it."""
    inout: bool
    """Whether the function writes its array arguments (requested with COMIN_FLAG_WRITE)."""
    ack_key: str
    """The carrier metadata whose value the plugin writes into the carrier after the call."""


FUNCTIONS: Final[Mapping[str, FunctionEntry]] = {
    "grid_init": FunctionEntry(
        grid_wrapper.grid_init, "EP_ATM_TIMELOOP_BEFORE", inout=False, ack_key=_marshal.PASS_KEY
    ),
    "diffusion_init": FunctionEntry(
        diffusion_wrapper.diffusion_init,
        "EP_ATM_TIMELOOP_BEFORE",
        inout=False,
        ack_key=_marshal.PASS_KEY,
    ),
    "diffusion_run": FunctionEntry(
        diffusion_wrapper.diffusion_run,
        "EP_ATM_DIFFUSION_ENTER",
        inout=True,
        ack_key=_marshal.CALL_COUNT_KEY,
    ),
}
"""What the plugin calls where, in call order (per entry point). ICON's side must match."""

IDLE_MARKER: Final = (_marshal.carrier_name("diffusion_run"), _marshal.DOMAIN_ID)
"""Without this variable ICON does not delegate to ComIn, and the plugin stays idle."""


def source_sha1() -> str:
    """SHA-1 over this package's Python sources (file names and contents, sorted by name)."""
    digest = hashlib.sha1(usedforsecurity=False)
    for path in sorted(pathlib.Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class _PhaseClock:
    """Split times of one call (ICON4PY_COMIN_TIMING); does nothing when disabled."""

    def __init__(self, enabled: bool, start: float) -> None:
        self.enabled = enabled
        self._last = start
        self.phases: dict[str, float] = {}

    def __call__(self, phase: str) -> None:
        if self.enabled:
            now = time.perf_counter()
            self.phases[phase] = now - self._last
            self._last = now


def _format_ms(phases: Mapping[str, float]) -> str:
    return ", ".join(f"{phase} {1e3 * seconds:.3f}" for phase, seconds in phases.items())


class Plugin:
    """The plugin's state and callbacks; 'comin' is ComIn's Python module (or a test double)."""

    def __init__(
        self,
        comin: ModuleType,
        functions: Mapping[str, FunctionEntry] = FUNCTIONS,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        self._comin = comin
        self._functions = functions
        self._check = environ.get(CHECK_ENV, "1") != "0"
        self._timing = environ.get(TIMING_ENV, "0") == "1"
        profile = environ.get(PROFILE_ENV, "").strip()
        if profile and not profile.isdigit():
            raise ValueError(f"{PROFILE_ENV}={profile!r}: expected the number of a call.")
        self._profile_call = int(profile) if profile else None
        self._skip_ack = environ.get(SKIP_ACK_ENV, "0") == "1"
        self.rank = int(comin.parallel_get_host_mpi_rank())
        self._device_xp: ModuleType | None = None
        self._device_flag = 0
        self._bound: dict[str, _marshal.BoundFunction] = {}
        self._active = False
        self._first_call_done: set[str] = set()
        self._timings: dict[str, list[dict[str, float]]] = {}

    @property
    def active(self) -> bool:
        return self._active

    def _entry_point(self, name: str) -> Any:
        entry_point = getattr(self._comin, name, None)
        if entry_point is None:
            raise RuntimeError(
                f"ComIn has no entry point '{name}': ICON must be built with a ComIn that"
                " provides the diffusion entry points EP_ATM_DIFFUSION_ENTER/_LEAVE."
            )
        return entry_point

    def register(self) -> None:
        """Primary constructor: select host or device, log the setup, register the callbacks."""
        comin = self._comin
        if comin.descrdata_get_global().has_device:
            if cp is None:
                raise RuntimeError("ICON runs on the GPU, but CuPy cannot be imported.")
            self._device_xp = cp
            self._device_flag = comin.COMIN_FLAG_DEVICE
        numbers = {name: operator.index(self._entry_point(name)) for name in EXPECTED_ENTRY_POINTS}
        log.info(
            f"source sha1 {source_sha1()}, icon4py.bindings {icon4py.bindings.__version__}"
            f" at {pathlib.Path(icon4py.bindings.__file__).parent}, device {self._device_xp is not None}"
        )
        log.info("entry points " + ", ".join(f"{k}={v}" for k, v in numbers.items()))
        if numbers != EXPECTED_ENTRY_POINTS:
            log.warning(f"entry point numbers differ from {dict(EXPECTED_ENTRY_POINTS)}.")
        if self._skip_ack:
            log.warning(f"{SKIP_ACK_ENV}=1, 'diffusion_run' is not acknowledged (negative test).")
        if not self._check:
            log.info(f"{CHECK_ENV}=0: in-job consistency checks are off.")
        if self._timing:
            log.info(f"{TIMING_ENV}=1: per-call phase timing is on.")
            self._log_state("primary constructor")
        if self._profile_call is not None:
            log.info(f"{PROFILE_ENV}={self._profile_call}: that call is profiled.")

        callbacks: dict[str, Callable[[], None]] = {
            SECONDARY_CONSTRUCTOR: self.secondary_constructor
        }
        for entry in self._functions.values():
            callbacks[entry.entry_point] = functools.partial(self.on_entry_point, entry.entry_point)
        callbacks[DESTRUCTOR] = self.destructor
        for name, callback in callbacks.items():
            self._entry_point(name)(callback)  # 'comin.entry_point' registers as a decorator

    def secondary_constructor(self) -> None:
        """Request and check all argument variables, or stay idle."""
        comin = self._comin
        exposed = set(comin.var_list())
        if IDLE_MARKER not in exposed:
            log.info(f"idle: ICON exposes no '{IDLE_MARKER[0]}', it does not delegate to ComIn.")
            return
        errors: list[str] = []
        read, write = comin.COMIN_FLAG_READ, comin.COMIN_FLAG_WRITE
        carrier_flags = read | write | self._device_flag
        for name, entry in self._functions.items():
            flags = read | (write if entry.inout else 0) | self._device_flag
            self._bound[name] = _marshal.bind(
                comin,
                _marshal.signature(name, entry.exported),
                context=[self._entry_point(entry.entry_point)],
                flags=flags,
                carrier_flags=carrier_flags,
                exposed=exposed,
                errors=errors,
            )
        if errors:
            raise RuntimeError(
                f"icon4py ComIn plugin: {len(errors)} problem(s) with ICON's icon4py variables:\n  "
                + "\n  ".join(errors)
            )
        self._active = True
        count = sum(len(bound.arrays) + 1 for bound in self._bound.values())
        log.info(f"active: requested {count} variables for {', '.join(self._functions)}.")

    def _domain_id(self) -> int | None:
        try:
            return int(self._comin.current_get_domain_id())
        except Exception:
            return None  # the adapter raises for the negative ids of calls outside the domain loop

    def on_entry_point(self, entry_point: str) -> None:
        """
        Call the functions of 'entry_point'.

        A mis-fire (outside the domain loop, or for another domain) only logs a warning and
        touches nothing: the adapter turns an exception into ComIn's finish, which stops ICON.
        """
        start = time.perf_counter()
        if not self._active:
            if entry_point in DOMAIN_ENTRY_POINTS:
                log.warning(
                    f"{entry_point} fired, but the plugin is idle; ignored.", extra=ALL_RANKS
                )
            elif self._timing:
                self._log_state(f"{entry_point}, idle")
            return
        if entry_point in DOMAIN_ENTRY_POINTS:
            domain_id = self._domain_id()
            if domain_id != _marshal.DOMAIN_ID:
                where = (
                    "outside the domain loop" if domain_id is None else f"for domain {domain_id}"
                )
                log.warning(f"{entry_point} fired {where}; ignored.", extra=ALL_RANKS)
                return
        # the adapter holds the GIL between callbacks: GT4Py must not compile in the background
        wrapper_config.WAIT_FOR_COMPILATION = True
        log_state = self._timing and any(
            e.entry_point == entry_point and not e.inout for e in self._functions.values()
        )
        if log_state:
            self._log_state(f"{entry_point}, before")
        for name, entry in self._functions.items():
            if entry.entry_point == entry_point:
                self._call(name, entry, start)
        if log_state:
            self._log_state(f"{entry_point}, after")

    def _call(self, name: str, entry: FunctionEntry, start: float) -> None:
        clock = _PhaseClock(self._timing and entry.inout, start)
        clock("guard")
        bound = self._bound[name]
        ack_value = self._ack_value(bound, entry)
        clock("ack")
        kwargs = _marshal.arguments(self._comin, bound, self._device_xp)
        clock("args")
        first_call = name not in self._first_call_done
        if first_call:
            self._log_optional_arguments(name, bound, kwargs)
        log_values = entry.inout and self._check and first_call
        sums_before = self._value_sums(bound) if log_values else {}
        clock("sums")
        if clock.enabled:
            self._synchronize()
            clock("presync")

        profiled = entry.ack_key == _marshal.CALL_COUNT_KEY and ack_value == self._profile_call
        if profiled:
            self._profile(name, ack_value, bound.signature.function, kwargs)
        else:
            bound.signature.function(**kwargs)
        clock("run")
        self._synchronize()
        clock("sync")

        if entry.inout and self._check:
            self._check_pointer_identity(name, bound, kwargs, first_call)
        if log_values:
            self._log_value_sums(name, sums_before, self._value_sums(bound))
        self._first_call_done.add(name)
        clock("check")

        acknowledge = True
        if entry.ack_key == _marshal.CALL_COUNT_KEY:
            log.info(f"{name} call {ack_value} done.", extra=ALL_RANKS)
            if self._skip_ack:
                log.warning(
                    f"{name} call {ack_value} not acknowledged ({SKIP_ACK_ENV}).", extra=ALL_RANKS
                )
                acknowledge = False
        else:
            log.info(f"{name} done ({entry.ack_key} {ack_value}).")
        if acknowledge:
            _marshal.write_carrier(bound, ack_value)
        clock("done")

        if clock.enabled:
            self._log_timing(name, ack_value, clock.phases, profiled)

    def _ack_value(self, bound: _marshal.BoundFunction, entry: FunctionEntry) -> int:
        metadata = _marshal.carrier_metadata(self._comin, bound)
        value = metadata.get(entry.ack_key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(
                f"icon4py ComIn plugin: '{bound.signature.carrier}' has '{entry.ack_key}' = {value!r},"
                " expected an integer: ICON did not bind the arguments."
            )
        return value

    def _synchronize(self) -> None:
        # ICON's '!$ACC WAIT' does not wait for CuPy/DaCe work outside OpenACC's queues
        if self._device_xp is not None:
            self._device_xp.cuda.runtime.deviceSynchronize()

    def _profile(
        self, name: str, call: int, function: Callable[..., None], kwargs: Mapping[str, Any]
    ) -> None:
        profiler = cProfile.Profile()
        profiler.runcall(function, **kwargs)
        for sort in ("cumulative", "tottime"):
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).sort_stats(sort).print_stats(PROFILE_LINES)
            log.info(f"profile of {name} call {call}, sorted by {sort}:", extra=ALL_RANKS)
            for line in text.getvalue().splitlines():
                if line.strip():
                    log.info(f"  {line}", extra=ALL_RANKS)

    def _log_timing(
        self, name: str, call: int, phases: Mapping[str, float], profiled: bool
    ) -> None:
        if name not in self._timings:
            threads = ", ".join(
                f"{t.name}{' (daemon)' if t.daemon else ''}" for t in threading.enumerate()
            )
            log.info(
                f"timing: Python threads {threads}; switch interval"
                f" {1e3 * sys.getswitchinterval():.1f} ms",
                extra=ALL_RANKS,
            )
            self._timings[name] = []
        if not profiled:
            self._timings[name].append({"call": call, **phases})
        total = sum(phases.values())
        log.info(
            f"timing {name} call {call} [ms]: {_format_ms(phases)}, total {1e3 * total:.3f}"
            f"{' (profiled)' if profiled else ''}",
            extra=ALL_RANKS,
        )
        if call <= 2 or call % 3 == 0:
            self._log_state(f"after {name} call {call}")

    def _log_state(self, where: str) -> None:
        log.info(f"state at {where}: {_diagnostics.process_state()}", extra=ALL_RANKS)

    def _log_timing_summary(self) -> None:
        for name, calls in self._timings.items():
            steady = [c for c in calls if c["call"] > 1]
            if not steady:
                continue
            phases = [p for p in steady[0] if p != "call"]
            medians = {p: statistics.median(c[p] for c in steady) for p in phases}
            total = statistics.median(sum(c[p] for p in phases) for c in steady)
            log.info(
                f"timing {name}, median of {len(steady)} calls after the first [ms]:"
                f" {_format_ms(medians)}, total {1e3 * total:.3f}",
                extra=ALL_RANKS,
            )

    def _log_optional_arguments(
        self, name: str, bound: _marshal.BoundFunction, kwargs: Mapping[str, Any]
    ) -> None:
        """Log what the first call gets for each optional array argument (e.g. empty 'zd_*' lists)."""
        parts = []
        for array in bound.arrays:
            if not array.param.descriptor.is_optional:
                continue
            value = kwargs[array.param.name]
            if not array.present:
                parts.append(f"{array.param.name} None (absent)")
            elif value is None:
                parts.append(f"{array.param.name} None (NULL address, ICON extent {array.shape})")
            else:
                parts.append(f"{array.param.name} {tuple(value.shape)}")
        if parts:
            log.info(f"{name} optional arguments: {', '.join(parts)}", extra=ALL_RANKS)

    def _check_pointer_identity(
        self,
        name: str,
        bound: _marshal.BoundFunction,
        kwargs: Mapping[str, Any],
        first_call: bool,
    ) -> None:
        """
        Consistency guard: every Field argument still aliases the memory ComIn hands out for its
        variable after the call (a copying Field helper, or a Field whose buffer GT4Py replaced,
        would fail here). It does not show that the function wrote through the Fields; the
        comparisons of ICON's results with the py2fgen path do.
        """
        count = 0
        for array in bound.arrays:
            value = kwargs[array.param.name]
            if array.param.dims is None or value is None or 0 in array.shape:
                continue
            field_ptr = _views.data_ptr(value.ndarray)
            comin_ptr = _marshal.variable_data_ptr(array, self._device_xp)
            if field_ptr != comin_ptr:
                raise RuntimeError(
                    f"icon4py ComIn plugin: '{array.param.name}' of '{name}' does not alias ICON's"
                    f" array (Field {field_ptr:#x}, ComIn {comin_ptr:#x}); the results would be lost."
                )
            count += 1
        if first_call:
            log.info(f"pointer identity OK ({count} fields)", extra=ALL_RANKS)

    def _value_sums(self, bound: _marshal.BoundFunction) -> dict[str, float]:
        """Sums of fresh views of some fields; information only, so it never raises."""
        sums: dict[str, float] = {}
        try:
            for array in bound.arrays:
                if array.param.name in VALUE_LOG_FIELDS and array.present and 0 not in array.shape:
                    on_device = _marshal.is_on_device(array, self._device_xp)
                    xp = self._device_xp if (on_device and self._device_xp is not None) else np
                    sums[array.param.name] = float(xp.asarray(array.variable).sum())
        except Exception as error:
            log.warning(f"value sums failed: {error!r}", extra=ALL_RANKS)
        return sums

    def _log_value_sums(
        self, name: str, before: Mapping[str, float], after: Mapping[str, float]
    ) -> None:
        parts = []
        for field, value in before.items():
            if field in after:
                state = "changed" if after[field] != value else "unchanged"
                parts.append(f"{field} {value!r} -> {after[field]!r} ({state})")
        log.info(f"{name} first call, sums before -> after: {'; '.join(parts)}", extra=ALL_RANKS)

    def destructor(self) -> None:
        """Release the granule and CuPy's cached memory."""
        if self._timing:
            self._log_state(f"destructor{'' if self._active else ', idle'}")
        if not self._active:
            return
        self._log_timing_summary()
        diffusion_wrapper.granule = None
        grid_wrapper.grid_state = None
        self._bound.clear()
        self._active = False
        if self._device_xp is not None:
            self._device_xp.get_default_memory_pool().free_all_blocks()
            self._device_xp.get_default_pinned_memory_pool().free_all_blocks()
        log.info("released the granule.")


_instance: Plugin | None = None
"""The registered plugin; a second one is refused, so that the diffusion cannot run twice."""


def register() -> Plugin:
    """Create and register the plugin in ComIn's primary constructor ('plugin_main.py')."""
    global _instance  # noqa: PLW0603 [global-statement]
    if _instance is not None:
        raise RuntimeError(
            "The icon4py ComIn plugin is loaded twice; keep one '&comin_plugin_nml' group for it."
        )
    # 'comin' exists only inside ICON, as the module of ComIn's Python adapter
    plugin = Plugin(importlib.import_module("comin"))
    configure_logging(plugin.rank)
    plugin.register()
    _instance = plugin
    return plugin
