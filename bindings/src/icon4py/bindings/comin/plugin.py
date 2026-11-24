# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
ComIn plugin that runs icon4py's diffusion for ICON.

ICON's icon4py switches decide what happens ('run_nml': icon4py_interface,
luse_icon4py_diffusion, icon4py_mode). With icon4py_interface=1 ICON calls no icon4py function:
it fires ComIn's dycore entry points around each diffusion call
(EP_ATM_DYCORE_DIFFUSION_BEFORE/_AFTER), runs its own diffusion in VERIFY and OFF, and skips it
in SUBSTITUTE. ICON trusts that a plugin computes it: nothing in ICON checks. The plugin
- reads ICON's namelist output in its primary constructor ('_config.py'): ICON's icon4py mode,
  as ICON's dispatcher decides it, and the configuration: the values of 'grid_init', the backend,
  icon4py's DiffusionConfig ('_granule.diffusion_config') and the two values that icon4py's
  'Diffusion' takes besides it; it logs one line on what ICON and the plugin will do, and
  computes iff ICON delegates the diffusion to ComIn
  (icon4py_interface=1, mode SUBSTITUTE or VERIFY);
- requests ICON's variables in the secondary constructor, or stays idle; there it also gives
  the DiffusionConfig ICON's value of 'loutshs' at run time, which ICON's NWP physics set to
  whether the output lists 'ddt_tke_hsh': whether ICON has that variable
  ('_config.runtime_loutshs');
- builds icon4py's grid and diffusion granule through icon4py's API ('_granule.py': 'grid_init',
  'diffusion_init') at EP_ATM_TIMELOOP_BEFORE (every pass); the grid, geometry and
  interpolation inputs come from ComIn's descriptive data of domain 1 ('_descrdata.py': the
  plugin's own copies, a fresh one per pass); the metrics that 'diffusion_init' gets are ICON's
  own variables (STATIC_VARIABLES: zero-copy views, on the device on the GPU, READ and DEVICE
  access), fetched in every pass at the addresses of the first one (ICON allocates them once);
  the granule keeps them, so the plugin also fetches them in every callback in which the
  granule computes and stops if one is not at the address of the first pass (ComIn hands out
  an ICON variable per callback);
- in SUBSTITUTE, calls 'diffusion_run' at EP_ATM_DYCORE_DIFFUSION_BEFORE of domain 1 on ICON's
  own variables of the same names (zero-copy views, on the device on the GPU; READ, WRITE and
  DEVICE access, so ComIn copies nothing), with the time step from ComIn's descriptive data and
  the initial-call flag from the order of the entry points (below);
- in VERIFY ('_verify.py'), copies ICON's variables of the same names at
  EP_ATM_DYCORE_DIFFUSION_BEFORE of domain 1, before ICON computes, into its own buffers (on the
  device on the GPU; READ and DEVICE access: the plugin never writes ICON's variables in
  VERIFY). At EP_ATM_DYCORE_DIFFUSION_AFTER, where ICON's variables hold ICON's own results, it
  calls 'diffusion_run' on its copies with the time step and the initial-call flag it derived,
  compares its results with ICON's and prints the comparison in the format of ICON's tables,
  after a line that starts with VERIFY_HEADER. These tables are the only comparison: ICON
  continues with its own result and compares nothing.
The granule's inputs are zero-copy views of ICON's arrays in the forms py2fgen builds from them
('_arguments.py'), so the granule computes what it computes through py2fgen.

The guards at EP_ATM_DYCORE_DIFFUSION_BEFORE: the plugin acts on domain 1 only, and only if
'lhdiff_vn': ICON fires the pair of the regular call site also when the diffusion is off.

The initial-call flag ('linit'): in a real-data run ICON calls the diffusion once more before
the first dynamics step of a domain, i.e. before any EP_ATM_DYCORE_SOLVE_NH_BEFORE of that step;
the regular call follows the dynamics. So an EP_ATM_DYCORE_DIFFUSION_BEFORE of a domain is the
initial call iff no EP_ATM_DYCORE_SOLVE_NH_BEFORE of that domain has fired since its last
EP_ATM_INTEGRATE_START. The plugin checks this against ICON's condition for the initial call
from the namelist output and the descriptive data ('_config.InitialCall': ldynamics, not
ltestcase, lhdiff_vn, init_mode not IAU, no restart, the first step, the first call of the
step), logs both per call and stops if they disagree.

It also registers callbacks at ComIn's dycore entry points at ICON's call sites
(EP_ATM_DYCORE_DIFFUSION_BEFORE/_AFTER, EP_ATM_DYCORE_SOLVE_NH_BEFORE/_AFTER; ICON fires them
whether or not icon4py runs), counts them and logs the counts when it is released; ComIn must
provide them, else 'register' stops.

Every input of the granule's functions ('_granule.PARAMS') has a native route, checked when the
plugin is created ('check_routes'): ICON's namelist output (the configuration,
'_config.ARGUMENTS' and CONFIGURATIONS); ComIn's descriptive data ('_descrdata.ROUTES', among
them 'inv_dual_edge_length' as '1 / dual_edge_length' with ICON's 0 on the first row of the
lateral boundary, and the owner masks of cells, edges and vertices as 'decomp_domain == 0',
which ICON's decomposition makes them); ICON's variables (STATIC_VARIABLES, and the per-call
arrays of 'diffusion_run'); the time step of the descriptive data and the initial-call flag
from the order of the entry points. The py2fgen probe ('_probe.py', imported only in probe
mode) compares every input, and the objects built from them, with py2fgen's own in a run that
computes through py2fgen: run it for every new configuration.

Environment:
- ICON4PY_COMIN_CHECK=0 turns the in-job consistency checks off (default: on): after every call
  of a writing function, each Field argument must still alias the array it was built from; the
  first call logs the value sums of some fields before and after.
- ICON4PY_COMIN_TIMING=1 logs the time of each phase of every call of a writing function, and
  their medians when the plugin is released (default: off). It adds a device synchronization
  before the call, so that GPU work still pending at the entry point is timed separately. It also
  logs the state of the process at some entry points, also when the plugin is idle: native
  threads and their CPU time, child processes, the main thread's CPU affinity and context
  switches, the scheduling flags of the primary CUDA context ('_diagnostics.py').
- ICON4PY_COMIN_PROFILE=<n> profiles call <n> of 'diffusion_run' with cProfile and logs the most
  expensive functions (default: off).
- ICON4PY_COMIN_PROBE=1 (the py2fgen probe, '_probe.py'): ICON computes the diffusion through
  py2fgen, and the plugin compares every input it would hand the granule, and the grid and
  granule it builds from them, with py2fgen's (with
  PY2FGEN_EXTRA_CALLABLES=icon4py.bindings.comin._probe:install); it runs no diffusion and
  writes nothing.
Any other ICON4PY_COMIN_* variable (e.g. one of a check that no longer exists) is ignored, with
one warning at start-up.

ICON's namelist output is read from the working directory of ICON (its run directory).

Log records go to the logger of this package. When ICON loads the plugin ('register'), they are
printed on standard output: on every rank for per-rank lines, else on rank 0 only.
"""

import cProfile
import dataclasses
import enum
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
from icon4py.bindings.comin import (
    _arguments,
    _config,
    _descrdata,
    _diagnostics,
    _granule,
    _verify,
    _views,
)


try:
    import cupy as cp  # type: ignore[import-not-found, import-untyped, unused-ignore]
except ImportError:
    cp = None

log = logging.getLogger(__name__)

CHECK_ENV: Final = "ICON4PY_COMIN_CHECK"
TIMING_ENV: Final = "ICON4PY_COMIN_TIMING"
PROFILE_ENV: Final = "ICON4PY_COMIN_PROFILE"
PROBE_ENV: Final = "ICON4PY_COMIN_PROBE"
PROBE_MODULE: Final = f"{__package__}._probe"
"""The py2fgen probe, imported only with ICON4PY_COMIN_PROBE=1."""
ENV_PREFIX: Final = "ICON4PY_COMIN_"
KNOWN_ENV: Final = frozenset({CHECK_ENV, TIMING_ENV, PROFILE_ENV, PROBE_ENV})
"""The ICON4PY_COMIN_* variables this plugin reads; it warns once about any other."""

LOG_FORMAT: Final = "icon4py-comin: %(level_prefix)s%(message)s [rank %(rank)d]"
ALL_RANKS: Final[Mapping[str, Any]] = {"all_ranks": True}
"""'extra' of a log record that every rank prints; the others are printed on rank 0 only."""

SECONDARY_CONSTRUCTOR: Final = "EP_SECONDARY_CONSTRUCTOR"
DESTRUCTOR: Final = "EP_DESTRUCTOR"
TIMELOOP_BEFORE: Final = "EP_ATM_TIMELOOP_BEFORE"
"""Fired before the time loop of every pass: the plugin calls the functions that do not write."""
DIFFUSION_BEFORE: Final = "EP_ATM_DYCORE_DIFFUSION_BEFORE"
DIFFUSION_AFTER: Final = "EP_ATM_DYCORE_DIFFUSION_AFTER"
SOLVE_NH_BEFORE: Final = "EP_ATM_DYCORE_SOLVE_NH_BEFORE"
DYCORE_ENTRY_POINTS: Final = (
    DIFFUSION_BEFORE,
    DIFFUSION_AFTER,
    SOLVE_NH_BEFORE,
    "EP_ATM_DYCORE_SOLVE_NH_AFTER",
)
"""ComIn's dycore entry points at ICON's call sites; the plugin counts their callbacks."""
INTEGRATE_START: Final = "EP_ATM_INTEGRATE_START"
"""Fired at the start of every time step of a domain (with SOLVE_NH_BEFORE: the initial call)."""
PER_CALL_SCALARS: Final = ("dtime", "linit")
"""
The scalar arguments of a writing function that the plugin derives per call at
DIFFUSION_BEFORE: the time step (ComIn's descriptive data) and the initial-call flag (the order
of the entry points). With the function's arrays (ICON's variables of the same names) they are
its per-call arguments.
"""
VERIFY_HEADER: Final = "plugin verifying"
"""The start of the line before the plugin's comparison tables in VERIFY."""
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
    params: Mapping[str, _arguments.Param]
    """The function's inputs ('_granule.PARAMS'); the plugin calls the method of the same name of
    its granule ('_granule.Granule')."""
    entry_point: str
    """
    The ComIn entry point at which the plugin builds the function's arguments: a function that
    does not write is called there; a writing function is called there in SUBSTITUTE (on ICON's
    variables), and in VERIFY at DIFFUSION_AFTER on the plugin's copies of ICON's variables,
    taken there.
    """
    inout: bool
    """Whether the function writes its array arguments (ICON's variables in SUBSTITUTE)."""


FUNCTIONS: Final[Mapping[str, FunctionEntry]] = {
    "grid_init": FunctionEntry(_granule.GRID_INIT, TIMELOOP_BEFORE, inout=False),
    "diffusion_init": FunctionEntry(_granule.DIFFUSION_INIT, TIMELOOP_BEFORE, inout=False),
    "diffusion_run": FunctionEntry(_granule.DIFFUSION_RUN, DIFFUSION_BEFORE, inout=True),
}
"""What the plugin calls where, in call order (per entry point)."""

STATIC_VARIABLES: Final[Mapping[str, Mapping[str, str]]] = {
    "diffusion_init": {
        "theta_ref_mc": "theta_ref_mc",
        "wgtfac_c": "wgtfac_c",
        "zd_cellidx": "zd_indlist",
        "zd_vertidx": "zd_vertidx",
        "zd_intcoef": "zd_intcoef",
        "zd_diffcoef": "zd_diffcoef",
    },
}
"""
ICON's variables that are array arguments of a function that does not write (argument: ICON's
name). ICON allocates them once, before ComIn's secondary constructor; the 'zd_*' lists of the
truly horizontal temperature diffusion exist iff 'l_zdiffu_t' (an absent one is an absent
optional argument). The plugin fetches them in every call of the function, at its entry point,
and in every callback in which the granule computes (it keeps them), and stops if one is not at
the address of the first call.
"""


CONFIGURATIONS: Final[
    Mapping[str, Mapping[str, Callable[[_config.Namelists], tuple[Any, list[str]]]]]
] = {
    "diffusion_init": {"config": _granule.diffusion_config},
}
"""The configuration objects of the functions that icon4py builds from ICON's namelist output."""

LOCAL_COUNT: Final[Mapping[str, tuple[str, str]]] = {
    "cell": ("cells", "ncells"),
    "edge": ("edges", "nedges"),
    "vertex": ("verts", "nverts"),
}
"""Descriptive data (domain) of each location and its number of local entries."""
ROUTES: Final = (
    "ICON's namelist output",
    "ComIn's descriptive data",
    "ICON's variables",
    f"per call at {DIFFUSION_BEFORE}",
)
"""The native routes of the inputs, in the order of their log line."""


def configuration_route(name: str, param: str, kind: _arguments.Param) -> bool:
    """The input is a scalar with an entry in ICON's namelist output ('_config.ARGUMENTS') or a
    configuration object built from it (CONFIGURATIONS)."""
    return not isinstance(kind, _arguments.ArrayParam) and (
        param in _config.ARGUMENTS.get(name, {}) or param in CONFIGURATIONS.get(name, {})
    )


def static_route(name: str, entry: FunctionEntry, param: str) -> bool:
    """The input is an ICON variable of STATIC_VARIABLES (of a function that does not write)."""
    return not entry.inout and param in STATIC_VARIABLES.get(name, {})


def per_call_route(entry: FunctionEntry, param: str, kind: _arguments.Param) -> bool:
    """The input is a per-call input of a writing function (an array, or PER_CALL_SCALARS)."""
    return entry.inout and (isinstance(kind, _arguments.ArrayParam) or param in PER_CALL_SCALARS)


def route(name: str, entry: FunctionEntry, param: str) -> str | None:
    """The native route of an input (one of ROUTES), or 'None' if it has none."""
    kind = entry.params[param]
    if per_call_route(entry, param, kind):
        return ROUTES[3]
    if static_route(name, entry, param):
        return ROUTES[2]
    if _descrdata.has_route(name, param):
        return ROUTES[1]
    if configuration_route(name, param, kind):
        return ROUTES[0]
    return None


def check_routes(functions: Mapping[str, FunctionEntry]) -> list[str]:
    """Every input of every function has a native route."""
    return [
        f"{name}: '{param}' has no native route (the configuration from ICON's namelist output,"
        " ComIn's descriptive data, ICON's variables of STATIC_VARIABLES, and the per-call inputs"
        " of a writing function)."
        for name, entry in functions.items()
        for param in entry.params
        if route(name, entry, param) is None
    ]


def same_memory(name: str, entry: FunctionEntry) -> frozenset[str]:
    """The inputs that are ICON's own memory: ICON's variables, static or per call."""
    return frozenset(
        param
        for param, kind in entry.params.items()
        if static_route(name, entry, param)
        or (entry.inout and isinstance(kind, _arguments.ArrayParam))
    )


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


def _logical(value: bool) -> str:
    return "T" if value else "F"


def _describe(value: Any) -> str:
    """A configuration value for the log: a dataclass by its fields, an enumeration by its value."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = ", ".join(
            f"{f.name}={_describe(getattr(value, f.name))}" for f in dataclasses.fields(value)
        )
        return f"{type(value).__name__}({fields})"
    if isinstance(value, enum.Enum):
        return repr(value.value)
    return repr(value)


def _format_ms(phases: Mapping[str, float]) -> str:
    return ", ".join(f"{phase} {1e3 * seconds:.3f}" for phase, seconds in phases.items())


@dataclasses.dataclass
class EntryPointOrder:
    """
    The order of ICON's entry points per domain, from which the plugin derives the initial-call
    flag: a diffusion call is the initial one iff no SOLVE_NH_BEFORE of its domain has fired
    since the domain's last INTEGRATE_START.
    """

    steps: dict[int, int] = dataclasses.field(default_factory=dict)
    """INTEGRATE_START callbacks per domain: the number of the current time step."""
    solves: dict[int, int] = dataclasses.field(default_factory=dict)
    """SOLVE_NH_BEFORE callbacks per domain since its last INTEGRATE_START."""
    calls: dict[int, int] = dataclasses.field(default_factory=dict)
    """Diffusion calls per domain since its last INTEGRATE_START."""

    def integrate_start(self, jg: int) -> None:
        self.steps[jg] = self.steps.get(jg, 0) + 1
        self.solves[jg] = 0
        self.calls[jg] = 0

    def solve_nh(self, jg: int) -> None:
        self.solves[jg] = self.solves.get(jg, 0) + 1

    def diffusion_call(self, jg: int) -> tuple[int, int, int]:
        """Count a diffusion call of 'jg'; return (step, SOLVE_NH_BEFOREs since, call of the step)."""
        self.calls[jg] = self.calls.get(jg, 0) + 1
        return self.steps.get(jg, 0), self.solves.get(jg, 0), self.calls[jg]


@dataclasses.dataclass
class CallState:
    """What the plugin derived, and used, at one DIFFUSION_BEFORE of domain 1."""

    call: int
    """The number of the diffusion call (DIFFUSION_BEFORE of domain 1 with 'lhdiff_vn')."""
    linit: bool
    dtime: float
    icon_pointers: dict[str, tuple[int, int | None]] = dataclasses.field(default_factory=dict)
    """VERIFY: host and device address of ICON's variables at DIFFUSION_BEFORE."""
    routes: dict[str, Any] = dataclasses.field(default_factory=dict)
    """The per-call arguments (in VERIFY the plugin's copies as the granule gets them; the
    scalars)."""
    copies: dict[str, _arguments.BoundFunction] = dataclasses.field(default_factory=dict)
    """VERIFY: per writing function, the plugin's copies of ICON's input as bound arrays."""


@dataclasses.dataclass
class DiffusionCalls:
    """The diffusion calls of domain 1 while the plugin is active."""

    count: int = 0
    """DIFFUSION_BEFORE callbacks of domain 1 with 'lhdiff_vn': ICON's diffusion calls."""
    computed: int = 0
    """SUBSTITUTE: of those, computed at DIFFUSION_BEFORE."""
    copied: int = 0
    """VERIFY: of those, ICON's input copied at DIFFUSION_BEFORE."""
    compared: int = 0
    """VERIFY: calls computed on the copies and compared with ICON's results at DIFFUSION_AFTER."""
    not_close: dict[str, int] = dataclasses.field(default_factory=dict)
    """VERIFY: the rows 'No' of the plugin's tables, per table."""
    pending: CallState | None = None
    """VERIFY: the state of the last call copied at DIFFUSION_BEFORE, until DIFFUSION_AFTER."""
    skipped: int = 0
    """DIFFUSION_BEFORE callbacks of domain 1 without 'lhdiff_vn' (no diffusion call)."""
    linit: list[tuple[int, bool, bool]] = dataclasses.field(default_factory=list)
    """Per call: its number, the derived 'linit', whether ICON's condition agrees."""


@dataclasses.dataclass
class StaticVariables:
    """ICON's variables of STATIC_VARIABLES that the plugin requested."""

    handles: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    """Per function and argument: the 'comin.var_get' handle ('None': ICON does not expose it)."""
    first: dict[str, tuple[int, dict[str, _arguments.Pointer]]] = dataclasses.field(
        default_factory=dict
    )
    """Per function: the pass of the first fetch, and the pointers then."""
    checks: dict[str, int] = dataclasses.field(default_factory=dict)
    """Per function: the callbacks in which the granule computed and they were checked."""


@dataclasses.dataclass
class Verification:
    """VERIFY: the plugin's copies of ICON's input, and the comparison with ICON's results."""

    functions: tuple[str, ...] = ()
    """The writing functions computed on the plugin's copies."""
    copies: _verify.Copies = dataclasses.field(default_factory=_verify.Copies)
    """The plugin's buffers for ICON's input (kept for the whole run)."""
    allreduce: _verify.Allreduce = _verify.local_allreduce
    """The reductions of the comparison tables over ICON's work PEs ('register')."""
    real_entries: dict[str, int] = dataclasses.field(default_factory=dict)
    """Per location, the real entries of the first block (ICON's 'domain' table)."""


class Plugin:
    """The plugin's state and callbacks; 'comin' is ComIn's Python module (or a test double)."""

    def __init__(
        self,
        comin: ModuleType,
        functions: Mapping[str, FunctionEntry] = FUNCTIONS,
        environ: Mapping[str, str] = os.environ,
        granule: Any = None,
        *,
        run_dir: str | os.PathLike[str] | None = None,
        host_comm: Any = None,
    ) -> None:
        """
        'granule': the object whose methods of the functions' names the plugin calls, and its
        'release' (default: a new '_granule.Granule'); 'run_dir': where ICON wrote its namelist
        output (default: the working directory); 'host_comm': ComIn's host communicator as an
        mpi4py communicator (default: from 'comin', in 'register').
        """
        errors = check_routes(functions)
        if errors:
            raise ValueError(
                "icon4py ComIn plugin: inputs without a route:\n  " + "\n  ".join(errors)
            )
        self._comin = comin
        self._functions = functions
        self._granule = _granule.Granule() if granule is None else granule
        self._run_dir = pathlib.Path.cwd() if run_dir is None else pathlib.Path(run_dir)
        self._host_comm = host_comm
        self._mode: _config.IconMode | None = None
        """ICON's icon4py mode, from its namelist output ('register')."""
        self._configuration: dict[str, dict[str, Any]] = {}
        """The configuration inputs per function, from ICON's namelist output (and the
        DiffusionConfig's 'loutshs' at run time from the secondary constructor on)."""
        self._hdiff_vn = True
        """ICON calls the diffusion ('lhdiff_vn' of 'diffusion_nml')."""
        self._initial: _config.InitialCall | None = None
        """ICON's condition for the initial diffusion call ('register', if the plugin computes)."""
        self._before: tuple[str, ...] = ()
        """SUBSTITUTE: the writing functions, called at DIFFUSION_BEFORE on ICON's variables."""
        self._verification = Verification()
        self._icon_handles: dict[str, Any] = {}
        """ICON's variables of the names of the writing functions' array arguments."""
        self._order = EntryPointOrder()
        self._calls = DiffusionCalls()
        self._signatures = {
            name: _arguments.Signature(name, getattr(self._granule, name), entry.params)
            for name, entry in functions.items()
        }
        """The inputs of each function and the granule's method."""
        self.probe: Any = self._make_probe(environ)
        """The py2fgen probe ('_probe.Probe', ICON4PY_COMIN_PROBE=1): py2fgen's calls and objects
        and the comparisons; else 'None'."""
        self._probing = False
        """The probe is on and ICON computes through py2fgen ('register')."""
        self._probed: tuple[str, ...] = ()
        """Probe mode: the writing functions whose per-call arguments the probe compares."""
        self._passes: dict[str, int] = {}
        """Callbacks per entry point of the functions that do not write: the pass number."""
        self._descrdata = _descrdata.DescriptiveData(comin)
        """The arguments from ComIn's descriptive data (the plugin's copies)."""
        self._counts: dict[str, int] = {}
        """The local number of cells, edges and vertices (LOCAL_COUNT), once read."""
        self._check = environ.get(CHECK_ENV, "1") != "0"
        self._timing = environ.get(TIMING_ENV, "0") == "1"
        profile = environ.get(PROFILE_ENV, "").strip()
        if profile and not profile.isdigit():
            raise ValueError(f"{PROFILE_ENV}={profile!r}: expected the number of a call.")
        self._profile_call = int(profile) if profile else None
        self._static = StaticVariables()  # ICON's variables of STATIC_VARIABLES
        self._ep_counts: dict[str, int] = dict.fromkeys(DYCORE_ENTRY_POINTS, 0)
        self._ep_domains: dict[str, set[int | None]] = {name: set() for name in DYCORE_ENTRY_POINTS}
        self._ignored_env = sorted(
            k for k in environ if k.startswith(ENV_PREFIX) and k not in KNOWN_ENV
        )
        """The ICON4PY_COMIN_* variables that this plugin does not read ('register' warns)."""
        self.rank = int(comin.parallel_get_host_mpi_rank())
        self._device_xp: ModuleType | None = None
        self._device_flag = 0
        self._active = False
        self._first_call_done: set[str] = set()
        self._timings: dict[str, list[dict[str, float]]] = {}

    def _make_probe(self, environ: Mapping[str, str]) -> Any:
        """The py2fgen probe if ICON4PY_COMIN_PROBE=1 (its module is imported only then)."""
        value = environ.get(PROBE_ENV, "0").strip() or "0"
        if value not in ("0", "1"):
            raise ValueError(f"{PROBE_ENV}={value!r}: expected 0 or 1.")
        if value == "0":
            return None
        probe = importlib.import_module(PROBE_MODULE)
        return probe.Probe(
            {
                name: probe.Function(
                    entry.params,
                    same_memory(name, entry),
                    per_call=entry.inout,
                    entry_point=entry.entry_point,
                )
                for name, entry in self._functions.items()
            },
            self._local_count,
            self._synchronize,
        )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def _copied(self) -> tuple[str, ...]:
        """VERIFY: the writing functions computed on the plugin's copies of ICON's input."""
        return self._verification.functions

    @property
    def mode(self) -> _config.IconMode:
        """ICON's icon4py mode (known after 'register')."""
        if self._mode is None:
            raise RuntimeError("icon4py ComIn plugin: ICON's mode is read in 'register'.")
        return self._mode

    def _entry_point(self, name: str) -> Any:
        entry_point = getattr(self._comin, name, None)
        if entry_point is None:
            raise RuntimeError(
                f"ComIn has no entry point '{name}': ICON must be built with a ComIn that"
                " provides the dycore entry points EP_ATM_DYCORE_DIFFUSION_BEFORE/_AFTER and"
                " EP_ATM_DYCORE_SOLVE_NH_BEFORE/_AFTER."
            )
        return entry_point

    def _compute_entry_point(self) -> str:
        """The entry point at which the plugin calls the functions that write: ICON's fields in
        SUBSTITUTE, its copies of them in VERIFY; '-' if it computes nowhere (idle, probe)."""
        if self._before:
            return DIFFUSION_BEFORE
        if self._copied:
            return DIFFUSION_AFTER
        return "-"

    def _read_configuration(self) -> None:
        """
        Read ICON's namelist output (rank 0 reads, every rank gets it): ICON's mode (logged in
        one line) and, if the plugin computes or probes, the configuration arguments.
        """
        comm = self._host_comm if self._host_comm is not None else _config.host_comm(self._comin)
        if comm is not None:
            self._verification.allreduce = _verify.mpi_allreduce(comm)
        namelists = _config.load(comm, self._run_dir / _config.NAMELIST_FILE)
        mode = _config.icon_mode(namelists)
        self._mode = mode
        if self.probe is not None:
            self._check_probe_mode(mode)
        if mode.computes:
            self._select_compute_sites(mode.mode)
        log.info(
            mode.describe(self._compute_entry_point(), DIFFUSION_BEFORE if self._copied else None)
        )
        if mode.computes or self._probing:
            self._read_arguments(namelists)

    def _read_arguments(self, namelists: _config.Namelists) -> None:
        """'lhdiff_vn', the configuration inputs and ICON's initial-call condition."""
        errors: list[str] = []
        lhdiff_vn, kind = _config.INITIAL_CALL["lhdiff_vn"]
        try:
            self._hdiff_vn = bool(_config.argument(namelists, lhdiff_vn, kind))
        except _config.NamelistError as error:
            errors.append(f"lhdiff_vn: {error}")
        for name, entry in self._functions.items():
            kinds = {
                param: kind
                for param, kind in entry.params.items()
                if param in _config.ARGUMENTS.get(name, {}) and isinstance(kind, type)
            }
            values, problems = _config.arguments(namelists, name, kinds) if kinds else ({}, [])
            errors += problems
            for param, build in CONFIGURATIONS.get(name, {}).items():
                if param in entry.params:
                    value, problems = build(namelists)
                    values[param] = value
                    errors += problems
            if values:  # in the order of the inputs
                self._configuration[name] = {p: values[p] for p in entry.params if p in values}
        initial, problems = _config.initial_call(namelists)
        errors += problems
        if initial is not None:
            lrestartrun = bool(self._comin.descrdata_get_global().lrestartrun)
            self._initial = dataclasses.replace(initial, lrestartrun=lrestartrun)
        if errors:
            raise RuntimeError(
                f"icon4py ComIn plugin: {len(errors)} problem(s) with ICON's namelist output"
                f" {str(self._run_dir / _config.NAMELIST_FILE)!r}:\n  " + "\n  ".join(errors)
            )
        self._log_configuration(namelists)
        if self._initial is not None:
            log.info(
                f"initial diffusion call: {self._initial.describe()}; 'linit' from"
                f" the order of {INTEGRATE_START}, {SOLVE_NH_BEFORE} and {DIFFUSION_BEFORE},"
                " checked against ICON's condition at every call"
            )

    def _check_probe_mode(self, mode: _config.IconMode) -> None:
        """The probe compares py2fgen's arguments: ICON must compute through py2fgen in
        SUBSTITUTE, where the per-call arguments are ICON's own variables."""
        if mode.icon4py_interface != _config.INTERFACE_PY2FGEN or mode.mode != _config.SUBSTITUTE:
            raise RuntimeError(
                f"icon4py ComIn plugin: {PROBE_ENV}=1 compares the arguments of py2fgen's calls,"
                f" so ICON must compute the diffusion through py2fgen in SUBSTITUTE"
                f" (icon4py_interface={_config.INTERFACE_PY2FGEN}, icon4py_mode={_config.SUBSTITUTE},"
                f" luse_icon4py_diffusion=T); ICON's switches are {mode.switches}."
            )
        self._probing = True
        self._probed = tuple(name for name, entry in self._functions.items() if entry.inout)

    def _select_compute_sites(self, mode: int) -> None:
        """The writing functions: in SUBSTITUTE the plugin calls them at DIFFUSION_BEFORE, in
        VERIFY at DIFFUSION_AFTER on its copies of ICON's input taken at DIFFUSION_BEFORE."""
        writing = tuple(name for name, entry in self._functions.items() if entry.inout)
        if mode == _config.VERIFY:
            self._verification.functions = writing
        else:
            self._before = writing

    def _log_configuration(self, namelists: _config.Namelists) -> None:
        """The configuration inputs, their precision, and what the reader could not read (the
        plugin does not need it: what it needs raised before)."""
        for name, values in self._configuration.items():
            log.info(
                f"{name} configuration from {_config.NAMELIST_FILE}: "
                + ", ".join(f"{k} {_describe(v)}" for k, v in values.items())
            )
        self._log_configuration_precision(namelists)
        problems = namelists.problems()
        if problems:
            log.info(
                f"{_config.NAMELIST_FILE}: the plugin does not need what it could not read,"
                f" ignored: {'; '.join(problems)}"
            )

    def _log_configuration_precision(self, namelists: _config.Namelists) -> None:
        """
        A real configuration value is exact if ICON wrote it with 17 significant digits; with 16
        (nvfortran's F format) if ICON's value is the nearest double to a decimal of at most 15
        digits, as a namelist literal is ('_config.maybe_inexact'). Log the ones whose value is
        no such decimal: they may differ from ICON's values in the last bit.
        """
        entries = self._real_entries(namelists)
        count = len(entries)
        if not count:
            return
        inexact = [
            f"{label} ({group}: {entry})"
            for label, (group, entry) in entries.items()
            if namelists.maybe_inexact(group, entry)
        ]
        if inexact:
            log.warning(
                f"configuration: {len(inexact)} of {count} real values have 16 significant"
                f" digits in {_config.NAMELIST_FILE} and no shorter decimal, so they may differ"
                f" from ICON's values in the last bit: {', '.join(inexact)}"
            )
        else:
            log.info(
                f"configuration: {count} real values, each written with 17 significant digits"
                " or the nearest double to a decimal of at most 15 (exact if ICON's value is that"
                " decimal's double, as for namelist literals)"
            )

    def _real_entries(self, namelists: _config.Namelists) -> dict[str, tuple[str, str]]:
        """The real configuration values and their namelist entries (group, name)."""
        entries = {}
        for name, values in self._configuration.items():
            for param, value in values.items():
                if type(value) is float:
                    setting = _config.ARGUMENTS[name][param]
                    entries[f"{name}.{param}"] = _config.source(namelists, setting)
                elif value is not None and isinstance(value, _granule.DiffusionConfig):
                    for entry in _granule.diffusion_entries():
                        if entry.kind is float:
                            entries[f"{name}.{param}.{entry.field}"] = (entry.group, entry.name)
        return entries

    def register(self) -> None:
        """Primary constructor: select host or device, log the setup, register the callbacks."""
        comin = self._comin
        if comin.descrdata_get_global().has_device:
            if cp is None:
                raise RuntimeError("ICON runs on the GPU, but CuPy cannot be imported.")
            self._device_xp = cp
            self._device_flag = comin.COMIN_FLAG_DEVICE
        dycore = {name: operator.index(self._entry_point(name)) for name in DYCORE_ENTRY_POINTS}
        self._entry_point(INTEGRATE_START)
        log.info(
            f"source sha1 {source_sha1()}, icon4py.bindings {icon4py.bindings.__version__}"
            f" at {pathlib.Path(icon4py.bindings.__file__).parent}, device {self._device_xp is not None}"
        )
        if self._ignored_env:
            log.warning(
                f"ignored: {', '.join(self._ignored_env)}; this plugin reads only"
                f" {', '.join(sorted(KNOWN_ENV))} (the variables of removed checks have no effect)."
            )
        self._read_configuration()
        log.info("counting the callbacks at " + ", ".join(f"{k}={v}" for k, v in dycore.items()))
        if not self._check:
            log.info(f"{CHECK_ENV}=0: in-job consistency checks are off.")
        if self._timing:
            log.info(f"{TIMING_ENV}=1: per-call phase timing is on.")
            self._log_state("primary constructor")
        if self._profile_call is not None:
            log.info(f"{PROFILE_ENV}={self._profile_call}: that call is profiled.")
        self._log_routes()
        if self._probing:
            self._log_probe()

        callbacks: dict[str, Callable[[], None]] = {
            SECONDARY_CONSTRUCTOR: self.secondary_constructor
        }
        for entry in self._functions.values():
            if not entry.inout:
                callbacks[entry.entry_point] = functools.partial(
                    self.on_entry_point, entry.entry_point
                )
        for name in DYCORE_ENTRY_POINTS:
            callbacks[name] = functools.partial(self.on_dycore_entry_point, name)
        callbacks[INTEGRATE_START] = self.on_integrate_start
        callbacks[DESTRUCTOR] = self.destructor
        for name, callback in callbacks.items():
            self._entry_point(name)(callback)  # 'comin.entry_point' registers as a decorator

    def _log_routes(self) -> None:
        """Log the inputs once: counts per kind and per native route."""
        rows = [
            (name, entry, param, kind)
            for name, entry in self._functions.items()
            for param, kind in entry.params.items()
        ]
        n_arrays = sum(isinstance(kind, _arguments.ArrayParam) for *_, kind in rows)
        routes = [route(name, entry, param) for name, entry, param, _ in rows]
        counts = ", ".join(f"{r} {routes.count(r)}" for r in ROUTES)
        log.info(
            f"inputs: {len(rows)} ({n_arrays} arrays, {len(rows) - n_arrays} scalars and"
            f" configurations) of {', '.join(self._functions)}; every one from a native route:"
            f" {counts}"
        )

    def _domain(self) -> Any:
        return self._comin.descrdata_get_domain(_arguments.DOMAIN_ID)

    def _local_count(self, location: str | None) -> int | None:
        """The local number of cells, edges or vertices ('location') in domain 1."""
        if location is None:
            return None
        if location not in self._counts:
            kind, count = LOCAL_COUNT[location]
            self._counts[location] = int(getattr(getattr(self._domain(), kind), count))
        return self._counts[location]

    def _log_probe(self) -> None:
        """Probe mode: what the plugin compares where (it computes nothing)."""
        sites = [
            f"{name} at {entry.entry_point} (py2fgen's call of the same pass, recorded)"
            for name, entry in self._functions.items()
            if name not in self._probed
        ] + [f"{name} at py2fgen's call (kept at {DIFFUSION_BEFORE})" for name in self._probed]
        log.info(
            f"{PROBE_ENV}=1: the py2fgen probe compares the inputs the plugin would hand the"
            f" granule with py2fgen's: {'; '.join(sites)}; and the grid and granule the plugin"
            f" builds at {TIMELOOP_BEFORE} with those py2fgen's bindings built; py2fgen's glue"
            f" installs the recorders with PY2FGEN_EXTRA_CALLABLES={PROBE_MODULE}:install; the"
            " plugin computes nothing"
        )

    def secondary_constructor(self) -> None:
        """Request ICON's variables, or stay idle (ICON's mode decides)."""
        mode = self.mode
        exposed = set(self._comin.var_list())
        self._set_runtime_loutshs(exposed)
        if self._probing:
            self._probe_secondary_constructor(exposed)
            return
        if not mode.computes:
            log.info(f"idle: ICON mode {mode.name} ({mode.switches}); requested nothing.")
            return
        errors: list[str] = []
        self._request_static_variables(exposed, errors)
        self._request_icon_variables(exposed, errors)
        if errors:
            raise RuntimeError(
                f"icon4py ComIn plugin: {len(errors)} problem(s) with ICON's variables:\n  "
                + "\n  ".join(errors)
            )
        self._active = True
        count = len(self._icon_handles) + sum(
            variable is not None
            for handles in self._static.handles.values()
            for variable in handles.values()
        )
        log.info(f"active: requested {count} of ICON's variables for {', '.join(self._functions)}.")

    def _set_runtime_loutshs(self, exposed: set[tuple[str, int]]) -> None:
        """
        The DiffusionConfig of 'diffusion_init' gets ICON's value of 'loutshs' when ICON
        initialises icon4py's diffusion ('_config.runtime_loutshs'): with NWP physics, whether
        ICON has the variable 'ddt_tke_hsh' on domain 1; else the configured value it holds.
        """
        values = self._configuration.get("diffusion_init", {})
        config = values.get("config")
        if not isinstance(config, _granule.DiffusionConfig):
            return
        value, reason = _config.runtime_loutshs(
            int(config.iforcing), config.loutshs, exposed, _arguments.DOMAIN_ID
        )
        if value != config.loutshs:
            values["config"] = dataclasses.replace(config, loutshs=value)
        log.info(f"diffusion_init configuration at run time: {reason}")

    def _probe_secondary_constructor(self, exposed: set[tuple[str, int]]) -> None:
        """Probe mode: request ICON's variables that the granules get (READ only), where the
        plugin would hand them to the granule."""
        errors: list[str] = []
        self._request_static_variables(exposed, errors)
        self._request_icon_variables(exposed, errors)
        if errors:
            raise RuntimeError(
                f"icon4py ComIn plugin: {len(errors)} problem(s) with ICON's variables for the"
                " py2fgen probe:\n  " + "\n  ".join(errors)
            )
        mode = self.mode
        log.info(
            f"idle: ICON mode {mode.name} ({mode.switches}); computes nothing; requested ICON's"
            f" variables for the py2fgen probe ({PROBE_ENV}=1)."
        )

    def _request_static_variables(self, exposed: set[tuple[str, int]], errors: list[str]) -> None:
        """
        Request ICON's variables of STATIC_VARIABLES: READ, on the device on the GPU (no copies),
        at the entry point of their function, and where the granule computes, which keeps them
        (ComIn hands out a variable per callback). An absent one must be an optional argument.
        """
        comin = self._comin
        flags = comin.COMIN_FLAG_READ | self._device_flag
        computes = self._compute_entry_point()
        for name, entry in self._functions.items():
            wanted = [param for param in entry.params if static_route(name, entry, param)]
            if not wanted:
                continue
            arrays = {a.name: a for a in self._signatures[name].arrays}
            names = [entry.entry_point] + ([computes] if computes != "-" else [])
            context = [self._entry_point(e) for e in dict.fromkeys(names)]
            handles: dict[str, Any] = {}
            requested, absent = [], []
            for param in wanted:
                icon_name = STATIC_VARIABLES[name][param]
                label = icon_name if icon_name == param else f"{icon_name} (as {param})"
                descriptor = (icon_name, _arguments.DOMAIN_ID)
                if descriptor not in exposed:
                    if arrays[param].optional:
                        handles[param] = None
                        absent.append(label)
                    else:
                        errors.append(
                            f"'{icon_name}': ICON does not expose its variable, the argument"
                            f" '{param}' of '{name}'."
                        )
                    continue
                try:
                    handles[param] = comin.var_get(list(context), descriptor, flags)
                    requested.append(label)
                except Exception as error:
                    errors.append(f"'{icon_name}' (ICON's variable): 'var_get' failed: {error}")
            self._static.handles[name] = handles
            log.info(
                f"requested ICON's {' '.join(requested) or 'nothing'} (READ"
                f"{' | DEVICE' if self._device_flag else ''}) at {entry.entry_point} for {name}"
                + (
                    f"; not exposed (absent optional arguments): {' '.join(absent)}"
                    if absent
                    else ""
                )
                + (
                    f"; also at {computes}, where the granule computes with them"
                    if computes not in ("-", entry.entry_point)
                    else ""
                )
            )

    def _static_bound(self, name: str) -> dict[str, _arguments.BoundArray]:
        """ICON's variables of STATIC_VARIABLES that 'name' gets, as its array arguments (only
        valid inside a callback that requested them)."""
        arrays = {a.name: a for a in self._signatures[name].arrays}
        return {
            param: (
                _arguments.BoundArray(arrays[param], None, (), present=False)
                if variable is None
                else _arguments.icon_bound_array(
                    variable, arrays[param], STATIC_VARIABLES[name][param]
                )
            )
            for param, variable in self._static.handles.get(name, {}).items()
        }

    def _static_arguments(self, name: str, k: int) -> dict[str, _arguments.BoundArray]:
        """
        ICON's variables of STATIC_VARIABLES that 'name' gets, fetched now. Raises if one is not
        where it was at the first call (address, shape, presence; 'k': the pass).
        """
        bound = self._static_bound(name)
        if not bound:
            return {}
        pointers = {param: self._pointer(array) for param, array in bound.items()}
        first, recorded = self._static.first.setdefault(name, (k, pointers))
        moved = [STATIC_VARIABLES[name][p] for p, ptr in pointers.items() if ptr != recorded[p]]
        if moved:
            raise RuntimeError(
                f"icon4py ComIn plugin: ICON's variable(s) {' '.join(moved)} of '{name}' moved"
                f" between pass {first} and pass {k} (address, shape or presence); the plugin"
                " expects ICON to allocate them once."
            )
        present = sum(pointer.present for pointer in pointers.values())
        log.info(
            f"{name} pass {k}: ICON's variables {' '.join(bound)} fetched at"
            f" {self._functions[name].entry_point} ({present} present, {len(pointers) - present}"
            " absent); "
            + ("addresses recorded" if first == k else f"addresses as at pass {first}"),
            extra=ALL_RANKS,
        )
        return bound

    def _request_icon_variables(self, exposed: set[tuple[str, int]], errors: list[str]) -> None:
        """
        Request ICON's own variables of the names of the array arguments of the writing
        functions, on the device on the GPU, so that ComIn copies nothing before or after the
        callback: in SUBSTITUTE READ and WRITE at DIFFUSION_BEFORE, where the functions compute
        on them; in VERIFY READ only, at DIFFUSION_BEFORE, where the plugin copies them, and at
        DIFFUSION_AFTER, where it compares its results with them; in probe mode READ only, at
        DIFFUSION_BEFORE. An absent one must be optional.
        """
        comin = self._comin
        if self._copied:
            flags = comin.COMIN_FLAG_READ | self._device_flag
            names = [DIFFUSION_BEFORE, DIFFUSION_AFTER]
        elif self._probed:
            flags = comin.COMIN_FLAG_READ | self._device_flag
            names = [DIFFUSION_BEFORE]
        else:
            flags = comin.COMIN_FLAG_READ | comin.COMIN_FLAG_WRITE | self._device_flag
            names = [DIFFUSION_BEFORE]
        context = [self._entry_point(e) for e in names]
        absent = []
        for name in self._before + self._copied + self._probed:
            for param in self._signatures[name].arrays:
                descriptor = (param.name, _arguments.DOMAIN_ID)
                if descriptor not in exposed:
                    if param.optional:
                        absent.append(param.name)
                    else:
                        errors.append(
                            f"'{param.name}': ICON does not expose its variable, which '{name}'"
                            f" gets at {DIFFUSION_BEFORE}."
                        )
                    continue
                try:
                    self._icon_handles[param.name] = comin.var_get(list(context), descriptor, flags)
                except Exception as error:
                    errors.append(f"'{param.name}' (ICON's variable): 'var_get' failed: {error}")
        access = "READ | WRITE" if self._before else "READ"
        log.info(
            f"requested ICON's {' '.join(self._icon_handles) or 'nothing'} ({access}"
            f"{' | DEVICE' if self._device_flag else ''}) at {' and '.join(names)} for"
            f" {', '.join(self._before + self._copied + self._probed)}"
            + (f"; not exposed (absent optional arguments): {' '.join(absent)}" if absent else "")
            + (
                f"; copies them at {DIFFUSION_BEFORE} into the plugin's buffers, computes on the"
                f" copies and compares its results with them at {DIFFUSION_AFTER}"
                if self._copied
                else ""
            )
            + ("; for the py2fgen probe" if self._probed else "")
        )

    def _pointers(self, variable: Any, on_device: bool) -> tuple[int, int | None]:
        """Host and (on the GPU) device address of a ComIn variable."""
        host = _views.data_ptr(np.asarray(variable))
        if not on_device:
            return host, None
        return host, int(variable.__cuda_array_interface__["data"][0])

    def on_integrate_start(self) -> None:
        """A new time step of the domain: restart the order of its entry points."""
        domain_id = self._domain_id()
        if domain_id is None:
            return
        self._order.integrate_start(domain_id)

    def on_dycore_entry_point(self, entry_point: str) -> None:
        """
        Count the callback; follow the order of SOLVE_NH_BEFORE; at DIFFUSION_BEFORE of domain 1
        derive the per-call state and compute (SUBSTITUTE) or copy ICON's input (VERIFY); at
        DIFFUSION_AFTER of domain 1 compute on the copies and compare the results (VERIFY).
        """
        self._ep_counts[entry_point] += 1
        domain_id = self._domain_id()
        self._ep_domains[entry_point].add(domain_id)
        if domain_id is None:
            return
        if entry_point == SOLVE_NH_BEFORE:
            self._order.solve_nh(domain_id)
        elif entry_point == DIFFUSION_BEFORE and domain_id == _arguments.DOMAIN_ID:
            if self._active or self._probed:
                self._on_diffusion_before()
        elif entry_point == DIFFUSION_AFTER and domain_id == _arguments.DOMAIN_ID:
            if self._active and self._copied:
                self._on_diffusion_after()
            elif self.probe is not None:
                for name in self._probed:
                    self.probe.after(name)
        elif entry_point == DIFFUSION_BEFORE and self._active:
            log.warning(
                f"{entry_point} fired for domain {domain_id}; the plugin computes domain 1 only;"
                " ignored.",
                extra=ALL_RANKS,
            )

    def _on_diffusion_before(self) -> None:
        """DIFFUSION_BEFORE of domain 1: the guards, the per-call state, the compute or copy."""
        jg = _arguments.DOMAIN_ID
        calls = self._calls
        if not self._hdiff_vn:
            # the regular pair sits outside ICON's 'IF (lhdiff_vn)': ICON calls no diffusion
            calls.skipped += 1
            log.info(
                f"{DIFFUSION_BEFORE} {self._ep_counts[DIFFUSION_BEFORE]}: lhdiff_vn F, ICON calls"
                " no diffusion; nothing to do.",
                extra=ALL_RANKS,
            )
            return
        if calls.pending is not None:
            raise RuntimeError(
                f"icon4py ComIn plugin: diffusion call {calls.pending.call} of domain 1 was not"
                f" followed by {DIFFUSION_AFTER} before the next {DIFFUSION_BEFORE}: the plugin"
                " has not compared its results with ICON's."
            )
        calls.count += 1
        state = CallState(
            call=calls.count,
            linit=self._derive_linit(jg, calls.count),
            dtime=float(self._comin.descrdata_get_timesteplength(jg)),
        )
        state.routes = {"dtime": state.dtime, "linit": state.linit}
        if self._before:
            self._compute(state)
        elif self._copied:
            self._copy_inputs(state)
            calls.pending = state
        elif self._probed:
            self._probe_expect(state)

    def _probe_expect(self, state: CallState) -> None:
        """
        Probe mode, at DIFFUSION_BEFORE of domain 1: keep the arguments the plugin would hand the
        granule here in SUBSTITUTE (ICON's variables as zero-copy views; the time step and the
        initial-call flag), for py2fgen's call that follows.
        """
        assert self.probe is not None
        for name in self._probed:
            _, kwargs = self._native_arguments(name)
            params = self._functions[name].params
            kwargs.update({p: state.routes[p] for p in PER_CALL_SCALARS if p in params})
            self.probe.expect(name, state.call, self.probe.natives(name, kwargs, values=False))
        log.info(
            f"{', '.join(self._probed)} call {state.call}: kept the arguments for the py2fgen probe"
            f" at {DIFFUSION_BEFORE} (linit {_logical(state.linit)}, dtime {state.dtime!r})",
            extra=ALL_RANKS,
        )

    def _init_arguments(self, name: str, k: int) -> tuple[_arguments.BoundFunction, dict[str, Any]]:
        """
        The inputs of a function that does not write, as the granule gets them in pass 'k': the
        configuration, the descriptive data and ICON's variables of STATIC_VARIABLES.
        """
        kwargs = self._routes(name, None)
        static = self._static_arguments(name, k)
        kwargs.update({p: _arguments.array_argument(b, self._device_xp) for p, b in static.items()})
        missing = [p for p in self._functions[name].params if p not in kwargs]
        if missing:
            raise RuntimeError(
                f"icon4py ComIn plugin: no value for {', '.join(missing)} of '{name}'."
            )
        return _arguments.BoundFunction(self._signatures[name], tuple(static.values())), kwargs

    def _probe_init(self, entry_point: str, k: int) -> None:
        """
        Probe mode, at the entry point of the functions that do not write (where the plugin
        calls them when it computes): build their inputs as for the granule and compare them
        with py2fgen's call of the same pass; then build the grid and the granule from them, as
        when the plugin computes, and compare them with the objects py2fgen's bindings built.
        """
        assert self.probe is not None
        built = True
        for name, entry in self._functions.items():
            if entry.entry_point != entry_point or name in self._probed:
                continue
            _, arguments = self._init_arguments(name, k)
            self.probe.compare_init(name, k, self.probe.natives(name, arguments, values=True))
            if not built:
                continue
            try:
                self._signatures[name].function(**arguments)
            except Exception as error:  # the probe reports it; py2fgen computes on
                built = False
                self.probe.not_probed("objects", f"pass {k}", f"{name}: {error!r}")
        if built:
            grid_state = getattr(self._granule, "grid_state", None)
            diffusion = getattr(self._granule, "diffusion", None)
            self.probe.compare_objects(k, grid_state, diffusion)

    def _copy_inputs(self, state: CallState) -> None:
        """
        VERIFY, at DIFFUSION_BEFORE of domain 1, before ICON computes: copy ICON's variables
        that are the array arguments of the writing functions into the plugin's buffers (the
        first block, in py2fgen's layout; allocated at the first call, kept for the whole run).
        An absent variable is an absent optional argument.
        """
        on_device = self._device_xp is not None
        present, absent = [], []
        for name in self._copied:
            arrays = []
            for param in self._signatures[name].arrays:
                variable = self._icon_handles.get(param.name)
                source = (
                    None
                    if variable is None
                    else _arguments.array_argument(
                        _arguments.icon_bound_array(variable, param), self._device_xp
                    )
                )
                if source is None:
                    arrays.append(_arguments.BoundArray(param, None, (), present=False))
                    state.routes[param.name] = None
                    absent.append(param.name)
                    continue
                buffer = self._verification.copies.take(
                    f"{name}.{param.name}", getattr(source, "ndarray", source)
                )
                arrays.append(
                    _arguments.BoundArray(param, buffer, tuple(buffer.shape), present=True)
                )
                state.routes[param.name] = (
                    buffer if param.dims is None else _views.field_view(buffer, param.dims)
                )
                present.append(param.name)
            state.copies[name] = _arguments.BoundFunction(self._signatures[name], tuple(arrays))
        state.icon_pointers = {
            n: self._pointers(v, on_device) for n, v in self._icon_handles.items()
        }
        self._synchronize()  # ICON overwrites its variables next, outside CuPy's stream
        self._calls.copied += 1
        mib = self._verification.copies.nbytes() / 2**20
        log.info(
            f"{', '.join(self._copied)} call {state.call}: copied ICON's {' '.join(present)} at"
            f" {DIFFUSION_BEFORE} into the plugin's buffers ({mib:.2f}"
            f" MiB{' on the device' if on_device else ''}, kept)"
            + (f"; absent: {' '.join(absent)}" if absent else ""),
            extra=ALL_RANKS,
        )

    def _on_diffusion_after(self) -> None:
        """
        VERIFY, at DIFFUSION_AFTER of domain 1: compute the last call on the plugin's copies of
        ICON's input, compare the results with ICON's variables, which hold ICON's own results
        now, and print the comparison in the format of ICON's tables (rank 0). Collective over
        ICON's work PEs.
        """
        start = time.perf_counter()
        calls = self._calls
        state, calls.pending = calls.pending, None
        if state is None:
            return  # no diffusion call (lhdiff_vn F)
        on_device = self._device_xp is not None
        now = {n: self._pointers(v, on_device) for n, v in self._icon_handles.items()}
        moved = [n for n in now if now[n] != state.icon_pointers.get(n)]
        if moved:
            raise RuntimeError(
                f"icon4py ComIn plugin: ICON's variables {' '.join(moved)} at {DIFFUSION_AFTER}"
                f" are not those copied at {DIFFUSION_BEFORE} of diffusion call {state.call}: the"
                " plugin cannot compare its results with ICON's."
            )
        self._compute_on_copies(state, start)
        for name in self._copied:
            pairs = self._verify_pairs(name, state.copies[name])
            log.info(
                f"{VERIFY_HEADER} {name} call {state.call} (linit {_logical(state.linit)}):"
                f" icon4py on the plugin's copies of ICON's input ({DIFFUSION_BEFORE},"
                f" computed at {DIFFUSION_AFTER}) against ICON at {DIFFUSION_AFTER}; atol"
                f" {_verify.ATOL:g}, rtol {_verify.RTOL:g}"
            )
            not_close = {}
            for which in _verify.TABLES:
                rows = _verify.table(pairs, which, self._verification.allreduce)
                for line in _verify.header_lines(which):
                    log.info(line)
                for row in rows:
                    log.info(_verify.format_row(row))
                log.info(_verify.footer_line())
                not_close[which] = sum(not row.is_close for row in rows)
                calls.not_close[which] = calls.not_close.get(which, 0) + not_close[which]
            log.info(
                f"{name} call {state.call} compared at {DIFFUSION_AFTER}: {len(pairs)} fields,"
                f" ICON's variables as at {DIFFUSION_BEFORE}; rows not close: "
                + ", ".join(f"{k} {v}" for k, v in not_close.items()),
                extra=ALL_RANKS,
            )
        calls.compared += 1

    def _compute_on_copies(self, state: CallState, start: float) -> None:
        """VERIFY: call the writing functions on the plugin's copies of ICON's input, with the
        time step and the initial-call flag derived at DIFFUSION_BEFORE."""
        for name in self._copied:
            entry = self._functions[name]
            clock = _PhaseClock(self._timing, start)
            clock("guard")
            kwargs = dict(self._configuration.get(name, {}))
            kwargs.update({p: v for p, v in state.routes.items() if p in entry.params})
            missing = [p for p in entry.params if p not in kwargs]
            if missing:
                raise RuntimeError(
                    f"icon4py ComIn plugin: no value for {', '.join(missing)} of '{name}' call"
                    f" {state.call}."
                )
            self._check_static(state.call)
            clock("args")
            profiled = self._run(
                name, entry, state.copies[name], kwargs, call=state.call, clock=clock
            )
            log.info(
                f"{name} call {state.call} done at {DIFFUSION_AFTER} on the plugin's copies of"
                f" ICON's input from {DIFFUSION_BEFORE} (linit {_logical(state.linit)}).",
                extra=ALL_RANKS,
            )
            clock("done")
            if clock.enabled:
                self._log_timing(name, state.call, clock.phases, profiled)

    def _verify_pairs(self, name: str, copies: _arguments.BoundFunction) -> list[_verify.Pair]:
        """ICON's variables and the plugin's results, in the order of ICON's tables: the
        edge fields first, then the cell fields, each in the order of the arguments."""
        pairs = []
        for array in sorted(copies.arrays, key=lambda a: a.param.location != "edge"):
            if not array.present:
                continue
            variable = self._icon_handles[array.param.name]
            reference = _arguments.array_argument(
                _arguments.icon_bound_array(variable, array.param), self._device_xp
            )
            pairs.append(
                _verify.Pair(
                    name=array.param.name,
                    reference=getattr(reference, "ndarray", reference),
                    field=array.variable,
                    real=self._real_entry_count(array.param.location),
                )
            )
        return pairs

    def _real_entry_count(self, location: str | None) -> int:
        """
        The end of ICON's 'domain' table for the location (default: cells): the end index of
        the outermost halo row in the first block (ICON's 'get_indices_c/e' from refin_ctrl 1 to
        'min_rlcell/min_rledge'; ComIn's 'end_index' starts at the lowest refin_ctrl).
        """
        location = location or "cell"
        if location not in self._verification.real_entries:
            kind, _ = LOCAL_COUNT[location]
            end_index = np.asarray(getattr(self._domain(), kind).end_index)
            end = int(end_index.reshape(-1)[0])
            self._verification.real_entries[location] = end
            log.info(
                f"verification: the 'domain' table of the {location}s ends at {end} (end_index of"
                f" the outermost halo row; {LOCAL_COUNT[location][1]} {self._local_count(location)})",
                extra=ALL_RANKS,
            )
        return self._verification.real_entries[location]

    def _derive_linit(self, jg: int, call: int) -> bool:
        """
        The initial-call flag from the order of the entry points: no SOLVE_NH_BEFORE of the
        domain since its last INTEGRATE_START. Checked against ICON's condition; raises if they
        disagree.
        """
        step, solves, position = self._order.diffusion_call(jg)
        if step == 0:
            raise RuntimeError(
                f"icon4py ComIn plugin: {DIFFUSION_BEFORE} of domain {jg} before any"
                f" {INTEGRATE_START} of it: the plugin cannot tell the initial diffusion call from"
                " a regular one."
            )
        derived = solves == 0
        if self._initial is None:
            raise RuntimeError("icon4py ComIn plugin: ICON's initial-call condition is unknown.")
        expected = self._initial.expected(step, position)
        agree = derived == expected
        self._calls.linit.append((call, derived, agree))
        text = (
            f"linit call {call}: {_logical(derived)} from the entry-point order ({solves}"
            f" {SOLVE_NH_BEFORE} since {INTEGRATE_START} {step}); ICON's condition:"
            f" {_logical(expected)} ({self._initial.describe()}, step"
            f" {step}, call {position} of the step): {'agree' if agree else 'disagree'}"
        )
        log.info(text, extra=ALL_RANKS)
        if not agree:
            raise RuntimeError(f"icon4py ComIn plugin: {text}.")
        return derived

    def _pointer(self, bound: _arguments.BoundArray) -> _arguments.Pointer:
        """An array input as the granule sees it: addresses, shape, presence."""
        if not bound.present:
            return _arguments.Pointer(device=None, host=None, shape=(), present=False)
        host, device = self._pointers(
            bound.variable, _arguments.is_on_device(bound, self._device_xp)
        )
        return _arguments.Pointer(device=device, host=host, shape=bound.shape, present=True)

    def _native_arguments(self, name: str) -> tuple[_arguments.BoundFunction, dict[str, Any]]:
        """The array arguments of 'name' from ICON's own variables (zero-copy, py2fgen's layout)."""
        arrays = []
        for param in self._signatures[name].arrays:
            variable = self._icon_handles.get(param.name)
            if variable is None:  # ICON has no such variable: an absent optional argument
                arrays.append(_arguments.BoundArray(param, None, (), present=False))
            else:
                arrays.append(_arguments.icon_bound_array(variable, param))
        bound = _arguments.BoundFunction(self._signatures[name], tuple(arrays))
        kwargs = {b.param.name: _arguments.array_argument(b, self._device_xp) for b in arrays}
        return bound, kwargs

    def _compute(self, state: CallState) -> None:
        """SUBSTITUTE: call the writing functions on ICON's own variables, at DIFFUSION_BEFORE."""
        for name in self._before:
            entry = self._functions[name]
            clock = _PhaseClock(self._timing, time.perf_counter())
            clock("guard")
            bound, kwargs = self._native_arguments(name)
            kwargs.update(self._configuration.get(name, {}))
            kwargs.update({p: state.routes[p] for p in PER_CALL_SCALARS if p in entry.params})
            self._check_static(state.call)
            clock("args")
            profiled = self._run(name, entry, bound, kwargs, call=state.call, clock=clock)
            self._calls.computed += 1
            log.info(
                f"{name} call {state.call} done at {DIFFUSION_BEFORE} (linit"
                f" {_logical(state.linit)}).",
                extra=ALL_RANKS,
            )
            clock("done")
            if clock.enabled:
                self._log_timing(name, state.call, clock.phases, profiled)

    def _log_counts(self) -> None:
        counts = ", ".join(f"{k} {v}" for k, v in self._ep_counts.items())
        domains = sorted({d for s in self._ep_domains.values() for d in s}, key=str)
        log.info(f"entry point callbacks: {counts} (domains {domains})", extra=ALL_RANKS)
        calls = self._calls
        if calls.count or calls.skipped:
            initial = sum(derived for _, derived, _ in calls.linit)
            agree = sum(ok for *_, ok in calls.linit)
            log.info(
                f"per-call state: {calls.count} diffusion calls of domain 1 at {DIFFUSION_BEFORE},"
                f" {calls.computed} computed there, {calls.copied} copied there, {calls.skipped}"
                f" without lhdiff_vn; linit: initial {initial}, regular"
                f" {len(calls.linit) - initial}, ICON's condition agrees at {agree} of"
                f" {len(calls.linit)}; {INTEGRATE_START}"
                f" {self._order.steps.get(_arguments.DOMAIN_ID, 0)} (domain 1)",
                extra=ALL_RANKS,
            )

    def _domain_id(self) -> int | None:
        try:
            return int(self._comin.current_get_domain_id())
        except Exception:
            return None  # the adapter raises for the negative ids of calls outside the domain loop

    def on_entry_point(self, entry_point: str) -> None:
        """Call the functions that do not write at 'entry_point' (every pass)."""
        start = time.perf_counter()
        k = self._passes[entry_point] = self._passes.get(entry_point, 0) + 1
        if self._probing:
            self._probe_init(entry_point, k)
            return
        if not self._active:
            if self._timing:
                self._log_state(f"{entry_point}, idle")
            return
        if self._timing:
            self._log_state(f"{entry_point}, before")
        for name, entry in self._functions.items():
            if entry.entry_point == entry_point and not entry.inout:
                bound, kwargs = self._init_arguments(name, k)
                clock = _PhaseClock(False, start)
                self._run(name, entry, bound, kwargs, call=k, clock=clock)
                log.info(f"{name} done (pass {k}).")
        if self._timing:
            self._log_state(f"{entry_point}, after")

    def _routes(self, name: str, state: CallState | None) -> dict[str, Any]:
        """The values of the native routes of 'name' that the plugin has: the configuration, the
        inputs from the descriptive data, and the per-call state of this call."""
        routes = dict(self._configuration.get(name, {}))
        routes.update(self._descriptive_arguments(name))
        if state is not None:
            params = self._functions[name].params
            routes.update({p: v for p, v in state.routes.items() if p in params})
        return routes

    def _descriptive_arguments(self, name: str) -> dict[str, Any]:
        """The inputs of 'name' from ComIn's descriptive data, as the granule gets them (a fresh
        copy per call). A failing one stops ICON."""
        wanted = [
            param for param in self._functions[name].params if _descrdata.has_route(name, param)
        ]
        if not wanted:
            return {}
        if not self._descrdata.checked:
            log.info(self._descrdata.describe(), extra=ALL_RANKS)
        arrays = {a.name: a for a in self._signatures[name].arrays}
        values: dict[str, Any] = {}
        for param in wanted:
            try:
                values[param] = self._descrdata.argument(
                    name, param, arrays.get(param), self._device_xp
                )
            except Exception as error:
                raise RuntimeError(
                    f"icon4py ComIn plugin: '{param}' of '{name}' from ComIn's descriptive"
                    f" data ({_descrdata.ROUTES[name][param].source}): {error}"
                ) from error
        log.info(
            f"{name}: {len(wanted)} arguments from ComIn's descriptive data (host copies"
            f" {self._descrdata.host_bytes() / 2**20:.2f} MiB, a fresh copy"
            f"{' on the device' if self._device_xp is not None else ''} per pass)",
            extra=ALL_RANKS,
        )
        return values

    def _check_static(self, call: int) -> None:
        """
        ICON's variables of STATIC_VARIABLES, which the granule keeps since the init call,
        fetched again where the granule computes: they must be where the first pass found them
        (ComIn hands out an ICON variable per callback). Raises if one moved.
        """
        where = self._compute_entry_point()
        for name, handles in self._static.handles.items():
            if not handles:
                continue
            if name not in self._static.first:
                raise RuntimeError(
                    f"icon4py ComIn plugin: '{name}' has not fetched ICON's variables before"
                    f" diffusion call {call}."
                )
            first, recorded = self._static.first[name]
            moved = [
                STATIC_VARIABLES[name][param]
                for param, bound in self._static_bound(name).items()
                if self._pointer(bound) != recorded[param]
            ]
            if moved:
                raise RuntimeError(
                    f"icon4py ComIn plugin: ICON's variable(s) {' '.join(moved)} of '{name}' at"
                    f" {where} of diffusion call {call} are not where pass {first} found them,"
                    " and the granule keeps them from there (address, shape or presence)."
                )
            self._static.checks[name] = self._static.checks.get(name, 0) + 1
            if self._static.checks[name] == 1:
                log.info(
                    f"{name}: ICON's variables {' '.join(handles)} fetched at {where} (diffusion"
                    f" call {call}): at the addresses of pass {first}, which the granule keeps;"
                    " checked at every call",
                    extra=ALL_RANKS,
                )

    def _run(
        self,
        name: str,
        entry: FunctionEntry,
        bound: _arguments.BoundFunction,
        kwargs: Mapping[str, Any],
        *,
        call: int,
        clock: _PhaseClock,
    ) -> bool:
        """Run the granule's function with the consistency checks; True if it was profiled."""
        first_call = name not in self._first_call_done
        if first_call:
            self._log_optional_arguments(name, bound, kwargs)
        log_values = entry.inout and self._check and first_call
        sums_before = self._value_sums(bound) if log_values else {}
        clock("sums")
        if clock.enabled:
            self._synchronize()
            clock("presync")

        profiled = entry.inout and call == self._profile_call
        if profiled:
            self._profile(name, call, bound.signature.function, kwargs)
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
        return profiled

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
        self, name: str, bound: _arguments.BoundFunction, kwargs: Mapping[str, Any]
    ) -> None:
        """Log what the first call gets for each optional array argument (e.g. empty 'zd_*' lists)."""
        parts = []
        for array in bound.arrays:
            if not array.param.optional:
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
        bound: _arguments.BoundFunction,
        kwargs: Mapping[str, Any],
        first_call: bool,
    ) -> None:
        """
        Consistency guard: every Field argument still aliases the memory it was built from
        (ICON's variable, or in VERIFY the plugin's buffer) after the call (a copying Field
        helper, or a Field whose buffer GT4Py replaced, would fail here). It does not show that
        the function wrote through the Fields; the comparisons of ICON's results with the
        py2fgen path do.
        """
        count = 0
        for array in bound.arrays:
            value = kwargs[array.param.name]
            if array.param.dims is None or value is None or 0 in array.shape:
                continue
            field_ptr = _views.data_ptr(value.ndarray)
            buffer_ptr = _arguments.variable_data_ptr(array, self._device_xp)
            if field_ptr != buffer_ptr:
                raise RuntimeError(
                    f"icon4py ComIn plugin: '{array.param.name}' of '{name}' does not alias the"
                    f" array it was built from (Field {field_ptr:#x}, array {buffer_ptr:#x}); the"
                    " results would be lost."
                )
            count += 1
        if first_call:
            log.info(f"pointer identity OK ({count} fields)", extra=ALL_RANKS)

    def _value_sums(self, bound: _arguments.BoundFunction) -> dict[str, float]:
        """Sums of fresh views of some fields; information only, so it never raises."""
        sums: dict[str, float] = {}
        try:
            for array in bound.arrays:
                if array.param.name in VALUE_LOG_FIELDS and array.present and 0 not in array.shape:
                    on_device = _arguments.is_on_device(array, self._device_xp)
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
        """Release the granule and CuPy's cached memory; log the counts."""
        self._log_counts()
        if self.probe is not None and self._probing:
            self.probe.summary()
            self._granule.release()
            self._static.handles.clear()
            self._icon_handles.clear()
            self._descrdata.release()
        for name, checks in self._static.checks.items():
            first = self._static.first.get(name, (0, {}))[0]
            log.info(
                f"{name}: ICON's variables at the addresses of pass {first} at all {checks}"
                f" diffusion calls ({self._compute_entry_point()})",
                extra=ALL_RANKS,
            )
        state = self._calls
        if self._copied and (state.copied or state.compared):
            log.info(
                f"verification: {state.copied} calls copied at {DIFFUSION_BEFORE}, computed on"
                f" the copies and compared at {DIFFUSION_AFTER}: {state.compared}; rows not"
                " close: " + ", ".join(f"{k} {state.not_close.get(k, 0)}" for k in _verify.TABLES),
                extra=ALL_RANKS,
            )
            if state.pending is not None or state.compared != state.copied:
                log.warning(
                    f"verification: {state.copied - state.compared} call(s) not compared"
                    f" (no {DIFFUSION_AFTER}).",
                    extra=ALL_RANKS,
                )
        if self._timing:
            self._log_state(f"destructor{'' if self._active else ', idle'}")
        if self._active:
            self._log_timing_summary()
            self._granule.release()
            self._static.handles.clear()
            self._icon_handles.clear()
            self._verification.copies.release()
            self._descrdata.release()
            self._active = False
            if self._device_xp is not None:
                self._device_xp.get_default_memory_pool().free_all_blocks()
                self._device_xp.get_default_pinned_memory_pool().free_all_blocks()
            log.info("released the granule.")


_instance: Plugin | None = None
"""The registered plugin; a second one is refused, so that the diffusion cannot run twice."""


def instance() -> Plugin | None:
    """The plugin that ICON registered ('register'), if any (e.g. for the py2fgen probe)."""
    return _instance


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
