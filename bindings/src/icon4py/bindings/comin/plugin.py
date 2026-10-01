# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
ComIn plugin that runs icon4py's diffusion for ICON.

ICON's ComIn backend of the icon4py interface exposes the arguments of 'grid_init',
'diffusion_init' and 'diffusion_run' as ComIn variables ('_marshal.py'). The plugin
- reads ICON's namelist output in its primary constructor ('_config.py'): ICON's icon4py mode,
  as ICON's dispatcher decides it, and the configuration arguments of 'grid_init' and
  'diffusion_init'; it logs one line on what ICON and the plugin will do, and computes iff
  ICON delegates the diffusion to ComIn (icon4py_interface=1, mode SUBSTITUTE or VERIFY);
- requests the argument variables in the secondary constructor after checking them against
  the functions' 'param_descriptors', or stays idle;
- calls 'grid_init' and 'diffusion_init' at EP_ATM_TIMELOOP_BEFORE (every pass) and writes
  the pass number into their carriers; their grid, geometry and interpolation arguments come
  from ComIn's descriptive data of domain 1 ('_descrdata.py': the plugin's own copies, a fresh
  one per pass); the metrics that 'diffusion_init' gets are ICON's own variables
  (STATIC_VARIABLES: zero-copy views, on the device on the GPU, READ and DEVICE access), fetched
  in every pass at the addresses of the first one (ICON allocates them once);
- in SUBSTITUTE, calls 'diffusion_run' at EP_ATM_DYCORE_DIFFUSION_BEFORE of domain 1 on ICON's
  own variables of the same names (zero-copy views, on the device on the GPU; READ, WRITE and
  DEVICE access, so ComIn copies nothing), with the time step from ComIn's descriptive data and
  the initial-call flag from the order of the entry points (below). At EP_ATM_DIFFUSION_ENTER,
  where ICON's dispatcher binds the arguments of the same call, it compares them with what it
  used (the dual check) and writes the call count into the carrier, which ICON checks after the
  entry point;
- in VERIFY, calls 'diffusion_run' at EP_ATM_DIFFUSION_ENTER on the arguments, ICON's copies of
  the input, and writes the call count into the carrier.
It calls the undecorated functions ('__wrapped__') with the arguments py2fgen would build.

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
provide them, else 'register' stops. When active, it requests ICON's own variables that are
also arguments of 'diffusion_run' (TIMELEVEL_VARIABLES, READ, on the device) at
EP_ATM_DIFFUSION_ENTER (and, in VERIFY, at EP_ATM_DYCORE_DIFFUSION_BEFORE), and at
EP_ATM_DIFFUSION_ENTER compares the pointers ICON's variables had at
EP_ATM_DYCORE_DIFFUSION_BEFORE, where the plugin computes in SUBSTITUTE, with the argument
variables (the time-level check: ComIn must show ICON's variables at the time level the
diffusion works on), and checks that ICON's pointers did not move in between.

Every argument has a source ('SOURCES', the provider of '_dual.py'): OLD (the argument variable
or carrier of ICON's ComIn backend), NEW (a native route that the granule gets) or OBSERVE (a
native route that is computed and compared with the argument, never used). The dual check
('_dual.py') compares the NEW and OBSERVE items with the arguments that ICON binds: those of
'grid_init' and 'diffusion_init' at EP_ATM_TIMELOOP_BEFORE before the call, those of
'diffusion_run' at EP_ATM_DIFFUSION_ENTER (in SUBSTITUTE after the call that the plugin made at
EP_ATM_DYCORE_DIFFUSION_BEFORE). NEW: the configuration arguments, from ICON's namelist output
('_config.ARGUMENTS'); the arguments from ComIn's descriptive data ('_descrdata.ROUTES', among
them 'inv_dual_edge_length' as '1 / dual_edge_length' with ICON's 0 on the first row of the
lateral boundary, and the owner masks of cells, edges and vertices as 'decomp_domain == 0',
which ICON's decomposition makes them; compared bit by bit on the real entries and the padding,
in the same form, the communicator by MPI_Comm_compare); ICON's variables that 'diffusion_init' gets
(STATIC_VARIABLES; compared with the arguments: the same pointers, shapes and presence); and the
per-call arguments of 'diffusion_run' in SUBSTITUTE (ICON's variables, the time step, the
initial-call flag; compared with the arguments: the same pointers, shapes and presence, the same
scalar bits). So in SUBSTITUTE no argument comes from the old route (checked at registration).
In VERIFY the plugin computes on the arguments, and the time step and the initial-call flag are
only observed. The dual check also compares the mode with ICON's behaviour: ICON exposes the
argument variables iff icon4py_interface=1 and icon4py_mode is not OFF (secondary
constructor), and fires EP_ATM_DIFFUSION_ENTER at every diffusion call of domain 1 iff the
plugin computes (counted, compared when the plugin is released).

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
- ICON4PY_COMIN_DUAL=strict|report|off and ICON4PY_COMIN_DUAL_SELFTEST=all|<item>[,...]: the dual
  check ('_dual.py'; default strict, no self-test).
- ICON4PY_COMIN_TIMELEVEL_CHECK=report|strict|off: the time-level check (report: log each call
  and continue; strict: raise at the first call with a differing pointer; off: request
  nothing). Default: strict in SUBSTITUTE, where the arguments must be ICON's own arrays;
  report in VERIFY, where they are ICON's copies, so every pointer differs; off when idle.
- ICON4PY_COMIN_PROBE=1 (py2fgen probe): the primary constructor computes the native routes of
  PROBE_ITEMS from the descriptive data and keeps them ('Plugin.probe_values'), for a recorder
  that py2fgen runs (PY2FGEN_EXTRA_CALLABLES) to compare them with py2fgen's own arguments.

ICON's namelist output is read from the working directory of ICON (its run directory).

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
from icon4py.bindings.comin import _config, _descrdata, _diagnostics, _dual, _marshal, _views
from icon4py.tools import py2fgen


try:
    import cupy as cp  # type: ignore[import-not-found, import-untyped, unused-ignore]
except ImportError:
    cp = None

log = logging.getLogger(__name__)

CHECK_ENV: Final = "ICON4PY_COMIN_CHECK"
TIMING_ENV: Final = "ICON4PY_COMIN_TIMING"
PROFILE_ENV: Final = "ICON4PY_COMIN_PROFILE"
SKIP_ACK_ENV: Final = "ICON4PY_COMIN_TEST_SKIP_ACK"
PROBE_ENV: Final = "ICON4PY_COMIN_PROBE"
TIMELEVEL_CHECK_ENV: Final = "ICON4PY_COMIN_TIMELEVEL_CHECK"

LOG_FORMAT: Final = "icon4py-comin: %(level_prefix)s%(message)s [rank %(rank)d]"
ALL_RANKS: Final[Mapping[str, Any]] = {"all_ranks": True}
"""'extra' of a log record that every rank prints; the others are printed on rank 0 only."""

SECONDARY_CONSTRUCTOR: Final = "EP_SECONDARY_CONSTRUCTOR"
DESTRUCTOR: Final = "EP_DESTRUCTOR"
DIFFUSION_ENTER: Final = "EP_ATM_DIFFUSION_ENTER"
"""ICON's icon4py dispatcher fires it at every diffusion call iff ICON delegates to ComIn."""
DOMAIN_ENTRY_POINTS: Final = frozenset({DIFFUSION_ENTER})
"""Entry points ICON fires for a domain; there the plugin acts on domain 1 only."""
EXPECTED_ENTRY_POINTS: Final[Mapping[str, int]] = {
    "EP_FINISH": 72,
    "EP_DESTRUCTOR": 73,
    "EP_ATM_DIFFUSION_ENTER": 74,
    "EP_ATM_DIFFUSION_LEAVE": 75,
}
"""
Entry points whose numbers 'register' logs, with their values in the ComIn build this plugin
was written for (the diffusion entry points appended after EP_DESTRUCTOR).
"""
DIFFUSION_BEFORE: Final = "EP_ATM_DYCORE_DIFFUSION_BEFORE"
SOLVE_NH_BEFORE: Final = "EP_ATM_DYCORE_SOLVE_NH_BEFORE"
DYCORE_ENTRY_POINTS: Final = (
    DIFFUSION_BEFORE,
    "EP_ATM_DYCORE_DIFFUSION_AFTER",
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
TIMELEVEL_CHECK_ENTRY_POINT: Final = "EP_ATM_DIFFUSION_ENTER"
TIMELEVEL_VARIABLES: Final = (
    "vn",
    "w",
    "theta_v",
    "exner",
    "rho",
    "hdef_ic",
    "div_ic",
    "dwdx",
    "dwdy",
)
"""
ICON's variables that are also array arguments (same names) of the function at
TIMELEVEL_CHECK_ENTRY_POINT.
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
    """
    The ComIn entry point at which ICON binds its arguments. The plugin calls the function there,
    except a writing function whose per-call arguments are NEW: that one it calls at
    DIFFUSION_BEFORE and checks there.
    """
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
"""
ICON exposes this variable iff icon4py_interface=1 and icon4py_mode is not OFF (compared with
ICON's mode in the secondary constructor).
"""

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
and stops if one is not at the address of the first call.
"""


def _sources(
    klass: str, names: str, location: str | None = None, axis: int = 0, provider: str = _dual.OLD
) -> dict[str, _dual.Source]:
    return {name: _dual.Source(klass, provider, location, axis) for name in names.split()}


SOURCES: Final[Mapping[str, Mapping[str, _dual.Source]]] = {
    # class per argument (A: an ICON variable, B: ComIn's descriptive data, C: derived from it,
    # D: configuration, E: not in ComIn 1.0; '_dual.CLASSES'); 'location' marks the arrays whose
    # entries beyond the local number of cells, edges or vertices are padding
    "grid_init": {
        # NEW from ComIn's descriptive data ('_descrdata.ROUTES'): the index ranges, the
        # connectivities, the global indices, the owner masks, the counts, the communicator,
        # the geometry and the vertical coordinate table
        **_sources(
            "B",
            "cell_starts cell_ends vertex_starts vertex_ends edge_starts edge_ends vct_a",
            provider=_dual.NEW,
        ),
        **_sources(
            "B",
            "c2e c2e2c c2v c_glb_index cell_center_lat cell_center_lon cell_areas",
            "cell",
            provider=_dual.NEW,
        ),
        **_sources(
            "B",
            "e2c e2v e2c2v e_glb_index tangent_orientation primal_normal_cell_x"
            " primal_normal_cell_y edge_center_lat edge_center_lon",
            "edge",
            provider=_dual.NEW,
        ),
        **_sources("B", "v2e v2c v_glb_index", "vertex", provider=_dual.NEW),
        # 1 / primal_edge_length; 1 / dual_edge_length with ICON's 0 on the first row of the
        # lateral boundary, where ICON does not compute it ('_descrdata._inverse')
        **_sources(
            "C",
            "inverse_primal_edge_lengths inv_dual_edge_length",
            "edge",
            provider=_dual.NEW,
        ),
        **_sources(
            "E",
            "e2c2e inv_vert_vert_length edge_areas f_e primal_normal_vert_x primal_normal_vert_y"
            " dual_normal_vert_x dual_normal_vert_y dual_normal_cell_x dual_normal_cell_y"
            " primal_normal_x primal_normal_y",
            "edge",
            provider=_dual.NEW,
        ),
        # the owner masks: decomp_domain == 0, as ICON's decomposition makes them (see
        # _descrdata._owner_mask)
        **_sources("C", "c_owner_mask", "cell", provider=_dual.NEW),
        **_sources("C", "e_owner_mask", "edge", provider=_dual.NEW),
        **_sources("C", "v_owner_mask", "vertex", provider=_dual.NEW),
        **_sources(
            "D",
            "lowest_layer_thickness model_top_height stretch_factor flat_height"
            " rayleigh_damping_height backend",
            provider=_dual.NEW,
        ),
        **_sources(
            "B",
            "mean_cell_area comm_id num_vertices num_cells num_edges vertical_size limited_area",
            provider=_dual.NEW,
        ),
    },
    "diffusion_init": {
        # NEW: ICON's variables (STATIC_VARIABLES), compared by pointer, shape and presence
        **_sources("A", "theta_ref_mc wgtfac_c", "cell", provider=_dual.NEW),
        **_sources("A", "zd_cellidx zd_vertidx zd_intcoef zd_diffcoef", provider=_dual.NEW),
        # NEW from ComIn's descriptive data ('_descrdata.ROUTES'): the interpolation coefficients
        **_sources(
            "B",
            "e_bln_c_s geofac_div geofac_grg_x geofac_grg_y geofac_n2s",
            "cell",
            provider=_dual.NEW,
        ),
        **_sources("B", "nudgecoeff_e", "edge", provider=_dual.NEW),
        **_sources("B", "rbf_vec_coeff_v", "vertex", axis=2, provider=_dual.NEW),  # (6, 2, nproma)
        **_sources(
            "D",
            "ndyn_substeps diffusion_type hdiff_w hdiff_vn hdiff_smag_w zdiffu_t type_t_diffu"
            " type_vn_diffu hdiff_efdt_ratio hdiff_w_efdt_ratio smagorinski_scaling_factor"
            " smagorinski_scaling_factor2 smagorinski_scaling_factor3 smagorinski_scaling_factor4"
            " smagorinski_scaling_height smagorinski_scaling_height2 smagorinski_scaling_height3"
            " smagorinski_scaling_height4 hdiff_temp denom_diffu_v nudge_max_coeff itype_sher"
            " iforcing a_hshr loutshs backend",
            provider=_dual.NEW,
        ),
    },
    "diffusion_run": {
        # per call, at DIFFUSION_BEFORE: ICON's variables of the same names, the time step of
        # the descriptive data, the initial-call flag from the order of the entry points
        **_sources("A", "w exner theta_v rho hdef_ic div_ic dwdx dwdy", "cell", provider=_dual.NEW),
        **_sources("A", "vn", "edge", provider=_dual.NEW),
        **_sources("B", "dtime", provider=_dual.NEW),
        **_sources("E", "linit", provider=_dual.NEW),
    },
}
"""
Where each argument of FUNCTIONS comes from (the providers of the dual check, '_dual.py'). The
configuration arguments (class D) are NEW: ICON's namelist output ('_config.ARGUMENTS'). The
arguments of '_descrdata.ROUTES' are NEW: ComIn's descriptive data. The arguments of
STATIC_VARIABLES are NEW: ICON's variables. The per-call arguments of 'diffusion_run' are NEW in
SUBSTITUTE ('sources_for_mode'). So no argument is OLD in SUBSTITUTE ('old_arguments', checked at
registration).
"""

LOCAL_COUNT: Final[Mapping[str, tuple[str, str]]] = {
    "cell": ("cells", "ncells"),
    "edge": ("edges", "nedges"),
    "vertex": ("verts", "nverts"),
}
"""Descriptive data (domain) of each location and its number of local entries."""
PROBE_ROUTES: Final[Mapping[str, Callable[[Any], np.ndarray]]] = {
    name: _descrdata.domain_value("grid_init", name)
    for name in ("c2e", "cell_areas", "c_owner_mask", "e_owner_mask", "v_owner_mask")
}
"""
The native routes of the arguments of 'grid_init' that the py2fgen probe compares, from ComIn's
descriptive data of domain 1 (host arrays in py2fgen's shape and dtype).
"""
PROBE_ITEMS: Final = tuple(PROBE_ROUTES)


@dataclasses.dataclass(frozen=True)
class ProbeValue:
    """A native-route value kept for the py2fgen probe."""

    value: np.ndarray
    real: int | None
    axis: int


SCALAR_TYPES: Final[Mapping[Any, type]] = {
    py2fgen.BOOL: bool,
    py2fgen.INT32: int,
    py2fgen.INT64: int,
    py2fgen.FLOAT32: float,
    py2fgen.FLOAT64: float,
}
"""The Python type py2fgen passes for a scalar argument."""


def configuration_route(name: str, param: str, descriptor: Any) -> bool:
    """The argument is a scalar with an entry in ICON's namelist output ('_config.ARGUMENTS')."""
    return param in _config.ARGUMENTS.get(name, {}) and isinstance(
        descriptor, py2fgen.ScalarParamDescriptor
    )


def static_route(name: str, entry: FunctionEntry, param: str) -> bool:
    """The argument is an ICON variable of STATIC_VARIABLES (of a function that does not write)."""
    return not entry.inout and param in STATIC_VARIABLES.get(name, {})


def per_call_route(entry: FunctionEntry, param: str, descriptor: Any) -> bool:
    """The argument is a per-call argument of a writing function (an array, or PER_CALL_SCALARS)."""
    return entry.inout and (
        isinstance(descriptor, py2fgen.ArrayParamDescriptor) or param in PER_CALL_SCALARS
    )


def check_sources(
    functions: Mapping[str, FunctionEntry], sources: Mapping[str, Mapping[str, _dual.Source]]
) -> list[str]:
    """Every argument of every function has exactly one source, and every checked one a route."""
    errors = []
    for name, entry in functions.items():
        table = sources.get(name, {})
        descriptors = entry.exported.param_descriptors
        params = set(descriptors)
        errors += [f"{name}: no source for '{p}'." for p in sorted(params - set(table))]
        errors += [f"{name}: '{p}' is not an argument." for p in sorted(set(table) - params)]
        for param, source in table.items():
            if param not in params:
                continue
            per_call = per_call_route(entry, param, descriptors[param])
            descriptive = _descrdata.has_route(name, param)
            static = static_route(name, entry, param)
            if source.provider == _dual.NEW and not (
                per_call
                or descriptive
                or static
                or configuration_route(name, param, descriptors[param])
            ):
                errors.append(
                    f"{name}: '{param}' is NEW, but has no native route that this plugin passes"
                    " to the granule (configuration scalars from ICON's namelist output, ComIn's"
                    " descriptive data, ICON's variables of STATIC_VARIABLES, and the per-call"
                    " arguments of a writing function)."
                )
            elif source.provider == _dual.OBSERVE and not (
                descriptive or static or (per_call and param in PER_CALL_SCALARS)
            ):
                errors.append(f"{name}: '{param}' is OBSERVE, but has no native route.")
        per_call_params = [
            p for p in descriptors if p in table and per_call_route(entry, p, descriptors[p])
        ]
        arrays = [
            p for p in per_call_params if isinstance(descriptors[p], py2fgen.ArrayParamDescriptor)
        ]
        if any(table[p].provider == _dual.NEW for p in arrays):
            others = [p for p in per_call_params if table[p].provider != _dual.NEW]
            if others:
                errors.append(
                    f"{name}: its per-call arguments are NEW only all together (the plugin then"
                    f" calls it at {DIFFUSION_BEFORE}), but {', '.join(others)} are not."
                )
    return errors


def sources_for_mode(
    sources: Mapping[str, Mapping[str, _dual.Source]],
    functions: Mapping[str, FunctionEntry],
    mode: int,
) -> dict[str, dict[str, _dual.Source]]:
    """
    The source table in ICON's mode. In VERIFY the plugin computes on ICON's copies of the
    input, which only the arguments carry: the per-call arrays of a writing function come from
    the arguments, and its per-call scalars are only observed.
    """
    table = {name: dict(rows) for name, rows in sources.items()}
    if mode != _config.VERIFY:
        return table
    for name, rows in table.items():
        entry = functions[name]
        for param, source in rows.items():
            descriptor = entry.exported.param_descriptors[param]
            if source.provider != _dual.NEW or not per_call_route(entry, param, descriptor):
                continue
            is_array = isinstance(descriptor, py2fgen.ArrayParamDescriptor)
            rows[param] = dataclasses.replace(
                source, provider=_dual.OLD if is_array else _dual.OBSERVE
            )
    return table


def computes_before(entry: FunctionEntry, table: Mapping[str, _dual.Source]) -> bool:
    """The plugin calls this writing function at DIFFUSION_BEFORE: its per-call arrays are NEW."""
    return entry.inout and any(
        table[param].provider == _dual.NEW
        for param, descriptor in entry.exported.param_descriptors.items()
        if isinstance(descriptor, py2fgen.ArrayParamDescriptor)
    )


def old_arguments(sources: Mapping[str, Mapping[str, _dual.Source]]) -> list[str]:
    """The arguments that a source table takes from the old route ('function.argument')."""
    return [
        f"{name}.{param}"
        for name, table in sources.items()
        for param, source in table.items()
        if source.provider == _dual.OLD
    ]


def icon_bound_array(
    variable: Any, param: _marshal.ArrayParam, icon_name: str | None = None
) -> _marshal.BoundArray:
    """
    ICON's own variable ('icon_name', default: the argument's name) as the array argument
    'param': py2fgen's argument is the first 'rank' extents of ComIn's 5-D shape, so every other
    extent must be 1, the block axis included (one block per patch, nblks == 1). Only valid
    inside a callback that requested it.
    """
    shape = np.asarray(variable).shape  # the host buffer's ('__cuda_array_interface__' too)
    rank = param.descriptor.rank
    if len(shape) != _marshal.MAX_RANK or any(e != 1 for e in shape[rank:]):
        raise ValueError(
            f"ICON's variable '{icon_name or param.name}' has the shape {shape}: expected {rank}"
            " used extents and 1 elsewhere (one block, nblks == 1), as py2fgen's argument"
            f" '{param.name}'."
        )
    return _marshal.BoundArray(param=param, variable=variable, shape=shape[:rank], present=True)


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
class DiffusionCalls:
    """The diffusion calls of domain 1 while the plugin is active."""

    count: int = 0
    """DIFFUSION_BEFORE callbacks of domain 1 with 'lhdiff_vn': ICON's diffusion calls."""
    computed: int = 0
    """Of those, computed at DIFFUSION_BEFORE."""
    skipped: int = 0
    """DIFFUSION_BEFORE callbacks of domain 1 without 'lhdiff_vn' (no diffusion call)."""
    state: "CallState | None" = None
    """The state of the last call, until EP_ATM_DIFFUSION_ENTER takes it."""
    linit: list[tuple[int, bool, bool]] = dataclasses.field(default_factory=list)
    """Per call: its number, the derived 'linit', whether ICON's condition agrees."""


@dataclasses.dataclass
class StaticVariables:
    """ICON's variables of STATIC_VARIABLES that the plugin requested."""

    handles: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    """Per function and argument: the 'comin.var_get' handle ('None': ICON does not expose it)."""
    first: dict[str, tuple[int, dict[str, _dual.Pointer]]] = dataclasses.field(default_factory=dict)
    """Per function: the pass of the first fetch, and the pointers then."""


@dataclasses.dataclass
class CallState:
    """What the plugin derived, and in SUBSTITUTE used, at one DIFFUSION_BEFORE of domain 1."""

    call: int
    """The number of the diffusion call (DIFFUSION_BEFORE of domain 1 with 'lhdiff_vn')."""
    linit: bool
    dtime: float
    before_count: int
    """The DIFFUSION_BEFORE callbacks so far (all domains)."""
    icon_pointers: dict[str, tuple[int, int | None]] = dataclasses.field(default_factory=dict)
    """Host and device address of ICON's variables there (the time-level check)."""
    routes: dict[str, Any] = dataclasses.field(default_factory=dict)
    """The values of the per-call NEW or OBSERVE arguments (pointers of the arrays, scalars)."""
    computed: bool = False


def validated_sources(
    functions: Mapping[str, FunctionEntry], sources: Mapping[str, Mapping[str, _dual.Source]]
) -> dict[str, Mapping[str, _dual.Source]]:
    """The source table of 'functions'; raises if 'check_sources' finds problems."""
    errors = check_sources(functions, sources)
    if errors:
        raise ValueError("icon4py ComIn plugin: bad source table:\n  " + "\n  ".join(errors))
    return {name: sources[name] for name in functions}


def dual_checker(
    sources: Mapping[str, Mapping[str, _dual.Source]], environ: Mapping[str, str]
) -> _dual.Checker:
    """The dual check with the mode and the self-test items that 'environ' sets."""
    checked = {
        param
        for table in sources.values()
        for param, source in table.items()
        if source.provider in _dual.CHECKED
    }
    return _dual.Checker(
        _dual.parse_mode(environ),
        _dual.parse_selftest(environ, checked),
        functools.partial(log.info, extra=ALL_RANKS),
    )


def parse_timelevel(environ: Mapping[str, str]) -> str | None:
    """The time-level check's mode that 'environ' sets, if any."""
    timelevel = environ.get(TIMELEVEL_CHECK_ENV, "").strip()
    if timelevel and timelevel not in _dual.MODES:
        raise ValueError(
            f"{TIMELEVEL_CHECK_ENV}={timelevel!r}: expected one of {', '.join(_dual.MODES)}."
        )
    return timelevel or None


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


def _format_ms(phases: Mapping[str, float]) -> str:
    return ", ".join(f"{phase} {1e3 * seconds:.3f}" for phase, seconds in phases.items())


class Plugin:
    """The plugin's state and callbacks; 'comin' is ComIn's Python module (or a test double)."""

    def __init__(
        self,
        comin: ModuleType,
        functions: Mapping[str, FunctionEntry] = FUNCTIONS,
        environ: Mapping[str, str] = os.environ,
        sources: Mapping[str, Mapping[str, _dual.Source]] = SOURCES,
        *,
        run_dir: str | os.PathLike[str] | None = None,
        host_comm: Any = None,
    ) -> None:
        """
        'run_dir': where ICON wrote its namelist output (default: the working directory);
        'host_comm': ComIn's host communicator as an mpi4py communicator (default: from
        'comin', in 'register').
        """
        self._comin = comin
        self._functions = functions
        self._run_dir = pathlib.Path.cwd() if run_dir is None else pathlib.Path(run_dir)
        self._host_comm = host_comm
        self._mode: _config.IconMode | None = None
        """ICON's icon4py mode, from its namelist output ('register')."""
        self._configuration: dict[str, dict[str, Any]] = {}
        """The NEW configuration arguments per function, from ICON's namelist output."""
        self._hdiff_vn = True
        """ICON calls the diffusion ('lhdiff_vn' of 'diffusion_nml')."""
        # ICON's condition for the initial diffusion call ('register', if the plugin computes)
        self._initial: _config.InitialCall | None = None
        self._static_sources = validated_sources(functions, sources)
        # the source table in ICON's mode ('sources_for_mode'; set in 'register'), the writing
        # functions the plugin calls at DIFFUSION_BEFORE ('computes_before'), and ICON's
        # variables that they get there, by name
        self._sources: Mapping[str, Mapping[str, _dual.Source]] = self._static_sources
        self._before: tuple[str, ...] = ()
        self._icon_handles: dict[str, Any] = {}
        self._order = EntryPointOrder()
        self._calls = DiffusionCalls()
        self._dual = dual_checker(self._sources, environ)
        # entry points whose first call logged every item (later calls: differences only)
        self._dual_logged: set[str] = set()
        self._dual_pending: tuple[str, _dual.Summary] | None = None
        probe = environ.get(PROBE_ENV, "0").strip() or "0"
        if probe not in ("0", "1"):
            raise ValueError(f"{PROBE_ENV}={probe!r}: expected 0 or 1.")
        self._probe = probe == "1"
        self.probe_values: dict[str, ProbeValue] = {}
        """The native routes of PROBE_ITEMS, for the py2fgen probe (ICON4PY_COMIN_PROBE=1)."""
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
        self._skip_ack = environ.get(SKIP_ACK_ENV, "0") == "1"
        self._timelevel_env = parse_timelevel(environ)
        """The time-level check's mode set by the environment, else derived from ICON's mode."""
        self._timelevel = _dual.OFF
        """The time-level check's mode ('register')."""
        self._timelevel_handles: dict[str, Any] = {}
        """ICON's variables of the time-level check ('comin.var_get' handles), by name."""
        self._static = StaticVariables()  # ICON's variables of STATIC_VARIABLES
        self._timelevel_results: list[tuple[int, bool, int, int]] = []
        """Per time-level check: call, linit, number of variables checked, number identical."""
        self._ep_counts: dict[str, int] = dict.fromkeys(DYCORE_ENTRY_POINTS, 0)
        self._ep_domains: dict[str, set[int | None]] = {name: set() for name in DYCORE_ENTRY_POINTS}
        self._diffusion_calls = 0
        """DIFFUSION_BEFORE callbacks for domain 1 (one per diffusion call site that ICON passes)."""
        self._enter_counts: dict[int | None, int] = {}
        """EP_ATM_DIFFUSION_ENTER callbacks per domain (also when idle)."""
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

    @property
    def _native_only(self) -> bool:
        """
        The plugin's own source table (SOURCES): in SUBSTITUTE no argument of it may come from
        the old route (a table that a test passes may).
        """
        return all(table is SOURCES.get(name) for name, table in self._static_sources.items())

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
                " provides the diffusion entry points EP_ATM_DIFFUSION_ENTER/_LEAVE and"
                " EP_ATM_DYCORE_DIFFUSION_BEFORE/_AFTER, EP_ATM_DYCORE_SOLVE_NH_BEFORE/_AFTER."
            )
        return entry_point

    def _compute_entry_point(self) -> str:
        """The entry point at which the plugin calls the function that writes ICON's fields."""
        if self._before:
            return DIFFUSION_BEFORE
        return next((e.entry_point for e in self._functions.values() if e.inout), "-")

    def _read_configuration(self) -> None:
        """
        Read ICON's namelist output (rank 0 reads, every rank gets it): ICON's mode (logged in
        one line), the time-level check's mode and, if the plugin computes, the NEW
        configuration arguments.
        """
        comm = self._host_comm if self._host_comm is not None else _config.host_comm(self._comin)
        namelists = _config.load(comm, self._run_dir / _config.NAMELIST_FILE)
        mode = _config.icon_mode(namelists)
        self._mode = mode
        if mode.computes:
            self._sources = sources_for_mode(self._static_sources, self._functions, mode.mode)
            self._before = tuple(
                name
                for name, entry in self._functions.items()
                if computes_before(entry, self._sources[name])
            )
        log.info(mode.describe(self._compute_entry_point()))
        if not mode.computes:
            self._timelevel = _dual.OFF
            log.info(f"time-level check {self._timelevel}: the plugin is idle")
            return
        default = _dual.STRICT if mode.mode == _config.SUBSTITUTE else _dual.REPORT
        self._timelevel = self._timelevel_env or default
        log.info(
            f"time-level check {self._timelevel}: "
            + (
                f"set by {TIMELEVEL_CHECK_ENV}"
                if self._timelevel_env
                else f"the default in {mode.name}"
            )
        )
        if mode.mode == _config.SUBSTITUTE and self._native_only:
            old = old_arguments(self._sources)
            if old:
                raise RuntimeError(
                    "icon4py ComIn plugin: in SUBSTITUTE every argument must come from a native"
                    " route, but these come from ICON's argument variables: " + ", ".join(old)
                )
            log.info(
                f"{mode.name}: every argument of {', '.join(self._functions)} from a native route"
                " (ICON's namelist output, ComIn's descriptive data, ICON's variables, the entry"
                " points); none from ICON's argument variables"
            )
        errors: list[str] = []
        hdiff_vn = _config.ARGUMENTS["diffusion_init"]["hdiff_vn"]
        try:
            self._hdiff_vn = bool(_config.argument(namelists, hdiff_vn, bool))
        except _config.NamelistError as error:
            errors.append(f"lhdiff_vn: {error}")
        for name, entry in self._functions.items():
            descriptors = entry.exported.param_descriptors
            kinds = {
                param: SCALAR_TYPES[descriptors[param].dtype]
                for param, source in self._sources[name].items()
                if source.provider == _dual.NEW
                and configuration_route(name, param, descriptors[param])
            }
            if kinds:
                values, problems = _config.arguments(namelists, name, kinds)
                self._configuration[name] = values
                errors += problems
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
        for name, values in self._configuration.items():
            log.info(
                f"{name} configuration from {_config.NAMELIST_FILE}: "
                + ", ".join(f"{k} {v!r}" for k, v in values.items())
            )
        if self._initial is not None:
            log.info(
                f"initial diffusion call: {self._initial.describe()}; 'linit' from"
                f" the order of {INTEGRATE_START}, {SOLVE_NH_BEFORE} and {DIFFUSION_BEFORE},"
                " checked against ICON's condition at every call"
            )

    def register(self) -> None:
        """Primary constructor: select host or device, log the setup, register the callbacks."""
        comin = self._comin
        if comin.descrdata_get_global().has_device:
            if cp is None:
                raise RuntimeError("ICON runs on the GPU, but CuPy cannot be imported.")
            self._device_xp = cp
            self._device_flag = comin.COMIN_FLAG_DEVICE
        numbers = {name: operator.index(self._entry_point(name)) for name in EXPECTED_ENTRY_POINTS}
        dycore = {name: operator.index(self._entry_point(name)) for name in DYCORE_ENTRY_POINTS}
        self._entry_point(INTEGRATE_START)
        log.info(
            f"source sha1 {source_sha1()}, icon4py.bindings {icon4py.bindings.__version__}"
            f" at {pathlib.Path(icon4py.bindings.__file__).parent}, device {self._device_xp is not None}"
        )
        self._read_configuration()
        log.info("entry points " + ", ".join(f"{k}={v}" for k, v in numbers.items()))
        if numbers != EXPECTED_ENTRY_POINTS:
            log.warning(f"entry point numbers differ from {dict(EXPECTED_ENTRY_POINTS)}.")
        log.info(
            "counting the callbacks at "
            + ", ".join(f"{k}={v}" for k, v in dycore.items())
            + f"; time-level check {self._timelevel}"
        )
        if self._skip_ack:
            log.warning(f"{SKIP_ACK_ENV}=1, 'diffusion_run' is not acknowledged (negative test).")
        if not self._check:
            log.info(f"{CHECK_ENV}=0: in-job consistency checks are off.")
        if self._timing:
            log.info(f"{TIMING_ENV}=1: per-call phase timing is on.")
            self._log_state("primary constructor")
        if self._profile_call is not None:
            log.info(f"{PROFILE_ENV}={self._profile_call}: that call is profiled.")
        self._log_sources()
        if self._probe:
            self._keep_probe_values()

        callbacks: dict[str, Callable[[], None]] = {
            SECONDARY_CONSTRUCTOR: self.secondary_constructor
        }
        for entry in self._functions.values():
            callbacks[entry.entry_point] = functools.partial(self.on_entry_point, entry.entry_point)
        for name in DYCORE_ENTRY_POINTS:
            callbacks[name] = functools.partial(self.on_dycore_entry_point, name)
        callbacks[INTEGRATE_START] = self.on_integrate_start
        callbacks[DESTRUCTOR] = self.destructor
        for name, callback in callbacks.items():
            self._entry_point(name)(callback)  # 'comin.entry_point' registers as a decorator

    def _log_sources(self) -> None:
        """Log the source table once: counts per class and provider, and what is checked where."""
        rows = [
            (entry, param, self._sources[name][param])
            for name, entry in self._functions.items()
            for param in entry.exported.param_descriptors
        ]
        n_arrays = sum(
            isinstance(entry.exported.param_descriptors[param], py2fgen.ArrayParamDescriptor)
            for entry, param, _ in rows
        )
        classes = ", ".join(f"{k} {sum(s.klass == k for *_, s in rows)}" for k in _dual.CLASSES)
        providers = ", ".join(
            f"{p} {sum(s.provider == p for *_, s in rows)}" for p in _dual.PROVIDERS
        )
        selftest = ",".join(sorted(self._dual.selftest)) or "none"
        log.info(
            f"dual source table: {len(rows)} arguments ({n_arrays} arrays,"
            f" {len(rows) - n_arrays} scalars; {classes}); {providers};"
            f" mode {self._dual.mode}; selftest {selftest}"
        )
        for entry_point in dict.fromkeys(e.entry_point for e in self._functions.values()):
            checked = [
                (param, source.provider)
                for entry, param, source in rows
                if entry.entry_point == entry_point and source.provider in _dual.CHECKED
            ]
            counts = ", ".join(f"{p} {sum(q == p for _, q in checked)}" for p in _dual.CHECKED)
            names = ", ".join(param for param, _ in checked)
            log.info(
                f"dual plan {entry_point}: {len(checked)} items ({counts})"
                + (f": {names}" if names else "")
            )
        if self._dual.selftest and self._dual.mode != _dual.OFF:
            log.warning(
                f"{_dual.SELFTEST_ENV}: the dual check must report {selftest} as differing"
                " (self-test; the granule's input is unchanged)."
            )

    def _domain(self) -> Any:
        return self._comin.descrdata_get_domain(_marshal.DOMAIN_ID)

    def _local_count(self, source: _dual.Source) -> int | None:
        """The local number of the source's location (cells, edges, vertices) in domain 1."""
        if source.location is None:
            return None
        if source.location not in self._counts:
            kind, count = LOCAL_COUNT[source.location]
            self._counts[source.location] = int(getattr(getattr(self._domain(), kind), count))
        return self._counts[source.location]

    def _keep_probe_values(self) -> None:
        """Probe mode: compute the native routes of PROBE_ITEMS now (primary constructor)."""
        domain = self._domain()
        for name in PROBE_ITEMS:
            source = self._sources["grid_init"][name]
            self.probe_values[name] = ProbeValue(
                value=np.array(PROBE_ROUTES[name](domain), copy=True),
                real=self._local_count(source),
                axis=source.axis,
            )
        log.info(
            f"{PROBE_ENV}=1: kept the native routes of {', '.join(self.probe_values)} for the"
            " py2fgen probe.",
            extra=ALL_RANKS,
        )

    def _dual_check(
        self,
        name: str,
        entry: FunctionEntry,
        ack_value: int,
        kwargs: Mapping[str, Any],
        new: Mapping[str, Any],
    ) -> None:
        """
        Compare the NEW and OBSERVE arguments of one call with the old route ('kwargs'); 'new'
        holds the values of the NEW arguments that the granule gets.
        """
        if self._dual.mode == _dual.OFF:
            return
        checked = [
            (param, source)
            for param, source in self._sources[name].items()
            if source.provider in _dual.CHECKED
        ]
        if not checked:
            return
        descriptors = entry.exported.param_descriptors
        items = []
        for param, source in checked:
            if param not in new:
                raise RuntimeError(
                    f"icon4py ComIn plugin: '{param}' of '{name}' is {source.provider}, but the"
                    " plugin has no value of its native route."
                )
            value, reference = new[param], kwargs[param]
            real, form = None, None
            if _descrdata.has_route(name, param):
                if isinstance(descriptors[param], py2fgen.ArrayParamDescriptor):
                    # the plugin's copy: its values on the real entries, and its form as well
                    real, form = self._local_count(source), _dual.form(reference)
                elif (name, param) in _descrdata.COMMUNICATORS:
                    value, reference = _communicators(value, reference)
            items.append(
                _dual.Item(
                    name=param,
                    provider=source.provider,
                    new=value,
                    reference=_dual.host_array(reference),
                    real=real,
                    axis=source.axis,
                    form=form,
                )
            )
        counter = "pass" if entry.ack_key == _marshal.PASS_KEY else "call"
        where = f"{entry.entry_point} {counter} {ack_value}"
        summary = self._dual.check(where, items, entry.entry_point not in self._dual_logged)
        if self._dual_pending is None:
            self._dual_pending = (where, summary)
        else:
            total = self._dual_pending[1]
            for provider in _dual.CHECKED:
                total.checked[provider] += summary.checked[provider]
                total.differ[provider] += summary.differ[provider]
            total.selftest += summary.selftest

    def _mode_check(self, what: str, identical: bool) -> None:
        """Compare ICON's mode with ICON's behaviour (the dual check's mode applies)."""
        if identical:
            log.info(f"mode check: {what}: identical", extra=ALL_RANKS)
            return
        message = f"icon4py ComIn plugin: mode check: {what}: ICON's mode and behaviour differ."
        if self._dual.mode == _dual.STRICT:
            raise _dual.DualCheckError(message + f" {_dual.MODE_ENV}={_dual.REPORT} continues.")
        log.warning(message, extra=ALL_RANKS)

    def secondary_constructor(self) -> None:
        """Request and check all argument variables, or stay idle (ICON's mode decides)."""
        comin = self._comin
        mode = self.mode
        exposed = set(comin.var_list())
        has_arguments = IDLE_MARKER in exposed
        if self._dual.mode != _dual.OFF:
            self._mode_check(
                f"ICON's icon4py variables exposed {'yes' if has_arguments else 'no'},"
                f" expected {'yes' if mode.exposes_arguments else 'no'} ({mode.switches})",
                has_arguments == mode.exposes_arguments,
            )
        if not mode.computes:
            log.info(f"idle: ICON mode {mode.name} ({mode.switches}); requested nothing.")
            return
        if not has_arguments:
            raise RuntimeError(
                f"icon4py ComIn plugin: ICON mode {mode.name} delegates the diffusion to ComIn,"
                f" but ICON exposes no '{IDLE_MARKER[0]}' (ICON's icon4py variables)."
            )
        errors: list[str] = []
        read, write = comin.COMIN_FLAG_READ, comin.COMIN_FLAG_WRITE
        carrier_flags = read | write | self._device_flag
        for name, entry in self._functions.items():
            # a function the plugin calls at DIFFUSION_BEFORE only reads its argument variables
            writes = entry.inout and name not in self._before
            flags = read | (write if writes else 0) | self._device_flag
            self._bound[name] = _marshal.bind(
                comin,
                _marshal.signature(name, entry.exported),
                context=[self._entry_point(entry.entry_point)],
                flags=flags,
                carrier_flags=carrier_flags,
                exposed=exposed,
                errors=errors,
            )
        self._request_static_variables(exposed, errors)
        if self._before:
            self._request_icon_variables(exposed, errors)
        if self._timelevel != _dual.OFF:
            self._request_timelevel_variables(exposed, errors)
        if errors:
            raise RuntimeError(
                f"icon4py ComIn plugin: {len(errors)} problem(s) with ICON's icon4py variables:\n  "
                + "\n  ".join(errors)
            )
        self._active = True
        count = sum(len(bound.arrays) + 1 for bound in self._bound.values())
        log.info(f"active: requested {count} variables for {', '.join(self._functions)}.")

    def _timelevel_arrays(self) -> dict[str, _marshal.BoundArray]:
        """The argument variables of the function at TIMELEVEL_CHECK_ENTRY_POINT that ICON has."""
        return {
            array.param.name: array
            for name, entry in self._functions.items()
            if entry.entry_point == TIMELEVEL_CHECK_ENTRY_POINT and name in self._bound
            for array in self._bound[name].arrays
            if array.param.name in TIMELEVEL_VARIABLES
        }

    def _request_static_variables(self, exposed: set[tuple[str, int]], errors: list[str]) -> None:
        """
        Request ICON's variables of STATIC_VARIABLES whose arguments are NEW or OBSERVE: READ, on
        the device on the GPU (no copies), at the entry point of their function. An absent one
        must be an optional argument.
        """
        comin = self._comin
        flags = comin.COMIN_FLAG_READ | self._device_flag
        for name, entry in self._functions.items():
            wanted = [
                param
                for param, source in self._sources[name].items()
                if static_route(name, entry, param) and source.provider in _dual.CHECKED
            ]
            if not wanted:
                continue
            arrays = {a.name: a for a in self._bound[name].signature.arrays}
            context = [self._entry_point(entry.entry_point)]
            handles: dict[str, Any] = {}
            requested, absent = [], []
            for param in wanted:
                icon_name = STATIC_VARIABLES[name][param]
                label = icon_name if icon_name == param else f"{icon_name} (as {param})"
                descriptor = (icon_name, _marshal.DOMAIN_ID)
                if descriptor not in exposed:
                    if arrays[param].descriptor.is_optional:
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
            )

    def _static_arguments(self, name: str, ack_value: int) -> dict[str, tuple[Any, _dual.Pointer]]:
        """
        ICON's variables of STATIC_VARIABLES that 'name' gets, fetched now: per argument, the
        value py2fgen would build from it and its pointer (address, shape, presence). Raises if
        one is not where it was at the first call ('ack_value': the pass).
        """
        handles = self._static.handles.get(name, {})
        if not handles:
            return {}
        arrays = {a.name: a for a in self._bound[name].signature.arrays}
        values: dict[str, tuple[Any, _dual.Pointer]] = {}
        for param, variable in handles.items():
            if variable is None:
                bound = _marshal.BoundArray(arrays[param], None, (), present=False)
            else:
                bound = icon_bound_array(variable, arrays[param], STATIC_VARIABLES[name][param])
            values[param] = (_marshal.array_argument(bound, self._device_xp), self._pointer(bound))
        pointers = {param: pointer for param, (_, pointer) in values.items()}
        first, recorded = self._static.first.setdefault(name, (ack_value, pointers))
        moved = [STATIC_VARIABLES[name][p] for p, ptr in pointers.items() if ptr != recorded[p]]
        if moved:
            raise RuntimeError(
                f"icon4py ComIn plugin: ICON's variable(s) {' '.join(moved)} of '{name}' moved"
                f" between pass {first} and pass {ack_value} (address, shape or presence); the"
                " plugin expects ICON to allocate them once."
            )
        present = sum(pointer.present for pointer in pointers.values())
        log.info(
            f"{name} pass {ack_value}: ICON's variables {' '.join(handles)} fetched at"
            f" {self._functions[name].entry_point} ({present} present, {len(pointers) - present}"
            " absent); "
            + ("addresses recorded" if first == ack_value else f"addresses as at pass {first}"),
            extra=ALL_RANKS,
        )
        return values

    def _request_icon_variables(self, exposed: set[tuple[str, int]], errors: list[str]) -> None:
        """
        Request ICON's own variables that the functions called at DIFFUSION_BEFORE get, of the
        names of their array arguments: READ and WRITE, on the device on the GPU, so that ComIn
        copies nothing before or after the callback. An absent one must be optional.
        """
        comin = self._comin
        flags = comin.COMIN_FLAG_READ | comin.COMIN_FLAG_WRITE | self._device_flag
        context = [self._entry_point(DIFFUSION_BEFORE)]
        absent = []
        for name in self._before:
            for param in self._bound[name].signature.arrays:
                descriptor = (param.name, _marshal.DOMAIN_ID)
                if descriptor not in exposed:
                    if param.descriptor.is_optional:
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
        log.info(
            f"requested ICON's {' '.join(self._icon_handles) or 'nothing'} (READ | WRITE"
            f"{' | DEVICE' if self._device_flag else ''}) at {DIFFUSION_BEFORE} for"
            f" {', '.join(self._before)}"
            + (f"; not exposed (absent optional arguments): {' '.join(absent)}" if absent else "")
        )

    def _request_timelevel_variables(
        self, exposed: set[tuple[str, int]], errors: list[str]
    ) -> None:
        """
        Request ICON's variables of the time-level check (READ, on the device: no copies): at
        TIMELEVEL_CHECK_ENTRY_POINT, and at DIFFUSION_BEFORE unless the plugin computes there
        (then it has them there already).
        """
        comin = self._comin
        names = ([] if self._before else [DIFFUSION_BEFORE]) + [TIMELEVEL_CHECK_ENTRY_POINT]
        context = [self._entry_point(name) for name in names]
        flags = comin.COMIN_FLAG_READ | self._device_flag
        missing = []
        for name in self._timelevel_arrays():
            descriptor = (name, _marshal.DOMAIN_ID)
            if descriptor not in exposed:
                missing.append(name)
                continue
            try:
                self._timelevel_handles[name] = comin.var_get(list(context), descriptor, flags)
            except Exception as error:
                errors.append(f"'{name}' (time-level check): 'var_get' failed: {error}")
        log.info(
            f"time-level check {self._timelevel}: requested ICON's"
            f" {' '.join(self._timelevel_handles) or 'nothing'}"
            f" (READ{' | DEVICE' if self._device_flag else ''}) at {' and '.join(names)}"
            + (f"; not exposed: {' '.join(missing)}" if missing else "")
            + (
                f"; compares the pointers at {DIFFUSION_BEFORE}, where the plugin computes"
                if self._before
                else ""
            )
        )

    def _pointers(self, variable: Any, on_device: bool) -> tuple[int, int | None]:
        """Host and (on the GPU) device address of a ComIn variable."""
        host = _views.data_ptr(np.asarray(variable))
        if not on_device:
            return host, None
        return host, int(variable.__cuda_array_interface__["data"][0])

    def _timelevel_pointers(self) -> dict[str, tuple[int, int | None]]:
        on_device = self._device_xp is not None
        return {name: self._pointers(v, on_device) for name, v in self._timelevel_handles.items()}

    def on_integrate_start(self) -> None:
        """A new time step of the domain: restart the order of its entry points."""
        domain_id = self._domain_id()
        if domain_id is None:
            return
        self._order.integrate_start(domain_id)

    def on_dycore_entry_point(self, entry_point: str) -> None:
        """
        Count the callback; follow the order of SOLVE_NH_BEFORE; at DIFFUSION_BEFORE of domain 1
        derive the per-call state and, if the plugin computes there, compute.
        """
        self._ep_counts[entry_point] += 1
        domain_id = self._domain_id()
        self._ep_domains[entry_point].add(domain_id)
        if domain_id is None:
            return
        if entry_point == SOLVE_NH_BEFORE:
            self._order.solve_nh(domain_id)
        elif entry_point == DIFFUSION_BEFORE and domain_id == _marshal.DOMAIN_ID:
            self._diffusion_calls += 1
            if self._active:
                self._on_diffusion_before()
        elif entry_point == DIFFUSION_BEFORE and self._active and self._before:
            log.warning(
                f"{entry_point} fired for domain {domain_id}; the plugin computes domain 1 only;"
                " ignored.",
                extra=ALL_RANKS,
            )

    def _on_diffusion_before(self) -> None:
        """DIFFUSION_BEFORE of domain 1: the guards, the per-call state, the compute."""
        jg = _marshal.DOMAIN_ID
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
        if calls.state is not None:
            raise RuntimeError(
                f"icon4py ComIn plugin: diffusion call {calls.state.call} of domain 1 was not"
                f" followed by {DIFFUSION_ENTER} before the next {DIFFUSION_BEFORE}: ICON did not"
                " delegate it, so ICON's mode and the plugin's differ."
            )
        calls.count += 1
        state = CallState(
            call=calls.count,
            linit=self._derive_linit(jg, calls.count),
            dtime=float(self._comin.descrdata_get_timesteplength(jg)),
            before_count=self._ep_counts[DIFFUSION_BEFORE],
        )
        state.routes = {"dtime": state.dtime, "linit": state.linit}
        if self._before:
            self._compute(state)
        elif self._timelevel_handles:
            state.icon_pointers = self._timelevel_pointers()
        calls.state = state

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

    def _pointer(self, bound: _marshal.BoundArray) -> _dual.Pointer:
        """An array argument as the granule sees it: addresses, shape, presence."""
        if not bound.present:
            return _dual.Pointer(device=None, host=None, shape=(), present=False)
        host, device = self._pointers(bound.variable, _marshal.is_on_device(bound, self._device_xp))
        return _dual.Pointer(device=device, host=host, shape=bound.shape, present=True)

    def _native_arguments(self, name: str) -> tuple[_marshal.BoundFunction, dict[str, Any]]:
        """The array arguments of 'name' from ICON's own variables (zero-copy, py2fgen's layout)."""
        arrays = []
        for param in self._bound[name].signature.arrays:
            variable = self._icon_handles.get(param.name)
            if variable is None:  # ICON has no such variable: an absent optional argument
                arrays.append(_marshal.BoundArray(param, None, (), present=False))
            else:
                arrays.append(icon_bound_array(variable, param))
        bound = _marshal.BoundFunction(self._bound[name].signature, tuple(arrays), carrier=None)
        kwargs = {b.param.name: _marshal.array_argument(b, self._device_xp) for b in arrays}
        return bound, kwargs

    def _compute(self, state: CallState) -> None:
        """SUBSTITUTE: call the writing functions on ICON's own variables, at DIFFUSION_BEFORE."""
        # the adapter holds the GIL between callbacks: GT4Py must not compile in the background
        wrapper_config.WAIT_FOR_COMPILATION = True
        for name in self._before:
            entry = self._functions[name]
            clock = _PhaseClock(self._timing, time.perf_counter())
            clock("guard")
            bound, kwargs = self._native_arguments(name)
            state.routes.update({b.param.name: self._pointer(b) for b in bound.arrays})
            state.icon_pointers = {
                p: (r.host if r.host is not None else 0, r.device)
                for p, r in state.routes.items()
                if isinstance(r, _dual.Pointer) and r.present
            }
            kwargs.update(self._configuration.get(name, {}))
            kwargs.update(
                {
                    p: state.routes[p]
                    for p in PER_CALL_SCALARS
                    if p in entry.exported.param_descriptors
                }
            )
            clock("args")
            profiled = self._run(name, entry, bound, kwargs, call=state.call, clock=clock)
            state.computed = True
            self._calls.computed += 1
            log.info(
                f"{name} call {state.call} done at {DIFFUSION_BEFORE} (linit"
                f" {_logical(state.linit)}).",
                extra=ALL_RANKS,
            )
            clock("done")
            if clock.enabled:
                self._log_timing(name, state.call, clock.phases, profiled)

    def _timelevel_check(
        self, bound: _marshal.BoundFunction, call: int, linit: bool, state: CallState | None
    ) -> None:
        """
        The time-level check: ICON's variables at DIFFUSION_BEFORE (ComIn's view of the current
        time level; in SUBSTITUTE the arrays the plugin computed on) and the argument variables
        (exactly what py2fgen gets) point at the same memory, on the host and the device; and
        ICON's variables still point there.
        """
        now = self._timelevel_pointers()
        arrays = self._timelevel_arrays()
        on_device = self._device_xp is not None
        before_count, before = (state.before_count, state.icon_pointers) if state else (0, {})
        differ = [
            name
            for name in now
            if arrays[name].present
            and self._pointers(arrays[name].variable, on_device) != before.get(name)
        ]
        checked = [name for name in now if arrays[name].present]
        moved = [name for name in now if before.get(name) != now[name]]
        site = "initial" if linit else "regular"
        self._timelevel_results.append((call, linit, len(checked), len(checked) - len(differ)))
        log.info(
            f"time-level check call {call} {site}: {len(checked)} checked,"
            f" {len(checked) - len(differ)} identical, {len(differ)} differ"
            f"{': ' + ' '.join(differ) if differ else ''};"
            f" {DIFFUSION_BEFORE} {before_count}, EP_ATM_DYCORE_DIFFUSION_AFTER"
            f" {self._ep_counts['EP_ATM_DYCORE_DIFFUSION_AFTER']}; ICON's pointers as at"
            f" {DIFFUSION_BEFORE}: {'no: ' + ' '.join(moved) if moved else 'yes'}"
            f" [{self._timelevel}]",
            extra=ALL_RANKS,
        )
        if self._timelevel == _dual.STRICT and (differ or moved):
            raise RuntimeError(
                f"icon4py ComIn plugin: time-level check, diffusion call {call}: ComIn's ICON"
                f" variables {' '.join(differ or moved)} do not point at the arrays the diffusion"
                " works on."
            )

    def _log_counts(self) -> None:
        counts = ", ".join(f"{k} {v}" for k, v in self._ep_counts.items())
        domains = sorted({d for s in self._ep_domains.values() for d in s}, key=str)
        log.info(f"entry point callbacks: {counts} (domains {domains})", extra=ALL_RANKS)
        if self._timelevel_results:
            parts = []
            for site, linit in (("initial", True), ("regular", False)):
                rows = [r for r in self._timelevel_results if r[1] == linit]
                same = sum(r[2] == r[3] for r in rows)
                parts.append(f"{site} {len(rows)} calls, {same} all identical")
            log.info(
                f"time-level check summary: {'; '.join(parts)} [{self._timelevel}]",
                extra=ALL_RANKS,
            )
        calls = self._calls
        if calls.count or calls.skipped:
            initial = sum(derived for _, derived, _ in calls.linit)
            agree = sum(ok for *_, ok in calls.linit)
            log.info(
                f"per-call state: {calls.count} diffusion calls of domain 1 at {DIFFUSION_BEFORE},"
                f" {calls.computed} computed there, {calls.skipped} without lhdiff_vn; linit:"
                f" initial {initial}, regular {len(calls.linit) - initial}, ICON's condition"
                f" agrees at {agree} of {len(calls.linit)}; {INTEGRATE_START}"
                f" {self._order.steps.get(_marshal.DOMAIN_ID, 0)} (domain 1)",
                extra=ALL_RANKS,
            )

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
        domain_id = self._domain_id() if entry_point in DOMAIN_ENTRY_POINTS else None
        if entry_point == DIFFUSION_ENTER:
            self._enter_counts[domain_id] = self._enter_counts.get(domain_id, 0) + 1
        if not self._active:
            if entry_point in DOMAIN_ENTRY_POINTS:
                log.warning(
                    f"{entry_point} fired, but the plugin is idle; ignored.", extra=ALL_RANKS
                )
            elif self._timing:
                self._log_state(f"{entry_point}, idle")
            return
        if entry_point in DOMAIN_ENTRY_POINTS:
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
        self._dual_pending = None
        for name, entry in self._functions.items():
            if entry.entry_point == entry_point:
                self._call_or_check(name, entry, start)
        if self._dual_pending is not None:
            self._dual.log_summary(*self._dual_pending)
            self._dual_logged.add(entry_point)
        if log_state:
            self._log_state(f"{entry_point}, after")

    def _take_state(self, entry: FunctionEntry, ack_value: int) -> CallState | None:
        """
        The per-call state of a writing function's call: the one derived at the last
        DIFFUSION_BEFORE of domain 1, which must be this call of ICON's.
        """
        if not entry.inout:
            return None
        state, self._calls.state = self._calls.state, None
        if state is None or state.call != ack_value:
            last = "none" if state is None else f"call {state.call}"
            raise RuntimeError(
                f"icon4py ComIn plugin: ICON binds diffusion call {ack_value} at {DIFFUSION_ENTER},"
                f" but the plugin's last diffusion call of domain 1 at {DIFFUSION_BEFORE} is"
                f" {last}: the entry points are out of order."
            )
        return state

    def _routes(self, name: str, state: CallState | None) -> dict[str, Any]:
        """The values of the native routes of 'name' that the plugin has: the configuration
        arguments, the arguments from the descriptive data, and the per-call state of this
        call."""
        routes = dict(self._configuration.get(name, {}))
        routes.update(self._descriptive_arguments(name))
        if state is not None:
            table = self._sources[name]
            routes.update({p: v for p, v in state.routes.items() if p in table})
        return routes

    def _descriptive_arguments(self, name: str) -> dict[str, Any]:
        """
        The NEW (and, unless the dual check is off, the OBSERVE) arguments of 'name' from
        ComIn's descriptive data, as the granule gets them (a fresh copy per call). A NEW one
        that fails stops ICON; an OBSERVE one is reported by the dual check.
        """
        table = self._sources[name]
        wanted = [
            param
            for param, source in table.items()
            if _descrdata.has_route(name, param)
            and (
                source.provider == _dual.NEW
                or (source.provider == _dual.OBSERVE and self._dual.mode != _dual.OFF)
            )
        ]
        if not wanted:
            return {}
        if not self._descrdata.checked:
            log.info(self._descrdata.describe(), extra=ALL_RANKS)
        arrays = {a.name: a for a in self._bound[name].signature.arrays}
        values: dict[str, Any] = {}
        for param in wanted:
            try:
                values[param] = self._descrdata.argument(
                    name, param, arrays.get(param), self._device_xp
                )
            except Exception as error:
                if table[param].provider == _dual.NEW:
                    raise RuntimeError(
                        f"icon4py ComIn plugin: '{param}' of '{name}' from ComIn's descriptive"
                        f" data ({_descrdata.ROUTES[name][param].source}): {error}"
                    ) from error
                values[param] = functools.partial(_fail, error)  # reported by the dual check
        new = sum(table[p].provider == _dual.NEW for p in wanted)
        log.info(
            f"{name}: {len(wanted)} arguments from ComIn's descriptive data (NEW {new}, OBSERVE"
            f" {len(wanted) - new}; host copies {self._descrdata.host_bytes() / 2**20:.2f} MiB,"
            f" a fresh copy{' on the device' if self._device_xp is not None else ''} per pass)",
            extra=ALL_RANKS,
        )
        return values

    def _call(self, name: str, entry: FunctionEntry, start: float) -> None:
        """Call 'name' on its argument variables (and the NEW configuration arguments)."""
        clock = _PhaseClock(self._timing and entry.inout, start)
        clock("guard")
        bound = self._bound[name]
        ack_value = self._ack_value(bound, entry)
        clock("ack")
        state = self._take_state(entry, ack_value)
        kwargs = _marshal.arguments(self._comin, bound, self._device_xp)
        routes = self._routes(name, state)
        # ICON's variables: compared by pointer with the argument variables
        static = self._static_arguments(name, ack_value)
        reference: dict[str, Any] = dict(kwargs)
        compared = dict(routes)
        if static:
            old = {a.param.name: a for a in bound.arrays}
            reference.update({p: self._pointer(old[p]) for p in static})
            compared.update({p: pointer for p, (_, pointer) in static.items()})
            routes.update({p: value for p, (value, _) in static.items()})
        self._dual_check(name, entry, ack_value, reference, compared)
        # the NEW arguments: the granule gets the native route (OBSERVE ones are only compared)
        kwargs.update(
            {p: v for p, v in routes.items() if self._sources[name][p].provider == _dual.NEW}
        )
        if entry.entry_point == TIMELEVEL_CHECK_ENTRY_POINT and self._timelevel_handles:
            self._timelevel_check(bound, ack_value, bool(kwargs.get("linit", False)), state)
        clock("args")
        profiled = self._run(name, entry, bound, kwargs, call=ack_value, clock=clock)
        self._acknowledge(name, entry, bound, ack_value, "done.")
        clock("done")
        if clock.enabled:
            self._log_timing(name, ack_value, clock.phases, profiled)

    def _call_or_check(self, name: str, entry: FunctionEntry, start: float) -> None:
        """Call 'name' here, or check the call made at DIFFUSION_BEFORE."""
        if name in self._before:
            self._check_call(name, entry)
        else:
            self._call(name, entry, start)

    def _check_call(self, name: str, entry: FunctionEntry) -> None:
        """
        A writing function that the plugin called at DIFFUSION_BEFORE, where ICON binds its
        arguments: compare them with what the granule got (the dual check: the same pointers,
        shapes and presence; the same time step and initial-call flag), run the time-level
        check, acknowledge. Nothing is computed here.
        """
        bound = self._bound[name]
        ack_value = self._ack_value(bound, entry)
        state = self._take_state(entry, ack_value)
        if state is None or not state.computed:
            raise RuntimeError(
                f"icon4py ComIn plugin: '{name}' call {ack_value} was not computed at"
                f" {DIFFUSION_BEFORE}."
            )
        reference: dict[str, Any] = {a.param.name: self._pointer(a) for a in bound.arrays}
        metadata = _marshal.carrier_metadata(self._comin, bound)
        reference.update(
            {s.name: _marshal.scalar_value(s, metadata[s.key]) for s in bound.signature.scalars}
        )
        self._dual_check(name, entry, ack_value, reference, self._routes(name, state))
        if self._timelevel_handles:
            self._timelevel_check(bound, ack_value, state.linit, state)
        self._acknowledge(
            name,
            entry,
            bound,
            ack_value,
            f"checked at {DIFFUSION_ENTER} (computed at {DIFFUSION_BEFORE}).",
        )

    def _run(
        self,
        name: str,
        entry: FunctionEntry,
        bound: _marshal.BoundFunction,
        kwargs: Mapping[str, Any],
        *,
        call: int,
        clock: _PhaseClock,
    ) -> bool:
        """Run the undecorated function with the consistency checks; True if it was profiled."""
        first_call = name not in self._first_call_done
        if first_call:
            self._log_optional_arguments(name, bound, kwargs)
        log_values = entry.inout and self._check and first_call
        sums_before = self._value_sums(bound) if log_values else {}
        clock("sums")
        if clock.enabled:
            self._synchronize()
            clock("presync")

        profiled = entry.ack_key == _marshal.CALL_COUNT_KEY and call == self._profile_call
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

    def _acknowledge(
        self, name: str, entry: FunctionEntry, bound: _marshal.BoundFunction, value: int, what: str
    ) -> None:
        """Write the call count or the pass number into the carrier, which ICON checks."""
        acknowledge = True
        if entry.ack_key == _marshal.CALL_COUNT_KEY:
            log.info(f"{name} call {value} {what}", extra=ALL_RANKS)
            if self._skip_ack:
                log.warning(
                    f"{name} call {value} not acknowledged ({SKIP_ACK_ENV}).", extra=ALL_RANKS
                )
                acknowledge = False
        else:
            log.info(f"{name} done ({entry.ack_key} {value}).")
        if acknowledge:
            _marshal.write_carrier(bound, value)

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
        """Release the granule and CuPy's cached memory; compare the diffusion calls with the mode."""
        self._log_counts()
        if self._timing:
            self._log_state(f"destructor{'' if self._active else ', idle'}")
        if self._active:
            self._log_timing_summary()
            diffusion_wrapper.granule = None
            grid_wrapper.grid_state = None
            self._bound.clear()
            self._timelevel_handles.clear()
            self._static.handles.clear()
            self._descrdata.release()
            self._active = False
            if self._device_xp is not None:
                self._device_xp.get_default_memory_pool().free_all_blocks()
                self._device_xp.get_default_pinned_memory_pool().free_all_blocks()
            log.info("released the granule.")
        if self._mode is not None and self._dual.mode != _dual.OFF:
            # ICON calls the diffusion at every DIFFUSION_BEFORE of domain 1 iff lhdiff_vn
            enter = self._enter_counts.get(_marshal.DOMAIN_ID, 0)
            calls = self._diffusion_calls if self._hdiff_vn else 0
            expected = calls if self._mode.computes else 0
            self._mode_check(
                f"{DIFFUSION_ENTER} callbacks for domain 1: {enter}, expected {expected}"
                f" (ICON mode {self._mode.name}; {DIFFUSION_BEFORE} {self._diffusion_calls},"
                f" lhdiff_vn {'T' if self._hdiff_vn else 'F'})",
                enter == expected,
            )


def _fail(error: Exception) -> Any:
    raise error


def _communicators(new: int, reference: int) -> tuple[Any, Any]:
    """The two communicators for the dual check ('_descrdata.communicators'), or, if that fails,
    a new route that raises (reported by the dual check)."""
    try:
        return _descrdata.communicators(new, reference)
    except Exception as error:
        return functools.partial(_fail, error), reference


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
