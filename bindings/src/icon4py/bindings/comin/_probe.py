# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The py2fgen probe: compare what this plugin hands icon4py's granule, and what it builds from
it, with what py2fgen hands it and builds, item by item, in the same run.

A diagnostic, not part of the plugin's normal path: the plugin imports this module only with
ICON4PY_COMIN_PROBE=1, and only this module imports py2fgen's bindings ('grid_wrapper',
'diffusion_wrapper'). Run it for every new configuration: it is the per-field reference of the
plugin's inputs and objects.

ICON computes the diffusion through py2fgen ('icon4py_interface=0', 'icon4py_mode=1') and
loads this plugin with ICON4PY_COMIN_PROBE=1. py2fgen's glue runs the callables named in
PY2FGEN_EXTRA_CALLABLES before it imports the exported functions, so with

    PY2FGEN_EXTRA_CALLABLES=icon4py.bindings.comin._probe:install

it binds the recorders that 'install' puts in place of 'grid_init', 'diffusion_init' and
'diffusion_run': copies of the exported functions whose inner function records or compares
py2fgen's arguments, the very values the granule gets, and then calls the original function
unchanged. 'install' replaces nothing unless the plugin is registered in probe mode.

In probe mode the plugin runs no diffusion and writes nothing of ICON's. It reads ICON's
namelist output, ComIn's descriptive data and ICON's variables (READ, on the device on the GPU)
as it does when it computes, and builds its inputs where it would hand them to the granule:
- 'grid_init', 'diffusion_init': ICON calls them in 'icon4py_init', before EP_ATM_TIMELOOP_BEFORE
  of the same pass. The recorder keeps py2fgen's arguments of each call (host copies of the
  arrays, with their form and address; the scalars), and after 'diffusion_init' the objects
  py2fgen's bindings built ('object_items': the grid, its decomposition and geometry, the
  vertical grid, the granule's configuration, parameters, metric and interpolation state and
  its own fields and values). At EP_ATM_TIMELOOP_BEFORE, where the plugin builds the granule
  when it computes, it builds its inputs and compares them with the recorded call of that pass
  ('Probe.compare_init'; 23 of the 25 configuration arguments of py2fgen's 'diffusion_init'
  from the plugin's DiffusionConfig, 'DIFFUSION_CONFIG_ARGUMENTS'; 'ndyn_substeps' and
  'nudge_max_coeff' are inputs of the plugin's own); then it builds its grid and granule
  ('_granule.Granule') and compares their objects ('Probe.compare_objects').
- 'diffusion_run': at EP_ATM_DYCORE_DIFFUSION_BEFORE of domain 1 the plugin keeps what it would
  hand the granule there (ICON's variables as zero-copy views: address, form, presence; the
  time step; the initial-call flag; 'Probe.expect'). ICON then calls py2fgen's 'diffusion_run'
  and the recorder compares its arguments with them, before the granule runs.

The comparison of one item ('compare'):
- an array: the presence, the form ('_compare.form': Field or plain array, array module, dtype,
  shape, element strides), the address of the first element where the plugin hands the granule
  ICON's own memory (ICON's variables), and the values where they were copied, bit by bit, for
  an input real entries and padding reported separately ('_compare.compare_arrays'; the verdict
  is on the real entries; every entry of an object is real). Bool arrays are compared as values:
  py2fgen views Fortran's logicals without normalising them (nvfortran writes .TRUE. as 255);
  the byte values are logged.
- a scalar: the same Python type and bits ('_compare.compare_scalars'); the communicator by
  MPI_Comm_compare and its members ('communicators').
The positive control: every comparison of an identical item is repeated with a perturbed copy
of the plugin's value ('perturb'), which must be reported as differing (a differing item is
reported already).

Log lines (one per item at the first comparison of a function, later the differing ones):
  probe <function> <pass|call> <k> <item> vs py2fgen: identical|differ (<details>)
  probe <function> <pass|call> <k> control <item>: not reported (<details>)
  probe <function> <pass|call> <k>: <n> compared, <n> identical, <n> differ[: <items>];
      control <n> of <n> reported; not probed <n>[: <items>]
  probe summary <function>: py2fgen calls <n>, compared <n> (<n> items each), ...
  probe summary: <n> items compared, ... (the arguments of the three functions)
with <function> 'objects' for the objects ('probe objects pass <k>: ...'; summary 'probe
summary objects').
"""

import copy
import dataclasses
import enum
import functools
import importlib
import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np

from icon4py.bindings import diffusion_wrapper, grid_wrapper
from icon4py.bindings.comin import _arguments, _compare, _views
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.tools import py2fgen


ENV: Final = "ICON4PY_COMIN_PROBE"
INSTALL: Final = f"{__name__}:install"
"""The value of PY2FGEN_EXTRA_CALLABLES that installs the recorders."""
ABSENT: Final = "None"
"""The form of an absent array ('_compare.form(None)')."""
ALL_RANKS: Final[Mapping[str, Any]] = {"all_ranks": True}
OBJECTS: Final = "objects"
"""The name of the comparisons of the objects (as a function's in the log and the tallies)."""

EXPORTED: Final[Mapping[str, Any]] = {
    "grid_init": grid_wrapper.grid_init,
    "diffusion_init": diffusion_wrapper.diffusion_init,
    "diffusion_run": diffusion_wrapper.diffusion_run,
}
"""py2fgen's exported functions, which the recorders replace."""
SCALAR_TYPES: Final[Mapping[Any, type]] = {
    py2fgen.BOOL: bool,
    py2fgen.INT32: int,
    py2fgen.INT64: int,
    py2fgen.FLOAT32: float,
    py2fgen.FLOAT64: float,
}
"""The Python type py2fgen passes for a scalar argument."""
DIFFUSION_CONFIG_ARGUMENTS: Final[Mapping[str, str]] = {
    "diffusion_type": "diffusion_type",
    "hdiff_w": "apply_to_vertical_wind",
    "hdiff_vn": "apply_to_horizontal_wind",
    "hdiff_smag_w": "apply_smag_diff_to_vertical_wind",
    "zdiffu_t": "apply_zdiffusion_t",
    "type_t_diffu": "type_t_diffu",
    "type_vn_diffu": "type_vn_diffu",
    "hdiff_efdt_ratio": "hdiff_efdt_ratio",
    "hdiff_w_efdt_ratio": "hdiff_w_efdt_ratio",
    **{
        f"smagorinski_scaling_{kind}{i}": f"smagorinski_scaling_{kind}{i}"
        for kind in ("factor", "height")
        for i in ("", "2", "3", "4")
    },
    "hdiff_temp": "apply_to_temperature",
    "denom_diffu_v": "velocity_boundary_diffusion_denominator",
    "itype_sher": "shear_type",
    "iforcing": "iforcing",
    "a_hshr": "a_hshr",
    "loutshs": "loutshs",
}
"""
py2fgen's 'diffusion_init': its configuration arguments that set a DiffusionConfig field, and
that field (the wrapper's constructor call, inverted). 'ndyn_substeps' and 'nudge_max_coeff',
which the wrapper hands to 'Diffusion' itself, are inputs of the plugin's 'diffusion_init'.
"""
CONFIGURATIONS: Final[Mapping[str, tuple[str, Mapping[str, str]]]] = {
    "diffusion_init": ("config", DIFFUSION_CONFIG_ARGUMENTS),
}
"""The plugin's configuration object of a function and py2fgen's arguments it replaces."""
COMMUNICATORS: Final = frozenset({("grid_init", "comm_id")})
"""The arguments that are MPI communicators (Fortran handles), compared by 'communicators'."""
DIFFUSION_SHARED: Final = frozenset(
    {"_grid", "_vertical_grid", "_edge_params", "_cell_params", "_exchange"}
)
"""The granule's references to the objects of 'grid_init' (compared there)."""
DIFFUSION_RUNTIME: Final = ("_allocator", "halo_exchange_wait")
"""The granule's runtime objects (the allocator, the exchange's wait): not compared."""
MAX_DEPTH: Final = 4
"""Nesting depth of the objects' items; deeper values are not compared."""

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
    """An array as a granule gets it, reduced to what the probe compares."""

    form: str
    """'_compare.form': Field or plain array, array module, dtype, shape, element strides."""
    address: int | None
    """The address of the first element (on the device for a device array); 'None' if absent."""
    values: np.ndarray | None = None
    """A host copy (bools as values), where one was taken."""
    raw: tuple[int, ...] = ()
    """For a bool array with a copy: the byte values in its memory."""


def capture(value: Any, values: bool) -> Array:
    """An array (gtx.Field, CuPy or NumPy array, or 'None'); a host copy if 'values'."""
    if value is None:
        return Array(ABSENT, None)
    data = getattr(value, "ndarray", value)
    address = _views.data_ptr(data)
    if not values:
        return Array(_compare.form(value), address)
    host = np.array(_compare.host_array(value), copy=True)
    raw: tuple[int, ...] = ()
    if host.dtype == np.bool_:
        raw = tuple(int(b) for b in np.unique(np.ascontiguousarray(host).view(np.uint8)))
        host = host != 0
    return Array(_compare.form(value), address, host, raw)


def _address(value: Array) -> str:
    return "-" if value.address is None else f"{value.address:#x}"


def compare_arrays(
    new: Array, reference: Array, real: int | None, axis: int, same_memory: bool
) -> _compare.Comparison:
    """Presence, form, address (if 'same_memory') and copied values ('real': see '_compare')."""
    if ABSENT in (new.form, reference.form):
        if new.form == reference.form:
            return _compare.Comparison(True, "absent on both routes")
        where = "reference" if new.form == ABSENT else "new route"
        return _compare.Comparison(False, f"present only on the {where}")
    if new.form != reference.form:
        return _compare.Comparison(False, f"form {new.form} vs {reference.form}")
    parts = []
    if same_memory:
        if new.address != reference.address:
            return _compare.Comparison(
                False, f"address {_address(new)} vs {_address(reference)}; {new.form}"
            )
        parts.append(f"same address {_address(new)}")
    if new.values is None or reference.values is None:
        return _compare.Comparison(True, "; ".join([*parts, new.form]))
    result = _compare.compare_arrays(new.values, reference.values, real, axis)
    text = result.details
    if new.raw or reference.raw:
        text += f"; as values, bytes {list(new.raw)} vs {list(reference.raw)}"
    return dataclasses.replace(result, details="; ".join([*parts, text]))


def compare(
    new: Any, reference: Any, real: int | None = None, axis: int = 0, same_memory: bool = False
) -> _compare.Comparison:
    """One item: arrays by 'compare_arrays', anything else by '_compare.compare'."""
    if isinstance(new, Array) and isinstance(reference, Array):
        return compare_arrays(new, reference, real, axis, same_memory)
    if isinstance(new, Array) or isinstance(reference, Array):
        return _compare.Comparison(
            False, f"array vs value: {type(new).__name__}, {type(reference).__name__}"
        )
    return _compare.compare(new, reference)


def perturb(value: Any) -> Any:
    """
    A perturbed copy for the positive control: one value changed ('_compare.perturb'); an array
    without a copy at another address, an absent one present, an empty one in another form.
    """
    if not isinstance(value, Array):
        return _compare.perturb(value)
    if value.form == ABSENT:
        return Array(_compare.form(np.zeros(1, dtype=np.uint8)), 0)
    if value.values is not None and value.values.size:
        return dataclasses.replace(value, values=_compare.perturb(value.values))
    if value.values is None and value.address is not None:
        return dataclasses.replace(value, address=value.address + 8)
    return dataclasses.replace(value, form=f"{value.form} (perturbed)")


_RELATIONS: Final = ("IDENT", "CONGRUENT", "SIMILAR", "UNEQUAL")


def communicators(new: int, reference: int) -> tuple[_compare.Communicator, _compare.Communicator]:
    """
    The communicators of two Fortran handles: their members (ranks in MPI_COMM_WORLD) and, for
    'new', MPI_Comm_compare's result against 'reference'.
    """
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]: only inside ICON

    if not MPI.Is_initialized() or MPI.Is_finalized():
        raise RuntimeError("MPI is not initialized (ICON initializes it).")
    new_comm, reference_comm = MPI.Comm.f2py(new), MPI.Comm.f2py(reference)
    codes = {getattr(MPI, name): name for name in _RELATIONS}
    relation = codes.get(new_comm.Compare(reference_comm), "unknown")
    return (
        _compare.Communicator(new, _members(MPI, new_comm), relation),
        _compare.Communicator(reference, _members(MPI, reference_comm)),
    )


def _members(mpi: Any, comm: Any) -> tuple[int, ...]:
    group, world = comm.Get_group(), mpi.COMM_WORLD.Get_group()
    try:
        ranks = group.Translate_ranks(list(range(group.Get_size())), world)
        return tuple(int(r) for r in ranks)
    finally:
        group.Free()
        world.Free()


# ---- the objects -------------------------------------------------------------------------------


def _flatten(
    items: dict[str, Any], skipped: list[str], prefix: str, value: Any, depth: int = 0
) -> None:
    """
    Add 'value' as items: arrays and Fields as host copies ('capture'), scalars, strings and
    'None' as they are, a non-integer enumeration member by its name; dictionaries, sequences
    and dataclasses by their entries (up to MAX_DEPTH). Anything else (programs, runtime
    objects) is listed in 'skipped'.
    """
    if depth > MAX_DEPTH:
        skipped.append(f"{prefix} ({type(value).__name__}, nested deeper than {MAX_DEPTH})")
    elif value is None or isinstance(value, (bool, int, float, str, np.generic)):
        items[prefix] = value
    elif isinstance(value, enum.Enum):
        items[prefix] = f"{type(value).__name__}.{value.name}"
    elif (
        hasattr(value, "ndarray")
        or isinstance(value, np.ndarray)
        or hasattr(value, "__cuda_array_interface__")
    ):
        items[prefix] = capture(value, values=True)
    elif isinstance(value, dict):
        for key, entry in value.items():
            _flatten(items, skipped, f"{prefix}.{key}", entry, depth + 1)
    elif isinstance(value, (tuple, list)):
        for i, entry in enumerate(value):
            _flatten(items, skipped, f"{prefix}[{i}]", entry, depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _flatten(
                items, skipped, f"{prefix}.{field.name}", getattr(value, field.name), depth + 1
            )
    else:
        skipped.append(f"{prefix} ({type(value).__name__})")


def object_items(grid_state: Any, diffusion: Any) -> tuple[dict[str, Any], list[str]]:
    """
    The objects of 'grid_init' ('grid_state': the grid's identity, configuration,
    connectivities and index ranges, the vertical grid, the edge and cell geometry, the
    exchange and its decomposition) and of 'diffusion_init' (the granule's attributes but its
    references to the former), as items; and what was not compared.
    """
    items: dict[str, Any] = {}
    skipped: list[str] = []
    if grid_state is not None:
        grid = grid_state.grid
        items["grid.id"] = grid.id
        _flatten(items, skipped, "grid.config", grid.config)
        for offset in sorted(grid.connectivities):
            connectivity = grid.connectivities[offset]
            _flatten(items, skipped, f"grid.connectivity.{offset}", connectivity)
            items[f"grid.connectivity.{offset}.skip_value"] = getattr(
                connectivity, "skip_value", None
            )
        for dim in dims.horizontal_dims():
            for domain in h_grid.get_domains_for_dim(dim):
                where = f"{dim.value}.{getattr(domain.zone, 'name', domain.zone)}"
                items[f"grid.start_index.{where}"] = grid.start_index(domain)
                items[f"grid.end_index.{where}"] = grid.end_index(domain)
        _flatten(items, skipped, "vertical_grid", dict(vars(grid_state.vertical_grid)))
        _flatten(items, skipped, "edge_geometry", dict(vars(grid_state.edge_geometry)))
        _flatten(items, skipped, "cell_geometry", grid_state.cell_geometry)
        exchange = grid_state.exchange_runtime
        items["exchange.type"] = type(exchange).__name__
        for method in ("get_size", "my_rank"):
            if callable(getattr(exchange, method, None)):
                items[f"exchange.{method}"] = int(getattr(exchange, method)())
        info = getattr(exchange, "_decomposition_info", None)
        if info is not None:
            for dim in dims.horizontal_dims():
                _flatten(
                    items, skipped, f"exchange.global_index.{dim.value}", info.global_index(dim)
                )
                _flatten(items, skipped, f"exchange.owner_mask.{dim.value}", info.owner_mask(dim))
    if diffusion is not None:
        attributes = vars(diffusion)
        skipped += [
            f"diffusion.{k} ({type(attributes[k]).__name__})"
            for k in DIFFUSION_RUNTIME
            if k in attributes
        ]
        _flatten(
            items,
            skipped,
            "diffusion",
            {
                k: v
                for k, v in attributes.items()
                if k not in DIFFUSION_SHARED and k not in DIFFUSION_RUNTIME
            },
        )
    return items, skipped


# ---- the probe ---------------------------------------------------------------------------------


@dataclasses.dataclass
class Tally:
    """The comparisons of one function (or of the objects)."""

    calls: int = 0
    """py2fgen's calls of the function (seen by the recorder); for the objects, the records."""
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
    """What the probe needs to know of one function of the plugin."""

    params: Mapping[str, _arguments.Param]
    """The plugin's inputs of the function ('_granule.PARAMS')."""
    same_memory: frozenset[str]
    """The inputs that are ICON's own memory (ICON's variables): compared by address too."""
    per_call: bool
    """Compared at py2fgen's call (else recorded there and compared by 'compare_init')."""
    entry_point: str
    """Where the plugin builds its inputs."""


@dataclasses.dataclass(frozen=True)
class _Spec:
    """How one item is compared."""

    real: int | None = None
    axis: int = 0
    same_memory: bool = False
    communicator: bool = False


class Probe:
    """
    The records of py2fgen's calls and objects and the plugin's inputs and objects, their
    comparisons and counts. 'real(location)' is the local number of entries of a location
    ('cell', 'edge', 'vertex'; else 'None'); 'synchronize' waits for the device.
    """

    def __init__(
        self,
        functions: Mapping[str, Function],
        real: Callable[[str | None], int | None],
        synchronize: Callable[[], None] = lambda: None,
    ) -> None:
        self._functions = functions
        self._real = real
        self._synchronize = synchronize
        self._records: dict[str, list[dict[str, Any]]] = {name: [] for name in functions}
        self._objects: list[dict[str, Any]] = []
        self._expected: dict[str, tuple[int, dict[str, Any]]] = {}
        self.tallies: dict[str, Tally] = {name: Tally() for name in [*functions, OBJECTS]}
        self.installed: list[str] = []

    # --- py2fgen's side (the recorders)
    def install(self) -> None:
        """Replace each exported function in its module by a recorder (once)."""
        if self.installed:
            log.warning("py2fgen probe: installed already; nothing replaced", extra=ALL_RANKS)
            return
        for name in self._functions:
            module = importlib.import_module(EXPORTED[name].__module__)
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
            if name == "diffusion_init":
                try:
                    self.record_objects()
                except Exception as error:
                    tally = self.tallies[OBJECTS]
                    tally.not_probed[f"record {tally.calls + 1}"] = repr(error)
                    log.error(f"py2fgen probe: objects: {error!r}", extra=ALL_RANKS)

        return recorder

    def on_call(self, name: str, kwargs: Mapping[str, Any]) -> None:
        """py2fgen calls 'name' with 'kwargs' (its converted arguments)."""
        tally = self.tallies[name]
        tally.calls += 1
        function = self._functions[name]
        references = self._references(name, kwargs, values=not function.per_call)
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

    def record_objects(self) -> None:
        """After py2fgen's 'diffusion_init': the objects py2fgen's bindings built (host copies)."""
        granule = diffusion_wrapper.granule
        self._synchronize()
        items, _ = object_items(
            grid_wrapper.grid_state, None if granule is None else granule.diffusion
        )
        self._objects.append(items)
        self.tallies[OBJECTS].calls += 1
        log.info(
            f"probe {OBJECTS} record {len(self._objects)}: recorded the {len(items)} items of the"
            " objects py2fgen's grid_init and diffusion_init built",
            extra=ALL_RANKS,
        )

    def _references(self, name: str, kwargs: Mapping[str, Any], values: bool) -> dict[str, Any]:
        descriptors = EXPORTED[name].param_descriptors
        return {
            param: capture(value, values)
            if isinstance(descriptors[param], py2fgen.ArrayParamDescriptor)
            else value
            for param, value in kwargs.items()
        }

    # --- the plugin's side
    def natives(self, name: str, arguments: Mapping[str, Any], values: bool) -> dict[str, Any]:
        """
        The plugin's inputs of 'name' as py2fgen's arguments, in the form the probe compares: a
        configuration object as the arguments it replaces (CONFIGURATIONS).
        """
        params = self._functions[name].params
        configuration, fields = CONFIGURATIONS.get(name, ("", {}))
        descriptors = EXPORTED[name].param_descriptors
        natives: dict[str, Any] = {}
        for param, value in arguments.items():
            if param == configuration:
                natives.update(
                    {
                        argument: SCALAR_TYPES[descriptors[argument].dtype](getattr(value, field))
                        for argument, field in fields.items()
                    }
                )
            elif isinstance(params.get(param), _arguments.ArrayParam):
                natives[param] = capture(value, values)
            else:
                natives[param] = value
        return natives

    def compare_init(self, name: str, k: int, natives: Mapping[str, Any]) -> None:
        """Pass 'k': compare the plugin's inputs with py2fgen's call 'k' of 'name'."""
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

    def compare_objects(self, k: int, grid_state: Any, diffusion: Any) -> None:
        """Pass 'k': compare the plugin's objects with those py2fgen's bindings built."""
        tally = self.tallies[OBJECTS]
        if len(self._objects) < k:
            tally.not_probed[f"pass {k}"] = f"{len(self._objects)} record(s) of py2fgen's objects"
            log.warning(
                f"probe {OBJECTS} pass {k}: {len(self._objects)} record(s) of py2fgen's objects;"
                " not probed",
                extra=ALL_RANKS,
            )
            return
        self._synchronize()
        natives, skipped = object_items(grid_state, diffusion)
        if tally.compared == 0:
            log.info(
                f"probe {OBJECTS} pass {k}: {len(natives)} items of the plugin's objects; not"
                f" compared (programs and runtime objects): {', '.join(skipped) or 'nothing'}",
                extra=ALL_RANKS,
            )
        references = self._objects[k - 1]
        names = list(dict.fromkeys([*references, *natives]))
        self._compare_items(OBJECTS, "pass", k, names, natives, references, lambda name: _Spec())
        self._objects[k - 1] = {}

    def not_probed(self, function: str, key: str, reason: str) -> None:
        """Record what could not be compared ('function': a function or OBJECTS)."""
        self.tallies[function].not_probed[key] = reason
        log.error(f"probe {function} {key}: not probed: {reason}", extra=ALL_RANKS)

    def expect(self, name: str, call: int, natives: Mapping[str, Any]) -> None:
        """The plugin's inputs of call 'call' of 'name', which py2fgen's next call must get."""
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
        """After the call site: py2fgen must have taken the inputs kept for it."""
        expected = self._expected.pop(name, None)
        if expected is not None:
            self.tallies[name].not_probed[f"call {expected[0]}"] = "py2fgen made no call"
            log.warning(
                f"probe {name} call {expected[0]}: py2fgen made no call at this call site; not"
                " probed",
                extra=ALL_RANKS,
            )

    def _spec(self, name: str, param: str) -> _Spec:
        function = self._functions[name]
        array = function.params.get(param)
        location, axis = (
            (array.location, array.axis) if isinstance(array, _arguments.ArrayParam) else (None, 0)
        )
        return _Spec(
            real=self._real(location),
            axis=axis,
            same_memory=param in function.same_memory,
            communicator=(name, param) in COMMUNICATORS,
        )

    def _compare(
        self,
        name: str,
        counter: str,
        k: int,
        natives: Mapping[str, Any],
        references: Mapping[str, Any],
    ) -> None:
        names = list(EXPORTED[name].param_descriptors)
        self._compare_items(
            name, counter, k, names, natives, references, functools.partial(self._spec, name)
        )

    def _compare_items(  # noqa: PLR0917 [too-many-positional-arguments]
        self,
        name: str,
        counter: str,
        k: int,
        names: list[str],
        natives: Mapping[str, Any],
        references: Mapping[str, Any],
        spec_of: Callable[[str], _Spec],
    ) -> None:
        tally = self.tallies[name]
        first = tally.compared == 0
        tally.compared += 1
        where = f"{name} {counter} {k}"
        differ, missing, unreported = [], [], []
        compared = 0
        for param in names:
            if param not in natives or param not in references:
                side = "the plugin" if param not in natives else "py2fgen"
                missing.append(param)
                tally.not_probed[f"{counter} {k} {param}"] = f"no value from {side}"
                log.warning(
                    f"probe {where} {param}: no value from {side}; not probed", extra=ALL_RANKS
                )
                continue
            spec = spec_of(param)
            new, reference = natives[param], references[param]
            failure = None
            if spec.communicator:
                try:
                    new, reference = communicators(new, reference)
                except Exception as error:
                    failure = f"MPI_Comm_compare failed: {error!r}"
            copied = isinstance(reference, Array) and reference.values is not None
            real = spec.real if copied else None  # the values' real entries
            if failure is not None:
                result = _compare.Comparison(False, failure)
            else:
                result = compare(new, reference, real, spec.axis, spec.same_memory)
            # the positive control: a perturbed copy of the plugin's value must be reported; an
            # item that differs is reported already (its perturbed copy may even match)
            control = (
                compare(perturb(new), reference, real, spec.axis, spec.same_memory)
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
        """Log the counts per function, of the objects, and of the arguments in total."""
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
        tallies = [t for name, t in self.tallies.items() if name != OBJECTS]
        items = sum(t.items for t in tallies)
        identical = sum(t.identical for t in tallies)
        reported = sum(t.reported for t in tallies)
        controls = sum(t.controls for t in tallies)
        not_probed = sum(len(t.not_probed) for t in tallies)
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
