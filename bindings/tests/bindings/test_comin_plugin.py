# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the icon4py ComIn plugin against a fake 'comin' module and a fake ICON side.

'FakeComIn' mimics the parts of ComIn's Python API that the plugin uses (ComIn's
'plugins/python_adapter/comin.py'), including the access-scope rule and the exception of
'current_get_domain_id' outside the domain loop. 'FakeIcon' does what ICON does with
icon4py_interface=1: expose its variables, fire the secondary constructor and
EP_ATM_TIMELOOP_BEFORE (once per pass), and around a diffusion call what 'mo_nh_stepping'
fires: EP_ATM_INTEGRATE_START at the start of a time step, the dynamics'
EP_ATM_DYCORE_SOLVE_NH_BEFORE/_AFTER before a regular call, and the pair
EP_ATM_DYCORE_DIFFUSION_BEFORE/_AFTER around ICON's dispatcher, which runs ICON's own diffusion
unless in SUBSTITUTE.
ICON's namelist output is an excerpt of a real run's ('comin_run_dir/'; ComIn diffusion
SUBSTITUTE of 'mch_icon-ch1_small'), copied into each test's working directory.
"""

import collections
import io
import logging
import pathlib
import re
import shutil
import sys
import types
import typing
from typing import Any

import cffi
import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import (
    config as wrapper_config,
    diffusion_wrapper,
    grid_wrapper,
    icon4py_export,
)
from icon4py.bindings.comin import _arguments, _config, _descrdata, _granule, _views, plugin
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.tools import py2fgen


FLAG_READ, FLAG_WRITE, FLAG_SYNC_HALO, FLAG_DEVICE = 1 << 1, 1 << 2, 1 << 3, 1 << 4
DOUBLE, FLOAT, INT = 0, 1, 2
ENTRY_POINTS = {
    "EP_SECONDARY_CONSTRUCTOR": 0,
    "EP_ATM_TIMELOOP_BEFORE": 8,
    "EP_ATM_INTEGRATE_START": 10,
    "EP_ATM_DYCORE_DIFFUSION_BEFORE": 20,
    "EP_ATM_DYCORE_DIFFUSION_AFTER": 21,
    "EP_ATM_DYCORE_SOLVE_NH_BEFORE": 22,
    "EP_ATM_DYCORE_SOLVE_NH_AFTER": 23,
    "EP_FINISH": 72,
    "EP_DESTRUCTOR": 73,
}
"""Entry point numbers of the ComIn build of these tests (the plugin uses only the names)."""
INIT, BEFORE, AFTER = (
    "EP_ATM_TIMELOOP_BEFORE",
    "EP_ATM_DYCORE_DIFFUSION_BEFORE",
    "EP_ATM_DYCORE_DIFFUSION_AFTER",
)


# ---- fake ComIn ------------------------------------------------------------------------------


class FakeComInError(RuntimeError):
    pass


class FakeEntryPoint:
    def __init__(self, comin: "FakeComIn", number: int):
        self._comin = comin
        self._number = number

    def __index__(self) -> int:
        return self._number

    def __call__(self, fun):
        self._comin.callbacks.setdefault(self._number, []).append(fun)
        return fun


class FakeVariable:
    def __init__(self, comin: "FakeComIn", descriptor: tuple[str, int]):
        self._comin = comin
        self.descriptor = descriptor

    def __array__(self, dtype=None, copy=None):
        # like '_var_get_buffer': a writable PEP 3118 view of ICON's (host) memory
        return np.asarray(memoryview(self._comin.get_ptr(self)))

    @property
    def __cuda_array_interface__(self):
        # like 'comin.variable': shape and strides of the host buffer, ICON's device pointer
        host = self._comin.get_ptr(self)
        device = self._comin.device_buffers.get(self.descriptor)
        return {
            "shape": host.shape,
            "typestr": host.dtype.str,
            "data": (0 if device is None else device.data.ptr, False),
            "version": 3,
            "strides": host.strides,
        }


class FakeComIn:
    COMIN_FLAG_READ, COMIN_FLAG_WRITE = FLAG_READ, FLAG_WRITE
    COMIN_FLAG_SYNC_HALO, COMIN_FLAG_DEVICE = FLAG_SYNC_HALO, FLAG_DEVICE
    COMIN_VAR_DATATYPE_DOUBLE, COMIN_VAR_DATATYPE_FLOAT, COMIN_VAR_DATATYPE_INT = DOUBLE, FLOAT, INT

    def __init__(self, has_device: bool = False, entry_points: dict[str, int] = ENTRY_POINTS):
        self.buffers: dict[tuple[str, int], np.ndarray | None] = {}
        self.device_buffers: dict[tuple[str, int], Any] = {}
        self.meta: dict[tuple[str, int], dict[str, Any]] = {}
        self.callbacks: dict[int, list] = {}
        self.contexts: dict[tuple[int, tuple[str, int]], int] = {}
        self.accessed: list[tuple[int, tuple[str, int]]] = []
        self.current_ep: int | None = None
        self.domain_id = -1
        self.has_device = has_device
        self.lrestartrun = False
        self.timestep = 0.0
        """ICON's time step of domain 1 ('comin_descrdata_get_timesteplength')."""
        self.domain: Any = None
        """Domain 1's descriptive data ('comin_descrdata_get_domain'), where a test needs it."""
        for name, number in entry_points.items():
            setattr(self, name, FakeEntryPoint(self, number))

    # --- host side (what ICON does through comin_host_interface)
    def add(
        self, name: str, buffer: np.ndarray | None, device: Any = None, **metadata: Any
    ) -> None:
        self.buffers[name, 1] = buffer
        if device is not None:
            self.device_buffers[name, 1] = device
        self.meta[name, 1] = dict(metadata)

    def fire(self, name: str, domain_id: int = -1) -> None:
        number = ENTRY_POINTS[name]
        self.current_ep, self.domain_id = number, domain_id
        try:
            for fun in self.callbacks.get(number, []):
                fun()
        finally:
            self.current_ep, self.domain_id = None, -1

    def get_ptr(self, variable: FakeVariable) -> np.ndarray:
        key = (self.current_ep, variable.descriptor)
        if key not in self.contexts:
            raise FakeComInError("Access out of requested scope.")
        self.accessed.append(key)
        buffer = self.buffers[variable.descriptor]
        if buffer is None:
            raise ValueError("comin_var_get_ptr failed")
        return buffer

    # --- the plugin's API
    def descrdata_get_global(self):
        return types.SimpleNamespace(has_device=self.has_device, lrestartrun=self.lrestartrun)

    def descrdata_get_timesteplength(self, jg: int) -> float:
        assert jg == 1
        return self.timestep

    def descrdata_get_domain(self, jg: int):
        assert jg == 1 and self.domain is not None
        return self.domain

    def parallel_get_host_mpi_rank(self) -> int:
        return 0

    def current_get_domain_id(self) -> int:
        if self.domain_id < 0:
            raise RuntimeError(f"comin_current_get_domain_id failed. (jg={self.domain_id})")
        return self.domain_id

    def var_list(self):
        yield from list(self.buffers)

    def metadata(self, descriptor: tuple[str, int]) -> dict[str, Any]:
        if descriptor not in self.meta:
            raise FakeComInError(f"Invalid handle: {descriptor}")
        return self.meta[descriptor]

    def var_get(self, context: list, descriptor: tuple[str, int], flag: int) -> FakeVariable:
        if self.current_ep != ENTRY_POINTS["EP_SECONDARY_CONSTRUCTOR"]:
            raise FakeComInError("Not inside secondary constructor.")
        if not isinstance(context, list) or descriptor not in self.buffers:
            raise FakeComInError(f"Invalid handle: {descriptor}")
        for entry_point in context:
            key = (int(entry_point.__index__()), descriptor)
            if self.contexts.get(key, flag) != flag:
                raise FakeComInError("Inconsistent access flags.")
            self.contexts[key] = flag
        return FakeVariable(self, descriptor)


# ---- fake ICON --------------------------------------------------------------------------------

MAX_RANK = _arguments.MAX_RANK


def fortran_buffer(shape: tuple[int, ...], dtype, fill=0) -> np.ndarray:
    """A 5-D Fortran-order buffer, like a Fortran array padded to ComIn's 5 dimensions."""
    padded = shape + (1,) * (MAX_RANK - len(shape))
    buffer = np.zeros(padded, dtype=dtype, order="F")
    rank_view(buffer, len(shape))[...] = fill
    return buffer


def rank_view(buffer: Any, rank: int) -> Any:
    return buffer[(slice(None),) * rank + (0,) * (MAX_RANK - rank)]


class FakeIcon:
    """
    ICON with icon4py_interface=1 in miniature: its variables, its passes, and per diffusion
    call the entry points of 'mo_nh_stepping' around ICON's dispatcher, which runs ICON's own
    diffusion ('fortran': 'w' += fortran_w * dtime, 'vn' *= 2) in OFF and VERIFY, and nothing
    in SUBSTITUTE ('mode': ICON's mode).
    """

    def __init__(
        self, comin: FakeComIn, device_xp: Any = None, mode: int = 1, fortran_w: float = 1.0
    ):
        self.comin = comin
        self.device_xp = device_xp
        self.mode = mode
        self.fortran_w = fortran_w
        self.after_initial = False
        self.fortran_calls = 0

    def add(self, name: str, buffer: np.ndarray) -> Any:
        """Expose ICON's variable 'name' (on the GPU also its device copy); ICON's array."""
        device = None if self.device_xp is None else self.device_xp.array(buffer, order="F")
        datatype = DOUBLE if buffer.dtype == np.float64 else INT
        self.comin.add(name, buffer, device, datatype=datatype)
        return buffer if device is None else device

    def array(self, name: str) -> Any:
        """ICON's array of a variable (on the device on the GPU)."""
        device = self.comin.device_buffers.get((name, 1))
        return self.comin.buffers[name, 1] if device is None else device

    def move(self, name: str, buffer: np.ndarray) -> Any:
        """ICON re-targets its variable to other memory (another time level); ICON's array."""
        assert buffer.shape == self.comin.buffers[name, 1].shape
        self.comin.buffers[name, 1] = buffer
        if self.device_xp is not None:
            self.comin.device_buffers[name, 1] = self.device_xp.array(buffer, order="F")
        return self.array(name)

    def host(self, array: Any) -> np.ndarray:
        return np.asarray(array) if self.device_xp is None else self.device_xp.asnumpy(array)

    def secondary_constructor(self) -> None:
        self.comin.fire("EP_SECONDARY_CONSTRUCTOR")

    def new_pass(self) -> None:
        self.comin.fire(INIT)

    def dynamics(self, domain_id: int = 1, substeps: int = 2) -> None:
        """'perform_dyn_substepping': the SOLVE_NH pair per substep."""
        for _ in range(substeps):
            self.comin.fire("EP_ATM_DYCORE_SOLVE_NH_BEFORE", domain_id)
            self.comin.fire("EP_ATM_DYCORE_SOLVE_NH_AFTER", domain_id)

    def diffusion_call(
        self, dtime: float, linit: bool, domain_id: int = 1, *, lhdiff_vn: bool = True
    ) -> None:
        """
        One diffusion call as 'mo_nh_stepping' makes it: the initial one ('linit') opens a time
        step (INTEGRATE_START); a regular one follows the dynamics, in a new time step unless it
        follows the initial one. Then the pair DIFFUSION_BEFORE/_AFTER around the dispatcher,
        which ICON calls only with 'lhdiff_vn' (the regular pair fires anyway).
        """
        if linit or not self.after_initial:
            self.comin.fire("EP_ATM_INTEGRATE_START", domain_id)
        if not linit:
            self.dynamics(domain_id)
        self.after_initial = linit
        self.comin.timestep = dtime
        self.comin.fire(BEFORE, domain_id)
        if lhdiff_vn:
            self.dispatch(dtime)
        self.comin.fire(AFTER, domain_id)

    def dispatch(self, dtime: float) -> None:
        """ICON's dispatcher with icon4py_interface=1: its own diffusion unless SUBSTITUTE."""
        if self.mode == 1:
            return
        self.fortran_calls += 1
        if ("w", 1) in self.comin.buffers:
            self.array("w")[...] += self.fortran_w * dtime
        if ("vn", 1) in self.comin.buffers:
            self.array("vn")[...] *= 2.0


# ---- a toy granule: every kind of input, cheap to call ----------------------------------------

CALLS: list[tuple[str, dict[str, Any]]] = []


class ToyGranule:
    """The granule's functions in miniature: they record their inputs; 'diffusion_run' writes."""

    def __init__(self) -> None:
        self.released = False

    def diffusion_init(self, **kwargs: Any) -> None:
        CALLS.append(("init", kwargs))

    def diffusion_run(self, **kwargs: Any) -> None:
        CALLS.append(("run", kwargs))
        kwargs["w"].ndarray[...] += kwargs["dtime"]
        kwargs["vn"].ndarray[...] *= 2.0

    def release(self) -> None:
        self.released = True


F64, I32 = np.dtype(np.float64), np.dtype(np.int32)
TOY_INIT: dict[str, _arguments.Param] = {
    "theta_ref_mc": _arguments.ArrayParam(
        "theta_ref_mc", F64, 2, (dims.CellDim, dims.KDim), location="cell"
    ),
    "zd_cellidx": _arguments.ArrayParam("zd_cellidx", I32, 2, optional=True),
    "zd_diffcoef": _arguments.ArrayParam("zd_diffcoef", F64, 1, optional=True),
    "config": diffusion.DiffusionConfig,
    "backend": int,
}
TOY_RUN: dict[str, _arguments.Param] = {
    "w": _arguments.ArrayParam("w", F64, 2, (dims.CellDim, dims.KDim), location="cell"),
    "vn": _arguments.ArrayParam("vn", F64, 2, (dims.EdgeDim, dims.KDim), location="edge"),
    "opt": _arguments.ArrayParam(
        "opt", F64, 2, (dims.CellDim, dims.KDim), optional=True, location="cell"
    ),
    "dtime": float,
    "linit": bool,
}
TOY_FUNCTIONS = {
    # the names must be the real ones: they select the routes ('_config.ARGUMENTS',
    # 'plugin.CONFIGURATIONS', 'plugin.STATIC_VARIABLES')
    "diffusion_init": plugin.FunctionEntry(TOY_INIT, INIT, inout=False),
    "diffusion_run": plugin.FunctionEntry(TOY_RUN, BEFORE, inout=True),
}
"""The toy functions' inputs: ICON's variables (static and per call), the configuration."""
NC, NE, NLEV, NPOINTS = 6, 9, 4, 3
REAL_CELLS, REAL_EDGES = NC - 1, NE - 2
"""The toy domain's real cells and edges (the rest of the first block is padding)."""
STATIC_NAMES = {
    "theta_ref_mc": "theta_ref_mc",
    "zd_cellidx": "zd_indlist",
    "zd_diffcoef": "zd_diffcoef",
}


def toy_icon(
    comin: FakeComIn,
    device_xp: Any = None,
    mode: int = 1,
    *,
    zdiffu: bool = True,
    fortran_w: float = 1.0,
    w: Any = 1.0,
    vn: Any = 3.0,
) -> tuple[FakeIcon, dict[str, Any]]:
    """
    ICON's variables of the toy functions ('w' and 'vn' filled with 'w' and 'vn'; without
    'zdiffu' no 'zd_*' lists: l_zdiffu_t=.FALSE.) and domain 1's counts; ICON's arrays by name.
    """
    icon = FakeIcon(comin, device_xp, mode, fortran_w)
    buffers = {"theta_ref_mc": fortran_buffer((NC, NLEV), np.float64, fill=290.0)}
    if zdiffu:
        buffers["zd_indlist"] = fortran_buffer((4, NPOINTS), np.int32, fill=2)
        buffers["zd_diffcoef"] = fortran_buffer((NPOINTS,), np.float64, fill=0.5)
    buffers["w"] = fortran_buffer((NC, NLEV + 1), np.float64, fill=w)
    buffers["vn"] = fortran_buffer((NE, NLEV), np.float64, fill=vn)
    live = {name: icon.add(name, buffer) for name, buffer in buffers.items()}
    ns = types.SimpleNamespace
    comin.domain = ns(
        cells=ns(ncells=REAL_CELLS, end_index=np.array([REAL_CELLS, 1, 1])),
        edges=ns(nedges=REAL_EDGES, end_index=np.array([REAL_EDGES, 1, 1])),
    )
    return icon, live


NAMELIST_EXCERPT = pathlib.Path(__file__).parent / "comin_run_dir" / _config.NAMELIST_FILE
"""ICON's namelist output of a ComIn diffusion SUBSTITUTE run (the groups the plugin reads)."""


@pytest.fixture(autouse=True)
def icon_run_dir(tmp_path, monkeypatch):
    """ICON's run directory: the working directory, with ICON's namelist output; one process."""
    shutil.copyfile(NAMELIST_EXCERPT, tmp_path / _config.NAMELIST_FILE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_config, "host_comm", lambda comin: None)
    return tmp_path


def set_namelist(run_dir: pathlib.Path, **entries: str) -> None:
    """Change entries (upper-case names, as ICON writes them) of the namelist output in place."""
    path = run_dir / _config.NAMELIST_FILE
    text = path.read_text()
    for name, value in entries.items():
        text, n = re.subn(rf"(?m)^ {name} =\s*[^,\n]*", f" {name} = {value}", text)
        assert n == 1, name
    path.write_text(text)


def set_mode(run_dir: pathlib.Path, mode: int, interface: int = 1, luse: str = "T") -> None:
    set_namelist(
        run_dir,
        ICON4PY_MODE=str(mode),
        ICON4PY_INTERFACE=str(interface),
        LUSE_ICON4PY_DIFFUSION=luse,
    )


@pytest.fixture
def calls(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "CALLS", [])
    return sys.modules[__name__].CALLS


@pytest.fixture
def wrappers(monkeypatch):
    """py2fgen's module state, which the plugin must not touch (it builds its own granule)."""
    granule, grid_state = object(), object()
    monkeypatch.setattr(diffusion_wrapper, "granule", granule)
    monkeypatch.setattr(grid_wrapper, "grid_state", grid_state)
    monkeypatch.setattr(wrapper_config, "WAIT_FOR_COMPILATION", False)
    return granule, grid_state


PACKAGE_LOGGER = logging.getLogger(plugin.__package__)


@pytest.fixture(autouse=True)
def restore_package_logger():
    """'plugin.register' configures the package logger; undo that after each test."""
    state = (list(PACKAGE_LOGGER.handlers), PACKAGE_LOGGER.propagate, PACKAGE_LOGGER.level)
    yield
    PACKAGE_LOGGER.handlers[:], PACKAGE_LOGGER.propagate = state[0], state[1]
    PACKAGE_LOGGER.setLevel(state[2])


@pytest.fixture
def logs(caplog):
    """The plugin's log messages (records propagate to pytest's handler unless 'register' ran)."""
    caplog.set_level(logging.INFO, logger=plugin.__package__)
    return caplog


def cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No GPU.")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No GPU.")
    return cp


def make_plugin(
    comin: FakeComIn, functions=TOY_FUNCTIONS, granule=None, **environ: str
) -> plugin.Plugin:
    if granule is None and functions is TOY_FUNCTIONS:
        granule = ToyGranule()
    instance = plugin.Plugin(comin, functions=functions, environ=environ, granule=granule)
    instance.register()
    return instance


def started(
    icon_run_dir: pathlib.Path, mode: int = 1, device_xp: Any = None, **kwargs: Any
) -> tuple[FakeComIn, FakeIcon, dict[str, Any], plugin.Plugin]:
    """ICON's mode in its namelist output, the toy ICON, the plugin through the secondary
    constructor ('kwargs': for 'toy_icon', and the environment as 'environ')."""
    set_mode(icon_run_dir, mode)
    environ = kwargs.pop("environ", {})
    comin = FakeComIn(has_device=device_xp is not None)
    icon, live = toy_icon(comin, device_xp, mode, **kwargs)
    instance = make_plugin(comin, **environ)
    icon.secondary_constructor()
    return comin, icon, live, instance


# ---- the granule's inputs: py2fgen's, in py2fgen's forms ----------------------------------------

REAL = {
    "grid_init": grid_wrapper.grid_init,
    "diffusion_init": diffusion_wrapper.diffusion_init,
    "diffusion_run": diffusion_wrapper.diffusion_run,
}
"""py2fgen's exported functions, the reference of the granule's inputs."""
NUMPY_DTYPE = {py2fgen.FLOAT64: F64, py2fgen.INT32: I32, py2fgen.BOOL: np.dtype(np.bool_)}
PYTHON_TYPE = {py2fgen.FLOAT64: float, py2fgen.INT32: int, py2fgen.BOOL: bool}


def _py2fgen_dims(function: Any, name: str) -> tuple[gtx.Dimension, ...] | None:
    # the decision of py2fgen's 'icon4py_export.field_annotation_mapping_hook'
    hint = typing.get_type_hints(function.__wrapped__, include_extras=True)[name]
    gt4py_type = icon4py_export._get_gt4py_type(hint)
    return None if gt4py_type is None else tuple(icon4py_export._parse_type_spec(gt4py_type[0])[0])


def test_the_inputs_are_py2fgens():
    """The granule's inputs are py2fgen's arguments: names, order, dtype, rank, Field dimensions,
    memory space and presence; 'diffusion_init' gets a DiffusionConfig for 23 of its 25
    values ('ndyn_substeps' and 'nudge_max_coeff' it gets as they are)."""
    replaced = set(_granule.DIFFUSION_INIT) ^ set(REAL["diffusion_init"].param_descriptors)
    assert replaced == {"config"} | {
        "diffusion_type", "hdiff_w", "hdiff_vn", "hdiff_smag_w", "zdiffu_t",
        "type_t_diffu", "type_vn_diffu", "hdiff_efdt_ratio", "hdiff_w_efdt_ratio",
        *(f"smagorinski_scaling_{k}{i}" for k in ("factor", "height") for i in ("", "2", "3", "4")),
        "hdiff_temp", "denom_diffu_v", "itype_sher", "iforcing", "a_hshr",
        "loutshs",
    }  # fmt: skip
    assert _granule.DIFFUSION_INIT["config"] is diffusion.DiffusionConfig
    for name, params in _granule.PARAMS.items():
        descriptors = {p: d for p, d in REAL[name].param_descriptors.items() if p in params}
        assert [p for p in params if p != "config"] == list(descriptors), name
        for param, descriptor in descriptors.items():
            kind = params[param]
            if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
                assert isinstance(kind, _arguments.ArrayParam) and kind.name == param
                assert (kind.dtype, kind.rank, kind.host, kind.optional) == (
                    NUMPY_DTYPE[descriptor.dtype],
                    descriptor.rank,
                    descriptor.memory_space == py2fgen.MemorySpace.HOST,
                    descriptor.is_optional,
                ), param
                assert kind.dims == _py2fgen_dims(REAL[name], param), param
            else:
                assert kind is PYTHON_TYPE[descriptor.dtype], param
    signatures = {
        name: _arguments.Signature(name, getattr(_granule.Granule(), name), params)
        for name, params in _granule.PARAMS.items()
    }
    counts = {n: (len(s.arrays), len(s.params) - len(s.arrays)) for n, s in signatures.items()}
    assert counts == {"grid_init": (43, 13), "diffusion_init": (13, 4), "diffusion_run": (9, 2)}
    arrays = [a for s in signatures.values() for a in s.arrays]
    assert {a.name for a in arrays if a.host} == {
        *(f"{e}_{b}" for e in ("cell", "vertex", "edge") for b in ("starts", "ends")),
        *(f"{e}_owner_mask" for e in "cev"),
        *(f"{e}_glb_index" for e in "cev"),
    }
    assert {a.name for a in arrays if a.optional} == {
        "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef", "hdef_ic", "div_ic", "dwdx", "dwdy"
    }  # fmt: skip
    # the locations of the padding (for the probe and VERIFY's tables)
    unlocated = {a.name for a in arrays if a.location is None}
    assert unlocated == {
        *(f"{e}_{b}" for e in ("cell", "vertex", "edge") for b in ("starts", "ends")),
        "vct_a", "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef",
    }  # fmt: skip
    located = collections.Counter(a.location for a in arrays if a.location is not None)
    assert located == {"cell": 23, "edge": 26, "vertex": 5}
    assert [(a.name, a.axis) for a in arrays if a.axis] == [("rbf_vec_coeff_v", 2)]
    run = signatures["diffusion_run"]
    assert [a.name for a in run.arrays][:2] == ["w", "vn"]
    assert run.arrays[1].dims == (dims.EdgeDim, dims.KDim)
    assert run.arrays[0].dims == (dims.CellDim, dims.KHalfDim)  # 'w' on the half levels


def _py2fgen_argument(address: int, shape: tuple[int, ...], optional: bool) -> Any:
    """What py2fgen's conversion makes of an address (host; a NULL address short-cuts)."""
    ffi = cffi.FFI()
    keep = ffi.new("int[1]")
    pointer = ffi.NULL if address == 0 else ffi.cast("int *", keep)
    return py2fgen._conversion.as_array(ffi, (pointer, shape, False, optional))


class _DeviceVariable:
    """A variable whose device address is 'address' (as ICON's CAI would report it)."""

    def __init__(self, shape: tuple[int, ...], address: int):
        self.shape5 = shape + (1,) * (MAX_RANK - len(shape))
        self.address = address

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.shape5, "typestr": "<i4", "data": (self.address, False), "version": 3}


def _bound(param: _arguments.ArrayParam, variable: Any, shape: tuple[int, ...], present=True):
    return _arguments.BoundArray(param=param, variable=variable, shape=shape, present=present)


def _int_param(name: str, optional: bool) -> _arguments.ArrayParam:
    return _arguments.ArrayParam(name, I32, 2, optional=optional)


@pytest.mark.parametrize(
    "case, address, shape, optional",
    [
        ("absent", 0, (4, 0), True),  # ICON has no such variable
        ("zero-size on the GPU", 0, (4, 0), True),  # device address of a zero-size array: NULL
        ("zero-size on the host", 1, (4, 0), True),  # host address of a zero-size array: not NULL
        ("NULL, not optional", 0, (4, 0), False),
    ],
)
def test_optional_arguments_as_py2fgen(case, address, shape, optional):
    """
    py2fgen passes None for an optional argument with a NULL address and an array of the given
    shape otherwise (an empty 'zd_*' list on a rank: None on the GPU, where the device address of
    a zero-size array is NULL, an empty array on the host); the plugin does the same.
    """
    try:
        expected = _py2fgen_argument(address, shape, optional)
    except RuntimeError:
        expected = RuntimeError
    param = _int_param("zd", optional)
    if case == "zero-size on the host":
        buffer = fortran_buffer(shape, np.int32)  # host path: NumPy's address of the empty buffer
        bound = _bound(param, buffer, shape)
        device_xp = None
    else:
        bound = _bound(param, _DeviceVariable(shape, address), shape, case != "absent")
        device_xp = types.ModuleType("cupy")  # never used: nothing is viewed

    if expected is RuntimeError:
        with pytest.raises(ValueError, match="NULL address for a non-optional input"):
            _arguments.array_argument(bound, device_xp)
        return
    value = _arguments.array_argument(bound, device_xp)
    if expected is None:
        assert value is None
    else:
        assert value.shape == expected.shape == shape and value.dtype == expected.dtype


def test_real_zd_lists_are_none_when_empty_on_the_gpu():
    """The 'zd_*' arguments of 'diffusion_init' on a rank without such points, device build:
    ICON's device address of a zero-size array is NULL, so None (diffusion_init's empty case)."""
    params = _granule.DIFFUSION_INIT.values()
    zd = [a for a in params if isinstance(a, _arguments.ArrayParam) and a.name.startswith("zd_")]
    assert [a.name for a in zd] == ["zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef"]
    for param in zd:
        shape = (0,) if param.rank == 1 else (4, 0)
        bound = _bound(param, _DeviceVariable(shape, 0), shape)
        assert _arguments.array_argument(bound, types.ModuleType("cupy")) is None


def test_icon_variable_with_more_than_one_block():
    comin = FakeComIn()
    buffer = np.zeros((NE, NLEV, 2, 1, 1), order="F")
    comin.add("vn", buffer, datatype=DOUBLE)
    comin.contexts[None, ("vn", 1)] = FLAG_READ
    param = TOY_RUN["vn"]
    assert isinstance(param, _arguments.ArrayParam)
    with pytest.raises(ValueError, match=r"'vn' has the shape \(9, 4, 2, 1, 1\).*nblks == 1"):
        _arguments.icon_bound_array(FakeVariable(comin, ("vn", 1)), param)
    comin.buffers["vn", 1] = np.zeros((NE, NLEV, 1, 1, 1), order="F")
    bound = _arguments.icon_bound_array(FakeVariable(comin, ("vn", 1)), param)
    assert bound.shape == (NE, NLEV) and bound.present


@gtx.field_operator
def _double(a: fa.EdgeKField[gtx.float64]) -> fa.EdgeKField[gtx.float64]:
    return 2.0 * a


@gtx.program
def double(a: fa.EdgeKField[gtx.float64], out: fa.EdgeKField[gtx.float64]):
    _double(a, out=out)


def test_icon_variables_as_arguments_write_through():
    """The argument of an ICON variable is a zero-copy view: GT4Py writes ICON's memory."""
    comin = FakeComIn()
    vn = fortran_buffer((NE, NLEV), np.float64, fill=7.0)
    comin.add("vn", vn, datatype=DOUBLE)
    comin.contexts[ENTRY_POINTS[BEFORE], ("vn", 1)] = FLAG_READ | FLAG_WRITE
    comin.current_ep = ENTRY_POINTS[BEFORE]
    param = TOY_RUN["vn"]
    assert isinstance(param, _arguments.ArrayParam)
    argument = _arguments.array_argument(
        _arguments.icon_bound_array(FakeVariable(comin, ("vn", 1)), param), None
    )
    double(argument, argument, offset_provider={})  # a GT4Py program, in place
    assert isinstance(argument, gtx.Field) and np.all(vn == 14.0)
    assert _views.data_ptr(argument.ndarray) == _views.data_ptr(vn)
    comin.current_ep = ENTRY_POINTS[INIT]  # requested at DIFFUSION_BEFORE only
    with pytest.raises(FakeComInError, match="out of requested scope"):
        _arguments.icon_bound_array(FakeVariable(comin, ("vn", 1)), param)


# ---- the routes of the real functions ------------------------------------------------------------

DESCRIPTIVE = (
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
"""The inputs that the plugin takes from ComIn's descriptive data (all of 'grid_init')."""
DESCRIPTIVE_INIT = (
    *("e_bln_c_s", "geofac_div", "geofac_grg_x", "geofac_grg_y", "geofac_n2s", "nudgecoeff_e"),
    "rbf_vec_coeff_v",
)
"""The inputs of 'diffusion_init' from ComIn's descriptive data (the interpolation coefficients)."""
STATIC = ("theta_ref_mc", "wgtfac_c", "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef")
"""The inputs of 'diffusion_init' that are ICON's variables (plugin.STATIC_VARIABLES)."""


def test_routes_cover_the_84_inputs():
    assert plugin.check_routes(plugin.FUNCTIONS) == []
    rows = [(n, p) for n, entry in plugin.FUNCTIONS.items() for p in entry.params]
    assert len(rows) == 84
    routes = {(n, p): plugin.route(n, plugin.FUNCTIONS[n], p) for n, p in rows}
    by_route = collections.defaultdict(set)
    for key, value in routes.items():
        by_route[value].add(key)
    # the configuration: ICON's namelist output (the vertical grid's values, the backend,
    # icon4py's DiffusionConfig, and the two values Diffusion takes besides it)
    configuration = {(n, p) for n, table in _config.ARGUMENTS.items() for p in table}
    configuration |= {(n, p) for n, table in plugin.CONFIGURATIONS.items() for p in table}
    assert by_route["ICON's namelist output"] == configuration
    assert configuration == {
        *(("grid_init", p) for p in ("lowest_layer_thickness", "model_top_height")),
        *(("grid_init", p) for p in ("stretch_factor", "flat_height", "rayleigh_damping_height")),
        ("grid_init", "backend"), ("diffusion_init", "config"), ("diffusion_init", "backend"),
        ("diffusion_init", "ndyn_substeps"), ("diffusion_init", "nudge_max_coeff"),
    }  # fmt: skip
    descriptive = {(n, p) for n, table in _descrdata.ROUTES.items() for p in table}
    assert descriptive == {("grid_init", p) for p in DESCRIPTIVE} | {
        ("diffusion_init", p) for p in DESCRIPTIVE_INIT
    }
    assert by_route["ComIn's descriptive data"] == descriptive
    static = {(n, p) for n, table in plugin.STATIC_VARIABLES.items() for p in table}
    assert static == {("diffusion_init", p) for p in STATIC}
    assert by_route["ICON's variables"] == static
    per_call = {("diffusion_run", p) for p in plugin.FUNCTIONS["diffusion_run"].params}
    assert by_route[f"per call at {BEFORE}"] == per_call
    assert set(by_route) == set(plugin.ROUTES)  # no input without a route
    assert [len(by_route[r]) for r in plugin.ROUTES] == [10, 57, 6, 11]
    # ICON's own memory: the static variables and the per-call arrays (compared by address)
    memory = {(n, p) for n, entry in plugin.FUNCTIONS.items() for p in plugin.same_memory(n, entry)}
    assert memory == static | {
        ("diffusion_run", p)
        for p in ("w", "vn", "exner", "theta_v", "rho", "hdef_ic", "div_ic", "dwdx", "dwdy")
    }


NO_ROUTE = (
    "has no native route (the configuration from ICON's namelist output, ComIn's descriptive"
    " data, ICON's variables of STATIC_VARIABLES, and the per-call inputs of a writing function)."
)


def test_route_errors(monkeypatch):
    table = {
        p: v for p, v in plugin.STATIC_VARIABLES["diffusion_init"].items() if p != "theta_ref_mc"
    }
    monkeypatch.setattr(plugin, "STATIC_VARIABLES", {"diffusion_init": table})
    functions = {name: plugin.FUNCTIONS[name] for name in ("grid_init", "diffusion_init")}
    nonsense = {**plugin.FUNCTIONS["grid_init"].params, "nonsense": int}
    functions["grid_init"] = plugin.FunctionEntry(nonsense, INIT, inout=False)
    assert plugin.check_routes(functions) == [
        f"grid_init: 'nonsense' {NO_ROUTE}",
        f"diffusion_init: 'theta_ref_mc' {NO_ROUTE}",
    ]
    with pytest.raises(ValueError, match="inputs without a route"):
        plugin.Plugin(FakeComIn(), functions=functions)
    # ICON's variables are a route of a function that does not write only
    writing = plugin.FunctionEntry(TOY_INIT, INIT, inout=True)
    assert not plugin.static_route("diffusion_init", writing, "zd_cellidx")
    assert plugin.static_route("diffusion_init", plugin.FUNCTIONS["diffusion_init"], "zd_cellidx")


def test_register_logs_the_inputs(logs):
    plugin.Plugin(FakeComIn(), environ={}).register()
    assert (
        "inputs: 84 (65 arrays, 19 scalars and configurations) of grid_init, diffusion_init,"
        " diffusion_run; every one from a native route: ICON's namelist output 10, ComIn's"
        f" descriptive data 57, ICON's variables 6, per call at {BEFORE} 11" in logs.messages
    )


# ---- registration ------------------------------------------------------------------------------


def test_register_logs_setup(logs):
    make_plugin(FakeComIn())
    assert logs.messages[0].startswith(f"source sha1 {plugin.source_sha1()}, icon4py.bindings")
    assert logs.messages[1] == (
        "ICON mode SUBSTITUTE (icon4py_interface=1, luse_icon4py_diffusion=T, icon4py_mode=1):"
        f" this plugin computes the horizontal diffusion of domain 1 at {BEFORE};"
        " ICON skips its own"
    )
    assert not any(m.startswith("entry points") for m in logs.messages)
    assert (
        f"counting the callbacks at {BEFORE}=20, {AFTER}=21, EP_ATM_DYCORE_SOLVE_NH_BEFORE=22,"
        " EP_ATM_DYCORE_SOLVE_NH_AFTER=23" in logs.messages
    )
    assert (
        "inputs: 10 (6 arrays, 4 scalars and configurations) of diffusion_init, diffusion_run;"
        " every one from a native route: ICON's namelist output 2, ComIn's descriptive data 0,"
        f" ICON's variables 3, per call at {BEFORE} 5" in logs.messages
    )
    assert all(r.levelno == logging.INFO for r in logs.records)


def test_register_uses_entry_points_by_name(logs):
    """The plugin uses ComIn's names only: other numbers (e.g. EPs appended to ComIn's
    enumeration) change nothing but the logged numbers."""
    numbers = {k: v + 3 for k, v in ENTRY_POINTS.items()}
    make_plugin(FakeComIn(entry_points=numbers))
    assert f"counting the callbacks at {BEFORE}=23, {AFTER}=24," in logs.text
    assert all(r.levelno == logging.INFO for r in logs.records)


def test_register_warns_once_about_other_variables(logs):
    removed = {"ICON4PY_COMIN_DUAL": "strict", "ICON4PY_COMIN_TIMELEVEL_CHECK": "strict"}
    make_plugin(FakeComIn(), **removed, **{plugin.CHECK_ENV: "1", "OTHER": "x"})
    warnings = [r.message for r in logs.records if r.levelno >= logging.WARNING]
    assert warnings == [
        "ignored: ICON4PY_COMIN_DUAL, ICON4PY_COMIN_TIMELEVEL_CHECK; this plugin reads only"
        " ICON4PY_COMIN_CHECK, ICON4PY_COMIN_PROBE, ICON4PY_COMIN_PROFILE, ICON4PY_COMIN_TIMING"
        " (the variables of removed checks have no effect)."
    ]


@pytest.mark.parametrize(
    "missing", [BEFORE, "EP_ATM_DYCORE_SOLVE_NH_AFTER", "EP_ATM_INTEGRATE_START", "EP_DESTRUCTOR"]
)
def test_register_needs_the_entry_points(missing):
    entry_points = {k: v for k, v in ENTRY_POINTS.items() if k != missing}
    with pytest.raises(RuntimeError, match=f"no entry point '{missing}'"):
        make_plugin(FakeComIn(entry_points=entry_points))


def test_register_refuses_a_second_instance(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "comin", FakeComIn())
    monkeypatch.setattr(plugin, "_instance", None)
    first = plugin.register()
    assert isinstance(first, plugin.Plugin)
    with pytest.raises(RuntimeError, match="loaded twice"):
        plugin.register()
    # 'register' prints the package's records on standard output, with the rank
    assert "icon4py-comin: counting the callbacks at " in capsys.readouterr().out


@pytest.mark.parametrize("rank", [0, 1])
def test_log_format_and_ranks(rank):
    stream = io.StringIO()
    handler = plugin.configure_logging(rank, stream)
    assert PACKAGE_LOGGER.propagate is False and handler in PACKAGE_LOGGER.handlers
    log = logging.getLogger(_arguments.__name__)  # any logger of the package
    log.info("setup line")
    log.info("per-rank line", extra=plugin.ALL_RANKS)
    log.warning("something odd")
    log.debug("not shown")
    lines = stream.getvalue().splitlines()
    expected = [
        f"icon4py-comin: per-rank line [rank {rank}]",
        f"icon4py-comin: WARNING: something odd [rank {rank}]",
    ]
    assert lines == ([f"icon4py-comin: setup line [rank {rank}]"] if rank == 0 else []) + expected


# ---- idle: ICON computes the diffusion itself or through py2fgen ------------------------------


@pytest.mark.parametrize(
    "mode, interface, line",
    [
        (0, 1, "ICON computes the horizontal diffusion; this plugin stays idle"),
        (
            1,
            0,
            "icon4py computes the horizontal diffusion through py2fgen, not ComIn; this plugin"
            " stays idle",
        ),
        (
            2,
            0,
            "icon4py computes the horizontal diffusion through py2fgen, not ComIn; this plugin"
            " stays idle",
        ),
    ],
)
def test_idle(  # noqa: PLR0917 [too-many-positional-arguments]
    calls, wrappers, logs, icon_run_dir, mode, interface, line
):
    set_mode(icon_run_dir, mode, interface)
    comin = FakeComIn()
    icon, live = toy_icon(comin, mode=0)  # ICON computes its own diffusion
    instance = make_plugin(comin)
    icon.secondary_constructor()
    assert not instance.active and comin.contexts == {}
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    icon.diffusion_call(dtime=10.0, linit=False)
    comin.fire("EP_DESTRUCTOR")
    assert calls == [] and comin.accessed == []
    assert (diffusion_wrapper.granule, grid_wrapper.grid_state) == wrappers
    assert wrapper_config.WAIT_FOR_COMPILATION is False
    assert np.all(live["w"] == 21.0)  # ICON's own diffusion only
    switches = f"icon4py_interface={interface}, luse_icon4py_diffusion=T, icon4py_mode={mode}"
    assert (
        logs.messages[1]
        == f"ICON mode {('OFF', 'SUBSTITUTE', 'VERIFY')[mode]} ({switches}): {line}"
    )
    assert any(m.startswith("idle: ICON mode") and "requested nothing." in m for m in logs.messages)
    assert (
        f"entry point callbacks: {BEFORE} 2, {AFTER} 2, EP_ATM_DYCORE_SOLVE_NH_BEFORE 2,"
        " EP_ATM_DYCORE_SOLVE_NH_AFTER 2 (domains [1])" in logs.messages
    )
    assert not any(m.startswith("per-call state") for m in logs.messages)
    assert not any(r.levelno >= logging.WARNING for r in logs.records)


# ---- what the plugin requests, with the real functions -----------------------------------------

PER_CALL_VARIABLES = ("w", "vn", "exner", "theta_v", "rho", "hdef_ic", "div_ic")
"""ICON's variables of the names of diffusion_run's arrays (ICON has no 'dwdx' and 'dwdy' here)."""


def real_icon(comin: FakeComIn) -> None:
    """ICON's variables that the real functions get (the shapes do not matter here)."""
    for name in PER_CALL_VARIABLES:
        comin.add(name, fortran_buffer((NC, NLEV), np.float64), datatype=DOUBLE)
    for icon_name in plugin.STATIC_VARIABLES["diffusion_init"].values():
        comin.add(icon_name, fortran_buffer((NC, NLEV), np.float64), datatype=DOUBLE)


@pytest.mark.parametrize("has_device", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("mode", [1, 2], ids=["substitute", "verify"])
def test_real_functions_request(monkeypatch, icon_run_dir, logs, has_device, mode):
    """SUBSTITUTE: ICON's per-call variables READ | WRITE at DIFFUSION_BEFORE; VERIFY: READ only,
    at DIFFUSION_BEFORE (copied) and DIFFUSION_AFTER (compared); ICON's static variables READ,
    where diffusion_init gets them and where the granule computes. Nothing else."""
    if has_device:
        monkeypatch.setattr(plugin, "cp", types.ModuleType("cupy"))  # selected, not used
    set_mode(icon_run_dir, mode)
    comin = FakeComIn(has_device=has_device)
    real_icon(comin)
    instance = make_plugin(comin, functions=plugin.FUNCTIONS)
    comin.fire("EP_SECONDARY_CONSTRUCTOR")
    assert instance.active
    device = FLAG_DEVICE if has_device else 0
    init_ep, before_ep, after_ep = (ENTRY_POINTS[e] for e in (INIT, BEFORE, AFTER))
    expected = {}
    for name in PER_CALL_VARIABLES:
        if mode == 1:
            expected[before_ep, (name, 1)] = FLAG_READ | FLAG_WRITE | device
        else:
            expected[before_ep, (name, 1)] = FLAG_READ | device
            expected[after_ep, (name, 1)] = FLAG_READ | device
    computes = before_ep if mode == 1 else after_ep
    for name in plugin.STATIC_VARIABLES["diffusion_init"].values():
        expected[init_ep, (name, 1)] = FLAG_READ | device
        expected[computes, (name, 1)] = FLAG_READ | device
    assert comin.contexts == expected  # nothing else, never SYNC_HALO
    assert len(expected) == (7 if mode == 1 else 14) + 12
    assert (
        "active: requested 13 of ICON's variables for grid_init, diffusion_init, diffusion_run."
        in logs.messages
    )
    assert (
        f"requested ICON's w vn exner theta_v rho hdef_ic div_ic"
        f" ({'READ | WRITE' if mode == 1 else 'READ'}{' | DEVICE' if has_device else ''}) at"
        f" {BEFORE}{'' if mode == 1 else f' and {AFTER}'} for diffusion_run; not exposed (absent"
        " optional arguments): dwdx dwdy" in " ".join(logs.messages)
    )


def test_substitute_needs_icons_variables(icon_run_dir):
    comin = FakeComIn()
    comin.add("w", fortran_buffer((NC, NLEV + 1), np.float64), datatype=DOUBLE)
    comin.add("theta_ref_mc", fortran_buffer((NC, NLEV), np.float64), datatype=DOUBLE)
    make_plugin(comin)
    with pytest.raises(RuntimeError, match="1 problem") as error:
        comin.fire("EP_SECONDARY_CONSTRUCTOR")
    assert (
        f"'vn': ICON does not expose its variable, which 'diffusion_run' gets at {BEFORE}"
        in str(error.value)
    )


# ---- SUBSTITUTE: diffusion_run at EP_ATM_DYCORE_DIFFUSION_BEFORE on ICON's own variables ------


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_substitute_life_cycle(calls, wrappers, logs, icon_run_dir, on_device):
    device_xp = cupy_or_skip() if on_device else None
    set_mode(icon_run_dir, 1)
    comin = FakeComIn(has_device=on_device)
    icon, live = toy_icon(comin, device_xp)
    instance = make_plugin(comin)
    # a plugin registered after this one sees at DIFFUSION_BEFORE what the granule has done
    seen: list[int] = []
    comin.EP_ATM_DYCORE_DIFFUSION_BEFORE(lambda: seen.append(len(calls)))
    icon.secondary_constructor()
    assert instance.active
    device = FLAG_DEVICE if on_device else 0
    for name in ("w", "vn"):
        assert comin.contexts[ENTRY_POINTS[BEFORE], (name, 1)] == FLAG_READ | FLAG_WRITE | device
    assert not any(ep == ENTRY_POINTS[AFTER] for ep, _ in comin.contexts)

    # pass 1: diffusion_init at EP_ATM_TIMELOOP_BEFORE, on ICON's variables and the configuration
    icon.new_pass()
    assert wrapper_config.WAIT_FOR_COMPILATION is False  # py2fgen's flag: untouched
    ((name, init),) = calls
    theta, zd_cellidx = init["theta_ref_mc"], init["zd_cellidx"]
    assert isinstance(theta, gtx.Field) and tuple(theta.domain.dims) == (dims.CellDim, dims.KDim)
    assert _views.data_ptr(theta.ndarray) == _views.data_ptr(live["theta_ref_mc"])
    assert type(zd_cellidx).__module__.split(".")[0] == ("cupy" if on_device else "numpy")
    assert _views.data_ptr(zd_cellidx) == _views.data_ptr(live["zd_indlist"])
    assert zd_cellidx.shape == (4, NPOINTS)
    # ICON's namelist output: icon4py's DiffusionConfig and the backend
    assert isinstance(init["config"], diffusion.DiffusionConfig) and init["backend"] == 0
    config = init["config"]
    assert (config.hdiff_efdt_ratio, config.apply_to_horizontal_wind) == (24.0, True)

    # two calls at DIFFUSION_BEFORE, before ICON's dispatcher, which skips its own diffusion
    icon.diffusion_call(dtime=10.0, linit=True)
    assert seen == [2]
    assert np.all(icon.host(live["w"]) == 11.0) and np.all(icon.host(live["vn"]) == 6.0)
    # ICON re-targets its 'w' to another time level: the next call computes on that one
    other_w = icon.move("w", fortran_buffer((NC, NLEV + 1), np.float64, fill=100.0))
    icon.diffusion_call(dtime=5.0, linit=False)
    assert seen == [2, 3]
    assert np.all(icon.host(other_w) == 105.0) and np.all(icon.host(live["w"]) == 11.0)
    assert np.all(icon.host(live["vn"]) == 12.0)
    # the granule got ICON's time step and the derived flag
    assert [(c[0], c[1]["dtime"], c[1]["linit"]) for c in calls[1:]] == [
        ("run", 10.0, True),
        ("run", 5.0, False),
    ]
    assert calls[1][1]["opt"] is None
    assert _views.data_ptr(calls[2][1]["w"].ndarray) == _views.data_ptr(other_w)
    assert icon.fortran_calls == 0

    # pass 2 (e.g. iterative IAU): diffusion_init again
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=False)
    assert [c[0] for c in calls] == ["init", "run", "run", "init", "run"]
    comin.fire("EP_DESTRUCTOR")
    assert instance._granule.released
    assert (diffusion_wrapper.granule, grid_wrapper.grid_state) == wrappers  # py2fgen's: untouched
    assert not instance.active

    messages = logs.messages
    assert f"device {on_device}" in messages[0]
    assert (
        f"requested ICON's w vn (READ | WRITE{' | DEVICE' if on_device else ''}) at {BEFORE}"
        " for diffusion_run; not exposed (absent optional arguments): opt" in messages
    )
    assert "active: requested 5 of ICON's variables for diffusion_init, diffusion_run." in messages
    assert (
        "diffusion_init done (pass 1)." in messages and "diffusion_init done (pass 2)." in messages
    )
    condition = "ldynamics T, ltestcase F, lhdiff_vn T, init_mode 7, lrestartrun F"
    assert f"initial diffusion call: {condition}; 'linit' from the order of" in logs.text
    order = "EP_ATM_DYCORE_SOLVE_NH_BEFORE since EP_ATM_INTEGRATE_START"
    assert (
        f"linit call 1: T from the entry-point order (0 {order} 1); ICON's condition: T"
        f" ({condition}, step 1, call 1 of the step): agree" in messages
    )
    assert (
        f"linit call 2: F from the entry-point order (2 {order} 1); ICON's condition: F"
        f" ({condition}, step 1, call 2 of the step): agree" in messages
    )
    for n, flag in ((1, "T"), (2, "F"), (3, "F")):
        assert f"diffusion_run call {n} done at {BEFORE} (linit {flag})." in messages
    assert messages.count("pointer identity OK (2 fields)") == 1
    assert sum("sums before -> after" in m for m in messages) == 1
    assert "w 30.0 -> 330.0 (changed)" in logs.text and "vn 108.0 -> 216.0 (changed)" in logs.text
    assert "diffusion_run optional arguments: opt None (absent)" in messages
    assert not any(m.startswith("timing") for m in messages)  # off by default
    assert (
        f"entry point callbacks: {BEFORE} 3, {AFTER} 3, EP_ATM_DYCORE_SOLVE_NH_BEFORE 4,"
        " EP_ATM_DYCORE_SOLVE_NH_AFTER 4 (domains [1])" in messages
    )
    assert (
        f"per-call state: 3 diffusion calls of domain 1 at {BEFORE}, 3 computed there, 0 copied"
        " there, 0 without lhdiff_vn; linit: initial 1, regular 2, ICON's condition agrees at 3"
        " of 3; EP_ATM_INTEGRATE_START 2 (domain 1)" in messages
    )
    assert (
        "diffusion_init: ICON's variables at the addresses of pass 1 at all 3 diffusion calls"
        f" ({BEFORE})" in messages
    )
    assert messages[-1] == "released the granule."
    assert not any(r.levelno >= logging.WARNING for r in logs.records)


def test_substitute_without_lhdiff_vn(calls, wrappers, logs, icon_run_dir):
    set_namelist(icon_run_dir, LHDIFF_VN="F")
    comin, icon, live, _ = started(icon_run_dir)
    icon.new_pass()
    for _ in range(2):  # ICON fires the regular pair, but calls no diffusion
        icon.diffusion_call(dtime=10.0, linit=False, lhdiff_vn=False)
    comin.fire("EP_DESTRUCTOR")
    assert [c[0] for c in calls] == ["init"] and np.all(live["w"] == 1.0)
    assert f"{BEFORE} 2: lhdiff_vn F, ICON calls no diffusion; nothing to do." in logs.messages
    assert not any(m.startswith("linit call") for m in logs.messages)
    assert (
        f"per-call state: 0 diffusion calls of domain 1 at {BEFORE}, 0 computed there, 0 copied"
        " there, 2 without lhdiff_vn; linit: initial 0, regular 0, ICON's condition agrees at 0"
        " of 0; EP_ATM_INTEGRATE_START 2 (domain 1)" in logs.messages
    )


@pytest.mark.parametrize(
    "entry, value", [("INIT_MODE", "5"), ("LTESTCASE", "T"), ("LDYNAMICS", "F"), ("restart", "")]
)
def test_linit_disagreeing_with_icons_condition_stops(calls, wrappers, icon_run_dir, entry, value):
    set_mode(icon_run_dir, 1)
    comin = FakeComIn()
    if entry == "restart":
        comin.lrestartrun = True
    else:
        set_namelist(icon_run_dir, **{entry: value})
    if entry == "LDYNAMICS":  # without dynamics the plugin also reads 'l_scm_mode' ('loutshs')
        with (icon_run_dir / _config.NAMELIST_FILE).open("a") as f:
            f.write(" &GRID_NML\n L_SCM_MODE =  F,\n /\n")
    icon, live = toy_icon(comin)
    make_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    # ICON's condition excludes an initial call, but a DIFFUSION_BEFORE comes before the dynamics
    with pytest.raises(
        RuntimeError, match=r"linit call 1: T from .* ICON's condition: F .* disagree"
    ):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in calls] == ["init"] and np.all(live["w"] == 1.0)


def test_linit_needs_integrate_start(calls, wrappers, icon_run_dir):
    comin, icon, _, _ = started(icon_run_dir)
    icon.new_pass()
    with pytest.raises(RuntimeError, match="before any EP_ATM_INTEGRATE_START"):
        comin.fire(BEFORE, 1)


def test_linit_in_later_steps_and_domains(calls, wrappers, logs, icon_run_dir):
    comin, icon, _, _ = started(icon_run_dir)
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    comin.fire("EP_ATM_INTEGRATE_START", 2)  # another domain's order does not count
    comin.fire("EP_ATM_DYCORE_SOLVE_NH_BEFORE", 2)
    icon.diffusion_call(dtime=10.0, linit=False)
    icon.diffusion_call(dtime=10.0, linit=False)
    lines = [m for m in logs.messages if m.startswith("linit call")]
    assert [m.split(":")[1].split(" from")[0].strip() for m in lines] == ["T", "F", "F"]
    assert "step 2, call 1 of the step): agree" in lines[2]


@pytest.mark.parametrize("domain_id", [-1, 2])
def test_diffusion_before_of_another_domain(calls, wrappers, logs, icon_run_dir, domain_id):
    comin, icon, live, _ = started(icon_run_dir)
    icon.new_pass()
    comin.accessed.clear()
    comin.fire("EP_ATM_INTEGRATE_START", domain_id)
    comin.fire(BEFORE, domain_id)
    comin.fire(AFTER, domain_id)
    assert [c[0] for c in calls] == ["init"]
    assert comin.accessed == []  # touched nothing
    assert np.all(live["w"] == 1.0)
    warnings = [r.message for r in logs.records if r.levelno == logging.WARNING]
    if domain_id < 0:  # outside the domain loop: not a diffusion call of any domain
        assert warnings == []
    else:
        assert warnings == [
            f"{BEFORE} fired for domain 2; the plugin computes domain 1 only; ignored."
        ]


def test_pointer_identity_check_catches_a_copy(calls, wrappers, monkeypatch, icon_run_dir):
    _, icon, _, _ = started(icon_run_dir)
    icon.new_pass()
    # a helper that copies ('gtx.as_field') instead of 'field_view'
    monkeypatch.setattr(_views, "field_view", lambda array, dims: gtx.as_field(list(dims), array))
    with pytest.raises(
        RuntimeError, match="'w' of 'diffusion_run' does not alias the array it was built from"
    ):
        icon.diffusion_call(dtime=10.0, linit=True)


def test_checks_off(calls, wrappers, logs, icon_run_dir):
    _, icon, live, _ = started(icon_run_dir, environ={plugin.CHECK_ENV: "0"})
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    assert np.all(live["w"] == 11.0)
    assert "in-job consistency checks are off" in logs.text
    assert "pointer identity" not in logs.text and "sums before" not in logs.text


# ---- ICON's variables that diffusion_init gets (plugin.STATIC_VARIABLES) -----------------------


def test_static_variables_are_the_real_names():
    real = plugin.STATIC_VARIABLES["diffusion_init"]
    assert {p: real[p] for p in STATIC_NAMES} == STATIC_NAMES
    assert set(real) <= set(diffusion_wrapper.diffusion_init.param_descriptors)
    assert real["zd_cellidx"] == "zd_indlist"  # ICON's name of py2fgen's 'zd_cellidx'


@pytest.mark.parametrize("mode", [1, 2], ids=["substitute", "verify"])
@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_static_variables_end_to_end(  # noqa: PLR0917 [too-many-positional-arguments]
    calls, wrappers, logs, icon_run_dir, on_device, mode
):
    device_xp = cupy_or_skip() if on_device else None
    comin, icon, _, _ = started(icon_run_dir, mode, device_xp)
    device = FLAG_DEVICE if on_device else 0
    computes = BEFORE if mode == 1 else AFTER
    contexts = {k: v for k, v in comin.contexts.items() if k[1][0] in STATIC_NAMES.values()}
    assert contexts == {
        (ENTRY_POINTS[ep], (n, 1)): FLAG_READ | device
        for n in STATIC_NAMES.values()
        for ep in (INIT, computes)
    }
    assert (
        "requested ICON's theta_ref_mc zd_indlist (as zd_cellidx) zd_diffcoef"
        f" (READ{' | DEVICE' if on_device else ''}) at {INIT} for diffusion_init; also at"
        f" {computes}, where the granule computes with them" in logs.messages
    )
    icon.new_pass()
    assert (
        "diffusion_init pass 1: ICON's variables theta_ref_mc zd_cellidx zd_diffcoef fetched at"
        f" {INIT} (3 present, 0 absent); addresses recorded" in logs.messages
    )
    icon.new_pass()  # pass 2: fetched again, at the same addresses
    assert (
        "diffusion_init pass 2: ICON's variables theta_ref_mc zd_cellidx zd_diffcoef fetched at"
        f" {INIT} (3 present, 0 absent); addresses as at pass 1" in logs.messages
    )
    # where the granule computes, fetched again at every call: at the addresses of pass 1
    icon.diffusion_call(dtime=10.0, linit=True)
    icon.diffusion_call(dtime=10.0, linit=False)
    comin.fire("EP_DESTRUCTOR")
    assert (
        "diffusion_init: ICON's variables theta_ref_mc zd_cellidx zd_diffcoef fetched at"
        f" {computes} (diffusion call 1): at the addresses of pass 1, which the granule keeps;"
        " checked at every call" in logs.messages
    )
    assert (
        "diffusion_init: ICON's variables at the addresses of pass 1 at all 2 diffusion calls"
        f" ({computes})" in logs.messages
    )


@pytest.mark.parametrize("mode", [1, 2], ids=["substitute", "verify"])
def test_static_variable_moved_before_a_call(calls, wrappers, icon_run_dir, mode):
    """The granule keeps ICON's variables from the init call: one that moved stops the call."""
    _, icon, live, _ = started(icon_run_dir, mode)
    icon.new_pass()
    icon.move("theta_ref_mc", live["theta_ref_mc"].copy(order="F"))
    where = BEFORE if mode == 1 else AFTER
    with pytest.raises(
        RuntimeError, match=rf"theta_ref_mc of 'diffusion_init' at {where} of diffusion call 1"
    ):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in calls] == ["init"]  # the granule did not run


def test_static_variables_absent_without_the_lists(calls, wrappers, logs, icon_run_dir):
    """l_zdiffu_t=.FALSE.: ICON has no 'zd_*' lists; None for the granule."""
    _, icon, _, _ = started(icon_run_dir, zdiffu=False)
    assert (
        f"requested ICON's theta_ref_mc (READ) at {INIT} for diffusion_init; not exposed (absent"
        " optional arguments): zd_indlist (as zd_cellidx) zd_diffcoef; also at"
        f" {BEFORE}, where the granule computes with them" in logs.messages
    )
    icon.new_pass()
    assert "(1 present, 2 absent); addresses recorded" in logs.text
    init = calls[0][1]
    assert init["zd_cellidx"] is None and init["zd_diffcoef"] is None
    assert (
        "diffusion_init optional arguments: zd_cellidx None (absent), zd_diffcoef None (absent)"
        in logs.messages
    )


def test_static_variable_moved_between_passes(calls, wrappers, icon_run_dir):
    _, icon, live, _ = started(icon_run_dir)
    icon.new_pass()
    icon.move("zd_diffcoef", live["zd_diffcoef"].copy(order="F"))
    with pytest.raises(RuntimeError, match=r"zd_diffcoef of 'diffusion_init' moved between pass 1"):
        icon.new_pass()
    assert len(calls) == 1


def test_static_variable_must_be_exposed(icon_run_dir):
    set_mode(icon_run_dir, 1)
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    del comin.buffers["theta_ref_mc", 1]
    make_plugin(comin)
    with pytest.raises(RuntimeError, match="1 problem") as error:
        icon.secondary_constructor()
    assert (
        "'theta_ref_mc': ICON does not expose its variable, the argument 'theta_ref_mc' of"
        " 'diffusion_init'." in str(error.value)
    )


# ---- VERIFY: the plugin computes on its own copies of ICON's input and compares ----------------


def table_rows(messages: list[str], which: str) -> list[list[str]]:
    """The rows of the plugin's tables 'which' (all calls), split into columns."""
    rows, current = [], None
    for m in messages:
        title = re.fullmatch(r"-{20} (\w+) -{20}", m)
        if title:
            current = title.group(1)
        elif current == which and (" Yes " in m or " No " in m):
            rows.append(m.split())
    return rows


def random_fields(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"w": rng.normal(size=(NC, NLEV + 1)), "vn": rng.normal(size=(NE, NLEV)) + 3.0}


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_verify_copies_computes_and_compares(calls, wrappers, logs, icon_run_dir, on_device):
    device_xp = cupy_or_skip() if on_device else None
    start = random_fields()
    comin, icon, live, _ = started(icon_run_dir, 2, device_xp, **start)
    device = FLAG_DEVICE if on_device else 0
    for name in ("w", "vn"):  # ICON's variables: only read, never written
        assert comin.contexts[ENTRY_POINTS[BEFORE], (name, 1)] == FLAG_READ | device
        assert comin.contexts[ENTRY_POINTS[AFTER], (name, 1)] == FLAG_READ | device
    # the granule computes when ICON's variables hold ICON's results, on the plugin's copies
    at_call: list[np.ndarray] = []
    comin.EP_ATM_DYCORE_DIFFUSION_AFTER(lambda: at_call.append(icon.host(live["w"]).copy()))

    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in calls] == ["init", "run"]  # at DIFFUSION_AFTER, after ICON's own
    icon.diffusion_call(dtime=5.0, linit=False)
    comin.fire("EP_DESTRUCTOR")
    # ICON continues with its own result, untouched by the plugin
    assert icon.fortran_calls == 2
    assert np.array_equal(icon.host(live["w"])[:, :, 0, 0, 0], start["w"] + 15.0)
    assert np.array_equal(icon.host(live["vn"])[:, :, 0, 0, 0], start["vn"] * 4.0)
    assert np.array_equal(at_call[0][:, :, 0, 0, 0], start["w"] + 10.0)
    # the granule got the plugin's copies (not ICON's), the derived time step and flag
    assert [(c[1]["dtime"], c[1]["linit"]) for c in calls[1:]] == [(10.0, True), (5.0, False)]
    w = calls[1][1]["w"]
    assert isinstance(w, gtx.Field) and _views.data_ptr(w.ndarray) != _views.data_ptr(live["w"])
    assert calls[1][1]["opt"] is None

    messages = logs.messages
    assert messages[1].endswith(
        "ICON computes the horizontal diffusion and continues with its own result; this plugin"
        f" computes it too, at {AFTER} on its own copies of ICON's input from {BEFORE}, and"
        " compares its results with ICON's there; ICON compares nothing"
    )
    assert (
        f"requested ICON's w vn (READ{' | DEVICE' if on_device else ''}) at {BEFORE} and"
        f" {AFTER} for diffusion_run; not exposed (absent optional arguments): opt; copies them"
        f" at {BEFORE} into the plugin's buffers, computes on the copies and compares its results"
        f" with them at {AFTER}" in messages
    )
    nbytes = 8 * (NC * (NLEV + 1) + NE * NLEV) / 2**20
    assert (
        f"diffusion_run call 1: copied ICON's w vn at {BEFORE} into the plugin's buffers"
        f" ({nbytes:.2f} MiB{' on the device' if on_device else ''}, kept); absent: opt" in messages
    )
    assert (
        f"diffusion_run call 2 done at {AFTER} on the plugin's copies of ICON's input from"
        f" {BEFORE} (linit F)." in messages
    )
    # the plugin's tables at DIFFUSION_AFTER, in ICON's format
    header = [m for m in messages if m.startswith(plugin.VERIFY_HEADER)]
    assert header[0] == (
        f"plugin verifying diffusion_run call 1 (linit T): icon4py on the plugin's copies of"
        f" ICON's input ({BEFORE}, computed at {AFTER}) against ICON at {AFTER}; atol 1e-08,"
        " rtol 1e-05"
    )
    domain, padding = table_rows(messages, "domain"), table_rows(messages, "padding")
    assert [r[0] for r in domain] == ["vn", "w"] * 2  # ICON's order: edges first
    for row in domain:
        assert row[1:4] == ["Yes", "100.0", "0.0000E+00"]  # bitwise: no error at all
    assert [r[0] for r in padding] == ["vn", "w"] * 2
    assert (
        "verification: the 'domain' table of the edges ends at 7 (end_index of the outermost"
        " halo row; nedges 7)" in messages
    )
    assert (
        f"diffusion_run call 2 compared at {AFTER}: 2 fields, ICON's variables as at {BEFORE};"
        " rows not close: domain 0, padding 0" in messages
    )
    assert (
        f"verification: 2 calls copied at {BEFORE}, computed on the copies and compared at"
        f" {AFTER}: 2; rows not close: domain 0, padding 0" in messages
    )
    assert (
        f"per-call state: 2 diffusion calls of domain 1 at {BEFORE}, 0 computed there, 2 copied"
        " there, 0 without lhdiff_vn; linit: initial 1, regular 1, ICON's condition agrees at 2"
        " of 2; EP_ATM_INTEGRATE_START 1 (domain 1)" in messages
    )
    assert not any(r.levelno >= logging.WARNING for r in logs.records)


def test_verify_tables_show_icons_difference(calls, wrappers, logs, icon_run_dir):
    """ICON's own diffusion differs from icon4py's: the plugin's tables show it."""
    _, icon, _, _ = started(icon_run_dir, 2, fortran_w=1.5, **random_fields())
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    domain = table_rows(logs.messages, "domain")
    assert [r[:3] for r in domain] == [["vn", "Yes", "100.0"], ["w", "No", "0.0"]]
    # w: ICON's +15 against the plugin's +10, on every real entry
    assert domain[1][9] == "0.5000E+01"  # max_abs_err
    assert domain[1][4] == "0.2500E+02"  # abs MSE
    assert (
        f"diffusion_run call 1 compared at {AFTER}: 2 fields, ICON's variables as at {BEFORE};"
        " rows not close: domain 1, padding 1" in logs.messages
    )


def test_verify_needs_diffusion_after(calls, wrappers, icon_run_dir):
    comin, icon, _, _ = started(icon_run_dir, 2)
    icon.new_pass()
    comin.fire("EP_ATM_INTEGRATE_START", 1)
    comin.timestep = 10.0
    comin.fire(BEFORE, 1)  # no DIFFUSION_AFTER follows
    with pytest.raises(RuntimeError, match=f"call 1 of domain 1 was not followed by {AFTER}"):
        icon.diffusion_call(dtime=10.0, linit=False)


def test_verify_icon_variables_moved_stops(calls, wrappers, icon_run_dir):
    _, icon, live, _ = started(icon_run_dir, 2)
    icon.new_pass()
    original = icon.dispatch

    def moved(dtime: float) -> None:
        original(dtime)
        icon.move("w", np.asarray(live["w"]).copy(order="F"))  # another time level at AFTER

    icon.dispatch = moved  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=f"ICON's variables w at {AFTER} are not those copied"):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in calls] == ["init"]  # the granule did not run


# ---- timing and profiling (ICON4PY_COMIN_TIMING, ICON4PY_COMIN_PROFILE) ------------------------

PHASES = ["guard", "args", "sums", "presync", "run", "sync", "check", "done"]


@pytest.mark.parametrize("mode", [1, 2], ids=["substitute", "verify"])
def test_timing(calls, wrappers, logs, icon_run_dir, mode):
    comin, icon, _, _ = started(icon_run_dir, mode, environ={plugin.TIMING_ENV: "1"})
    icon.new_pass()
    for n in range(3):
        icon.diffusion_call(dtime=1.0, linit=n == 0)
    comin.fire("EP_DESTRUCTOR")

    timing = [m for m in logs.messages if m.startswith("timing")]
    assert timing[0].startswith("timing: Python threads MainThread")
    for n, line in zip((1, 2, 3), timing[1:4], strict=True):
        assert line.startswith(f"timing diffusion_run call {n} [ms]: ")
        assert [p.split()[0] for p in line.split(": ", 1)[1].split(", ")] == [*PHASES, "total"]
    assert timing[4].startswith("timing diffusion_run, median of 2 calls after the first [ms]: ")
    assert len(timing) == 5  # only the writing function is timed
    states = [m.split(":", 1)[0] for m in logs.messages if m.startswith("state at")]
    assert states == [
        "state at primary constructor",
        f"state at {INIT}, before",
        f"state at {INIT}, after",
        "state at after diffusion_run call 1",
        "state at after diffusion_run call 2",
        "state at after diffusion_run call 3",
        "state at destructor",
    ]
    (state,) = [m for m in logs.messages if m.startswith("state at destructor")]
    assert "native threads [name:CPU s]" in state


def test_timing_idle(calls, wrappers, logs, icon_run_dir):
    set_mode(icon_run_dir, 0)
    comin = FakeComIn()
    make_plugin(comin, **{plugin.TIMING_ENV: "1"})
    for entry_point in ("EP_SECONDARY_CONSTRUCTOR", INIT, "EP_DESTRUCTOR"):
        comin.fire(entry_point)
    states = [m.split(":", 1)[0] for m in logs.messages if m.startswith("state at")]
    assert states == [
        "state at primary constructor",
        f"state at {INIT}, idle",
        "state at destructor, idle",
    ]


@pytest.mark.parametrize("mode", [1, 2], ids=["substitute", "verify"])
def test_profile(calls, wrappers, logs, icon_run_dir, mode):
    _, icon, _, _ = started(icon_run_dir, mode, environ={plugin.PROFILE_ENV: "2"})
    icon.new_pass()
    for n in range(3):
        icon.diffusion_call(dtime=1.0, linit=n == 0)
    assert [c[0] for c in calls] == ["init", "run", "run", "run"]  # the profiled call ran once
    headers = [m for m in logs.messages if m.startswith("profile of")]
    assert headers == [
        "profile of diffusion_run call 2, sorted by cumulative:",
        "profile of diffusion_run call 2, sorted by tottime:",
    ]
    assert any("test_comin_plugin.py" in m and "(diffusion_run)" in m for m in logs.messages)


def test_profile_needs_a_call_number():
    with pytest.raises(ValueError, match="expected the number of a call"):
        plugin.Plugin(FakeComIn(), environ={plugin.PROFILE_ENV: "all"})
