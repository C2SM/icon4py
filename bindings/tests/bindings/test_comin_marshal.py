# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the icon4py ComIn plugin against a fake 'comin' module and a fake ICON side.

'FakeComIn' mimics the parts of ComIn's Python API that the plugin uses (ComIn 6f4029b6,
'plugins/python_adapter/comin.py'), including the access-scope rule and the exception of
'current_get_domain_id' outside the domain loop. 'FakeIcon' does what ICON's ComIn backend
('mo_icon4py_comin.f90') does: expose the arguments, fire the secondary constructor, check the
handshake, bind per pass and per call, fire the entry points and check the acknowledgement;
and what 'mo_nh_stepping' does around a diffusion call: EP_ATM_INTEGRATE_START at the start of
a time step, the dynamics' EP_ATM_DYCORE_SOLVE_NH_BEFORE/_AFTER before a regular call, the
pair EP_ATM_DYCORE_DIFFUSION_BEFORE/_AFTER around the dispatcher.
ICON's namelist output is an excerpt of a real run's ('comin_run_dir/'; ComIn diffusion
SUBSTITUTE of 'mch_icon-ch1_small'), copied into each test's working directory.
"""

import io
import logging
import pathlib
import re
import shutil
import sys
import types
from typing import Any

import cffi
import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import (
    common as wrapper_common,
    config as wrapper_config,
    diffusion_wrapper,
    grid_wrapper,
    icon4py_export,
)
from icon4py.bindings.comin import _config, _dual, _marshal, _views, plugin
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
    "EP_ATM_DIFFUSION_ENTER": 74,
    "EP_ATM_DIFFUSION_LEAVE": 75,
}


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

    def is_used(self, descriptor: tuple[str, int]) -> bool:
        return any(d == descriptor for _, d in self.contexts)

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


class IconFinish(RuntimeError):
    pass


def fortran_buffer(shape: tuple[int, ...], dtype, fill=0) -> np.ndarray:
    """A 5-D Fortran-order buffer, like a Fortran array padded to ComIn's 5 dimensions."""
    padded = shape + (1,) * (_marshal.MAX_RANK - len(shape))
    buffer = np.zeros(padded, dtype=dtype, order="F")
    rank_view(buffer, len(shape))[...] = fill
    return buffer


def rank_view(buffer: np.ndarray, rank: int) -> np.ndarray:
    return buffer[(slice(None),) * rank + (0,) * (_marshal.MAX_RANK - rank)]


def comin_datatype(kind) -> int:
    return {py2fgen.FLOAT64: DOUBLE, py2fgen.FLOAT32: FLOAT}.get(kind, INT)


def to_metadata(value: Any) -> Any:
    return int(value) if isinstance(value, bool) else value  # ICON stores a logical as 0/1


class FakeIcon:
    """ICON's ComIn backend in miniature ('mo_icon4py_comin.f90')."""

    def __init__(
        self, comin: FakeComIn, functions: dict[str, plugin.FunctionEntry], device_xp: Any = None
    ):
        self.comin = comin
        self.functions = functions
        self.device_xp = device_xp
        self.live: dict[str, Any] = {}
        """The arrays the functions work on: the device copy of MAYBE_DEVICE arguments on GPU."""
        self.carriers: dict[str, np.ndarray] = {}
        self.pass_count = 0
        self.call_count = 0
        self.calls_in_pass = 0
        self.after_initial = False

    def expose(self, name: str, arrays: dict[str, np.ndarray | None], scalars: dict[str, Any]):
        """'arrays' holds the 5-D buffers; a missing or 'None' entry is an absent optional."""
        for param, descriptor in self.functions[name].exported.param_descriptors.items():
            if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
                buffer = arrays.get(param)
                device = self.device_copy(buffer, descriptor)
                self.live[param] = buffer if device is None else device
                self.comin.add(
                    _marshal.variable_name(name, param),
                    buffer,
                    device,
                    datatype=comin_datatype(descriptor.dtype),
                    icon4py_rank=descriptor.rank,
                    icon4py_shape=[0] * 5 if buffer is None else list(buffer.shape),
                    icon4py_present=0 if buffer is None else 1,
                )
        carrier = fortran_buffer((1,), np.int32, fill=-1)
        self.carriers[name] = carrier
        self.comin.add(
            _marshal.carrier_name(name),
            carrier,
            datatype=INT,
            icon4py_shape=[1] * 5,
            **{_marshal.scalar_key(k): to_metadata(v) for k, v in scalars.items()},
        )

    def device_copy(self, buffer: np.ndarray | None, descriptor: py2fgen.ArrayParamDescriptor):
        if self.device_xp is None or buffer is None:
            return None
        if descriptor.memory_space == py2fgen.MemorySpace.HOST:
            return None
        return self.device_xp.array(buffer, order="F")

    def rebind(self, name: str, param: str, buffer: np.ndarray) -> Any:
        """'comin_var_set_cptr': same shape, other memory; returns the live array."""
        descriptor = (_marshal.variable_name(name, param), 1)
        assert buffer.shape == self.comin.buffers[descriptor].shape
        self.comin.buffers[descriptor] = buffer
        device = self.device_copy(buffer, self.functions[name].exported.param_descriptors[param])
        if device is not None:
            self.comin.device_buffers[descriptor] = device
        return buffer if device is None else device

    def host(self, array: Any) -> np.ndarray:
        return np.asarray(array) if self.device_xp is None else self.device_xp.asnumpy(array)

    def set_scalar(self, name: str, key: str, value: Any) -> None:
        self.comin.meta[_marshal.carrier_name(name), 1][key] = to_metadata(value)

    def secondary_constructor(self) -> None:
        self.comin.fire("EP_SECONDARY_CONSTRUCTOR")

    def handshake(self) -> None:
        for name in self.functions:
            if not self.comin.is_used((_marshal.carrier_name(name), 1)):
                raise IconFinish("icon4py ComIn plugin did not request its variables")

    def init_names(self) -> list[str]:
        return [n for n, e in self.functions.items() if e.ack_key == _marshal.PASS_KEY]

    def new_pass(self) -> None:
        self.pass_count += 1
        self.calls_in_pass = 0
        for name in self.init_names():
            self.set_scalar(name, _marshal.PASS_KEY, self.pass_count)
        self.handshake()
        self.comin.fire("EP_ATM_TIMELOOP_BEFORE")

    def dynamics(self, domain_id: int = 1, substeps: int = 2) -> None:
        """'perform_dyn_substepping': the SOLVE_NH pair per substep."""
        for _ in range(substeps):
            self.comin.fire("EP_ATM_DYCORE_SOLVE_NH_BEFORE", domain_id)
            self.comin.fire("EP_ATM_DYCORE_SOLVE_NH_AFTER", domain_id)

    def diffusion_call(
        self,
        dtime: float,
        linit: bool,
        domain_id: int = 1,
        *,
        delegate: bool = True,
        carrier_linit: bool | None = None,
    ) -> None:
        """
        One diffusion call as 'mo_nh_stepping' makes it: the initial one ('linit') opens a time
        step (INTEGRATE_START); a regular one follows the dynamics, in a new time step unless it
        follows the initial one. Then the pair DIFFUSION_BEFORE/_AFTER around the dispatcher,
        which binds the arguments and fires EP_ATM_DIFFUSION_ENTER ('delegate'; ICON does not
        call it without 'lhdiff_vn'); 'carrier_linit' overrides the flag it binds (a wrong one).
        """
        if linit or not self.after_initial:
            self.comin.fire("EP_ATM_INTEGRATE_START", domain_id)
        if not linit:
            self.dynamics(domain_id)
        self.after_initial = linit
        self.comin.timestep = dtime
        self.comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", domain_id)
        if delegate:
            self.dispatch(dtime, linit if carrier_linit is None else carrier_linit, domain_id)
        self.comin.fire("EP_ATM_DYCORE_DIFFUSION_AFTER", domain_id)

    def dispatch(self, dtime: float, linit: bool, domain_id: int = 1) -> None:
        """ICON's dispatcher in ComIn SUBSTITUTE or VERIFY: bind the call, fire ENTER, check."""
        self.call_count += 1
        self.calls_in_pass += 1
        self.set_scalar("diffusion_run", _marshal.CALL_COUNT_KEY, self.call_count)
        self.set_scalar("diffusion_run", _marshal.scalar_key("dtime"), dtime)
        self.set_scalar("diffusion_run", _marshal.scalar_key("linit"), linit)
        ctl = self.carriers["diffusion_run"]
        ctl[...] = -1
        if self.calls_in_pass == 1:
            for name in self.init_names():
                if self.carriers[name].item() != self.pass_count:
                    raise IconFinish(f"icon4py ComIn plugin did not run {name}")
        self.comin.fire("EP_ATM_DIFFUSION_ENTER", domain_id)
        if ctl.item() != self.call_count:
            raise IconFinish("icon4py ComIn plugin did not run diffusion_run")
        self.comin.fire("EP_ATM_DIFFUSION_LEAVE", domain_id)


# ---- toy functions: every kind of parameter, cheap to call ------------------------------------

CALLS: list[tuple[str, dict[str, Any]]] = []


@icon4py_export.export
def toy_init(  # noqa: PLR0917 [too-many-positional-arguments]
    starts: grid_wrapper.NumpyInt32Array1D,
    mask: grid_wrapper.NumpyBoolArray1D,
    area: fa.CellField[gtx.float64],
    coeff: wrapper_common.Float64Array3D,
    idx: wrapper_common.OptionalInt32Array2D,
    empty: wrapper_common.OptionalFloat64Array1D,
    n: gtx.int32,
    flag: bool,
    x: gtx.float64,
) -> None:
    CALLS.append(
        (
            "init",
            dict(
                starts=starts,
                mask=mask,
                area=area,
                coeff=coeff,
                idx=idx,
                empty=empty,
                n=n,
                flag=flag,
                x=x,
            ),
        )
    )


@icon4py_export.export
def toy_run(
    w: fa.CellKField[gtx.float64],
    vn: fa.EdgeKField[gtx.float64],
    opt: fa.CellKField[gtx.float64] | None,
    dtime: gtx.float64,
    linit: bool,
) -> None:
    CALLS.append(("run", dict(w=w, vn=vn, opt=opt, dtime=dtime, linit=linit)))
    w.ndarray[...] += dtime
    vn.ndarray[...] *= 2.0


TOY_FUNCTIONS = {
    # the ComIn names must be the real ones: the idle check looks for 'icon4py_diffusion_run_ctl'
    "grid_init": plugin.FunctionEntry(toy_init, "EP_ATM_TIMELOOP_BEFORE", False, _marshal.PASS_KEY),
    "diffusion_run": plugin.FunctionEntry(
        toy_run, "EP_ATM_DIFFUSION_ENTER", True, _marshal.CALL_COUNT_KEY
    ),
}
NC, NE, NLEV = 6, 9, 4


def toy_icon(
    comin: FakeComIn, device_xp: Any = None, icon_variables: bool = False
) -> tuple[FakeIcon, dict[str, np.ndarray]]:
    """The toy functions' variables; with 'icon_variables', also ICON's own 'w' and 'vn' on the
    same memory as the arguments (as in SUBSTITUTE at the initial call site)."""
    icon = FakeIcon(comin, TOY_FUNCTIONS, device_xp)
    buffers = {
        "starts": fortran_buffer((NC,), np.int32, fill=np.arange(1, NC + 1)),
        "mask": fortran_buffer((NC,), np.int32, fill=[1, 1, 0, 1, 0, 0]),
        "area": fortran_buffer((NC,), np.float64, fill=2.5),
        "coeff": fortran_buffer((NC, 2, 3), np.float64, fill=0.5),
        "empty": fortran_buffer((0,), np.float64),
        "w": fortran_buffer((NC, NLEV + 1), np.float64, fill=1.0),
        "vn": fortran_buffer((NE, NLEV), np.float64, fill=3.0),
    }
    init = {k: buffers[k] for k in ("starts", "mask", "area", "coeff", "empty")}
    icon.expose("grid_init", init, dict(n=NC, flag=True, x=0.25))
    icon.expose("diffusion_run", {k: buffers[k] for k in ("w", "vn")}, dict(dtime=0.0, linit=False))
    if icon_variables:
        for name in ("w", "vn"):
            device = icon.live[name] if device_xp is not None else None
            comin.add(name, buffers[name], device, datatype=DOUBLE)
    return icon, buffers


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


def all_old(functions) -> dict[str, dict[str, _dual.Source]]:
    """A source table that takes every argument from the old route (the toy functions)."""
    return {
        name: {p: _dual.Source("B", _dual.OLD) for p in entry.exported.param_descriptors}
        for name, entry in functions.items()
    }


def make_plugin(comin: FakeComIn, functions=TOY_FUNCTIONS, **environ: str) -> plugin.Plugin:
    sources = all_old(functions) if functions is TOY_FUNCTIONS else plugin.SOURCES
    instance = plugin.Plugin(comin, functions=functions, environ=environ, sources=sources)
    instance.register()
    return instance


# ---- signatures of the real functions ----------------------------------------------------------

REAL = {
    "grid_init": grid_wrapper.grid_init,
    "diffusion_init": diffusion_wrapper.diffusion_init,
    "diffusion_run": diffusion_wrapper.diffusion_run,
}


def test_real_signatures():
    signatures = {name: _marshal.signature(name, fn) for name, fn in REAL.items()}
    counts = {n: (len(s.arrays), len(s.scalars)) for n, s in signatures.items()}
    assert counts == {"grid_init": (43, 13), "diffusion_init": (13, 26), "diffusion_run": (9, 2)}

    arrays = [a for s in signatures.values() for a in s.arrays]
    assert {a.name for a in arrays if a.is_host} == {
        *(f"{e}_{b}" for e in ("cell", "vertex", "edge") for b in ("starts", "ends")),
        *(f"{e}_owner_mask" for e in "cev"),
        *(f"{e}_glb_index" for e in "cev"),
    }
    assert {a.name for a in arrays if a.descriptor.is_optional} == {
        "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef", "hdef_ic", "div_ic", "dwdx", "dwdy"
    }  # fmt: skip
    raw = {a.name for a in arrays if a.dims is None}
    assert raw == {a.name for a in arrays if a.is_host} | {
        "rbf_vec_coeff_v", "zd_cellidx", "zd_vertidx", "zd_intcoef", "zd_diffcoef"
    }  # fmt: skip
    assert all(len(a.variable) <= 64 for a in arrays)  # COMIN_MAX_LEN_VAR_NAME
    assert max(len(a.variable) for a in arrays) == len(
        "icon4py_grid_init_inverse_primal_edge_lengths"
    )

    run = signatures["diffusion_run"]
    assert run.carrier == "icon4py_diffusion_run_ctl"
    assert [a.variable for a in run.arrays][:2] == [
        "icon4py_diffusion_run_w",
        "icon4py_diffusion_run_vn",
    ]
    assert [s.key for s in run.scalars] == ["icon4py_dtime", "icon4py_linit"]
    assert run.function is diffusion_wrapper.diffusion_run.__wrapped__
    assert dict(zip((a.name for a in run.arrays), (a.dims for a in run.arrays)))["vn"] == (
        dims.EdgeDim,
        dims.KDim,
    )


# ---- binding and marshalling of the real functions ---------------------------------------------

LOCAL_SIZE = 3
SIZES = {dims.CellDim: NC, dims.EdgeDim: NE, dims.VertexDim: 5, dims.KDim: NLEV}
RAW_SHAPES = {1: (NC,), 2: (4, 5), 3: (5, 2, 6)}
ZERO_EXTENT = {
    "zd_cellidx": (4, 0),
    "zd_vertidx": (4, 0),
    "zd_intcoef": (4, 0),
    "zd_diffcoef": (0,),
}
ABSENT = {"dwdx", "dwdy"}


def real_icon(comin: FakeComIn) -> tuple[FakeIcon, dict[str, dict[str, np.ndarray]]]:
    functions = {
        name: plugin.FunctionEntry(fn, e.entry_point, e.inout, e.ack_key)
        for (name, fn), e in zip(REAL.items(), plugin.FUNCTIONS.values(), strict=True)
    }
    icon = FakeIcon(comin, functions)
    all_buffers = {}
    rng = np.random.default_rng(42)
    for name, fn in REAL.items():
        signature = _marshal.signature(name, fn)
        buffers = {}
        for a in signature.arrays:
            if a.name in ABSENT:
                continue
            if a.name in ZERO_EXTENT:
                shape = ZERO_EXTENT[a.name]
            elif a.dims is not None:
                shape = tuple(SIZES.get(d, LOCAL_SIZE) for d in a.dims)
            else:
                shape = RAW_SHAPES[a.descriptor.rank]
            dtype = np.float64 if a.descriptor.dtype == py2fgen.FLOAT64 else np.int32
            fill = rng.integers(0, 2, size=shape) if a.descriptor.dtype == py2fgen.BOOL else 7
            buffers[a.name] = fortran_buffer(shape, dtype, fill=fill)
        scalars = {}
        for s in signature.scalars:
            kind = s.descriptor.dtype
            scalars[s.name] = (
                True if kind == py2fgen.BOOL else (3 if kind == py2fgen.INT32 else 0.1)
            )
        icon.expose(name, buffers, scalars)
        all_buffers[name] = buffers
    # ICON's own variables of the names of diffusion_run's arrays, on the same memory (as in
    # SUBSTITUTE); ICON has no 'dwdx' and 'dwdy' here (ABSENT)
    for name, buffer in all_buffers["diffusion_run"].items():
        comin.add(name, buffer, datatype=DOUBLE)
    # ICON's variables that diffusion_init gets (plugin.STATIC_VARIABLES), on the same memory
    for param, icon_name in plugin.STATIC_VARIABLES["diffusion_init"].items():
        buffer = all_buffers["diffusion_init"][param]
        comin.add(icon_name, buffer, datatype=DOUBLE if buffer.dtype == np.float64 else INT)
    return icon, all_buffers


@pytest.mark.parametrize("has_device", [False, True], ids=["host", "device"])
def test_real_functions_bind(monkeypatch, has_device):
    if has_device:
        monkeypatch.setattr(plugin, "cp", types.ModuleType("cupy"))  # selected, not used
    comin = FakeComIn(has_device=has_device)
    icon, _ = real_icon(comin)
    instance = make_plugin(comin, functions=icon.functions)
    icon.secondary_constructor()
    assert instance.active
    icon.handshake()  # all three carriers requested

    device = FLAG_DEVICE if has_device else 0
    init_ep, enter_ep, before_ep = (
        ENTRY_POINTS["EP_ATM_TIMELOOP_BEFORE"],
        ENTRY_POINTS["EP_ATM_DIFFUSION_ENTER"],
        ENTRY_POINTS["EP_ATM_DYCORE_DIFFUSION_BEFORE"],
    )
    expected = {}
    for name, fn in REAL.items():
        ep = enter_ep if name == "diffusion_run" else init_ep
        # SUBSTITUTE: diffusion_run runs at DIFFUSION_BEFORE on ICON's variables, so its
        # arguments are only read (compared) at ENTER
        for a in _marshal.signature(name, fn).arrays:
            expected[ep, (a.variable, 1)] = FLAG_READ | device
        expected[ep, (_marshal.carrier_name(name), 1)] = FLAG_READ | FLAG_WRITE | device
    icon_variables = [n for n in ("w", "vn", "exner", "theta_v", "rho", "hdef_ic", "div_ic")]
    for name in icon_variables:
        expected[before_ep, (name, 1)] = FLAG_READ | FLAG_WRITE | device  # computes on them
        expected[enter_ep, (name, 1)] = FLAG_READ | device  # the time-level check (strict)
    for name in plugin.STATIC_VARIABLES["diffusion_init"].values():
        expected[init_ep, (name, 1)] = FLAG_READ | device  # diffusion_init gets them
    assert comin.contexts == expected  # nothing else, never SYNC_HALO
    assert len(expected) == 65 + 3 + 2 * 7 + 6


def test_real_functions_marshal(monkeypatch):
    comin = FakeComIn()
    icon, buffers = real_icon(comin)
    instance = make_plugin(comin, functions=icon.functions)
    icon.secondary_constructor()
    bound = instance._bound

    for name in REAL:
        comin.current_ep = ENTRY_POINTS[icon.functions[name].entry_point]
        comin.accessed.clear()
        kwargs = _marshal.arguments(comin, bound[name], None)
        signature = bound[name].signature
        assert list(kwargs) == [a.name for a in signature.arrays] + [
            s.name for s in signature.scalars
        ]
        for a in signature.arrays:
            value = kwargs[a.name]
            descriptor = (a.variable, 1)
            if a.name in ABSENT:
                assert value is None
                assert (comin.current_ep, descriptor) not in comin.accessed
                continue
            buffer = rank_view(buffers[name][a.name], a.descriptor.rank)
            if a.name in ZERO_EXTENT:
                # host address of a zero-size array: not NULL, so an empty array (as py2fgen)
                assert value.shape == ZERO_EXTENT[a.name] and value.size == 0
                assert (comin.current_ep, descriptor) in comin.accessed
            elif a.descriptor.dtype == py2fgen.BOOL:
                assert value.dtype == np.bool_ and isinstance(value, np.ndarray)
                np.testing.assert_array_equal(value, buffer != 0)
            elif a.dims is not None:
                assert isinstance(value, gtx.Field)
                assert tuple(value.domain.dims) == a.dims
                assert (
                    value.ndarray.shape == buffer.shape and value.ndarray.strides == buffer.strides
                )
                assert _views.data_ptr(value.ndarray) == _views.data_ptr(buffer)
            else:
                assert isinstance(value, np.ndarray) and value.dtype == buffer.dtype
                assert _views.data_ptr(value) == _views.data_ptr(buffer)
                assert value.strides == buffer.strides
        for s in signature.scalars:
            expected_type = {py2fgen.BOOL: bool, py2fgen.INT32: int, py2fgen.FLOAT64: float}
            assert type(kwargs[s.name]) is expected_type[s.descriptor.dtype]
    comin.current_ep = None


@gtx.field_operator
def _double(a: fa.EdgeKField[gtx.float64]) -> fa.EdgeKField[gtx.float64]:
    return 2.0 * a


@gtx.program
def double(a: fa.EdgeKField[gtx.float64], out: fa.EdgeKField[gtx.float64]):
    _double(a, out=out)


def test_real_diffusion_run_arguments_write_through():
    comin = FakeComIn()
    icon, buffers = real_icon(comin)
    instance = make_plugin(comin, functions=icon.functions)
    icon.secondary_constructor()
    comin.current_ep = ENTRY_POINTS["EP_ATM_DIFFUSION_ENTER"]
    kwargs = _marshal.arguments(comin, instance._bound["diffusion_run"], None)
    comin.current_ep = None

    double(kwargs["vn"], kwargs["vn"], offset_provider={})  # a GT4Py program, in place
    kwargs["w"].ndarray[...] = -1.0
    assert np.all(buffers["diffusion_run"]["vn"] == 14.0)
    assert np.all(buffers["diffusion_run"]["w"] == -1.0)


def test_access_outside_the_requested_entry_point_fails():
    comin = FakeComIn()
    icon, _ = real_icon(comin)
    instance = make_plugin(comin, functions=icon.functions)
    icon.secondary_constructor()
    comin.current_ep = ENTRY_POINTS["EP_ATM_TIMELOOP_BEFORE"]  # run arguments belong to ENTER
    with pytest.raises(FakeComInError, match="out of requested scope"):
        _marshal.arguments(comin, instance._bound["diffusion_run"], None)


# ---- metadata checks ---------------------------------------------------------------------------


def test_metadata_errors_are_collected(wrappers):
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    meta = comin.meta
    meta["icon4py_grid_init_area", 1]["datatype"] = INT
    meta["icon4py_grid_init_coeff", 1]["icon4py_rank"] = 2
    meta["icon4py_grid_init_starts", 1]["icon4py_shape"] = [NC, 2, 1, 1, 1]
    meta["icon4py_grid_init_mask", 1]["icon4py_present"] = 0
    del meta["icon4py_grid_init_ctl", 1]["icon4py_n"]
    meta["icon4py_grid_init_ctl", 1]["icon4py_x"] = 1  # an integer for a real
    meta["icon4py_grid_init_ctl", 1]["icon4py_flag"] = 2
    meta["icon4py_diffusion_run_ctl", 1]["datatype"] = DOUBLE
    del meta["icon4py_diffusion_run_w", 1]["icon4py_shape"]
    del comin.buffers["icon4py_diffusion_run_vn", 1]
    make_plugin(comin)

    with pytest.raises(RuntimeError) as error:
        icon.secondary_constructor()
    message = str(error.value)
    assert "10 problem(s)" in message
    for fragment in (
        "'icon4py_grid_init_area': 'datatype' is 2, expected COMIN_VAR_DATATYPE_DOUBLE",
        "'icon4py_grid_init_coeff': 'icon4py_rank' is 2, expected 3",
        "'icon4py_grid_init_starts': 'icon4py_shape' [6, 2, 1, 1, 1] pads a rank-1 array",
        "'icon4py_grid_init_mask': the argument is not optional, but ICON marks it absent",
        "scalar 'n' is missing (metadata 'icon4py_n')",
        "Metadata 'icon4py_x': expected a FLOAT64 value, got 1",
        "Metadata 'icon4py_flag': expected a BOOL value, got 2",
        "'icon4py_diffusion_run_ctl': 'datatype' is 0, expected COMIN_VAR_DATATYPE_INT",
        "'icon4py_diffusion_run_w': 'icon4py_shape' is missing, expected 5 extents",
        "'icon4py_diffusion_run_vn': ICON does not expose this variable",
    ):
        assert fragment in message


def test_absent_optional_shape_is_not_checked():
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    comin.meta["icon4py_grid_init_idx", 1]["icon4py_shape"] = "anything"
    instance = make_plugin(comin)
    icon.secondary_constructor()
    assert instance.active


# ---- the plugin's life cycle -------------------------------------------------------------------


def test_register_logs_setup(logs):
    make_plugin(FakeComIn())
    assert logs.messages[0].startswith(f"source sha1 {plugin.source_sha1()}, icon4py.bindings")
    assert logs.messages[1] == (
        "ICON mode SUBSTITUTE (icon4py_interface=1, luse_icon4py_diffusion=T, icon4py_mode=1):"
        " this plugin computes the horizontal diffusion of domain 1 at EP_ATM_DIFFUSION_ENTER;"
        " ICON skips its own"
    )
    assert logs.messages[2] == "time-level check strict: the default in SUBSTITUTE"
    assert (
        "entry points EP_FINISH=72, EP_DESTRUCTOR=73, EP_ATM_DIFFUSION_ENTER=74,"
        " EP_ATM_DIFFUSION_LEAVE=75" in logs.messages
    )
    assert (
        "counting the callbacks at EP_ATM_DYCORE_DIFFUSION_BEFORE=20,"
        " EP_ATM_DYCORE_DIFFUSION_AFTER=21, EP_ATM_DYCORE_SOLVE_NH_BEFORE=22,"
        " EP_ATM_DYCORE_SOLVE_NH_AFTER=23; time-level check strict" in logs.messages
    )
    assert all(r.levelno == logging.INFO for r in logs.records)


def test_register_warns_about_other_entry_point_numbers(logs):
    make_plugin(FakeComIn(entry_points={**ENTRY_POINTS, "EP_ATM_DIFFUSION_ENTER": 70}))
    assert [r.levelno for r in logs.records if "entry point numbers differ" in r.message] == [
        logging.WARNING
    ]


@pytest.mark.parametrize(
    "missing",
    ["EP_ATM_DIFFUSION_ENTER", "EP_ATM_DYCORE_DIFFUSION_BEFORE", "EP_ATM_DYCORE_SOLVE_NH_AFTER"],
)
def test_register_needs_the_new_entry_points(missing):
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
    assert "icon4py-comin: entry points EP_FINISH=72," in capsys.readouterr().out


@pytest.mark.parametrize("rank", [0, 1])
def test_log_format_and_ranks(rank):
    stream = io.StringIO()
    handler = plugin.configure_logging(rank, stream)
    assert PACKAGE_LOGGER.propagate is False and handler in PACKAGE_LOGGER.handlers
    log = logging.getLogger(_marshal.__name__)  # any logger of the package
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


def test_idle_in_mode_off(calls, wrappers, logs, icon_run_dir):
    set_mode(icon_run_dir, 0)
    comin = FakeComIn()
    comin.add("vn", fortran_buffer((NE, NLEV), np.float64), datatype=DOUBLE)
    # report: the misfired ENTER below (ICON never fires it in OFF) is logged, not raised
    instance = make_plugin(comin, **{_dual.MODE_ENV: "report"})
    comin.fire("EP_SECONDARY_CONSTRUCTOR")
    assert not instance.active
    assert comin.contexts == {}
    comin.fire("EP_ATM_TIMELOOP_BEFORE")
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)
    comin.fire("EP_ATM_DIFFUSION_ENTER", 1)
    for _ in range(2):
        comin.fire("EP_ATM_DYCORE_SOLVE_NH_BEFORE", 1)
    comin.fire("EP_DESTRUCTOR")
    assert calls == [] and comin.accessed == []
    assert (diffusion_wrapper.granule, grid_wrapper.grid_state) == wrappers
    assert (
        "entry point callbacks: EP_ATM_DYCORE_DIFFUSION_BEFORE 1, EP_ATM_DYCORE_DIFFUSION_AFTER 0,"
        " EP_ATM_DYCORE_SOLVE_NH_BEFORE 2, EP_ATM_DYCORE_SOLVE_NH_AFTER 0 (domains [1])"
        in logs.messages
    )
    assert wrapper_config.WAIT_FOR_COMPILATION is False
    switches = "icon4py_interface=1, luse_icon4py_diffusion=T, icon4py_mode=0"
    assert logs.messages[1] == (
        f"ICON mode OFF ({switches}): ICON computes the horizontal diffusion;"
        " this plugin stays idle"
    )
    assert "time-level check off: the plugin is idle" in logs.messages
    assert f"idle: ICON mode OFF ({switches}); requested nothing." in logs.messages
    assert (
        f"mode check: ICON's icon4py variables exposed no, expected no ({switches}): identical"
        in logs.messages
    )
    # ENTER must not fire at all in OFF: the destructor compares the counts
    assert [r.levelname for r in logs.records if "but the plugin is idle" in r.message] == [
        "WARNING"
    ]
    (line,) = [r for r in logs.records if "mode check: EP_ATM_DIFFUSION_ENTER" in r.message]
    assert line.levelname == "WARNING"
    assert "callbacks for domain 1: 1, expected 0 (ICON mode OFF;" in line.message


def cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No GPU.")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No GPU.")
    return cp


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_life_cycle(calls, wrappers, logs, on_device):
    device_xp = cupy_or_skip() if on_device else None
    comin = FakeComIn(has_device=on_device)
    icon, _ = toy_icon(comin, device_xp)
    live = icon.live
    instance = make_plugin(comin)
    icon.secondary_constructor()
    assert instance.active

    # pass 1: init at EP_ATM_TIMELOOP_BEFORE, acknowledged with the pass number
    icon.new_pass()
    assert wrapper_config.WAIT_FOR_COMPILATION is True
    assert [c[0] for c in calls] == ["init"]
    init = calls[0][1]
    assert isinstance(init["starts"], np.ndarray) and init["starts"].dtype == np.int32
    assert list(init["starts"]) == list(range(1, NC + 1))
    assert isinstance(init["mask"], np.ndarray)
    assert list(init["mask"]) == [True, True, False, True, False, False]
    assert tuple(init["area"].domain.dims) == (dims.CellDim,)
    assert _views.data_ptr(init["area"].ndarray) == _views.data_ptr(live["area"])
    assert init["coeff"].shape == (NC, 2, 3) and _views.data_ptr(init["coeff"]) == _views.data_ptr(
        live["coeff"]
    )
    assert init["idx"] is None
    # a zero-size optional: NULL device address on the GPU, so None (as py2fgen); on the host an
    # empty array
    assert init["empty"] is None if on_device else init["empty"].size == 0
    assert (init["n"], init["flag"], init["x"]) == (NC, True, 0.25)
    assert icon.carriers["grid_init"].item() == 1

    # two calls; the second on other memory (SUBSTITUTE alternates nnow/nnew, VERIFY copies)
    icon.diffusion_call(dtime=10.0, linit=True)
    assert np.all(icon.host(live["w"]) == 11.0) and np.all(icon.host(live["vn"]) == 6.0)
    other_w = icon.rebind(
        "diffusion_run", "w", fortran_buffer((NC, NLEV + 1), np.float64, fill=100.0)
    )
    icon.diffusion_call(dtime=5.0, linit=False)
    assert np.all(icon.host(other_w) == 105.0) and np.all(icon.host(live["w"]) == 11.0)
    assert np.all(icon.host(live["vn"]) == 12.0)
    assert [(c[0], c[1]["dtime"], c[1]["linit"]) for c in calls[1:]] == [
        ("run", 10.0, True),
        ("run", 5.0, False),
    ]
    assert calls[1][1]["opt"] is None
    assert icon.carriers["diffusion_run"].item() == 2

    # pass 2 (e.g. iterative IAU): init again
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=False)
    assert [c[0] for c in calls] == ["init", "run", "run", "init", "run"]
    assert icon.carriers["grid_init"].item() == 2 and icon.carriers["diffusion_run"].item() == 3

    comin.fire("EP_DESTRUCTOR")
    assert diffusion_wrapper.granule is None and grid_wrapper.grid_state is None
    assert not instance.active

    messages = logs.messages
    assert f"device {on_device}" in messages[0]
    assert messages.count("pointer identity OK (2 fields)") == 1
    assert sum("sums before -> after" in m for m in messages) == 1
    assert "w 30.0 -> 330.0 (changed)" in logs.text and "vn 108.0 -> 216.0 (changed)" in logs.text
    for n in (1, 2, 3):
        assert f"diffusion_run call {n} done." in messages
    assert "grid_init done (icon4py_pass 2)." in messages
    empty = "empty None (NULL address, ICON extent (0,))" if on_device else "empty (0,)"
    assert f"grid_init optional arguments: idx None (absent), {empty}" in messages
    assert "diffusion_run optional arguments: opt None (absent)" in messages
    assert not any(m.startswith("timing") for m in messages)  # off by default
    # the toy ICON exposes no 'w' and 'vn' of its own: nothing to compare
    assert (
        "time-level check strict: requested ICON's nothing (READ"
        f"{' | DEVICE' if on_device else ''}) at EP_ATM_DYCORE_DIFFUSION_BEFORE and"
        " EP_ATM_DIFFUSION_ENTER; not exposed: w vn" in messages
    )
    assert not any(m.startswith("time-level check call") for m in messages)
    assert (
        "entry point callbacks: EP_ATM_DYCORE_DIFFUSION_BEFORE 3, EP_ATM_DYCORE_DIFFUSION_AFTER 3,"
        " EP_ATM_DYCORE_SOLVE_NH_BEFORE 4, EP_ATM_DYCORE_SOLVE_NH_AFTER 4 (domains [1])" in messages
    )
    assert (
        "per-call state: 3 diffusion calls of domain 1 at EP_ATM_DYCORE_DIFFUSION_BEFORE,"
        " 0 computed there, 0 without lhdiff_vn; linit: initial 1, regular 2, ICON's condition"
        " agrees at 3 of 3; EP_ATM_INTEGRATE_START 2 (domain 1)" in messages
    )
    assert messages[-1] == (
        "mode check: EP_ATM_DIFFUSION_ENTER callbacks for domain 1: 3, expected 3 (ICON mode"
        " SUBSTITUTE; EP_ATM_DYCORE_DIFFUSION_BEFORE 3, lhdiff_vn T): identical"
    )


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_timelevel_check_report(calls, wrappers, logs, on_device):
    device_xp = cupy_or_skip() if on_device else None
    comin = FakeComIn(has_device=on_device)
    icon, _ = toy_icon(comin, device_xp, icon_variables=True)
    make_plugin(comin, **{plugin.TIMELEVEL_CHECK_ENV: "report"})
    icon.secondary_constructor()
    flags = FLAG_READ | (FLAG_DEVICE if on_device else 0)
    for entry_point in ("EP_ATM_DYCORE_DIFFUSION_BEFORE", "EP_ATM_DIFFUSION_ENTER"):
        for name in ("w", "vn"):
            assert comin.contexts[ENTRY_POINTS[entry_point], (name, 1)] == flags
    icon.new_pass()
    # initial call site: the arguments are ICON's arrays, which ComIn's 'w' and 'vn' show
    icon.diffusion_call(dtime=10.0, linit=True)
    # regular call site, ComIn's pointers not re-targeted: the argument is another time level,
    # ComIn's 'w' is stale
    icon.rebind("diffusion_run", "w", fortran_buffer((NC, NLEV + 1), np.float64, fill=100.0))
    icon.diffusion_call(dtime=5.0, linit=False)
    comin.fire("EP_DESTRUCTOR")
    assert [c[0] for c in calls] == ["init", "run", "run"]  # report mode: the run goes on
    messages = logs.messages
    before, after = "EP_ATM_DYCORE_DIFFUSION_BEFORE", "EP_ATM_DYCORE_DIFFUSION_AFTER"
    assert "time-level check report: set by ICON4PY_COMIN_TIMELEVEL_CHECK" in messages
    assert (
        "time-level check report: requested ICON's w vn"
        f" (READ{' | DEVICE' if on_device else ''}) at {before} and EP_ATM_DIFFUSION_ENTER"
        in messages
    )
    assert (
        "time-level check call 1 initial: 2 checked, 2 identical, 0 differ;"
        f" {before} 1, {after} 0; ICON's pointers as at {before}: yes [report]" in messages
    )
    assert (
        "time-level check call 2 regular: 2 checked, 1 identical, 1 differ: w;"
        f" {before} 2, {after} 1; ICON's pointers as at {before}: yes [report]" in messages
    )
    assert (
        "time-level check summary: initial 1 calls, 1 all identical;"
        " regular 1 calls, 0 all identical [report]" in messages
    )
    assert (
        f"entry point callbacks: {before} 2, {after} 2, EP_ATM_DYCORE_SOLVE_NH_BEFORE 2,"
        " EP_ATM_DYCORE_SOLVE_NH_AFTER 2 (domains [1])" in messages
    )


def test_timelevel_check_strict_stops_before_the_granule(calls, wrappers):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    make_plugin(comin, **{plugin.TIMELEVEL_CHECK_ENV: "strict"})
    icon.secondary_constructor()
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    icon.rebind("diffusion_run", "vn", fortran_buffer((NE, NLEV), np.float64, fill=3.0))
    with pytest.raises(
        RuntimeError, match=r"time-level check, diffusion call 2: .* vn do not point"
    ):
        icon.diffusion_call(dtime=5.0, linit=False)
    assert [c[0] for c in calls] == ["init", "run"]


def test_timelevel_check_sees_a_moved_icon_pointer(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    make_plugin(comin, **{plugin.TIMELEVEL_CHECK_ENV: "strict"})
    icon.secondary_constructor()
    icon.new_pass()
    comin.fire("EP_ATM_INTEGRATE_START", 1)
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)
    # ICON re-targets its 'vn' after DIFFUSION_BEFORE (it must not), and binds the argument there
    other = fortran_buffer((NE, NLEV), np.float64, fill=3.0)
    comin.buffers["vn", 1] = other
    icon.rebind("diffusion_run", "vn", other)
    icon.set_scalar("diffusion_run", _marshal.CALL_COUNT_KEY, 1)
    icon.set_scalar("diffusion_run", _marshal.scalar_key("linit"), True)
    with pytest.raises(
        RuntimeError, match=r"time-level check, diffusion call 1: .* vn do not point"
    ):
        comin.fire("EP_ATM_DIFFUSION_ENTER", 1)
    assert [c[0] for c in calls] == ["init"]
    assert any(
        m.endswith("ICON's pointers as at EP_ATM_DYCORE_DIFFUSION_BEFORE: no: vn [strict]")
        for m in logs.messages
    )


def test_timelevel_check_off_requests_nothing(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    make_plugin(comin, **{plugin.TIMELEVEL_CHECK_ENV: "off"})
    icon.secondary_constructor()
    assert not any(descriptor[0] in ("w", "vn") for _, descriptor in comin.contexts)
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    lines = [m for m in logs.messages if m.startswith("time-level check")]
    assert lines == ["time-level check off: set by ICON4PY_COMIN_TIMELEVEL_CHECK"]


def test_timelevel_check_mode_is_checked():
    with pytest.raises(ValueError, match="ICON4PY_COMIN_TIMELEVEL_CHECK='on'"):
        plugin.Plugin(FakeComIn(), environ={plugin.TIMELEVEL_CHECK_ENV: "on"})


@pytest.mark.parametrize("domain_id", [-1, 2])
def test_enter_guard(calls, wrappers, logs, domain_id):
    comin = FakeComIn()
    icon, buffers = toy_icon(comin)
    make_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    comin.accessed.clear()

    with pytest.raises(IconFinish, match="did not run diffusion_run"):
        icon.diffusion_call(dtime=10.0, linit=True, domain_id=domain_id)
    assert [c[0] for c in calls] == ["init"]
    assert comin.accessed == []  # touched nothing
    assert icon.carriers["diffusion_run"].item() == -1
    assert np.all(buffers["w"] == 1.0)
    where = "outside the domain loop" if domain_id < 0 else f"for domain {domain_id}"
    warnings = [r.message for r in logs.records if r.levelno == logging.WARNING]
    assert warnings == [f"EP_ATM_DIFFUSION_ENTER fired {where}; ignored."]


def test_skip_ack(calls, wrappers, logs):
    comin = FakeComIn()
    icon, buffers = toy_icon(comin)
    make_plugin(comin, **{plugin.SKIP_ACK_ENV: "1"})
    icon.secondary_constructor()
    icon.new_pass()
    assert icon.carriers["grid_init"].item() == 1  # init still acknowledged

    with pytest.raises(IconFinish, match="did not run diffusion_run"):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in calls] == ["init", "run"]  # it ran, but did not acknowledge
    assert np.all(buffers["w"] == 11.0)
    warnings = [r.message for r in logs.records if r.levelno == logging.WARNING]
    assert warnings[0].startswith("ICON4PY_COMIN_TEST_SKIP_ACK=1")
    assert warnings[1].startswith("diffusion_run call 1 not acknowledged")


def test_handshake_fails_without_the_plugin():
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    icon.secondary_constructor()  # no plugin loaded: nobody requests the carriers
    with pytest.raises(IconFinish, match="did not request"):
        icon.new_pass()


def test_pointer_identity_check_catches_a_copy(calls, wrappers, monkeypatch):
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    make_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    # a helper that copies ('gtx.as_field') instead of 'field_view'
    monkeypatch.setattr(_views, "field_view", lambda array, dims: gtx.as_field(list(dims), array))
    with pytest.raises(RuntimeError, match="'w' of 'diffusion_run' does not alias ICON's array"):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert icon.carriers["diffusion_run"].item() == -1


def test_checks_off(calls, wrappers, logs):
    comin = FakeComIn()
    icon, buffers = toy_icon(comin)
    make_plugin(comin, **{plugin.CHECK_ENV: "0"})
    icon.secondary_constructor()
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    assert np.all(buffers["w"] == 11.0)
    assert "in-job consistency checks are off" in logs.text
    assert "pointer identity" not in logs.text and "sums before" not in logs.text


def test_missing_pass_number_is_reported(calls, wrappers):
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    make_plugin(comin)
    icon.secondary_constructor()
    with pytest.raises(RuntimeError, match="'icon4py_pass' = None"):
        comin.fire("EP_ATM_TIMELOOP_BEFORE")  # ICON did not bind: no pass number
    assert calls == []


def test_scalar_value():
    flag = _marshal.ScalarParam("f", "icon4py_f", py2fgen.ScalarParamDescriptor(dtype=py2fgen.BOOL))
    real = _marshal.ScalarParam(
        "r", "icon4py_r", py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT64)
    )
    count = _marshal.ScalarParam(
        "i", "icon4py_i", py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)
    )
    assert _marshal.scalar_value(flag, 1) is True and _marshal.scalar_value(flag, 0) is False
    assert _marshal.scalar_value(real, 0.1) == 0.1 and _marshal.scalar_value(count, 7) == 7
    for param, bad in (
        (flag, 2),
        (flag, 1.0),
        (real, 1),
        (real, True),
        (count, True),
        (count, 1.0),
    ):
        with pytest.raises(TypeError):
            _marshal.scalar_value(param, bad)


# ---- timing and profiling (ICON4PY_COMIN_TIMING, ICON4PY_COMIN_PROFILE) ------------------------


def test_timing(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    make_plugin(comin, **{plugin.TIMING_ENV: "1"})
    icon.secondary_constructor()
    icon.new_pass()
    for n in range(3):
        icon.diffusion_call(dtime=1.0, linit=n == 0)
    comin.fire("EP_DESTRUCTOR")

    timing = [m for m in logs.messages if m.startswith("timing")]
    assert timing[0].startswith("timing: Python threads MainThread")
    phases = ["guard", "ack", "args", "sums", "presync", "run", "sync", "check", "done"]
    for n, line in zip((1, 2, 3), timing[1:4], strict=True):
        assert line.startswith(f"timing diffusion_run call {n} [ms]: ")
        assert [p.split()[0] for p in line.split(": ", 1)[1].split(", ")] == [*phases, "total"]
    assert timing[4].startswith("timing diffusion_run, median of 2 calls after the first [ms]: ")
    assert len(timing) == 5  # only the writing function is timed
    states = [m.split(":", 1)[0] for m in logs.messages if m.startswith("state at")]
    assert states == [
        "state at primary constructor",
        "state at EP_ATM_TIMELOOP_BEFORE, before",
        "state at EP_ATM_TIMELOOP_BEFORE, after",
        "state at after diffusion_run call 1",
        "state at after diffusion_run call 2",
        "state at after diffusion_run call 3",
        "state at destructor",
    ]
    (state,) = [m for m in logs.messages if m.startswith("state at destructor")]
    assert "native threads [name:CPU s]" in state


def test_timing_at_diffusion_before(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    native_plugin(comin, **{plugin.TIMING_ENV: "1"})
    icon.secondary_constructor()
    icon.new_pass()
    for n in range(3):
        icon.diffusion_call(dtime=1.0, linit=n == 0)
    comin.fire("EP_DESTRUCTOR")
    timing = [m for m in logs.messages if m.startswith("timing diffusion_run")]
    phases = ["guard", "args", "sums", "presync", "run", "sync", "check", "done"]
    for n, line in zip((1, 2, 3), timing[:3], strict=True):
        assert line.startswith(f"timing diffusion_run call {n} [ms]: ")
        assert [p.split()[0] for p in line.split(": ", 1)[1].split(", ")] == [*phases, "total"]
    assert timing[3].startswith("timing diffusion_run, median of 2 calls after the first [ms]: ")
    assert len(timing) == 4  # the check at ENTER is not timed


def test_timing_idle(calls, wrappers, logs, icon_run_dir):
    set_mode(icon_run_dir, 0)
    comin = FakeComIn()
    make_plugin(comin, **{plugin.TIMING_ENV: "1"})
    for entry_point in ("EP_SECONDARY_CONSTRUCTOR", "EP_ATM_TIMELOOP_BEFORE", "EP_DESTRUCTOR"):
        comin.fire(entry_point)
    states = [m.split(":", 1)[0] for m in logs.messages if m.startswith("state at")]
    assert states == [
        "state at primary constructor",
        "state at EP_ATM_TIMELOOP_BEFORE, idle",
        "state at destructor, idle",
    ]


def test_profile(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin)
    make_plugin(comin, **{plugin.PROFILE_ENV: "2"})
    icon.secondary_constructor()
    icon.new_pass()
    for n in range(3):
        icon.diffusion_call(dtime=1.0, linit=n == 0)
    assert [c[0] for c in calls] == ["init", "run", "run", "run"]  # the profiled call ran once
    headers = [m for m in logs.messages if m.startswith("profile of")]
    assert headers == [
        "profile of diffusion_run call 2, sorted by cumulative:",
        "profile of diffusion_run call 2, sorted by tottime:",
    ]
    assert any("toy_run" in m for m in logs.messages)


def test_profile_needs_a_call_number():
    with pytest.raises(ValueError, match="expected the number of a call"):
        plugin.Plugin(FakeComIn(), environ={plugin.PROFILE_ENV: "all"})


# ---- optional and zero-size arguments: the same as py2fgen -------------------------------------


def _py2fgen_argument(address: int, shape: tuple[int, ...], optional: bool) -> Any:
    """What py2fgen's conversion makes of an address (host; a NULL address short-cuts)."""
    ffi = cffi.FFI()
    keep = ffi.new("int[1]")
    pointer = ffi.NULL if address == 0 else ffi.cast("int *", keep)
    return py2fgen._conversion.as_array(ffi, (pointer, shape, False, optional))


class _DeviceVariable:
    """A variable whose device address is 'address' (as ICON's CAI would report it)."""

    def __init__(self, shape: tuple[int, ...], address: int):
        self.shape5 = shape + (1,) * (_marshal.MAX_RANK - len(shape))
        self.address = address

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.shape5, "typestr": "<i4", "data": (self.address, False), "version": 3}


def _bound(name: str, optional: bool, variable: Any, shape: tuple[int, ...], present=True):
    descriptor = py2fgen.ArrayParamDescriptor(
        rank=2,
        dtype=py2fgen.INT32,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=optional,
    )
    param = _marshal.ArrayParam(name, _marshal.variable_name("f", name), descriptor, None)
    return _marshal.BoundArray(param=param, variable=variable, shape=shape, present=present)


@pytest.mark.parametrize(
    "case, address, shape, optional",
    [
        ("absent", 0, (4, 0), True),  # an unassociated optional: ICON exposes NULL addresses
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
    if case == "zero-size on the host":
        buffer = fortran_buffer(shape, np.int32)  # host path: NumPy's address of the empty buffer
        bound = _bound("zd", optional, buffer, shape)
        device_xp = None
    else:
        bound = _bound("zd", optional, _DeviceVariable(shape, address), shape, case != "absent")
        device_xp = types.ModuleType("cupy")  # never used: nothing is viewed

    if expected is RuntimeError:
        with pytest.raises(ValueError, match="NULL pointer to a non-optional argument"):
            _marshal.array_argument(bound, device_xp)
        return
    value = _marshal.array_argument(bound, device_xp)
    if expected is None:
        assert value is None
    else:
        assert value.shape == expected.shape == shape and value.dtype == expected.dtype


def test_real_diffusion_init_gets_none_for_an_empty_list_on_the_gpu(monkeypatch):
    """The 'zd_*' arguments of 'diffusion_init' on a rank without such points, device build."""
    comin = FakeComIn()
    icon, _ = real_icon(comin)
    instance = make_plugin(comin, functions=icon.functions)
    icon.secondary_constructor()
    bound = instance._bound["diffusion_init"]
    comin.current_ep = ENTRY_POINTS["EP_ATM_TIMELOOP_BEFORE"]
    for array in bound.arrays:
        if array.param.name in ZERO_EXTENT:  # zero-size: ICON's device address is NULL
            comin.device_buffers[array.param.variable, 1] = types.SimpleNamespace(
                data=types.SimpleNamespace(ptr=0)
            )
    arguments = {
        a.param.name: _marshal.array_argument(a, types.ModuleType("cupy"))
        for a in bound.arrays
        if a.param.name in ZERO_EXTENT
    }
    comin.current_ep = None
    assert arguments == dict.fromkeys(ZERO_EXTENT)  # all None: diffusion_init's "empty list" branch


# ---- SUBSTITUTE: diffusion_run at EP_ATM_DYCORE_DIFFUSION_BEFORE on ICON's own variables ------

NATIVE_SOURCES = {
    "grid_init": {p: _dual.Source("B", _dual.OLD) for p in toy_init.param_descriptors},
    "diffusion_run": {
        "w": _dual.Source("A", _dual.NEW, "cell"),
        "vn": _dual.Source("A", _dual.NEW, "edge"),
        "opt": _dual.Source("A", _dual.NEW, "cell"),
        "dtime": _dual.Source("B", _dual.NEW),
        "linit": _dual.Source("E", _dual.NEW),
    },
}
"""The toy functions with the per-call arguments of 'toy_run' NEW, as the real ones."""
BEFORE, AFTER, ENTER = (
    "EP_ATM_DYCORE_DIFFUSION_BEFORE",
    "EP_ATM_DYCORE_DIFFUSION_AFTER",
    "EP_ATM_DIFFUSION_ENTER",
)


def native_plugin(comin: FakeComIn, **environ: str) -> plugin.Plugin:
    instance = plugin.Plugin(
        comin, functions=TOY_FUNCTIONS, environ=environ, sources=NATIVE_SOURCES
    )
    instance.register()
    return instance


def dual_lines(logs) -> list[str]:
    return [m for m in logs.messages if m.startswith("dual EP_")]


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_substitute_computes_at_diffusion_before(calls, wrappers, logs, on_device):
    device_xp = cupy_or_skip() if on_device else None
    comin = FakeComIn(has_device=on_device)
    icon, _ = toy_icon(comin, device_xp, icon_variables=True)
    live = icon.live
    instance = native_plugin(comin)
    # a plugin registered after this one sees at DIFFUSION_BEFORE what the granule has done
    seen: list[int] = []
    comin.EP_ATM_DYCORE_DIFFUSION_BEFORE(lambda: seen.append(len(calls)))
    icon.secondary_constructor()
    assert instance.active
    device = FLAG_DEVICE if on_device else 0
    before, enter = ENTRY_POINTS[BEFORE], ENTRY_POINTS[ENTER]
    for name in ("w", "vn"):
        assert comin.contexts[before, (name, 1)] == FLAG_READ | FLAG_WRITE | device
        assert comin.contexts[enter, (name, 1)] == FLAG_READ | device  # the time-level check
        assert comin.contexts[enter, (f"icon4py_diffusion_run_{name}", 1)] == FLAG_READ | device
        assert (before, (f"icon4py_diffusion_run_{name}", 1)) not in comin.contexts

    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    assert seen == [2]  # grid_init, then diffusion_run: already at DIFFUSION_BEFORE
    assert np.all(icon.host(live["w"]) == 11.0) and np.all(icon.host(live["vn"]) == 6.0)
    icon.diffusion_call(dtime=5.0, linit=False)
    assert seen == [2, 3]
    assert np.all(icon.host(live["w"]) == 16.0) and np.all(icon.host(live["vn"]) == 12.0)
    # the granule got ICON's time step and the derived flag, not the carrier's scalars
    assert [(c[1]["dtime"], c[1]["linit"]) for c in calls[1:]] == [(10.0, True), (5.0, False)]
    assert calls[1][1]["opt"] is None
    assert icon.carriers["diffusion_run"].item() == 2
    comin.fire("EP_DESTRUCTOR")

    messages = logs.messages
    assert messages[1].endswith(
        "this plugin computes the horizontal diffusion of domain 1 at"
        " EP_ATM_DYCORE_DIFFUSION_BEFORE; ICON skips its own"
    )
    assert (
        f"requested ICON's w vn (READ | WRITE{' | DEVICE' if on_device else ''}) at {BEFORE}"
        " for diffusion_run; not exposed (absent optional arguments): opt" in messages
    )
    assert (
        f"time-level check strict: requested ICON's w vn (READ{' | DEVICE' if on_device else ''})"
        f" at {ENTER}; compares the pointers at {BEFORE}, where the plugin computes" in messages
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
    assert f"diffusion_run call 1 done at {BEFORE} (linit T)." in messages
    assert f"diffusion_run call 2 done at {BEFORE} (linit F)." in messages
    assert f"diffusion_run call 2 checked at {ENTER} (computed at {BEFORE})." in messages
    assert not any(m.startswith("diffusion_run call") and m.endswith(" done.") for m in messages)
    assert messages.count("pointer identity OK (2 fields)") == 1
    assert "w 30.0 -> 330.0 (changed)" in logs.text  # sums of ICON's arrays, first call
    lines = dual_lines(logs)
    first = f"dual {ENTER} call 1 NEW"
    for name, shape in (("w", (NC, NLEV + 1)), ("vn", (NE, NLEV))):
        (line,) = [m for m in lines if m.startswith(f"{first} {name} vs old: identical (device ")]
        assert f", shape {shape}, present True)" in line
        assert ("device -," in line) != on_device
    assert f"{first} opt vs old: identical (device -, host -, shape (), present False)" in lines
    assert f"{first} linit vs old: identical (bool True)" in lines
    assert any(m.startswith(f"{first} dtime vs old: identical (float 10.0") for m in lines)
    for call in (1, 2):
        assert (
            f"dual {ENTER} call {call}: 5 checked (NEW 5, OBSERVE 0), 5 identical, 0 differ"
            " (NEW 0, OBSERVE 0); selftest 0" in lines
        )
    assert (
        f"time-level check call 2 regular: 2 checked, 2 identical, 0 differ; {BEFORE} 2,"
        f" {AFTER} 1; ICON's pointers as at {BEFORE}: yes [strict]" in messages
    )
    assert (
        f"per-call state: 2 diffusion calls of domain 1 at {BEFORE}, 2 computed there,"
        " 0 without lhdiff_vn; linit: initial 1, regular 1, ICON's condition agrees at 2 of 2;"
        " EP_ATM_INTEGRATE_START 1 (domain 1)" in messages
    )
    assert messages[-1].endswith(
        "callbacks for domain 1: 2, expected 2 (ICON mode SUBSTITUTE;"
        f" {BEFORE} 2, lhdiff_vn T): identical"
    )
    assert not any(r.levelno >= logging.WARNING for r in logs.records)


def test_substitute_names_an_argument_on_other_memory(calls, wrappers, logs):
    comin = FakeComIn()
    icon, buffers = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    # ICON's dispatcher binds another array than ComIn's 'vn' (a stale time level)
    icon.rebind("diffusion_run", "vn", fortran_buffer((NE, NLEV), np.float64, fill=3.0))
    with pytest.raises(_dual.DualCheckError, match="new route of 'vn' differs"):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert np.all(buffers["vn"] == 6.0)  # the granule ran at DIFFUSION_BEFORE on ICON's 'vn'
    (line,) = [m for m in dual_lines(logs) if " vn vs old" in m]
    assert line.startswith(f"dual {ENTER} call 1 NEW vn vs old: differ (device -, host 0x")
    assert icon.carriers["diffusion_run"].item() == -1  # not acknowledged


def test_substitute_report_mode_continues(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    native_plugin(comin, **{_dual.MODE_ENV: "report", plugin.TIMELEVEL_CHECK_ENV: "report"})
    icon.secondary_constructor()
    icon.new_pass()
    icon.rebind("diffusion_run", "w", fortran_buffer((NC, NLEV + 1), np.float64, fill=1.0))
    icon.diffusion_call(dtime=10.0, linit=True)
    assert icon.carriers["diffusion_run"].item() == 1
    assert any(m.startswith(f"dual {ENTER} call 1 NEW w vs old: differ") for m in logs.messages)
    assert any(
        m.startswith("time-level check call 1 initial: 2 checked, 1 identical, 1 differ: w;")
        for m in logs.messages
    )


def test_substitute_selftest_of_the_per_call_arguments(calls, wrappers, logs):
    comin = FakeComIn()
    icon, buffers = toy_icon(comin, icon_variables=True)
    native_plugin(comin, **{_dual.MODE_ENV: "report", _dual.SELFTEST_ENV: "all"})
    icon.secondary_constructor()
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    lines = [m for m in dual_lines(logs) if m.startswith(f"dual {ENTER} call 1")]
    for name in ("w", "vn", "opt", "dtime", "linit"):
        assert any(f" NEW {name} vs old [selftest]: differ" in m for m in lines), name
    assert lines[-1].endswith("0 identical, 5 differ (NEW 5, OBSERVE 0); selftest 5")
    # the granule's input is ICON's state, not the perturbed copies
    assert (calls[1][1]["dtime"], calls[1][1]["linit"]) == (10.0, True)
    assert np.all(buffers["w"] == 11.0)


def test_substitute_without_lhdiff_vn(calls, wrappers, logs, icon_run_dir):
    set_namelist(icon_run_dir, LHDIFF_VN="F")
    comin = FakeComIn()
    icon, buffers = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    for _ in range(2):  # ICON fires the regular pair, but calls no diffusion
        icon.diffusion_call(dtime=10.0, linit=False, delegate=False)
    comin.fire("EP_DESTRUCTOR")
    assert [c[0] for c in calls] == ["init"] and np.all(buffers["w"] == 1.0)
    assert f"{BEFORE} 2: lhdiff_vn F, ICON calls no diffusion; nothing to do." in logs.messages
    assert not any(m.startswith("linit call") for m in logs.messages)
    assert logs.messages[-1].endswith(
        f"domain 1: 0, expected 0 (ICON mode SUBSTITUTE; {BEFORE} 2, lhdiff_vn F): identical"
    )


@pytest.mark.parametrize(
    "entry, value", [("INIT_MODE", "5"), ("LTESTCASE", "T"), ("LDYNAMICS", "F"), ("restart", "")]
)
def test_linit_disagreeing_with_icons_condition_stops(calls, wrappers, icon_run_dir, entry, value):
    comin = FakeComIn()
    if entry == "restart":
        comin.lrestartrun = True
    else:
        set_namelist(icon_run_dir, **{entry: value})
    icon, buffers = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    # ICON's condition excludes an initial call, but a DIFFUSION_BEFORE comes before the dynamics
    with pytest.raises(
        RuntimeError, match=r"linit call 1: T from .* ICON's condition: F .* disagree"
    ):
        icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in calls] == ["init"] and np.all(buffers["w"] == 1.0)


def test_linit_needs_integrate_start(calls, wrappers):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    with pytest.raises(RuntimeError, match="before any EP_ATM_INTEGRATE_START"):
        comin.fire(BEFORE, 1)


def test_linit_in_later_steps_and_domains(calls, wrappers, logs):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    comin.fire("EP_ATM_INTEGRATE_START", 2)  # another domain's order does not count
    comin.fire("EP_ATM_DYCORE_SOLVE_NH_BEFORE", 2)
    icon.diffusion_call(dtime=10.0, linit=False)
    icon.diffusion_call(dtime=10.0, linit=False)
    lines = [m for m in logs.messages if m.startswith("linit call")]
    assert [m.split(":")[1].split(" from")[0].strip() for m in lines] == ["T", "F", "F"]
    assert "step 2, call 1 of the step): agree" in lines[2]


def test_diffusion_before_without_enter_stops(calls, wrappers):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True, delegate=False)  # ICON did not delegate
    with pytest.raises(RuntimeError, match="call 1 of domain 1 was not followed by"):
        icon.diffusion_call(dtime=10.0, linit=False)


def test_enter_out_of_order_stops(calls, wrappers):
    comin = FakeComIn()
    icon, _ = toy_icon(comin, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    icon.call_count = 5  # ICON's call count does not match the plugin's
    with pytest.raises(RuntimeError, match=r"binds diffusion call 6 .* is call 1"):
        icon.diffusion_call(dtime=10.0, linit=True)


def test_substitute_needs_icons_variables():
    comin = FakeComIn()
    icon, _ = toy_icon(comin)  # ICON exposes no 'w' and 'vn' of its own
    native_plugin(comin)
    with pytest.raises(RuntimeError, match="2 problem") as error:
        icon.secondary_constructor()
    assert f"'w': ICON does not expose its variable, which 'diffusion_run' gets at {BEFORE}" in str(
        error.value
    )


def test_icon_variable_with_more_than_one_block():
    comin = FakeComIn()
    buffer = np.zeros((NE, NLEV, 2, 1, 1), order="F")
    comin.add("vn", buffer, datatype=DOUBLE)
    comin.contexts[None, ("vn", 1)] = FLAG_READ
    param = _marshal.signature("diffusion_run", toy_run).arrays[1]
    with pytest.raises(ValueError, match=r"'vn' has the shape \(9, 4, 2, 1, 1\).*nblks == 1"):
        plugin.icon_bound_array(FakeVariable(comin, ("vn", 1)), param)
    buffer = np.zeros((NE, NLEV, 1, 1, 1), order="F")
    comin.buffers["vn", 1] = buffer
    bound = plugin.icon_bound_array(FakeVariable(comin, ("vn", 1)), param)
    assert bound.shape == (NE, NLEV) and bound.present


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_verify_computes_at_enter_and_observes_the_per_call_scalars(
    calls, wrappers, logs, icon_run_dir, on_device
):
    device_xp = cupy_or_skip() if on_device else None
    set_mode(icon_run_dir, 2)
    comin = FakeComIn(has_device=on_device)
    icon, buffers = toy_icon(comin, device_xp, icon_variables=True)
    native_plugin(comin)
    icon.secondary_constructor()
    device = FLAG_DEVICE if on_device else 0
    before, enter = ENTRY_POINTS[BEFORE], ENTRY_POINTS[ENTER]
    assert comin.contexts[before, ("w", 1)] == FLAG_READ | device  # the time-level check only
    assert comin.contexts[enter, ("icon4py_diffusion_run_w", 1)] == FLAG_READ | FLAG_WRITE | device
    icon.new_pass()
    # VERIFY: ICON binds copies of its input; the granule works on them
    copy_w = icon.rebind("diffusion_run", "w", fortran_buffer((NC, NLEV + 1), np.float64, 1.0))
    icon.diffusion_call(dtime=10.0, linit=True)
    assert np.all(icon.host(copy_w) == 11.0) and np.all(buffers["w"] == 1.0)
    icon.diffusion_call(dtime=5.0, linit=False, carrier_linit=True)  # observed, not used
    assert (calls[2][1]["dtime"], calls[2][1]["linit"]) == (5.0, True)
    messages = logs.messages
    assert "this plugin computes it too, at EP_ATM_DIFFUSION_ENTER on ICON's copies" in messages[1]
    assert (
        "dual source table: 14 arguments (9 arrays, 5 scalars; A 3, B 10, C 0, D 0, E 1);"
        " OLD 12, NEW 0, OBSERVE 2; mode strict; selftest none" in messages
    )
    assert f"dual plan {ENTER}: 2 items (NEW 0, OBSERVE 2): dtime, linit" in messages
    lines = dual_lines(logs)
    assert f"dual {ENTER} call 1 OBSERVE linit vs old: identical (bool True)" in lines
    assert f"dual {ENTER} call 2 OBSERVE linit vs old: differ (bool False vs bool True)" in lines
    assert "diffusion_run call 2 done." in messages
    assert not any("done at" in m for m in messages)


# ---- ICON's variables that diffusion_init gets (plugin.STATIC_VARIABLES) -----------------------


@icon4py_export.export
def toy_diffusion_init(
    theta_ref_mc: fa.CellKField[gtx.float64],
    zd_cellidx: wrapper_common.OptionalInt32Array2D,
    zd_diffcoef: wrapper_common.OptionalFloat64Array1D,
    ndyn_substeps: gtx.int32,
) -> None:
    CALLS.append(
        (
            "diffusion_init",
            dict(
                theta_ref_mc=theta_ref_mc,
                zd_cellidx=zd_cellidx,
                zd_diffcoef=zd_diffcoef,
                ndyn_substeps=ndyn_substeps,
            ),
        )
    )


STATIC_FUNCTIONS = {
    "diffusion_init": plugin.FunctionEntry(
        toy_diffusion_init, "EP_ATM_TIMELOOP_BEFORE", False, _marshal.PASS_KEY
    ),
    "diffusion_run": TOY_FUNCTIONS["diffusion_run"],
}
STATIC_SOURCES = {
    "diffusion_init": {
        "theta_ref_mc": _dual.Source("A", _dual.NEW, "cell"),
        "zd_cellidx": _dual.Source("A", _dual.NEW),
        "zd_diffcoef": _dual.Source("A", _dual.NEW),
        "ndyn_substeps": _dual.Source("D", _dual.OLD),
    },
    "diffusion_run": {p: _dual.Source("A", _dual.OLD) for p in toy_run.param_descriptors},
}
"""The toy diffusion_init with its arguments of STATIC_VARIABLES NEW, as the real one."""
NPOINTS = 3
STATIC_NAMES = {
    "theta_ref_mc": "theta_ref_mc",
    "zd_cellidx": "zd_indlist",
    "zd_diffcoef": "zd_diffcoef",
}
INIT = "EP_ATM_TIMELOOP_BEFORE"


def static_icon(
    comin: FakeComIn, device_xp: Any = None, zdiffu: bool = True
) -> tuple[FakeIcon, dict[str, np.ndarray]]:
    """
    The argument variables of the toy diffusion_init and ICON's own variables of their names in
    STATIC_VARIABLES, on the same memory (host and device); without 'zdiffu' ICON has no 'zd_*'
    lists (l_zdiffu_t=.FALSE.: absent optional arguments, no variables).
    """
    icon = FakeIcon(comin, STATIC_FUNCTIONS, device_xp)
    buffers = {"theta_ref_mc": fortran_buffer((NC, NLEV), np.float64, fill=290.0)}
    if zdiffu:
        buffers["zd_cellidx"] = fortran_buffer((4, NPOINTS), np.int32, fill=2)
        buffers["zd_diffcoef"] = fortran_buffer((NPOINTS,), np.float64, fill=0.5)
    icon.expose("diffusion_init", buffers, dict(ndyn_substeps=5))
    run = {
        "w": fortran_buffer((NC, NLEV + 1), np.float64, fill=1.0),
        "vn": fortran_buffer((NE, NLEV), np.float64, fill=3.0),
    }
    icon.expose("diffusion_run", run, dict(dtime=0.0, linit=False))
    for param, buffer in buffers.items():
        device = icon.live[param] if device_xp is not None else None
        datatype = DOUBLE if buffer.dtype == np.float64 else INT
        comin.add(
            plugin.STATIC_VARIABLES["diffusion_init"][param], buffer, device, datatype=datatype
        )
    return icon, buffers


def static_plugin(comin: FakeComIn, **environ: str) -> plugin.Plugin:
    instance = plugin.Plugin(
        comin, functions=STATIC_FUNCTIONS, environ=environ, sources=STATIC_SOURCES
    )
    instance.register()
    return instance


def test_static_variables_are_the_real_names():
    real = plugin.STATIC_VARIABLES["diffusion_init"]
    assert {p: real[p] for p in STATIC_NAMES} == STATIC_NAMES
    assert set(real) <= set(diffusion_wrapper.diffusion_init.param_descriptors)
    assert real["zd_cellidx"] == "zd_indlist"  # ICON's name of py2fgen's 'zd_cellidx'


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_static_variables_end_to_end(calls, wrappers, logs, on_device):
    device_xp = cupy_or_skip() if on_device else None
    comin = FakeComIn(has_device=on_device)
    icon, _ = static_icon(comin, device_xp)
    static_plugin(comin)
    icon.secondary_constructor()
    device = FLAG_DEVICE if on_device else 0
    init_ep = ENTRY_POINTS[INIT]
    contexts = {k: v for k, v in comin.contexts.items() if k[1][0] in STATIC_NAMES.values()}
    assert contexts == {(init_ep, (n, 1)): FLAG_READ | device for n in STATIC_NAMES.values()}
    assert (
        "requested ICON's theta_ref_mc zd_indlist (as zd_cellidx) zd_diffcoef"
        f" (READ{' | DEVICE' if on_device else ''}) at {INIT} for diffusion_init" in logs.messages
    )
    icon.new_pass()
    where = f"dual {INIT} pass 1"
    lines = dual_lines(logs)
    for name, shape in (
        ("theta_ref_mc", (NC, NLEV)),
        ("zd_cellidx", (4, NPOINTS)),
        ("zd_diffcoef", (NPOINTS,)),
    ):
        (line,) = [m for m in lines if f" {name} vs " in m]
        assert line.startswith(f"{where} NEW {name} vs old: identical (device ")
        assert line.endswith(f", shape {shape}, present True)")
        assert ("device -," in line) != on_device
    assert lines[-1] == (
        f"{where}: 3 checked (NEW 3, OBSERVE 0), 3 identical, 0 differ (NEW 0, OBSERVE 0);"
        " selftest 0"
    )
    assert (
        "diffusion_init pass 1: ICON's variables theta_ref_mc zd_cellidx zd_diffcoef fetched at"
        f" {INIT} (3 present, 0 absent); addresses recorded" in logs.messages
    )
    # the granule gets zero-copy views of ICON's variables, as py2fgen builds them
    ((name, init),) = calls
    theta, zd_cellidx = init["theta_ref_mc"], init["zd_cellidx"]
    assert isinstance(theta, gtx.Field) and tuple(theta.domain.dims) == (dims.CellDim, dims.KDim)
    assert _views.data_ptr(theta.ndarray) == _views.data_ptr(icon.live["theta_ref_mc"])
    assert type(zd_cellidx).__module__.split(".")[0] == ("cupy" if on_device else "numpy")
    assert _views.data_ptr(zd_cellidx) == _views.data_ptr(icon.live["zd_cellidx"])
    assert zd_cellidx.shape == (4, NPOINTS) and init["ndyn_substeps"] == 5
    # pass 2: fetched again, at the same addresses
    icon.new_pass()
    assert (
        "diffusion_init pass 2: ICON's variables theta_ref_mc zd_cellidx zd_diffcoef fetched at"
        f" {INIT} (3 present, 0 absent); addresses as at pass 1" in logs.messages
    )
    assert [m.split(":")[0] for m in dual_lines(logs) if "pass 2" in m] == [f"dual {INIT} pass 2"]


def test_static_variables_absent_without_the_lists(calls, wrappers, logs):
    """l_zdiffu_t=.FALSE.: ICON has no 'zd_*' lists; absent on both routes, None for the granule."""
    comin = FakeComIn()
    icon, _ = static_icon(comin, zdiffu=False)
    static_plugin(comin)
    icon.secondary_constructor()
    assert (
        f"requested ICON's theta_ref_mc (READ) at {INIT} for diffusion_init; not exposed (absent"
        " optional arguments): zd_indlist (as zd_cellidx) zd_diffcoef" in logs.messages
    )
    icon.new_pass()
    lines = dual_lines(logs)
    for name in ("zd_cellidx", "zd_diffcoef"):
        assert (
            f"dual {INIT} pass 1 NEW {name} vs old: identical (device -, host -, shape (),"
            " present False)" in lines
        )
    assert lines[-1].endswith("3 identical, 0 differ (NEW 0, OBSERVE 0); selftest 0")
    assert "(1 present, 2 absent); addresses recorded" in logs.text
    init = calls[0][1]
    assert init["zd_cellidx"] is None and init["zd_diffcoef"] is None


def test_static_variable_on_other_memory(calls, wrappers, logs):
    """ICON's variable is not what py2fgen gets: strict stops before the granule; report goes on."""
    for mode in (_dual.STRICT, _dual.REPORT):
        comin = FakeComIn()
        icon, buffers = static_icon(comin)
        comin.buffers["theta_ref_mc", 1] = buffers["theta_ref_mc"].copy(order="F")
        static_plugin(comin, **{_dual.MODE_ENV: mode})
        icon.secondary_constructor()
        if mode == _dual.STRICT:
            with pytest.raises(_dual.DualCheckError, match="new route of 'theta_ref_mc' differs"):
                icon.new_pass()
            assert calls == []
            continue
        icon.new_pass()
        lines = [m for m in dual_lines(logs) if " theta_ref_mc vs " in m]
        assert len(lines) == 2  # strict logged it, then stopped
        assert all(
            " differ (device -, host 0x" in m and " vs device -, host 0x" in m for m in lines
        )
        theta = calls[0][1]["theta_ref_mc"]  # NEW: the granule gets ICON's variable
        assert _views.data_ptr(theta.ndarray) == _views.data_ptr(comin.buffers["theta_ref_mc", 1])


def test_static_variable_moved_between_passes(calls, wrappers):
    comin = FakeComIn()
    icon, buffers = static_icon(comin)
    static_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    moved = buffers["zd_diffcoef"].copy(order="F")
    comin.buffers["zd_diffcoef", 1] = moved
    icon.rebind("diffusion_init", "zd_diffcoef", moved)  # both routes agree, but it moved
    with pytest.raises(RuntimeError, match=r"zd_diffcoef of 'diffusion_init' moved between pass 1"):
        icon.new_pass()
    assert len(calls) == 1


def test_static_variables_selftest(calls, wrappers, logs):
    comin = FakeComIn()
    icon, buffers = static_icon(comin)
    static_plugin(comin, **{_dual.MODE_ENV: "report", _dual.SELFTEST_ENV: "all"})
    icon.secondary_constructor()
    icon.new_pass()
    lines = dual_lines(logs)
    for name in STATIC_NAMES:
        (line,) = [m for m in lines if f" {name} vs " in m]
        assert f"NEW {name} vs old [selftest]: differ" in line
    assert lines[-1].endswith("0 identical, 3 differ (NEW 3, OBSERVE 0); selftest 3")
    theta = calls[0][1]["theta_ref_mc"]  # the granule's input is untouched
    assert _views.data_ptr(theta.ndarray) == _views.data_ptr(buffers["theta_ref_mc"])


def test_static_variable_must_be_exposed():
    comin = FakeComIn()
    icon, _ = static_icon(comin)
    del comin.buffers["theta_ref_mc", 1]
    static_plugin(comin)
    with pytest.raises(RuntimeError, match="1 problem") as error:
        icon.secondary_constructor()
    assert (
        "'theta_ref_mc': ICON does not expose its variable, the argument 'theta_ref_mc' of"
        " 'diffusion_init'." in str(error.value)
    )
