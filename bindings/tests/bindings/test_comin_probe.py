# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the py2fgen probe ('_probe.py'): the comparisons, the objects, the recorders as
py2fgen's glue installs them, and the plugin in probe mode with the real inputs against the fake
ComIn of 'test_comin_plugin.py' and the descriptive data of 'test_comin_descrdata.py'. py2fgen's
arguments are built by py2fgen's own conversion (cffi pointers, 'ArrayInfo's) from the arrays
ICON's py2fgen interface passes; the granules on both routes are replaced by stand-ins that
build the same kind of objects from what they get. Also: the runtime modules of the plugin
import neither the probe nor py2fgen's bindings.
"""

import ast
import copy
import dataclasses
import functools
import pathlib
import pkgutil
import types
from typing import Any

import cffi
import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import diffusion_wrapper, grid_wrapper
from icon4py.bindings.comin import _compare, _probe, _views, plugin
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, states as grid_states
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _codegen

from .test_comin_config import PY2FGEN_DIFFUSION
from .test_comin_descrdata import NLEV, NPROMA, PY2FGEN, Icon
from .test_comin_plugin import (  # fixtures: icon_run_dir, logs, restore_package_logger, wrappers
    DOUBLE,
    INT,
    FakeComIn,
    cupy_or_skip,
    fortran_buffer,
    icon_run_dir,
    logs,
    rank_view,
    restore_package_logger,
    set_mode,
    wrappers,
)


# ---- the comparisons ----------------------------------------------------------------------------


def test_capture_and_compare_arrays():
    a = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(4, 3))
    new, ref = _probe.capture(a, values=True), _probe.capture(a.copy(order="F"), values=True)
    assert new.form == "numpy float64 (4, 3) strides (1, 4)" and new.values is not a
    result = _probe.compare(new, ref, real=3)
    assert result.identical
    assert result.details == "float64 (4, 3); real 9: 0 differ; padding 3: 0 differ"
    # the same memory is required for ICON's variables
    assert not _probe.compare(new, ref, real=3, same_memory=True).identical
    same = _probe.compare(new, _probe.capture(a, values=True), real=3, same_memory=True)
    assert same.identical and same.details.startswith(f"same address {a.ctypes.data:#x}; ")
    # without copies: form and address only
    lean = _probe.capture(a, values=False)
    assert lean.values is None
    assert _probe.compare(lean, lean, same_memory=True).details == (
        f"same address {a.ctypes.data:#x}; numpy float64 (4, 3) strides (1, 4)"
    )
    # another form
    c_order = _probe.capture(np.ascontiguousarray(a), values=True)
    assert _probe.compare(c_order, ref).details.startswith(
        "form numpy float64 (4, 3) strides (3, 1)"
    )
    # a real entry and a padding entry
    b = a.copy(order="F")
    b[3, 0] = -1.0
    assert _probe.compare(_probe.capture(b, values=True), ref, real=3).identical
    b[1, 2] = -1.0
    result = _probe.compare(_probe.capture(b, values=True), ref, real=3)
    assert not result.identical and "real 9: 1 differ, first (1, 2)" in result.details


def test_absent_arrays():
    absent = _probe.capture(None, values=True)
    present = _probe.capture(np.zeros(2), values=True)
    assert _probe.compare(absent, _probe.capture(None, values=False)).details == (
        "absent on both routes"
    )
    assert _probe.compare(absent, present).details == "present only on the reference"
    assert _probe.compare(present, absent).details == "present only on the new route"
    assert not _probe.compare(present, 1.0).identical


def test_bools_are_compared_as_values():
    """py2fgen views Fortran's logicals as bools without normalising them (255 for .TRUE.)."""
    raw = np.array([255, 0, 255, 0], dtype=np.uint8).view(np.bool_)
    native = np.array([True, False, True, False])
    new, ref = _probe.capture(native, values=True), _probe.capture(raw, values=True)
    assert ref.raw == (0, 255) and new.raw == (0, 1)
    result = _probe.compare(new, ref, real=4)
    assert result.identical
    assert result.details.endswith("; as values, bytes [0, 1] vs [0, 255]")
    assert not _probe.compare(_probe.capture(~native, values=True), ref, real=4).identical


def test_scalars_and_communicators():
    assert _probe.compare(1.5, 1.5).identical
    assert not _probe.compare(1.5, np.nextafter(1.5, 2.0)).identical
    assert not _probe.compare(True, 1).identical
    same = _compare.Communicator(1, (0, 1), "CONGRUENT"), _compare.Communicator(2, (0, 1))
    assert _probe.compare(*same).identical


@pytest.mark.parametrize(
    "value",
    [
        np.arange(3.0),
        np.array([True, False]),
        None,
        np.zeros((0, 2)),
        2.5,
        7,
        False,
        _compare.Communicator(5, (0, 1), "IDENT"),
    ],
    ids=["float", "bool", "absent", "empty", "real", "int", "logical", "communicator"],
)
def test_perturbation_is_reported(value):
    for values in (True, False):
        new = (
            _probe.capture(value, values)
            if value is None or isinstance(value, np.ndarray)
            else value
        )
        reference = (
            _compare.Communicator(6, (0, 1)) if isinstance(value, _compare.Communicator) else new
        )
        assert _probe.compare(new, reference, same_memory=True).identical
        assert not _probe.compare(_probe.perturb(new), reference, same_memory=True).identical


def test_switch():
    assert _probe.parse({}) is False and _probe.parse({_probe.ENV: "1"}) is True
    assert plugin.PROBE_ENV == _probe.ENV and _probe.__name__ == plugin.PROBE_MODULE
    with pytest.raises(ValueError, match="expected 0 or 1"):
        _probe.parse({_probe.ENV: "yes"})
    with pytest.raises(ValueError, match="expected 0 or 1"):
        plugin.Plugin(FakeComIn(), environ={plugin.PROBE_ENV: "yes"})
    assert plugin.Plugin(FakeComIn(), environ={plugin.PROBE_ENV: "0"}).probe is None


RUNTIME_FORBIDDEN = ("grid_wrapper", "diffusion_wrapper", "icon4py.tools", "py2fgen", "_probe")
"""What the plugin's runtime modules must not import: py2fgen's bindings and the probe."""


def test_the_runtime_modules_import_neither_py2fgens_bindings_nor_the_probe():
    package = pathlib.Path(plugin.__file__).parent
    runtime = sorted(set(package.glob("*.py")) - {package / "_probe.py", package / "_compare.py"})
    assert {p.name for p in runtime} >= {"plugin.py", "_granule.py", "_arguments.py"}
    for path in runtime:
        names = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names += [f"{node.module}.{alias.name}" for alias in node.names]
        bad = [n for n in names if any(f in n for f in RUNTIME_FORBIDDEN)]
        assert bad == [], path.name


# ---- py2fgen's side ------------------------------------------------------------------------------

CALLS: list[tuple[str, dict[str, Any]]] = []
MODULES = {
    "grid_init": grid_wrapper,
    "diffusion_init": diffusion_wrapper,
    "diffusion_run": diffusion_wrapper,
}


@dataclasses.dataclass
class StandInGridConfig:
    num_cells: int
    limited_area: bool


class StandInGrid:
    """A grid as icon4py's: an id, a configuration, connectivities, index ranges."""

    def __init__(self, c2e: Any, num_cells: int) -> None:
        self.id = "icon_grid"
        self.config = StandInGridConfig(num_cells, True)
        self.connectivities = {"C2E": gtx.as_field([dims.CellDim, dims.C2EDim], c2e.asnumpy())}

    def start_index(self, domain: Any) -> np.int32:
        return np.int32(0)

    def end_index(self, domain: Any) -> np.int32:
        return np.int32(self.config.num_cells)


class StandInExchange:
    def get_size(self) -> int:
        return 1

    def my_rank(self) -> int:
        return 0


def stand_in_grid_state(kwargs: dict[str, Any]) -> types.SimpleNamespace:
    """What a 'grid_init' builds, in miniature, from its inputs."""
    return types.SimpleNamespace(
        grid=StandInGrid(kwargs["c2e"], kwargs["num_cells"]),
        vertical_grid=types.SimpleNamespace(
            config=StandInGridConfig(kwargs["vertical_size"], kwargs["limited_area"]),
            _vct_a=gtx.as_field([dims.KHalfDim], kwargs["vct_a"].asnumpy()),
            _vct_b=None,
        ),
        edge_geometry=types.SimpleNamespace(
            tangent_orientation=kwargs["tangent_orientation"],
            primal_normal_vert=(kwargs["primal_normal_vert_x"], kwargs["primal_normal_vert_y"]),
            dual_edge_lengths=None,
        ),
        cell_geometry=grid_states.CellParams(
            cell_center_lat=kwargs["cell_center_lat"],
            cell_center_lon=kwargs["cell_center_lon"],
            area=kwargs["cell_areas"],
            mean_cell_area=kwargs["mean_cell_area"],
        ),
        exchange_runtime=StandInExchange(),
    )


@dataclasses.dataclass
class StandInMetricState:
    theta_ref_mc: Any


def stand_in_diffusion(config: diffusion.DiffusionConfig, theta_ref_mc: Any) -> Any:
    """What a 'diffusion_init' builds, in miniature: a granule's configuration, a field it keeps,
    one it computes, values, a program and a runtime object."""
    return types.SimpleNamespace(
        config=config,
        _metric_state=StandInMetricState(theta_ref_mc),
        diff_multfac_vn=gtx.as_field([dims.KDim], np.full(3, config.hdiff_efdt_ratio)),
        rd_o_cvd=0.4,
        _cell_start_nudging=np.int32(4),
        halo_exchange_wait=object(),
        apply_diffusion_to_vn=functools.partial(print),
        _grid=object(),
    )


def wrapper_config(kwargs: dict[str, Any]) -> diffusion.DiffusionConfig:
    """The DiffusionConfig of py2fgen's 'diffusion_init' (its constructor call)."""
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType(kwargs["diffusion_type"]),
        apply_to_vertical_wind=kwargs["hdiff_w"],
        apply_to_horizontal_wind=kwargs["hdiff_vn"],
        apply_smag_diff_to_vertical_wind=kwargs["hdiff_smag_w"],
        apply_zdiffusion_t=kwargs["zdiffu_t"],
        type_t_diffu=diffusion.TemperatureDiscretizationType(kwargs["type_t_diffu"]),
        type_vn_diffu=diffusion.SmagorinskyStencilType(kwargs["type_vn_diffu"]),
        hdiff_efdt_ratio=kwargs["hdiff_efdt_ratio"],
        hdiff_w_efdt_ratio=kwargs["hdiff_w_efdt_ratio"],
        **{
            f"smagorinski_scaling_{k}{i}": kwargs[f"smagorinski_scaling_{k}{i}"]
            for k in ("factor", "height")
            for i in ("", "2", "3", "4")
        },
        apply_to_temperature=kwargs["hdiff_temp"],
        velocity_boundary_diffusion_denominator=kwargs["denom_diffu_v"],
        shear_type=diffusion.TurbulenceShearForcingType(kwargs["itype_sher"]),
        iforcing=diffusion.ForcingType(kwargs["iforcing"]),
        a_hshr=kwargs["a_hshr"],
        loutshs=kwargs["loutshs"],
    )


@pytest.fixture
def stand_ins(monkeypatch):
    """The exported functions with stand-ins for the granules (py2fgen converts as usual): they
    record their arguments and build the objects of 'grid_init' and 'diffusion_init'."""
    CALLS.clear()
    monkeypatch.setattr(grid_wrapper, "grid_state", None)
    monkeypatch.setattr(diffusion_wrapper, "granule", None)
    for name, module in MODULES.items():
        stand_in = copy.copy(getattr(module, name))

        def fun(_name=name, **kwargs: Any) -> None:
            CALLS.append((_name, kwargs))
            if _name == "grid_init":
                grid_wrapper.grid_state = stand_in_grid_state(kwargs)
            elif _name == "diffusion_init":
                diffusion_wrapper.granule = types.SimpleNamespace(
                    diffusion=stand_in_diffusion(wrapper_config(kwargs), kwargs["theta_ref_mc"])
                )

        stand_in._fun = fun
        monkeypatch.setattr(module, name, stand_in)
    yield CALLS
    CALLS.clear()


class StandInGranule:
    """The plugin's granule in probe mode: the same objects from the plugin's inputs."""

    def __init__(self) -> None:
        self.grid_state: Any = None
        self.diffusion: Any = None
        self.released = False

    def grid_init(self, **kwargs: Any) -> None:
        self.grid_state = stand_in_grid_state(kwargs)

    def diffusion_init(self, **kwargs: Any) -> None:
        self.diffusion = stand_in_diffusion(kwargs["config"], kwargs["theta_ref_mc"])

    def diffusion_run(self, **kwargs: Any) -> None:
        raise AssertionError("the plugin runs no diffusion in probe mode")

    def release(self) -> None:
        self.released = True


def test_install_is_inert_without_the_probe(stand_ins, monkeypatch, capsys):
    before = {name: getattr(module, name) for name, module in MODULES.items()}
    monkeypatch.setattr(plugin, "_instance", None)
    pkgutil.resolve_name(_probe.INSTALL)()
    assert (
        "py2fgen probe: not installed: the icon4py ComIn plugin is not loaded"
        in capsys.readouterr().out
    )
    monkeypatch.setattr(
        plugin, "_instance", plugin.Plugin(FakeComIn(), environ={}, granule=StandInGranule())
    )
    pkgutil.resolve_name(_probe.INSTALL)()
    assert "does not run in probe mode (ICON4PY_COMIN_PROBE=1)" in capsys.readouterr().out
    assert {name: getattr(module, name) for name, module in MODULES.items()} == before


class Py2fgen:
    """py2fgen's glue in miniature: 'ArrayInfo's of Fortran arrays, the exported function's
    conversion, the function bound after the extra callables ran."""

    def __init__(self) -> None:
        self.ffi = cffi.FFI()
        self.keep: list[Any] = []

    def info(self, array: Any, descriptor: py2fgen.ArrayParamDescriptor) -> tuple:
        """A Fortran array as the glue passes it; a CuPy array is ICON's device array."""
        if array is None:
            return (self.ffi.NULL, (), False, descriptor.is_optional)
        assert array.flags.f_contiguous or array.ndim <= 1
        self.keep.append(array)
        ctype = _codegen.BUILTIN_TO_CPP_TYPE[descriptor.dtype]
        on_gpu = not isinstance(array, np.ndarray)
        assert not (on_gpu and descriptor.memory_space == py2fgen.MemorySpace.HOST)
        pointer = self.ffi.cast(f"{ctype} *", _views.data_ptr(array))
        return (pointer, array.shape, on_gpu, descriptor.is_optional)

    def call(self, name: str, arguments: dict[str, Any]) -> None:
        exported = getattr(MODULES[name], name)  # what the glue imports after the extra callables
        kwargs = {}
        for param, descriptor in exported.param_descriptors.items():
            value = arguments[param]
            if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
                kwargs[param] = self.info(value, descriptor)
            elif descriptor.dtype == py2fgen.BOOL:
                kwargs[param] = int(value)  # cffi hands over the C byte as an int
            else:
                kwargs[param] = value
        exported(ffi=self.ffi, perf_counters=None, **kwargs)


# ---- the plugin in probe mode, end to end ----------------------------------------------------------

NZ = 2
"""Entries of the 'zd_*' lists."""
TRUE_BYTE = 255
"""nvfortran's .TRUE. in a logical(c_bool) array."""


class ProbeComIn(FakeComIn):
    """FakeComIn with the complete descriptive data of domain 1 ('test_comin_descrdata.Icon')."""

    def __init__(self, icon: Icon, has_device: bool = False) -> None:
        super().__init__(has_device=has_device)
        self.icon = icon
        self.domain = icon.domain
        self.timestep = 10.0

    def descrdata_get_global(self):
        glob = self.icon.glob
        return types.SimpleNamespace(
            has_device=self.has_device,
            lrestartrun=False,
            n_dom=glob.n_dom,
            l_limited_area=glob.l_limited_area,
            vct_a=glob.vct_a,
        )

    def parallel_get_host_mpi_comm(self) -> int:
        return self.icon.host_comm


def icon_variables(
    comin: ProbeComIn, device_xp: Any = None, ddt_tke_hsh: bool = True
) -> dict[str, Any]:
    """ICON's variables that the granules get (5-D Fortran buffers, on the device with
    'device_xp': what ICON passes); no 'dwdx', 'dwdy'; 'ddt_tke_hsh' iff the output lists it."""
    rng = np.random.default_rng(3)
    shapes = {
        "theta_ref_mc": ((NPROMA, NLEV), np.float64),
        "wgtfac_c": ((NPROMA, NLEV + 1), np.float64),
        "zd_indlist": ((4, NZ), np.int32),
        "zd_vertidx": ((4, NZ), np.int32),
        "zd_intcoef": ((3, NZ), np.float64),
        "zd_diffcoef": ((NZ,), np.float64),
        "w": ((NPROMA, NLEV + 1), np.float64),
        "vn": ((NPROMA, NLEV), np.float64),
        "exner": ((NPROMA, NLEV), np.float64),
        "theta_v": ((NPROMA, NLEV), np.float64),
        "rho": ((NPROMA, NLEV), np.float64),
        "hdef_ic": ((NPROMA, NLEV + 1), np.float64),
        "div_ic": ((NPROMA, NLEV + 1), np.float64),
        # the output of the excerpt's run lists it, so ICON passed loutshs = True
        "ddt_tke_hsh": ((NPROMA, NLEV + 1), np.float64),
    }
    if not ddt_tke_hsh:
        del shapes["ddt_tke_hsh"]
    live = {}
    for name, (shape, dtype) in shapes.items():
        fill = rng.integers(1, 9, shape) if dtype == np.int32 else rng.normal(size=shape)
        buffer = fortran_buffer(shape, dtype, fill=fill)
        device = None if device_xp is None else device_xp.array(buffer, order="F")
        comin.add(name, buffer, device, datatype=DOUBLE if dtype == np.float64 else INT)
        live[name] = buffer if device is None else device
    return live


def view(buffer: Any, rank: int) -> Any:
    return rank_view(buffer, rank)


def to_device(arguments: dict[str, Any], exported: Any, device_xp: Any) -> dict[str, Any]:
    """On the GPU ICON passes the device copies of the arrays py2fgen may take there."""
    if device_xp is None:
        return arguments
    descriptors = exported.param_descriptors
    return {
        param: device_xp.array(value, order="F")
        if isinstance(value, np.ndarray)
        and descriptors[param].memory_space == py2fgen.MemorySpace.MAYBE_DEVICE
        else value
        for param, value in arguments.items()
    }


def py2fgen_arguments(
    icon: Icon,
    variables: dict[str, Any],
    configuration: dict[str, dict[str, Any]],
    device_xp: Any = None,
) -> dict[str, dict[str, Any]]:
    """What ICON's py2fgen interface passes ('build_grid_init', 'build_diffusion_init'): the
    configuration of 'grid_init' as the plugin read it, the 25 values of 'diffusion_init' as ICON
    passed them in the run of the namelist excerpt ('PY2FGEN_DIFFUSION')."""
    grid: dict[str, Any] = {}
    for param, expected in PY2FGEN["grid_init"].items():
        value = expected(icon)
        if isinstance(value, np.ndarray):
            value = np.asfortranarray(np.array(value))
            if value.dtype == np.bool_:  # Fortran's logical(c_bool) bytes, as nvfortran writes them
                value = (value.astype(np.uint8) * TRUE_BYTE).view(np.bool_)
        grid[param] = value
    grid.update(configuration["grid_init"])
    init: dict[str, Any] = {
        param: np.asfortranarray(np.array(value(icon)))
        for param, value in PY2FGEN["diffusion_init"].items()
    }
    for param, icon_name in plugin.STATIC_VARIABLES["diffusion_init"].items():
        rank = diffusion_wrapper.diffusion_init.param_descriptors[param].rank
        init[param] = view(variables[icon_name], rank)
    init.update(PY2FGEN_DIFFUSION, backend=configuration["diffusion_init"]["backend"])
    return {
        "grid_init": to_device(grid, grid_wrapper.grid_init, device_xp),
        "diffusion_init": to_device(init, diffusion_wrapper.diffusion_init, device_xp),
    }


def run_arguments(variables: dict[str, Any], linit: bool, **override: Any) -> dict[str, Any]:
    """'build_diffusion_run' on ICON's own variables (SUBSTITUTE): no 'dwdx', 'dwdy'."""
    arguments: dict[str, Any] = {
        name: view(variables[name], 2)
        for name in ("w", "vn", "exner", "theta_v", "rho", "hdef_ic", "div_ic")
    }
    arguments.update(dwdx=None, dwdy=None, dtime=10.0, linit=linit)
    arguments.update(override)
    return arguments


@pytest.fixture
def make_run(stand_ins, monkeypatch, icon_run_dir, logs):
    """The plugin registered in probe mode with the real functions, the recorders installed
    (py2fgen's glue runs the extra callable), the secondary constructor done."""

    def make(device_xp: Any = None, ddt_tke_hsh: bool = True) -> types.SimpleNamespace:
        set_mode(icon_run_dir, mode=1, interface=0)
        monkeypatch.setattr(
            _probe,
            "communicators",
            lambda new, ref: (
                _compare.Communicator(new, (0,), "IDENT"),
                _compare.Communicator(ref, (0,)),
            ),
        )
        icon = Icon()
        comin = ProbeComIn(icon, has_device=device_xp is not None)
        variables = icon_variables(comin, device_xp, ddt_tke_hsh)
        instance = plugin.Plugin(comin, environ={plugin.PROBE_ENV: "1"}, granule=StandInGranule())
        instance.register()
        monkeypatch.setattr(plugin, "_instance", instance)
        pkgutil.resolve_name(_probe.INSTALL)()
        comin.fire("EP_SECONDARY_CONSTRUCTOR")
        arguments = py2fgen_arguments(icon, variables, instance._configuration, device_xp)
        return types.SimpleNamespace(
            icon=icon,
            comin=comin,
            variables=variables,
            plugin=instance,
            arguments=arguments,
            glue=Py2fgen(),
        )

    return make


def probe_lines(logs, prefix: str = "probe ") -> list[str]:
    return [m for m in logs.messages if m.startswith(prefix)]


def diffusion_step(run, initial: bool, call: bool = True, **override: Any) -> None:
    """One diffusion call as ICON makes it: DIFFUSION_BEFORE, py2fgen's diffusion_run, _AFTER;
    the initial one at the start of a time step, a regular one after the dynamics."""
    comin = run.comin
    if initial:
        comin.fire("EP_ATM_INTEGRATE_START", 1)
    else:
        comin.fire("EP_ATM_DYCORE_SOLVE_NH_BEFORE", 1)
        comin.fire("EP_ATM_DYCORE_SOLVE_NH_AFTER", 1)
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)
    if call:
        run.glue.call(
            "diffusion_run", run_arguments(run.variables, **{"linit": initial, **override})
        )
    comin.fire("EP_ATM_DYCORE_DIFFUSION_AFTER", 1)


N_DOMAINS = sum(len(list(h_grid.get_domains_for_dim(d))) for d in dims.horizontal_dims())
N_OBJECTS = 1 + 2 + 2 + 2 * N_DOMAINS + 4 + 4 + 4 + 3 + 25 + 1 + 1 + 1 + 1
"""
The stand-ins' objects: the grid's id, configuration (2), connectivity and its skip value, the
start and end index of every domain of cells, edges and vertices, the vertical grid
(configuration 2, vct_a, vct_b), the edge geometry (4), the cell geometry (4), the exchange
(type, size, rank), the granule (DiffusionConfig 25, a kept field, a computed field, two
values).
"""


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_probe_end_to_end(make_run, logs, on_device):
    device_xp = cupy_or_skip() if on_device else None
    module = "cupy" if on_device else "numpy"
    run = make_run(device_xp)
    setup = list(logs.messages)
    assert [m for m in setup if m.startswith("py2fgen probe: installed;")] == [
        "py2fgen probe: installed; grid_init, diffusion_init, diffusion_run replaced by recorders"
        " before py2fgen's glue imports them"
    ]
    assert any(m.startswith("ICON4PY_COMIN_PROBE=1: the py2fgen probe compares") for m in setup)
    assert any(
        m.startswith("idle: ICON mode SUBSTITUTE (icon4py_interface=0,") and "py2fgen probe" in m
        for m in setup
    )
    # icon4py_init: py2fgen's grid_init and diffusion_init; then EP_ATM_TIMELOOP_BEFORE
    for name in ("grid_init", "diffusion_init"):
        run.glue.call(name, run.arguments[name])
    assert [name for name, _ in CALLS] == ["grid_init", "diffusion_init"]
    run.comin.fire("EP_ATM_TIMELOOP_BEFORE")
    lines = probe_lines(logs)
    for name, n in (("grid_init", 56), ("diffusion_init", 39)):
        items = [m for m in lines if m.startswith(f"probe {name} pass 1 ") and " vs py2fgen: " in m]
        assert len(items) == n and all(" vs py2fgen: identical (" in m for m in items), items
        assert (
            f"probe {name} pass 1: {n} compared, {n} identical, 0 differ; control {n} of {n} reported; not probed 0"
            in lines
        )
    (c2e,) = [m for m in lines if m.startswith("probe grid_init pass 1 c2e ")]
    assert c2e == (
        "probe grid_init pass 1 c2e vs py2fgen: identical (int32 (8, 3); real 15: 0 differ;"
        " padding 9: 0 differ)"
    )
    (mask,) = [m for m in lines if m.startswith("probe grid_init pass 1 e_owner_mask ")]
    assert mask.endswith("; as values, bytes [0, 1] vs [0, 255])")
    (theta,) = [m for m in lines if m.startswith("probe diffusion_init pass 1 theta_ref_mc ")]
    address = _views.data_ptr(run.variables["theta_ref_mc"])
    assert (
        f"(same address {address:#x}; float64 (8, 3); real 15: 0 differ; padding 9: 0 differ)"
        in theta
    )
    (geofac,) = [m for m in lines if m.startswith("probe diffusion_init pass 1 geofac_div ")]
    assert "identical (float64 (8, 3); real 15: 0 differ; padding 9: 0 differ)" in geofac
    (comm,) = [m for m in lines if m.startswith("probe grid_init pass 1 comm_id ")]
    assert "MPI_Comm_compare IDENT" in comm
    # the objects: the plugin built its grid and granule from its inputs, as when it computes
    objects = [m for m in lines if m.startswith("probe objects pass 1 ") and " vs py2fgen: " in m]
    assert len(objects) == N_OBJECTS and all(" vs py2fgen: identical (" in m for m in objects)
    assert (
        f"probe objects pass 1: {N_OBJECTS} compared, {N_OBJECTS} identical, 0 differ; control"
        f" {N_OBJECTS} of {N_OBJECTS} reported; not probed 0" in lines
    )
    assert (
        f"probe objects pass 1: {N_OBJECTS} items of the plugin's objects; not compared (programs"
        " and runtime objects): diffusion.halo_exchange_wait (object),"
        " diffusion.apply_diffusion_to_vn (partial)" in lines
    )
    (connectivity,) = [
        m for m in objects if m.startswith("probe objects pass 1 grid.connectivity.C2E ")
    ]
    assert connectivity.endswith("identical (int32 (8, 3); real 24: 0 differ; padding 0: 0 differ)")
    (hdiff,) = [m for m in objects if " diffusion.config.hdiff_efdt_ratio " in m]
    assert hdiff.endswith("identical (float 24.0 (0x1.8000000000000p+4, bits 0000000000003840))")
    assert any(" diffusion.config.diffusion_type vs py2fgen: identical (" in m for m in objects)
    assert any(m.startswith("probe objects pass 1 exchange.get_size ") for m in objects)
    # the granule got py2fgen's arguments, unchanged
    grid_kwargs = CALLS[0][1]
    assert grid_kwargs["c_owner_mask"].view(np.uint8).max() == TRUE_BYTE
    # the diffusion calls: the initial one and a regular one
    diffusion_step(run, initial=True)
    diffusion_step(run, initial=False)
    lines = probe_lines(logs)
    first = [
        m for m in lines if m.startswith("probe diffusion_run call 1 ") and " vs py2fgen: " in m
    ]
    assert len(first) == 11 and all(" vs py2fgen: identical (" in m for m in first)
    (vn,) = [m for m in first if " vn " in m]
    assert vn == (
        f"probe diffusion_run call 1 vn vs py2fgen: identical (same address"
        f" {_views.data_ptr(run.variables['vn']):#x}; Field[Edge, K] over {module} float64 (8, 3)"
        " strides (1, 8))"
    )
    assert "probe diffusion_run call 1 dwdx vs py2fgen: identical (absent on both routes)" in first
    assert "probe diffusion_run call 1 linit vs py2fgen: identical (bool True)" in first
    for k in (1, 2):
        assert (
            f"probe diffusion_run call {k}: 11 compared, 11 identical, 0 differ; control 11 of 11 reported; not probed 0"
            in lines
        )
    assert not [
        m for m in lines if m.startswith("probe diffusion_run call 2 ") and " vs py2fgen: " in m
    ]
    assert [name for name, _ in CALLS] == [
        "grid_init",
        "diffusion_init",
        "diffusion_run",
        "diffusion_run",
    ]
    run.comin.fire("EP_DESTRUCTOR")
    summary = probe_lines(logs, "probe summary")
    assert summary == [
        "probe summary grid_init: py2fgen calls 1, compared 1 (56 items each), items 56, identical 56,"
        " differ 0; control 56 of 56 reported; not probed 0",
        "probe summary diffusion_init: py2fgen calls 1, compared 1 (39 items each), items 39,"
        " identical 39, differ 0; control 39 of 39 reported; not probed 0",
        "probe summary diffusion_run: py2fgen calls 2, compared 2 (11 items each), items 22,"
        " identical 22, differ 0; control 22 of 22 reported; not probed 0",
        f"probe summary objects: py2fgen calls 1, compared 1 ({N_OBJECTS} items each), items"
        f" {N_OBJECTS}, identical {N_OBJECTS}, differ 0; control {N_OBJECTS} of {N_OBJECTS}"
        " reported; not probed 0",
        "probe summary: 117 items compared, 117 identical, 0 differ; control 117 of 117 reported;"
        " not probed 0",
    ]
    assert run.plugin._granule.released
    # the plugin computed nothing, wrote nothing and requested nothing for writing
    assert not run.plugin.active
    assert all(flag & run.comin.COMIN_FLAG_WRITE == 0 for flag in run.comin.contexts.values())


@pytest.mark.parametrize("py2fgen_loutshs", [False, True])
def test_probe_compares_icons_loutshs_at_run_time(make_run, logs, py2fgen_loutshs):
    """An output without ddt_tke_hsh: ICON passes loutshs = False to py2fgen, and so does the
    plugin (its configured value is True); the probe reports a py2fgen value that differs."""
    run = make_run(ddt_tke_hsh=False)
    init = dict(run.arguments["diffusion_init"], loutshs=py2fgen_loutshs)
    run.glue.call("grid_init", run.arguments["grid_init"])
    run.glue.call("diffusion_init", init)
    run.comin.fire("EP_ATM_TIMELOOP_BEFORE")
    (line,) = [m for m in probe_lines(logs) if m.startswith("probe diffusion_init pass 1 loutshs ")]
    if py2fgen_loutshs:
        assert (
            line
            == "probe diffusion_init pass 1 loutshs vs py2fgen: differ (bool False vs bool True)"
        )
    else:
        assert line == "probe diffusion_init pass 1 loutshs vs py2fgen: identical (bool False)"
    assert any(
        m.startswith("diffusion_init configuration at run time: loutshs F: ") for m in logs.messages
    )


def test_probe_reports_differences(make_run, logs):
    run = make_run()
    grid = dict(run.arguments["grid_init"])
    grid["c2e"] = grid["c2e"].copy(order="F")
    grid["c2e"][2, 1] += 1  # a real entry
    init = dict(run.arguments["diffusion_init"])
    init["theta_ref_mc"] = init["theta_ref_mc"].copy(order="F")  # equal values, other memory
    init["hdiff_efdt_ratio"] += 0.5
    run.glue.call("grid_init", grid)
    run.glue.call("diffusion_init", init)
    run.comin.fire("EP_ATM_TIMELOOP_BEFORE")
    lines = probe_lines(logs)
    (c2e,) = [m for m in lines if m.startswith("probe grid_init pass 1 c2e ")]
    assert "differ (int32 (8, 3); real 15: 1 differ, first (2, 1), max abs 1" in c2e
    (theta,) = [m for m in lines if m.startswith("probe diffusion_init pass 1 theta_ref_mc ")]
    assert " differ (address 0x" in theta
    assert any(
        m.startswith("probe diffusion_init pass 1 hdiff_efdt_ratio vs py2fgen: differ (float")
        for m in lines
    )
    assert (
        "probe grid_init pass 1: 56 compared, 55 identical, 1 differ: c2e; control 56 of 56 reported; not probed 0"
        in lines
    )
    assert (
        "probe diffusion_init pass 1: 39 compared, 37 identical, 2 differ: theta_ref_mc"
        " hdiff_efdt_ratio; control 39 of 39 reported; not probed 0" in lines
    )
    # the objects py2fgen's bindings built from them differ there too (values only: theta_ref_mc
    # has the same values in other memory)
    assert (
        f"probe objects pass 1: {N_OBJECTS} compared, {N_OBJECTS - 3} identical, 3 differ:"
        " grid.connectivity.C2E diffusion.config.hdiff_efdt_ratio diffusion.diff_multfac_vn;"
        f" control {N_OBJECTS} of {N_OBJECTS} reported; not probed 0" in lines
    )
    # diffusion_run: vn on other memory, linit wrong; then a call site without py2fgen's call
    diffusion_step(
        run, initial=True, vn=run.variables["vn"][:, :, 0, 0, 0].copy(order="F"), linit=False
    )
    diffusion_step(run, initial=False, call=False)
    lines = probe_lines(logs)
    assert any(
        m.startswith("probe diffusion_run call 1 vn vs py2fgen: differ (address 0x") for m in lines
    )
    assert "probe diffusion_run call 1 linit vs py2fgen: differ (bool True vs bool False)" in lines
    assert "probe diffusion_run call 2: py2fgen made no call at this call site; not probed" in lines
    run.comin.fire("EP_DESTRUCTOR")
    (total,) = probe_lines(logs, "probe summary:")
    assert total == (
        "probe summary: 106 items compared, 101 identical, 5 differ; control 106 of 106 reported;"
        " not probed 1"
    )


def test_probe_without_the_recorders(make_run, logs):
    """py2fgen made no call (no recorders installed): nothing compared, everything reported."""
    run = make_run()
    run.comin.fire("EP_ATM_TIMELOOP_BEFORE")
    lines = probe_lines(logs)
    assert any(
        m.startswith("probe grid_init pass 1: py2fgen made 0 call(s) of grid_init; not probed")
        for m in lines
    )
    assert "probe objects pass 1: 0 record(s) of py2fgen's objects; not probed" in probe_lines(logs)
    run.comin.fire("EP_DESTRUCTOR")
    (total,) = probe_lines(logs, "probe summary:")
    assert total.endswith("not probed 2")
    (objects,) = probe_lines(logs, "probe summary objects:")
    assert objects.endswith("not probed 1: pass 1 (0 record(s) of py2fgen's objects)")


@pytest.mark.parametrize("mode, interface", [(1, 1), (2, 0), (0, 0)])
def test_probe_needs_py2fgen_substitute(icon_run_dir, mode, interface):
    set_mode(icon_run_dir, mode=mode, interface=interface)
    instance = plugin.Plugin(
        ProbeComIn(Icon()), environ={plugin.PROBE_ENV: "1"}, granule=StandInGranule()
    )
    with pytest.raises(
        RuntimeError, match="must compute the diffusion through py2fgen in SUBSTITUTE"
    ):
        instance.register()


def test_recorder_failure_does_not_stop_py2fgen(make_run, logs, monkeypatch):
    run = make_run()

    def fail(name, kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(run.plugin.probe, "on_call", fail)
    run.glue.call("grid_init", run.arguments["grid_init"])
    assert [name for name, _ in CALLS] == ["grid_init"]
    assert "py2fgen probe: grid_init: ValueError('boom')" in logs.messages


def test_a_failing_build_is_reported(make_run, logs, monkeypatch):
    """The plugin's grid or granule cannot be built: reported, py2fgen computes on."""
    run = make_run()
    for name in ("grid_init", "diffusion_init"):
        run.glue.call(name, run.arguments[name])

    def fail(**kwargs: Any) -> None:
        raise ValueError("no grid")

    object.__setattr__(run.plugin._signatures["grid_init"], "function", fail)
    run.comin.fire("EP_ATM_TIMELOOP_BEFORE")
    assert "probe objects pass 1: not probed: grid_init: ValueError('no grid')" in logs.messages
    run.comin.fire("EP_DESTRUCTOR")
    (objects,) = probe_lines(logs, "probe summary objects:")
    assert objects.endswith("not probed 1: pass 1 (grid_init: ValueError('no grid'))")


def test_object_items():
    """What the probe compares of an object, and what it skips."""
    field = gtx.as_field([dims.CellDim], np.arange(3.0))
    value = types.SimpleNamespace(
        values={"a": 1, "b": (field, None)},
        mode=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        zone=types.SimpleNamespace(),
    )
    items: dict[str, Any] = {}
    skipped: list[str] = []
    _probe._flatten(items, skipped, "x", vars(value))
    assert list(items) == ["x.values.a", "x.values.b[0]", "x.values.b[1]", "x.mode"]
    assert isinstance(items["x.values.b[0]"], _probe.Array) and items["x.values.b[1]"] is None
    assert items["x.mode"] == diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER  # an int enumeration
    assert skipped == ["x.zone (SimpleNamespace)"]
    deep: Any = 1
    for _ in range(6):
        deep = {"d": deep}
    items, skipped = {}, []
    _probe._flatten(items, skipped, "y", deep)
    assert items == {} and skipped == ["y.d.d.d.d.d (dict, nested deeper than 4)"]
