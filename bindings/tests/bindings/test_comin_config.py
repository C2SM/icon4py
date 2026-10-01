# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the icon4py ComIn plugin's configuration from ICON's namelist output ('_config.py'):
the reader, ICON's mode, the configuration arguments, and what the plugin does with them.

The fixture 'comin_run_dir/NAMELIST_ICON_output_atm' is a verbatim excerpt (the groups run_nml,
gridref_nml, interpol_nml, sleve_nml, nonhydrostatic_nml, diffusion_nml, turbdiff_nml, and the
first three entries of initicon_nml, whose other 28 000 lines are file names) of the namelist
output of a ComIn diffusion SUBSTITUTE run of 'mch_icon-ch1_small' (nvfortran).
"""

import dataclasses
import logging
import sys
import types
import typing

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import diffusion_wrapper, grid_wrapper, icon4py_export
from icon4py.bindings.comin import _config, _dual, _marshal, plugin
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.config import options
from icon4py.model.common.grid import vertical

from .test_comin_marshal import (  # fixtures: icon_run_dir, logs, restore_package_logger, wrappers
    NAMELIST_EXCERPT,
    FakeComIn,
    FakeIcon,
    fortran_buffer,
    icon_run_dir,
    logs,
    restore_package_logger,
    set_mode,
    set_namelist,
    wrappers,
)


REAL_HOST_COMM = _config.host_comm
"""'_config.host_comm' before the 'icon_run_dir' fixture replaces it."""


def excerpt() -> _config.Namelists:
    return _config.Namelists.from_text(NAMELIST_EXCERPT.read_text())


# ---- the reader --------------------------------------------------------------------------------


def test_parse_the_excerpt():
    nml = excerpt()
    assert list(nml.groups) == [
        "run_nml",
        "gridref_nml",
        "interpol_nml",
        "sleve_nml",
        "nonhydrostatic_nml",
        "diffusion_nml",
        "turbdiff_nml",
        "initicon_nml",
    ]
    assert nml.get("initicon_nml", "init_mode") == (7,)
    assert nml.get("run_nml", "icon4py_mode") == (1,)
    assert nml.get("RUN_NML", "LUSE_ICON4PY_DIFFUSION") == (True,)  # names are case-insensitive
    assert nml.get("run_nml", "luse_radarfwo") == (False,) * 10  # continued on 9 lines
    assert nml.get("run_nml", "num_lev") == (80,) + (31,) * 9
    assert nml.get("run_nml", "nsteps") == (-999,)
    output = nml.get("run_nml", "output")
    assert output is not None and len(output) == 5 and output[0] == "nml" + " " * 29
    (restart,) = nml.get("run_nml", "restart_filename") or ()
    assert isinstance(restart, str) and restart.startswith("<gridfile>_restart_<mtype>_<rsttime>.")
    assert nml.get("run_nml", "lmsgwam") == (False,) * 10  # the group's last entry: no comma
    assert nml.get("gridref_nml", "fbk_relax_timescale") == (10800.0,)
    assert nml.get("nonhydrostatic_nml", "damp_height") == (12250.0,) * 10
    # E format with 17 significant digits, F format with 16: both read back bit for bit
    (fac,) = nml.get("diffusion_nml", "hdiff_smag_fac") or ()
    assert fac.hex() == (0.025).hex()  # "2.5000000000000001E-002"
    (fac2,) = nml.get("diffusion_nml", "hdiff_smag_fac2") or ()
    assert fac2.hex() == "0x1.2457800000000p-4"  # "7.1372509002685547E-002"
    assert nml.get("sleve_nml", "stretch_fac") == (0.65,)  # "0.6500000000000000"
    assert nml.get("interpol_nml", "nudge_max_coeff") == (0.075,)
    assert len(nml.get("interpol_nml", "rbf_coeffs_filename") or ()) == 10
    assert nml.get("run_nml", "no_such_entry") is None
    assert nml.get("no_such_nml", "x") is None


@pytest.mark.parametrize(
    "value", [0.1, 1 / 3, 12250.0, 7.137250900268555e-2, -2.5e-300, 1.7976931348623157e308]
)
def test_reals_read_back_bitwise(value):
    """ICON writes reals with 17 significant digits in E format; D exponents are read too."""
    mantissa, exponent = f"{value:.16E}".split("E")
    for text in (f"{mantissa}E{int(exponent):+04d}", f"{mantissa}D{int(exponent):+04d}"):
        (read,) = _config.parse(f" &A_NML\n X = {text}\n /\n")["a_nml"][0]["x"]
        assert read.hex() == value.hex(), text


def test_values_of_every_kind():
    text = (
        " &KINDS_NML\n"
        " A = .TRUE., b=.false.,  C = T,\n"
        " D = -12, E = +3,\n"
        " F = 3*F,\n"
        " G = 2*1.5, 1.0D-3,\n"
        ' H = \'it\'\'s, a/b = &c\', "say ""hi""",\n'
        " I(1) = 5,\n"
        " J = 2*,\n"
        " K = 1, 2*, 3\n"
        " /\n"
        " &KINDS_NML\n"
        " A = F\n"
        " /\n"
    )
    groups = _config.parse(text)
    first, second = groups["kinds_nml"]
    assert first == {
        "a": (True,),
        "b": (False,),
        "c": (True,),
        "d": (-12,),
        "e": (3,),
        "f": (False, False, False),
        "g": (1.5, 1.5, 1e-3),
        "h": ("it's, a/b = &c", 'say "hi"'),
        "i(1)": (5,),
        "j": (None, None),
        "k": (1, None, None, 3),
    }
    assert second == {"a": (False,)}
    assert type(first["d"][0]) is int and type(first["g"][0]) is float
    nml = _config.Namelists(groups)
    with pytest.raises(_config.NamelistError, match="'&kinds_nml' occurs 2 times"):
        nml.get("kinds_nml", "a")
    assert nml.fortran_dict()["kinds_nml"][0]["f"] == [False, False, False]
    assert nml.fortran_dict()["kinds_nml"][1] == {"a": False}


def test_an_entry_written_twice():
    # ICON lists 'nrestart_streams' twice in 'io_nml', and nvfortran writes it twice
    text = " &IO_NML\n N = 1,\n X = 2,\n N = 1\n /\n"
    assert _config.parse(text) == {"io_nml": [{"n": (1,), "x": (2,)}]}
    with pytest.raises(_config.NamelistError, match=r"line 5: entry 'n' twice .* other values"):
        _config.parse(" &IO_NML\n N = 1,\n X = 2,\n N = 2\n /\n")


@pytest.mark.parametrize(
    "text, message",
    [
        (" &A_NML\n X = 1,\n", "line 3: '&a_nml' is not closed"),
        (" X = 1\n", "line 1: entry 'X' outside a namelist group"),
        (" /\n", "line 1: '/' outside a namelist group"),
        (" &A_NML\n &B_NML\n /\n", "line 2: '&B_NML' inside '&a_nml'"),
        (" &A_NML\n 5,\n /\n", "line 2: value 5 outside a namelist entry"),
        (" &A_NML\n X = (1.0, 2.0)\n /\n", "line 2: cannot read the value '\\(1.0'"),
        (" &A_NML\n X = 1.2.3\n /\n", "line 2: cannot read the value '1.2.3'"),
    ],
)
def test_parse_errors(text, message):
    with pytest.raises(_config.NamelistError, match=message):
        _config.parse(text)


def test_fortran_dict_feeds_icon4py_readers():
    """The layout of f90nml's dictionaries: icon4py's own readers take it."""
    config = excerpt().fortran_dict()
    assert config["run_nml"]["icon4py_mode"] == 1 and config["run_nml"]["num_lev"][0] == 80
    vertical_config = vertical.VerticalGridConfig.from_fortran_dict(config)
    assert vertical_config.num_levels == 80
    diffusion_config = diffusion.DiffusionConfig.from_fortran_dict(config)
    assert diffusion_config.ndyn_substeps == 5


# ---- the configuration arguments ---------------------------------------------------------------

REAL = {"grid_init": grid_wrapper.grid_init, "diffusion_init": diffusion_wrapper.diffusion_init}


def kinds(function: str) -> dict[str, type]:
    descriptors = REAL[function].param_descriptors
    return {
        name: plugin.SCALAR_TYPES[descriptors[name].dtype] for name in _config.ARGUMENTS[function]
    }


EXPECTED = {
    "grid_init": {
        "lowest_layer_thickness": 20.0,
        "model_top_height": 22000.0,
        "stretch_factor": 0.65,
        "flat_height": 16000.0,
        "rayleigh_damping_height": 12250.0,
        "backend": 0,
    },
    "diffusion_init": {
        "ndyn_substeps": 5,
        "diffusion_type": 5,
        "hdiff_w": True,
        "hdiff_vn": True,
        "hdiff_smag_w": False,
        "zdiffu_t": True,
        "type_t_diffu": 2,
        "type_vn_diffu": 1,
        "hdiff_efdt_ratio": 24.0,
        "hdiff_w_efdt_ratio": 15.0,
        "smagorinski_scaling_factor": 0.025,
        "smagorinski_scaling_factor2": float.fromhex("0x1.2457800000000p-4"),
        "smagorinski_scaling_factor3": 0.0,
        "smagorinski_scaling_factor4": 1.0,
        "smagorinski_scaling_height": 32500.0,
        "smagorinski_scaling_height2": 60686.25390625,
        "smagorinski_scaling_height3": 50000.0,
        "smagorinski_scaling_height4": 90000.0,
        "hdiff_temp": True,
        "denom_diffu_v": 150.0,
        "nudge_max_coeff": 5.0 * 0.075,
        "itype_sher": 2,
        "iforcing": 3,
        "a_hshr": 2.0,
        "loutshs": True,
        "backend": 0,
    },
}
"""ICON's values in the run of the excerpt."""


def _bits(value: typing.Any) -> tuple[type, str]:
    return type(value), value.hex() if isinstance(value, float) else repr(value)


def test_the_32_configuration_arguments():
    nml = excerpt()
    count = 0
    for function, expected in EXPECTED.items():
        # every scalar of class D of the source table, and nothing else
        assert set(_config.ARGUMENTS[function]) == {
            p for p, s in plugin.SOURCES[function].items() if s.klass == "D"
        }
        values, errors = _config.arguments(nml, function, kinds(function))
        assert errors == []
        assert {k: _bits(v) for k, v in values.items()} == {
            k: _bits(v) for k, v in expected.items()
        }
        count += len(values)
    assert count == 32


def test_arguments_are_named_after_icon4pys_config():
    """Each diffusion argument reads the entry of icon4py's 'IconOption' of its config field."""
    fields = {  # 'diffusion_wrapper.diffusion_init': argument -> DiffusionConfig field
        "diffusion_type": "diffusion_type",
        "hdiff_w": "apply_to_vertical_wind",
        "hdiff_vn": "apply_to_horizontal_wind",
        "hdiff_smag_w": "apply_smag_diff_to_vertical_wind",
        "zdiffu_t": "apply_zdiffusion_t",
        "type_t_diffu": "type_t_diffu",
        "type_vn_diffu": "type_vn_diffu",
        "hdiff_efdt_ratio": "hdiff_efdt_ratio",
        "hdiff_w_efdt_ratio": "hdiff_w_efdt_ratio",
        **{f"smagorinski_scaling_{k}{i}": f"smagorinski_scaling_{k}{i}"
           for k in ("factor", "height") for i in ("", "2", "3", "4")},
        "hdiff_temp": "apply_to_temperature",
        "ndyn_substeps": "ndyn_substeps",
        "denom_diffu_v": "velocity_boundary_diffusion_denominator",
        "nudge_max_coeff": "max_nudging_coefficient",
        "itype_sher": "shear_type",
        "iforcing": "iforcing",
        "a_hshr": "a_hshr",
        "loutshs": "loutshs",
    }  # fmt: skip
    icon_options = {
        name: option.icon_equivalent
        for name, option in options.ConfigOption.iter_from_config_class(diffusion.DiffusionConfig)
    }
    config = diffusion.DiffusionConfig.from_fortran_dict(excerpt().fortran_dict())
    ours, _ = _config.arguments(excerpt(), "diffusion_init", kinds("diffusion_init"))
    for argument, setting in _config.ARGUMENTS["diffusion_init"].items():
        if argument == "backend":
            continue  # EXCLAIM's switch, no diffusion option
        icon_option = icon_options[fields[argument]]
        if argument == "loutshs":
            assert icon_option is None  # not a namelist entry
            continue
        assert isinstance(icon_option, options.IconOption)
        assert (setting.group, setting.name) == (*icon_option.path, icon_option.name), argument
        assert (setting.index == 0) == icon_option.list_to_value, argument
        if icon_option.read_from_icon:  # all but nudge_max_coeff, which ICON scales
            value = getattr(config, fields[argument])
            assert _bits(type(ours[argument])(value)) == _bits(ours[argument]), argument
    # the vertical ones: the entries of 'VerticalGridConfig.from_fortran_dict'
    vertical_config = vertical.VerticalGridConfig.from_fortran_dict(excerpt().fortran_dict())
    grid, _ = _config.arguments(excerpt(), "grid_init", kinds("grid_init"))
    for argument in grid.keys() - {"backend"}:
        assert _bits(getattr(vertical_config, argument)) == _bits(grid[argument]), argument


def test_effective_values_where_icon_differs_from_the_namelist():
    def values(text: str) -> dict:
        found, errors = _config.arguments(
            _config.Namelists.from_text(text),
            "diffusion_init",
            {"nudge_max_coeff": float, "loutshs": bool},
        )
        assert errors == []
        return found

    run = " &RUN_NML\n LDYNAMICS = {}\n /\n"
    interpol = " &INTERPOL_NML\n NUDGE_MAX_COEFF = 2.0000000000000000E-002\n /\n"
    got = values(run.format("T") + interpol + " &TURBDIFF_NML\n A_HSHR = 1.0\n /\n")
    assert got == {"nudge_max_coeff": 5.0 * 0.02, "loutshs": True}  # ICON's default
    assert values(run.format("F") + interpol + " &TURBDIFF_NML\n /\n")["loutshs"] is False
    # an ICON that writes it: the entry wins
    assert values(run.format("T") + interpol + " &TURBDIFF_NML\n LOUTSHS = F\n /\n") == {
        "nudge_max_coeff": 0.1,
        "loutshs": False,
    }
    damp = " &NONHYDROSTATIC_NML\n DAMP_HEIGHT = 1000.0, 2000.0\n /\n"
    got, errors = _config.arguments(
        _config.Namelists.from_text(damp), "grid_init", {"rayleigh_damping_height": float}
    )
    assert got == {"rayleigh_damping_height": 1000.0} and errors == []  # domain 1


def test_argument_problems_are_collected():
    text = (
        " &DIFFUSION_NML\n HDIFF_ORDER = 5.0,\n LHDIFF_W = 1,\n LHDIFF_VN = T, F\n /\n"
        " &INTERPOL_NML\n NUDGE_MAX_COEFF = 2\n /\n"
    )
    found, errors = _config.arguments(
        _config.Namelists.from_text(text),
        "diffusion_init",
        {
            "diffusion_type": int,
            "hdiff_w": bool,
            "hdiff_vn": bool,
            "nudge_max_coeff": float,
            "ndyn_substeps": int,
            "no_such_argument": int,
        },
    )
    assert found == {}
    assert errors == [
        "diffusion_init: 'diffusion_type': diffusion_nml: hdiff_order = 5.0, expected a int.",
        "diffusion_init: 'hdiff_w': diffusion_nml: lhdiff_w = 1, expected a bool.",
        "diffusion_init: 'hdiff_vn': diffusion_nml: lhdiff_vn: 2 values, expected one.",
        "diffusion_init: 'nudge_max_coeff': interpol_nml: nudge_max_coeff = 2, expected a float.",
        "diffusion_init: 'ndyn_substeps': NAMELIST_ICON_output_atm has no nonhydrostatic_nml:"
        " ndyn_substeps.",
        "diffusion_init: 'no_such_argument' has no namelist entry.",
    ]


# ---- ICON's mode -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interface, luse, mode, name, computes, exposes, says",
    [
        (1, True, 1, "SUBSTITUTE", True, True, "this plugin computes the horizontal diffusion"),
        (1, True, 2, "VERIFY", True, True, "this plugin computes it too, at EP_X"),
        (1, True, 0, "OFF", False, False, "ICON computes the horizontal diffusion; this plugin"),
        (0, True, 1, "SUBSTITUTE", False, False, "through py2fgen, not ComIn; this plugin stays"),
        (0, True, 0, "OFF", False, False, "ICON computes the horizontal diffusion; this plugin"),
        # ICON stops at its namelist check with these, but the dispatcher's rule is clear
        (1, False, 1, "OFF", False, True, "ICON computes the horizontal diffusion; this plugin"),
        (None, None, None, "OFF", False, False, "ICON has no icon4py switches"),
    ],
)
def test_icon_mode(  # noqa: PLR0917 [too-many-positional-arguments]
    interface, luse, mode, name, computes, exposes, says
):
    icon_mode = _config.IconMode(interface, luse, mode)
    assert (icon_mode.name, icon_mode.computes, icon_mode.exposes_arguments) == (
        name,
        computes,
        exposes,
    )
    line = icon_mode.describe("EP_X")
    assert line.startswith(f"ICON mode {name} (icon4py_interface=") and says in line
    assert line.endswith("ICON skips its own" if name == "SUBSTITUTE" and computes else "")


def test_icon_mode_from_the_namelist_output():
    assert _config.icon_mode(excerpt()) == _config.IconMode(1, True, 1)
    missing = _config.icon_mode(_config.Namelists.from_text(" &RUN_NML\n NSTEPS = 5\n /\n"))
    assert missing == _config.IconMode(None, None, None) and not missing.computes
    with pytest.raises(_config.NamelistError, match=r"icon4py_mode = 1\.0, expected a int"):
        _config.icon_mode(_config.Namelists.from_text(" &RUN_NML\n ICON4PY_MODE = 1.0\n /\n"))


# ---- reading on rank 0 -------------------------------------------------------------------------


class FakeComm:
    """Two ranks of an mpi4py communicator, run one after the other: rank 0 broadcasts."""

    def __init__(self, rank: int, wire: dict) -> None:
        self.rank, self.wire = rank, wire

    def Get_rank(self) -> int:  # mpi4py's name
        return self.rank

    def bcast(self, obj, root=0):
        assert root == 0
        if self.rank == 0:
            self.wire["message"] = obj
            return obj
        assert obj == (None, None)  # rank 1 reads nothing
        return self.wire["message"]


def test_load_reads_on_rank_zero_only(tmp_path):
    wire: dict = {}
    on_rank0 = _config.load(FakeComm(0, wire), NAMELIST_EXCERPT)
    on_rank1 = _config.load(FakeComm(1, wire), tmp_path / "not_there")
    assert on_rank0 == on_rank1 == excerpt()


def test_load_failure_raises_on_every_rank(tmp_path):
    wire: dict = {}
    for rank in (0, 1):
        with pytest.raises(RuntimeError, match=r"cannot read ICON's namelist output .*not_there"):
            _config.load(FakeComm(rank, wire), tmp_path / "not_there")


def test_host_comm_is_comins_communicator(monkeypatch):
    """mpi4py's view of ComIn's host communicator (a Fortran handle); ICON initializes MPI."""
    handles = []

    def f2py(handle):
        handles.append(handle)
        return f"communicator {handle}"

    mpi = types.SimpleNamespace(Comm=types.SimpleNamespace(f2py=f2py))
    monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=mpi))
    comin = types.SimpleNamespace(parallel_get_host_mpi_comm=lambda: 42)
    assert REAL_HOST_COMM(comin) == "communicator 42" and handles == [42]


# ---- the plugin: ICON's mode and the configuration arguments -------------------------------------

CALLS: list[tuple[str, dict]] = []


@icon4py_export.export
def cfg_grid_init(
    area: fa.CellField[gtx.float64], rayleigh_damping_height: gtx.float64, backend: gtx.int32
) -> None:
    CALLS.append(
        ("grid_init", dict(rayleigh_damping_height=rayleigh_damping_height, backend=backend))
    )


@icon4py_export.export
def cfg_diffusion_init(
    ndyn_substeps: gtx.int32,
    hdiff_vn: bool,
    nudge_max_coeff: gtx.float64,
    loutshs: bool,
    backend: gtx.int32,
) -> None:
    CALLS.append(
        (
            "diffusion_init",
            dict(
                ndyn_substeps=ndyn_substeps,
                hdiff_vn=hdiff_vn,
                nudge_max_coeff=nudge_max_coeff,
                loutshs=loutshs,
                backend=backend,
            ),
        )
    )


@icon4py_export.export
def cfg_diffusion_run(w: fa.CellKField[gtx.float64], dtime: gtx.float64, linit: bool) -> None:
    CALLS.append(("diffusion_run", dict(dtime=dtime, linit=linit)))
    w.ndarray[...] += dtime


CFG_FUNCTIONS = {
    "grid_init": plugin.FunctionEntry(
        cfg_grid_init, "EP_ATM_TIMELOOP_BEFORE", False, _marshal.PASS_KEY
    ),
    "diffusion_init": plugin.FunctionEntry(
        cfg_diffusion_init, "EP_ATM_TIMELOOP_BEFORE", False, _marshal.PASS_KEY
    ),
    "diffusion_run": plugin.FunctionEntry(
        cfg_diffusion_run, "EP_ATM_DIFFUSION_ENTER", True, _marshal.CALL_COUNT_KEY
    ),
}
CFG_SOURCES = {
    name: {
        param: plugin.SOURCES[name].get(param, _dual.Source("B", _dual.OLD, "cell"))
        if name != "diffusion_run"
        else _dual.Source("B", _dual.OLD)  # called at ENTER on the arguments (the old route)
        for param in entry.exported.param_descriptors
    }
    for name, entry in CFG_FUNCTIONS.items()
}
CONFIGURATION = {
    name: {p: EXPECTED[name][p] for p, s in CFG_SOURCES[name].items() if s.provider == _dual.NEW}
    for name in ("grid_init", "diffusion_init")
}
"""The toy functions' NEW arguments and ICON's values of them in the run of the excerpt."""
NC, NLEV = 4, 3


@pytest.fixture
def cfg_calls():
    CALLS.clear()
    yield CALLS
    CALLS.clear()


def cfg_icon(comin: FakeComIn, **carrier: typing.Any) -> FakeIcon:
    """ICON's side of the toy functions; the carriers hold ICON's values, or 'carrier'."""
    icon = FakeIcon(comin, CFG_FUNCTIONS)
    scalars = {
        name: {k: carrier.get(k, v) for k, v in values.items()}
        for name, values in CONFIGURATION.items()
    }
    icon.expose("grid_init", {"area": fortran_buffer((NC,), np.float64, 1.0)}, scalars["grid_init"])
    icon.expose("diffusion_init", {}, scalars["diffusion_init"])
    icon.expose(
        "diffusion_run",
        {"w": fortran_buffer((NC, NLEV), np.float64, 1.0)},
        dict(dtime=0.0, linit=False),
    )
    return icon


def cfg_plugin(comin: FakeComIn, **environ: str) -> plugin.Plugin:
    instance = plugin.Plugin(comin, functions=CFG_FUNCTIONS, environ=environ, sources=CFG_SOURCES)
    instance.register()
    return instance


def dual_lines(logs) -> list[str]:
    return [m for m in logs.messages if m.startswith("dual EP_")]


def test_configuration_is_new_and_checked(cfg_calls, wrappers, logs):
    comin = FakeComIn()
    icon = cfg_icon(comin)
    cfg_plugin(comin)
    assert (
        "diffusion_init configuration from NAMELIST_ICON_output_atm: ndyn_substeps 5,"
        " hdiff_vn True, nudge_max_coeff 0.375, loutshs True, backend 0" in logs.messages
    )
    icon.secondary_constructor()
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in cfg_calls] == ["grid_init", "diffusion_init", "diffusion_run"]
    for (_, got), expected in zip(cfg_calls[:2], CONFIGURATION.values(), strict=True):
        assert {k: _bits(v) for k, v in got.items()} == {k: _bits(v) for k, v in expected.items()}
    lines = dual_lines(logs)
    assert (
        "dual EP_ATM_TIMELOOP_BEFORE pass 1 NEW nudge_max_coeff vs old: identical"
        f" (float 0.375 (0x1.8000000000000p-2, bits {'0' * 12}d83f))" in lines
    )
    assert "dual EP_ATM_TIMELOOP_BEFORE pass 1 NEW loutshs vs old: identical (bool True)" in lines
    assert lines[-1] == (
        "dual EP_ATM_TIMELOOP_BEFORE pass 1: 7 checked (NEW 7, OBSERVE 0), 7 identical,"
        " 0 differ (NEW 0, OBSERVE 0); selftest 0"
    )


def test_configuration_difference_stops_in_strict_mode(cfg_calls, wrappers):
    comin = FakeComIn()
    icon = cfg_icon(comin, nudge_max_coeff=0.075)  # e.g. the namelist value, not ICON's
    cfg_plugin(comin)
    icon.secondary_constructor()
    with pytest.raises(_dual.DualCheckError, match="new route of 'nudge_max_coeff' differs"):
        icon.new_pass()
    assert [c[0] for c in cfg_calls] == ["grid_init"]  # diffusion_init did not run


def test_configuration_difference_in_report_mode(cfg_calls, wrappers, logs):
    comin = FakeComIn()
    icon = cfg_icon(comin, backend=1, loutshs=False)
    cfg_plugin(comin, **{_dual.MODE_ENV: "report"})
    icon.secondary_constructor()
    icon.new_pass()
    # the granule gets the NEW route, ICON's namelist values
    assert cfg_calls[0][1]["backend"] == 0 and cfg_calls[1][1]["loutshs"] is True
    differ = [m for m in dual_lines(logs) if ": differ" in m]
    assert [m.split(" vs ")[0].rsplit(" ", 1)[1] for m in differ] == [
        "backend",
        "loutshs",
        "backend",
    ]
    assert "(bool True vs bool False)" in differ[1]


def test_configuration_selftest(cfg_calls, wrappers, logs):
    comin = FakeComIn()
    icon = cfg_icon(comin)
    cfg_plugin(comin, **{_dual.MODE_ENV: "report", _dual.SELFTEST_ENV: "all"})
    icon.secondary_constructor()
    icon.new_pass()
    lines = dual_lines(logs)
    assert sum("[selftest]: differ" in m for m in lines) == 7
    assert lines[-1].endswith("0 identical, 7 differ (NEW 7, OBSERVE 0); selftest 7")
    assert [c[1] for c in cfg_calls] == [
        CONFIGURATION["grid_init"],
        CONFIGURATION["diffusion_init"],
    ]


def test_configuration_problems_stop_the_registration(icon_run_dir):
    text = (icon_run_dir / _config.NAMELIST_FILE).read_text()
    (icon_run_dir / _config.NAMELIST_FILE).write_text(text.replace(" NDYN_SUBSTEPS =", " NDYN_X ="))
    with pytest.raises(RuntimeError, match=r"1 problem\(s\) with ICON's namelist output") as error:
        cfg_plugin(FakeComIn())
    assert "'ndyn_substeps': NAMELIST_ICON_output_atm has no nonhydrostatic_nml" in str(error.value)
    set_mode(icon_run_dir, 0)
    cfg_plugin(FakeComIn())  # idle: the configuration is not read


def test_missing_namelist_output_stops_the_registration(icon_run_dir):
    (icon_run_dir / _config.NAMELIST_FILE).unlink()
    with pytest.raises(RuntimeError, match="cannot read ICON's namelist output"):
        cfg_plugin(FakeComIn())


@pytest.mark.parametrize(
    "mode, interface, environ, timelevel, source",
    [
        (1, 1, {}, "strict", "the default in SUBSTITUTE"),
        (2, 1, {}, "report", "the default in VERIFY"),
        (1, 1, {"ICON4PY_COMIN_TIMELEVEL_CHECK": "report"}, "report", "set by"),
        (2, 1, {"ICON4PY_COMIN_TIMELEVEL_CHECK": "strict"}, "strict", "set by"),
        (0, 1, {}, "off", "the plugin is idle"),
        (1, 0, {"ICON4PY_COMIN_TIMELEVEL_CHECK": "strict"}, "off", "the plugin is idle"),
    ],
)
def test_startup_line_and_time_level_check(  # noqa: PLR0917 [too-many-positional-arguments]
    logs, icon_run_dir, mode, interface, environ, timelevel, source
):
    set_mode(icon_run_dir, mode, interface)
    instance = cfg_plugin(FakeComIn(), **environ)
    assert logs.messages[1] == instance.mode.describe("EP_ATM_DIFFUSION_ENTER")
    assert logs.messages[1].startswith(
        f"ICON mode {_config.MODE_NAMES[mode]} (icon4py_interface={interface},"
    )
    assert logs.messages[2].startswith(f"time-level check {timelevel}: {source}")
    assert f"; time-level check {timelevel}" in logs.text


def test_mode_off_but_icon_exposes_its_variables(cfg_calls, wrappers, logs, icon_run_dir):
    set_mode(icon_run_dir, 0)
    comin = FakeComIn()
    icon = cfg_icon(comin)
    cfg_plugin(comin)
    with pytest.raises(_dual.DualCheckError, match="exposed yes, expected no"):
        icon.secondary_constructor()
    comin = FakeComIn()
    icon = cfg_icon(comin)
    instance = cfg_plugin(comin, **{_dual.MODE_ENV: "report"})
    icon.secondary_constructor()
    assert not instance.active and comin.contexts == {}
    assert [r.levelname for r in logs.records if "exposed yes, expected no" in r.message] == [
        "WARNING"
    ]


def test_mode_computes_but_icon_exposes_nothing(logs):
    with pytest.raises(_dual.DualCheckError, match="exposed no, expected yes"):
        cfg_plugin(FakeComIn()).secondary_constructor()
    instance = cfg_plugin(FakeComIn(), **{_dual.MODE_ENV: "report"})
    with pytest.raises(RuntimeError, match=r"ICON mode SUBSTITUTE delegates .* exposes no"):
        instance.secondary_constructor()


def _run(comin: FakeComIn, calls: int, **environ: str) -> FakeIcon:
    """Register the plugin, then one pass with 'calls' diffusion calls."""
    icon = cfg_icon(comin)
    cfg_plugin(comin, **environ)
    icon.secondary_constructor()
    icon.new_pass()
    for n in range(calls):
        icon.diffusion_call(dtime=1.0, linit=n == 0)
    return icon


@pytest.mark.parametrize("mode", [1, 2])
def test_enter_at_every_diffusion_call(cfg_calls, wrappers, logs, icon_run_dir, mode):
    set_mode(icon_run_dir, mode)
    comin = FakeComIn()
    _run(comin, 3)
    comin.fire("EP_DESTRUCTOR")
    assert logs.messages[-1] == (
        f"mode check: EP_ATM_DIFFUSION_ENTER callbacks for domain 1: 3, expected 3 (ICON mode"
        f" {_config.MODE_NAMES[mode]}; EP_ATM_DYCORE_DIFFUSION_BEFORE 3, lhdiff_vn T): identical"
    )


def test_enter_missing_at_a_diffusion_call(cfg_calls, wrappers, logs):
    comin = FakeComIn()
    _run(comin, 2)
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)  # a call without ENTER
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 2)  # another domain: not counted
    with pytest.raises(_dual.DualCheckError, match="domain 1: 2, expected 3"):
        comin.fire("EP_DESTRUCTOR")
    assert diffusion_wrapper.granule is None  # released before the check


def test_no_diffusion_call_without_lhdiff_vn(cfg_calls, wrappers, logs, icon_run_dir):
    set_namelist(icon_run_dir, LHDIFF_VN="F")
    comin = FakeComIn()
    icon = cfg_icon(comin, hdiff_vn=False)
    cfg_plugin(comin)
    icon.secondary_constructor()
    icon.new_pass()
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)  # the regular site, outside IF (lhdiff_vn)
    comin.fire("EP_DESTRUCTOR")
    assert logs.messages[-1].endswith(
        "callbacks for domain 1: 0, expected 0 (ICON mode SUBSTITUTE;"
        " EP_ATM_DYCORE_DIFFUSION_BEFORE 1, lhdiff_vn F): identical"
    )


def test_idle_plugin_runs_nothing(cfg_calls, wrappers, logs, icon_run_dir):
    set_mode(icon_run_dir, 1, interface=0)  # py2fgen computes; ICON exposes nothing to ComIn
    comin = FakeComIn()
    instance = cfg_plugin(comin)
    comin.fire("EP_SECONDARY_CONSTRUCTOR")
    comin.fire("EP_ATM_TIMELOOP_BEFORE")
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)
    comin.fire("EP_DESTRUCTOR")
    assert not instance.active and cfg_calls == [] and comin.contexts == {}
    assert (diffusion_wrapper.granule, grid_wrapper.grid_state) == wrappers
    assert "idle: ICON mode SUBSTITUTE (icon4py_interface=0," in logs.text
    assert logs.messages[-1].endswith(
        "domain 1: 0, expected 0 (ICON mode SUBSTITUTE; EP_ATM_DYCORE_DIFFUSION_BEFORE 1,"
        " lhdiff_vn T): identical"
    )
    assert not any(r.levelno >= logging.WARNING for r in logs.records)


def test_mode_check_off(cfg_calls, wrappers, logs):
    comin = FakeComIn()
    icon = _run(comin, 1, **{_dual.MODE_ENV: "off"})
    icon.diffusion_call(dtime=1.0, linit=False, delegate=False)  # a call without ENTER
    comin.fire("EP_DESTRUCTOR")
    assert not any(m.startswith(("mode check", "dual EP_")) for m in logs.messages)
    # the granule still gets the NEW arguments
    assert [c[1] for c in cfg_calls[:2]] == list(CONFIGURATION.values())


# ---- ICON's condition for the initial diffusion call ------------------------------------------


def test_initial_call_of_the_excerpt():
    initial, errors = _config.initial_call(excerpt())
    assert errors == []
    assert initial == _config.InitialCall(
        ldynamics=True, ltestcase=False, lhdiff_vn=True, init_mode=7, lrestartrun=False
    )
    assert initial.describe() == (
        "ldynamics T, ltestcase F, lhdiff_vn T, init_mode 7, lrestartrun F"
    )
    # the first call of the first step only
    assert initial.expected(1, 1)
    assert not initial.expected(1, 2) and not initial.expected(2, 1)


@pytest.mark.parametrize(
    "change",
    [
        dict(ldynamics=False),
        dict(ltestcase=True),
        dict(lhdiff_vn=False),
        dict(init_mode=_config.MODE_IAU),
        dict(lrestartrun=True),
    ],
)
def test_no_initial_call(change):
    initial, _ = _config.initial_call(excerpt())
    assert initial is not None
    assert not dataclasses.replace(initial, **change).expected(1, 1)
    assert dataclasses.replace(initial, init_mode=1).expected(1, 1)  # other init modes: yes


def test_initial_call_problems_stop_the_registration(icon_run_dir):
    text = (icon_run_dir / _config.NAMELIST_FILE).read_text()
    (icon_run_dir / _config.NAMELIST_FILE).write_text(text.replace(" INIT_MODE =", " INIT_X ="))
    with pytest.raises(RuntimeError, match=r"1 problem\(s\)") as error:
        cfg_plugin(FakeComIn())
    assert "initial call: 'init_mode': NAMELIST_ICON_output_atm has no initicon_nml" in str(
        error.value
    )
