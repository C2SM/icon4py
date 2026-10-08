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

import collections
import dataclasses
import logging
import re
import sys
import types
import typing

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import diffusion_wrapper, grid_wrapper
from icon4py.bindings.comin import _arguments, _config, _granule, plugin
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common import dimension as dims
from icon4py.model.common.config import options
from icon4py.model.common.grid import vertical

from .test_comin_plugin import (  # fixtures: icon_run_dir, logs, restore_package_logger, wrappers
    AFTER,
    BEFORE,
    INIT,
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


def excerpt_text() -> str:
    return NAMELIST_EXCERPT.read_text()


def excerpt() -> _config.Namelists:
    return _config.Namelists.from_text(excerpt_text())


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
        (read,) = _config.parse(f" &A_NML\n X = {text}\n /\n").groups["a_nml"][0]["x"]
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
    groups = _config.parse(text).groups
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
    assert _config.parse(text).groups == {"io_nml": [{"n": (1,), "x": (2,)}]}
    parsed = _config.parse(" &IO_NML\n N = 1,\n X = 2,\n N = 2\n /\n &B_NML\n Y = 1\n /\n")
    assert parsed.groups == {"b_nml": [{"y": (1,)}]}  # the next group is read
    assert re.search(r"line 5: entry 'n' twice .* other values", parsed.problems["io_nml"])


B_NML: typing.Final = " &B_NML\n Y = 2\n /\n"
"""A readable group after the one with a problem: the reader resumes there."""


@pytest.mark.parametrize(
    "text, group, message",
    [
        (" &A_NML\n X = 1,\n", "a_nml", "line 3: '&a_nml' is not closed"),
        (" X = 1\n" + B_NML, _config.OUTSIDE, "line 1: entry 'X' outside a namelist group"),
        (" /\n" + B_NML, _config.OUTSIDE, "line 1: '/' outside a namelist group"),
        (" &A_NML\n &B_NML\n Y = 2\n /\n", "a_nml", "line 2: '&B_NML' inside '&a_nml'"),
        (" &A_NML\n 5,\n /\n" + B_NML, "a_nml", "line 2: value 5 outside a namelist entry"),
        (" &1\n" + B_NML, _config.OUTSIDE, "line 1: cannot read '&1"),
    ],
)
def test_parse_problems_skip_the_group(text, group, message):
    """A group the reader cannot read is skipped, up to the next group, and kept as a problem."""
    parsed = _config.parse(text)
    assert list(parsed.problems) == [group]
    assert parsed.problems[group].startswith(f"{_config.NAMELIST_FILE}, {message}")
    assert parsed.groups == ({"b_nml": [{"y": (2,)}]} if "B_NML" in text else {})


def test_unreadable_values_raise_only_when_used():
    text = (
        " &A_NML\n X = (1.0, 2.0), Y = 1.2.3, Z = 3\n /\n"
        " &C_NML\n X = 1,\n &END\n"  # some compilers close a group with '&END'
        " &D_NML\n TINY = 0.1000000000000000-100, HUGE = -0.25+101\n /\n"  # E format, 3 digits
        " &E_NML\n /\n &E_NML\n K = 1,\n"  # an unclosed second instance
    )
    nml = _config.Namelists.from_text(text)
    assert nml.get("a_nml", "z") == (3,)
    for key, shown in (("x", "\\(1.0, 2.0\\)"), ("y", "1.2.3")):
        with pytest.raises(_config.NamelistError, match=rf"a_nml: {key} = {shown}: cannot read"):
            nml.get("a_nml", key)
    assert nml.get("c_nml", "x") == (1,)
    assert nml.get("d_nml", "tiny") == (1e-101,) and nml.get("d_nml", "huge") == (-2.5e100,)
    with pytest.raises(_config.NamelistError, match=r"'&e_nml' could not be read \(.*line 14"):
        nml.get("e_nml", "k")
    assert nml.problems() == [
        f"&e_nml: {_config.NAMELIST_FILE}, line 14: '&e_nml' is not closed with '/'",
        "&a_nml: x = (1.0, 2.0)",
        "&a_nml: y = 1.2.3",
    ]


@pytest.mark.parametrize(
    "text, inexact",
    [
        ("150.0000000000000", False),  # nvfortran's F format: 16 significant digits
        ("0.6500000000000000", False),  # the nearest double to 0.65: exact
        ("2.5000000000000001E-002", False),  # E format: 17 digits, always exact
        ("7.1372509002685547E-002", False),
        ("60686.25390625000", False),  # a short dyadic value
        ("1.234567890123457", True),  # 16 digits, and no shorter decimal: may be 1 ulp off
        ("0.3000000000000000", False),  # e.g. 0.1 + 0.2 = 0.30000000000000004 (undetectable)
        ("0.000000000000000", False),
        ("12", False),
    ],
)
def test_reals_that_may_be_inexact(text, inexact):
    nml = _config.Namelists.from_text(f" &A_NML\n X = {text}\n /\n")
    assert nml.maybe_inexact("a_nml", "x") is inexact


def test_the_gap_of_the_precision_check():
    """A value 1 ulp off a short decimal reads back as that decimal: not detectable."""
    text = f"{0.1 + 0.2:.16f}"  # nvfortran's F format of 0.30000000000000004
    assert float(text) != 0.1 + 0.2 and not _config.maybe_inexact(text, float(text))
    assert _config.significant_digits("-0.30000000000000004") == 17


def test_fortran_dict_feeds_icon4py_readers():
    """The layout of f90nml's dictionaries: icon4py's own readers take it."""
    config = excerpt().fortran_dict()
    assert config["run_nml"]["icon4py_mode"] == 1 and config["run_nml"]["num_lev"][0] == 80
    vertical_config = vertical.VerticalGridConfig.from_fortran_dict(config)
    assert vertical_config.num_levels == 80
    diffusion_config = diffusion.DiffusionConfig.from_fortran_dict(config)
    assert diffusion_config.hdiff_efdt_ratio == 24.0


# ---- the configuration ------------------------------------------------------------------------

EXPECTED_GRID = {
    "lowest_layer_thickness": 20.0,
    "model_top_height": 22000.0,
    "stretch_factor": 0.65,
    "flat_height": 16000.0,
    "rayleigh_damping_height": 12250.0,
    "backend": 0,
}
"""ICON's values of the configuration of 'grid_init' in the run of the excerpt."""
PY2FGEN_DIFFUSION = {
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
}
"""The 25 configuration values that ICON passed to py2fgen's 'diffusion_init' in the run of the
excerpt (the py2fgen probe's reference of them)."""


def _bits(value: typing.Any) -> tuple[type, str]:
    return type(value), value.hex() if isinstance(value, float) else repr(value)


def test_the_configuration_of_the_excerpt():
    nml = excerpt()
    kinds = {p: _granule.GRID_INIT[p] for p in _config.ARGUMENTS["grid_init"]}
    values, errors = _config.arguments(nml, "grid_init", kinds)
    assert errors == []
    assert {k: _bits(v) for k, v in values.items()} == {
        k: _bits(v) for k, v in EXPECTED_GRID.items()
    }
    kinds = {p: _granule.DIFFUSION_INIT[p] for p in _config.ARGUMENTS["diffusion_init"]}
    values, errors = _config.arguments(nml, "diffusion_init", kinds)
    assert errors == []
    expected = {k: PY2FGEN_DIFFUSION[k] for k in ("ndyn_substeps", "nudge_max_coeff")}
    assert {k: _bits(v) for k, v in values.items()} == {
        k: _bits(v) for k, v in {**expected, "backend": 0}.items()
    }
    config, errors = _granule.diffusion_config(nml)
    assert errors == [] and isinstance(config, diffusion.DiffusionConfig)
    assert config.loutshs is True


def test_the_diffusion_config_is_py2fgens():
    """icon4py's DiffusionConfig from ICON's namelist output = the one py2fgen's wrapper builds
    from 23 of the 25 values ICON passes, field by field and bit by bit (also the fields the
    wrapper leaves at their defaults: ICON's values are the defaults in this run)."""
    ours, errors = _granule.diffusion_config(excerpt())
    assert errors == []
    theirs = diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType(PY2FGEN_DIFFUSION["diffusion_type"]),
        apply_to_vertical_wind=PY2FGEN_DIFFUSION["hdiff_w"],
        apply_to_horizontal_wind=PY2FGEN_DIFFUSION["hdiff_vn"],
        apply_smag_diff_to_vertical_wind=PY2FGEN_DIFFUSION["hdiff_smag_w"],
        apply_zdiffusion_t=PY2FGEN_DIFFUSION["zdiffu_t"],
        type_t_diffu=diffusion.TemperatureDiscretizationType(PY2FGEN_DIFFUSION["type_t_diffu"]),
        type_vn_diffu=diffusion.SmagorinskyStencilType(PY2FGEN_DIFFUSION["type_vn_diffu"]),
        hdiff_efdt_ratio=PY2FGEN_DIFFUSION["hdiff_efdt_ratio"],
        hdiff_w_efdt_ratio=PY2FGEN_DIFFUSION["hdiff_w_efdt_ratio"],
        **{
            f"smagorinski_scaling_{k}{i}": PY2FGEN_DIFFUSION[f"smagorinski_scaling_{k}{i}"]
            for k in ("factor", "height")
            for i in ("", "2", "3", "4")
        },
        apply_to_temperature=PY2FGEN_DIFFUSION["hdiff_temp"],
        velocity_boundary_diffusion_denominator=PY2FGEN_DIFFUSION["denom_diffu_v"],
        shear_type=diffusion.TurbulenceShearForcingType(PY2FGEN_DIFFUSION["itype_sher"]),
        iforcing=diffusion.ForcingType(PY2FGEN_DIFFUSION["iforcing"]),
        a_hshr=PY2FGEN_DIFFUSION["a_hshr"],
        loutshs=PY2FGEN_DIFFUSION["loutshs"],
    )  # the constructor call of py2fgen's 'diffusion_wrapper.diffusion_init'
    fields = [f.name for f in dataclasses.fields(diffusion.DiffusionConfig)]
    assert len(fields) == 25
    assert {f: _bits(getattr(ours, f)) for f in fields} == {
        f: _bits(getattr(theirs, f)) for f in fields
    }


def test_the_diffusion_entries_are_icon4pys():
    """The plugin checks exactly the entries 'DiffusionConfig.from_fortran_dict' reads, and
    gives ICON's value of the other field that the wrapper sets."""
    icon_options = dict(options.ConfigOption.iter_from_config_class(diffusion.DiffusionConfig))
    read = {
        name: (o.icon_equivalent.path, o.icon_equivalent.name)
        for name, o in icon_options.items()
        if isinstance(o.icon_equivalent, options.IconOption) and o.icon_equivalent.read_from_icon
    }
    entries = _granule.diffusion_entries()
    assert {e.field: ((e.group,), e.name) for e in entries} == read
    assert len(entries) == 24
    assert {e.field for e in entries if e.domain_value} == {
        "apply_smag_diff_to_vertical_wind",
        "compute_3d_smag_coeff",
    }
    kinds = collections.Counter(e.kind.__name__ for e in entries)
    assert kinds == {"bool": 6, "int": 5, "float": 13}
    assert set(icon_options) - set(read) == {_config.LOUTSHS_FIELD} == {"loutshs"}
    # the vertical ones: the entries of 'VerticalGridConfig.from_fortran_dict'
    vertical_config = vertical.VerticalGridConfig.from_fortran_dict(excerpt().fortran_dict())
    kinds = {p: _granule.GRID_INIT[p] for p in _config.ARGUMENTS["grid_init"]}
    grid, _ = _config.arguments(excerpt(), "grid_init", kinds)
    for argument in grid.keys() - {"backend"}:
        assert _bits(getattr(vertical_config, argument)) == _bits(grid[argument]), argument


def test_effective_values_where_icon_differs_from_the_namelist():
    interpol = " &INTERPOL_NML\n NUDGE_MAX_COEFF = 2.0000000000000000E-002\n /\n"
    nudge = _config.ARGUMENTS["diffusion_init"]["nudge_max_coeff"]
    assert _config.argument(_config.Namelists.from_text(interpol), nudge, float) == 5.0 * 0.02
    damp = " &NONHYDROSTATIC_NML\n DAMP_HEIGHT = 1000.0, 2000.0\n /\n"
    got, errors = _config.arguments(
        _config.Namelists.from_text(damp), "grid_init", {"rayleigh_damping_height": float}
    )
    assert got == {"rayleigh_damping_height": 1000.0} and errors == []  # domain 1


def configured_loutshs(text: str) -> _config.Loutshs:
    loutshs, errors = _config.loutshs(_config.Namelists.from_text(text))
    assert errors == [] and loutshs is not None
    return loutshs


def test_configured_loutshs_is_icons_default():
    """No namelist entry: ICON's default true, false in single-column runs without dynamics."""
    run = " &RUN_NML\n LDYNAMICS = {}\n IFORCING = {}\n /\n"
    grid = " &GRID_NML\n L_SCM_MODE = {}\n /\n"
    with_dynamics = configured_loutshs(run.format("T", 3))
    assert with_dynamics == _config.Loutshs(iforcing=3, ldynamics=True, l_scm_mode=None)
    assert with_dynamics.configured is True  # grid_nml not needed with dynamics
    assert configured_loutshs(run.format("F", 3) + grid.format("F")).configured is True
    assert configured_loutshs(run.format("F", 0) + grid.format("T")).configured is False
    # an entry LOUTSHS is ignored: ICON has none (and resets the value with NWP physics)
    turbdiff = " &TURBDIFF_NML\n LOUTSHS = F\n /\n"
    assert configured_loutshs(run.format("T", 0) + turbdiff).configured is True
    loutshs, errors = _config.loutshs(_config.Namelists.from_text(run.format("F", 3)))
    assert loutshs is None
    assert errors == [
        "DiffusionConfig.loutshs: 'l_scm_mode': NAMELIST_ICON_output_atm has no grid_nml:"
        " l_scm_mode."
    ]


def test_runtime_loutshs_follows_icon():
    """With NWP physics ICON's value is whether ICON has the variable ddt_tke_hsh on the
    domain (the output lists it); without, the configured value."""
    with_hsh = {("ddt_tke_hsh", 1), ("vn", 1)}
    without = {("vn", 1), ("ddt_tke_hsh", 2)}
    value, reason = _config.runtime_loutshs(3, True, with_hsh, 1)
    assert value is True
    assert reason == (
        "loutshs T: with NWP physics ICON sets it to whether the output of domain 1 lists"
        " ddt_tke_hsh; it lists it (ICON has the variable ddt_tke_hsh on domain 1); configured T"
    )
    value, reason = _config.runtime_loutshs(3, True, without, 1)
    assert value is False
    assert reason == (
        "loutshs F: with NWP physics ICON sets it to whether the output of domain 1 lists"
        " ddt_tke_hsh; it does not list it (ICON has no variable ddt_tke_hsh on domain 1);"
        " configured T"
    )
    assert _config.runtime_loutshs(3, False, with_hsh, 1)[0] is True  # NWP resets it
    for iforcing in (0, 2):
        for configured in (True, False):
            value, reason = _config.runtime_loutshs(iforcing, configured, with_hsh, 1)
            assert value is configured
            assert reason == (
                f"loutshs {'T' if configured else 'F'}: ICON's configured value (iforcing"
                f" {iforcing}, no NWP physics)"
            )


def test_configuration_problems_are_collected():
    text = excerpt_text()
    for old, new in (
        (" HDIFF_ORDER =            5,", " HDIFF_ORDER = 5.0,"),
        (" LHDIFF_W =  T,", " LHDIFF_W = 1,"),
        (" LHDIFF_VN =  T,", " LHDIFF_VN = T, F"),
        (" NUDGE_MAX_COEFF =   7.4999999999999997E-002,", " NUDGE_MAX_COEFF = 2,"),
        (" NDYN_SUBSTEPS =", " NDYN_X ="),
    ):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    config, errors = _granule.diffusion_config(_config.Namelists.from_text(text))
    assert config is None
    assert errors == [
        "diffusion_nml: hdiff_order (DiffusionConfig.diffusion_type) = 5.0, expected a int.",
        "diffusion_nml: lhdiff_w (DiffusionConfig.apply_to_vertical_wind) = 1, expected a bool.",
        "diffusion_nml: lhdiff_vn (DiffusionConfig.apply_to_horizontal_wind): 2 values, expected"
        " one.",
    ]
    found, errors = _config.arguments(
        _config.Namelists.from_text(text),
        "diffusion_init",
        {"ndyn_substeps": int, "nudge_max_coeff": float},
    )
    assert found == {} and errors == [
        "diffusion_init: 'ndyn_substeps': NAMELIST_ICON_output_atm has no nonhydrostatic_nml:"
        " ndyn_substeps.",
        "diffusion_init: 'nudge_max_coeff': interpol_nml: nudge_max_coeff = 2, expected a float.",
    ]
    found, errors = _config.arguments(
        _config.Namelists.from_text(text), "grid_init", {"no_such_argument": int}
    )
    assert found == {} and errors == ["grid_init: 'no_such_argument' has no namelist entry."]
    # an option icon4py does not implement: from_fortran_dict's validation
    text = excerpt_text().replace(" LSMAG_3D =  F,", " LSMAG_3D =  T,")
    config, errors = _granule.diffusion_config(_config.Namelists.from_text(text))
    assert config is None and len(errors) == 1
    assert errors[0].startswith("DiffusionConfig.from_fortran_dict: NotImplementedError('3D")


# ---- ICON's mode -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interface, luse, mode, name, computes, says",
    [
        (1, True, 1, "SUBSTITUTE", True, "this plugin computes the horizontal diffusion"),
        (
            1,
            True,
            2,
            "VERIFY",
            True,
            "ICON computes the horizontal diffusion and continues with its own result; this"
            " plugin computes it too, at EP_X on its own copies of ICON's input from EP_Y, and"
            " compares its results with ICON's there; ICON compares nothing",
        ),
        (1, True, 0, "OFF", False, "ICON computes the horizontal diffusion; this plugin"),
        (0, True, 1, "SUBSTITUTE", False, "through py2fgen, not ComIn; this plugin stays"),
        (0, True, 0, "OFF", False, "ICON computes the horizontal diffusion; this plugin"),
        # ICON stops at its namelist check with this, but the dispatcher's rule is clear
        (1, False, 1, "OFF", False, "ICON computes the horizontal diffusion; this plugin"),
        (None, None, None, "OFF", False, "ICON has no icon4py switches"),
    ],
)
def test_icon_mode(  # noqa: PLR0917 [too-many-positional-arguments]
    interface, luse, mode, name, computes, says
):
    icon_mode = _config.IconMode(interface, luse, mode)
    assert (icon_mode.name, icon_mode.computes) == (name, computes)
    line = icon_mode.describe("EP_X", "EP_Y")
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


# ---- the plugin: ICON's mode and the configuration ---------------------------------------------

CALLS: list[tuple[str, dict]] = []


class CfgGranule:
    """A granule that records what it gets."""

    def grid_init(self, **kwargs: typing.Any) -> None:
        CALLS.append(("grid_init", kwargs))

    def diffusion_init(self, **kwargs: typing.Any) -> None:
        CALLS.append(("diffusion_init", kwargs))

    def diffusion_run(self, **kwargs: typing.Any) -> None:
        CALLS.append(("diffusion_run", {k: kwargs[k] for k in ("dtime", "linit")}))
        kwargs["w"].ndarray[...] += kwargs["dtime"]

    def release(self) -> None:
        pass


CFG_FUNCTIONS = {
    "grid_init": plugin.FunctionEntry(
        {"rayleigh_damping_height": float, "backend": int}, INIT, inout=False
    ),
    "diffusion_init": plugin.FunctionEntry(
        {
            "config": diffusion.DiffusionConfig,
            "ndyn_substeps": int,
            "nudge_max_coeff": float,
            "backend": int,
        },
        INIT,
        inout=False,
    ),
    "diffusion_run": plugin.FunctionEntry(
        {
            "w": _arguments.ArrayParam(
                "w", np.float64, 2, (dims.CellDim, dims.KDim), location="cell"
            ),
            "dtime": float,
            "linit": bool,
        },
        BEFORE,
        inout=True,
    ),
}
NC, NLEV = 4, 3


@pytest.fixture
def cfg_calls():
    CALLS.clear()
    yield CALLS
    CALLS.clear()


def cfg_icon(comin: FakeComIn, ddt_tke_hsh: bool = True) -> FakeIcon:
    """ICON's side of the toy functions: its variable 'w', and 'ddt_tke_hsh' as in the run of the
    excerpt, whose output lists it (ICON's loutshs is then true)."""
    icon = FakeIcon(comin)
    icon.add("w", fortran_buffer((NC, NLEV), np.float64, 1.0))
    if ddt_tke_hsh:
        icon.add("ddt_tke_hsh", fortran_buffer((NC, NLEV + 1), np.float64, 0.0))
    return icon


def cfg_plugin(comin: FakeComIn, **environ: str) -> plugin.Plugin:
    instance = plugin.Plugin(comin, functions=CFG_FUNCTIONS, environ=environ, granule=CfgGranule())
    instance.register()
    return instance


def test_configuration_reaches_the_granule(cfg_calls, wrappers, logs):
    comin = FakeComIn()
    icon = cfg_icon(comin)
    cfg_plugin(comin)
    (line,) = [m for m in logs.messages if m.startswith("diffusion_init configuration from")]
    assert line.startswith(
        "diffusion_init configuration from NAMELIST_ICON_output_atm: config"
        " DiffusionConfig(diffusion_type=5, apply_to_vertical_wind=True,"
    )
    assert line.endswith(
        "shear_type=2, iforcing=3, a_hshr=2.0, loutshs=True), ndyn_substeps 5,"
        " nudge_max_coeff 0.375, backend 0"
    )
    assert (
        "grid_init configuration from NAMELIST_ICON_output_atm: rayleigh_damping_height 12250.0,"
        " backend 0" in logs.messages
    )
    icon.secondary_constructor()
    assert (
        "diffusion_init configuration at run time: loutshs T: with NWP physics ICON sets it to"
        " whether the output of domain 1 lists ddt_tke_hsh; it lists it (ICON has the variable"
        " ddt_tke_hsh on domain 1); configured T" in logs.messages
    )
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    assert [c[0] for c in cfg_calls] == ["grid_init", "diffusion_init", "diffusion_run"]
    grid, init = cfg_calls[0][1], cfg_calls[1][1]
    assert {k: _bits(v) for k, v in grid.items()} == {
        k: _bits(EXPECTED_GRID[k]) for k in ("rayleigh_damping_height", "backend")
    }
    assert init["config"] == _granule.diffusion_config(excerpt())[0] and init["backend"] == 0
    assert (_bits(init["ndyn_substeps"]), _bits(init["nudge_max_coeff"])) == (
        _bits(PY2FGEN_DIFFUSION["ndyn_substeps"]),
        _bits(PY2FGEN_DIFFUSION["nudge_max_coeff"]),
    )


def test_loutshs_at_run_time_is_icons(cfg_calls, wrappers, logs):
    """With NWP physics ICON passes loutshs = whether the output lists ddt_tke_hsh, i.e. whether
    ICON has that variable: without it the granule gets false, whatever the configured value."""
    comin = FakeComIn()
    icon = cfg_icon(comin, ddt_tke_hsh=False)
    cfg_plugin(comin)
    (line,) = [m for m in logs.messages if m.startswith("diffusion_init configuration from")]
    assert "loutshs=True)" in line  # the configured value
    icon.secondary_constructor()
    assert (
        "diffusion_init configuration at run time: loutshs F: with NWP physics ICON sets it to"
        " whether the output of domain 1 lists ddt_tke_hsh; it does not list it (ICON has no"
        " variable ddt_tke_hsh on domain 1); configured T" in logs.messages
    )
    icon.new_pass()
    icon.diffusion_call(dtime=10.0, linit=True)
    config = cfg_calls[1][1]["config"]
    assert config.loutshs is False
    assert config == dataclasses.replace(_granule.diffusion_config(excerpt())[0], loutshs=False)


def test_configuration_problems_stop_the_registration(icon_run_dir):
    text = (icon_run_dir / _config.NAMELIST_FILE).read_text()
    (icon_run_dir / _config.NAMELIST_FILE).write_text(text.replace(" NDYN_SUBSTEPS =", " NDYN_X ="))
    with pytest.raises(RuntimeError, match=r"1 problem\(s\) with ICON's namelist output") as error:
        cfg_plugin(FakeComIn())
    assert (
        "diffusion_init: 'ndyn_substeps': NAMELIST_ICON_output_atm has no nonhydrostatic_nml:"
        " ndyn_substeps." in str(error.value)
    )
    set_mode(icon_run_dir, 0)
    cfg_plugin(FakeComIn())  # idle: the configuration is not read


def test_unread_groups_are_logged_and_ignored(cfg_calls, wrappers, logs, icon_run_dir):
    """A group or value the plugin does not need may be unreadable; one it needs may not."""
    path = icon_run_dir / _config.NAMELIST_FILE
    text = path.read_text()
    extra = " &OTHER_NML\n X = (1.0, 2.0)\n /\n &BROKEN_NML\n 5,\n /\n"
    path.write_text(extra + text.replace(" LHDIFF_W =", " UNUSED = abc, LHDIFF_W ="))
    cfg_plugin(FakeComIn())
    (line,) = [m for m in logs.messages if "could not read" in m]
    assert line == (
        "NAMELIST_ICON_output_atm: the plugin does not need what it could not read, ignored:"
        " &broken_nml: NAMELIST_ICON_output_atm, line 5: value 5 outside a namelist entry;"
        " &other_nml: x = (1.0, 2.0); &diffusion_nml: unused = abc"
    )
    # a value the plugin needs
    path.write_text(text.replace(" NDYN_SUBSTEPS =", " NDYN_SUBSTEPS = abc, X ="))
    with pytest.raises(RuntimeError, match=r"1 problem\(s\)") as error:
        cfg_plugin(FakeComIn())
    assert (
        "diffusion_init: 'ndyn_substeps': nonhydrostatic_nml: ndyn_substeps = abc: cannot read"
        " the value" in str(error.value)
    )
    # a group the plugin needs
    path.write_text(text.replace(" &NONHYDROSTATIC_NML", " &NONHYDROSTATIC_NML\n 7,"))
    with pytest.raises(RuntimeError, match=r"3 problem\(s\)") as error:
        cfg_plugin(FakeComIn())  # rayleigh_damping_height, l_zdiffu_t and ndyn_substeps
    assert "'&nonhydrostatic_nml' could not be read" in str(error.value)


def test_configuration_precision_is_logged(cfg_calls, wrappers, logs, icon_run_dir):
    cfg_plugin(FakeComIn())
    (line,) = [m for m in logs.messages if m.startswith("configuration: ")]
    assert line.startswith("configuration: 15 real values, each written with 17 significant")
    path = icon_run_dir / _config.NAMELIST_FILE
    path.write_text(
        path.read_text().replace(
            "NUDGE_MAX_COEFF =   7.4999999999999997E-002", "NUDGE_MAX_COEFF = 0.7499999999999997"
        )
    )
    cfg_plugin(FakeComIn())
    (warning,) = [r for r in logs.records if r.levelno == logging.WARNING]
    assert warning.message == (
        "configuration: 1 of 15 real values have 16 significant digits in"
        " NAMELIST_ICON_output_atm and no shorter decimal, so they may differ from ICON's values"
        " in the last bit: diffusion_init.nudge_max_coeff (interpol_nml: nudge_max_coeff)"
    )


def test_missing_namelist_output_stops_the_registration(icon_run_dir):
    (icon_run_dir / _config.NAMELIST_FILE).unlink()
    with pytest.raises(RuntimeError, match="cannot read ICON's namelist output"):
        cfg_plugin(FakeComIn())


@pytest.mark.parametrize(
    "mode, interface, where, copied",
    [(1, 1, BEFORE, None), (2, 1, AFTER, BEFORE), (0, 1, "-", None), (1, 0, "-", None)],
)
def test_startup_line(logs, icon_run_dir, mode, interface, where, copied):  # noqa: PLR0917
    set_mode(icon_run_dir, mode, interface)
    instance = cfg_plugin(FakeComIn())
    assert logs.messages[1] == instance.mode.describe(where, copied)
    assert logs.messages[1].startswith(
        f"ICON mode {_config.MODE_NAMES[mode]} (icon4py_interface={interface},"
    )


def test_idle_plugin_runs_nothing(cfg_calls, wrappers, logs, icon_run_dir):
    set_mode(icon_run_dir, 1, interface=0)  # py2fgen computes
    comin = FakeComIn()
    instance = cfg_plugin(comin)
    comin.fire("EP_SECONDARY_CONSTRUCTOR")
    comin.fire("EP_ATM_TIMELOOP_BEFORE")
    comin.fire("EP_ATM_INTEGRATE_START", 1)
    comin.fire("EP_ATM_DYCORE_DIFFUSION_BEFORE", 1)
    comin.fire("EP_ATM_DYCORE_DIFFUSION_AFTER", 1)
    comin.fire("EP_DESTRUCTOR")
    assert not instance.active and cfg_calls == [] and comin.contexts == {}
    assert (diffusion_wrapper.granule, grid_wrapper.grid_state) == wrappers
    assert "idle: ICON mode SUBSTITUTE (icon4py_interface=0," in logs.text
    assert not any(r.levelno >= logging.WARNING for r in logs.records)


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
