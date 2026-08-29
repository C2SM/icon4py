# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the py2fgen wrapper of the NWP 1D turbulence granule.

What is tested here is the BOUNDARY, not the scheme. 'Turbulence.run' is validated against
serialized ICON output by the turbulence package's own datatests; what no test of that package
can see is whether the flat argument list of 'turbulence_run' lands in the right slot of the
right state container, and whether the Fortran interface py2fgen renders from it is the one the
ICON side will be written against.

So the granule is mocked and every field is given a value of its own: a mis-wiring that swapped
two arguments of the same shape would still produce a plausible run, and only distinct values
catch it. No serialized data is needed and no stencil is compiled, which keeps the module fast
enough to run on every change to the wrapper.

The codegen test at the end is the counterpart of 'test_codegen_references.py'. That module
snapshots the whole rendered library and reports any difference; this one states what the
turbulence half of the snapshot has to contain, so that a change to the wrapper's signature
fails with the argument that moved rather than with a diff of a hundred and fifty thousand
lines.
"""

import dataclasses
import math
from unittest import mock

import cffi
import numpy as np
import pytest
from click.testing import CliRunner

from icon4py.bindings import (
    all_bindings,
    common as wrapper_common,
    grid_wrapper,
    turbulence_wrapper,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple as simple_grid, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.tools.py2fgen import test_utils


#: 'exp.mch_icon-ch2_small's 'turbdiff_nml', the namelist the granule is validated against. The
#: same values as 'test_turbdiff_granule.CONFIG'; repeated rather than imported because the
#: turbulence package's test tree is not importable from here.
MCH_NAMELIST = {
    "tkhmin": 0.5,
    "tkmmin": 0.75,
    "pat_len": 750.0,
    "tur_len": 300.0,
    "rat_sea": 0.8,
    "ltkesso": True,
    "frcsmot": 0.0,
    "imode_frcsmot": 2,
    "itype_sher": 2,
    "ltkeshs": True,
    "a_hshr": 2.0,
    "icldm_turb": 2,
    "q_crit": 2.0,
    "imode_tkesso": 2,
    "rlam_heat": 10.0,
    "alpha1": 0.125,
    "imode_charpar": 3,
}

#: Every field of 'turbulence_run', with the number of vertical levels it carries: 'half' for
#: the 'nlev + 1' half levels, 'main' for the 'nlev' main levels and 'surface' for one value per
#: column. Taken from the declarations of 'turbulence_states.py', which are the Fortran's.
RUN_FIELDS: dict[str, str] = {
    "u": "main",
    "v": "main",
    "t": "main",
    "qv": "main",
    "qc": "main",
    "prs": "main",
    "rhoh": "main",
    "epr": "main",
    "ut_sso": "main",
    "vt_sso": "main",
    "hdef2": "half",
    "hdiv": "half",
    "dwdx": "half",
    "dwdy": "half",
    "t_g": "surface",
    "qv_s": "surface",
    "ps": "surface",
    "l_pat": "surface",
    "gz0": "surface",
    "tvm": "surface",
    "tvh": "surface",
    "tfm": "surface",
    "tfh": "surface",
    "tkred_sfc": "surface",
    "tkred_sfc_h": "surface",
    "shfl_s": "surface",
    "qvfl_s": "surface",
    "tke": "half",
    "tkvm": "half",
    "tkvh": "half",
    "rcld": "half",
    "rhon": "half",
    "tketens": "half",
    "tket_hshr": "half",
    "u_tens": "main",
    "v_tens": "main",
    "t_tens": "main",
    "qv_tens": "main",
    "qc_tens": "main",
}

#: Where each field of 'turbulence_run' has to arrive: container attribute, keyed by argument
#: name. 'tke' is absent because ICON's one array becomes two granule fields; it has its own
#: test.
RUN_DESTINATIONS: dict[str, tuple[str, str]] = {
    "u": ("input_state", "u"),
    "v": ("input_state", "v"),
    "t": ("input_state", "t"),
    "qv": ("input_state", "qv"),
    "qc": ("input_state", "qc"),
    "prs": ("input_state", "prs"),
    "rhoh": ("input_state", "rhoh"),
    "epr": ("input_state", "epr"),
    "ut_sso": ("input_state", "ut_sso"),
    "vt_sso": ("input_state", "vt_sso"),
    "hdef2": ("input_state", "hdef2"),
    "hdiv": ("input_state", "hdiv"),
    "dwdx": ("input_state", "dwdx"),
    "dwdy": ("input_state", "dwdy"),
    "t_g": ("surface_state", "t_g"),
    "qv_s": ("surface_state", "qv_s"),
    "ps": ("surface_state", "ps"),
    "l_pat": ("surface_state", "l_pat"),
    "gz0": ("diagnostic_state", "gz0"),
    "tvm": ("diagnostic_state", "tvm"),
    "tvh": ("diagnostic_state", "tvh"),
    "tfm": ("diagnostic_state", "tfm"),
    "tfh": ("diagnostic_state", "tfh"),
    "tkred_sfc": ("diagnostic_state", "tkred_sfc"),
    "tkred_sfc_h": ("diagnostic_state", "tkred_sfc_h"),
    "shfl_s": ("diagnostic_state", "shfl_s"),
    "qvfl_s": ("diagnostic_state", "qvfl_s"),
    "tkvm": ("diagnostic_state", "tkvm"),
    "tkvh": ("diagnostic_state", "tkvh"),
    "rcld": ("diagnostic_state", "rcld"),
    "rhon": ("diagnostic_state", "rhon"),
    "tketens": ("tendency_state", "ddt_tke"),
    "tket_hshr": ("tendency_state", "tket_hshr"),
    "u_tens": ("tendency_state", "ddt_u"),
    "v_tens": ("tendency_state", "ddt_v"),
    "t_tens": ("tendency_state", "ddt_t"),
    "qv_tens": ("tendency_state", "ddt_qv"),
    "qc_tens": ("tendency_state", "ddt_qc"),
}


@pytest.fixture
def grid():
    return simple_grid.simple_grid()


@pytest.fixture
def grid_state(grid, monkeypatch):
    """A 'GridState' built on the simple grid, in place of a 'grid_init' from savepoint data.

    'turbulence_init' takes the grid and the vertical grid straight from the module global that
    'grid_init' fills, and hands them to 'Turbulence', which is mocked here. Only the cell count
    and the level count are used on this side, to size the wrapper's own buffers.
    """
    state = grid_wrapper.GridState(
        grid=grid,
        vertical_grid=v_grid.VerticalGrid(
            config=v_grid.VerticalGridConfig(num_levels=grid.num_levels),
            vct_a=data_alloc.zero_field(grid, dims.KDim, extend={dims.KDim: 1}),
            vct_b=None,
        ),
        edge_geometry=None,
        cell_geometry=None,
        exchange_runtime=None,
    )
    monkeypatch.setattr(grid_wrapper, "grid_state", state)
    return state


@pytest.fixture(autouse=True)
def _forget_the_granule(monkeypatch):
    """'turbulence_init' writes a module global; no test may inherit another's."""
    monkeypatch.setattr(turbulence_wrapper, "granule", None)


def _levels(grid, shape: str) -> tuple[int, ...]:
    return {
        "surface": (grid.num_cells,),
        "main": (grid.num_cells, grid.num_levels),
        "half": (grid.num_cells, grid.num_levels + 1),
    }[shape]


def _distinct_arrays(grid) -> dict[str, np.ndarray]:
    """One array per field of 'turbulence_run', each holding values no other field holds.

    Two arrays of the same shape swapped at the boundary is the failure this module exists to
    catch, so 'np.full' with a per-field offset is not laziness: it makes every swap visible.

    Fortran layout, because 'array_to_array_info' silently COPIES an array that is not
    F-contiguous and the copy is what the wrapper would then write into. The tests that read a
    result back out of one of these arrays depend on it aliasing.
    """
    return {
        name: np.full(_levels(grid, shape), float(index + 1), dtype=np.float64, order="F")
        for index, (name, shape) in enumerate(RUN_FIELDS.items())
    }


def _metric_arrays(grid) -> dict[str, np.ndarray]:
    return {
        "hhl": np.full(_levels(grid, "half"), 101.0, order="F"),
        "dp0": np.full(_levels(grid, "main"), 102.0, order="F"),
        "l_hori": np.full(_levels(grid, "surface"), 103.0),
        "trop_mask": np.full(_levels(grid, "surface"), 104.0),
        "innertrop_mask": np.full(_levels(grid, "surface"), 105.0),
    }


def _config_kwargs(config: turbulence.TurbulenceConfig) -> dict:
    """Every member of 'TurbulenceConfig' as a flat keyword argument.

    Built from 'dataclasses.fields' rather than written out, so that a member the wrapper's
    signature does not accept fails here with 'unexpected keyword argument' -- which is the
    point: the interface is the contract and a switch ICON can set must be expressible
    (port spec D5/D6).
    """
    return {
        field.name: int(value) if isinstance(value, int) else value
        for field in dataclasses.fields(config)
        if (value := getattr(config, field.name)) is not None
    }


def _init(grid, config, ffi, metric=None, nturb_tracer_tot=0):
    """Run 'turbulence_init' with 'Turbulence.__init__' mocked out; return its call kwargs.

    'nturb_tracer_tot' is spelled out rather than taken from '_config_kwargs' because it is not
    a member of 'TurbulenceConfig': ICON computes it at the interface, not in 'turbdiff_nml'.
    """
    arrays = _metric_arrays(grid) if metric is None else metric
    with mock.patch.object(turbulence.Turbulence, "__init__", return_value=None) as mocked:
        turbulence_wrapper.turbulence_init(
            ffi=ffi,
            perf_counters=None,
            **{name: test_utils.array_to_array_info(a) for name, a in arrays.items()},
            **_config_kwargs(config),
            nturb_tracer_tot=nturb_tracer_tot,
            backend=wrapper_common.BackendIntEnum.DEFAULT,
        )
    return mocked.call_args.kwargs, arrays


def test_turbulence_init_builds_the_configuration_the_flat_arguments_describe(grid, grid_state):
    """Every member of 'TurbulenceConfig' survives the round trip through the C boundary."""
    expected = turbulence.TurbulenceConfig(**MCH_NAMELIST)

    captured, _ = _init(grid, expected, cffi.FFI())

    assert captured["config"] == expected
    assert captured["params"] == turbulence.TurbulenceParams(expected)
    assert captured["grid"] is grid
    assert captured["vertical_grid"] is grid_state.vertical_grid


def test_turbulence_init_builds_the_metric_state_from_the_flat_arguments(grid, grid_state):
    ffi = cffi.FFI()
    captured, arrays = _init(grid, turbulence.TurbulenceConfig(**MCH_NAMELIST), ffi)

    metric_state = captured["metric_state"]
    for name, array in arrays.items():
        assert np.array_equal(getattr(metric_state, name).asnumpy(), array), name


def test_turbulence_init_defaults_are_a_configuration_the_granule_accepts(grid, grid_state):
    """'TurbulenceConfig()' is the compiled-in default of mo_turbdiff_config.f90.

    A Fortran caller that fills the list from an unmodified 'tdc' must not be refused before it
    has said anything.
    """
    captured, _ = _init(grid, turbulence.TurbulenceConfig(), cffi.FFI())

    assert captured["config"] == turbulence.TurbulenceConfig()


@pytest.mark.parametrize("count", [1, 2, 7])
def test_turbulence_init_refuses_the_tracers_it_cannot_be_given(grid, grid_state, count):
    """The guard that replaces ICON's 'finish', and the reason the tuples below may stay empty.

    'turbulence_run' passes 'tracers=()' and 'ddt_tracers=()' unconditionally, so a run with
    'ndtr > 0' would drop a physical process and report nothing. Refusing the count at init is
    what makes the empty tuples honest; before this existed the only guard was
    'check_supported_configuration' in ICON's 'mo_icon4py_turbulence.f90', in another repository
    and invisible from the wrapper.
    """
    with pytest.raises(NotImplementedError, match="nturb_tracer_tot"):
        _init(
            grid,
            turbulence.TurbulenceConfig(**MCH_NAMELIST),
            cffi.FFI(),
            nturb_tracer_tot=count,
        )


def test_turbulence_run_before_init_says_so(grid):
    with pytest.raises(RuntimeError, match="turbulence_init"):
        turbulence_wrapper.turbulence_run(
            ffi=cffi.FFI(),
            perf_counters=None,
            **{
                name: test_utils.array_to_array_info(array)
                for name, array in _distinct_arrays(grid).items()
            },
            dt_var=1.0,
            dt_tke=1.0,
        )


def _run(grid, ffi, *, side_effect=None):
    """Run 'turbulence_init' then 'turbulence_run', both mocked; return run's call kwargs."""
    _init(grid, turbulence.TurbulenceConfig(**MCH_NAMELIST), ffi)
    arrays = _distinct_arrays(grid)
    with mock.patch.object(turbulence.Turbulence, "run", side_effect=side_effect) as mocked:
        turbulence_wrapper.turbulence_run(
            ffi=ffi,
            perf_counters=None,
            **{name: test_utils.array_to_array_info(a) for name, a in arrays.items()},
            dt_var=17.0,
            dt_tke=19.0,
        )
    return mocked.call_args.kwargs, arrays


def test_turbulence_run_puts_every_field_in_the_slot_the_granule_reads_it_from(grid, grid_state):
    captured, arrays = _run(grid, cffi.FFI())

    for argument, (container, member) in RUN_DESTINATIONS.items():
        arrived = getattr(captured[container], member)
        assert np.array_equal(arrived.asnumpy(), arrays[argument]), (
            f"'{argument}' did not arrive at '{container}.{member}'"
        )
    assert captured["dt_var"] == 17.0
    assert captured["dt_tke"] == 19.0


def test_turbulence_run_passes_icons_tke_as_the_granules_input_tke(grid, grid_state):
    captured, arrays = _run(grid, cffi.FFI())

    assert np.array_equal(captured["input_state"].tke.asnumpy(), arrays["tke"])


def test_turbulence_run_hands_the_updated_tke_back_in_icons_array(grid, grid_state):
    """ICON has one 'z_tvs' and the granule has two fields; the wrapper closes the gap.

    'mo_nwp_turbdiff_interface.f90:812,:825' reads the array back to update 'p_prog_rcf%tke', so
    a wrapper that left the granule's output in its own buffer would silently feed the next step
    the previous step's TKE.
    """
    ffi = cffi.FFI()

    def write_the_output(**kwargs):
        kwargs["diagnostic_state"].updated_tke.ndarray[...] = 42.0

    _, arrays = _run(grid, ffi, side_effect=write_the_output)

    assert np.array_equal(arrays["tke"], np.full(arrays["tke"].shape, 42.0))


def test_turbulence_run_seeds_the_tke_buffer_so_untouched_columns_survive(grid, grid_state):
    """The granule computes 'ivstart..ivend' only; the rest of the array must come back as it
    went in."""
    ffi = cffi.FFI()

    seen = {}

    def look(**kwargs):
        seen["tke"] = kwargs["diagnostic_state"].updated_tke.asnumpy().copy()

    _, arrays = _run(grid, ffi, side_effect=look)

    assert np.array_equal(seen["tke"], np.full(arrays["tke"].shape, seen["tke"].flat[0]))
    assert seen["tke"].flat[0] == float(list(RUN_FIELDS).index("tke") + 1)


def test_the_state_members_the_granule_ignores_are_poisoned(grid, grid_state):
    """A field that is not an argument of 'turbulence_run' must not look like a valid value.

    The wrapper leaves out the members of the state containers that 'Turbulence.run' provably
    neither reads nor writes. Zero-filling them would make a granule that started reading one
    keep producing plausible numbers; NaN makes it fail. The two boolean masks cannot carry NaN
    and are the stated exception.
    """
    captured, _ = _run(grid, cffi.FFI())

    states = {
        "input_state": ("w", "tket_conv"),
        "surface_state": ("fr_land", "urb_isa", "rlamh_fac", "z0_waves"),
        "diagnostic_state": (
            "tfv",
            "tprn",
            "edr",
            "tur_len_scale",
            "tcm",
            "tch",
            "umfl_s",
            "vmfl_s",
            "t_2m",
            "qv_2m",
            "td_2m",
            "rh_2m",
            "u_10m",
            "v_10m",
        ),
    }
    for container, members in states.items():
        for member in members:
            values = getattr(captured[container], member).asnumpy()
            assert np.isnan(values).all(), f"'{container}.{member}' is not poisoned"

    assert not captured["surface_state"].l_lake.asnumpy().any()
    assert not captured["surface_state"].l_sice.asnumpy().any()
    assert math.isnan(float(np.max(captured["input_state"].w.asnumpy()))) is True


def test_the_tracer_tuples_are_empty(grid, grid_state):
    """Nothing to diffuse and nothing to pass -- 'turbulence_init' has refused any other count."""
    captured, _ = _run(grid, cffi.FFI())

    assert captured["input_state"].tracers == ()
    assert captured["tendency_state"].ddt_tracers == ()


# --------------------------------------------------------------------------- codegen ---


@pytest.fixture(scope="module")
def rendered_fortran(tmp_path_factory) -> str:
    """The '.f90' the registered wrappers render to, as 'test_codegen_references.py' makes it."""
    path = tmp_path_factory.mktemp("bindings") / f"{all_bindings.LIBRARY_NAME}.f90"
    result = CliRunner().invoke(all_bindings.main, ["--output-f90", str(path)])
    assert result.exit_code == 0, result.output
    return path.read_text()


@pytest.mark.parametrize("name", ["turbulence_init", "turbulence_run"])
def test_the_generated_fortran_exposes_the_turbulence_entry_points(rendered_fortran, name):
    assert f"public :: {name}" in rendered_fortran
    assert f"   subroutine {name}(" in rendered_fortran
    assert f'bind(c, name="{name}_wrapper")' in rendered_fortran


@pytest.mark.parametrize(
    "wrapper", [turbulence_wrapper.turbulence_init, turbulence_wrapper.turbulence_run]
)
def test_the_generated_fortran_takes_the_wrappers_arguments_in_order(rendered_fortran, wrapper):
    """The Fortran dummy-argument list is the wrapper's signature plus 'rc'.

    This is the check that makes the ICON side (plan task 4.2) writable against this module
    rather than against the rendered file: if an argument is added, removed or moved here, this
    fails naming it.
    """
    name = wrapper.__name__
    start = rendered_fortran.index(f"   subroutine {name}(")
    declaration = rendered_fortran[start : rendered_fortran.index(")\n", start)]
    arguments = [part.strip().rstrip(",&").strip() for part in declaration.split("\n")]
    arguments[0] = arguments[0].split("(", 1)[1]
    arguments = [a.rstrip(", &").rstrip(",").strip() for a in arguments if a]

    assert arguments == [*wrapper.param_descriptors, "rc"]
