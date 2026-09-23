# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The C kind of each wrapper argument follows ICON's precision, not icon4py's.

None of these tests needs serialized data.
"""

import json
import os
import subprocess
import sys
import typing

import cffi
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.bindings import (
    all_bindings,
    common as wrapper_common,
    config,
    dycore_wrapper,
    icon4py_export,
)
from icon4py.bindings.icon4py_export import Vp, Wp
from icon4py.model.common import type_alias as ta
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils


SomeDim = gtx.Dimension("SomeDim")

# Arguments that ICON declares `REAL(vp)`; every other float argument is `REAL(wp)`.
# Metric fields: `icon/src/atm_dyn_iconam/mo_nonhydro_types.f90:348` (t_nh_metrics vp block).
# Diagnostic fields: same file, lines 142 and 265 (t_nh_diag vp block, `max_vcfl_dyn`).
# Call-site locals: `icon/src/atm_dyn_iconam/mo_icon4py_interfaces.f90:657-660` and
# `:1605-1608`. Checked against icon HEAD 554c4ef194. This is the substitute-mode call sites
# only: the verify-mode `solve_nh_run` call (same file, :1058) passes `REAL(wp)` `*_before`
# buffers for 13 of these, so it does not match a mixed interface until ICON declares them `vp`.
ICON_VP_ARGUMENTS: typing.Final = {
    "diffusion_init": {"theta_ref_mc", "wgtfac_c"},
    "diffusion_run": {"hdef_ic", "div_ic", "dwdx", "dwdy"},
    "grid_init": set(),
    "solve_nh_init": {
        "exner_exfac",
        "exner_ref_mc",
        "wgtfac_c",
        "wgtfacq_c",
        "inv_ddqz_z_full",
        "rho_ref_mc",
        "theta_ref_mc",
        "d_exner_dz_ref_ic",
        "ddqz_z_half",
        "theta_ref_ic",
        "d2dexdz2_fac1_mc",
        "d2dexdz2_fac2_mc",
        "rho_ref_me",
        "theta_ref_me",
        "ddxn_z_full",
        "zdiff_gradp",
        "pg_exdist",
        "ddqz_z_full_e",
        "ddxt_z_full",
        "wgtfac_e",
        "wgtfacq_e",
        "coeff1_dwdz",
        "coeff2_dwdz",
        "coeff_gradekin",
    },
    "solve_nh_run": {
        "w_concorr_c",
        "ddt_vn_apc_ntl1",
        "ddt_vn_apc_ntl2",
        "ddt_w_adv_ntl1",
        "ddt_w_adv_ntl2",
        "exner_dyn_incr",
        "ddt_exner_phy",
        "ddt_vn_phy",
        "vn_ie",
        "vt",
        "vn_incr",
        "rho_incr",
        "exner_incr",
        "max_vcfl_size1_array",
    },
}

_FLOAT32_ARGUMENTS_SCRIPT = """
import json
from icon4py.bindings import all_bindings
from icon4py.tools import py2fgen
print(json.dumps({
    f.__name__: sorted(n for n, d in f.param_descriptors.items() if d.dtype == py2fgen.FLOAT32)
    for f in all_bindings.FUNCTIONS
}))
"""


def _float32_arguments(**env: str) -> dict[str, set[str]]:
    """Import the bindings in a fresh interpreter, since both precisions are read at import."""
    result = subprocess.run(
        [sys.executable, "-c", _FLOAT32_ARGUMENTS_SCRIPT],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return {name: set(args) for name, args in json.loads(result.stdout).items()}


@pytest.mark.single_precision_ready
@pytest.mark.parametrize("icon4py_precision", ["double", "single"])
def test_icon_mixed_changes_exactly_the_vp_arguments(icon4py_precision):
    result = _float32_arguments(
        ICON4PY_BINDINGS_ICON_PRECISION="mixed", ICON4PY_FLOAT_PRECISION=icon4py_precision
    )

    assert result == ICON_VP_ARGUMENTS


@pytest.mark.single_precision_ready
@pytest.mark.parametrize("icon4py_precision", ["double", "single"])
def test_icon_double_interface_is_all_double(icon4py_precision):
    result = _float32_arguments(
        ICON4PY_BINDINGS_ICON_PRECISION="double", ICON4PY_FLOAT_PRECISION=icon4py_precision
    )

    assert result == {name: set() for name in ICON_VP_ARGUMENTS}


@pytest.mark.single_precision_ready
@pytest.mark.parametrize(
    "icon_precision, kind, expected",
    [
        ("double", icon4py_export.IconKind.WP, py2fgen.FLOAT64),
        ("double", icon4py_export.IconKind.VP, py2fgen.FLOAT64),
        ("mixed", icon4py_export.IconKind.WP, py2fgen.FLOAT64),
        ("mixed", icon4py_export.IconKind.VP, py2fgen.FLOAT32),
    ],
)
def test_icon_scalar_kind(monkeypatch, icon_precision, kind, expected):
    monkeypatch.setattr(config, "ICON_PRECISION", icon_precision)

    assert icon4py_export.icon_scalar_kind(kind) == expected


def _vp_optional_inside(_: Vp[gtx.Field[gtx.Dims[SomeDim], gtx.float32] | None]):
    pass


def _vp_optional_outside(_: Vp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]] | None):
    pass


@pytest.mark.single_precision_ready
@pytest.mark.parametrize("fun", [_vp_optional_inside, _vp_optional_outside])
def test_split_boundary_accepts_both_optional_spellings(fun):
    hint = typing.get_type_hints(fun, include_extras=True)["_"]

    base, boundary = icon4py_export._split_boundary(hint)

    assert boundary == icon4py_export.Boundary(icon4py_export.IconKind.VP)
    assert base == (gtx.Field[gtx.Dims[SomeDim], gtx.float32] | None)


@pytest.mark.single_precision_ready
def test_split_boundary_leaves_unmarked_hints_alone():
    hint = gtx.Field[gtx.Dims[SomeDim], gtx.int32]

    assert icon4py_export._split_boundary(hint) == (hint, None)


@pytest.mark.single_precision_ready
@pytest.mark.parametrize("icon_precision", ["double", "mixed"])
def test_descriptor_is_icon_kind_not_compute_dtype(monkeypatch, icon_precision):
    monkeypatch.setattr(config, "ICON_PRECISION", icon_precision)
    # icon4py computes in float32 here; what crosses is decided by ICON alone
    hint = Vp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]]

    descriptor = icon4py_export.field_annotation_descriptor_hook(hint)

    assert descriptor.dtype == icon4py_export.icon_scalar_kind(icon4py_export.IconKind.VP)
    assert descriptor.rank == 1


@pytest.mark.single_precision_ready
@pytest.mark.parametrize(
    "hint, message",
    [
        (gtx.Field[gtx.Dims[SomeDim], gtx.float64], "needs its ICON kind"),
        (Wp[gtx.Field[gtx.Dims[SomeDim], gtx.int32]], "non-float field"),
        (Wp[gtx.float64], "scalars cross by value"),
        (gtx.float32, "always double"),
    ],
)
def test_descriptor_hook_rejects(hint, message):
    with pytest.raises(TypeError, match=message):
        icon4py_export.field_annotation_descriptor_hook(hint)


def _field_mappers():
    """Yield (function, name, descriptor, mapper) for every argument mapped to a GT4Py field."""
    for fun in all_bindings.FUNCTIONS:
        hints = typing.get_type_hints(fun.__wrapped__, include_extras=True)
        for name, descriptor in fun.param_descriptors.items():
            mapper = icon4py_export.field_annotation_mapping_hook(hints[name], descriptor)
            if mapper is not None:
                yield fun.__name__, name, descriptor, mapper


@pytest.mark.single_precision_ready
def test_every_field_argument_is_a_view_or_refuses():
    """
    Every field argument sees ICON's own memory when ICON's dtype and icon4py's agree.

    When they differ it must refuse loudly, because no wrapper argument has a declared intent
    yet. Holds for any combination of the two precision settings.
    """
    ffi = cffi.FFI()
    viewed, refused = [], []
    for fun_name, name, descriptor, mapper in _field_mappers():
        icon_dtype = icon4py_export._NUMPY_DTYPE[descriptor.dtype]
        fortran_array = np.zeros((2, 3, 4)[: descriptor.rank], dtype=icon_dtype, order="F")
        array_info = test_utils.array_to_array_info(fortran_array, ffi=ffi)
        try:
            field = mapper(array_info, ffi=ffi)
        except TypeError as error:
            assert "declare whether ICON reads it back" in str(error), f"{fun_name}.{name}"
            refused.append(f"{fun_name}.{name}")
            continue

        assert field.dtype.scalar_type == icon_dtype, f"{fun_name}.{name}"
        assert np.shares_memory(field.ndarray, fortran_array), f"{fun_name}.{name} was copied"
        viewed.append(f"{fun_name}.{name}")

    # 115 float fields and 12 integer or bool fields; guards against checking nothing
    assert len(viewed) + len(refused) == 127
    assert set(refused) == _expected_refusals()


def _expected_refusals() -> set[str]:
    """Float fields whose ICON dtype differs from icon4py's, derived from the ICON oracle."""
    icon_vp_is_single = config.ICON_PRECISION == "mixed"
    expected = set()
    for fun_name, name, descriptor, _ in _field_mappers():
        if descriptor.dtype not in (py2fgen.FLOAT32, py2fgen.FLOAT64):
            continue
        icon_is_single = icon_vp_is_single and name in ICON_VP_ARGUMENTS[fun_name]
        # icon4py computes every float field in single or in double, never mixed, here
        compute_is_single = ta.precision == "single"
        if icon_is_single != compute_is_single:
            expected.add(f"{fun_name}.{name}")
    return expected


@pytest.mark.single_precision_ready
def test_field_mapper_refuses_to_copy_an_argument_without_declared_intent():
    ffi = cffi.FFI()
    hint = Wp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]]  # ICON passes double, always
    descriptor = icon4py_export.field_annotation_descriptor_hook(hint)
    mapper = icon4py_export.field_annotation_mapping_hook(hint, descriptor)
    fortran_array = np.zeros(5, dtype=np.float64)

    with pytest.raises(TypeError, match="declare whether ICON reads it back"):
        mapper(test_utils.array_to_array_info(fortran_array, ffi=ffi), ffi=ffi)


def _mapper_for(hint):
    descriptor = icon4py_export.field_annotation_descriptor_hook(hint)
    return icon4py_export.field_annotation_mapping_hook(hint, descriptor)


@pytest.mark.single_precision_ready
@pytest.mark.parametrize(
    "hint",
    [
        Wp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]],  # wp is always double in ICON
        Vp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]],  # vp is double unless ICON is mixed
        icon4py_export.VpInOut[gtx.Field[gtx.Dims[SomeDim], gtx.float32]],
    ],
)
def test_pointer_contradicting_the_configured_icon_precision_raises(monkeypatch, hint):
    monkeypatch.setattr(config, "ICON_PRECISION", "double")
    ffi = cffi.FFI()
    mapper = _mapper_for(hint)
    single = np.zeros(5, dtype=np.float32)

    with pytest.raises(TypeError, match="ICON4PY_BINDINGS_ICON_PRECISION"):
        mapper(test_utils.array_to_array_info(single, ffi=ffi), ffi=ffi)


@pytest.mark.single_precision_ready
def test_single_vp_pointer_is_accepted_when_icon_is_mixed(monkeypatch):
    monkeypatch.setattr(config, "ICON_PRECISION", "mixed")
    ffi = cffi.FFI()
    mapper = _mapper_for(Vp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]])
    single = np.zeros(5, dtype=np.float32)

    field = mapper(test_utils.array_to_array_info(single, ffi=ffi), ffi=ffi)

    assert np.shares_memory(field.ndarray, single)


@pytest.mark.single_precision_ready
def test_every_raw_float_array_declares_its_icon_kind():
    """Raw arrays bypass `Wp`/`Vp`, so their alias is the only place the ICON kind lives."""
    icon_kind_aliases = {
        wrapper_common.IconWpArray2D,
        wrapper_common.IconWpArray3D,
        wrapper_common.OptionalIconWpArray1D,
        wrapper_common.OptionalIconWpArray2D,
        wrapper_common.OptionalIconVpArray1D,
        dycore_wrapper.NumpyIconVpArray1D,
    }
    raw_float_arrays = []
    for fun in all_bindings.FUNCTIONS:
        hints = typing.get_type_hints(fun.__wrapped__, include_extras=True)
        for name, descriptor in fun.param_descriptors.items():
            is_raw_float_array = (
                isinstance(descriptor, py2fgen.ArrayParamDescriptor)
                and descriptor.dtype in (py2fgen.FLOAT32, py2fgen.FLOAT64)
                and icon4py_export.field_annotation_mapping_hook(hints[name], descriptor) is None
            )
            if is_raw_float_array:
                assert hints[name] in icon_kind_aliases, f"{fun.__name__}.{name}: {hints[name]}"
                raw_float_arrays.append(f"{fun.__name__}.{name}")

    # rbf_vec_coeff_{e,v}, pg_exdist, max_vcfl_size1_array, rbf_vec_coeff_v, zd_{intcoef,diffcoef}
    assert len(raw_float_arrays) == 7
