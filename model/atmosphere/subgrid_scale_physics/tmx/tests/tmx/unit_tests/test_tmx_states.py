# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Wiring of the tmx static states onto the common field factories.

The field *values* are validated where the providers are registered, in
``common/tests/common/metrics/unit_tests/test_metrics_factory.py`` and
``common/tests/common/interpolation/unit_tests/test_interpolation_factory.py``.
What belongs to tmx is only which attribute each state member is fetched from.
"""

from __future__ import annotations

import dataclasses

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common.grid import geometry_attributes
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes


class _NamingSource:
    """A ``FieldSource`` stand-in that hands back the requested attribute name."""

    def get(self, field_name: str) -> str:
        return field_name


_METRICS_MEMBERS: dict[str, str] = {
    "ddqz_z_full": metrics_attributes.DDQZ_Z_FULL,
    "inv_ddqz_z_full": metrics_attributes.INV_DDQZ_Z_FULL,
    "ddqz_z_half": metrics_attributes.DDQZ_Z_HALF,
    "inv_ddqz_z_half": metrics_attributes.INV_DDQZ_Z_HALF,
    "inv_ddqz_z_full_e": metrics_attributes.INV_DDQZ_Z_FULL_E,
    "inv_ddqz_z_half_e": metrics_attributes.INV_DDQZ_Z_HALF_E,
    "inv_ddqz_z_half_v": metrics_attributes.INV_DDQZ_Z_HALF_V,
    "wgtfac_c": metrics_attributes.WGTFAC_C,
    "wgtfac_e": metrics_attributes.WGTFAC_E,
    "wgtfacq_c": metrics_attributes.WGTFACQ_C,
    "wgtfacq1_c": metrics_attributes.WGTFACQ1_C,
    "wgtfacq_e": metrics_attributes.WGTFACQ_E,
    "wgtfacq1_e": metrics_attributes.WGTFACQ1_E,
    "geopot_agl_ifc": metrics_attributes.GEOPOT_AGL_IFC,
    "height_above_ground": metrics_attributes.HEIGHT_ABOVE_GROUND,
    "z_mc": metrics_attributes.Z_MC,
    "z_ifc": metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL,
}

_GEOMETRY_MEMBERS: dict[str, str] = {
    "edge_cell_length": geometry_attributes.EDGE_CELL_DISTANCE,
}

_INTERPOLATION_MEMBERS: dict[str, str] = {
    "c_lin_e": interpolation_attributes.C_LIN_E,
    "e_bln_c_s": interpolation_attributes.E_BLN_C_S,
    "geofac_div": interpolation_attributes.GEOFAC_DIV,
    "cells_aw_verts": interpolation_attributes.CELL_AW_VERTS,
    "rbf_coeff_v1": interpolation_attributes.RBF_VEC_COEFF_V1,
    "rbf_coeff_v2": interpolation_attributes.RBF_VEC_COEFF_V2,
    "rbf_coeff_e": interpolation_attributes.RBF_VEC_COEFF_E,
    "rbf_coeff_c1": interpolation_attributes.RBF_VEC_COEFF_C1,
    "rbf_coeff_c2": interpolation_attributes.RBF_VEC_COEFF_C2,
}


def test_metric_state_members_come_from_the_expected_attributes() -> None:
    expected = {**_METRICS_MEMBERS, **_GEOMETRY_MEMBERS}
    state = tmx_states.TmxMetricState.from_sources(
        metrics=_NamingSource(),  # type: ignore[arg-type]
        geometry=_NamingSource(),  # type: ignore[arg-type]
    )
    assert {field.name for field in dataclasses.fields(state)} == set(expected)
    for member, attribute in expected.items():
        assert getattr(state, member) == attribute, member


def test_interpolation_state_members_come_from_the_expected_attributes() -> None:
    state = tmx_states.TmxInterpolationState.from_sources(
        interpolation=_NamingSource(),  # type: ignore[arg-type]
    )
    assert {field.name for field in dataclasses.fields(state)} == set(_INTERPOLATION_MEMBERS)
    for member, attribute in _INTERPOLATION_MEMBERS.items():
        assert getattr(state, member) == attribute, member


def test_requested_attributes_are_registered_in_the_common_metadata() -> None:
    for attribute in _METRICS_MEMBERS.values():
        assert attribute in metrics_attributes.attrs
    for attribute in _GEOMETRY_MEMBERS.values():
        assert attribute in geometry_attributes.attrs
    for attribute in _INTERPOLATION_MEMBERS.values():
        assert attribute in interpolation_attributes.attrs
