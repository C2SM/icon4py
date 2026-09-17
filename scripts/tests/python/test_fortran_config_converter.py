# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Fortran namelist-to-YAML converter."""

from __future__ import annotations

import dataclasses
import datetime

import pytest

import fortran_config_converter as fcc


def _make_dicts(run_nml: dict) -> tuple[dict, dict]:
    """Minimal atm/master dicts for exercising make_driver_config."""
    atm_dict = {
        "nonhydrostatic_nml": {"vcfl_threshold": 0.85, "ndyn_substeps": 5},
        "run_nml": {"ltestcase": True, "ltransport": False} | run_nml,
    }
    master_dict = {
        "master_time_control_nml": {
            "experimentstartdate": "2000-01-01T00:00:00Z",
            "experimentstopdate": "2000-01-01T01:00:00Z",
        },
        "master_model_nml": {"model_namelist_filename": "NAMELIST_test_sb_atm"},
    }
    return atm_dict, master_dict


def test_modeltimestep_takes_priority_over_dtime() -> None:
    # Trailing whitespace mimics the fixed-width Fortran string.
    atm_dict, master_dict = _make_dicts(
        {"dtime": 999.0, "modeltimestep": "PT300S                          "}
    )
    config = fcc.make_driver_config(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.dtime == datetime.timedelta(seconds=300)


def test_empty_modeltimestep_falls_back_to_dtime() -> None:
    atm_dict, master_dict = _make_dicts({"dtime": 120.0, "modeltimestep": "        "})
    config = fcc.make_driver_config(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.dtime == datetime.timedelta(seconds=120)


@pytest.mark.parametrize("ltransport", [True, False])
def test_do_prep_adv_from_ltransport(ltransport: bool) -> None:
    atm_dict, master_dict = _make_dicts(
        {"dtime": 10.0, "modeltimestep": "  ", "ltransport": ltransport}
    )
    config = fcc.make_driver_config(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.do_prep_adv is ltransport


@pytest.mark.parametrize("ltestcase", [True, False])
def test_diffuse_before_time_loop(ltestcase: bool) -> None:
    atm_dict, master_dict = _make_dicts(
        {"dtime": 10.0, "modeltimestep": "  ", "ltestcase": ltestcase}
    )
    config = fcc.make_driver_config(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.diffuse_before_time_loop is (not ltestcase)
    assert config.apply_extra_second_order_divdamp is (not ltestcase)


@dataclasses.dataclass
class _SampleConfig:
    field_a: int = 0
    field_b: str = "default"


def test_config_dataclass_from_dict_uses_name_map() -> None:
    config = fcc.config_dataclass_from_dict(
        _SampleConfig,
        {"fortran_a": 42, "unknown": "ignored"},
        name_map={"fortran_a": "field_a"},
    )
    assert config.field_a == 42
    assert config.field_b == "default"
