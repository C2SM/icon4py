# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Fortran namelist-to-ExperimentConfig converter."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import pathlib

import f90nml
import fortran_config_converter as fcc
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.common import constants, prescribed_tendencies
from icon4py.model.common.config import config_io
from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.initial_condition.analytical import gauss3d as gauss_ic
from icon4py.model.common.topography import from_file as from_file_topo
from icon4py.model.common.topography.analytical import gaussian_hill as gausshill_topo
from icon4py.model.driver import config as driver_config


# The subset of the exclaim_gauss3d namelists read by the converter. `max_dom`-sized
# lists are shortened to two entries.
ATM_NML = {
    "diffusion_nml": {
        "hdiff_efdt_ratio": 36.0,
        "hdiff_order": 5,
        "hdiff_smag_fac": 0.015,
        "hdiff_smag_fac2": 0.07137250900268555,
        "hdiff_smag_fac3": 0.0,
        "hdiff_smag_fac4": 1.0,
        "hdiff_smag_z": 32500.0,
        "hdiff_smag_z2": 60686.25390625,
        "hdiff_smag_z3": 50000.0,
        "hdiff_smag_z4": 90000.0,
        "hdiff_w_efdt_ratio": 15.0,
        "itype_t_diffu": 2,
        "itype_vn_diffu": 1,
        "lhdiff_smag_w": [False, False],
        "lhdiff_temp": True,
        "lhdiff_vn": True,
        "lhdiff_w": True,
        "lsmag_3d": [False, False],
    },
    "dynamics_nml": {"divavg_cntrwgt": 0.5, "ldeepatmo": False},
    "gridref_nml": {"denom_diffu_t": 135.0, "denom_diffu_v": 200.0},
    "initicon_nml": {"init_mode": 2},
    "interpol_nml": {
        "lsq_high_ord": 3,
        "nudge_efold_width": 2.0,
        "nudge_max_coeff": 0.02,
        "nudge_zone_width": 8,
        "rbf_vec_kern_c": 1,
        "rbf_vec_kern_e": 3,
        "rbf_vec_kern_v": 1,
    },
    "nonhydrostatic_nml": {
        "damp_height": [45000.0, 45000.0],
        "divdamp_fac": 0.0025,
        "divdamp_fac2": 0.004,
        "divdamp_fac3": 0.004,
        "divdamp_fac4": 0.004,
        "divdamp_order": 24,
        "divdamp_trans_end": 17500.0,
        "divdamp_trans_start": 12500.0,
        "divdamp_type": 3,
        "divdamp_z": 32500.0,
        "divdamp_z2": 40000.0,
        "divdamp_z3": 60000.0,
        "divdamp_z4": 80000.0,
        "exner_expol": 0.3333333333333333,
        "htop_moist_proc": 22500.0,
        "iadv_rhotheta": 2,
        "igradp_method": 3,
        "itime_scheme": 4,
        "l_zdiffu_t": True,
        "lextra_diffu": True,
        "ndyn_substeps": 5,
        "rayleigh_coeff": [0.1, 0.1],
        "rayleigh_type": 2,
        "rhotheta_offctr": -0.1,
        "thhgtd_zdiffu": 200.0,
        "thslp_zdiffu": 0.025,
        "vcfl_threshold": 1.05,
        "veladv_offctr": 0.25,
        "vwind_offctr": 0.15,
    },
    "run_nml": {
        "dtime": 4.0,
        "iforcing": 0,
        "ltestcase": True,
        "ltransport": False,
        "lvert_nest": False,
        # ICON writes the ISO 8601 duration as a fixed-width, blank-padded string.
        "modeltimestep": "                                ",
        "ntracer": 0,
        "num_lev": [35, 31],
    },
    "sleve_nml": {
        "decay_exp": 1.2,
        "decay_scale_1": 4000.0,
        "decay_scale_2": 2500.0,
        "flat_height": 16000.0,
        "htop_thcknlimit": 15000.0,
        "max_lay_thckn": 25000.0,
        "min_lay_thckn": 50.0,
        "stretch_fac": 1.0,
        "top_height": 23500.0,
    },
    "transport_nml": {
        "ihadv_tracer": [2, 2],
        "itype_hlimit": [4, 4],
        "itype_vlimit": [1, 1],
        "ivadv_tracer": [3, 3],
    },
    "turbdiff_nml": {"a_hshr": 1.0, "itype_sher": 0},
}
MASTER_NML = {
    "master_model_nml": {"model_namelist_filename": "NAMELIST_exclaim_gauss3d_sb_atm"},
    "master_time_control_nml": {
        "experimentstartdate": "2008-09-01T00:00:00Z",
        "experimentstopdate": "2008-09-01T00:00:40Z",
    },
}
INPUT_NML = {
    "nh_testcase_nml": {
        "nh_test_name": "gauss3D",
        "mount_height": 100.0,
        "mount_width": 500.0,
        "nh_u0": 0.0,
        "nh_t0": 300.0,
        "nh_brunt_vais": 0.01,
    }
}
INPUT_NML_FNAME = "NAMELIST_exclaim_gauss3d_sb"


def _write_namelists(
    namelist_dir: pathlib.Path, *, atm: dict = ATM_NML, input_fname: str = INPUT_NML_FNAME
) -> None:
    f90nml.Namelist(atm).write(namelist_dir / fcc.NAMELIST_ATM_FNAME)
    f90nml.Namelist(MASTER_NML).write(namelist_dir / fcc.NAMELIST_MASTER_FNAME)
    f90nml.Namelist(INPUT_NML).write(namelist_dir / input_fname)


def _driver_namelists(run_nml: dict) -> dict:
    """Minimal master/model namelists for exercising the DRIVER mapping."""
    return {
        "master_cfg": {
            "master_time_control_nml": {
                "experimentstartdate": "2000-01-01T00:00:00Z",
                "experimentstopdate": "2000-01-01T01:00:00Z",
            },
            "master_model_nml": {"model_namelist_filename": "NAMELIST_test_sb_atm"},
        },
        "model_cfg": {
            "nonhydrostatic_nml": {"vcfl_threshold": 0.85, "ndyn_substeps": 5},
            "run_nml": {"ltestcase": True, "ltransport": False} | run_nml,
        },
    }


def test_modeltimestep_takes_priority_over_dtime() -> None:
    # Trailing whitespace mimics the fixed-width Fortran string.
    icon_config = _driver_namelists(
        {"dtime": 999.0, "modeltimestep": "PT300S                          "}
    )
    config = fcc.DRIVER.build(icon_config, profiling_options=None)
    assert config.dtime == datetime.timedelta(seconds=300)


def test_empty_modeltimestep_falls_back_to_dtime() -> None:
    icon_config = _driver_namelists({"dtime": 120.0, "modeltimestep": "        "})
    config = fcc.DRIVER.build(icon_config, profiling_options=None)
    assert config.dtime == datetime.timedelta(seconds=120)


# The extra diffusion call before the time loop is only made for real data runs, which
# are the ones that are not a testcase. MCH_CH_R04B09 is the only one.
@pytest.mark.parametrize("ltestcase", [True, False])
def test_diffuse_before_time_loop(ltestcase: bool) -> None:
    icon_config = _driver_namelists({"dtime": 10.0, "modeltimestep": "  ", "ltestcase": ltestcase})
    config = fcc.DRIVER.build(icon_config, profiling_options=None)
    assert config.diffuse_before_time_loop is (not ltestcase)
    assert config.apply_extra_second_order_divdamp is (not ltestcase)


def test_max_dom_lists_are_reduced_to_their_first_entry() -> None:
    config = fcc.DIFFUSION.build(ATM_NML)
    assert config.apply_smag_diff_to_vertical_wind is False
    assert config.compute_3d_smag_coeff is False


def test_field_type_is_the_fallback_converter() -> None:
    config = fcc.NONHYDROSTATIC.build(ATM_NML)
    assert config.rayleigh_type is constants.RayleighType.KLEMP


def test_final_field_type_is_unwrapped_for_the_fallback_converter() -> None:
    config = fcc.VERTICAL_GRID.build(ATM_NML)
    assert type(config.model_top_height) is np.float64


def test_missing_namelist_entry_is_an_error() -> None:
    atm = copy.deepcopy(ATM_NML)
    del atm["diffusion_nml"]["hdiff_order"]
    with pytest.raises(KeyError, match="hdiff_order"):
        fcc.DIFFUSION.build(atm)


@dataclasses.dataclass
class _SampleConfig:
    field_a: int = 0
    field_b: str = "default"


def test_missing_optional_entry_keeps_the_default() -> None:
    mapping = fcc.ConfigMapping(
        _SampleConfig,
        [
            fcc.IconOption("field_a", ("fortran_a",), required=False),
            fcc.IconOption("field_b", ("fortran_b",), required=False),
        ],
    )
    config = mapping.build({"fortran_a": "42"})
    assert config.field_a == 42
    assert config.field_b == "default"


def test_convert_experiment_testcase(tmp_path: pathlib.Path) -> None:
    _write_namelists(tmp_path)

    config = fcc.convert_experiment(tmp_path)

    assert config.driver.experiment_name == "exclaim_gauss3d"
    assert config.driver.dtime == datetime.timedelta(seconds=4)
    assert config.driver.start_of_timestepping == config.driver.start_of_simulation
    assert config.vertical_grid.num_levels == 35
    assert config.topography == gausshill_topo.GaussianHillConfig(
        mount_height=100.0, mount_width=500.0
    )
    assert config.initial_condition == gauss_ic.Gauss3DConfig(u0=0.0, t0=300.0, brunt_vais=0.01)
    assert config.prescribed_tendencies == prescribed_tendencies.PrescribedTendenciesConfig(
        data_path=None
    )
    assert config.graupel is None
    assert config.muphys is None


def test_convert_experiment_from_file_paths_resolve_against_the_config_file(
    tmp_path: pathlib.Path,
) -> None:
    atm = copy.deepcopy(ATM_NML)
    atm["run_nml"]["ltestcase"] = False
    _write_namelists(tmp_path, atm=atm)

    config = fcc.convert_experiment(tmp_path)

    # the generated config is portable: paths are relative to the namelist directory
    relative = pathlib.Path("ser_data")
    assert isinstance(config.topography, from_file_topo.FromFileConfig)
    assert isinstance(config.initial_condition, from_file_ic.FromFileConfig)
    assert config.topography.data_path == relative
    assert config.initial_condition.data_path == relative
    assert config.prescribed_tendencies.data_path == relative

    config_file = tmp_path / "config.yml"
    config_file.write_text(config_io.write_yaml_str(config))
    read_back = driver_config.read_experiment_config_from_yaml(config_file)
    resolved = tmp_path.resolve() / "ser_data"
    assert read_back.topography.data_path == resolved
    assert read_back.initial_condition.data_path == resolved
    assert read_back.prescribed_tendencies.data_path == resolved


def test_convert_experiment_with_explicit_namelist_expname(tmp_path: pathlib.Path) -> None:
    _write_namelists(tmp_path, input_fname="NAMELIST_other")
    (tmp_path / "NAMELIST_decoy").write_text("&nh_testcase_nml nh_test_name='jabw' /\n")

    with pytest.raises(FileNotFoundError, match="Expected exactly one"):
        fcc.convert_experiment(tmp_path)
    config = fcc.convert_experiment(tmp_path, namelist_expname="NAMELIST_other")
    assert isinstance(config.initial_condition, gauss_ic.Gauss3DConfig)


def _echoed_vdf_record(**overrides: object) -> list[object]:
    """A positional t_vdiff_config record as echoed in aes_vdf_nml.

    Positions not pinned by a TmxConfig option get a dummy value; the
    overrides are placed at the pinned 'unnamed_index' positions.
    """
    positions = {
        "use_tmx": 22,
        "solver_type": 23,
        "energy_type": 24,
        "dissipation_factor": 25,
        "use_louis": 26,
        "use_louis_land": 27,
        "use_louis_ice": 28,
        "louis_constant_b": 29,
        "use_km_const": 30,
        "km_const": 31,
        "use_scale_turb_energy_flux": 32,
        "scale_turb_energy_flux": 33,
        "smag_constant": 34,
        "turb_prandtl": 35,
        "km_min": 37,
        "max_turb_scale": 38,
    }
    record: list[object] = [0.0] * 42
    record[positions["use_tmx"]] = True
    for name, value in overrides.items():
        record[positions[name]] = value
    return record


def test_tmx_config() -> None:
    fortran_dict = {
        "aes_vdf_nml": {
            "aes_vdf_config": _echoed_vdf_record(
                solver_type=1,
                energy_type=1,
                dissipation_factor=0.5,
                use_louis=False,
                use_louis_land=False,
                use_louis_ice=False,
                louis_constant_b=2.1,
                use_km_const=True,
                km_const=2.0,
                use_scale_turb_energy_flux=True,
                scale_turb_energy_flux=0.9,
                smag_constant=0.28,
                turb_prandtl=0.5,
                km_min=0.002,
                max_turb_scale=150.0,
            )
        }
    }
    assert fcc.tmx_is_active(fortran_dict)
    config = fcc.TMX.build(fortran_dict)
    assert config.solver_type is tmx_config.SolverType.EXPLICIT
    assert config.energy_type is tmx_config.EnergyType.DRY_STATIC
    assert config.dissipation_factor == 0.5
    assert config.use_louis is False
    assert config.use_louis_land is False
    assert config.use_louis_ice is False
    assert config.louis_constant_b == 2.1
    assert config.use_km_const is True
    assert config.km_const == 2.0
    assert config.use_scale_turb_energy_flux is True
    assert config.scale_turb_energy_flux == 0.9
    assert config.smag_constant == 0.28
    assert config.turb_prandtl == 0.5
    assert config.km_min == 0.002
    assert config.max_turb_scale == 150.0


def test_tmx_rejects_changed_member_count() -> None:
    record = _echoed_vdf_record()
    with pytest.raises(ValueError, match="not a multiple"):
        fcc.tmx_is_active({"aes_vdf_nml": {"aes_vdf_config": [*record, 0.0]}})


def test_tmx_is_inactive_when_use_tmx_is_false() -> None:
    record = _echoed_vdf_record()
    record[22] = False
    assert not fcc.tmx_is_active({"aes_vdf_nml": {"aes_vdf_config": record}})
