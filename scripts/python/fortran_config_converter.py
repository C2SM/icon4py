#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert Fortran namelist files into an icon4py :class:`ExperimentConfig` YAML.

This module replaces the previous two-step pipeline (Fortran ``.nml`` → JSON
via ``f90nml`` → :class:`Config.from_fortran_dict`).  It reads the namelists
directly and assembles a :class:`driver.config.ExperimentConfig` that can be
serialized to YAML with :func:`icon4py.model.common.config.config_io.write_yaml_str`.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing
from typing import Any

import f90nml

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common import constants, initial_condition, prescribed_tendencies, time, topography
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.interpolation.rbf_interpolation import InterpolationKernel
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_states
from icon4py.model.driver import config as driver_config


# Time-format helpers remain in driver.config for ISO 8601 parsing.
absolutetime_from_iconformat = driver_config.absolutetime_from_iconformat
relativetime_from_iconformat = driver_config.relativetime_from_iconformat
relativetime_from_iso8601 = driver_config.relativetime_from_iso8601


# ---------------------------------------------------------------------------
# Helpers (moved from model/common/utils/fortran_config.py)
# ---------------------------------------------------------------------------


def list_to_value[T](obj: list[T] | T) -> T:
    # Some parameters are allocated as `max_dom`-sized lists, with one value
    # per domain. ICON4Py (for now) only runs on one domain.
    return obj[0] if isinstance(obj, list) else obj


def _translate_fields(
    source: dict[str, Any],
    name_map: dict[str, str],
    known_fields: set[str],
) -> dict[str, Any]:
    """Map Fortran namelist keys to Python field names, keeping only known dataclass fields."""
    params: dict[str, Any] = {}
    for key, value in source.items():
        python_name = name_map.get(key, key)
        if python_name in known_fields:
            params[python_name] = value
    return params


def config_dataclass_from_dict[T](
    cls: type[T], source: dict[str, Any], name_map: dict[str, str]
) -> T:
    """Construct a dataclass from a Fortran namelist dict.

    Unknown keys are ignored.  Missing keys fall back to dataclass defaults.
    Fortran→Python name translation is driven by the supplied ``name_map``:
    ``{fortran_key: python_field_name}``.
    """
    known_fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    kwargs = _translate_fields(source, name_map, known_fields)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Config builders
#
# DiffusionConfig, NonHydrostaticConfig, InterpolationConfig and DriverConfig
# previously used ``ConfigOption(icon_equivalent=IconOption(...))`` annotations
# read at runtime by ``construct_config_from_icon`` / ``iter_pairs_from_icon``.
# Those annotations and helpers have been removed from the model packages.
# The Fortran→Python mapping now lives here as explicit dicts.
# ---------------------------------------------------------------------------


def _extract(atm_dict: dict[str, Any], section: str, name: str, *, list_to_value: bool = False) -> Any:
    """Read a value from a namelist section, optionally de-listifying."""
    raw = atm_dict[section][name]
    return list_to_value(raw) if list_to_value else raw


_DIFFUSION_FIELDS = [
    # (python_field, nml_section, nml_name, list_to_value, converter)
    ("diffusion_type", "diffusion_nml", "hdiff_order", False, diffusion.DiffusionType),
    ("apply_to_vertical_wind", "diffusion_nml", "lhdiff_w", False, None),
    ("apply_to_horizontal_wind", "diffusion_nml", "lhdiff_vn", False, None),
    ("apply_to_temperature", "diffusion_nml", "lhdiff_temp", False, None),
    ("apply_smag_diff_to_vertical_wind", "diffusion_nml", "lhdiff_smag_w", True, None),
    ("compute_3d_smag_coeff", "diffusion_nml", "lsmag_3d", True, None),
    ("type_vn_diffu", "diffusion_nml", "itype_vn_diffu", False, diffusion.SmagorinskyStencilType),
    ("type_t_diffu", "diffusion_nml", "itype_t_diffu", False, diffusion.TemperatureDiscretizationType),
    ("hdiff_efdt_ratio", "diffusion_nml", "hdiff_efdt_ratio", False, None),
    ("hdiff_w_efdt_ratio", "diffusion_nml", "hdiff_w_efdt_ratio", False, None),
    ("smagorinski_scaling_factor", "diffusion_nml", "hdiff_smag_fac", False, None),
    ("smagorinski_scaling_factor2", "diffusion_nml", "hdiff_smag_fac2", False, None),
    ("smagorinski_scaling_factor3", "diffusion_nml", "hdiff_smag_fac3", False, None),
    ("smagorinski_scaling_factor4", "diffusion_nml", "hdiff_smag_fac4", False, None),
    ("smagorinski_scaling_height", "diffusion_nml", "hdiff_smag_z", False, None),
    ("smagorinski_scaling_height2", "diffusion_nml", "hdiff_smag_z2", False, None),
    ("smagorinski_scaling_height3", "diffusion_nml", "hdiff_smag_z3", False, None),
    ("smagorinski_scaling_height4", "diffusion_nml", "hdiff_smag_z4", False, None),
    ("apply_zdiffusion_t", "nonhydrostatic_nml", "l_zdiffu_t", False, None),
    ("temperature_boundary_diffusion_denominator", "gridref_nml", "denom_diffu_t", False, None),
    ("velocity_boundary_diffusion_denominator", "gridref_nml", "denom_diffu_v", False, None),
    ("shear_type", "turbdiff_nml", "itype_sher", False, diffusion.TurbulenceShearForcingType),
    ("iforcing", "run_nml", "iforcing", False, diffusion.ForcingType),
    ("a_hshr", "turbdiff_nml", "a_hshr", False, None),
]


def make_diffusion_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> diffusion.DiffusionConfig:
    kwargs: dict[str, Any] = {}
    for field, section, name, de_list, conv in _DIFFUSION_FIELDS:
        try:
            value = _extract(atm_dict, section, name, list_to_value=de_list)
        except KeyError:
            continue
        if conv is not None:
            value = conv(value)
        kwargs[field] = value
    kwargs.update(overrides)
    return diffusion.DiffusionConfig(**kwargs)


_NONHYDROSTATIC_FIELDS = [
    # (python_field, nml_section, nml_name, list_to_value, converter)
    ("itime_scheme", "nonhydrostatic_nml", "itime_scheme", False, dycore_states.TimeSteppingScheme),
    ("iadv_rhotheta", "nonhydrostatic_nml", "iadv_rhotheta", False, dycore_states.RhoThetaAdvectionType),
    ("igradp_method", "nonhydrostatic_nml", "igradp_method", False, dycore_states.HorizontalPressureDiscretizationType),
    ("rayleigh_type", "nonhydrostatic_nml", "rayleigh_type", False, constants.RayleighType),
    ("divdamp_order", "nonhydrostatic_nml", "divdamp_order", False, dycore_states.DivergenceDampingOrder),
    ("divdamp_type", "nonhydrostatic_nml", "divdamp_type", False, dycore_states.DivergenceDampingType),
    ("l_vert_nested", "run_nml", "lvert_nest", False, None),
    ("deepatmos_mode", "dynamics_nml", "ldeepatmo", False, None),
    ("iau_init", "initicon_nml", "init_mode", False, lambda v: bool(v == 5)),
    ("extra_diffu", "nonhydrostatic_nml", "lextra_diffu", False, None),
    ("rhotheta_offctr", "nonhydrostatic_nml", "rhotheta_offctr", False, None),
    ("veladv_offctr", "nonhydrostatic_nml", "veladv_offctr", False, None),
    ("fourth_order_divdamp_factor", "nonhydrostatic_nml", "divdamp_fac", False, None),
    ("fourth_order_divdamp_factor2", "nonhydrostatic_nml", "divdamp_fac2", False, None),
    ("fourth_order_divdamp_factor3", "nonhydrostatic_nml", "divdamp_fac3", False, None),
    ("fourth_order_divdamp_factor4", "nonhydrostatic_nml", "divdamp_fac4", False, None),
    ("fourth_order_divdamp_z", "nonhydrostatic_nml", "divdamp_z", False, None),
    ("fourth_order_divdamp_z2", "nonhydrostatic_nml", "divdamp_z2", False, None),
    ("fourth_order_divdamp_z3", "nonhydrostatic_nml", "divdamp_z3", False, None),
    ("fourth_order_divdamp_z4", "nonhydrostatic_nml", "divdamp_z4", False, None),
]


def make_nonhydrostatic_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> solve_nh.NonHydrostaticConfig:
    kwargs: dict[str, Any] = {}
    for field, section, name, de_list, conv in _NONHYDROSTATIC_FIELDS:
        try:
            value = _extract(atm_dict, section, name, list_to_value=de_list)
        except KeyError:
            continue
        if conv is not None:
            value = conv(value)
        kwargs[field] = value
    kwargs.update(overrides)
    return solve_nh.NonHydrostaticConfig(**kwargs)


def _convert_nudge_max_coeff(nudge_max_coeff: float) -> float:
    return constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * nudge_max_coeff


_INTERPOLATION_FIELDS = [
    # (python_field, nml_section, nml_name, list_to_value, converter)
    ("divergence_averaging_central_cell_weight", "dynamics_nml", "divavg_cntrwgt", False, None),
    ("max_nudging_coefficient", "interpol_nml", "nudge_max_coeff", False, _convert_nudge_max_coeff),
    ("nudge_efold_width", "interpol_nml", "nudge_efold_width", False, None),
    ("nudge_zone_width", "interpol_nml", "nudge_zone_width", False, None),
    ("rbf_kernel_cell", "interpol_nml", "rbf_vec_kern_c", False, InterpolationKernel),
    ("rbf_kernel_edge", "interpol_nml", "rbf_vec_kern_e", False, InterpolationKernel),
    ("rbf_kernel_vertex", "interpol_nml", "rbf_vec_kern_v", False, InterpolationKernel),
    ("lsq_high_ord", "interpol_nml", "lsq_high_ord", False, None),
]


def make_interpolation_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> interpolation_factory.InterpolationConfig:
    kwargs: dict[str, Any] = {}
    for field, section, name, de_list, conv in _INTERPOLATION_FIELDS:
        try:
            value = _extract(atm_dict, section, name, list_to_value=de_list)
        except KeyError:
            continue
        if conv is not None:
            value = conv(value)
        kwargs[field] = value
    kwargs.update(overrides)
    return interpolation_factory.InterpolationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Driver config (merged master + atm dicts, with converters)
# ---------------------------------------------------------------------------


# (field, source_key, section, name, list_to_value, converter)
# source_key is "master_cfg" or "model_cfg"
_DRIVER_FIELDS = [
    ("experiment_name", "master_cfg", "master_model_nml", "model_namelist_filename", False,
     lambda v: v.removeprefix("NAMELIST_").removesuffix("_sb_atm")),
    ("start_of_simulation", "master_cfg", "master_time_control_nml", "experimentstartdate", False,
     absolutetime_from_iconformat),
    ("start_of_timestepping", "master_cfg", "master_time_control_nml", "experimentstartdate", False,
     absolutetime_from_iconformat),
    ("end_of_simulation", "master_cfg", "master_time_control_nml", "experimentstopdate", False,
     absolutetime_from_iconformat),
    ("apply_extra_second_order_divdamp", "model_cfg", "run_nml", "ltestcase", False,
     lambda v: not v),
    ("do_prep_adv", "model_cfg", "run_nml", "ltransport", False, None),
    ("diffuse_before_time_loop", "model_cfg", "run_nml", "ltestcase", False,
     lambda v: not v),
    ("vertical_cfl_threshold", "model_cfg", "nonhydrostatic_nml", "vcfl_threshold", False, None),
    ("ndyn_substeps", "model_cfg", "nonhydrostatic_nml", "ndyn_substeps", False, None),
]


def make_driver_config(
    *, atm_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
) -> driver_config.DriverConfig:
    icon_config = {"master_cfg": master_dict, "model_cfg": atm_dict}
    kwargs: dict[str, Any] = {}
    for field, src, section, name, de_list, conv in _DRIVER_FIELDS:
        try:
            raw = icon_config[src][section][name]
            value = list_to_value(raw) if de_list else raw
            if conv is not None:
                value = conv(value)
            kwargs[field] = value
        except KeyError:
            continue

    # dtime is an IconMultiOption: modeltimestep takes priority over dtime
    try:
        run_nml = atm_dict["run_nml"]
        kwargs["dtime"] = relativetime_from_iconformat(
            run_nml["dtime"], run_nml.get("modeltimestep", "").strip()
        )
    except KeyError:
        pass

    kwargs.update(overrides)
    return driver_config.DriverConfig.make_initial(**kwargs)


# ---------------------------------------------------------------------------
# Manual config builders
# ---------------------------------------------------------------------------


def make_metrics_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> metrics_factory.MetricsConfig:
    nonhydrostatic_nml = atm_dict["nonhydrostatic_nml"]
    return metrics_factory.MetricsConfig(
        exner_expol=nonhydrostatic_nml["exner_expol"],
        vwind_offctr=nonhydrostatic_nml["vwind_offctr"],
        thslp_zdiffu=nonhydrostatic_nml["thslp_zdiffu"],
        thhgtd_zdiffu=nonhydrostatic_nml["thhgtd_zdiffu"],
        rayleigh_type=constants.RayleighType(nonhydrostatic_nml["rayleigh_type"]),
        rayleigh_coeff=list_to_value(nonhydrostatic_nml["rayleigh_coeff"]),
        divdamp_trans_start=nonhydrostatic_nml["divdamp_trans_start"],
        divdamp_trans_end=nonhydrostatic_nml["divdamp_trans_end"],
        divdamp_type=nonhydrostatic_nml["divdamp_type"],
        igradp_method=nonhydrostatic_nml["igradp_method"],
        **overrides,
    )


def make_vertical_grid_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> v_grid.VerticalGridConfig:
    sleve_nml = atm_dict["sleve_nml"]
    nonhydrostatic_nml = atm_dict["nonhydrostatic_nml"]
    run_nml = atm_dict["run_nml"]
    return v_grid.VerticalGridConfig(
        num_levels=list_to_value(run_nml["num_lev"]),
        maximal_layer_thickness=sleve_nml["max_lay_thckn"],
        top_height_limit_for_maximal_layer_thickness=sleve_nml["htop_thcknlimit"],
        lowest_layer_thickness=sleve_nml["min_lay_thckn"],
        model_top_height=sleve_nml["top_height"],
        flat_height=sleve_nml["flat_height"],
        stretch_factor=sleve_nml["stretch_fac"],
        rayleigh_damping_height=list_to_value(nonhydrostatic_nml["damp_height"]),
        htop_moist_proc=nonhydrostatic_nml["htop_moist_proc"],
        SLEVE_decay_scale_1=sleve_nml["decay_scale_1"],
        SLEVE_decay_scale_2=sleve_nml["decay_scale_2"],
        SLEVE_decay_exponent=sleve_nml["decay_exp"],
        **overrides,
    )


def make_advection_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> tracer_advection.AdvectionConfig:
    transport_nml = atm_dict["transport_nml"]
    return tracer_advection.AdvectionConfig(
        horizontal_advection_type=tracer_advection.HorizontalAdvectionType(
            list_to_value(transport_nml["ihadv_tracer"])
        ),
        horizontal_advection_limiter=tracer_advection.HorizontalAdvectionLimiter(
            list_to_value(transport_nml["itype_hlimit"])
        ),
        vertical_advection_type=tracer_advection.VerticalAdvectionType(
            list_to_value(transport_nml["ivadv_tracer"])
        ),
        vertical_advection_limiter=tracer_advection.VerticalAdvectionLimiter(
            list_to_value(transport_nml["itype_vlimit"])
        ),
        **overrides,
    )


def make_graupel_config(
    atm_dict: dict[str, Any], **overrides: Any
) -> graupel.SingleMomentSixClassIconGraupelConfig:
    run_nml = atm_dict["run_nml"]
    nwp_phy_nml = atm_dict["nwp_phy_nml"]
    nwp_tuning_nml = atm_dict["nwp_tuning_nml"]
    return graupel.SingleMomentSixClassIconGraupelConfig(
        do_latent_heat_nudging=run_nml["ldass_lhn"],
        use_constant_latent_heat=list_to_value(nwp_phy_nml["ithermo_water"]) == 0,
        ice_stickeff_min=nwp_tuning_nml["tune_zceff_min"],
        power_law_coeff_for_ice_mean_fall_speed=nwp_tuning_nml["tune_zvz0i"],
        exponent_for_density_factor_in_ice_sedimentation=nwp_tuning_nml[
            "tune_icesedi_exp"
        ],
        power_law_coeff_for_snow_fall_speed=nwp_tuning_nml["tune_v0snow"],
        rain_mu=nwp_phy_nml["mu_rain"],
        rain_n0=nwp_phy_nml["rain_n0_factor"],
        snow2graupel_riming_coeff=nwp_tuning_nml["tune_zcsg"],
        **overrides,
    )


def make_topography_config(
    *,
    atm_dict: dict[str, Any],
    input_dict: dict[str, Any],
    data_path: pathlib.Path,
):
    run_nml = atm_dict["run_nml"]
    if not run_nml["ltestcase"]:
        from icon4py.model.common.topography import from_file as from_file_topo

        return from_file_topo.FromFileConfig(
            data_path=data_path / "ser_data",
        )

    testcase_nml = input_dict.get("nh_testcase_nml", {})
    test_name = testcase_nml.get("nh_test_name")
    config: typing.Any
    from icon4py.model.common.topography.analytical import (
        flat_topography as flat_topo,
        gaussian_hill as gausshill_topo,
        jablonowski_williamson as jw_topo,
    )

    # Map Fortran namelist keys to analytical-topography dataclass fields.
    jw_topo_name_map = {"jw_u0": "u0"}

    match test_name:
        case "APE_nwp" | "APE_aes" | "wk82":
            config = flat_topo.FlatTopographyConfig()
        case "jabw" | "jabw_s":
            config = config_dataclass_from_dict(
                jw_topo.JablonowskiWilliamsonConfig, testcase_nml, jw_topo_name_map
            )
        case "gauss3D":
            config = config_dataclass_from_dict(
                gausshill_topo.GaussianHillConfig, testcase_nml, {}
            )
        case name:
            raise ValueError(f"Unknown or missing test case name: {name!r}")

    return config


def make_initial_condition_config(
    *,
    atm_dict: dict[str, Any],
    input_dict: dict[str, Any],
    data_path: pathlib.Path,
    start_of_simulation: time.AbsoluteTime,
    start_of_timestepping: time.AbsoluteTime,
    dtime: time.RelativeTime,
):
    run_nml = atm_dict["run_nml"]
    if not run_nml["ltestcase"]:
        return from_file_ic.FromFileConfig(
            data_path=data_path / "ser_data",
            start_of_simulation=start_of_simulation,
            start_of_timestepping=start_of_timestepping,
            dtime=dtime,
            ntracer=list_to_value(run_nml["ntracer"]),
        )

    testcase_nml = input_dict.get("nh_testcase_nml", {})
    test_name = testcase_nml.get("nh_test_name")
    config: typing.Any
    from icon4py.model.common.initial_condition.analytical import (
        gauss3d as gauss_ic,
        jablonowski_williamson as jw_ic,
        weisman_klemp as wk_ic,
    )

    # Map Fortran namelist keys to analytical-initial-condition dataclass fields.
    jw_ic_name_map = {
        "jw_up": "baroclinic_amplitude",
        "jw_u0": "u0",
        "jw_temp0": "temp0",
        "zp_ape": "p_sfc",
        "rh_at_1000hpa": "rh_at_1000hpa",
        "qv_max": "qv_max",
        "ztmc_ape": "global_moisture_content",
    }
    gauss3d_ic_name_map = {
        "nh_u0": "u0",
        "nh_t0": "t0",
        "nh_brunt_vais": "brunt_vais",
    }
    wk_ic_name_map = {
        "qv_max_wk": "qv_max",
        "u_infty_wk": "max_wind_speed",
        "bub_hor_width": "bubble_horizontal_width",
        "bub_ver_width": "bubble_vertical_width",
        "bubctr_lon": "bubble_center_x",
        "bubctr_lat": "bubble_center_y",
        "bubctr_z": "bubble_center_z",
        "bub_amp": "bubble_amplitude",
    }

    match test_name:
        case "jabw" | "jabw_s" | "APE_nwp" | "APE_aes":
            config = config_dataclass_from_dict(
                jw_ic.JablonowskiWilliamsonConfig, testcase_nml, jw_ic_name_map
            )
            config.normalize_global_moisture = test_name in ("APE_nwp", "APE_aes")
            if test_name == "jabw_s":
                config.baroclinic_amplitude = 0.0
        case "gauss3D":
            config = config_dataclass_from_dict(
                gauss_ic.Gauss3DConfig, testcase_nml, gauss3d_ic_name_map
            )
        case "wk82":
            config = config_dataclass_from_dict(
                wk_ic.WeismanKlempConfig, testcase_nml, wk_ic_name_map
            )
        case name:
            raise ValueError(f"Unknown or missing test case name: {name!r}")

    return config


def make_prescribed_tendencies_config(
    *,
    atm_dict: dict[str, Any],
    data_path: pathlib.Path,
) -> prescribed_tendencies.PrescribedTendenciesConfig:
    run_nml = atm_dict["run_nml"]
    if run_nml["ltestcase"]:
        return prescribed_tendencies.PrescribedTendenciesConfig(data_path=None)
    return prescribed_tendencies.PrescribedTendenciesConfig(
        data_path=data_path / "ser_data"
    )


# ---------------------------------------------------------------------------
# Orchestration (moved from driver.config.read_experiment_config_from_fortran)
# ---------------------------------------------------------------------------

#: Filenames of the Fortran namelists read by the converter.
NAMELIST_ATM_FNAME: typing.Final = "NAMELIST_ICON_output_atm"
NAMELIST_MASTER_FNAME: typing.Final = "icon_master.namelist"


def _namelist_to_dict(namelist_dir: pathlib.Path, fname: str) -> dict[str, Any]:
    nml_path = namelist_dir / fname
    if not nml_path.exists():
        raise FileNotFoundError(f"Missing namelist file: {nml_path}")
    return f90nml.read(nml_path).todict()


def _discover_experiment_namelist(namelist_dir: pathlib.Path) -> str:
    """Find the experiment-specific NAMELIST file, excluding .json and known namelists."""
    known = {NAMELIST_ATM_FNAME, NAMELIST_MASTER_FNAME, "NAMELIST_expname"}
    candidates = sorted(
        c.name
        for c in namelist_dir.glob("NAMELIST_*")
        if c.name not in known and not c.name.endswith(".json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No experiment-specific NAMELIST found in {namelist_dir} "
            f"(looked for NAMELIST_* excluding .json, "
            f"{NAMELIST_ATM_FNAME!r}, {NAMELIST_MASTER_FNAME!r})."
        )
    return candidates[0]


def convert_experiment(
    namelist_dir: pathlib.Path,
    *,
    enable_profiling: bool = False,
    enable_statistics_output: bool = False,
    namelist_expname: str | None = None,
) -> driver_config.ExperimentConfig:
    """Assemble an :class:`ExperimentConfig` from a directory of Fortran namelists.

    Parameters
    ----------
    namelist_dir:
        Directory containing ``NAMELIST_ICON_output_atm``, ``icon_master.namelist``
        and (when applicable) the experiment-specific namelist.
    enable_profiling:
        Include a :class:`ProfilingConfig` in the driver config.
    enable_statistics_output:
        Enable variable-statistics logging in the driver config.
    namelist_expname:
        Filename of the experiment-specific namelist (e.g.
        ``NAMELIST_exclaim_gauss3d_sb``).  When ``None`` the converter
        discovers it by globbing ``NAMELIST_*`` excluding ``.json`` files
        and the known atm/master namelists.
    """
    atm_dict = _namelist_to_dict(namelist_dir, NAMELIST_ATM_FNAME)
    master_dict = _namelist_to_dict(namelist_dir, NAMELIST_MASTER_FNAME)

    if namelist_expname is None:
        input_fname = _discover_experiment_namelist(namelist_dir)
    else:
        input_fname = namelist_expname
    input_dict = _namelist_to_dict(namelist_dir, input_fname)

    geometry_cfg = GeometryConfig(use_analytical_means=True)

    metrics_cfg = make_metrics_config(atm_dict)

    interpolation_cfg = make_interpolation_config(atm_dict)

    vertical_grid_cfg = make_vertical_grid_config(atm_dict)

    topography_cfg = make_topography_config(
        atm_dict=atm_dict, input_dict=input_dict, data_path=namelist_dir
    )

    nonhydro_cfg = make_nonhydrostatic_config(atm_dict)

    diffusion_cfg = make_diffusion_config(atm_dict)

    do_tracer_advection = not (
        "exclaim_ch_r04b09_dsl" in namelist_dir.name
        or "exclaim_ape_R02B04" in namelist_dir.name
    )
    tracer_advection_cfg = (
        make_advection_config(atm_dict) if do_tracer_advection else None
    )
    ntracer = list_to_value(atm_dict["run_nml"]["ntracer"]) if do_tracer_advection else 0

    aes_physics_on = "aes_phy_nml" in atm_dict
    tracer_cfg = (
        tracer_states.TracerConfig.all()
        if aes_physics_on
        else tracer_states.TracerConfig.from_ntracer(ntracer)
    )

    do_physics = "nwp_phy_nml" in atm_dict and "nwp_tuning_nml" in atm_dict
    graupel_cfg = make_graupel_config(atm_dict) if do_physics else None

    profiling_stats = driver_config.ProfilingConfig() if enable_profiling else None
    driver_cfg = make_driver_config(
        atm_dict=atm_dict,
        master_dict=master_dict,
        profiling_options=profiling_stats,
        enable_statistics_logging=enable_statistics_output,
    )

    initial_condition_cfg = make_initial_condition_config(
        atm_dict=atm_dict,
        input_dict=input_dict,
        data_path=namelist_dir,
        start_of_simulation=driver_cfg.start_of_simulation,
        start_of_timestepping=driver_cfg.start_of_timestepping,
        dtime=driver_cfg.dtime,
    )

    if not do_tracer_advection and isinstance(
        initial_condition_cfg, from_file_ic.FromFileConfig
    ):
        initial_condition_cfg = dataclasses.replace(initial_condition_cfg, ntracer=0)

    muphys_cfg: typing.Any = None
    if aes_physics_on:
        from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
            config as muphys_config,
        )

        muphys_cfg = muphys_config.MuphysConfig()

    return driver_config.ExperimentConfig(
        geometry=geometry_cfg,
        metrics=metrics_cfg,
        interpolation=interpolation_cfg,
        vertical_grid=vertical_grid_cfg,
        nonhydrostatic=nonhydro_cfg,
        diffusion=diffusion_cfg,
        tracer_config=tracer_cfg,
        tracer_advection=tracer_advection_cfg,
        graupel=graupel_cfg,
        muphys=muphys_cfg,
        topography=topography_cfg,
        initial_condition=initial_condition_cfg,
        prescribed_tendencies=make_prescribed_tendencies_config(
            atm_dict=atm_dict, data_path=namelist_dir
        ),
        driver=driver_cfg,
    )
