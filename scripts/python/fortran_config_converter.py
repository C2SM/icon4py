# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert Fortran namelist files into an icon4py :class:`ExperimentConfig`.

Reads the namelists of an ICON experiment directly with ``f90nml`` and assembles a
:class:`icon4py.model.driver.config.ExperimentConfig` that can be serialized to YAML
with :func:`icon4py.model.common.config.config_io.write_yaml_str`.

The Fortran-to-ICON4Py option mapping lives here, in the tables below, and nowhere in
the model packages.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
import typing
from typing import Any

import f90nml

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys import config as muphys_config
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common import constants, prescribed_tendencies, time
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.initial_condition.analytical import (
    gauss3d as gauss_ic,
    jablonowski_williamson as jw_ic,
    weisman_klemp as wk_ic,
)
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_states
from icon4py.model.common.topography import from_file as from_file_topo
from icon4py.model.common.topography.analytical import (
    flat_topography as flat_topo,
    gaussian_hill as gausshill_topo,
    jablonowski_williamson as jw_topo,
)
from icon4py.model.driver import config as driver_config
from icon4py.model.testing import definitions as test_defs


log = logging.getLogger(__name__)

NAMELIST_ATM_FNAME: typing.Final = "NAMELIST_ICON_output_atm"
NAMELIST_MASTER_FNAME: typing.Final = "icon_master.namelist"

# Paths to serialized data are written relative to the namelist directory, so that
# the generated config stays valid when the archive is moved to another machine
# (see `driver.config.read_experiment_config_from_yaml`).
SER_DATA_PATH: typing.Final = pathlib.Path(test_defs.SERIALIZED_DATA_SUBDIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_to_value[T](obj: list[T] | T) -> T:
    # Some parameters are allocated as `max_dom`-sized lists, with one value
    # per domain. ICON4Py (for now) only runs on one domain.
    # Most parameters have the same value for all elements, others (such as
    # num_levels) have a default value different from domain[0].
    # TODO (ricoh,jcanton): stop using this for per-tracer values when enabling
    # that functionality Tracers are an even different case where there is one
    # value per tracer, but with the current version of ICON4Py all tracers get
    # the same config.
    return obj[0] if isinstance(obj, list) else obj


@dataclasses.dataclass(frozen=True)
class IconOption:
    """Where to find a config field in the ICON namelists and how to convert it."""

    #: ICON4Py config field name
    field: str
    #: path through nested namelist sections, ending with the option name, or with
    #: a section when `converter` combines several of its options
    path: tuple[str, ...]
    #: take the first element of a `max_dom`-sized list, see `list_to_value`
    list_to_value: bool = False
    #: applied to the namelist value; defaults to the type annotation of the field
    converter: typing.Callable[[Any], Any] | None = None
    #: position within an unnamed (positional) namelist record.
    #: Derived-type namelists (e.g. the AES physics `aes_*_nml`) are echoed by ICON as an
    #: anonymous array of the member values in declaration order, one record per domain.
    #: For these, `path` leads to that array and `unnamed_index` is the 0-based member
    #: position within the first record, while the option name only serves as documentation.
    unnamed_index: int | None = None
    #: when False, a missing namelist entry leaves the ICON4Py default in place
    required: bool = True


def _field_type(config_cls: type, field: str) -> typing.Callable[[Any], Any]:
    """Base type of a (possibly `Annotated` or `Final`) dataclass field, used as fallback converter."""
    hint = typing.get_type_hints(config_cls, include_extras=True)[field]
    while typing.get_origin(hint) in (typing.Annotated, typing.Final):
        hint = typing.get_args(hint)[0]
    return hint


@dataclasses.dataclass(frozen=True)
class ConfigMapping[T]:
    """The ICON namelist options a config class is read from."""

    config_cls: type[T]
    options: list[IconOption]

    def build(self, icon_config: dict[str, Any], **overrides: Any) -> T:
        """Construct `config_cls` from the `options` in `icon_config`, plus `overrides`.

        Options are required by default: ICON dumps every namelist in full, so a
        missing key means the mapping drifted from Fortran and must not be silently
        replaced by the ICON4Py default.
        """
        kwargs: dict[str, Any] = {}
        for opt in self.options:
            value: Any = icon_config
            try:
                for key in opt.path:
                    value = value[key]
            except KeyError:
                if opt.required:
                    raise
                continue
            if opt.unnamed_index is not None:
                value = value[opt.unnamed_index]
            if opt.list_to_value:
                value = list_to_value(value)
            converter = opt.converter or _field_type(self.config_cls, opt.field)
            kwargs[opt.field] = converter(value)
        return self.config_cls(**kwargs, **overrides)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


DIFFUSION = ConfigMapping(
    diffusion.DiffusionConfig,
    [
        IconOption("diffusion_type", ("diffusion_nml", "hdiff_order")),
        IconOption("apply_to_vertical_wind", ("diffusion_nml", "lhdiff_w")),
        IconOption("apply_to_horizontal_wind", ("diffusion_nml", "lhdiff_vn")),
        IconOption("apply_to_temperature", ("diffusion_nml", "lhdiff_temp")),
        IconOption(
            "apply_smag_diff_to_vertical_wind",
            ("diffusion_nml", "lhdiff_smag_w"),
            list_to_value=True,
        ),
        IconOption("compute_3d_smag_coeff", ("diffusion_nml", "lsmag_3d"), list_to_value=True),
        IconOption("type_vn_diffu", ("diffusion_nml", "itype_vn_diffu")),
        IconOption("type_t_diffu", ("diffusion_nml", "itype_t_diffu")),
        IconOption("hdiff_efdt_ratio", ("diffusion_nml", "hdiff_efdt_ratio")),
        IconOption("hdiff_w_efdt_ratio", ("diffusion_nml", "hdiff_w_efdt_ratio")),
        IconOption("smagorinski_scaling_factor", ("diffusion_nml", "hdiff_smag_fac")),
        IconOption("smagorinski_scaling_factor2", ("diffusion_nml", "hdiff_smag_fac2")),
        IconOption("smagorinski_scaling_factor3", ("diffusion_nml", "hdiff_smag_fac3")),
        IconOption("smagorinski_scaling_factor4", ("diffusion_nml", "hdiff_smag_fac4")),
        IconOption("smagorinski_scaling_height", ("diffusion_nml", "hdiff_smag_z")),
        IconOption("smagorinski_scaling_height2", ("diffusion_nml", "hdiff_smag_z2")),
        IconOption("smagorinski_scaling_height3", ("diffusion_nml", "hdiff_smag_z3")),
        IconOption("smagorinski_scaling_height4", ("diffusion_nml", "hdiff_smag_z4")),
        IconOption("apply_zdiffusion_t", ("nonhydrostatic_nml", "l_zdiffu_t")),
        IconOption("temperature_boundary_diffusion_denominator", ("gridref_nml", "denom_diffu_t")),
        IconOption("velocity_boundary_diffusion_denominator", ("gridref_nml", "denom_diffu_v")),
        IconOption("shear_type", ("turbdiff_nml", "itype_sher")),
        IconOption("iforcing", ("run_nml", "iforcing")),
        IconOption("a_hshr", ("turbdiff_nml", "a_hshr")),
    ],
)


NONHYDROSTATIC = ConfigMapping(
    solve_nh.NonHydrostaticConfig,
    [
        IconOption("itime_scheme", ("nonhydrostatic_nml", "itime_scheme")),
        IconOption("iadv_rhotheta", ("nonhydrostatic_nml", "iadv_rhotheta")),
        IconOption("igradp_method", ("nonhydrostatic_nml", "igradp_method")),
        IconOption("rayleigh_type", ("nonhydrostatic_nml", "rayleigh_type")),
        IconOption("divdamp_order", ("nonhydrostatic_nml", "divdamp_order")),
        IconOption("divdamp_type", ("nonhydrostatic_nml", "divdamp_type")),
        IconOption("l_vert_nested", ("run_nml", "lvert_nest")),
        IconOption("deepatmos_mode", ("dynamics_nml", "ldeepatmo")),
        IconOption(
            "iau_init", ("initicon_nml", "init_mode"), converter=lambda init_mode: init_mode == 5
        ),
        IconOption("extra_diffu", ("nonhydrostatic_nml", "lextra_diffu")),
        IconOption("rhotheta_offctr", ("nonhydrostatic_nml", "rhotheta_offctr")),
        IconOption("veladv_offctr", ("nonhydrostatic_nml", "veladv_offctr")),
        IconOption("fourth_order_divdamp_factor", ("nonhydrostatic_nml", "divdamp_fac")),
        IconOption("fourth_order_divdamp_factor2", ("nonhydrostatic_nml", "divdamp_fac2")),
        IconOption("fourth_order_divdamp_factor3", ("nonhydrostatic_nml", "divdamp_fac3")),
        IconOption("fourth_order_divdamp_factor4", ("nonhydrostatic_nml", "divdamp_fac4")),
        IconOption("fourth_order_divdamp_z", ("nonhydrostatic_nml", "divdamp_z")),
        IconOption("fourth_order_divdamp_z2", ("nonhydrostatic_nml", "divdamp_z2")),
        IconOption("fourth_order_divdamp_z3", ("nonhydrostatic_nml", "divdamp_z3")),
        IconOption("fourth_order_divdamp_z4", ("nonhydrostatic_nml", "divdamp_z4")),
    ],
)


def _convert_nudge_max_coeff(nudge_max_coeff: float) -> float:
    return constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * nudge_max_coeff


INTERPOLATION = ConfigMapping(
    interpolation_factory.InterpolationConfig,
    [
        IconOption("divergence_averaging_central_cell_weight", ("dynamics_nml", "divavg_cntrwgt")),
        IconOption(
            "max_nudging_coefficient",
            ("interpol_nml", "nudge_max_coeff"),
            converter=_convert_nudge_max_coeff,
        ),
        IconOption("nudge_efold_width", ("interpol_nml", "nudge_efold_width")),
        IconOption("nudge_zone_width", ("interpol_nml", "nudge_zone_width")),
        IconOption("rbf_kernel_cell", ("interpol_nml", "rbf_vec_kern_c")),
        IconOption("rbf_kernel_edge", ("interpol_nml", "rbf_vec_kern_e")),
        IconOption("rbf_kernel_vertex", ("interpol_nml", "rbf_vec_kern_v")),
        IconOption("lsq_high_ord", ("interpol_nml", "lsq_high_ord")),
    ],
)


def _experiment_name_from_namelist_filename(model_namelist_filename: str) -> str:
    return model_namelist_filename.removeprefix("NAMELIST_").removesuffix("_sb_atm")


def _dtime_from_run_nml(run_nml: dict[str, Any]) -> time.RelativeTime:
    # `modeltimestep` (ISO 8601 duration) takes priority over the legacy `dtime`
    # (seconds); ICON writes it as a fixed-width, blank-padded string.
    return driver_config.relativetime_from_iconformat(
        run_nml["dtime"], run_nml["modeltimestep"].strip()
    )


# The driver reads from both the master and the model namelists:
# `{"master_cfg": master_dict, "model_cfg": atm_dict}`.
DRIVER = ConfigMapping(
    driver_config.DriverConfig,
    [
        IconOption(
            "experiment_name",
            ("master_cfg", "master_model_nml", "model_namelist_filename"),
            converter=_experiment_name_from_namelist_filename,
        ),
        IconOption("dtime", ("model_cfg", "run_nml"), converter=_dtime_from_run_nml),
        IconOption(
            "start_of_simulation",
            ("master_cfg", "master_time_control_nml", "experimentstartdate"),
            converter=driver_config.absolutetime_from_iconformat,
        ),
        # start_of_timestepping is always equal to start_of_simulation when reading from ICON
        IconOption(
            "start_of_timestepping",
            ("master_cfg", "master_time_control_nml", "experimentstartdate"),
            converter=driver_config.absolutetime_from_iconformat,
        ),
        IconOption(
            "end_of_simulation",
            ("master_cfg", "master_time_control_nml", "experimentstopdate"),
            converter=driver_config.absolutetime_from_iconformat,
        ),
        # Not a namelist variable, coded as follows in mo_nh_stepping.f90:
        # IF (elapsed_time_global <= 7200._wp+0.5_wp*dtime .AND. .NOT. ltestcase)
        IconOption(
            "apply_extra_second_order_divdamp",
            ("model_cfg", "run_nml", "ltestcase"),
            converter=lambda ltestcase: not ltestcase,
        ),
        IconOption(
            "diffuse_before_time_loop",
            ("model_cfg", "run_nml", "ltestcase"),
            converter=lambda ltestcase: not ltestcase,
        ),
        IconOption("vertical_cfl_threshold", ("model_cfg", "nonhydrostatic_nml", "vcfl_threshold")),
        IconOption("ndyn_substeps", ("model_cfg", "nonhydrostatic_nml", "ndyn_substeps")),
    ],
)


METRICS = ConfigMapping(
    metrics_factory.MetricsConfig,
    [
        IconOption("exner_expol", ("nonhydrostatic_nml", "exner_expol")),
        IconOption("vwind_offctr", ("nonhydrostatic_nml", "vwind_offctr")),
        IconOption("thslp_zdiffu", ("nonhydrostatic_nml", "thslp_zdiffu")),
        IconOption("thhgtd_zdiffu", ("nonhydrostatic_nml", "thhgtd_zdiffu")),
        IconOption("rayleigh_type", ("nonhydrostatic_nml", "rayleigh_type")),
        IconOption("rayleigh_coeff", ("nonhydrostatic_nml", "rayleigh_coeff"), list_to_value=True),
        IconOption("divdamp_trans_start", ("nonhydrostatic_nml", "divdamp_trans_start")),
        IconOption("divdamp_trans_end", ("nonhydrostatic_nml", "divdamp_trans_end")),
        IconOption("divdamp_type", ("nonhydrostatic_nml", "divdamp_type")),
        IconOption("igradp_method", ("nonhydrostatic_nml", "igradp_method")),
    ],
)


VERTICAL_GRID = ConfigMapping(
    v_grid.VerticalGridConfig,
    [
        IconOption("num_levels", ("run_nml", "num_lev"), list_to_value=True),
        IconOption("maximal_layer_thickness", ("sleve_nml", "max_lay_thckn")),
        IconOption(
            "top_height_limit_for_maximal_layer_thickness", ("sleve_nml", "htop_thcknlimit")
        ),
        IconOption("lowest_layer_thickness", ("sleve_nml", "min_lay_thckn")),
        IconOption("model_top_height", ("sleve_nml", "top_height")),
        IconOption("flat_height", ("sleve_nml", "flat_height")),
        IconOption("stretch_factor", ("sleve_nml", "stretch_fac")),
        IconOption(
            "rayleigh_damping_height", ("nonhydrostatic_nml", "damp_height"), list_to_value=True
        ),
        IconOption("htop_moist_proc", ("nonhydrostatic_nml", "htop_moist_proc")),
        IconOption("SLEVE_decay_scale_1", ("sleve_nml", "decay_scale_1")),
        IconOption("SLEVE_decay_scale_2", ("sleve_nml", "decay_scale_2")),
        IconOption("SLEVE_decay_exponent", ("sleve_nml", "decay_exp")),
    ],
)


ADVECTION = ConfigMapping(
    tracer_advection.AdvectionConfig,
    [
        IconOption(
            "horizontal_advection_type", ("transport_nml", "ihadv_tracer"), list_to_value=True
        ),
        IconOption(
            "horizontal_advection_limiter", ("transport_nml", "itype_hlimit"), list_to_value=True
        ),
        IconOption(
            "vertical_advection_type", ("transport_nml", "ivadv_tracer"), list_to_value=True
        ),
        IconOption(
            "vertical_advection_limiter", ("transport_nml", "itype_vlimit"), list_to_value=True
        ),
    ],
)


GRAUPEL = ConfigMapping(
    graupel.SingleMomentSixClassIconGraupelConfig,
    [
        IconOption("do_latent_heat_nudging", ("run_nml", "ldass_lhn")),
        IconOption(
            "use_constant_latent_heat",
            ("nwp_phy_nml", "ithermo_water"),
            list_to_value=True,
            converter=lambda ithermo_water: ithermo_water == 0,
        ),
        IconOption("ice_stickeff_min", ("nwp_tuning_nml", "tune_zceff_min")),
        IconOption("power_law_coeff_for_ice_mean_fall_speed", ("nwp_tuning_nml", "tune_zvz0i")),
        IconOption(
            "exponent_for_density_factor_in_ice_sedimentation",
            ("nwp_tuning_nml", "tune_icesedi_exp"),
        ),
        IconOption("power_law_coeff_for_snow_fall_speed", ("nwp_tuning_nml", "tune_v0snow")),
        IconOption("rain_mu", ("nwp_phy_nml", "mu_rain")),
        IconOption("rain_n0", ("nwp_phy_nml", "rain_n0_factor")),
        IconOption("snow2graupel_riming_coeff", ("nwp_tuning_nml", "tune_zcsg")),
    ],
)


# ICON echoes `aes_vdf_nml` as an anonymous array of the `t_vdiff_config` members in
# declaration order, one record per domain; each option is pinned to its member position.
# Keep `_TMX_VDIFF_MEMBERS`, `_TMX_USE_TMX_INDEX` and the indices below in sync with
# `t_vdiff_config` in `mo_turb_vdiff_config.f90`.
_TMX_VDIFF_PATH: typing.Final = ("aes_vdf_nml", "aes_vdf_config")
#: number of members of `t_vdiff_config`, i.e. the length of one domain's record
_TMX_VDIFF_MEMBERS: typing.Final = 42
#: position of the `use_tmx` switch within a record
_TMX_USE_TMX_INDEX: typing.Final = 22

TMX = ConfigMapping(
    tmx_config.TmxConfig,
    [
        IconOption("solver_type", _TMX_VDIFF_PATH, unnamed_index=23),
        IconOption("energy_type", _TMX_VDIFF_PATH, unnamed_index=24),
        IconOption("dissipation_factor", _TMX_VDIFF_PATH, unnamed_index=25),
        IconOption("use_louis", _TMX_VDIFF_PATH, unnamed_index=26),
        IconOption("use_louis_land", _TMX_VDIFF_PATH, unnamed_index=27),
        IconOption("use_louis_ice", _TMX_VDIFF_PATH, unnamed_index=28),
        IconOption("louis_constant_b", _TMX_VDIFF_PATH, unnamed_index=29),
        IconOption("use_km_const", _TMX_VDIFF_PATH, unnamed_index=30),
        IconOption("km_const", _TMX_VDIFF_PATH, unnamed_index=31),
        IconOption("use_scale_turb_energy_flux", _TMX_VDIFF_PATH, unnamed_index=32),
        IconOption("scale_turb_energy_flux", _TMX_VDIFF_PATH, unnamed_index=33),
        IconOption("smag_constant", _TMX_VDIFF_PATH, unnamed_index=34),
        IconOption("turb_prandtl", _TMX_VDIFF_PATH, unnamed_index=35),
        IconOption("km_min", _TMX_VDIFF_PATH, unnamed_index=37),
        IconOption("max_turb_scale", _TMX_VDIFF_PATH, unnamed_index=38),
    ],
)


def _read_use_tmx(atm_dict: dict[str, Any]) -> bool:
    """Read the `use_tmx` switch, checking the layout the pinned positions rely on.

    The options are located positionally, so a change to `t_vdiff_config` must fail
    loudly rather than silently shift every value by one member.
    """
    flat = atm_dict["aes_vdf_nml"]["aes_vdf_config"]
    if len(flat) % _TMX_VDIFF_MEMBERS != 0:
        raise ValueError(
            f"'aes_vdf_config' has {len(flat)} values, not a multiple of the "
            f"{_TMX_VDIFF_MEMBERS} members of t_vdiff_config: the Fortran type changed "
            "and the pinned 'unnamed_index' positions must be revised."
        )
    use_tmx = flat[_TMX_USE_TMX_INDEX]
    if not isinstance(use_tmx, bool):
        raise ValueError(
            f"expected the 'use_tmx' switch at position {_TMX_USE_TMX_INDEX} of "
            f"'aes_vdf_config', found {use_tmx!r}: the t_vdiff_config member order "
            "changed and the pinned 'unnamed_index' positions must be revised."
        )
    return use_tmx


def tmx_is_active(atm_dict: dict[str, Any]) -> bool:
    """Whether the experiment ran the tmx turbulent mixing scheme."""
    return "aes_vdf_nml" in atm_dict and _read_use_tmx(atm_dict)


# The analytical test cases are read from `nh_testcase_nml`, which ICON does not dump
# to NAMELIST_ICON_output_atm: it comes from the experiment's input namelist, which
# only lists the values that differ from the ICON defaults.
_testcase_option = functools.partial(IconOption, required=False)

JW_TOPOGRAPHY = ConfigMapping(
    jw_topo.JablonowskiWilliamsonConfig,
    [_testcase_option("u0", ("jw_u0",))],
)

GAUSSIAN_HILL_TOPOGRAPHY = ConfigMapping(
    gausshill_topo.GaussianHillConfig,
    [
        _testcase_option("mount_height", ("mount_height",)),
        _testcase_option("mount_width", ("mount_width",)),
    ],
)

JW_INITIAL_CONDITION = ConfigMapping(
    jw_ic.JablonowskiWilliamsonConfig,
    [
        _testcase_option("baroclinic_amplitude", ("jw_up",)),
        _testcase_option("u0", ("jw_u0",)),
        _testcase_option("temp0", ("jw_temp0",)),
        _testcase_option("p_sfc", ("zp_ape",)),
        _testcase_option("rh_at_1000hpa", ("rh_at_1000hpa",)),
        _testcase_option("qv_max", ("qv_max",)),
        _testcase_option("global_moisture_content", ("ztmc_ape",)),
        # Only the APE cases rescale qv to a prescribed global moisture content.
        IconOption(
            "normalize_global_moisture",
            ("nh_test_name",),
            converter=lambda nh_test_name: nh_test_name in ("APE_nwp", "APE_aes"),
        ),
    ],
)

GAUSS3D_INITIAL_CONDITION = ConfigMapping(
    gauss_ic.Gauss3DConfig,
    [
        _testcase_option("u0", ("nh_u0",)),
        _testcase_option("t0", ("nh_t0",)),
        _testcase_option("brunt_vais", ("nh_brunt_vais",)),
    ],
)

WK_INITIAL_CONDITION = ConfigMapping(
    wk_ic.WeismanKlempConfig,
    [
        _testcase_option("qv_max", ("qv_max_wk",)),
        _testcase_option("max_wind_speed", ("u_infty_wk",)),
        _testcase_option("bubble_horizontal_width", ("bub_hor_width",)),
        _testcase_option("bubble_vertical_width", ("bub_ver_width",)),
        _testcase_option("bubble_center_x", ("bubctr_lon",)),
        _testcase_option("bubble_center_y", ("bubctr_lat",)),
        _testcase_option("bubble_center_z", ("bubctr_z",)),
        _testcase_option("bubble_amplitude", ("bub_amp",)),
    ],
)


def make_topography_config(
    *,
    atm_dict: dict[str, Any],
    input_dict: dict[str, Any],
) -> (
    from_file_topo.FromFileConfig
    | flat_topo.FlatTopographyConfig
    | jw_topo.JablonowskiWilliamsonConfig
    | gausshill_topo.GaussianHillConfig
):
    if not atm_dict["run_nml"]["ltestcase"]:
        log.info("Reading topography from file")
        return from_file_topo.FromFileConfig(data_path=SER_DATA_PATH)

    testcase_nml = input_dict.get("nh_testcase_nml", {})
    test_name = testcase_nml.get("nh_test_name")
    match test_name:
        case "APE_nwp" | "APE_aes" | "wk82":
            log.info("Flat topography")
            return flat_topo.FlatTopographyConfig()
        case "jabw" | "jabw_s":
            log.info("Analytical topography for Jablonowski-Williamson test case")
            return JW_TOPOGRAPHY.build(testcase_nml)
        case "gauss3D":
            log.info("Analytical Gaussian hill topography")
            return GAUSSIAN_HILL_TOPOGRAPHY.build(testcase_nml)
        case name:
            raise ValueError(f"Unknown or missing test case name: {name!r}")


def make_initial_condition_config(
    *,
    atm_dict: dict[str, Any],
    input_dict: dict[str, Any],
    start_of_simulation: time.AbsoluteTime,
    start_of_timestepping: time.AbsoluteTime,
    dtime: time.RelativeTime,
) -> (
    from_file_ic.FromFileConfig
    | jw_ic.JablonowskiWilliamsonConfig
    | gauss_ic.Gauss3DConfig
    | wk_ic.WeismanKlempConfig
):
    run_nml = atm_dict["run_nml"]
    if not run_nml["ltestcase"]:
        log.info("Reading initial condition from file")
        return from_file_ic.FromFileConfig(
            data_path=SER_DATA_PATH,
            start_of_simulation=start_of_simulation,
            start_of_timestepping=start_of_timestepping,
            dtime=dtime,
            ntracer=list_to_value(run_nml["ntracer"]),
        )

    testcase_nml = input_dict.get("nh_testcase_nml", {})
    test_name = testcase_nml.get("nh_test_name")
    match test_name:
        case "jabw" | "jabw_s" | "APE_nwp" | "APE_aes":
            log.info("Analytical initial condition for Jablonowski-Williamson test case")
            config = JW_INITIAL_CONDITION.build(testcase_nml)
            # Fortran resets jw_up to 0 only for jabw_s; other cases keep the default (1.0).
            if test_name == "jabw_s":
                config.baroclinic_amplitude = 0.0
            return config
        case "gauss3D":
            log.info("Analytical initial condition for Gauss 3D test case")
            return GAUSS3D_INITIAL_CONDITION.build(testcase_nml)
        case "wk82":
            log.info("Analytical initial condition for Weisman-Klemp test case")
            return WK_INITIAL_CONDITION.build(testcase_nml)
        case name:
            raise ValueError(f"Unknown or missing test case name: {name!r}")


def make_prescribed_tendencies_config(
    atm_dict: dict[str, Any],
) -> prescribed_tendencies.PrescribedTendenciesConfig:
    if atm_dict["run_nml"]["ltestcase"]:
        return prescribed_tendencies.PrescribedTendenciesConfig(data_path=None)
    return prescribed_tendencies.PrescribedTendenciesConfig(data_path=SER_DATA_PATH)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _namelist_to_dict(namelist_dir: pathlib.Path, fname: str) -> dict[str, Any]:
    nml_path = namelist_dir / fname
    if not nml_path.exists():
        raise FileNotFoundError(f"Missing namelist file: {nml_path}")
    return f90nml.read(nml_path).todict()


def _discover_experiment_namelist(namelist_dir: pathlib.Path) -> str:
    """Find the experiment-specific namelist: the `NAMELIST_*` file that is not the atm one."""
    candidates = sorted(
        c.name
        for c in namelist_dir.glob("NAMELIST_*")
        if c.name != NAMELIST_ATM_FNAME and c.suffix != ".json"
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one experiment-specific NAMELIST_* in {namelist_dir} "
            f"(besides {NAMELIST_ATM_FNAME!r}), found {candidates}. "
            "Pass `namelist_expname` explicitly."
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
        and the experiment-specific namelist.
    enable_profiling:
        Include a :class:`ProfilingConfig` in the driver config.
    enable_statistics_output:
        Enable variable-statistics logging in the driver config.
    namelist_expname:
        Filename of the experiment-specific namelist (e.g.
        ``NAMELIST_exclaim_gauss3d_sb``). When ``None`` it is discovered as the
        only other ``NAMELIST_*`` file in ``namelist_dir``.
    """
    atm_dict = _namelist_to_dict(namelist_dir, NAMELIST_ATM_FNAME)
    master_dict = _namelist_to_dict(namelist_dir, NAMELIST_MASTER_FNAME)
    input_fname = (
        _discover_experiment_namelist(namelist_dir)
        if namelist_expname is None
        else namelist_expname
    )
    input_dict = _namelist_to_dict(namelist_dir, input_fname)

    geometry_cfg = GeometryConfig(use_analytical_means=True)
    metrics_cfg = METRICS.build(atm_dict)
    interpolation_cfg = INTERPOLATION.build(atm_dict)
    vertical_grid_cfg = VERTICAL_GRID.build(atm_dict)
    topography_cfg = make_topography_config(atm_dict=atm_dict, input_dict=input_dict)
    nonhydro_cfg = NONHYDROSTATIC.build(atm_dict)
    diffusion_cfg = DIFFUSION.build(atm_dict)

    do_tracer_advection = not (
        "exclaim_ch_r04b09_dsl" in namelist_dir.name or "exclaim_ape_R02B04" in namelist_dir.name
    )
    # The driver supplies advection's inputs (airmass and the mass fluxes the dycore
    # accumulates over the substeps), and exclaim_ape_aesPhys runs tracer advection:
    # the driver test validates transport+muphys against the end-of-time-step
    # reference (hydrometeors bit-exact, see the test_driver docstring).
    # The two experiments above stay disabled until their runs are validated the same
    # way (their datatests do not compare tracers yet).
    # TODO (jcanton): this isn't the right place to keep a special case
    # handling. Either fix these experiments or move the special case handling.
    tracer_advection_cfg = ADVECTION.build(atm_dict) if do_tracer_advection else None
    ntracer = list_to_value(atm_dict["run_nml"]["ntracer"]) if do_tracer_advection else 0

    # AES physics implies muphys is active for the experiments we support today; the presence
    # of the aes_phy_nml namelist mirrors the graupel `do_physics` check below. A robust
    # dt_mig>0 check needs the raw namelist (see docs/2026-07-22-muphys-namelist-dt-mig-gate.md).
    aes_physics_on = "aes_phy_nml" in atm_dict
    tracer_cfg = (
        tracer_states.TracerConfig.all()
        if aes_physics_on
        else tracer_states.TracerConfig.from_ntracer(ntracer)
    )

    # If these two namelists are missing it means that the experiment was run
    # without microphysics and we have to skip parsing the graupel config which
    # relies on some of these parameters.
    do_physics = "nwp_phy_nml" in atm_dict and "nwp_tuning_nml" in atm_dict
    graupel_cfg = GRAUPEL.build(atm_dict) if do_physics else None

    profiling_stats = driver_config.ProfilingConfig() if enable_profiling else None
    driver_cfg = DRIVER.build(
        {"master_cfg": master_dict, "model_cfg": atm_dict},
        profiling_options=profiling_stats,
        enable_statistics_logging=enable_statistics_output,
    )

    # the file-based initial condition needs the clock of the driver to know which
    # savepoint to read: the initial state, or a later one when restarting
    initial_condition_cfg = make_initial_condition_config(
        atm_dict=atm_dict,
        input_dict=input_dict,
        start_of_simulation=driver_cfg.start_of_simulation,
        start_of_timestepping=driver_cfg.start_of_timestepping,
        dtime=driver_cfg.dtime,
    )
    if not do_tracer_advection and isinstance(initial_condition_cfg, from_file_ic.FromFileConfig):
        initial_condition_cfg = dataclasses.replace(initial_condition_cfg, ntracer=0)

    muphys_cfg = muphys_config.MuphysConfig() if aes_physics_on else None

    # tmx is configured by the AES vertical-diffusion namelist; the driver does not run
    # the granule yet (icon4py#1360), but the config travels with the experiment.
    tmx_cfg = TMX.build(atm_dict) if tmx_is_active(atm_dict) else None

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
        tmx=tmx_cfg,
        topography=topography_cfg,
        initial_condition=initial_condition_cfg,
        prescribed_tendencies=make_prescribed_tendencies_config(atm_dict),
        driver=driver_cfg,
    )
