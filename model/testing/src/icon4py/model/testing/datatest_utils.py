# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
import functools
import json
import pathlib
import urllib.parse
from typing import TypeVar

import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.diffusion import config as diffusion_config
from icon4py.model.atmosphere.dycore import config as dycore_config, dycore_states
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import constants
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_factory, rbf_interpolation as rbf
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.standalone_driver import config as driver_config
from icon4py.model.testing import data_handling, definitions, serialbox


def get_process_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_process_properties(run_instance)


def get_experiment_name_with_version(
    experiment_description: definitions.ExperimentDescription,
) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment_description.name}_v{experiment_description.version:02d}"


def get_ranked_experiment_name_with_version(
    experiment_description: definitions.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{get_experiment_name_with_version(experiment_description)}"


def get_experiment_archive_filename(
    experiment_description: definitions.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{get_ranked_experiment_name_with_version(experiment_description, comm_size)}.tar.gz"


def get_experiment_archive_url(root_url: str, filepath: str) -> str:
    """Build a download URL for experiment archive from root URL."""
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_grid_archive_filename(grid: definitions.GridDescription) -> str:
    return f"{grid.name}.tar.gz"


def get_grid_filename(grid: definitions.GridDescription) -> str:
    return f"{grid.name}.nc"


def get_grid_filepath(grid: definitions.GridDescription) -> pathlib.Path:
    return definitions.grids_path().joinpath(grid.name, get_grid_filename(grid))


def get_grid_archive_url(root_url: str, grid: definitions.GridDescription) -> str:
    """Build a download URL for a grid archive from root URL."""
    filepath = f"{definitions.GRID_DATA_DIR}/{get_grid_archive_filename(grid)}"
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_muphys_archive_url(root_url: str, experiment_type: str, experiment_name: str) -> str:
    """Build a download URL for a muphys archive from root URL."""
    filepath = f"{definitions.MUPHYS_DATA_DIR}/{experiment_type}/{experiment_name}.tar.gz"
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_datapath_for_experiment(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment."""

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment_description,
        process_props.comm_size,
    )
    return definitions.serialized_data_path().joinpath(
        experiment_dir, definitions.SERIALIZED_DATA_SUBDIR
    )


def create_icon_serial_data_provider(
    datapath: pathlib.Path,
    rank: int,
    backend: gtx_typing.Backend | None,
) -> serialbox.IconSerialDataProvider:
    return serialbox.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=rank,
        do_print=True,
    )


def download_experiment(
    experiment_description: definitions.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> None:
    """Download data and config for an experiment--if not already present."""
    comm_size = processor_props.comm_size
    root_url = definitions.TESTDATA_ROOT_URL
    archive_filename = get_experiment_archive_filename(experiment_description, comm_size)
    archive_path = definitions.EXPERIMENT_DATA_DIR + "/" + archive_filename
    uri = get_experiment_archive_url(root_url, archive_path)
    destination_path = get_datapath_for_experiment(experiment_description, processor_props)
    data_handling.download_test_data(destination_path.parent, uri)


@functools.cache
def _read_namelist_json(json_file_path: pathlib.Path) -> dict:
    """
    Read and cache the namelist JSON file.

    Args:
        json_file_path: Path to the NAMELIST_ICON_output_atm.json file

    Returns:
        Dictionary containing the parsed JSON data
    """
    with json_file_path.open() as f:
        return json.load(f)


_T = TypeVar("_T")


def _list_to_value(obj: list[_T] | _T) -> _T:
    # Some parameters are allocated as `max_dom`-sized lists, with one value
    # per domain. ICON4Py tests (for now) only run on one domain.
    # Most parameters have the same value for all elements, others (such as
    # num_levels) have a default value different from domain[0].
    return obj[0] if isinstance(obj, list) else obj


def create_experiment_configuration(
    experiment_description: definitions.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> definitions.ExperimentConfig:
    """
    Create configuration objects from the experiment's namelist JSON file.

    This function reads the NAMELIST_ICON_output_atm.json file that comes with
    the serialized experiment data and constructs configuration objects.
    The JSON file is cached so it's only read once per unique path.

    Args:
        experiment: The experiment definition
        processor_props: Processor properties containing comm_size

    Returns:
        ExperimentConfig object containing the configuration for the experiment
    """

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment_description,
        processor_props.comm_size,
    )
    json_file_path = (
        definitions.serialized_data_path()
        / experiment_dir
        / f"{definitions.NAMELIST_ATM_FNAME}.json"
    )

    nml_data = _read_namelist_json(json_file_path)

    gridref_nml = nml_data["gridref_nml"]
    sleve_nml = nml_data["sleve_nml"]
    nonhydrostatic_nml = nml_data["nonhydrostatic_nml"]
    interpol_nml = nml_data["interpol_nml"]
    diffusion_nml = nml_data["diffusion_nml"]
    turbdiff_nml = nml_data["turbdiff_nml"]
    run_nml = nml_data["run_nml"]
    nwp_phy_nml = nml_data.get("nwp_phy_nml")
    nwp_tuning_nml = nml_data.get("nwp_tuning_nml")

    # *** MetricsConfig ***
    rayleigh_coeff = _list_to_value(nonhydrostatic_nml["rayleigh_coeff"])

    metrics_config = metrics_factory.MetricsConfig(
        exner_expol=nonhydrostatic_nml["exner_expol"],
        vwind_offctr=nonhydrostatic_nml["vwind_offctr"],
        thslp_zdiffu=nonhydrostatic_nml["thslp_zdiffu"],
        thhgtd_zdiffu=nonhydrostatic_nml["thhgtd_zdiffu"],
        rayleigh_type=constants.RayleighType(nonhydrostatic_nml["rayleigh_type"]),
        rayleigh_coeff=rayleigh_coeff,
        divdamp_trans_start=nonhydrostatic_nml["divdamp_trans_start"],
        divdamp_trans_end=nonhydrostatic_nml["divdamp_trans_end"],
        divdamp_type=nonhydrostatic_nml["divdamp_type"],
        igradp_method=nonhydrostatic_nml["igradp_method"],
    )

    # *** InterpolationConfig ***
    interpolation_config = interpolation_factory.InterpolationConfig(
        divergence_averaging_central_cell_weight=nml_data["dynamics_nml"]["divavg_cntrwgt"],
        _nudge_max_coeff=interpol_nml["nudge_max_coeff"],
        nudge_efold_width=interpol_nml["nudge_efold_width"],
        nudge_zone_width=interpol_nml["nudge_zone_width"],
        rbf_kernel_cell=rbf.InterpolationKernel(interpol_nml["rbf_vec_kern_c"]),
        rbf_kernel_edge=rbf.InterpolationKernel(interpol_nml["rbf_vec_kern_e"]),
        rbf_kernel_vertex=rbf.InterpolationKernel(interpol_nml["rbf_vec_kern_v"]),
        lsq_dim_stencil=interpol_nml["lsq_high_ord"],
    )
    # _nudge_max_coeff was scaled to max_nudging_coefficient in __post_init__.
    assert interpolation_config.max_nudging_coefficient is not None

    # *** VerticalGridConfig ***
    rayleigh_damping_height = _list_to_value(nonhydrostatic_nml["damp_height"])

    vertical_grid_config = v_grid.VerticalGridConfig(
        num_levels=_list_to_value(run_nml["num_lev"]),
        maximal_layer_thickness=sleve_nml["max_lay_thckn"],
        top_height_limit_for_maximal_layer_thickness=sleve_nml["htop_thcknlimit"],
        lowest_layer_thickness=sleve_nml["min_lay_thckn"],
        model_top_height=sleve_nml["top_height"],
        flat_height=sleve_nml["flat_height"],
        stretch_factor=sleve_nml["stretch_fac"],
        rayleigh_damping_height=rayleigh_damping_height,
        htop_moist_proc=nonhydrostatic_nml["htop_moist_proc"],
        SLEVE_decay_scale_1=sleve_nml["decay_scale_1"],
        SLEVE_decay_scale_2=sleve_nml["decay_scale_2"],
        SLEVE_decay_exponent=sleve_nml["decay_exp"],
    )

    # *** NonHydrostaticConfig ***
    nonhydro_config = dycore_config.NonHydrostaticConfig(
        itime_scheme=dycore_states.TimeSteppingScheme(nonhydrostatic_nml["itime_scheme"]),
        iadv_rhotheta=dycore_states.RhoThetaAdvectionType(nonhydrostatic_nml["iadv_rhotheta"]),
        igradp_method=dycore_states.HorizontalPressureDiscretizationType(
            nonhydrostatic_nml["igradp_method"]
        ),
        rayleigh_type=constants.RayleighType(nonhydrostatic_nml["rayleigh_type"]),
        divdamp_order=dycore_states.DivergenceDampingOrder(nonhydrostatic_nml["divdamp_order"]),
        divdamp_type=dycore_states.DivergenceDampingType(nonhydrostatic_nml["divdamp_type"]),
        l_vert_nested=run_nml["lvert_nest"],
        deepatmos_mode=nml_data["dynamics_nml"]["ldeepatmo"],
        extra_diffu=nonhydrostatic_nml["lextra_diffu"],
        rhotheta_offctr=nonhydrostatic_nml["rhotheta_offctr"],
        veladv_offctr=nonhydrostatic_nml["veladv_offctr"],
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
        fourth_order_divdamp_factor=nonhydrostatic_nml["divdamp_fac"],
        fourth_order_divdamp_factor2=nonhydrostatic_nml["divdamp_fac2"],
        fourth_order_divdamp_factor3=nonhydrostatic_nml["divdamp_fac3"],
        fourth_order_divdamp_factor4=nonhydrostatic_nml["divdamp_fac4"],
        fourth_order_divdamp_z=nonhydrostatic_nml["divdamp_z"],
        fourth_order_divdamp_z2=nonhydrostatic_nml["divdamp_z2"],
        fourth_order_divdamp_z3=nonhydrostatic_nml["divdamp_z3"],
        fourth_order_divdamp_z4=nonhydrostatic_nml["divdamp_z4"],
    )

    # *** DiffusionConfig ***
    hdiff_smag_w = _list_to_value(diffusion_nml["lhdiff_smag_w"])
    smag_3d = _list_to_value(diffusion_nml["lsmag_3d"])

    diffusion_configuration = diffusion_config.DiffusionConfig(
        diffusion_type=diffusion_config.DiffusionType(diffusion_nml["hdiff_order"]),
        apply_to_vertical_wind=diffusion_nml["lhdiff_w"],
        apply_to_horizontal_wind=diffusion_nml["lhdiff_vn"],
        apply_to_temperature=diffusion_nml["lhdiff_temp"],
        apply_smag_diff_to_vertical_wind=hdiff_smag_w,
        type_vn_diffu=diffusion_config.SmagorinskyStencilType(diffusion_nml["itype_vn_diffu"]),
        compute_3d_smag_coeff=smag_3d,
        type_t_diffu=diffusion_config.TemperatureDiscretizationType(diffusion_nml["itype_t_diffu"]),
        hdiff_efdt_ratio=diffusion_nml["hdiff_efdt_ratio"],
        hdiff_w_efdt_ratio=diffusion_nml["hdiff_w_efdt_ratio"],
        smagorinski_scaling_factor=diffusion_nml["hdiff_smag_fac"],
        smagorinski_scaling_factor2=diffusion_nml["hdiff_smag_fac2"],
        smagorinski_scaling_factor3=diffusion_nml["hdiff_smag_fac3"],
        smagorinski_scaling_factor4=diffusion_nml["hdiff_smag_fac4"],
        smagorinski_scaling_height=diffusion_nml["hdiff_smag_z"],
        smagorinski_scaling_height2=diffusion_nml["hdiff_smag_z2"],
        smagorinski_scaling_height3=diffusion_nml["hdiff_smag_z3"],
        smagorinski_scaling_height4=diffusion_nml["hdiff_smag_z4"],
        ndyn_substeps=nonhydrostatic_nml["ndyn_substeps"],
        apply_zdiffusion_t=nonhydrostatic_nml["l_zdiffu_t"],
        velocity_boundary_diffusion_denominator=gridref_nml["denom_diffu_v"],
        temperature_boundary_diffusion_denominator=gridref_nml["denom_diffu_t"],
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
        shear_type=diffusion_config.TurbulenceShearForcingType(turbdiff_nml["itype_sher"]),
        iforcing=diffusion_config.ForcingType(run_nml["iforcing"]),
        a_hshr=turbdiff_nml["a_hshr"],
    )

    # *** DriverConfig ***
    driver_cfg = driver_config.DriverConfig(
        experiment_name=experiment_description.name,
        output_path=pathlib.Path(),  # TODO (jcanton): Placeholder
        profiling_stats=None,
        dtime=datetime.timedelta(seconds=run_nml["dtime"]),
        start_date=datetime.datetime(1, 1, 1, 0, 0, 0),  # TODO (jcanton): Placeholder
        end_date=datetime.datetime(1, 1, 1, 1, 0, 0),  # TODO (jcanton): Placeholder
        apply_extra_second_order_divdamp=nonhydrostatic_nml["lextra_diffu"],
        vertical_cfl_threshold=nonhydrostatic_nml["vcfl_threshold"],
        ndyn_substeps=nonhydrostatic_nml["ndyn_substeps"],
        enable_statistics_output=False,
    )

    # *** GraupelConfig ***
    if nwp_phy_nml is not None and nwp_tuning_nml is not None:
        graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
            do_latent_heat_nudging=run_nml["ldass_lhn"],
            # ithermo_water == 0 means constant latent heat (docstring in class definition).
            use_constant_latent_heat=_list_to_value(nwp_phy_nml["ithermo_water"]) == 0,
            ice_stickeff_min=nwp_tuning_nml["tune_zceff_min"],
            power_law_coeff_for_ice_mean_fall_speed=nwp_tuning_nml["tune_zvz0i"],
            exponent_for_density_factor_in_ice_sedimentation=nwp_tuning_nml["tune_icesedi_exp"],
            power_law_coeff_for_snow_fall_speed=nwp_tuning_nml["tune_v0snow"],
            rain_mu=nwp_phy_nml["mu_rain"],
            rain_n0=nwp_phy_nml["rain_n0_factor"],
            snow2graupel_riming_coeff=nwp_tuning_nml["tune_zcsg"],
        )
    else:
        graupel_config = graupel.SingleMomentSixClassIconGraupelConfig()

    return definitions.ExperimentConfig(
        driver=driver_cfg,
        vertical_grid=vertical_grid_config,
        nonhydrostatic=nonhydro_config,
        diffusion=diffusion_configuration,
        metrics=metrics_config,
        interpolation=interpolation_config,
        graupel=graupel_config,
    )
