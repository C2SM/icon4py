# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import json
import pathlib
import urllib.parse

import gt4py.next.typing as gtx_typing

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, serialbox


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_experiment_name_with_version(experiment: definitions.Experiment) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment.name}_v{experiment.version:02d}"


def get_ranked_experiment_name_with_version(
    experiment: definitions.Experiment, comm_size: int
) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{get_experiment_name_with_version(experiment)}"


def get_experiment_archive_filename(experiment: definitions.Experiment, comm_size: int) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{get_ranked_experiment_name_with_version(experiment, comm_size)}.tar.gz"


def get_serialized_data_url(root_url: str, filepath: str) -> str:
    """Build a download URL for serialized data file from root URL."""
    return f"{root_url}/download?path=%2F&files={urllib.parse.quote(filepath)}"


def get_datapath_for_experiment(
    experiment: definitions.Experiment,
    processor_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment."""

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment,
        processor_props.comm_size,
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


def create_experiment_configuration(
    experiment: definitions.Experiment,
    processor_props: decomposition.ProcessProperties,
) -> tuple:
    """
    Create configuration objects from the experiment's namelist JSON file.

    This function reads the NAMELIST_ICON_output_atm.json file that comes with
    the serialized experiment data and constructs configuration objects for:
    - DriverConfig
    - VerticalGridConfig
    - NonHydrostaticConfig
    - DiffusionConfig

    The JSON file is cached so it's only read once per unique path.

    Args:
        experiment: The experiment definition
        processor_props: Processor properties containing comm_size

    Returns:
        Tuple of (DriverConfig, VerticalGridConfig, NonHydrostaticConfig, DiffusionConfig)
    """
    from icon4py.model.atmosphere.diffusion import diffusion
    from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
    from icon4py.model.common.constants import RayleighType
    from icon4py.model.common.grid import vertical as v_grid
    from icon4py.model.standalone_driver import config as driver_config

    # Construct path to JSON file
    experiment_dir = get_ranked_experiment_name_with_version(
        experiment,
        processor_props.comm_size,
    )
    json_file_path = (
        definitions.serialized_data_path()
        / experiment_dir
        / f"{definitions.NAMELIST_ICON_FNAME}.json"
    )

    # Read the JSON file (cached)
    nml_data = _read_namelist_json(json_file_path)

    # Extract relevant namelist sections
    sleve_nml = nml_data.get("sleve_nml", {})
    nonhydrostatic_nml = nml_data.get("nonhydrostatic_nml", {})
    diffusion_nml = nml_data.get("diffusion_nml", {})
    run_nml = nml_data.get("run_nml", {})

    # Create VerticalGridConfig
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=experiment.num_levels,
        lowest_layer_thickness=sleve_nml.get("min_lay_thckn", 50.0),
        model_top_height=sleve_nml.get("top_height", 23500.0),
        maximal_layer_thickness=sleve_nml.get("max_lay_thckn", 25000.0),
        top_height_limit_for_maximal_layer_thickness=sleve_nml.get("htop_thcknlimit", 15000.0),
        flat_height=sleve_nml.get("flat_height", 16000.0),
        stretch_factor=sleve_nml.get("stretch_fac", 1.0),
        rayleigh_damping_height=(
            nonhydrostatic_nml.get("damp_height", [45000.0])[0]
            if isinstance(nonhydrostatic_nml.get("damp_height"), list)
            else nonhydrostatic_nml.get("damp_height", 45000.0)
        ),
    )

    # Create NonHydrostaticConfig
    # Map divdamp_order from JSON to enum
    divdamp_order_value = nonhydrostatic_nml.get("divdamp_order", 24)
    divdamp_order_map = {
        4: dycore_states.DivergenceDampingOrder.FOURTH_ORDER,
        24: dycore_states.DivergenceDampingOrder.COMBINED,
    }
    divdamp_order = divdamp_order_map.get(divdamp_order_value, dycore_states.DivergenceDampingOrder.COMBINED)

    # Map divdamp_type from JSON to enum
    divdamp_type_value = nonhydrostatic_nml.get("divdamp_type", 3)
    divdamp_type_map = {
        3: dycore_states.DivergenceDampingType.THREE_DIMENSIONAL,
        32: dycore_states.DivergenceDampingType.COMBINED,
    }
    divdamp_type = divdamp_type_map.get(divdamp_type_value, dycore_states.DivergenceDampingType.THREE_DIMENSIONAL)

    # Map itime_scheme from JSON to enum
    itime_scheme_value = nonhydrostatic_nml.get("itime_scheme", 4)
    itime_scheme_map = {
        4: dycore_states.TimeSteppingScheme.MOST_EFFICIENT,
        5: dycore_states.TimeSteppingScheme.STABLE,
        6: dycore_states.TimeSteppingScheme.EXPENSIVE,
    }
    itime_scheme = itime_scheme_map.get(itime_scheme_value, dycore_states.TimeSteppingScheme.MOST_EFFICIENT)

    # Map iadv_rhotheta from JSON to enum
    iadv_rhotheta_value = nonhydrostatic_nml.get("iadv_rhotheta", 2)
    iadv_rhotheta_map = {
        2: dycore_states.RhoThetaAdvectionType.MIURA,
    }
    iadv_rhotheta = iadv_rhotheta_map.get(iadv_rhotheta_value, dycore_states.RhoThetaAdvectionType.MIURA)

    # Map igradp_method from JSON to enum
    igradp_method_value = nonhydrostatic_nml.get("igradp_method", 3)
    igradp_method_map = {
        1: dycore_states.HorizontalPressureDiscretizationType.CONVENTIONAL,
        2: dycore_states.HorizontalPressureDiscretizationType.TAYLOR,
        3: dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO,
        4: dycore_states.HorizontalPressureDiscretizationType.POLYNOMIAL,
        5: dycore_states.HorizontalPressureDiscretizationType.POLYNOMIAL_HYDRO,
    }
    igradp_method = igradp_method_map.get(
        igradp_method_value, dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    )

    # Map rayleigh_type from JSON to enum
    rayleigh_type_value = nonhydrostatic_nml.get("rayleigh_type", 2)
    rayleigh_type_map = {
        2: RayleighType.KLEMP,
    }
    rayleigh_type = rayleigh_type_map.get(rayleigh_type_value, RayleighType.KLEMP)

    # Extract rayleigh_coeff (can be a list or single value)
    rayleigh_coeff = nonhydrostatic_nml.get("rayleigh_coeff", 0.05)
    if isinstance(rayleigh_coeff, list):
        rayleigh_coeff = rayleigh_coeff[0]

    nonhydro_config = solve_nh.NonHydrostaticConfig(
        itime_scheme=itime_scheme,  # type: ignore[arg-type]
        iadv_rhotheta=iadv_rhotheta,  # type: ignore[arg-type]
        igradp_method=igradp_method,  # type: ignore[arg-type]
        rayleigh_type=rayleigh_type,  # type: ignore[arg-type]
        rayleigh_coeff=rayleigh_coeff,
        divdamp_order=divdamp_order,  # type: ignore[arg-type]
        divdamp_type=divdamp_type,
        divdamp_trans_start=nonhydrostatic_nml.get("divdamp_trans_start", 12500.0),
        divdamp_trans_end=nonhydrostatic_nml.get("divdamp_trans_end", 17500.0),
        rhotheta_offctr=nonhydrostatic_nml.get("rhotheta_offctr", -0.1),
        veladv_offctr=nonhydrostatic_nml.get("veladv_offctr", 0.25),
        fourth_order_divdamp_factor=nonhydrostatic_nml.get("divdamp_fac", 0.0025),
        fourth_order_divdamp_factor2=nonhydrostatic_nml.get("divdamp_fac2", 0.004),
        fourth_order_divdamp_factor3=nonhydrostatic_nml.get("divdamp_fac3", 0.004),
        fourth_order_divdamp_factor4=nonhydrostatic_nml.get("divdamp_fac4", 0.004),
        fourth_order_divdamp_z=nonhydrostatic_nml.get("divdamp_z", 32500.0),
        fourth_order_divdamp_z2=nonhydrostatic_nml.get("divdamp_z2", 40000.0),
        fourth_order_divdamp_z3=nonhydrostatic_nml.get("divdamp_z3", 60000.0),
        fourth_order_divdamp_z4=nonhydrostatic_nml.get("divdamp_z4", 80000.0),
    )

    # Create DiffusionConfig
    # Map diffusion_type from hdiff_order
    hdiff_order_value = diffusion_nml.get("hdiff_order", 5)
    diffusion_type_map = {
        -1: diffusion.DiffusionType.NO_DIFFUSION,
        2: diffusion.DiffusionType.LINEAR_2ND_ORDER,
        3: diffusion.DiffusionType.SMAGORINSKY_NO_BACKGROUND,
        4: diffusion.DiffusionType.LINEAR_4TH_ORDER,
        5: diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
    }
    diffusion_type = diffusion_type_map.get(hdiff_order_value, diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER)

    # Map type_vn_diffu from itype_vn_diffu
    type_vn_diffu_value = diffusion_nml.get("itype_vn_diffu", 1)
    type_vn_diffu_map = {
        1: diffusion.SmagorinskyStencilType.DIAMOND_VERTICES,
        2: diffusion.SmagorinskyStencilType.CELLS_AND_VERTICES,
    }
    type_vn_diffu = type_vn_diffu_map.get(type_vn_diffu_value, diffusion.SmagorinskyStencilType.DIAMOND_VERTICES)

    # Map type_t_diffu from itype_t_diffu
    type_t_diffu_value = diffusion_nml.get("itype_t_diffu", 2)
    type_t_diffu_map = {
        1: diffusion.TemperatureDiscretizationType.HOMOGENEOUS,
        2: diffusion.TemperatureDiscretizationType.HETEROGENOUS,
    }
    type_t_diffu = type_t_diffu_map.get(type_t_diffu_value, diffusion.TemperatureDiscretizationType.HETEROGENOUS)

    # Extract hdiff_smag_w (can be a list or single value)
    lhdiff_smag_w = diffusion_nml.get("lhdiff_smag_w", [False])
    hdiff_smag_w = lhdiff_smag_w[0] if isinstance(lhdiff_smag_w, list) else lhdiff_smag_w

    # Extract lsmag_3d (can be a list or single value)
    lsmag_3d = diffusion_nml.get("lsmag_3d", [False])
    smag_3d = lsmag_3d[0] if isinstance(lsmag_3d, list) else lsmag_3d

    # Get ndyn_substeps from either nonhydrostatic_nml or run_nml
    ndyn_substeps = nonhydrostatic_nml.get("ndyn_substeps", run_nml.get("ndyn_substeps", 5))

    diffusion_config = diffusion.DiffusionConfig(
        diffusion_type=diffusion_type,  # type: ignore[arg-type]
        hdiff_w=diffusion_nml.get("lhdiff_w", True),
        hdiff_vn=diffusion_nml.get("lhdiff_vn", True),
        hdiff_temp=diffusion_nml.get("lhdiff_temp", True),
        hdiff_smag_w=hdiff_smag_w,
        type_vn_diffu=type_vn_diffu,  # type: ignore[arg-type]
        smag_3d=smag_3d,
        type_t_diffu=type_t_diffu,  # type: ignore[arg-type]
        hdiff_efdt_ratio=diffusion_nml.get("hdiff_efdt_ratio", 36.0),
        hdiff_w_efdt_ratio=diffusion_nml.get("hdiff_w_efdt_ratio", 15.0),
        smagorinski_scaling_factor=diffusion_nml.get("hdiff_smag_fac", 0.015),
        n_substeps=ndyn_substeps,
        zdiffu_t=nonhydrostatic_nml.get("l_zdiffu_t", True),
        thslp_zdiffu=nonhydrostatic_nml.get("thslp_zdiffu", 0.025),
        thhgtd_zdiffu=nonhydrostatic_nml.get("thhgtd_zdiffu", 200.0),
    )

    # Create DriverConfig (using defaults for now, as these are not in the JSON)
    # TODO(jcanton): Extract these from the JSON when available
    import datetime

    driver_cfg = driver_config.DriverConfig(
        experiment_name=experiment.name,
        output_path=pathlib.Path(),  # Placeholder
        profiling_stats=None,
        dtime=datetime.timedelta(seconds=run_nml.get("dtime", 600.0)),
        start_date=datetime.datetime(1, 1, 1, 0, 0, 0),  # Placeholder
        end_date=datetime.datetime(1, 1, 1, 1, 0, 0),  # Placeholder
        apply_extra_second_order_divdamp=nonhydrostatic_nml.get("lextra_diffu", False),
        vertical_cfl_threshold=nonhydrostatic_nml.get("vcfl_threshold", 0.85),
        ndyn_substeps=ndyn_substeps,
        enable_statistics_output=False,
    )

    return driver_cfg, vertical_config, nonhydro_config, diffusion_config
