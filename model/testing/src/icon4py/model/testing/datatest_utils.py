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
from icon4py.model.testing import definitions as test_defs, serialbox


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_experiment_name_with_version(experiment: test_defs.ExperimentDescription) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment.name}_v{experiment.version:02d}"


def get_ranked_experiment_name_with_version(
    experiment: test_defs.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{get_experiment_name_with_version(experiment)}"


def get_experiment_archive_filename(experiment: test_defs.ExperimentDescription, comm_size: int) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{get_ranked_experiment_name_with_version(experiment, comm_size)}.tar.gz"


def get_serialized_data_url(root_url: str, filepath: str) -> str:
    """Build a download URL for serialized data file from root URL."""
    return f"{root_url}/download?path=%2F&files={urllib.parse.quote(filepath)}"


def get_datapath_for_experiment(
    experiment: test_defs.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment."""

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment,
        processor_props.comm_size,
    )
    return test_defs.serialized_data_path().joinpath(
        experiment_dir, test_defs.SERIALIZED_DATA_SUBDIR
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
    experiment: test_defs.ExperimentDescription,
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

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment,
        processor_props.comm_size,
    )
    json_file_path = (
        test_defs.serialized_data_path()
        / experiment_dir
        / f"{test_defs.NAMELIST_ICON_FNAME}.json"
    )

    nml_data = _read_namelist_json(json_file_path)

    sleve_nml = nml_data["sleve_nml"]
    nonhydrostatic_nml = nml_data["nonhydrostatic_nml"]
    diffusion_nml = nml_data["diffusion_nml"]
    run_nml = nml_data["run_nml"]

    # Create VerticalGridConfig
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=experiment.num_levels,
        lowest_layer_thickness=sleve_nml["min_lay_thckn"],
        model_top_height=sleve_nml["top_height"],
        maximal_layer_thickness=sleve_nml["max_lay_thckn"],
        top_height_limit_for_maximal_layer_thickness=sleve_nml["htop_thcknlimit"],
        flat_height=sleve_nml["flat_height"],
        stretch_factor=sleve_nml["stretch_fac"],
        rayleigh_damping_height=(
            nonhydrostatic_nml["damp_height"][0]
            if isinstance(nonhydrostatic_nml["damp_height"], list)
            else nonhydrostatic_nml["damp_height"]
        ),
    )

    # Create NonHydrostaticConfig
    # Map divdamp_order from JSON to enum
    divdamp_order_value = nonhydrostatic_nml["divdamp_order"]
    divdamp_order_map = {
        4: dycore_states.DivergenceDampingOrder.FOURTH_ORDER,
        24: dycore_states.DivergenceDampingOrder.COMBINED,
    }
    divdamp_order = divdamp_order_map[divdamp_order_value]

    # Map divdamp_type from JSON to enum
    divdamp_type_value = nonhydrostatic_nml["divdamp_type"]
    divdamp_type_map = {
        3: dycore_states.DivergenceDampingType.THREE_DIMENSIONAL,
        32: dycore_states.DivergenceDampingType.COMBINED,
    }
    divdamp_type = divdamp_type_map[divdamp_type_value]

    # Map itime_scheme from JSON to enum
    itime_scheme_value = nonhydrostatic_nml["itime_scheme"]
    itime_scheme_map = {
        4: dycore_states.TimeSteppingScheme.MOST_EFFICIENT,
        5: dycore_states.TimeSteppingScheme.STABLE,
        6: dycore_states.TimeSteppingScheme.EXPENSIVE,
    }
    itime_scheme = itime_scheme_map[itime_scheme_value]

    # Map iadv_rhotheta from JSON to enum
    iadv_rhotheta_value = nonhydrostatic_nml["iadv_rhotheta"]
    iadv_rhotheta_map = {
        2: dycore_states.RhoThetaAdvectionType.MIURA,
    }
    iadv_rhotheta = iadv_rhotheta_map[iadv_rhotheta_value]

    # Map igradp_method from JSON to enum
    igradp_method_value = nonhydrostatic_nml["igradp_method"]
    igradp_method_map = {
        1: dycore_states.HorizontalPressureDiscretizationType.CONVENTIONAL,
        2: dycore_states.HorizontalPressureDiscretizationType.TAYLOR,
        3: dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO,
        4: dycore_states.HorizontalPressureDiscretizationType.POLYNOMIAL,
        5: dycore_states.HorizontalPressureDiscretizationType.POLYNOMIAL_HYDRO,
    }
    igradp_method = igradp_method_map[igradp_method_value]

    # Map rayleigh_type from JSON to enum
    rayleigh_type_value = nonhydrostatic_nml["rayleigh_type"]
    rayleigh_type_map = {
        2: RayleighType.KLEMP,
    }
    rayleigh_type = rayleigh_type_map[rayleigh_type_value]

    # Extract rayleigh_coeff (can be a list or single value)
    rayleigh_coeff = nonhydrostatic_nml["rayleigh_coeff"]
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
        divdamp_trans_start=nonhydrostatic_nml["divdamp_trans_start"],
        divdamp_trans_end=nonhydrostatic_nml["divdamp_trans_end"],
        rhotheta_offctr=nonhydrostatic_nml["rhotheta_offctr"],
        veladv_offctr=nonhydrostatic_nml["veladv_offctr"],
        fourth_order_divdamp_factor=nonhydrostatic_nml["divdamp_fac"],
        fourth_order_divdamp_factor2=nonhydrostatic_nml["divdamp_fac2"],
        fourth_order_divdamp_factor3=nonhydrostatic_nml["divdamp_fac3"],
        fourth_order_divdamp_factor4=nonhydrostatic_nml["divdamp_fac4"],
        fourth_order_divdamp_z=nonhydrostatic_nml["divdamp_z"],
        fourth_order_divdamp_z2=nonhydrostatic_nml["divdamp_z2"],
        fourth_order_divdamp_z3=nonhydrostatic_nml["divdamp_z3"],
        fourth_order_divdamp_z4=nonhydrostatic_nml["divdamp_z4"],
    )

    # Create DiffusionConfig
    # Map diffusion_type from hdiff_order
    hdiff_order_value = diffusion_nml["hdiff_order"]
    diffusion_type_map = {
        -1: diffusion.DiffusionType.NO_DIFFUSION,
        2: diffusion.DiffusionType.LINEAR_2ND_ORDER,
        3: diffusion.DiffusionType.SMAGORINSKY_NO_BACKGROUND,
        4: diffusion.DiffusionType.LINEAR_4TH_ORDER,
        5: diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
    }
    diffusion_type = diffusion_type_map[hdiff_order_value]

    # Map type_vn_diffu from itype_vn_diffu
    type_vn_diffu_value = diffusion_nml["itype_vn_diffu"]
    type_vn_diffu_map = {
        1: diffusion.SmagorinskyStencilType.DIAMOND_VERTICES,
        2: diffusion.SmagorinskyStencilType.CELLS_AND_VERTICES,
    }
    type_vn_diffu = type_vn_diffu_map[type_vn_diffu_value]

    # Map type_t_diffu from itype_t_diffu
    type_t_diffu_value = diffusion_nml["itype_t_diffu"]
    type_t_diffu_map = {
        1: diffusion.TemperatureDiscretizationType.HOMOGENEOUS,
        2: diffusion.TemperatureDiscretizationType.HETEROGENOUS,
    }
    type_t_diffu = type_t_diffu_map[type_t_diffu_value]

    # Extract hdiff_smag_w (can be a list or single value)
    lhdiff_smag_w = diffusion_nml["lhdiff_smag_w"]
    hdiff_smag_w = lhdiff_smag_w[0] if isinstance(lhdiff_smag_w, list) else lhdiff_smag_w

    # Extract lsmag_3d (can be a list or single value)
    lsmag_3d = diffusion_nml["lsmag_3d"]
    smag_3d = lsmag_3d[0] if isinstance(lsmag_3d, list) else lsmag_3d

    # Get ndyn_substeps from either nonhydrostatic_nml or run_nml
    ndyn_substeps = nonhydrostatic_nml["ndyn_substeps"]

    diffusion_config = diffusion.DiffusionConfig(
        diffusion_type=diffusion_type,  # type: ignore[arg-type]
        hdiff_w=diffusion_nml["lhdiff_w"],
        hdiff_vn=diffusion_nml["lhdiff_vn"],
        hdiff_temp=diffusion_nml["lhdiff_temp"],
        hdiff_smag_w=hdiff_smag_w,
        type_vn_diffu=type_vn_diffu,  # type: ignore[arg-type]
        smag_3d=smag_3d,
        type_t_diffu=type_t_diffu,  # type: ignore[arg-type]
        hdiff_efdt_ratio=diffusion_nml["hdiff_efdt_ratio"],
        hdiff_w_efdt_ratio=diffusion_nml["hdiff_w_efdt_ratio"],
        smagorinski_scaling_factor=diffusion_nml["hdiff_smag_fac"],
        n_substeps=ndyn_substeps,
        zdiffu_t=nonhydrostatic_nml["l_zdiffu_t"],
    )

    # Create DriverConfig (using defaults for now, as these are not in the JSON)
    # TODO(jcanton): Extract these from the JSON when available
    import datetime

    driver_config = driver_config.DriverConfig(
        experiment_name=experiment.name,
        output_path=pathlib.Path(),  # Placeholder
        profiling_stats=None,
        dtime=datetime.timedelta(seconds=run_nml["dtime"]),
        start_date=datetime.datetime(1, 1, 1, 0, 0, 0),  # Placeholder
        end_date=datetime.datetime(1, 1, 1, 1, 0, 0),  # Placeholder
        apply_extra_second_order_divdamp=nonhydrostatic_nml["lextra_diffu"],
        vertical_cfl_threshold=nonhydrostatic_nml["vcfl_threshold"],
        ndyn_substeps=ndyn_substeps,
        enable_statistics_output=False,
    )

    return driver_config, vertical_config, nonhydro_config, diffusion_config
