# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common.decomposition.definitions import DecompositionInfo, ProcessProperties
from icon4py.model.common.decomposition.mpi_decomposition import ParallelLogger
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils import serialbox_utils as sb

# TODO (magdalena) should be removed when fields can be calculated properly.
from model.atmosphere.diffusion.tests.diffusion_tests.utils import (
    construct_interpolation_state,
    construct_metric_state_for_diffusion,
    construct_diagnostics,
)

SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, Enum):
    SB = "serialbox"
    NC = "netcdf"


def read_icon_grid(
    path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads
        from ppser serialized test data
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid()
            .construct_icon_grid()
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_initial_state(
    gridfile_path: Path, rank=0
) -> tuple[sb.IconSerialDataProvider, DiffusionDiagnosticState, PrognosticState]:
    """
    Read prognostic and diagnostic state from serialized data.

    Args:
        gridfile_path: path the serialized input data

    Returns: a tuple containing the data_provider, the initial diagnostic and prognostic state.
        The data_provider is returned such that further timesteps of diagnostics and prognostics
        can be read from within the dummy timeloop

    """
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(gridfile_path), False, mpi_rank=rank
    )
    init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    prognostic_state = init_savepoint.construct_prognostics()
    diagnostic_state = construct_diagnostics(init_savepoint)
    return data_provider, diagnostic_state, prognostic_state


def read_geometry_fields(
    path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> tuple[EdgeParams, CellParams, VerticalModelParams]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        ).from_savepoint_grid()
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vertical_geometry = VerticalModelParams(vct_a=sp.vct_a(), rayleigh_damping_height=12500)
        return edge_geometry, cell_geometry, vertical_geometry
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_decomp_info(
    path: Path,
    procs_props: ProcessProperties,
    ser_type=SerializationType.SB,
) -> DecompositionInfo:
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), True, procs_props.rank
        )
        return sp.from_savepoint_grid().construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> tuple[DiffusionMetricState, DiffusionInterpolationState]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        dataprovider = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        )
        interpolation_state = construct_interpolation_state(
            dataprovider.from_interpolation_savepoint()
        )
        metric_state = construct_metric_state_for_diffusion(dataprovider.from_metrics_savepoint())
        return metric_state, interpolation_state
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(run_path: str, start_time, processor_procs: ProcessProperties = None) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        start_time: start time of the model run

    """
    run_dir = Path(run_path).absolute() if run_path else Path(__file__).absolute().parent
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(f"dummy_dycore_driver_{datetime.isoformat(start_time)}.log")
    logfile.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    console_handler.addFilter(ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)
