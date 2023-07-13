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
import importlib
import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from icon4py.decomposition.decomposed import ProcessProperties
from icon4py.decomposition.parallel_setup import DecompositionInfo
from icon4py.diffusion.horizontal import CellParams, EdgeParams
from icon4py.diffusion.icon_grid import IconGrid, VerticalModelParams
from icon4py.diffusion.state_utils import (
    DiagnosticState,
    InterpolationState,
    MetricState,
    PrognosticState,
)


SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


def import_testutils():
    testutils = (
        Path(__file__).parent.__str__() + "/../../../tests/test_utils/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("helpers", testutils)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = import_testutils()

from helpers import serialbox_utils as sb  # noqa


# TODO(Magdalena): for preliminary version of the driver we need serialbox data which is in
#  testutils, since that is no proper package we need to import it by hand here.
#  Hence: Turn testutils into a package again?
def import_testutils():
    testutils = (
        Path(__file__).parent.__str__() + "/../../../tests/test_utils/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("helpers", testutils)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = import_testutils()

from helpers import serialbox_utils as sb  # noqa


class SerializationType(str, Enum):
    SB = "serialbox"
    NC = "netcdf"


def read_icon_grid(path: Path, rank=0, ser_type:  SerializationType=SerializationType.SB) -> IconGrid:
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
            sb.IconSerialDataProvider(
                "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
            )
            .from_savepoint_grid()
            .construct_icon_grid()
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_initial_state(
    gridfile_path: Path, rank=0
) -> tuple[sb.IconSerialDataProvider, DiagnosticState, PrognosticState]:
    """
    Read prognostic and diagnostic state from serialized data.

    Args:
        gridfile_path: path the serialized input data

    Returns: a tuple containing the data_provider, the initial diagnostic and prognostic state.
        The data_provider is returned such that further timesteps of diagnostics and prognostics
        can be read from within the dummy timeloop

    """
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(gridfile_path), False
    )
    init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    prognostic_state = init_savepoint.construct_prognostics()
    diagnostic_state = init_savepoint.construct_diagnostics_for_diffusion()
    return data_provider, diagnostic_state, prognostic_state


def read_geometry_fields(
    path: Path, ser_type:SerializationType=SerializationType.SB, rank=0
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
        vertical_geometry = VerticalModelParams(
            vct_a=sp.vct_a(), rayleigh_damping_height=12500
        )
        return edge_geometry, cell_geometry, vertical_geometry
    else:
        raise NotImplementedError(SB_ONLY_MSG)

# /home/magdalena/data/exclaim/dycore/mch_ch_r04b09_dsl/node2/mch_ch_r04b09_dsl/icon_grid
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


def read_grid(
    path: Path,
    procs_props: ProcessProperties,
    ser_type=SerializationType.SB,
) -> IconGrid:
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_grid", str(path.absolute()), True, procs_props.rank
        )
        return sp.from_savepoint_grid().construct_icon_grid()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    path: Path, ser_type: SerializationType=SerializationType.SB, rank=0
) -> tuple[MetricState, InterpolationState]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        dataprovider = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        )
        interpolation_state = (
            dataprovider.from_interpolation_savepoint().construct_interpolation_state_for_diffusion()
        )
        metric_state = dataprovider.from_metrics_savepoint().construct_metric_state()
        return metric_state, interpolation_state
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(run_path: str, start_time) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        start_time: start time of the model run

    """
    run_dir = (
        Path(run_path).absolute() if run_path else Path(__file__).absolute().parent
    )
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(
        f"dummy_dycore_driver_{datetime.isoformat(start_time)}.log"
    )
    logfile.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)-20s : %(funcName)-20s:  %(levelname)-8s %(message)s"
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console_handler)
