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

from icon4py.diffusion.horizontal import CellParams, EdgeParams
from icon4py.diffusion.icon_grid import IconGrid, VerticalModelParams
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.testutils import serialbox_utils


SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, Enum):
    SB = "serialbox"
    NC = "netcdf"


def read_icon_grid(path: Path, ser_type=SerializationType.SB) -> IconGrid:
    """
    Return IconGrid parsed from a given input type.

    Factory method that returns an icon grid dependeing on the ser_type.

    Args:
        path: str - path where to find the input data
        ser_type: str - type of input data. Currently only 'sb (serialbox)' is supported. It reads from ppser serialized test data
    """
    if ser_type == SerializationType.SB:
        return (
            serialbox_utils.IconSerialDataProvider(
                "icon_pydycore", str(path.absolute()), False
            )
            .from_savepoint_grid()
            .construct_icon_grid()
        )
    else:
        raise NotImplementedError("Only ser_type='sb' is implemented so far.")


def read_initial_state(gridfile_path: Path):
    data_provider = serialbox_utils.IconSerialDataProvider(
        "icon_pydycore", str(gridfile_path), False
    )
    init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    prognostic_state = init_savepoint.construct_prognostics()
    diagnostic_state = init_savepoint.construct_diagnostics()
    return data_provider, diagnostic_state, prognostic_state


def read_geometry_fields(
    path: Path, ser_type=SerializationType.SB
) -> tuple[EdgeParams, CellParams, VerticalModelParams]:
    if ser_type == SerializationType.SB:
        sp = serialbox_utils.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False
        ).from_savepoint_grid()
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vertical_geometry = VerticalModelParams(
            vct_a=sp.vct_a(), rayleigh_damping_height=12500
        )
        return edge_geometry, cell_geometry, vertical_geometry
    else:
        raise NotImplementedError("Only ser_type='sb' is implemented so far.")


def read_static_fields(
    path: Path, ser_type=SerializationType.SB
) -> tuple[MetricState, InterpolationState]:
    if ser_type == SerializationType.SB:
        sp = serialbox_utils.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False
        ).from_savepoint_diffusion_init(linit=True, date=SIMULATION_START_DATE)
        metric_state = sp.construct_metric_state()
        interpolation_state = sp.construct_interpolation_state()
        return metric_state, interpolation_state
    else:
        raise NotImplementedError("Only ser_type='sb' is implemented so far.")


def configure_logging(run_path: str, start_time):
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
