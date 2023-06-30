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
import tarfile
from pathlib import Path

import pytest
import wget

from icon4py.diffusion.diffusion import DiffusionConfig

from .test_utils.serialbox_utils import IconSerialDataProvider


data_uri = "https://polybox.ethz.ch/index.php/s/LcAbscZqnsx4WCf/download"
data_path = Path(__file__).parent.joinpath("ser_icondata")
extracted_path = data_path.joinpath("mch_ch_r04b09_dsl/ser_data")
data_file = data_path.joinpath("mch_ch_r04b09_dsl_v2.tar.gz").name


@pytest.fixture(scope="session")
def setup_icon_data():
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    data_path.mkdir(parents=True, exist_ok=True)
    if not any(data_path.iterdir()):
        print(
            f"directory {data_path} is empty: downloading data from {data_uri} and extracting"
        )
        wget.download(data_uri, out=data_file)
        # extract downloaded file
        if not tarfile.is_tarfile(data_file):
            raise NotImplementedError(f"{data_file} needs to be a valid tar file")
        with tarfile.open(data_file, mode="r:*") as tf:
            tf.extractall(path=data_path)
        Path(data_file).unlink(missing_ok=True)


@pytest.fixture
def data_provider(setup_icon_data) -> IconSerialDataProvider:
    return IconSerialDataProvider("icon_pydycore", str(extracted_path), True)


@pytest.fixture
def linit():
    """
    Set the 'linit' flag for the ICON diffusion data savepoint.

    Defaults to False
    """
    return False


@pytest.fixture
def step_date_init():
    """
    Set the step date for the loaded ICON time stamp at start of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def step_date_exit():
    """
    Set the step date for the loaded ICON time stamp at the end of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def diffusion_savepoint_init(data_provider, linit, step_date_init):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'

    linit flag can be set by overriding the 'linit' fixture
    """
    return data_provider.from_savepoint_diffusion_init(linit=linit, date=step_date_init)


@pytest.fixture
def savepoint_velocity_init(data_provider, step_date_init, istep, vn_only, jstep):
    """
    Load data from ICON savepoint at start of velocity_advection module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_init(
        istep=istep, vn_only=vn_only, date=step_date_init, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_init(data_provider, step_date_init, istep, jstep):
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_init(
        istep=istep, date=step_date_init, jstep=jstep
    )


@pytest.fixture
def diffusion_savepoint_exit(data_provider, step_date_exit):
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=False, date=step_date_exit)
    return sp


@pytest.fixture
def savepoint_velocity_exit(data_provider, step_date_exit, istep, vn_only, jstep):
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_exit(
        istep=istep, vn_only=vn_only, date=step_date_exit, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_exit(data_provider, step_date_exit, istep, jstep):
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_exit(
        istep=istep, date=step_date_exit, jstep=jstep
    )


@pytest.fixture
def interpolation_savepoint(data_provider):
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider):
    """Load data from ICON mestric state savepoint."""
    return data_provider.from_metrics_savepoint()


@pytest.fixture
def metrics_nonhydro_savepoint(data_provider):
    """Load data from ICON mestric state nonhydro savepoint."""
    return data_provider.from_metrics_nonhydro_savepoint()


@pytest.fixture
def icon_grid(grid_savepoint):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid()


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


@pytest.fixture
def r04b09_diffusion_config(setup_icon_data) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=5,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
    )


@pytest.fixture
def damping_height():
    return 12500


@pytest.fixture
def istep():
    return 1


@pytest.fixture
def jstep():
    return 0


@pytest.fixture
def ntnd(savepoint_velocity_init):
    return savepoint_velocity_init.get_metadata("ntnd").get("ntnd")


@pytest.fixture
def vn_only():
    return False
