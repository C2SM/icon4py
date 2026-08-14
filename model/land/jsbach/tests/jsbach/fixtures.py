# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, serialbox
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    download_ser_data,
    experiment_description,
    process_props,
)


__all__ = [
    "backend",
    "backend_like",
    "data_provider",
    "download_ser_data",
    "experiment_description",
    "process_props",
]


@pytest.fixture
def data_provider(
    download_ser_data: None,  # downloads data as side-effect
    experiment_description: test_defs.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    backend,
) -> serialbox.IconSerialDataProvider:
    """Serialbox provider built from the experiment description alone.

    The shared fixture in icon4py.model.testing goes through `experiment`, which
    parses the three Fortran namelist JSONs next to the archive. The JSBACH SSE
    datatest needs none of that configuration -- everything it compares comes out of
    the savepoints -- so it takes the shorter path and works against a bare
    ser_data directory.
    """
    data_path = dt_utils.get_datapath_for_experiment(experiment_description, process_props)
    return dt_utils.create_icon_serial_data_provider(data_path, process_props.rank, backend)
