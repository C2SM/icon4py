# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import icon4py.model.common.states.factory as factory
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal
from icon4py.model.common.metrics import metric_fields as mf
from icon4py.model.common.test_utils import datatest_utils as dt_utils, serialbox_utils as sb


# we need to register a couple of fields from the serializer. Those should get replaced one by one.

dt_utils.TEST_DATA_ROOT = pathlib.Path(__file__).parent / "testdata"
properties = decomposition.get_processor_properties(decomposition.get_runtype(with_mpi=False))
path = dt_utils.get_ranked_data_path(dt_utils.SERIALIZED_DATA_PATH, properties)

data_provider = sb.IconSerialDataProvider(
    "icon_pydycore", str(path.absolute()), False, mpi_rank=properties.rank
)

# z_ifc (computable from vertical grid for model without topography)
metrics_savepoint = data_provider.from_metrics_savepoint()

# interpolation fields also for now passing as precomputed fields
interpolation_savepoint = data_provider.from_interpolation_savepoint()
# can get geometry fields as pre computed fields from the grid_savepoint
grid_savepoint = data_provider.from_savepoint_grid()
#######

# start build up factory:


interface_model_height = metrics_savepoint.z_ifc()
c_lin_e = interpolation_savepoint.c_lin_e()

fields_factory = factory.FieldsFactory()

# used for vertical domain below: should go away once vertical grid provids start_index and end_index like interface
grid = grid_savepoint.global_grid_params

fields_factory.register_provider(
    factory.PrecomputedFieldsProvider(
        {
            "height_on_interface_levels": interface_model_height,
            "cell_to_edge_interpolation_coefficient": c_lin_e,
        }
    )
)
height_provider = factory.ProgramFieldProvider(
    func=mf.compute_z_mc,
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, grid.num_levels),
    },
    fields={"z_mc": "height"},
    deps={"z_ifc": "height_on_interface_levels"},
)
