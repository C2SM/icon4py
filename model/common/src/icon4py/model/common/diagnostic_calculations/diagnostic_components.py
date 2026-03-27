# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime

import gt4py.next as gtx
from typing_extensions import Literal

from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils import diagnose_temperature
from icon4py.model.common.exceptions import IncompleteStateError
from icon4py.model.common.grid import base, horizontal, vertical as v_grid
from icon4py.model.common.states import model


Inputs = Literal["theta_v", "exner", "qv", "qc", "qi", "qr", "qs", "qg"]
Outputs = Literal["temperature", "virtual_temperature"]
cell_domain = horizontal.domain(dims.CellDim)


class TemperatureComponent:
    """
    Diagnostic component computing temperature and virtual temperature.

    Implements the [Model Component Protocol](../../components/components.py).

    TODO (@halungge):  it would be nice to call the @gtx.field_operator that returns the computed fields
        directly, as this currently does not work for compiled backend we pre-allocate the output buffers
        in the constructor. Hence each call to the component will overwrite those buffers with the
        diagnostics of the new time step. The reference to the grid (horizontal and vertical is needed only
        for this allocation)
    """

    def __init__(self, backend, grid: base.BaseGrid, vertical_grid: v_grid.VerticalGrid):
        self.backend = backend
        # TODO (@halungge) as running the gtx.field_operator directly (which would return the results) does not
        # work (for compiled backends) we allocate the output buffers here
        output_field_size = {
            dims.CellDim: (0, grid.size[dims.CellDim]),
            dims.KDim: (0, vertical_grid.size(dims.KDim)),
        }
        self._output_fields = {
            k: model.ModelField(
                attrs=d,
                data=(
                    gtx.constructors.zeros.partial(allocator=backend)(
                        output_field_size, dtype=ta.wpfloat
                    )
                ),
            )
            for k, d in self.output_properties.items()
        }

    @property
    def input_properties(self):
        return {
            "theta_v": dict(standard_name="theta_v", units="K"),
            "exner": dict(standard_name="dimensionless_exner_pressure", units="1"),
            "qv": dict(standard_name="specific_humidity", units="1"),
            "qc": dict(standard_name="specific_cloud_content", units="1"),
            "qi": dict(standard_name="specific_ice_content", units="1"),
            "qr": dict(standard_name="specific_rain_content", units="1"),
            "qs": dict(standard_name="specific_snow_content", units="1"),
            "qg": dict(standard_name="specific_graupel_content", units="1"),
        }

    @property
    def output_properties(self):
        return {
            "temperature": dict(standard_name="temperature", units="K"),
            "virtual_temperature": dict(standard_name="virtual_temperature", units="K"),
        }

    def __call__(self, model_state: dict, timestep: datetime.datetime):
        theta_v = self._get_or_raise(model_state, "theta_v")
        exner = self._get_or_raise(model_state, "exner")

        qv = self._get_or_raise(model_state, "qv")
        qc = self._get_or_raise(model_state, "qc")
        qi = self._get_or_raise(model_state, "qi")
        qr = self._get_or_raise(model_state, "qr")
        qs = self._get_or_raise(model_state, "qs")
        qg = self._get_or_raise(model_state, "qg")

        ## calling field operator directly by passing the as_program run works
        # - for embedded if no `field_provider` kwarg is given, if this is given it assumes that you want a as program run and asks for the `out` kwarg
        # - never works for the compiled backends: the allways require `offset_provider` and `out`
        diagnose_temperature._diagnose_virtual_temperature_and_temperature.with_backend(
            self.backend
        )(
            qv.data,
            qc.data,
            qi.data,
            qr.data,
            qs.data,
            qg.data,
            theta_v.data,
            exner.data,
            constants.RV_O_RD_MINUS_1,
            out=(
                self._output_fields["virtual_temperature"].data,
                self._output_fields["temperature"].data,
            ),
            offset_provider={},
        )
        return self._output_fields

    def _get_or_raise(self, model_state, name: str):
        try:
            return model_state[self.input_properties[name]["standard_name"]]
        except KeyError as err:
            raise IncompleteStateError(
                f"missing field in state {self.input_properties[name]}"
            ) from err
