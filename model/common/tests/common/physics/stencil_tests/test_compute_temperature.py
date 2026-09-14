# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants as phy_const, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics.compute_temperature import (
    compute_temperature_from_internal_energy_per_area,
    compute_virtual_temperature_and_temperature,
)
from icon4py.model.testing import stencil_tests


class TestComputeVirtualTemperatureAndTemperature(stencil_tests.StencilTest):
    PROGRAM = compute_virtual_temperature_and_temperature
    OUTPUTS = ("virtual_temperature", "temperature")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        qv: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        theta_v: np.ndarray,
        exner: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        qsum = qc + qi + qr + qs + qg
        virtual_temperature = theta_v * exner
        temperature = virtual_temperature / (1.0 + phy_const.RV_O_RD_MINUS_1 * qv - qsum)
        return dict(
            virtual_temperature=virtual_temperature,
            temperature=temperature,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        theta_v = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=1.0e-4, high=1.0, dtype=ta.wpfloat
        )
        exner = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=1.0e-4, high=1.0, dtype=ta.wpfloat
        )
        qv = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat)
        qc = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat)
        qi = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat)
        qr = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat)
        qs = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat)
        qg = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat)
        virtual_temperature = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        temperature = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            qv=qv,
            qc=qc,
            qi=qi,
            qr=qr,
            qs=qs,
            qg=qg,
            theta_v=theta_v,
            exner=exner,
            virtual_temperature=virtual_temperature,
            temperature=temperature,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestComputeTemperatureFromInternalEnergyPerArea(stencil_tests.StencilTest):
    PROGRAM = compute_temperature_from_internal_energy_per_area
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        internal_energy_per_area: np.ndarray,
        qv: np.ndarray,
        qliq: np.ndarray,
        qice: np.ndarray,
        rho: np.ndarray,
        dz: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(out=np.full(internal_energy_per_area.shape, 255.75599999999997))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid):
        return dict(
            internal_energy_per_area=data_alloc.constant_field(
                38265357.270336017, dims.CellDim, dims.KDim, dtype=ta.wpfloat
            ),
            qv=data_alloc.constant_field(0.00122576, dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            qliq=data_alloc.constant_field(1.63837e-20, dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            qice=data_alloc.constant_field(1.09462e-08, dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            rho=data_alloc.constant_field(0.83444, dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            dz=data_alloc.constant_field(249.569, dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
            out=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
        )
