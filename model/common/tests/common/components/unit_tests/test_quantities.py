# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.components import framework as fw, quantities as qty
from icon4py.model.common.states import data as cf_data


PLACES = {
    "OnCell": (dims.CellDim,),
    "OnCellK": (dims.CellDim, dims.KDim),
    "OnCellKHalf": (dims.CellDim, dims.KHalfDim),
    "OnEdgeK": (dims.EdgeDim, dims.KDim),
    "OnEdgeKHalf": (dims.EdgeDim, dims.KHalfDim),
}


def _quantities() -> list[type[fw.Quantity]]:
    return [
        cls
        for _, cls in inspect.getmembers(qty, inspect.isclass)
        if issubclass(cls, fw.Quantity)
        and cls.__module__ == qty.__name__
        and cls is not fw.Tendency
    ]


@pytest.mark.parametrize("quantity", _quantities(), ids=lambda q: q.__name__)
def test_quantity_is_named_after_its_place(quantity: type[fw.Quantity]) -> None:
    place = next(suffix for suffix in PLACES if quantity.__name__.endswith(suffix))
    assert quantity.dims == PLACES[place]
    assert isinstance(quantity.units, str)
    assert quantity.precision in ("wp", "vp")


def test_the_cf_attributes_are_those_of_the_tables() -> None:
    for quantity, attrs in (
        (qty.RhoOnCellK, cf_data.PROGNOSTIC_CF_ATTRIBUTES["air_density"]),
        (qty.VnOnEdgeK, cf_data.PROGNOSTIC_CF_ATTRIBUTES["normal_velocity"]),
        (qty.QvOnCellK, cf_data.COMMON_TRACER_CF_ATTRIBUTES["qv"]),
        (qty.TemperatureOnCellK, cf_data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"]),
        (qty.SurfacePressureOnCell, cf_data.DIAGNOSTIC_CF_ATTRIBUTES["surface_pressure"]),
        (qty.TendencyOfQvOnCellK, cf_data.TENDENCY_CF_ATTRIBUTES["qv"]),
        (qty.PrecipitationFluxOnCellK, cf_data.PRECIPITATION_CF_ATTRIBUTES["precipitation_flux"]),
    ):
        assert quantity.standard_name == attrs.standard_name
        assert quantity.units == attrs.units
        assert quantity.long_name == attrs.long_name


def test_tendencies_are_marked() -> None:
    assert issubclass(qty.TendencyOfTemperatureOnCellK, fw.Tendency)
    assert issubclass(qty.TendencyOfVnOnEdgeK, fw.Tendency)
    assert not issubclass(qty.TemperatureOnCellK, fw.Tendency)


def test_the_dycore_precisions_follow_the_diagnostic_state() -> None:
    assert qty.TangentialWindOnEdgeK.precision == "vp"
    assert qty.NormalWindAdvectiveTendencyOnEdgeK.precision == "vp"
    assert qty.ExnerDynamicalIncrementOnCellK.precision == "vp"
    assert qty.ThetaVOnCellKHalf.precision == "wp"
    assert qty.HorizontalWindDeformationOnCellKHalf.precision == "vp"
