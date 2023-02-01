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

import unittest

import pytest

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.codegen.deserialise import (
    EndCreateDataFactory,
    EndStencilDataFactory,
    ImportsDataFactory,
    StartCreateDataFactory,
    StartStencilDataFactory,
)
from icon4py.liskov.codegen.interface import (
    BoundsData,
    EndCreateData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    StartCreateData,
)
from icon4py.liskov.parsing.exceptions import (
    IncompatibleFieldError,
    MissingBoundsError,
    MissingDirectiveArgumentError,
)


@pytest.mark.parametrize(
    "factory_class, directive_type, startln, endln, string, expected",
    [
        (StartCreateDataFactory, ts.StartCreate, "start create", 1, 2, StartCreateData),
        (EndCreateDataFactory, ts.EndCreate, "end create", 3, 4, EndCreateData),
        (ImportsDataFactory, ts.Imports, "imports", 5, 6, ImportsData),
    ],
)
def test_data_factories(
    factory_class, directive_type, string, startln, endln, expected
):
    parsed = {
        "directives": [directive_type(string=string, startln=startln, endln=endln)],
        "content": {},
    }
    factory = factory_class()
    result = factory(parsed)
    assert isinstance(result, expected)
    assert result.startln == startln
    assert result.endln == endln


@pytest.mark.parametrize(
    "mock_data",
    [
        (
            {
                "directives": [
                    ts.EndStencil("END STENCIL(name=foo)", 5, 5),
                    ts.EndStencil("END STENCIL(name=bar)", 20, 20),
                ],
                "content": {"EndStencil": [{"name": "foo"}, {"name": "bar"}]},
            }
        ),
        (
            {
                "directives": [ts.EndStencil("END STENCIL(name=foo)", 5, 5)],
                "content": {"EndStencil": [{"name": "foo"}]},
            }
        ),
    ],
)
def test_end_stencil_data_factory(mock_data):
    factory = EndStencilDataFactory()
    result = factory(mock_data)
    assert all([isinstance(r, EndStencilData) for r in result])
    for r in result:
        assert hasattr(r, "name")


class TestStartStencilFactory(unittest.TestCase):
    def setUp(self):
        self.factory = StartStencilDataFactory()
        self.mock_fields = [
            FieldAssociationData("x", "i", 3, True, False),
            FieldAssociationData("y", "i", 3, True, False),
        ]

    def test_get_bounds(self):
        """Test that bounds are extracted correctly."""
        named_args = {
            "name": "stencil1",
            "horizontal_lower": 0,
            "horizontal_upper": 10,
            "vertical_lower": -5,
            "vertical_upper": 15,
        }
        assert self.factory._make_bounds(named_args) == BoundsData(0, 10, -5, 15)

    def test_get_bounds_missing_bounds(self):
        """Test that exception is raised if bounds are not provided."""
        named_args = {"name": "stencil1", "horizontal_upper": 10, "vertical_upper": 15}
        with pytest.raises(MissingBoundsError):
            self.factory._make_bounds(named_args)

    def test_get_field_associations(self):
        """Test that field associations are extracted correctly."""
        named_args = {
            "name": "mo_nh_diffusion_stencil_06",
            "z_nabla2_e": "z_nabla2_e(:,:,1)",
            "area_edge": "p_patch%edges%area_edge(:,1)",
            "fac_bdydiff_v": "fac_bdydiff_v",
            "vn": "p_nh_prog%vn(:,:,1)",
            "vertical_lower": "1",
            "vertical_upper": "nlev",
            "horizontal_lower": "i_startidx",
            "horizontal_upper": "i_endidx",
        }

        expected_fields = [
            FieldAssociationData("z_nabla2_e", "z_nabla2_e(:,:,1)", 3, True, False),
            FieldAssociationData(
                "area_edge", "p_patch%edges%area_edge(:,1)", 3, True, False
            ),
            FieldAssociationData("fac_bdydiff_v", "fac_bdydiff_v", 3, True, False),
            FieldAssociationData("vn", "p_nh_prog%vn(:,:,1)", 2, True, True),
        ]
        assert self.factory._make_fields(named_args) == expected_fields

    def test_missing_directive_argument_error(self):
        """Test that exception is raised if 'name' argument is not provided."""
        named_args = {
            "vn": "p_nh_prog%vn(:,:,1)",
            "vertical_lower": "1",
            "vertical_upper": "nlev",
            "horizontal_lower": "i_startidx",
            "horizontal_upper": "i_endidx",
        }
        with pytest.raises(MissingDirectiveArgumentError):
            self.factory._make_fields(named_args)

    def test_incompatible_field_error(self):
        named_args = {
            "name": "mo_nh_diffusion_stencil_06",
            "foo": "p_nh_prog%vn(:,:,1)",
            "vertical_lower": "1",
            "vertical_upper": "nlev",
            "horizontal_lower": "i_startidx",
            "horizontal_upper": "i_endidx",
        }
        with pytest.raises(IncompatibleFieldError):
            self.factory._make_fields(named_args)

    def test_update_field_tolerances(self):
        """Test that relative and absolute tolerances are set correctly for fields."""
        named_args = {
            "x_rel_tol": "0.01",
            "x_abs_tol": "0.1",
            "y_rel_tol": "0.001",
        }
        expected_fields = [
            FieldAssociationData(
                "x", "i", True, False, 3, rel_tol="0.01", abs_tol="0.1"
            ),
            FieldAssociationData("y", "i", True, False, 3, rel_tol="0.001"),
        ]
        assert (
            self.factory._update_tolerances(named_args, self.mock_fields)
            == expected_fields
        )

    def test_update_field_tolerances_not_all_fields(self):
        # Test that tolerance is not set for fields that are not provided in the named_args.
        named_args = {
            "x_rel_tol": "0.01",
            "x_abs_tol": "0.1",
        }
        expected_fields = [
            FieldAssociationData(
                "x", "i", True, False, 3, rel_tol="0.01", abs_tol="0.1"
            ),
            FieldAssociationData("y", "i", True, False, 3),
        ]
        assert (
            self.factory._update_tolerances(named_args, self.mock_fields)
            == expected_fields
        )

    def test_update_field_tolerances_no_tolerances(self):
        # Test that fields are not updated if named_args does not contain any tolerances.
        named_args = {}
        assert (
            self.factory._update_tolerances(named_args, self.mock_fields)
            == self.mock_fields
        )
