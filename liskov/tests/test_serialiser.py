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

# todo: add tests for serialiser
#   - check that each factory produces objects as expected.
import pytest

from icon4py.liskov.codegen.interface import BoundsData, FieldAssociationData
from icon4py.liskov.parsing.exceptions import (
    IncompatibleFieldError,
    MissingBoundsError,
    MissingDirectiveArgumentError,
)
from icon4py.liskov.parsing.serialise import StartStencilDataFactory


@pytest.fixture
def start_stencil_factory():
    return StartStencilDataFactory()


def test_get_bounds(start_stencil_factory):
    """Test that bounds are extracted correctly."""
    named_args = {
        "name": "stencil1",
        "horizontal_lower": 0,
        "horizontal_upper": 10,
        "vertical_lower": -5,
        "vertical_upper": 15,
    }
    assert start_stencil_factory._get_bounds(named_args) == BoundsData(0, 10, -5, 15)


def test_get_bounds_missing_bounds(start_stencil_factory):
    """Test that exception is raised if bounds are not provided."""
    named_args = {"name": "stencil1", "horizontal_upper": 10, "vertical_upper": 15}
    with pytest.raises(MissingBoundsError):
        start_stencil_factory._get_bounds(named_args)


def test_get_field_associations(start_stencil_factory):
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
        FieldAssociationData("z_nabla2_e", "z_nabla2_e(:,:,1)", True, False),
        FieldAssociationData("area_edge", "p_patch%edges%area_edge(:,1)", True, False),
        FieldAssociationData("fac_bdydiff_v", "fac_bdydiff_v", True, False),
        FieldAssociationData("vn", "p_nh_prog%vn(:,:,1)", True, True),
    ]
    assert start_stencil_factory._get_field_associations(named_args) == expected_fields


def test_missing_directive_argument_error(start_stencil_factory):
    """Test that exception is raised if 'name' argument is not provided."""
    named_args = {
        "vn": "p_nh_prog%vn(:,:,1)",
        "vertical_lower": "1",
        "vertical_upper": "nlev",
        "horizontal_lower": "i_startidx",
        "horizontal_upper": "i_endidx",
    }
    with pytest.raises(MissingDirectiveArgumentError):
        start_stencil_factory._get_field_associations(named_args)


def test_incompatible_field_error(start_stencil_factory):
    named_args = {
        "name": "mo_nh_diffusion_stencil_06",
        "foo": "p_nh_prog%vn(:,:,1)",
        "vertical_lower": "1",
        "vertical_upper": "nlev",
        "horizontal_lower": "i_startidx",
        "horizontal_upper": "i_endidx",
    }
    with pytest.raises(IncompatibleFieldError):
        start_stencil_factory._get_field_associations(named_args)


@pytest.fixture
def mock_fields():
    return [
        FieldAssociationData("x", "i", True, False),
        FieldAssociationData("y", "i", True, False),
    ]


def test_update_field_tolerances(start_stencil_factory, mock_fields):
    """Test that relative and absolute tolerances are set correctly for fields."""
    named_args = {
        "x_rel_tol": "0.01",
        "x_abs_tol": "0.1",
        "y_rel_tol": "0.001",
    }
    expected_fields = [
        FieldAssociationData("x", "i", True, False, rel_tol="0.01", abs_tol="0.1"),
        FieldAssociationData("y", "i", True, False, rel_tol="0.001"),
    ]
    assert (
        start_stencil_factory._update_field_tolerances(named_args, mock_fields)
        == expected_fields
    )


def test_update_field_tolerances_not_all_fields(start_stencil_factory, mock_fields):
    # Test that tolerance is not set for fields that are not provided in the named_args.
    named_args = {
        "x_rel_tol": "0.01",
        "x_abs_tol": "0.1",
    }
    expected_fields = [
        FieldAssociationData("x", "i", True, False, rel_tol="0.01", abs_tol="0.1"),
        FieldAssociationData("y", "i", True, False),
    ]
    assert (
        start_stencil_factory._update_field_tolerances(named_args, mock_fields)
        == expected_fields
    )


def test_update_field_tolerances_no_tolerances(start_stencil_factory, mock_fields):
    # Test that fields are not updated if named_args does not contain any tolerances.
    named_args = {}
    assert (
        start_stencil_factory._update_field_tolerances(named_args, mock_fields)
        == mock_fields
    )
