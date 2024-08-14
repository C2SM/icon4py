# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pytest

import icon4pytools.liskov.parsing.parse as ts
from icon4pytools.liskov.codegen.integration.deserialise import (
    DeclareDataFactory,
    EndCreateDataFactory,
    EndDeleteDataFactory,
    EndFusedStencilDataFactory,
    EndIfDataFactory,
    EndProfileDataFactory,
    EndStencilDataFactory,
    ImportsDataFactory,
    InsertDataFactory,
    StartCreateDataFactory,
    StartDeleteDataFactory,
    StartFusedStencilDataFactory,
    StartProfileDataFactory,
    StartStencilDataFactory,
)
from icon4pytools.liskov.codegen.integration.interface import (
    BoundsData,
    DeclareData,
    EndCreateData,
    EndDeleteData,
    EndFusedStencilData,
    EndIfData,
    EndProfileData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    InsertData,
    StartCreateData,
    StartDeleteData,
    StartProfileData,
)
from icon4pytools.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    MissingBoundsError,
    MissingDirectiveArgumentError,
)


@pytest.mark.parametrize(
    "factory_class, directive_type, string, startln, endln, expected",
    [
        (
            EndCreateDataFactory,
            ts.EndCreate,
            "END CREATE",
            2,
            2,
            EndCreateData,
        ),
        (
            ImportsDataFactory,
            ts.Imports,
            "IMPORTS",
            3,
            3,
            ImportsData,
        ),
        (
            EndIfDataFactory,
            ts.EndIf,
            "ENDIF",
            4,
            4,
            EndIfData,
        ),
        (
            EndProfileDataFactory,
            ts.EndProfile,
            "END PROFILE",
            5,
            5,
            EndProfileData,
        ),
        (
            StartDeleteDataFactory,
            ts.StartDelete,
            "START DELETE",
            6,
            6,
            StartDeleteData,
        ),
        (
            EndDeleteDataFactory,
            ts.EndDelete,
            "END DELETE",
            7,
            7,
            EndDeleteData,
        ),
    ],
)
def test_data_factories_no_args(factory_class, directive_type, string, startln, endln, expected):
    parsed = {
        "directives": [directive_type(string=string, startln=startln, endln=endln)],
        "content": {},
    }
    factory = factory_class()
    result = factory(parsed)

    if isinstance(result, list):
        result = result[0]

    assert isinstance(result, expected)
    assert result.startln == startln


@pytest.mark.parametrize(
    "factory,target,mock_data",
    [
        (
            EndStencilDataFactory,
            EndStencilData,
            {
                "directives": [
                    ts.EndStencil("END STENCIL(name=foo)", 5, 5),
                    ts.EndStencil("END STENCIL(name=bar; noendif=true; noprofile=true)", 20, 20),
                ],
                "content": {
                    "EndStencil": [
                        {"name": "foo"},
                        {"name": "bar", "noendif": "true", "noprofile": "true"},
                    ]
                },
            },
        ),
        (
            EndStencilDataFactory,
            EndStencilData,
            {
                "directives": [ts.EndStencil("END STENCIL(name=foo; noprofile=true)", 5, 5)],
                "content": {"EndStencil": [{"name": "foo"}]},
            },
        ),
        (
            EndFusedStencilDataFactory,
            EndFusedStencilData,
            {
                "directives": [
                    ts.EndFusedStencil("END FUSED STENCIL(name=foo)", 5, 5),
                    ts.EndFusedStencil("END FUSED STENCIL(name=bar)", 20, 20),
                ],
                "content": {
                    "EndFusedStencil": [
                        {"name": "foo"},
                        {"name": "bar"},
                    ]
                },
            },
        ),
        (
            StartProfileDataFactory,
            StartProfileData,
            {
                "directives": [
                    ts.StartProfile("START PROFILE(name=foo)", 5, 5),
                    ts.StartProfile("START PROFILE(name=bar)", 20, 20),
                ],
                "content": {"StartProfile": [{"name": "foo"}, {"name": "bar"}]},
            },
        ),
        (
            StartProfileDataFactory,
            StartProfileData,
            {
                "directives": [ts.StartProfile("START PROFILE(name=foo)", 5, 5)],
                "content": {"StartProfile": [{"name": "foo"}]},
            },
        ),
        (
            DeclareDataFactory,
            DeclareData,
            {
                "directives": [
                    ts.Declare(
                        "DECLARE(vn=nlev,nblks_c; w=nlevp1,nblks_e; suffix=dsl; type=LOGICAL)",
                        5,
                        5,
                    )
                ],
                "content": {
                    "Declare": [
                        {
                            "vn": "nlev,nblks_c",
                            "w": "nlevp1,nblks_e",
                            "suffix": "dsl",
                            "type": "LOGICAL",
                        }
                    ]
                },
            },
        ),
        (
            InsertDataFactory,
            InsertData,
            {
                "directives": [ts.Insert("INSERT(content=foo)", 5, 5)],
                "content": {"Insert": ["foo"]},
            },
        ),
    ],
)
def test_data_factories_with_args(factory, target, mock_data):
    factory_init = factory()
    result = factory_init(mock_data)
    assert all([isinstance(r, target) for r in result])


@pytest.mark.parametrize(
    "mock_data, extra_fields",
    [
        (
            {
                "directives": [ts.StartCreate("START CREATE(extra_fields=foo)", 5, 5)],
                "content": {"StartCreate": [{"extra_fields": "foo"}]},
            },
            ["foo"],
        ),
        (
            {
                "directives": [ts.StartCreate("START CREATE(extra_fields=foo,xyz)", 5, 5)],
                "content": {"StartCreate": [{"extra_fields": "foo,xyz"}]},
            },
            ["foo", "xyz"],
        ),
        (
            {
                "directives": [ts.StartCreate("START CREATE()", 5, 5)],
                "content": {"StartCreate": [None]},
            },
            None,
        ),
    ],
)
def test_start_create_factory(mock_data, extra_fields):
    factory = StartCreateDataFactory()
    result = factory(mock_data)
    if isinstance(result, list):
        result = result[0]
    assert isinstance(result, StartCreateData)
    assert result.extra_fields == extra_fields


@pytest.mark.parametrize(
    "factory,target,mock_data",
    [
        (
            EndStencilDataFactory,
            EndStencilData,
            {
                "directives": [
                    ts.EndStencil("END STENCIL(name=foo)", 5, 5),
                    ts.EndStencil("END STENCIL(name=bar; noendif=foo)", 20, 20),
                ],
                "content": {"EndStencil": [{"name": "foo"}, {"name": "bar", "noendif": "foo"}]},
            },
        ),
    ],
)
def test_data_factories_invalid_args(factory, target, mock_data):
    factory_init = factory()
    with pytest.raises(DirectiveSyntaxError):
        factory_init(mock_data)


class TestStartStencilFactory(unittest.TestCase):
    def setUp(self):
        self.factory = StartStencilDataFactory()
        self.mock_fields = [
            FieldAssociationData("x", "i", 3),
            FieldAssociationData("y", "i", 3),
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
        dimensions = {"z_nabla2_e": 3, "area_edge": 3, "fac_bdydiff_v": 3, "vn": 2}

        expected_fields = [
            FieldAssociationData("z_nabla2_e", "z_nabla2_e(:,:,1)", 3),
            FieldAssociationData("area_edge", "p_patch%edges%area_edge(:,1)", 3),
            FieldAssociationData("fac_bdydiff_v", "fac_bdydiff_v", 3),
            FieldAssociationData("vn", "p_nh_prog%vn(:,:,1)", 2),
        ]
        assert self.factory._make_fields(named_args, dimensions) == expected_fields

    def test_missing_directive_argument_error(self):
        """Test that exception is raised if 'name' argument is not provided."""
        named_args = {
            "vn": "p_nh_prog%vn(:,:,1)",
            "vertical_lower": "1",
            "vertical_upper": "nlev",
            "horizontal_lower": "i_startidx",
            "horizontal_upper": "i_endidx",
        }
        dimensions = {"vn": 2}
        with pytest.raises(MissingDirectiveArgumentError):
            self.factory._make_fields(named_args, dimensions)

    def test_update_field_tolerances(self):
        """Test that relative and absolute tolerances are set correctly for fields."""
        named_args = {
            "x_rel_tol": "0.01",
            "x_abs_tol": "0.1",
            "y_rel_tol": "0.001",
        }
        expected_fields = [
            FieldAssociationData("x", "i", 3, rel_tol="0.01", abs_tol="0.1"),
            FieldAssociationData("y", "i", 3, rel_tol="0.001"),
        ]
        assert self.factory._update_tolerances(named_args, self.mock_fields) == expected_fields

    def test_update_field_tolerances_not_all_fields(self):
        # Test that tolerance is not set for fields that are not provided in the named_args.
        named_args = {
            "x_rel_tol": "0.01",
            "x_abs_tol": "0.1",
        }
        expected_fields = [
            FieldAssociationData("x", "i", 3, rel_tol="0.01", abs_tol="0.1"),
            FieldAssociationData("y", "i", 3),
        ]
        assert self.factory._update_tolerances(named_args, self.mock_fields) == expected_fields

    def test_update_field_tolerances_no_tolerances(self):
        # Test that fields are not updated if named_args does not contain any tolerances.
        named_args = {}
        assert self.factory._update_tolerances(named_args, self.mock_fields) == self.mock_fields


class TestStartFusedStencilFactory(TestStartStencilFactory):
    def setUp(self):
        self.factory = StartFusedStencilDataFactory()
        self.mock_fields = [
            FieldAssociationData("x", "i", 3),
            FieldAssociationData("y", "i", 3),
        ]
