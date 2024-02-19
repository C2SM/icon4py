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


import pytest

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
    IntegrationCodeInterface,
    StartCreateData,
    StartDeleteData,
    StartFusedStencilData,
    StartProfileData,
    StartStencilData,
)
from icon4pytools.liskov.parsing.transform import (
    FusedStencilTransformer,
    OptionalModulesTransformer,
)


@pytest.fixture
def integration_code_interface():
    start_fused_stencil_data = StartFusedStencilData(
        name="fused_stencil1",
        fields=[
            FieldAssociationData("scalar1", "scalar1", inp=True, out=False, dims=None),
            FieldAssociationData("inp1", "inp1(:,:,1)", inp=True, out=False, dims=2),
            FieldAssociationData("out1", "out1(:,:,1)", inp=False, out=True, dims=2, abs_tol="0.5"),
            FieldAssociationData(
                "out2",
                "p_nh%prog(nnew)%out2(:,:,1)",
                inp=False,
                out=True,
                dims=3,
                abs_tol="0.2",
            ),
            FieldAssociationData("out3", "p_nh%prog(nnew)%w(:,:,jb)", inp=False, out=True, dims=2),
            FieldAssociationData("out4", "p_nh%prog(nnew)%w(:,:,1,2)", inp=False, out=True, dims=3),
            FieldAssociationData(
                "out5", "p_nh%prog(nnew)%w(:,:,:,ntnd)", inp=False, out=True, dims=3
            ),
            FieldAssociationData(
                "out6", "p_nh%prog(nnew)%w(:,:,1,ntnd)", inp=False, out=True, dims=3
            ),
        ],
        bounds=BoundsData("1", "10", "-1", "-10"),
        startln=1,
        acc_present=False,
    )
    end_fused_stencil_data = EndFusedStencilData(name="stencil1", startln=4)
    start_stencil_data1 = StartStencilData(
        name="stencil1",
        fields=[
            FieldAssociationData("scalar1", "scalar1", inp=True, out=False, dims=None),
            FieldAssociationData("inp1", "inp1(:,:,1)", inp=True, out=False, dims=2),
            FieldAssociationData("out1", "out1(:,:,1)", inp=False, out=True, dims=2, abs_tol="0.5"),
            FieldAssociationData(
                "out2",
                "p_nh%prog(nnew)%out2(:,:,1)",
                inp=False,
                out=True,
                dims=3,
                abs_tol="0.2",
            ),
            FieldAssociationData("out3", "p_nh%prog(nnew)%w(:,:,jb)", inp=False, out=True, dims=2),
            FieldAssociationData("out4", "p_nh%prog(nnew)%w(:,:,1,2)", inp=False, out=True, dims=3),
            FieldAssociationData(
                "out5", "p_nh%prog(nnew)%w(:,:,:,ntnd)", inp=False, out=True, dims=3
            ),
            FieldAssociationData(
                "out6", "p_nh%prog(nnew)%w(:,:,1,ntnd)", inp=False, out=True, dims=3
            ),
        ],
        bounds=BoundsData("1", "10", "-1", "-10"),
        startln=2,
        acc_present=False,
        mergecopy=False,
        copies=True,
        optional_module="None",
    )
    end_stencil_data1 = EndStencilData(
        name="stencil1", startln=3, noendif=False, noprofile=False, noaccenddata=False
    )
    start_stencil_data2 = StartStencilData(
        name="stencil2",
        fields=[
            FieldAssociationData("scalar1", "scalar1", inp=True, out=False, dims=None),
            FieldAssociationData("inp1", "inp1(:,:,1)", inp=True, out=False, dims=2),
            FieldAssociationData("out1", "out1(:,:,1)", inp=False, out=True, dims=2, abs_tol="0.5"),
            FieldAssociationData(
                "out2",
                "p_nh%prog(nnew)%out2(:,:,1)",
                inp=False,
                out=True,
                dims=3,
                abs_tol="0.2",
            ),
            FieldAssociationData("out3", "p_nh%prog(nnew)%w(:,:,jb)", inp=False, out=True, dims=2),
            FieldAssociationData("out4", "p_nh%prog(nnew)%w(:,:,1,2)", inp=False, out=True, dims=3),
            FieldAssociationData(
                "out5", "p_nh%prog(nnew)%w(:,:,:,ntnd)", inp=False, out=True, dims=3
            ),
            FieldAssociationData(
                "out6", "p_nh%prog(nnew)%w(:,:,1,ntnd)", inp=False, out=True, dims=3
            ),
        ],
        bounds=BoundsData("1", "10", "-1", "-10"),
        startln=5,
        acc_present=False,
        mergecopy=False,
        copies=True,
        optional_module="advection",
    )
    end_stencil_data2 = EndStencilData(
        name="stencil2", startln=6, noendif=False, noprofile=False, noaccenddata=False
    )
    declare_data = DeclareData(
        startln=7,
        declarations={"field2": "(nproma, p_patch%nlev, p_patch%nblks_e)"},
        ident_type="REAL(wp)",
        suffix="before",
    )
    imports_data = ImportsData(startln=8)
    start_create_data = StartCreateData(extra_fields=["foo", "bar"], startln=9)
    end_create_data = EndCreateData(startln=11)
    endif_data = EndIfData(startln=12)
    start_profile_data = StartProfileData(startln=13, name="test_stencil")
    end_profile_data = EndProfileData(startln=14)
    insert_data = InsertData(startln=15, content="print *, 'Hello, World!'")
    start_delete_data = StartDeleteData(startln=16)
    end_delete_data = EndDeleteData(startln=17)

    return IntegrationCodeInterface(
        StartStencil=[start_stencil_data1, start_stencil_data2],
        EndStencil=[end_stencil_data1, end_stencil_data2],
        StartFusedStencil=[start_fused_stencil_data],
        EndFusedStencil=[end_fused_stencil_data],
        StartDelete=[start_delete_data],
        EndDelete=[end_delete_data],
        Declare=[declare_data],
        Imports=imports_data,
        StartCreate=[start_create_data],
        EndCreate=[end_create_data],
        EndIf=[endif_data],
        StartProfile=[start_profile_data],
        EndProfile=[end_profile_data],
        Insert=[insert_data],
    )


@pytest.fixture
def fused_stencil_transform_fused(integration_code_interface):
    return FusedStencilTransformer(integration_code_interface, fused=True)


@pytest.fixture
def fused_stencil_transform_unfused(integration_code_interface):
    return FusedStencilTransformer(integration_code_interface, fused=False)


@pytest.fixture
def optional_modules_transform_enabled(integration_code_interface):
    return OptionalModulesTransformer(
        integration_code_interface, optional_modules_to_enable="advection"
    )


@pytest.fixture
def optional_modules_transform_disabled(integration_code_interface):
    return OptionalModulesTransformer(integration_code_interface, optional_modules_to_enable="no")


def test_transform_fused(
    fused_stencil_transform_fused,
):
    # Check that the transformed interface is as expected
    transformed = fused_stencil_transform_fused()
    assert len(transformed.StartFusedStencil) == 1
    assert len(transformed.EndFusedStencil) == 1
    assert len(transformed.StartStencil) == 1
    assert len(transformed.EndStencil) == 1
    assert len(transformed.StartDelete) == 2
    assert len(transformed.EndDelete) == 2


def test_transform_unfused(
    fused_stencil_transform_unfused,
):
    # Check that the transformed interface is as expected
    transformed = fused_stencil_transform_unfused()

    assert not transformed.StartFusedStencil
    assert not transformed.EndFusedStencil
    assert len(transformed.StartStencil) == 2
    assert len(transformed.EndStencil) == 2
    assert not transformed.StartDelete
    assert not transformed.EndDelete


def test_transform_optional_enabled(
    optional_modules_transform_enabled,
):
    # Check that the transformed interface is as expected
    transformed = optional_modules_transform_enabled()
    assert len(transformed.StartFusedStencil) == 1
    assert len(transformed.EndFusedStencil) == 1
    assert len(transformed.StartStencil) == 2
    assert len(transformed.EndStencil) == 2
    assert len(transformed.StartDelete) == 1
    assert len(transformed.EndDelete) == 1


def test_transform_optional_disabled(
    optional_modules_transform_disabled,
):
    # Check that the transformed interface is as expected
    transformed = optional_modules_transform_disabled()

    assert len(transformed.StartFusedStencil) == 1
    assert len(transformed.EndFusedStencil) == 1
    assert len(transformed.StartStencil) == 1
    assert len(transformed.EndStencil) == 1
    assert len(transformed.StartDelete) == 1
    assert len(transformed.EndDelete) == 1
