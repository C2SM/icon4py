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

from icon4py.liskov.codegen.generate import IntegrationGenerator
from icon4py.liskov.codegen.interface import (
    BoundsData,
    DeclareData,
    DeserialisedDirectives,
    EndCreateData,
    EndIfData,
    EndProfileData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    InsertData,
    StartCreateData,
    StartProfileData,
    StartStencilData,
)


# TODO: fix tests to adapt to new custom output fields
@pytest.fixture
def serialised_directives():
    start_stencil_data = StartStencilData(
        name="stencil1",
        fields=[
            FieldAssociationData("scalar1", "scalar1", inp=True, out=False, dims=None),
            FieldAssociationData("inp1", "inp1(:,:,1)", inp=True, out=False, dims=2),
            FieldAssociationData(
                "out1", "out1(:,:,1)", inp=False, out=True, dims=2, abs_tol="0.5"
            ),
            FieldAssociationData(
                "out2",
                "p_nh%prog(nnew)%out2(:,:,1)",
                inp=False,
                out=True,
                dims=3,
                abs_tol="0.2",
            ),
            FieldAssociationData(
                "out3", "p_nh%prog(nnew)%w(:,:,jb)", inp=False, out=True, dims=2
            ),
            FieldAssociationData(
                "out4", "p_nh%prog(nnew)%w(:,:,1,2)", inp=False, out=True, dims=3
            ),
            FieldAssociationData(
                "out5", "p_nh%prog(nnew)%w(:,:,:,ntnd)", inp=False, out=True, dims=3
            ),
            FieldAssociationData(
                "out6", "p_nh%prog(nnew)%w(:,:,1,ntnd)", inp=False, out=True, dims=3
            ),
        ],
        bounds=BoundsData("1", "10", "-1", "-10"),
        startln=1,
        endln=2,
        acc_present=False,
        mergecopy=False,
        copies=True,
    )
    end_stencil_data = EndStencilData(
        name="stencil1", startln=3, endln=4, noendif=False, noprofile=False
    )
    declare_data = DeclareData(
        startln=5,
        endln=6,
        declarations={"field2": "(nproma, p_patch%nlev, p_patch%nblks_e)"},
        ident_type="REAL(wp)",
        suffix="before",
    )
    imports_data = ImportsData(startln=7, endln=8)
    start_create_data = StartCreateData(startln=9, endln=10)
    end_create_data = EndCreateData(startln=11, endln=11)
    endif_data = EndIfData(startln=12, endln=12)
    start_profile_data = StartProfileData(startln=13, endln=13, name="test_stencil")
    end_profile_data = EndProfileData(startln=14, endln=14)
    insert_data = InsertData(startln=15, endln=15, content="print *, 'Hello, World!'")

    return DeserialisedDirectives(
        StartStencil=[start_stencil_data],
        EndStencil=[end_stencil_data],
        Declare=[declare_data],
        Imports=imports_data,
        StartCreate=start_create_data,
        EndCreate=end_create_data,
        EndIf=[endif_data],
        StartProfile=[start_profile_data],
        EndProfile=[end_profile_data],
        Insert=[insert_data],
    )


@pytest.fixture
def expected_start_create_source():
    return """
#ifdef __DSL_VERIFY
        dsl_verify = .TRUE.
#else
        dsl_verify = .FALSE.
#endif

        !$ACC DATA CREATE( &
        !$ACC   out1_before, &
        !$ACC   out2_before, &
        !$ACC   out3_before, &
        !$ACC   out4_before, &
        !$ACC   out5_before, &
        !$ACC   out6_before &
        !$ACC   ), &
        !$ACC      IF ( i_am_accel_node .AND. dsl_verify)"""


@pytest.fixture
def expected_end_create_source():
    return "!$ACC END DATA"


@pytest.fixture
def expected_imports_source():
    return "  USE stencil1, ONLY: wrap_run_stencil1"


@pytest.fixture
def expected_declare_source():
    return """
        ! DSL INPUT / OUTPUT FIELDS
        REAL(wp), DIMENSION((nproma, p_patch%nlev, p_patch%nblks_e)) :: field2_before"""


@pytest.fixture
def expected_start_stencil_source():
    return """
#ifdef __DSL_VERIFY
        !$ACC PARALLEL IF( i_am_accel_node ) DEFAULT(NONE) ASYNC(1)
        out1_before(:, :) = out1(:, :, 1)
        out2_before(:, :, :) = p_nh%prog(nnew)%out2(:, :, :)
        out3_before(:, :) = p_nh%prog(nnew)%w(:, :, jb)
        out4_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, 2)
        out5_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, ntnd)
        out6_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, ntnd)
        !$ACC END PARALLEL
        call nvtxStartRange("stencil1")"""


@pytest.fixture
def expected_end_stencil_source():
    return """
        call nvtxEndRange()
#endif
        call wrap_run_stencil1( &
           scalar1=scalar1, &
           inp1=inp1(:, :, 1), &
           out1=out1(:, :, 1), &
           out1_before=out1_before(:, :), &
           out2=p_nh%prog(nnew)%out2(:, :, 1), &
           out2_before=out2_before(:, :, 1), &
           out3=p_nh%prog(nnew)%w(:, :, jb), &
           out3_before=out3_before(:, :), &
           out4=p_nh%prog(nnew)%w(:, :, 1, 2), &
           out4_before=out4_before(:, :, 1), &
           out5=p_nh%prog(nnew)%w(:, :, :, ntnd), &
           out5_before=out5_before(:, :, 1), &
           out6=p_nh%prog(nnew)%w(:, :, 1, ntnd), &
           out6_before=out6_before(:, :, 1), &
           out1_abs_tol=0.5, &
           out2_abs_tol=0.2, &
           vertical_lower=-1, &
           vertical_upper=-10, &
           horizontal_lower=1, &
           horizontal_upper=10)"""


@pytest.fixture
def expected_endif_source():
    return "#endif"


@pytest.fixture
def expected_start_profile_source():
    return 'call nvtxStartRange("test_stencil")'


@pytest.fixture
def expected_end_profile_source():
    return "call nvtxEndRange()"


@pytest.fixture
def expected_insert_source():
    return "print *, 'Hello, World!'"


@pytest.fixture
def generator(serialised_directives):
    return IntegrationGenerator(serialised_directives, profile=True, metadata_gen=False)


def test_generate(
    generator,
    expected_start_create_source,
    expected_end_create_source,
    expected_imports_source,
    expected_declare_source,
    expected_start_stencil_source,
    expected_end_stencil_source,
    expected_endif_source,
    expected_start_profile_source,
    expected_end_profile_source,
    expected_insert_source,
):
    # Check that the generated code snippets are as expected
    generated = generator()
    assert len(generated) == 10
    assert generated[0].source == expected_start_create_source
    assert generated[1].source == expected_end_create_source
    assert generated[2].source == expected_imports_source
    assert generated[3].source == expected_declare_source
    assert generated[4].source == expected_start_stencil_source
    assert generated[5].source == expected_end_stencil_source
    assert generated[6].source == expected_endif_source
    assert generated[7].source == expected_start_profile_source
    assert generated[8].source == expected_end_profile_source
    assert generated[9].source == expected_insert_source
