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

from icon4py.common.dimension import C2E2CODim, CellDim, EdgeDim, KDim, VertexDim
from icon4py.liskov.codegen.generate import IntegrationGenerator
from icon4py.liskov.codegen.interface import (
    BoundsData,
    DeclareData,
    DeserialisedDirectives,
    EndCreateData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    StartCreateData,
    StartStencilData,
)


@pytest.fixture
def serialised_directives():
    start_stencil_data = StartStencilData(
        name="stencil1",
        fields=[
            FieldAssociationData("scalar1", "scalar1", True, False, None),
            FieldAssociationData(
                "inp1", "inp1(:,:,1)", True, False, [CellDim, C2E2CODim]
            ),
            FieldAssociationData(
                "out1", "out1(:,:,1)", False, True, [CellDim, KDim], abs_tol="0.5"
            ),
            FieldAssociationData(
                "out2",
                "p_nh%prog(nnew)%out2(:,:,1)",
                False,
                True,
                [VertexDim, KDim],
                abs_tol="0.2",
            ),
            FieldAssociationData(
                "out3", "p_nh%prog(nnew)%w(:,:,jb)", False, True, [KDim]
            ),
            FieldAssociationData(
                "out4", "p_nh%prog(nnew)%w(:,:,1,2)", False, True, [EdgeDim, KDim]
            ),
            FieldAssociationData(
                "out5", "p_nh%prog(nnew)%w(:,:,:,ntnd)", False, True, [VertexDim, KDim]
            ),
            FieldAssociationData(
                "out6", "p_nh%prog(nnew)%w(:,:,1,ntnd)", False, True, [CellDim, KDim]
            ),
        ],
        bounds=BoundsData("1", "10", "-1", "-10"),
        startln=1,
        endln=2,
    )
    end_stencil_data = EndStencilData(name="stencil1", startln=3, endln=4)
    declare_data = DeclareData(
        startln=5,
        endln=6,
        declarations=[{"field2": "(nproma, p_patch%nlev, p_patch%nblks_e)"}],
    )
    imports_data = ImportsData(startln=7, endln=8)
    start_create_data = StartCreateData(startln=9, endln=10)
    end_create_data = EndCreateData(startln=11, endln=11)

    return DeserialisedDirectives(
        StartStencil=[start_stencil_data],
        EndStencil=[end_stencil_data],
        Declare=declare_data,
        Imports=imports_data,
        StartCreate=start_create_data,
        EndCreate=end_create_data,
    )


@pytest.fixture
def expected_start_create_source():
    return """
#ifdef __DSL_VERIFY
        dsl_verify = .TRUE.
#elif
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
        !$ACC      IF ( i_am_accel_node .AND. acc_on .AND. dsl_verify)"""


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
        REAL(wp), DIMENSION((nproma, p_patch%nlev, p_patch%nblks_e)) :: field2_before
        LOGICAL :: dsl_verify"""


@pytest.fixture
def expected_start_stencil_source():
    return """
#ifdef __DSL_VERIFY
        !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
        out1_before(:, :, :) = out1(:, :, :)
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
           out1_before=out1_before(:, :, 1), &
           out2=p_nh%prog(nnew)%out2(:, :, 1), &
           out2_before=out2_before(:, :, 1), &
           out3=p_nh%prog(nnew)%w(:, :, jb), &
           out3_before=out3_before(:, 1), &
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


# todo: look into this: out3_before=out3_before(:, :), & check if this works (out3_before(:, 1))


@pytest.fixture
def generator(serialised_directives):
    return IntegrationGenerator(serialised_directives, profile=True)


def test_generate(
    generator,
    expected_start_create_source,
    expected_end_create_source,
    expected_imports_source,
    expected_declare_source,
    expected_start_stencil_source,
    expected_end_stencil_source,
):
    # Check that the generated code snippets are as expected
    assert len(generator.generated) == 6
    assert generator.generated[0].source == expected_start_create_source
    assert generator.generated[1].source == expected_end_create_source
    assert generator.generated[2].source == expected_imports_source
    assert generator.generated[3].source == expected_declare_source
    assert generator.generated[4].source == expected_start_stencil_source
    assert generator.generated[5].source == expected_end_stencil_source
