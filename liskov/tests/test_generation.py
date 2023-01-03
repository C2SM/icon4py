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
    CreateData,
    DeclareData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    SerialisedDirectives,
    StartStencilData,
)


@pytest.fixture
def serialised_directives():
    start_stencil_data = StartStencilData(
        name="stencil1",
        fields=[
            FieldAssociationData("field1", "field1(:, :, 1)", True, False),
            FieldAssociationData(
                "field2", "field2(:, :, 1)", False, True, abs_tol="0.5"
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
    create_data = CreateData(startln=9, endln=10)

    return SerialisedDirectives(
        start=[start_stencil_data],
        end=[end_stencil_data],
        declare=declare_data,
        imports=imports_data,
        create=create_data,
    )


@pytest.fixture
def expected_create_source():
    return """
#ifdef __DSL_VERIFY
        LOGICAL dsl_verify = .TRUE.
#elif
        LOGICAL dsl_verify = .FALSE.
#endif

        !$ACC DATA CREATE( &
        !$ACC   field2_before, &
        !$ACC   ) &
        !$ACC      IF ( i_am_accel_node .AND. acc_on .AND. dsl_verify)"""


@pytest.fixture
def expected_imports_source():
    return "USE stencil1, ONLY: wrap_run_stencil1"


@pytest.fixture
def expected_declare_source():
    return """
        ! DSL INPUT / OUTPUT FIELDS
        REAL(wp), DIMENSION((nproma, p_patch%nlev, p_patch%nblks_e)) :: field2_before"""


@pytest.fixture
def expected_stencil_start_source():
    return """
#ifdef __DSL_VERIFY
        !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
        field2_before(:, :, :) = field2(:, :, :)
        !$ACC END PARALLEL
        call nvtxStartRange("stencil1")"""


@pytest.fixture
def expected_stencil_end_source():
    return """
        call nvtxEndRange()
#endif
        call wrap_run_stencil1( &
           field1=field1(:, :, 1), &
           field2_before=field2_before(:, :, 1), &
           field2_abs_tol=0.5, &
           vertical_lower=-1, &
           vertical_upper=-10, &
           horizontal_lower=1, &
           horizontal_upper=10
        )"""


@pytest.fixture
def generator(serialised_directives):
    return IntegrationGenerator(serialised_directives, profile=True)


def test_generate(
    generator,
    expected_create_source,
    expected_imports_source,
    expected_declare_source,
    expected_stencil_start_source,
    expected_stencil_end_source,
):
    # Check that the generated code snippets are as expected
    assert len(generator.generated) == 5
    assert generator.generated[0].source == expected_create_source
    assert generator.generated[1].source == expected_imports_source
    assert generator.generated[2].source == expected_declare_source
    assert generator.generated[3].source == expected_stencil_start_source
    assert generator.generated[4].source == expected_stencil_end_source
