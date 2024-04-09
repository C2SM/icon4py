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

from icon4pytools.liskov.codegen.integration.generate import IntegrationCodeGenerator
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

# TODO: fix tests to adapt to new custom output fields
from icon4pytools.liskov.codegen.serialisation.generate import SerialisationCodeGenerator
from icon4pytools.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    ImportData,
    InitData,
    Metadata,
    SavepointData,
    SerialisationCodeInterface,
)


@pytest.fixture
def integration_code_interface():
    start_stencil_data = StartStencilData(
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
        startln=1,
        acc_present=False,
        mergecopy=False,
        copies=True,
        optional_module="None",
    )
    end_stencil_data = EndStencilData(
        name="stencil1", startln=3, noendif=False, noprofile=False, noaccenddata=False
    )
    declare_data = DeclareData(
        startln=5,
        declarations={"field2": "(nproma, p_patch%nlev, p_patch%nblks_e)"},
        ident_type="REAL(wp)",
        suffix="before",
    )
    imports_data = ImportsData(startln=7)
    start_create_data = StartCreateData(extra_fields=["foo", "bar"], startln=9)
    end_create_data = EndCreateData(startln=11)
    endif_data = EndIfData(startln=12)
    start_profile_data = StartProfileData(startln=13, name="test_stencil")
    end_profile_data = EndProfileData(startln=14)
    insert_data = InsertData(startln=15, content="print *, 'Hello, World!'")
    start_fused_stencil_data = StartFusedStencilData(
        startln=16,
        name="fused_stencil",
        acc_present=False,
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
    )
    end_fused_stencil_data = EndFusedStencilData(startln=17, name="fused_stencil1")
    start_delete_data = StartDeleteData(startln=18)
    end_delete_data = EndDeleteData(startln=19)

    return IntegrationCodeInterface(
        StartStencil=[start_stencil_data],
        EndStencil=[end_stencil_data],
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
def expected_start_create_source():
    return """
!$ACC DATA CREATE( &
!$ACC   foo, &
!$ACC   bar ) &
!$ACC   IF ( i_am_accel_node )"""


@pytest.fixture
def expected_end_create_source():
    return "!$ACC END DATA"


@pytest.fixture
def expected_imports_source():
    return """\
  USE fused_stencil, ONLY: wrap_run_and_verify_fused_stencil
  USE stencil1, ONLY: wrap_run_and_verify_stencil1"""


@pytest.fixture
def expected_declare_source():
    return """
        ! DSL INPUT / OUTPUT FIELDS
        REAL(wp), DIMENSION((nproma, p_patch%nlev, p_patch%nblks_e)) :: field2_before"""


@pytest.fixture
def expected_start_stencil_source():
    return """
        !$ACC DATA CREATE( &
        !$ACC   out1_before, &
        !$ACC   out2_before, &
        !$ACC   out3_before, &
        !$ACC   out4_before, &
        !$ACC   out5_before, &
        !$ACC   out6_before ) &
        !$ACC      IF ( i_am_accel_node )

#ifdef __DSL_VERIFY
        !$ACC KERNELS IF( i_am_accel_node ) DEFAULT(NONE) ASYNC(1)
        out1_before(:, :) = out1(:, :, 1)
        out2_before(:, :, :) = p_nh%prog(nnew)%out2(:, :, :)
        out3_before(:, :) = p_nh%prog(nnew)%w(:, :, jb)
        out4_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, 2)
        out5_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, ntnd)
        out6_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, ntnd)
        !$ACC END KERNELS
        call nvtxStartRange("stencil1")"""


@pytest.fixture
def expected_end_stencil_source():
    return """
        call nvtxEndRange()
#endif
        call wrap_run_and_verify_stencil1( &
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
           horizontal_upper=10)

        !$ACC END DATA"""


@pytest.fixture
def expected_start_fused_stencil_source():
    return """
        !$ACC ENTER DATA CREATE( &
        !$ACC   out1_before, &
        !$ACC   out2_before, &
        !$ACC   out3_before, &
        !$ACC   out4_before, &
        !$ACC   out5_before, &
        !$ACC   out6_before ) &
        !$ACC      IF ( i_am_accel_node )

#ifdef __DSL_VERIFY
        !$ACC KERNELS IF( i_am_accel_node ) DEFAULT(PRESENT) ASYNC(1)
        out1_before(:, :) = out1(:, :, 1)
        out2_before(:, :, :) = p_nh%prog(nnew)%out2(:, :, :)
        out3_before(:, :) = p_nh%prog(nnew)%w(:, :, jb)
        out4_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, 2)
        out5_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, ntnd)
        out6_before(:, :, :) = p_nh%prog(nnew)%w(:, :, :, ntnd)
        !$ACC END KERNELS
#endif"""


@pytest.fixture
def expected_end_fused_stencil_source():
    return """
        call wrap_run_and_verify_fused_stencil( &
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
           horizontal_upper=10)

        !$ACC EXIT DATA DELETE( &
        !$ACC   out1_before, &
        !$ACC   out2_before, &
        !$ACC   out3_before, &
        !$ACC   out4_before, &
        !$ACC   out5_before, &
        !$ACC   out6_before ) &
        !$ACC      IF ( i_am_accel_node )"""


@pytest.fixture
def expected_start_delete_source():
    return "#ifdef __DSL_VERIFY"


@pytest.fixture
def expected_end_delete_source():
    return "#endif"


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
def integration_code_generator(integration_code_interface):
    return IntegrationCodeGenerator(
        integration_code_interface, profile=True, metadatagen=False, verification=True
    )


def test_integration_code_generation(
    integration_code_generator,
    expected_start_create_source,
    expected_end_create_source,
    expected_imports_source,
    expected_declare_source,
    expected_start_stencil_source,
    expected_end_stencil_source,
    expected_start_fused_stencil_source,
    expected_end_fused_stencil_source,
    expected_start_delete_source,
    expected_end_delete_source,
    expected_endif_source,
    expected_start_profile_source,
    expected_end_profile_source,
    expected_insert_source,
):
    # Check that the generated code snippets are as expected
    generated = integration_code_generator()
    assert len(generated) == 14
    assert generated[0].source == expected_start_create_source
    assert generated[1].source == expected_end_create_source
    assert generated[2].source == expected_imports_source
    assert generated[3].source == expected_declare_source
    assert generated[4].source == expected_start_stencil_source
    assert generated[5].source == expected_end_stencil_source
    assert generated[6].source == expected_start_fused_stencil_source
    assert generated[7].source == expected_end_fused_stencil_source
    assert generated[8].source == expected_start_delete_source
    assert generated[9].source == expected_end_delete_source
    assert generated[10].source == expected_endif_source
    assert generated[11].source == expected_start_profile_source
    assert generated[12].source == expected_end_profile_source
    assert generated[13].source == expected_insert_source


@pytest.fixture
def serialisation_code_interface():
    interface = {
        "Import": ImportData(startln=0),
        "Init": InitData(startln=1, directory=".", prefix="liskov-serialisation"),
        "Savepoint": [
            SavepointData(
                startln=9,
                subroutine="apply_nabla2_to_vn_in_lateral_boundary",
                intent="start",
                fields=[
                    FieldSerialisationData(
                        variable="z_nabla2_e",
                        association="z_nabla2_e(:,:,1)",
                        decomposed=False,
                        dimension=None,
                        typespec=None,
                        typename=None,
                        ptr_var=None,
                    ),
                ],
                metadata=[
                    Metadata(key="jstep", value="jstep_ptr"),
                    Metadata(key="diffctr", value="diffctr"),
                ],
            ),
            SavepointData(
                startln=38,
                subroutine="apply_nabla2_to_vn_in_lateral_boundary",
                intent="end",
                fields=[
                    FieldSerialisationData(
                        variable="z_nabla2_e",
                        association="z_nabla2_e(:,:,1)",
                        decomposed=False,
                        dimension=None,
                        typespec=None,
                        typename=None,
                        ptr_var=None,
                    ),
                    FieldSerialisationData(
                        variable="vn",
                        association="p_nh_prog%vn(:,:,1)",
                        decomposed=False,
                        dimension=None,
                        typespec=None,
                        typename=None,
                        ptr_var=None,
                    ),
                ],
                metadata=[
                    Metadata(key="jstep", value="jstep_ptr"),
                    Metadata(key="diffctr", value="diffctr"),
                ],
            ),
        ],
    }

    return SerialisationCodeInterface(**interface)


@pytest.fixture
def expected_savepoints():
    return [
        "  USE mo_mpi, ONLY: get_my_mpi_work_id",
        """
    !$ser init directory="." prefix="liskov-serialisation" mpi_rank=get_my_mpi_work_id()

    !$ser savepoint apply_nabla2_to_vn_in_lateral_boundary_start jstep=jstep_ptr diffctr=diffctr

    PRINT *, 'Serializing z_nabla2_e=z_nabla2_e(:,:,1)'

    !$ser data z_nabla2_e=z_nabla2_e(:,:,1)""",
        """
    !$ser savepoint apply_nabla2_to_vn_in_lateral_boundary_end jstep=jstep_ptr diffctr=diffctr

    PRINT *, 'Serializing z_nabla2_e=z_nabla2_e(:,:,1)'

    !$ser data z_nabla2_e=z_nabla2_e(:,:,1)

    PRINT *, 'Serializing vn=p_nh_prog%vn(:,:,1)'

    !$ser data vn=p_nh_prog%vn(:,:,1)""",
    ]


@pytest.mark.parametrize("multinode", [False, True])
def test_serialisation_code_generation(
    serialisation_code_interface, expected_savepoints, multinode
):
    generated = SerialisationCodeGenerator(serialisation_code_interface, multinode=multinode)()

    if multinode:
        assert len(generated) == 3
        assert generated[0].source == expected_savepoints[0]
        assert generated[1].source == expected_savepoints[1]
        assert generated[2].source == expected_savepoints[2]
    else:
        assert len(generated) == 2
        assert generated[1].source == expected_savepoints[2]
