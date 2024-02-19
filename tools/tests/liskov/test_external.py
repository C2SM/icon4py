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

import os
from pathlib import Path

import pytest
from gt4py.next.ffront.decorator import Program

from icon4pytools.liskov.codegen.integration.interface import (
    FieldAssociationData,
    IntegrationCodeInterface,
    StartStencilData,
)
from icon4pytools.liskov.external.exceptions import IncompatibleFieldError, UnknownStencilError
from icon4pytools.liskov.external.gt4py import UpdateFieldsWithGt4PyStencils


def test_stencil_collector():
    name = "calculate_nabla4"
    updater = UpdateFieldsWithGt4PyStencils(None)
    assert isinstance(updater._collect_icon4py_stencil(name), Program)


def test_stencil_collector_invalid_module():
    name = "non_existent_module"
    updater = UpdateFieldsWithGt4PyStencils(None)
    with pytest.raises(UnknownStencilError, match=r"Did not find module: (\w*)"):
        updater._collect_icon4py_stencil(name)


def test_stencil_collector_invalid_member():
    from icon4py.model.atmosphere.diffusion.stencils import apply_nabla2_to_w

    module_path = Path(apply_nabla2_to_w.__file__)
    parents = module_path.parents[0]

    updater = UpdateFieldsWithGt4PyStencils(None)

    path = os.path.join(parents, "foo.py")
    with open(path, "w") as f:
        f.write("")

    with pytest.raises(UnknownStencilError, match=r"Did not find module member: (\w*)"):
        updater._collect_icon4py_stencil("foo")

    os.remove(path)


mock_deserialised_directives = IntegrationCodeInterface(
    StartStencil=[
        StartStencilData(
            name="apply_nabla2_to_w",
            fields=[
                FieldAssociationData(
                    variable="incompatible_field_name",
                    association="z_nabla2_e(:,:,1)",
                    dims=None,
                    abs_tol=None,
                    rel_tol=None,
                    inp=None,
                    out=None,
                )
            ],
            bounds=None,
            startln=None,
            acc_present=False,
            mergecopy=False,
            copies=True,
            optional_module="None"
        )
    ],
    Imports=None,
    Declare=None,
    EndStencil=None,
    StartFusedStencil=None,
    EndFusedStencil=None,
    StartDelete=None,
    EndDelete=None,
    StartCreate=None,
    EndCreate=None,
    EndIf=None,
    StartProfile=None,
    EndProfile=None,
    Insert=None,
)


def test_incompatible_field_error():
    updater = UpdateFieldsWithGt4PyStencils(mock_deserialised_directives)
    with pytest.raises(IncompatibleFieldError):
        updater()
