# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import types
import importlib
from typing import Optional

from gt4py import eve
from gt4py.next.ffront import program_ast as past
from gt4py.next.ffront.decorator import FieldOperator, Program


def _get_domain_arg_ids(fvprog: Program) -> set[Optional[eve.concepts.SymbolRef]]:
    """Collect all argument names that are used within the 'domain' keyword argument."""
    domain_arg_ids = []
    if "domain" in fvprog.past_stage.past_node.body[0].kwargs.keys():
        domain_arg = fvprog.past_stage.past_node.body[0].kwargs["domain"]
        assert isinstance(domain_arg, past.Dict)
        for arg in domain_arg.values_:
            for arg_elt in arg.elts:
                if isinstance(arg_elt, past.Name):
                    domain_arg_ids.append(arg_elt.id)
    return set(domain_arg_ids)


def import_definition(name: str) -> Program | FieldOperator | types.FunctionType:
    """Import a stencil from a given module.

    Note:
        The stencil program and module are assumed to have the same name.
    """
    module_name, member_name = name.split(":")
    program = getattr(importlib.import_module(module_name), member_name)
    return program
