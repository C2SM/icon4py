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
import icon4py.liskov.parsing.parse
import icon4py.liskov.parsing.types as ts
from icon4py.common.logger import setup_logger
from icon4py.liskov.codegen.integration.deserialise import (
    TOLERANCE_ARGS,
    _extract_stencil_name,
    pop_item_from_dict,
)
from icon4py.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    InitData,
    Metadata,
    SavepointData,
    SerialisationCodeInterface,
)
from icon4py.liskov.codegen.shared.deserialiser import Deserialiser
from icon4py.liskov.parsing.utils import extract_directive


logger = setup_logger(__name__)


class InitDataFactory:
    dtype = InitData

    def __call__(self, parsed: ts.ParsedDict) -> InitData:
        return self.dtype(startln=0, directory=".", prefix="liskov-serialisation")


class SavepointDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> list[SavepointData]:
        start_stencil = extract_directive(
            parsed["directives"], icon4py.liskov.parsing.parse.StartStencil
        )
        end_stencil = extract_directive(
            parsed["directives"], icon4py.liskov.parsing.parse.EndStencil
        )

        deserialised = []

        # todo: remove rel and abs tol values
        # todo: handle merge copies
        # todo: allow insertion of additional metadata?

        for i, (start, end) in enumerate(zip(start_stencil, end_stencil)):
            named_args = parsed["content"]["StartStencil"][i]
            stencil_name = _extract_stencil_name(named_args, start)
            to_remove = [
                "accpresent",
                "mergecopy",
                "copies",
                "horizontal_lower",
                "horizontal_upper",
                "vertical_lower",
                "vertical_upper",
                "name",
            ]
            [pop_item_from_dict(named_args, k, None) for k in to_remove]

            for tol in TOLERANCE_ARGS:
                for k in named_args.copy().keys():
                    if k.endswith(tol):
                        pop_item_from_dict(named_args, k, None)

            fields = [
                FieldSerialisationData(
                    variable=variable,
                    association=association,
                    decomposed=False,
                    dimension=None,
                    typespec=None,
                    typename=None,
                    ptr_var=None,
                )
                for variable, association in named_args.items()
            ]

            # todo: make this into a function
            timestep_variables = {}

            if "mo_velocity_advection" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["nstep"] = "nstep_ptr"
                timestep_variables["istep"] = "istep"

            diffusion_stencil_names = [
                "apply",
                "calculate",
                "enhance",
                "update",
                "temporary",
                "diffusion",
            ]

            if any([name in stencil_name for name in diffusion_stencil_names]):
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["diffctr"] = "diffctr"

            if "mo_intp_rbf" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["mo_intp_rbf_ctr"] = "mo_intp_rbf_ctr"

            if "mo_math_divrot" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["mo_math_divrot_ctr"] = "mo_math_divrot_ctr"

            if "grad_green_gauss" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["grad_green_gauss_ctr"] = "grad_green_gauss_ctr"

            if "mo_icon_interpolation_scalar" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables[
                    "mo_icon_interpolation_ctr"
                ] = "mo_icon_interpolation_ctr"

            if "mo_advection_traj" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["mo_advection_traj_ctr"] = "mo_advection_traj_ctr"

            if "mo_solve_nonhydro" in stencil_name:
                timestep_variables["jstep"] = "jstep_ptr"
                timestep_variables["nstep"] = "nstep_ptr"
                timestep_variables["mo_solve_nonhydro_ctr"] = "mo_solve_nonhydro_ctr"
                timestep_variables["istep"] = "istep"

            timestep_metadata = [
                Metadata(key=k, value=v) for k, v in timestep_variables.items()
            ]

            deserialised.append(
                SavepointData(
                    subroutine=f"{stencil_name}",
                    intent="start",
                    startln=start.startln,
                    fields=fields,
                    metadata=timestep_metadata,
                )
            )

            deserialised.append(
                SavepointData(
                    subroutine=f"{stencil_name}",
                    intent="end",
                    startln=end.startln,
                    fields=fields,
                    metadata=timestep_metadata,
                )
            )
        return deserialised


class SerialisationCodeDeserialiser(Deserialiser):
    _FACTORIES = {
        "Init": InitDataFactory(),
        "Savepoint": SavepointDataFactory(),
    }
    _INTERFACE_TYPE = SerialisationCodeInterface
