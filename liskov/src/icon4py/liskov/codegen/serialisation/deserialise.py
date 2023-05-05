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

KEYS_TO_REMOVE = [
    "accpresent",
    "mergecopy",
    "copies",
    "horizontal_lower",
    "horizontal_upper",
    "vertical_lower",
    "vertical_upper",
    "name",
]

SKIP_VARS = ["ikoffset", "pos_on_tplane_e_1", "pos_on_tplane_e_2"]


class InitDataFactory:
    dtype = InitData

    def __call__(self, parsed: ts.ParsedDict) -> InitData:
        return self.dtype(startln=0, directory=".", prefix="liskov-serialisation")


class SavepointDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> list[SavepointData]:
        """Create a list of Start and End Savepoints for each Start and End Stencil directive."""
        start_stencil = extract_directive(
            parsed["directives"], icon4py.liskov.parsing.parse.StartStencil
        )
        end_stencil = extract_directive(
            parsed["directives"], icon4py.liskov.parsing.parse.EndStencil
        )

        deserialised = []

        for i, (start, end) in enumerate(zip(start_stencil, end_stencil)):
            named_args = parsed["content"]["StartStencil"][i]
            stencil_name = _extract_stencil_name(named_args, start)

            field_names = self._remove_unnecessary_keys(named_args)

            metadata = [
                Metadata(key=k, value=v)
                for k, v in self._get_timestep_variables(stencil_name).items()
            ]

            fields = self._make_fields(field_names)

            for intent, ln in [("start", start.startln), ("end", end.startln)]:
                savepoint = SavepointData(
                    subroutine=f"{stencil_name}",
                    intent=intent,
                    startln=ln,
                    fields=fields,
                    metadata=metadata,
                )
                deserialised.append(savepoint)

        return deserialised

    @staticmethod
    def _remove_unnecessary_keys(named_args: dict) -> dict:
        """Remove unnecessary keys from named_args, and only return field names."""
        copy = named_args.copy()
        [pop_item_from_dict(copy, k, None) for k in KEYS_TO_REMOVE]
        for tol in TOLERANCE_ARGS:
            for k in named_args.copy().keys():
                if k.endswith(tol):
                    pop_item_from_dict(named_args, k, None)
        return copy

    @staticmethod
    def _make_fields(named_args: dict) -> list[FieldSerialisationData]:
        """Create a list of FieldSerialisationData objects based on named arguments."""
        fields = [
            FieldSerialisationData(
                variable=variable,
                association="z_hydro_corr(:,:,1)"
                if association == "z_hydro_corr(:,nlev,1)"
                else association,
                decomposed=False,
                dimension=None,
                typespec=None,
                typename=None,
                ptr_var=None,
            )
            for variable, association in named_args.items()
            if variable not in SKIP_VARS
        ]
        return fields

    @staticmethod
    def _get_timestep_variables(stencil_name: str) -> dict:
        """Get the corresponding timestep metadata variables for the stencil."""
        timestep_variables = {
            "jstep": "jstep_ptr"
        }  # jstep is always included as it is the main timestep counter

        diffusion_stencil_names = [
            "apply",
            "calculate",
            "enhance",
            "update",
            "temporary",
            "diffusion",
        ]
        if any(name in stencil_name for name in diffusion_stencil_names):
            timestep_variables["diffctr"] = "diffctr"

        if "mo_velocity_advection" in stencil_name:
            timestep_variables["nstep"] = "nstep_ptr"
            timestep_variables["istep"] = "istep"

        if "mo_intp_rbf" in stencil_name:
            timestep_variables["mo_intp_rbf_ctr"] = "mo_intp_rbf_ctr"

        if "mo_math_divrot" in stencil_name:
            timestep_variables["mo_math_divrot_ctr"] = "mo_math_divrot_ctr"

        if "grad_green_gauss" in stencil_name:
            timestep_variables["grad_green_gauss_ctr"] = "grad_green_gauss_ctr"

        if "mo_icon_interpolation_scalar" in stencil_name:
            timestep_variables[
                "mo_icon_interpolation_ctr"
            ] = "mo_icon_interpolation_ctr"

        if "mo_advection_traj" in stencil_name:
            timestep_variables["mo_advection_traj_ctr"] = "mo_advection_traj_ctr"

        if "mo_solve_nonhydro" in stencil_name:
            timestep_variables["nstep"] = "nstep_ptr"
            timestep_variables["istep"] = "istep"
            timestep_variables["mo_solve_nonhydro_ctr"] = "mo_solve_nonhydro_ctr"

        return timestep_variables


class SerialisationCodeDeserialiser(Deserialiser):
    _FACTORIES = {
        "Init": InitDataFactory(),
        "Savepoint": SavepointDataFactory(),
    }
    _INTERFACE_TYPE = SerialisationCodeInterface
