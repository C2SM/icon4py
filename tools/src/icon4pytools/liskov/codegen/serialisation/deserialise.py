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
import uuid
from typing import Callable, ClassVar

import icon4pytools.liskov.parsing.parse
import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.integration.deserialise import (
    TOLERANCE_ARGS,
    DeclareDataFactory,
    _extract_stencil_name,
    pop_item_from_dict,
)
from icon4pytools.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    ImportData,
    InitData,
    Metadata,
    SavepointData,
    SerialisationCodeInterface,
)
from icon4pytools.liskov.codegen.shared.deserialise import Deserialiser
from icon4pytools.liskov.parsing.utils import extract_directive


logger = setup_logger(__name__)

KEYS_TO_REMOVE = [
    "accpresent",
    "mergecopy",
    "copies",
    "optional_module",
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
    dtype = SavepointData

    def __call__(self, parsed: ts.ParsedDict) -> list[SavepointData]:
        """Create a list of Start and End Savepoints for each Start and End Stencil directive."""
        start_stencil = extract_directive(
            parsed["directives"], icon4pytools.liskov.parsing.parse.StartStencil
        )
        end_stencil = extract_directive(
            parsed["directives"], icon4pytools.liskov.parsing.parse.EndStencil
        )
        gpu_fields = self.get_gpu_fields(parsed)

        repeated = self._find_repeated_stencils(parsed["content"])

        deserialised = []

        for i, (start, end) in enumerate(zip(start_stencil, end_stencil, strict=False)):
            named_args = parsed["content"]["StartStencil"][i]
            stencil_name = _extract_stencil_name(named_args, start)

            if stencil_name in repeated:
                stencil_name = f"{stencil_name}_{uuid.uuid4()!s}"

            field_names = self._remove_unnecessary_keys(named_args)

            metadata = [
                Metadata(key=k, value=v)
                for k, v in self._get_timestep_variables(stencil_name).items()
            ]

            fields = self._make_fields(field_names, gpu_fields)

            for intent, ln in [("start", start.startln), ("end", end.startln)]:
                savepoint = self.dtype(
                    subroutine=f"{stencil_name}",
                    intent=intent,
                    startln=ln,
                    fields=fields,
                    metadata=metadata,
                )
                deserialised.append(savepoint)

        return deserialised

    def get_gpu_fields(self, parsed: ts.ParsedDict) -> set[str]:
        """Get declared fields which will be loaded on GPU and thus need to be serialised using accdata."""
        declare = DeclareDataFactory()(parsed)
        fields = []
        for d in declare:
            for f in d.declarations:
                fields.append(f)
        return set(fields)

    @staticmethod
    def _remove_unnecessary_keys(named_args: dict) -> dict:
        """Remove unnecessary keys from named_args, and only return field names."""
        copy = named_args.copy()
        [pop_item_from_dict(copy, k, None) for k in KEYS_TO_REMOVE]
        for tol in TOLERANCE_ARGS:
            for k in copy.copy().keys():
                if k.endswith(tol):
                    pop_item_from_dict(copy, k, None)
        return copy

    @staticmethod
    def _make_fields(named_args: dict, gpu_fields: set) -> list[FieldSerialisationData]:
        """Create a list of FieldSerialisationData objects based on named arguments."""
        fields = [
            FieldSerialisationData(
                variable=variable,
                association="z_hydro_corr(:,:,1)"  # special case
                if association == "z_hydro_corr(:,nlev,1)"
                else association,
                decomposed=False,
                dimension=None,
                typespec=None,
                typename=None,
                ptr_var=None,
                device="gpu" if variable in gpu_fields else "cpu",
            )
            for variable, association in named_args.items()
            if variable not in SKIP_VARS
        ]
        return fields

    @staticmethod
    def _get_timestep_variables(stencil_name: str) -> dict:
        """Get the corresponding timestep metadata variables for the stencil."""
        timestep_variables = {}

        diffusion_stencil_names = [
            "apply",
            "calculate",
            "enhance",
            "update",
            "temporary",
            "diffusion",
        ]
        if any(name in stencil_name for name in diffusion_stencil_names):
            timestep_variables["jstep"] = "jstep_ptr"
            timestep_variables["diffctr"] = "diffctr"

        if "mo_velocity_advection" in stencil_name:
            timestep_variables["jstep"] = "jstep_ptr"
            timestep_variables["nstep"] = "nstep_ptr"
            timestep_variables["istep"] = "istep"

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
            timestep_variables["mo_icon_interpolation_ctr"] = "mo_icon_interpolation_ctr"

        if "mo_advection_traj" in stencil_name:
            timestep_variables["jstep"] = "jstep_ptr"
            timestep_variables["mo_advection_traj_ctr"] = "mo_advection_traj_ctr"

        if "mo_solve_nonhydro" in stencil_name:
            timestep_variables["istep"] = "istep"
            timestep_variables["mo_solve_nonhydro_ctr"] = "mo_solve_nonhydro_ctr"

        return timestep_variables

    def _find_repeated_stencils(self, content: dict) -> set[str]:
        stencil_names: dict[str, str] = {}
        repeated_names = []
        for stencil in content["StartStencil"]:
            name = stencil["name"]
            if name in stencil_names:
                if stencil_names[name] not in repeated_names:
                    repeated_names.append(stencil_names[name])
                repeated_names.append(name)
            else:
                stencil_names[name] = name
        return set(repeated_names)


class ImportDataFactory:
    dtype = ImportData

    def __call__(self, parsed: ts.ParsedDict) -> ImportData:
        imports = extract_directive(
            parsed["directives"], icon4pytools.liskov.parsing.parse.Imports
        )[0]
        return self.dtype(startln=imports.startln)


class SerialisationCodeDeserialiser(Deserialiser):
    _FACTORIES: ClassVar[dict[str, Callable]] = {
        "Init": InitDataFactory(),
        "Savepoint": SavepointDataFactory(),
        "Import": ImportDataFactory(),
    }
    _INTERFACE_TYPE = SerialisationCodeInterface
