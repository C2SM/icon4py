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
from pathlib import Path
from typing import Any

from icon4pytools.liskov.codegen.integration.deserialise import IntegrationCodeDeserialiser
from icon4pytools.liskov.codegen.integration.generate import IntegrationCodeGenerator
from icon4pytools.liskov.codegen.integration.interface import IntegrationCodeInterface
from icon4pytools.liskov.codegen.serialisation.deserialise import SerialisationCodeDeserialiser
from icon4pytools.liskov.codegen.serialisation.generate import SerialisationCodeGenerator
from icon4pytools.liskov.codegen.shared.write import CodegenWriter
from icon4pytools.liskov.external.gt4py import UpdateFieldsWithGt4PyStencils
from icon4pytools.liskov.parsing.parse import DirectivesParser
from icon4pytools.liskov.parsing.scan import DirectivesScanner
from icon4pytools.liskov.parsing.transform import (
    FusedStencilTransformer,
    OptionalModulesTransformer,
)
from icon4pytools.liskov.pipeline.definition import Step, linear_pipeline


DESERIALISERS = {
    "integration": IntegrationCodeDeserialiser,
    "serialisation": SerialisationCodeDeserialiser,
}

CODEGENS = {
    "integration": IntegrationCodeGenerator,
    "serialisation": SerialisationCodeGenerator,
}


@linear_pipeline
def parse_fortran_file(
    input_filepath: Path,
    output_filepath: Path,
    deserialiser_type: str,
    **kwargs: Any,
) -> list[Step]:
    """Execute a pipeline to parse and deserialize directives from a file.

        The pipeline consists of three steps: DirectivesScanner, DirectivesParser, and
        DirectiveDeserialiser. The DirectivesScanner scans the file for directives,
        the DirectivesParser parses the directives into a dictionary, and the
        DirectiveDeserialiser deserializes the dictionary into a
        its corresponding Interface object.

    Args:
        input_filepath: Path to the input file to process.
        output_filepath: Path to the output file to generate.
        deserialiser_type: What deserialiser to use.

    Returns:
        IntegrationCodeInterface | SerialisationCodeInterface: The interface object.
    """
    deserialiser = DESERIALISERS[deserialiser_type]

    return [
        DirectivesScanner(input_filepath),
        DirectivesParser(input_filepath, output_filepath),
        deserialiser(**kwargs),
    ]


@linear_pipeline
def process_stencils(
    parsed: IntegrationCodeInterface, fused: bool, optional_modules_to_enable: list
) -> list[Step]:
    """Execute a linear pipeline to transform stencils and produce either fused or unfused execution.

    This function takes an input `parsed` object of type `IntegrationCodeInterface` and a `fused` boolean flag.
    It then executes a linear pipeline, consisting of two steps: transformation of stencils for fusion or unfusion,
    and updating fields with information from GT4Py stencils.

    Args:
        parsed (IntegrationCodeInterface): The input object containing parsed integration code.
        fused (bool): A boolean flag indicating whether to produce fused (True) or unfused (False) execution.

    Returns:
        The updated and transformed object with fields containing information from GT4Py stencils.
    """
    if optional_modules_to_enable == [False]:
        return [
            FusedStencilTransformer(parsed, fused),
            UpdateFieldsWithGt4PyStencils(parsed),
        ]
    else:
        return [
            FusedStencilTransformer(parsed, fused),
            OptionalModulesTransformer(parsed, optional_modules_to_enable),
            UpdateFieldsWithGt4PyStencils(parsed),
        ]


@linear_pipeline
def run_code_generation(
    input_filepath: Path,
    output_filepath: Path,
    codegen_type: str,
    *args: Any,
    **kwargs: Any,
) -> list[Step]:
    """Execute a pipeline to generate and write code.

    Args:
        input_filepath: The original file containing the DSL preprocessor directives.
        output_filepath: The file path to write the generated code to.
        codegen_type: Which type of code generator to use.

    Note:
        Additional positional and keyword arguments are passed to the code generator.
    """
    code_generator = CODEGENS[codegen_type]

    return [
        code_generator(*args, **kwargs),
        CodegenWriter(input_filepath, output_filepath),
    ]
