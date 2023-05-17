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

from icon4py.liskov.codegen.integration.deserialise import (
    IntegrationCodeDeserialiser,
)
from icon4py.liskov.codegen.integration.generate import IntegrationCodeGenerator
from icon4py.liskov.codegen.integration.interface import IntegrationCodeInterface
from icon4py.liskov.codegen.serialisation.deserialise import (
    SerialisationCodeDeserialiser,
)
from icon4py.liskov.codegen.serialisation.generate import (
    SerialisationCodeGenerator,
)
from icon4py.liskov.codegen.shared.writer import CodegenWriter
from icon4py.liskov.external.gt4py import UpdateFieldsWithGt4PyStencils
from icon4py.liskov.parsing.parse import DirectivesParser
from icon4py.liskov.parsing.scan import DirectivesScanner
from icon4py.liskov.pipeline.definition import Step, linear_pipeline


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
    **kwargs,
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
def load_gt4py_stencils(parsed: IntegrationCodeInterface) -> list[Step]:
    """Execute a pipeline to update fields of a IntegrationCodeInterface object with GT4Py stencils.

    Args:
        parsed: The input IntegrationCodeInterface object.

    Returns:
        The updated object with fields containing information from GT4Py stencils.
    """
    return [UpdateFieldsWithGt4PyStencils(parsed)]


@linear_pipeline
def run_code_generation(
    input_filepath: Path,
    output_filepath: Path,
    codegen_type: str,
    *args,
    **kwargs,
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
