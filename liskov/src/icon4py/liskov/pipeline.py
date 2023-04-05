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

from icon4py.liskov.codegen.integration.generate import IntegrationGenerator
from icon4py.liskov.codegen.integration.interface import DeserialisedDirectives
from icon4py.liskov.codegen.serialisation.deserialise import (
    SerialisationDeserialiser,
)
from icon4py.liskov.codegen.serialisation.generate import SerialisationGenerator
from icon4py.liskov.codegen.serialisation.interface import SerialisationInterface
from icon4py.liskov.codegen.write import CodegenWriter
from icon4py.liskov.common import Step, linear_pipeline
from icon4py.liskov.external.gt4py import UpdateFieldsWithGt4PyStencils
from icon4py.liskov.parsing.deserialise import DirectiveDeserialiser
from icon4py.liskov.parsing.parse import DirectivesParser
from icon4py.liskov.parsing.scan import DirectivesScanner


SERIALISERS = {
    "integration": DirectiveDeserialiser,
    "serialisation": SerialisationDeserialiser,
}


@linear_pipeline
def parse_fortran_file(
    input_filepath: Path, output_filepath: Path, deserialiser_type: str
) -> list[Step]:
    """Execute a pipeline to parse and deserialize directives from a file.

        The pipeline consists of three steps: DirectivesScanner, DirectivesParser, and
        DirectiveDeserialiser. The DirectivesScanner scans the file for directives,
        the DirectivesParser parses the directives into a dictionary, and the
        DirectiveDeserialiser deserializes the dictionary into a
        DeserialisedDirectives object.

    Args:
        input_filepath: Path to the input file to process.
        output_filepath: Path to the output file to generate.
        deserialiser_type: What deserialiser to use.

    Returns:
        DeserialisedDirectives: The deserialized directives object.
    """
    deserialiser = SERIALISERS[deserialiser_type]

    return [
        DirectivesScanner(input_filepath),
        DirectivesParser(input_filepath, output_filepath),
        deserialiser(),
    ]


@linear_pipeline
def load_gt4py_stencils(parsed: DeserialisedDirectives) -> list[Step]:
    """Execute a pipeline to update fields of a DeserialisedDirectives object with GT4Py stencils.

    Args:
        parsed: The input DeserialisedDirectives object.

    Returns:
        The updated object with fields containing information from GT4Py stencils.
    """
    return [UpdateFieldsWithGt4PyStencils(parsed)]


@linear_pipeline
def run_integration_code_generation(
    parsed: DeserialisedDirectives,
    input_filepath: Path,
    output_filepath: Path,
    profile: bool,
    metadatagen: bool,
) -> list[Step]:
    """Execute a pipeline to generate and write code for a set of directives.

    The pipeline consists of two steps: IntegrationGenerator and IntegrationWriter. The IntegrationGenerator generates
    code based on the parsed directives and profile flag. The IntegrationWriter writes the generated code to the
    specified filepath.

    Args:
        parsed: The deserialized directives object.
        input_filepath: The original file containing the DSL preprocessor directives.
        output_filepath: The file path to write the generated code to.
        profile: A flag to indicate if profiling information should be included in the generated code.
        metadatagen: A flag to indicate if a metadata header should be included in the generated code.
    """
    return [
        IntegrationGenerator(parsed, profile, metadatagen),
        CodegenWriter(input_filepath, output_filepath),
    ]


@linear_pipeline
def run_serialisation_code_generation(
    ser_iface: SerialisationInterface,
    input_filepath: Path,
    output_filepath: Path,
) -> list[Step]:
    """Execute a pipeline to generate and write serialisation statements.

    Args:
        ser_iface: The serialisation interface.
        input_filepath: The original file containing the DSL preprocessor directives.
        output_filepath: The file path to write the generated code to.
    """
    return [
        SerialisationGenerator(ser_iface),
        CodegenWriter(input_filepath, output_filepath),
    ]
