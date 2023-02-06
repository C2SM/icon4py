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

from icon4py.liskov.codegen.generate import IntegrationGenerator
from icon4py.liskov.codegen.interface import DeserialisedDirectives
from icon4py.liskov.codegen.write import IntegrationWriter
from icon4py.liskov.common import exec_pipeline
from icon4py.liskov.external.gt4py import UpdateFieldsWithGt4PyStencils
from icon4py.liskov.parsing.deserialise import DirectiveDeserialiser
from icon4py.liskov.parsing.parse import DirectivesParser
from icon4py.liskov.parsing.scan import DirectivesScanner


def parsing_pipeline(filepath: Path) -> DeserialisedDirectives:
    """Execute a pipeline to parse and deserialize directives from a file.

        The pipeline consists of three steps: DirectivesScanner, DirectivesParser, and
        DirectiveDeserialiser. The DirectivesScanner scans the file for directives,
        the DirectivesParser parses the directives into a dictionary, and the
        DirectiveDeserialiser deserializes the dictionary into a
        DeserialisedDirectives object.

    Args:
        filepath (Path): The file path of the directives file.

    Returns:
        DeserialisedDirectives: The deserialized directives object.
    """
    steps = [
        DirectivesScanner(filepath),
        DirectivesParser(filepath),
        DirectiveDeserialiser(),
    ]
    return exec_pipeline(steps)


def gt4py_pipeline(parsed: DeserialisedDirectives) -> DeserialisedDirectives:
    """Execute a pipeline to update fields of a DeserialisedDirectives object with GT4Py stencils.

    Args:
        parsed: The input DeserialisedDirectives object.

    Returns:
        The updated object with fields containing information from GT4Py stencils.
    """
    steps = [UpdateFieldsWithGt4PyStencils(parsed)]
    return exec_pipeline(steps)


def codegen_pipeline(
    parsed: DeserialisedDirectives, filepath: Path, profile: bool
) -> None:
    """Execute a pipeline to generate and write code for a set of directives.

    The pipeline consists of two steps: IntegrationGenerator and IntegrationWriter. The IntegrationGenerator generates
    code based on the parsed directives and profile flag. The IntegrationWriter writes the generated code to the
    specified filepath.

    Args:
        parsed: The deserialized directives object.
        filepath: The file path to write the generated code to.
        profile: A flag to indicate if profiling information should be included in the generated code.
    """
    steps = [IntegrationGenerator(parsed, profile), IntegrationWriter(filepath)]
    return exec_pipeline(steps)
