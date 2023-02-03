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
from icon4py.liskov.codegen.write import IntegrationWriter
from icon4py.liskov.parsing.deserialise import DirectiveDeserialiser
from icon4py.liskov.parsing.parse import DirectivesParser
from icon4py.liskov.parsing.scan import DirectivesScanner


class LinearPipelineComposer:
    def __init__(self, steps):
        self.steps = steps

    def execute(self, data=None):
        for step in self.steps:
            data = step(data)
        return data


class LinearStep:  # todo: Steps should inherit from this
    pass


def run_parsing_pipeline(filepath: Path):
    steps = [
        DirectivesScanner(filepath),
        DirectivesParser(filepath),
        DirectiveDeserialiser(),
    ]
    composer = LinearPipelineComposer(steps)
    return composer.execute()


def run_code_generation_pipeline(parsed, filepath: Path, profile: bool):
    steps = [IntegrationGenerator(parsed, profile), IntegrationWriter(filepath)]
    composer = LinearPipelineComposer(steps)
    return composer.execute()
