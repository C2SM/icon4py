#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate an OpenCode config for the weekly Slack summary."""

from __future__ import annotations

import json
import pathlib
from typing import Annotated

import typer


cli = typer.Typer(help=__doc__)


@cli.command(name="generate-weekly-summary-opencode-config")
def generate_cmd(
    output_path: Annotated[
        pathlib.Path,
        typer.Option(
            "--output-path",
            help="Path where the generated OpenCode config JSON is written.",
        ),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL of the CSCS Inference API.",
        ),
    ],
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            help="Model ID to use, e.g. google/gemma-4-31B-it.",
        ),
    ],
) -> None:
    """Write an OpenCode config for the weekly Slack summary."""
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"cscs-inference/{model_id}",
        "provider": {
            "cscs-inference": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "CSCS Inference",
                "options": {
                    "baseURL": base_url,
                    "apiKey": "{env:CSCS_INFERENCE_API_KEY}",
                },
                "models": {
                    model_id: {
                        "name": model_id,
                    }
                },
            }
        },
    }

    output_path.write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    cli()
