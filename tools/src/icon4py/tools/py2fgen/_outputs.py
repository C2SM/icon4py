# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import pathlib
from dataclasses import dataclass


class ArtifactKind(str, enum.Enum):
    PY = "py"
    F90 = "f90"
    C = "c"


@dataclass(frozen=True)
class OutputPlan:
    """Resolved on-disk targets for a py2fgen invocation.

    ``paths`` holds the path for each emitted source kind. ``h_path`` is the
    target for the C header when ``C`` is in the emit set, else ``None``.
    ``compile`` is True iff the shared library should be built.
    """

    paths: dict[ArtifactKind, pathlib.Path]
    h_path: pathlib.Path | None
    compile: bool


def resolve(
    *,
    library_name: str,
    output_path: pathlib.Path,
    overrides: dict[ArtifactKind, pathlib.Path | None],
    output_h: pathlib.Path | None,
    compile_lib: bool,
) -> OutputPlan:
    """Compute the emit set and resolve every artifact's on-disk path.

    - If any per-kind override is set, emit set = those kinds (selective mode).
    - Else emit set = all source kinds (default mode).
    Compilation runs iff ``compile_lib`` is True. Callers must reject the
    combination of ``compile_lib`` and any override before calling here.
    """
    explicit_overrides = {k: p for k, p in overrides.items() if p is not None}
    selective = bool(explicit_overrides) or output_h is not None

    if selective:
        kinds = set(explicit_overrides)
        if output_h is not None:
            kinds.add(ArtifactKind.C)
    else:
        kinds = set(ArtifactKind)

    paths: dict[ArtifactKind, pathlib.Path] = {}
    for kind in kinds:
        override = overrides.get(kind)
        if override is not None:
            paths[kind] = override
        else:
            paths[kind] = output_path / f"{library_name}.{kind.value}"

    h_path: pathlib.Path | None = None
    if ArtifactKind.C in kinds:
        if output_h is not None:
            h_path = output_h
        else:
            c_path = paths[ArtifactKind.C]
            h_path = c_path.with_suffix(".h")

    return OutputPlan(paths=paths, h_path=h_path, compile=compile_lib)
