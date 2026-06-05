# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any, Final

from icon4py.model.common.utils import fortran_config
from icon4py.model.standalone_driver.testcases import gauss3d, jablonowski_williamson as jabw
from icon4py.model.standalone_driver.testcases.from_file import FromFileParameters, read_from_file


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common import type_alias as ta
    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import geometry as grid_geometry, icon as icon_grid
    from icon4py.model.common.interpolation import interpolation_factory
    from icon4py.model.common.metrics import metrics_factory
    from icon4py.model.standalone_driver import driver_states


_SER_DATA_SUBDIR: Final = "ser_data"


def _params_from_dict(cls: type, source: dict[str, Any]):
    """Construct a dataclass from a namelist dict.

    Unknown keys are ignored (e.g. topography params mixed into the same nml block).
    Missing keys fall back to the dataclass field defaults.
    Fortran→Python name translation is driven by the required ``_fortran_name_map``
    class variable: ``{fortran_key: python_field_name}``.
    """
    name_map: dict[str, str] = cls._fortran_name_map  # type: ignore[attr-defined]
    known_fields = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in source.items():
        python_name = name_map.get(key, key)
        if python_name in known_fields:
            kwargs[python_name] = value
    return cls(**kwargs)


@dataclasses.dataclass
class InitialConditionConfig:
    parameters: (
        jabw.JablonowskiWilliamsonParameters | gauss3d.Gauss3DParameters | FromFileParameters
    )

    @classmethod
    def from_fortran_dict(
        cls,
        atm_dict: dict[str, Any],
        input_dict: dict[str, Any],
        *,
        data_path: pathlib.Path | None = None,
    ) -> InitialConditionConfig | None:
        run_nml = atm_dict.get("run_nml", {})
        if not run_nml.get("ltestcase", False):
            if data_path is None:
                return None
            ntracer = fortran_config.list_to_value(run_nml.get("ntracer", 0))
            return cls(
                parameters=FromFileParameters(
                    data_path=data_path / _SER_DATA_SUBDIR,
                    ntracer=ntracer,
                ),
            )

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        match testcase_nml.get("nh_test_name"):
            case "jabw" | "jabw_s":
                parameters = _params_from_dict(jabw.JablonowskiWilliamsonParameters, testcase_nml)
            case "gauss3D":
                parameters = _params_from_dict(gauss3d.Gauss3DParameters, testcase_nml)
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(parameters=parameters)


def create(
    *,
    config: InitialConditionConfig,
    grid: icon_grid.IconGrid,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend: gtx_typing.Backend | None,
    lowest_layer_thickness: ta.wpfloat,
    model_top_height: ta.wpfloat,
    stretch_factor: ta.wpfloat,
    damping_height: ta.wpfloat,
    exchange: decomposition_defs.ExchangeRuntime,
) -> driver_states.DriverStates:
    """Create initial driver states by dispatching on the type of ``config.parameters``."""
    kwargs: dict[str, Any] = dict(
        grid=grid,
        geometry_field_source=geometry_field_source,
        interpolation_field_source=interpolation_field_source,
        metrics_field_source=metrics_field_source,
        backend=backend,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        damping_height=damping_height,
        exchange=exchange,
    )
    match config.parameters:
        case jabw.JablonowskiWilliamsonParameters():
            return jabw.jablonowski_williamson(parameters=config.parameters, **kwargs)
        case gauss3d.Gauss3DParameters():
            return gauss3d.gauss3d(parameters=config.parameters, **kwargs)
        case FromFileParameters():
            return read_from_file(parameters=config.parameters, **kwargs)
        case _:
            raise TypeError(f"Unknown IC parameters type: {type(config.parameters)!r}")
