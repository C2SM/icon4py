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

import dataclasses
from typing import Any, Callable, Optional

import numpy as np
from gt4py import next as gtx
from gt4py.next.backend import ProgArgsInjector


try:
    import cupy as cp
    from gt4py.next.embedded.nd_array_field import CuPyArrayField
except ImportError:
    cp: Optional = None  # type:ignore[no-redef]

from gt4py.next.embedded.nd_array_field import NumPyArrayField
from gt4py.next.program_processors.runners.gtfn import extract_connectivity_args

from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl as add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_orig,
)
from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl as add_extra_diffusion_for_w_con_approaching_cfl_orig,
)
from icon4py.model.atmosphere.dycore.compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency as compute_advective_normal_wind_tendency_orig,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity as compute_horizontal_advection_term_for_vertical_velocity_orig,
)
from icon4py.model.atmosphere.dycore.compute_tangential_wind import (
    compute_tangential_wind as compute_tangential_wind_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels as interpolate_contravariant_vertical_velocity_to_full_levels_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import (
    interpolate_to_cell_center as interpolate_to_cell_center_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_to_ie_and_compute_ekin_on_edges as interpolate_vn_to_ie_and_compute_ekin_on_edges_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_vt_to_interface_edges import (
    interpolate_vt_to_interface_edges as interpolate_vt_to_interface_edges_orig,
)
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl as mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig,
)
from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl as mo_math_divrot_rot_vertex_ri_dsl_orig,
)
from icon4py.model.common.settings import device


def handle_numpy_integer(value):
    return int(value)


def handle_common_field(value, sizes):
    sizes.extend(value.shape)
    return value  # Return the value unmodified, but side-effect on sizes


def handle_default(value):
    return value  # Return the value unchanged


if cp:
    type_handlers = {
        np.integer: handle_numpy_integer,
        NumPyArrayField: handle_common_field,
        CuPyArrayField: handle_common_field,
    }
else:
    type_handlers = {
        np.integer: handle_numpy_integer,
        NumPyArrayField: handle_common_field,
    }


def process_arg(value, sizes):
    handler = type_handlers.get(type(value), handle_default)
    return handler(value, sizes) if handler == handle_common_field else handler(value)


@dataclasses.dataclass
class CachedProgram:
    program: gtx.ffront.decorator.Program
    with_domain: bool = True
    _compiled_program: Optional[Callable] = None
    _conn_args: Any = None
    _compiled_args: tuple = dataclasses.field(default_factory=tuple)

    @property
    def compiled_program(self) -> Callable:
        return self._compiled_program

    @property
    def conn_args(self) -> Callable:
        return self._conn_args

    def compile_the_program(
        self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any
    ) -> Callable:
        backend = self.program.backend
        transformer = backend.transforms_prog.replace(
            past_inject_args=ProgArgsInjector(
                args=args, kwargs=kwargs | {"offset_provider": offset_provider}
            )
        )
        program_call = transformer(self.program.definition_stage)
        self._compiled_args = program_call.args
        return backend.executor.otf_workflow(program_call)

    def __call__(self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any) -> None:
        if not self.compiled_program:
            self._compiled_program = self.compile_the_program(
                *args, offset_provider=offset_provider, **kwargs
            )
            self._conn_args = extract_connectivity_args(offset_provider, device)

        kwargs_as_tuples = tuple(kwargs.values())
        program_args = list(args) + list(kwargs_as_tuples)
        sizes = []

        # Convert numpy integers in args to int and handle gtx.common.Field
        for i in range(len(program_args)):
            program_args[i] = process_arg(program_args[i], sizes)

        if not self.with_domain:
            program_args.extend(sizes)

        # todo(samkellerhals): if we merge gt4py PR we can also pass connectivity args here conn_args=self.conn_args
        return self.compiled_program(*program_args, offset_provider=offset_provider)


# diffusion run stencils
add_extra_diffusion_for_normal_wind_tendency_approaching_cfl = CachedProgram(
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_orig
)
add_extra_diffusion_for_w_con_approaching_cfl = CachedProgram(
    add_extra_diffusion_for_w_con_approaching_cfl_orig
)
compute_advective_normal_wind_tendency = CachedProgram(compute_advective_normal_wind_tendency_orig)
compute_horizontal_advection_term_for_vertical_velocity = CachedProgram(
    compute_horizontal_advection_term_for_vertical_velocity_orig
)
compute_tangential_wind = CachedProgram(compute_tangential_wind_orig)
interpolate_contravariant_vertical_velocity_to_full_levels = CachedProgram(
    interpolate_contravariant_vertical_velocity_to_full_levels_orig
)
interpolate_to_cell_center = CachedProgram(interpolate_to_cell_center_orig)
interpolate_vn_to_ie_and_compute_ekin_on_edges = CachedProgram(
    interpolate_vn_to_ie_and_compute_ekin_on_edges_orig
)

interpolate_vt_to_interface_edges = CachedProgram(interpolate_vt_to_interface_edges_orig)

mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = CachedProgram(
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig
)

mo_math_divrot_rot_vertex_ri_dsl = CachedProgram(mo_math_divrot_rot_vertex_ri_dsl_orig)
