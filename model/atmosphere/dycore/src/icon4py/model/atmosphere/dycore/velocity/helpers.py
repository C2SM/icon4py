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

from gt4py import next as gtx

from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl as add_extra_diffusion_for_w_con_approaching_cfl_orig,
)
from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_wn_approaching_cfl import (
    add_extra_diffusion_for_wn_approaching_cfl as add_extra_diffusion_for_wn_approaching_cfl_orig,
)
from icon4py.model.atmosphere.dycore.compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency as compute_advective_normal_wind_tendency_orig,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity as compute_horizontal_advection_term_for_vertical_velocity_orig,
)
from icon4py.model.atmosphere.dycore.compute_tangential_wind import compute_tangential_wind as compute_tangential_wind_orig
from icon4py.model.atmosphere.dycore.interpolate_contravatiant_vertical_verlocity_to_full_levels import (
    interpolate_contravatiant_vertical_verlocity_to_full_levels as interpolate_contravatiant_vertical_verlocity_to_full_levels_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import interpolate_to_cell_center as interpolate_to_cell_center_orig
from icon4py.model.atmosphere.dycore.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_to_ie_and_compute_ekin_on_edges as interpolate_vn_to_ie_and_compute_ekin_on_edges_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_vt_to_ie import interpolate_vt_to_ie as interpolate_vt_to_ie_orig
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl as mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig,
)
from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl as mo_math_divrot_rot_vertex_ri_dsl_orig,
)
from icon4py.model.atmosphere.dycore.extrapolate_at_top import (
    extrapolate_at_top as extrapolate_at_top_orig,
)
from icon4py.model.atmosphere.dycore.velocity.velocity_advection_program import (
    fused_stencils_4_5 as fused_stencils_4_5_orig,
    fused_stencils_9_10 as fused_stencils_9_10_orig,
    fused_stencils_11_to_13 as fused_stencils_11_to_13_orig,
    fused_stencil_14 as fused_stencil_14_orig,
    fused_stencils_16_to_17 as fused_stencils_16_to_17_orig,
)



@dataclasses.dataclass
class CachedProgram:
    program: gtx.ffront.decorator.Program
    _compiled_program: Optional[Callable] = None
    _compiled_args: tuple = dataclasses.field(default_factory=tuple)

    @property
    def compiled_program(self) -> Callable:
        return self._compiled_program

    def compile_the_program(
        self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any
    ) -> Callable:
        backend = self.program.backend
        transformer = backend.transformer.replace(
            args=args, kwargs=kwargs | {"offset_provider": offset_provider}
        )
        program_call = transformer(self.program.definition_stage)
        self._compiled_args = program_call.args
        return backend.executor.otf_workflow(program_call)

    def __call__(self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any) -> None:
        if not self.compiled_program:
            self._compiled_program = self.compile_the_program(
                *args, offset_provider=offset_provider, **kwargs
            )

        size_args = self._compiled_args[len(args) :]
        return self.compiled_program(*args, *size_args, offset_provider=offset_provider)


add_extra_diffusion_for_w_con_approaching_cfl = CachedProgram(add_extra_diffusion_for_w_con_approaching_cfl_orig)
add_extra_diffusion_for_wn_approaching_cfl = CachedProgram(add_extra_diffusion_for_wn_approaching_cfl_orig)
compute_tangential_wind = CachedProgram(compute_tangential_wind_orig)
compute_advective_normal_wind_tendency = CachedProgram(compute_advective_normal_wind_tendency_orig)
compute_horizontal_advection_term_for_vertical_velocity = CachedProgram(compute_horizontal_advection_term_for_vertical_velocity_orig)
interpolate_to_cell_center = CachedProgram(interpolate_to_cell_center_orig)
interpolate_vt_to_ie = CachedProgram(interpolate_vt_to_ie_orig)
interpolate_contravatiant_vertical_verlocity_to_full_levels = CachedProgram(interpolate_contravatiant_vertical_verlocity_to_full_levels_orig)
interpolate_vn_to_ie_and_compute_ekin_on_edges = CachedProgram(interpolate_vn_to_ie_and_compute_ekin_on_edges_orig)
mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = CachedProgram(mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig)
mo_math_divrot_rot_vertex_ri_dsl = CachedProgram(mo_math_divrot_rot_vertex_ri_dsl_orig)
extrapolate_at_top = CachedProgram(extrapolate_at_top_orig)
fused_stencils_4_to_5 = CachedProgram(fused_stencils_4_5_orig)
fused_stencils_9_to_10 = CachedProgram(fused_stencils_9_10_orig)
fused_stencils_11_to_13 = CachedProgram(fused_stencils_11_to_13_orig)
fused_stencils_14 = CachedProgram(fused_stencil_14_orig)
fused_stencils_16_to_17 = CachedProgram(fused_stencils_16_to_17_orig)
