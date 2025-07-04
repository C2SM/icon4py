# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import math

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

import icon4py.model.common.math.helpers as math_helpers
import icon4py.model.common.metrics.compute_weight_factors as weight_factors
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.interpolation.stencils import cell_2_edge_interpolation
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.metrics import (
    compute_coeff_gradekin,
    compute_diffusion_metrics,
    compute_zdiff_gradp_dsl,
    metric_fields as mf,
    metrics_attributes as attrs,
    reference_atmosphere,
)
from icon4py.model.common.states import factory, model
from icon4py.model.common.utils import data_allocation as data_alloc


cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)
vertical_domain = v_grid.domain(dims.KDim)
vertical_half_domain = v_grid.domain(dims.KHalfDim)
log = logging.getLogger(__name__)


class MetricsFieldsFactory(factory.FieldSource, factory.GridProvider):
    def __init__(
        self,
        grid: icon.IconGrid,
        vertical_grid: v_grid.VerticalGrid,
        decomposition_info: definitions.DecompositionInfo,
        geometry_source: geometry.GridGeometry,
        topography: gtx.Field,
        interpolation_source: interpolation_factory.InterpolationFieldsFactory,
        backend: gtx_backend.Backend,
        metadata: dict[str, model.FieldMetaData],
        e_refin_ctrl: gtx.Field,
        c_refin_ctrl: gtx.Field,
        rayleigh_type: int,
        rayleigh_coeff: float,
        exner_expol: float,
        vwind_offctr: float,
    ):
        self._backend = backend
        self._xp = data_alloc.import_array_ns(backend)
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._providers: dict[str, factory.FieldProvider] = {}
        self._geometry = geometry_source
        self._interpolation_source = interpolation_source
        log.info(
            f"initialized metrics factory for backend = '{self._backend_name()}' and grid = '{self._grid}'"
        )
        log.debug(f"using array_ns {self._xp} ")
        vct_a = self._vertical_grid.vct_a
        vct_a_1 = vct_a.asnumpy()[0]
        self._config = {
            "divdamp_trans_start": 12500.0,
            "divdamp_trans_end": 17500.0,
            "divdamp_type": 3,
            "damping_height": vertical_grid.config.rayleigh_damping_height,
            "rayleigh_type": rayleigh_type,
            "rayleigh_coeff": rayleigh_coeff,
            "exner_expol": exner_expol,
            "vwind_offctr": vwind_offctr,
            "igradp_method": 3,
            "igradp_constant": 3,
            "thslp_zdiffu": 0.02,
            "thhgtd_zdiffu": 125.0,
            "vct_a_1": vct_a_1,
        }

        k_index = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        e_lev = data_alloc.index_field(self._grid, dims.EdgeDim, backend=self._backend)
        e_owner_mask = gtx.as_field(
            (dims.EdgeDim,), self._decomposition_info.owner_mask(dims.EdgeDim)
        )
        c_owner_mask = gtx.as_field(
            (dims.CellDim,), self._decomposition_info.owner_mask(dims.CellDim)
        )

        self.register_provider(
            factory.PrecomputedFieldProvider(
                {
                    "topography": topography,
                    "vct_a": vct_a,
                    "c_refin_ctrl": c_refin_ctrl,
                    "e_refin_ctrl": e_refin_ctrl,
                    "e_owner_mask": e_owner_mask,
                    "c_owner_mask": c_owner_mask,
                    "k_lev": k_index,
                    "e_lev": e_lev,
                }
            )
        )
        self._register_computed_fields()

    def __repr__(self):
        return f"{self.__class__.__name__} on (grid={self._grid!r}) providing fields f{self.metadata.keys()}"

    @property
    def _sources(self) -> factory.FieldSource:
        return factory.CompositeSource(self, (self._geometry, self._interpolation_source))

    def _register_computed_fields(self):
        vertical_coordinates_on_half_levels = factory.NumpyFieldsProvider(
            func=functools.partial(
                v_grid.compute_vertical_coordinate,
                array_ns=self._xp,
            ),
            fields=(attrs.CELL_HEIGHT_ON_HALF_LEVEL,),
            domain={
                dims.CellDim: (0, cell_domain(h_grid.Zone.END)),
                dims.KDim: (vertical_domain(v_grid.Zone.TOP), vertical_domain(v_grid.Zone.BOTTOM)),
            },
            deps={
                "vct_a": "vct_a",
                "topography": "topography",
                "cell_areas": geometry_attrs.CELL_AREA,
                "geofac_n2s": interpolation_attributes.GEOFAC_N2S,
            },
            connectivities={"c2e2co": dims.C2E2CODim},
            params={
                "nflatlev": self._vertical_grid.nflatlev,
                "model_top_height": self._vertical_grid.config.model_top_height,
                "SLEVE_decay_scale_1": self.vertical_grid.config.SLEVE_decay_scale_1,
                "SLEVE_decay_exponent": self._vertical_grid.config.SLEVE_decay_exponent,
                "SLEVE_decay_scale_2": self._vertical_grid.config.SLEVE_decay_scale_2,
                "SLEVE_minimum_layer_thickness_1": self._vertical_grid.config.SLEVE_minimum_layer_thickness_1,
                "SLEVE_minimum_relative_layer_thickness_1": self._vertical_grid.config.SLEVE_minimum_relative_layer_thickness_1,
                "SLEVE_minimum_layer_thickness_2": self._vertical_grid.config.SLEVE_minimum_layer_thickness_2,
                "SLEVE_minimum_relative_layer_thickness_2": self._vertical_grid.config.SLEVE_minimum_relative_layer_thickness_2,
                "lowest_layer_thickness": self._vertical_grid.config.lowest_layer_thickness,
            },
        )
        self.register_provider(vertical_coordinates_on_half_levels)

        height = factory.ProgramFieldProvider(
            func=math_helpers.average_two_vertical_levels_downwards_on_cells.with_backend(
                self._backend
            ),
            domain={
                dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.END)),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"average": attrs.Z_MC},
            deps={"input_field": attrs.CELL_HEIGHT_ON_HALF_LEVEL},
        )
        self.register_provider(height)

        compute_ddqz_z_half = factory.ProgramFieldProvider(
            func=mf.compute_ddqz_z_half.with_backend(self._backend),
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"ddqz_z_half": attrs.DDQZ_Z_HALF},
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "z_mc": attrs.Z_MC,
            },
            params={"nlev": self._grid.num_levels},
        )
        self.register_provider(compute_ddqz_z_half)

        ddqz_z_full_and_inverse = factory.ProgramFieldProvider(
            func=mf.compute_ddqz_z_full_and_inverse.with_backend(self._backend),
            deps={"z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL},
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"ddqz_z_full": attrs.DDQZ_Z_FULL, "inv_ddqz_z_full": attrs.INV_DDQZ_Z_FULL},
        )
        self.register_provider(ddqz_z_full_and_inverse)

        ddqz_full_on_edges = factory.ProgramFieldProvider(
            func=cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(self._backend),
            deps={"in_field": attrs.DDQZ_Z_FULL, "coeff": interpolation_attributes.C_LIN_E},
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"out_field": attrs.DDQZ_Z_FULL_E},
        )
        self.register_provider(ddqz_full_on_edges)

        compute_scaling_factor_for_3d_divdamp = factory.ProgramFieldProvider(
            func=mf.compute_scaling_factor_for_3d_divdamp.with_backend(self._backend),
            domain={
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                )
            },
            fields={"scaling_factor_for_3d_divdamp": attrs.SCALING_FACTOR_FOR_3D_DIVDAMP},
            deps={"vct_a": "vct_a"},
            params={
                "divdamp_trans_start": self._config["divdamp_trans_start"],
                "divdamp_trans_end": self._config["divdamp_trans_end"],
                "divdamp_type": self._config["divdamp_type"],
            },
        )
        self.register_provider(compute_scaling_factor_for_3d_divdamp)

        compute_rayleigh_w = factory.ProgramFieldProvider(
            func=mf.compute_rayleigh_w.with_backend(self._backend),
            deps={"vct_a": "vct_a"},
            domain={
                dims.KHalfDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    v_grid.Domain(dims.KHalfDim, v_grid.Zone.DAMPING, 1),
                )
            },
            fields={"rayleigh_w": attrs.RAYLEIGH_W},
            params={
                "damping_height": self._config["damping_height"],
                "rayleigh_type": self._config["rayleigh_type"],
                "rayleigh_coeff": self._config["rayleigh_coeff"],
                "vct_a_1": self._config["vct_a_1"],
                "pi_const": math.pi,
            },
        )
        self.register_provider(compute_rayleigh_w)

        compute_coeff_dwdz = factory.ProgramFieldProvider(
            func=mf.compute_coeff_dwdz.with_backend(self._backend),
            deps={
                "ddqz_z_full": attrs.DDQZ_Z_FULL,
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    v_grid.Domain(dims.KHalfDim, v_grid.Zone.TOP, 1),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"coeff1_dwdz": attrs.COEFF1_DWDZ, "coeff2_dwdz": attrs.COEFF2_DWDZ},
        )
        self.register_provider(compute_coeff_dwdz)

        compute_theta_exner_ref_mc = factory.ProgramFieldProvider(
            func=mf.compute_theta_exner_ref_mc.with_backend(self._backend),
            deps={
                "z_mc": attrs.Z_MC,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"exner_ref_mc": attrs.EXNER_REF_MC, "theta_ref_mc": attrs.THETA_REF_MC},
            params={
                "t0sl_bg": constants.SEA_LEVEL_TEMPERATURE,
                "del_t_bg": constants.DELTA_TEMPERATURE,
                "h_scal_bg": constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
                "grav": constants.GRAV,
                "rd": constants.RD,
                "p0sl_bg": constants.SEA_LEVEL_PRESSURE,
                "rd_o_cpd": constants.RD_O_CPD,
                "p0ref": constants.REFERENCE_PRESSURE,
            },
        )
        self.register_provider(compute_theta_exner_ref_mc)

        compute_d2dexdz2_fac_mc = factory.ProgramFieldProvider(
            func=reference_atmosphere.compute_d2dexdz2_fac_mc.with_backend(self._backend),
            deps={
                "theta_ref_mc": attrs.THETA_REF_MC,
                "inv_ddqz_z_full": attrs.INV_DDQZ_Z_FULL,
                "exner_ref_mc": attrs.EXNER_REF_MC,
                "z_mc": attrs.Z_MC,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={
                attrs.D2DEXDZ2_FAC1_MC: attrs.D2DEXDZ2_FAC1_MC,
                attrs.D2DEXDZ2_FAC2_MC: attrs.D2DEXDZ2_FAC2_MC,
            },
            params={
                "cpd": constants.CPD,
                "grav": constants.GRAV,
                "del_t_bg": constants.DEL_T_BG,
                "h_scal_bg": constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
            },
        )
        self.register_provider(compute_d2dexdz2_fac_mc)

        compute_cell_to_vertex_interpolation = factory.ProgramFieldProvider(
            func=compute_cell_2_vertex_interpolation.with_backend(self._backend),
            deps={
                "cell_in": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "c_int": interpolation_attributes.CELL_AW_VERTS,
            },
            domain={
                dims.VertexDim: (
                    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    vertex_domain(h_grid.Zone.INTERIOR),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.VERT_OUT: attrs.VERT_OUT},
        )
        self.register_provider(compute_cell_to_vertex_interpolation)

        compute_ddxt_z_half_e = factory.ProgramFieldProvider(
            func=mf.compute_ddxt_z_half_e.with_backend(self._backend),
            deps={
                "cell_in": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "c_int": interpolation_attributes.CELL_AW_VERTS,
                "inv_primal_edge_length": f"inverse_of_{geometry_attrs.EDGE_LENGTH}",
                "tangent_orientation": geometry_attrs.TANGENT_ORIENTATION,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3),
                    edge_domain(h_grid.Zone.INTERIOR),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.DDXT_Z_HALF_E: attrs.DDXT_Z_HALF_E},
        )
        self.register_provider(compute_ddxt_z_half_e)

        compute_ddxn_z_half_e = factory.ProgramFieldProvider(
            func=mf.compute_ddxn_z_half_e.with_backend(self._backend),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.INTERIOR),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.DDXN_Z_HALF_E: attrs.DDXN_Z_HALF_E},
        )
        self.register_provider(compute_ddxn_z_half_e)

        compute_ddxn_z_full = factory.ProgramFieldProvider(
            func=math_helpers.average_two_vertical_levels_downwards_on_edges.with_backend(
                self._backend
            ),
            deps={
                "input_field": attrs.DDXN_Z_HALF_E,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"average": attrs.DDXN_Z_FULL},
        )
        self.register_provider(compute_ddxn_z_full)
        compute_ddxt_z_full = factory.ProgramFieldProvider(
            func=math_helpers.average_two_vertical_levels_downwards_on_edges.with_backend(
                self._backend
            ),
            deps={
                "input_field": attrs.DDXT_Z_HALF_E,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"average": attrs.DDXT_Z_FULL},
        )
        self.register_provider(compute_ddxt_z_full)

        compute_exner_w_implicit_weight_parameter_np = factory.NumpyFieldsProvider(
            func=functools.partial(mf.compute_exner_w_implicit_weight_parameter, array_ns=self._xp),
            domain=(dims.CellDim,),
            connectivities={"c2e": dims.C2EDim},
            fields=(attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER,),
            deps={
                "vct_a": "vct_a",
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "z_ddxn_z_half_e": attrs.DDXN_Z_HALF_E,
                "z_ddxt_z_half_e": attrs.DDXT_Z_HALF_E,
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
            },
            params={
                "vwind_offctr": self._config["vwind_offctr"],
                "nlev": self._grid.num_levels,
                "horizontal_start_cell": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(compute_exner_w_implicit_weight_parameter_np)

        compute_exner_w_explicit_weight_parameter = factory.ProgramFieldProvider(
            func=mf.compute_exner_w_explicit_weight_parameter.with_backend(self._backend),
            deps={
                "exner_w_implicit_weight_parameter": attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
            },
            fields={"exner_w_explicit_weight_parameter": attrs.EXNER_W_EXPLICIT_WEIGHT_PARAMETER},
        )
        self.register_provider(compute_exner_w_explicit_weight_parameter)

        compute_exner_exfac = factory.ProgramFieldProvider(
            func=mf.compute_exner_exfac.with_backend(self._backend),
            deps={
                "ddxn_z_full": attrs.DDXN_Z_FULL,
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.EXNER_EXFAC: attrs.EXNER_EXFAC},
            params={
                "exner_expol": self._config["exner_expol"],
                "lateral_boundary_level_2": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(compute_exner_exfac)

        wgtfac_c_provider = factory.ProgramFieldProvider(
            func=weight_factors.compute_wgtfac_c.with_backend(self._backend),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
            },
            domain={
                dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.END)),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.WGTFAC_C: attrs.WGTFAC_C},
            params={"nlev": self._grid.num_levels},
        )
        self.register_provider(wgtfac_c_provider)

        compute_wgtfac_e = factory.ProgramFieldProvider(
            func=mf.compute_wgtfac_e.with_backend(self._backend),
            deps={
                attrs.WGTFAC_C: attrs.WGTFAC_C,
                "c_lin_e": interpolation_attributes.C_LIN_E,
            },
            domain={
                dims.CellDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.LOCAL),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.WGTFAC_E: attrs.WGTFAC_E},
        )
        self.register_provider(compute_wgtfac_e)
        compute_flat_edge_idx = factory.ProgramFieldProvider(
            func=mf.compute_flat_idx.with_backend(self._backend),
            deps={
                "z_mc": attrs.Z_MC,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "k_lev": "k_lev",
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.LOCAL),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"flat_idx": attrs.FLAT_EDGE_INDEX},
        )
        self.register_provider(compute_flat_edge_idx)
        max_flat_index_provider = factory.NumpyFieldsProvider(
            func=functools.partial(mf.compute_max_index, array_ns=self._xp),
            domain=(dims.EdgeDim,),
            fields=(attrs.FLAT_IDX_MAX,),
            deps={
                "flat_idx": attrs.FLAT_EDGE_INDEX,
            },
        )
        self.register_provider(max_flat_index_provider)

        pressure_gradient_fields = factory.ProgramFieldProvider(
            func=mf.compute_pressure_gradient_downward_extrapolation_mask_distance.with_backend(
                self._backend
            ),
            deps={
                "z_mc": attrs.Z_MC,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "topography": "topography",
                "e_owner_mask": "e_owner_mask",
                "flat_idx_max": attrs.FLAT_IDX_MAX,
                "e_lev": "e_lev",
                "k_lev": "k_lev",
            },
            params={
                "horizontal_start_distance": self._grid.end_index(edge_domain(h_grid.Zone.NUDGING)),
                "horizontal_end_distance": self._grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.NUDGING_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"pg_edgeidx_dsl": attrs.PG_EDGEIDX_DSL, "pg_exdist_dsl": attrs.PG_EDGEDIST_DSL},
        )
        self.register_provider(pressure_gradient_fields)

        compute_mask_bdy_halo_c = factory.ProgramFieldProvider(
            func=mf.compute_mask_bdy_halo_c.with_backend(self._backend),
            deps={
                "c_refin_ctrl": "c_refin_ctrl",
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.HALO),
                    cell_domain(h_grid.Zone.HALO),
                ),
            },
            fields={
                attrs.MASK_PROG_HALO_C: attrs.MASK_PROG_HALO_C,
                attrs.BDY_HALO_C: attrs.BDY_HALO_C,
            },
        )
        self.register_provider(compute_mask_bdy_halo_c)

        compute_horizontal_mask_for_3d_divdamp = factory.ProgramFieldProvider(
            func=mf.compute_horizontal_mask_for_3d_divdamp.with_backend(self._backend),
            deps={
                "e_refin_ctrl": "e_refin_ctrl",
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP: attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP},
            params={
                "grf_nudge_start_e": gtx.int32(h_grid._GRF_NUDGEZONE_START_EDGES),
                "grf_nudgezone_width": gtx.int32(h_grid._GRF_NUDGEZONE_WIDTH),
            },
        )
        self.register_provider(compute_horizontal_mask_for_3d_divdamp)

        compute_zdiff_gradp_dsl_np = factory.NumpyFieldsProvider(
            func=functools.partial(
                compute_zdiff_gradp_dsl.compute_zdiff_gradp_dsl, array_ns=self._xp
            ),
            deps={
                "z_mc": attrs.Z_MC,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "flat_idx": attrs.FLAT_IDX_MAX,
                "topography": "topography",
            },
            connectivities={"e2c": dims.E2CDim},
            domain=(dims.EdgeDim, dims.KDim),
            fields=(attrs.ZDIFF_GRADP,),
            params={
                "nlev": self._grid.num_levels,
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_start_1": self._grid.start_index(
                    edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
                ),
            },
        )
        self.register_provider(compute_zdiff_gradp_dsl_np)

        coeff_gradekin = factory.NumpyFieldsProvider(
            func=functools.partial(
                compute_coeff_gradekin.compute_coeff_gradekin, array_ns=self._xp
            ),
            domain=(dims.ECDim,),
            fields=(attrs.COEFF_GRADEKIN,),
            deps={
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
            },
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_end": self._grid.num_edges,
            },
        )
        self.register_provider(coeff_gradekin)

        compute_wgtfacq_c = factory.NumpyFieldsProvider(
            func=functools.partial(weight_factors.compute_wgtfacq_c_dsl, array_ns=self._xp),
            domain=(dims.CellDim, dims.KDim),
            fields=(attrs.WGTFACQ_C,),
            deps={"z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL},
            params={"nlev": self._grid.num_levels},
        )

        self.register_provider(compute_wgtfacq_c)

        compute_wgtfacq_e = factory.NumpyFieldsProvider(
            func=functools.partial(weight_factors.compute_wgtfacq_e_dsl, array_ns=self._xp),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "wgtfacq_c_dsl": attrs.WGTFACQ_C,
            },
            connectivities={"e2c": dims.E2CDim},
            domain=(dims.EdgeDim, dims.KDim),
            fields=(attrs.WGTFACQ_E,),
            params={"n_edges": self._grid.num_edges, "nlev": self._grid.num_levels},
        )

        self.register_provider(compute_wgtfacq_e)

        compute_maxslp_maxhgtd = factory.ProgramFieldProvider(
            func=mf.compute_maxslp_maxhgtd.with_backend(self._backend),
            deps={
                "ddxn_z_full": attrs.DDXN_Z_FULL,
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.MAXSLP: attrs.MAXSLP, attrs.MAXHGTD: attrs.MAXHGTD},
        )
        self.register_provider(compute_maxslp_maxhgtd)

        compute_weighted_cell_neighbor_sum = factory.ProgramFieldProvider(
            func=mf.compute_weighted_cell_neighbor_sum,
            deps={
                "maxslp": attrs.MAXSLP,
                "maxhgtd": attrs.MAXHGTD,
                "c_bln_avg": interpolation_attributes.C_BLN_AVG,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.MAXSLP_AVG: attrs.MAXSLP_AVG, attrs.MAXHGTD_AVG: attrs.MAXHGTD_AVG},
        )
        self.register_provider(compute_weighted_cell_neighbor_sum)

        compute_max_nbhgt = factory.NumpyFieldsProvider(
            func=functools.partial(
                compute_diffusion_metrics.compute_max_nbhgt_array_ns, array_ns=self._xp
            ),
            deps={
                "z_mc": attrs.Z_MC,
            },
            connectivities={"c2e2c": dims.C2E2CDim},
            domain=(dims.CellDim,),
            fields=(attrs.MAX_NBHGT,),
            params={
                "nlev": self._grid.num_levels,
            },
        )
        self.register_provider(compute_max_nbhgt)

        compute_diffusion_metrics_np = factory.NumpyFieldsProvider(
            func=functools.partial(
                compute_diffusion_metrics.compute_diffusion_metrics, array_ns=self._xp
            ),
            deps={
                "z_mc": attrs.Z_MC,
                "max_nbhgt": attrs.MAX_NBHGT,
                "c_owner_mask": "c_owner_mask",
                "maxslp_avg": attrs.MAXSLP_AVG,
                "maxhgtd_avg": attrs.MAXHGTD_AVG,
            },
            connectivities={"c2e2c": dims.C2E2CDim},
            domain=(dims.CellDim, dims.KDim),
            fields=(
                attrs.MASK_HDIFF,
                attrs.ZD_DIFFCOEF_DSL,
                attrs.ZD_INTCOEF_DSL,
                attrs.ZD_VERTOFFSET_DSL,
            ),
            params={
                "thslp_zdiffu": self._config["thslp_zdiffu"],
                "thhgtd_zdiffu": self._config["thhgtd_zdiffu"],
                "cell_nudging": self._grid.start_index(
                    h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING)
                ),
                "nlev": self._grid.num_levels,
            },
        )

        self.register_provider(compute_diffusion_metrics_np)

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_backend.Backend:
        return self._backend

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return self._vertical_grid
