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
import collections
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from gt4py import eve
from gt4py.eve.codegen import (
    JinjaTemplate as as_jinja,
    Node,
    TemplatedGenerator,
    format_source,
)

from icon4pytools.icon4pygen.bindings.codegen.header import (
    CppFreeFunc,
    CppRunAndVerifyFuncDeclaration,
    CppRunFuncDeclaration,
    CppSetupFuncDeclaration,
    CppVerifyFuncDeclaration,
    run_func_declaration,
    run_verify_func_declaration,
)
from icon4pytools.icon4pygen.bindings.codegen.render.offset import (
    GpuTriMeshOffsetRenderer,
)
from icon4pytools.icon4pygen.bindings.entities import Field, Offset
from icon4pytools.icon4pygen.bindings.exceptions import BindingsRenderingException
from icon4pytools.icon4pygen.bindings.utils import write_string


class CppDefGenerator(TemplatedGenerator):
    CppDefTemplate = as_jinja(
        """\
        {{ includes }}
        {{ stencil_class }}
        extern "C" {
        {{ run_func }}
        {{ verify_func }}
        {{ run_verify_func }}
        {{ setup_func }}
        {{ free_func }}
        }
        """
    )

    IncludeStatements = as_jinja(
        """\
        #include <chrono>
        #include <dace/dace.h>

        #include "cuda_utils.hpp"
        #include "cuda_verify.hpp"
        #include "to_json.hpp"
        #include "to_vtk.h"
        #include "unstructured_interface.hpp"
        #include "verification_metrics.hpp"

        struct {{funcname}}_state_t {
            dace::cuda::Context *gpu_context;
        };
        """
    )

    PublicUtilities = as_jinja(
        """
      static const GpuTriMesh & getMesh() {
        return mesh_;
      }

      static cudaStream_t getStream() {
        return stream_;
      }

      static json *getJsonRecord() {
        return jsonRecord_;
      }

      static MeshInfoVtk *getMeshInfoVtk() {
        return mesh_info_vtk_;
      }

      static verify *getVerify() {
        return verify_;
      }

      static void free() {
      }
      """
    )

    CopyPointers = as_jinja(
        """
      void copy_pointers(
        {%- for field in _this_node.fields %}
          {{field.renderer.render_ctype('c++')}}{{field.renderer.render_pointer()}} {{field.name}} {%- if not loop.last -%}, {%- endif -%}
        {% endfor %}
      ) {
        {% for field in _this_node.fields -%}
          {{field.name}}_ = {{field.name}};
        {% endfor %}
      }
      """
    )

    StencilClass = as_jinja(
        """
        namespace cuda_ico {

        class {{ funcname }} {
        {{ gpu_tri_mesh }}
        {{ private_members }}
        public:
        {{ public_utilities }}
        {{ setup_func }}
        {{ funcname }}() {}
        {{ run_fun }}
        {{ copy_pointers }}
        };
        } // namespace cuda_ico
        """
    )

    GpuTriMesh = as_jinja(
        """\
        public:
          struct GpuTriMesh {
            int Nproma;
            int NumVertices;
            int NumEdges;
            int NumCells;
            int VertexStride;
            int EdgeStride;
            int CellStride;
            {%- if has_offsets -%}
            {%- for var in _this_node.table_vars -%}
            int * {{ var }}Table;
            {%- endfor -%}
            {%- endif %}

            GpuTriMesh() {}

            GpuTriMesh(const GlobalGpuTriMesh *mesh) {
              Nproma = mesh->Nproma;
              NumVertices = mesh->NumVertices;
              NumCells = mesh->NumCells;
              NumEdges = mesh->NumEdges;
              VertexStride = mesh->VertexStride;
              CellStride = mesh->CellStride;
              EdgeStride = mesh->EdgeStride;
              {%- if has_offsets -%}
              {%- for table in _this_node.neighbor_tables -%}
              {{ table }}
              {%- endfor -%}
              {%- endif %}
            }
          };
        """
    )

    StencilClassSetupFunc = as_jinja(
        """\
        static void setup(
        const GlobalGpuTriMesh *mesh, cudaStream_t stream, json *jsonRecord, MeshInfoVtk *mesh_info_vtk, verify *verify) {
        mesh_ = GpuTriMesh(mesh);
        is_setup_ = true;
        stream_ = stream;
        jsonRecord_ = jsonRecord;
        mesh_info_vtk_ = mesh_info_vtk;
        verify_ = verify;
        }
        """
    )

    PrivateMembers = as_jinja(
        """\
        private:
        {%- for field in _this_node.fields -%}
        {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_;
        {%- endfor -%}
        inline static GpuTriMesh mesh_;
        inline static bool is_setup_;
        inline static cudaStream_t stream_;
        inline static json* jsonRecord_;
        inline static MeshInfoVtk* mesh_info_vtk_;
        inline static verify* verify_;
        """
    )

    StenClassRunFun = as_jinja(
        """
      void run({%- for arg in _this_node.domain_args -%}
        const int {{arg.cpp_arg_name()}}{%- if not loop.last -%}, {%- endif -%}
      {%- endfor -%}) {
      if (!is_setup_) {
        printf("{{ stencil_name }} has not been set up! make sure setup() is called before run!\\n");
        return;
      }

      dace::cuda::Context __dace_context(1, 1);
      __dace_context.streams[0] = stream_;
      DACE_GPU_CHECK(cudaEventCreateWithFlags(&__dace_context.events[0], cudaEventDisableTiming));

      {{ stencil_name }}_state_t handle{.gpu_context = &__dace_context};

      __dace_runkernel_tasklet_toplevel_map_0_0_1(&handle
      {%- for data_descr in _this_node.sorted_data_descriptors -%}
        , {{ data_descr.cpp_arg_name() }}
      {%- endfor -%},
      {{ ", ".join(_this_node.sorted_symbols) }});
      #ifndef NDEBUG
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
      #endif

        DACE_GPU_CHECK(cudaEventDestroy(__dace_context.events[0]));
      }
      """
    )

    CppRunFuncDeclaration = run_func_declaration

    RunFunc = as_jinja(
        """\
        {{run_func_declaration }} {
        cuda_ico::{{ funcname }} s;
        s.copy_pointers({{ params }});
        s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
        return;
        }
        """
    )

    Params = as_jinja(
        """\
        {%- for field in _this_node.fields -%}
        {{ field.name }}
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%}
        """
    )

    CppVerifyFuncDeclaration = as_jinja(
        """\
        bool verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_{{ before_suffix }},
        const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_{{ k_size_suffix }},
        {%- endfor -%}
        {%- for field in _this_node.tol_fields -%}
        const double {{ field.name }}_rel_tol,
        const double {{ field.name }}_abs_tol,
        {%- endfor -%}
        const int iteration)
        """
    )

    VerifyFunc = as_jinja(
        """\
        {{ verify_func_declaration }} {
        using namespace std::chrono;
        const auto &mesh = cuda_ico::{{ funcname }}::getMesh();
        cudaStream_t stream = cuda_ico::{{ funcname }}::getStream();
        MeshInfoVtk* mesh_info_vtk = cuda_ico::{{ funcname }}::getMeshInfoVtk();
        verify* verify = cuda_ico::{{ funcname }}::getVerify();
        high_resolution_clock::time_point t_start = high_resolution_clock::now();
        struct VerificationMetrics stencilMetrics;
        {{ metrics_serialisation }}
        }
        """
    )

    MetricsSerialisation = as_jinja(
        """\
        {%- for field in _this_node.out_fields %}
        int {{ field.name }}_kSize = {{ field.name }}_k_size;
        {% if field.is_integral() %}
        stencilMetrics = ::verify_field(
            stream, (mesh.{{ field.renderer.render_stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", iteration);
        {% elif field.is_bool() %}
        stencilMetrics = ::verify_bool_field(
            stream, verify, (mesh.{{ field.renderer.render_stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", iteration);
        {% else %}
        stencilMetrics = ::verify_field(
            stream, (mesh.{{ field.renderer.render_stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", {{ field.name }}_rel_tol, {{ field.name }}_abs_tol, iteration);
        {% endif %}
        #ifdef __SERIALIZE_METRICS
        MetricsSerialiser serialiser_{{ field.name }}(
            cuda_ico::{{ funcname }}::getJsonRecord(), stencilMetrics,
            \"{{ funcname }}\", \"{{ field.name }}\");
        serialiser_{{ field.name }}.writeJson(iteration);
        #endif
        if (!stencilMetrics.isValid) {
        #ifdef __SERIALIZE_ON_ERROR
        {{ field.renderer.render_serialise_func() }}(mesh_info_vtk, 0, (mesh.Num{{ field.location.render_location_type() }} - 1), {{ field.name }}_kSize,
                              (mesh.{{ field.renderer.render_stride_type() }}), {{ field.name }},
                              \"{{ funcname }}\", \"{{ field.name }}\", iteration);
        {{ field.renderer.render_serialise_func() }}(mesh_info_vtk, 0, (mesh.Num{{ field.location.render_location_type() }} - 1), {{ field.name }}_kSize,
                              (mesh.{{ field.renderer.render_stride_type() }}), {{ field.name }}_dsl,
                              \"{{ funcname }}\", \"{{ field.name }}_dsl\",
                              iteration);
        std::cout << "[DSL] serializing {{ field.name }} as error is high.\\n" << std::flush;
        #endif
        }
        {%- if loop.last -%}
        ;
        {%- endif -%}
        {%- endfor %}
        #ifdef __SERIALIZE_ON_ERROR
        serialize_flush_iter(\"{{ funcname }}\", iteration);
        #endif
        high_resolution_clock::time_point t_end = high_resolution_clock::now();
        duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
        std::cout << "[DSL] Verification took " << timing.count() << " seconds.\\n" << std::flush;
        return stencilMetrics.isValid;
        """
    )

    CppRunAndVerifyFuncDeclaration = run_verify_func_declaration

    RunFuncCall = as_jinja(
        """\
        run_{{funcname}}(
        {%- for field in _this_node.fields -%}
        {%- if field.intent.out -%}
        {{ field.name }}_before,
        {%- else -%}
        {{ field.name }},
        {%- endif -%}
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_{{ k_size_suffix }},
        {%- endfor -%}
        verticalStart, verticalEnd, horizontalStart, horizontalEnd) ;
        """
    )

    VerifyFuncCall = as_jinja(
        """\
        verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_{{ before_suffix }},
        {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_{{ k_size_suffix }},
        {%- endfor -%}
        {%- for field in _this_node.tol_fields -%}
        {{ field.name }}_rel_tol,
        {{ field.name }}_abs_tol,
        {%- endfor -%}
        iteration) ;
        """
    )

    RunAndVerifyFunc = as_jinja(
        """\
        {{ run_verify_func_declaration }} {
        static int iteration = 0;
        std::cout << \"[DSL] Running stencil {{ funcname }} (\"
                  << iteration << \") ...\\n\"
                  << std::flush;
        {{ run_func_call }}
        std::cout << \"[DSL] {{ funcname }} run time: \" << time
            << \"s\\n\"
            << std::flush;
        std::cout << \"[DSL] Verifying stencil {{ funcname }}...\\n\"
                  << std::flush;
        {{ verify_func_call }}
        iteration++;
        }
        """
    )

    CppSetupFuncDeclaration = as_jinja(
        """\
        void setup_{{funcname}}(
        GlobalGpuTriMesh *mesh, cudaStream_t stream, json *json_record, MeshInfoVtk *mesh_info_vtk, verify *verify)
        """
    )

    SetupFunc = as_jinja(
        """\
        {{ func_declaration }} {
        cuda_ico::{{ funcname }}::setup(mesh, stream, json_record, mesh_info_vtk, verify);
        }
        """
    )

    FreeFunc = as_jinja(
        """\
        void free_{{funcname}}() {
            cuda_ico::{{ funcname }}::free();
        }
        """
    )


class Scalar:
    name: str
    var_name: str

    def __init__(self, name: str, var_name: str):
        self.name = name
        self.var_name = var_name

    def sdfg_arg_name(self) -> str:
        return self.name

    def cpp_arg_name(self) -> str:
        return self.var_name


class DataDescriptor:
    data_descr: Union[Field, Offset]

    def __init__(self, data_descr: Field | Offset):
        self.data_descr = data_descr

    def sdfg_arg_name(self) -> str:
        if isinstance(self.data_descr, Field):
            return self.data_descr.name
        # for offset data arrays
        return f"__connectivity_{self.data_descr.renderer.render_uppercase_shorthand()}"

    def cpp_arg_name(self) -> str:
        if isinstance(self.data_descr, Field):
            return f"{self.data_descr.name}_"
        # for offset data arrays
        return f"mesh_.{self.data_descr.renderer.render_lowercase_shorthand()}Table"

    def strides(self) -> Optional[str]:
        if isinstance(self.data_descr, Field):
            try:
                strides_str = self.data_descr.renderer.render_strides(exclude_sparse_dim=False)
            except BindingsRenderingException:
                strides_str = "1"

            if strides_str == "1":
                return None
            return strides_str

        # for offset data arrays
        return "1, mesh_.Nproma"


class CppFunc(Node):
    funcname: str


class IncludeStatements(Node):
    funcname: str


class GpuTriMesh(Node):
    table_vars: list[str]
    neighbor_tables: list[str]
    has_offsets: bool


class StenClassRunFun(Node):
    stencil_name: str
    domain_args: Sequence[Scalar]
    sorted_data_descriptors: Sequence[DataDescriptor]
    sorted_symbols: Sequence[str]


class PublicUtilities(Node):
    fields: Sequence[Field]


class CopyPointers(Node):
    fields: Sequence[Field]


class PrivateMembers(Node):
    fields: Sequence[Field]
    out_fields: Sequence[Field]


class StencilClassSetupFunc(CppSetupFuncDeclaration):
    ...


class StencilClass(Node):
    funcname: str
    gpu_tri_mesh: GpuTriMesh
    run_fun: StenClassRunFun
    public_utilities: PublicUtilities
    copy_pointers: CopyPointers
    private_members: PrivateMembers
    setup_func: StencilClassSetupFunc


class Params(Node):
    fields: Sequence[Field]


class RunFunc(Node):
    funcname: str
    params: Params
    run_func_declaration: CppRunFuncDeclaration


class MetricsSerialisation(Node):
    funcname: str
    out_fields: Sequence[Field]


class VerifyFunc(Node):
    funcname: str
    verify_func_declaration: CppVerifyFuncDeclaration
    metrics_serialisation: MetricsSerialisation


class RunFuncCall(CppRunFuncDeclaration):
    ...


class VerifyFuncCall(CppVerifyFuncDeclaration):
    ...


class SetupFunc(CppFunc):
    func_declaration: CppSetupFuncDeclaration


class RunAndVerifyFunc(Node):
    funcname: str
    run_verify_func_declaration: CppRunAndVerifyFuncDeclaration
    run_func_call: RunFuncCall
    verify_func_call: VerifyFuncCall


class FreeFunc(CppFreeFunc):
    ...


class CppDefTemplate(Node):
    stencil_name: str
    fields: Sequence[Field]
    offsets: Sequence[Offset]

    includes: IncludeStatements = eve.datamodels.field(init=False)
    stencil_class: StencilClass = eve.datamodels.field(init=False)
    run_func: RunFunc = eve.datamodels.field(init=False)
    verify_func: VerifyFunc = eve.datamodels.field(init=False)
    run_verify_func: RunAndVerifyFunc = eve.datamodels.field(init=False)
    setup_func: SetupFunc = eve.datamodels.field(init=False)
    free_func: FreeFunc = eve.datamodels.field(init=False)

    def _get_field_data(self) -> tuple:
        output_fields = [field for field in self.fields if field.intent.out]
        tolerance_fields = [field for field in output_fields if not field.is_integral()]
        # since we can vertical fields as dense fields for the purpose of this function, lets include them here
        dense_fields = [
            field
            for field in self.fields
            if field.is_dense() or (field.has_vertical_dimension and field.rank() == 1)
        ]
        sparse_fields = [field for field in self.fields if field.is_sparse()]
        compound_fields = [field for field in self.fields if field.is_compound()]
        sparse_offsets = [offset for offset in self.offsets if not offset.is_compound_location()]
        strided_offsets = [offset for offset in self.offsets if offset.is_compound_location()]
        all_fields = self.fields

        offsets = dict(sparse=sparse_offsets, strided=strided_offsets)
        fields = dict(
            output=output_fields,
            dense=dense_fields,
            sparse=sparse_fields,
            compound=compound_fields,
            all_fields=all_fields,
            tolerance=tolerance_fields,
        )
        return fields, offsets

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        fields, offsets = self._get_field_data()
        data_descriptors = set(DataDescriptor(x) for x in [*fields["all_fields"], *self.offsets])
        offset_renderer = GpuTriMeshOffsetRenderer(self.offsets)
        domain_args = [
            Scalar("vertical_start", "verticalStart"),
            Scalar("vertical_end", "verticalEnd"),
            Scalar("horizontal_start", "horizontalStart"),
            Scalar("horizontal_end", "horizontalEnd"),
        ]

        symbol_map: dict[str, Optional[str]] = {}
        symbol_map.update({arg.sdfg_arg_name(): arg.cpp_arg_name() for arg in domain_args})
        symbol_map.update(
            {data_descr.sdfg_arg_name(): data_descr.strides() for data_descr in data_descriptors}
        )

        def key_data_descr(x: DataDescriptor | str) -> str:
            return x.sdfg_arg_name() if isinstance(x, DataDescriptor) else x

        self.includes = IncludeStatements(
            funcname=self.stencil_name,
        )

        self.stencil_class = StencilClass(
            funcname=self.stencil_name,
            gpu_tri_mesh=GpuTriMesh(
                table_vars=offset_renderer.make_table_vars(),
                neighbor_tables=offset_renderer.make_neighbor_tables(),
                has_offsets=offset_renderer.has_offsets,
            ),
            run_fun=StenClassRunFun(
                stencil_name=self.stencil_name,
                domain_args=domain_args,
                sorted_data_descriptors=sorted(data_descriptors, key=key_data_descr),
                sorted_symbols=[
                    symbols
                    for arg, symbols in collections.OrderedDict(sorted(symbol_map.items())).items()
                    if symbols is not None
                ],
            ),
            public_utilities=PublicUtilities(fields=fields["output"]),
            copy_pointers=CopyPointers(fields=self.fields),
            private_members=PrivateMembers(fields=self.fields, out_fields=fields["output"]),
            setup_func=StencilClassSetupFunc(
                funcname=self.stencil_name,
            ),
        )

        self.run_func = RunFunc(
            funcname=self.stencil_name,
            params=Params(fields=self.fields),
            run_func_declaration=CppRunFuncDeclaration(
                funcname=self.stencil_name,
                fields=self.fields,
                out_fields=fields["output"],
                k_size_suffix="k_size",
            ),
        )

        self.verify_func = VerifyFunc(
            funcname=self.stencil_name,
            verify_func_declaration=CppVerifyFuncDeclaration(
                funcname=self.stencil_name,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                before_suffix="dsl",
                k_size_suffix="k_size",
            ),
            metrics_serialisation=MetricsSerialisation(
                funcname=self.stencil_name, out_fields=fields["output"]
            ),
        )

        self.run_verify_func = RunAndVerifyFunc(
            funcname=self.stencil_name,
            run_verify_func_declaration=CppRunAndVerifyFuncDeclaration(
                funcname=self.stencil_name,
                fields=self.fields,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                before_suffix="before",
                k_size_suffix="k_size",
            ),
            run_func_call=RunFuncCall(
                funcname=self.stencil_name,
                fields=self.fields,
                out_fields=fields["output"],
                k_size_suffix="k_size",
            ),
            verify_func_call=VerifyFuncCall(
                funcname=self.stencil_name,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                before_suffix="before",
                k_size_suffix="k_size",
            ),
        )

        self.setup_func = SetupFunc(
            funcname=self.stencil_name,
            func_declaration=CppSetupFuncDeclaration(
                funcname=self.stencil_name,
            ),
        )

        self.free_func = FreeFunc(funcname=self.stencil_name)


def generate_cpp_definition(
    stencil_name: str,
    fields: Sequence[Field],
    offsets: Sequence[Offset],
    outpath: Path,
) -> None:
    definition = CppDefTemplate(
        stencil_name=stencil_name,
        fields=fields,
        offsets=offsets,
    )
    source = format_source("cpp", CppDefGenerator.apply(definition), style="LLVM")
    write_string(source, outpath, f"{stencil_name}.cpp")
