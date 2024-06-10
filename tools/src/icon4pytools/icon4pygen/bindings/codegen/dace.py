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
import warnings
from pathlib import Path
from typing import Any, Sequence

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
        #include \"{{ funcname }}_dace.h\"
        #include "unstructured_interface.hpp"
        #include "verification_metrics.hpp"
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
        (void)__dace_exit_{{ funcname }}(handle_);
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

        handle_ = __dace_init_{{ funcname }}(
        {%- for param in _this_node.sdfg_arglist -%}
            {{_this_node.sdfg_symbols[param] if param not in _this_node.sdfg_scalars else 0}}{%- if not loop.last -%}, {%- endif -%}
        {%- endfor -%});
        __set_stream_{{ funcname }}(handle_, stream);
        }
        """
    )

    PrivateMembers = as_jinja(
        """\
        private:
        {%- for field in _this_node.fields -%}
        inline static {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_;
        {%- endfor -%}
        inline static GpuTriMesh mesh_;
        inline static bool is_setup_;
        inline static cudaStream_t stream_;
        inline static json* jsonRecord_;
        inline static MeshInfoVtk* mesh_info_vtk_;
        inline static verify* verify_;
        inline static {{ funcname }}Handle_t handle_;
        """
    )

    StenClassRunFun = as_jinja(
        """
      void run(const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
        if (!is_setup_) {
            printf("{{ stencil_name }} has not been set up! make sure setup() is called before run!\\n");
            return;
        }

        __program_{{ stencil_name }}(handle_
        {%- for param in _this_node.sdfg_arglist -%}
            , {{_this_node.sdfg_args[param]}}
        {%- endfor -%});
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
    sdfg_arglist: list[str]
    sdfg_args: dict[str, str]


class PublicUtilities(Node):
    fields: Sequence[Field]
    funcname: str


class CopyPointers(Node):
    fields: Sequence[Field]


class PrivateMembers(Node):
    fields: Sequence[Field]
    out_fields: Sequence[Field]
    funcname: str


class StencilClassSetupFunc(CppSetupFuncDeclaration):
    sdfg_arglist: list[str]
    sdfg_scalars: set[str]
    sdfg_symbols: dict[str, str]


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
    fields: list[Field]
    offsets: list[Offset]
    arglist_init: list[str]
    arglist_run: list[str]

    includes: IncludeStatements = eve.datamodels.field(init=False)
    stencil_class: StencilClass = eve.datamodels.field(init=False)
    run_func: RunFunc = eve.datamodels.field(init=False)
    verify_func: VerifyFunc = eve.datamodels.field(init=False)
    run_verify_func: RunAndVerifyFunc = eve.datamodels.field(init=False)
    setup_func: SetupFunc = eve.datamodels.field(init=False)
    free_func: FreeFunc = eve.datamodels.field(init=False)
    k_size_suffix: str = "k_size"

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
        fields, _ = self._get_field_data()
        offset_renderer = GpuTriMeshOffsetRenderer(self.offsets)

        array_args = {}
        scalar_args = {
            "vertical_start",
            "vertical_end",
            "horizontal_start",
            "horizontal_end",
        }
        symbol_args = {}
        for f in fields["all_fields"]:
            if f.rank() == 0:
                # this is a scalar, therefore passed to the SDFG as a dace symbol
                symbol_args[f.name] = f"{f.name}_"
                scalar_args.add(f.name)
            else:
                # a regular field, therefore passed as an array with symbolic shape and strides
                array_args[f.name] = f"{f.name}_"
                strides = f.renderer.render_strides(use_dense_rank=False).split(",")
                for i, s in enumerate(strides):
                    symbol_args[f"__{f.name}_stride_{i}"] = s
        output_fields = {
            f.name for f in fields["output"]
        }
        for f in fields["all_fields"]:
            for i in range(f.rank()):
                size_symbol = f"__{f.name}_size_{i}"
                if f.name in output_fields and f.rank() != 0:
                    assert i < 2
                    if i == 0:
                        symbol_value = f"mesh_.{f.renderer.render_horizontal_size()}"
                    else:
                        symbol_value = f"{f.name}_{self.k_size_suffix}"
                    symbol_args[size_symbol] = symbol_value
                elif size_symbol in self.arglist_init and size_symbol not in scalar_args:
                    warnings.warn(f"Field size {size_symbol} not found as scalar argument.")
                    # TODO: the field size is supposed to be passed as a scalar argument
                    # but sometimes it is not, so we need to consider the max size of the field
                    symbol_args[size_symbol] = "0"
        for f in self.offsets:
            array_param = f"__connectivity_{f.renderer.render_uppercase_shorthand()}"
            array_arg = f"mesh_.{f.renderer.render_lowercase_shorthand()}Table"
            array_args[array_param] = array_arg
            symbol_args[f"__{array_param}_stride_0"] = "1"
            symbol_args[f"__{array_param}_stride_1"] = "mesh_.Nproma"
        # add symbols for domain arguments
        symbol_args["vertical_start"] = "verticalStart"
        symbol_args["vertical_end"] = "verticalEnd"
        symbol_args["horizontal_start"] = "horizontalStart"
        symbol_args["horizontal_end"] = "horizontalEnd"

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
                sdfg_arglist=self.arglist_run,
                sdfg_args=(array_args | symbol_args),
            ),
            public_utilities=PublicUtilities(fields=fields["output"], funcname=self.stencil_name),
            copy_pointers=CopyPointers(fields=self.fields),
            private_members=PrivateMembers(
                fields=self.fields, out_fields=fields["output"], funcname=self.stencil_name
            ),
            setup_func=StencilClassSetupFunc(
                funcname=self.stencil_name,
                sdfg_arglist=self.arglist_init,
                sdfg_scalars=scalar_args,
                sdfg_symbols=symbol_args,
            ),
        )

        self.run_func = RunFunc(
            funcname=self.stencil_name,
            params=Params(fields=self.fields),
            run_func_declaration=CppRunFuncDeclaration(
                funcname=self.stencil_name,
                fields=self.fields,
                out_fields=fields["output"],
                k_size_suffix=self.k_size_suffix,
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
    arglist_init: Sequence[str],
    arglist_run: Sequence[str],
) -> None:
    definition = CppDefTemplate(
        stencil_name=stencil_name,
        fields=fields,
        offsets=offsets,
        arglist_init=arglist_init,
        arglist_run=arglist_run,
    )
    source = format_source("cpp", CppDefGenerator.apply(definition), style="LLVM")
    write_string(source, outpath, f"{stencil_name}.cpp")
