# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
DaCe Orchestration Decorator

Creates a DaCe SDFG that fuses any GT4Py Program called by the decorated function.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union, get_type_hints

import gt4py.next as gtx
import numpy as np
from gt4py._core import definitions as core_defs

from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.orchestration import dtypes as orchestration_dtypes


try:
    import dace
except ImportError:
    dace: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import ghex
    from ghex import expose_cpp_ptr
except ImportError:
    ghex: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py: Optional[ModuleType] = None

if dace:
    from dace import hooks
    from dace.transformation.passes.simplify import SimplifyPass
    from gt4py.next.program_processors.runners.dace_common.utility import (
        connectivity_identifier,
    )


def orchestrate(func: Callable | None = None, *, method: bool | None = None):
    def _decorator(fuse_func: Callable):
        compiled_sdfgs = {}  # Caching

        def wrapper(*args, **kwargs):
            if settings.dace_orchestration is not None:
                if "dace" not in settings.backend.executor.name.lower():
                    raise ValueError(
                        "DaCe Orchestration works only with DaCe backends. Change the backend to a DaCe supported one."
                    )

                if method:
                    # self is used to retrieve the _exchange object -on the fly halo exchanges- and the grid object -offset providers-
                    self = args[0]
                    self_name = next(iter(inspect.signature(fuse_func).parameters))
                else:
                    raise ValueError(
                        "The orchestration decorator is only for methods -at least for now-."
                    )

                fuse_func_orig_annotations = copy.deepcopy(fuse_func.__annotations__)
                fuse_func.__annotations__ = to_dace_annotations(
                    fuse_func
                )  # every arg/kwarg is annotated with DaCe data types

                exchange_obj = None
                grid = None
                for attr_name, attr_value in self.__dict__.items():
                    if isinstance(attr_value, decomposition.ExchangeRuntime):
                        exchange_obj = getattr(self, attr_name)
                    if isinstance(attr_value, icon_grid.IconGrid):
                        grid = getattr(self, attr_name)

                if not grid:
                    raise ValueError("No grid object found.")

                order_kwargs_by_annotations(fuse_func, kwargs)

                compile_time_args_kwargs = {}
                all_args_kwargs = [*args, *kwargs.values()]
                for i, (k, v) in enumerate(fuse_func.__annotations__.items()):
                    if v is dace.compiletime:
                        compile_time_args_kwargs[k] = all_args_kwargs[i]

                unique_id = make_uid(fuse_func, compile_time_args_kwargs, exchange_obj)

                default_build_folder = Path(".dacecache") / f"uid_{unique_id}"

                parse_compile_cache_sdfg(
                    unique_id,
                    compiled_sdfgs,
                    default_build_folder,
                    exchange_obj,
                    fuse_func,
                    compile_time_args_kwargs,
                    self_name,
                    simplify_fused_sdfg=True,
                )
                dace_program = compiled_sdfgs[unique_id]["dace_program"]
                sdfg = compiled_sdfgs[unique_id]["sdfg"]
                compiled_sdfg = compiled_sdfgs[unique_id]["compiled_sdfg"]

                # update the args/kwargs with runtime related values, such as
                # concretized symbols, runtime connectivity tables, GHEX C++ pointers, and DaCe structures pointers
                updated_args, updated_kwargs = mod_xargs_for_dace_structures(
                    fuse_func, fuse_func_orig_annotations, args, kwargs
                )
                updated_kwargs = {
                    **updated_kwargs,
                    **dace_specific_kwargs(exchange_obj, grid.offset_providers),
                }
                updated_kwargs = {
                    **updated_kwargs,
                    **dace_symbols_concretization(
                        grid, fuse_func, fuse_func_orig_annotations, args, kwargs
                    ),
                }
                #

                sdfg_args = dace_program._create_sdfg_args(sdfg, updated_args, updated_kwargs)
                if method:
                    del sdfg_args[self_name]

                fuse_func.__annotations__ = (
                    fuse_func_orig_annotations  # restore the original annotations
                )

                with dace.config.temporary_config():
                    dace.config.Config.set(
                        "compiler", "allow_view_arguments", value=True
                    )  # Allow numpy views as arguments: If true, allows users to call DaCe programs with NumPy views (for example, “A[:,1]” or “w.T”)
                    return compiled_sdfg(**sdfg_args)
            else:
                return fuse_func(*args, **kwargs)

        return wrapper

    return _decorator(func) if func else _decorator


def make_uid(
    fuse_func: Callable,
    compile_time_args_kwargs: dict[str, Any],
    exchange_obj: Optional[decomposition.ExchangeRuntime],
) -> str:
    """Generate unique key to retrieve from compiled_sdfgs dict the cached objects."""
    if len(compile_time_args_kwargs) == 0:
        unique_id = fuse_func.__name__
    else:
        l_uids = []
        for ct_key, ct_val in compile_time_args_kwargs.items():
            if hasattr(ct_val, "orchestration_uid"):
                l_uids.append(ct_val.orchestration_uid)
            else:
                l_uids.append(generate_orchestration_uid(ct_val, ct_key))

        if exchange_obj:
            my_rank = exchange_obj.my_rank()
            mpi_size = exchange_obj.my_rank()
        else:
            my_rank = 0 if not mpi4py else MPI.COMM_WORLD.Get_rank()
            mpi_size = 1 if not mpi4py else MPI.COMM_WORLD.Get_size()

        l_uids.append(str({"MPI_rank": my_rank, "MPI_size": mpi_size}))

        unique_id = uid_from_hashlib(str(l_uids))

    return unique_id


def generate_orchestration_uid(obj: Any, obj_name: str = "") -> str:
    """Generate a unique id for a runtime object.

    The unique id is generated by creating a dictionary that describes the runtime state of the object.
    For primitive types, the dictionary contains the type and the value.
    For arrays, the dictionary contains the shape and the dtype -not their content-.
    """
    primitive_dtypes = (*orchestration_dtypes.ICON4PY_PRIMITIVE_DTYPES, str, uuid.UUID, np.dtype)

    unique_dict = {}

    def _populate_entry(key: str, value: Any, parent_key: str = ""):
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, primitive_dtypes):
            unique_dict[full_key] = {"type": "primitive_dtypes", "value": str(value)}
        elif isinstance(value, (np.ndarray, gtx.Field)):
            unique_dict[full_key] = {
                "type": "array/field",
                "shape": value.shape,
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (list, tuple)):
            if all(isinstance(i, primitive_dtypes) for i in value):
                unique_dict[full_key] = {
                    "type": "array-like[primitive_dtypes]",
                    "length": len(value),
                }
            else:
                for i, v in enumerate(value):
                    _populate_entry(str(i), v, full_key)
        elif value is None:
            unique_dict[full_key] = {"type": "None", "value": None}
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            _populate_unique_dict(value, full_key)
        else:
            raise ValueError(f"Type {type(value)} is not supported.")

    def _populate_unique_dict(obj: Any, parent_key: str = ""):
        if (hasattr(obj, "__dict__") or isinstance(obj, dict)) and not isinstance(
            obj, decomposition.ExchangeRuntime
        ):
            obj_to_traverse = obj.__dict__ if hasattr(obj, "__dict__") else obj
            for key, value in obj_to_traverse.items():
                _populate_entry(key, value, parent_key)

    if hasattr(obj, "__dict__") or isinstance(obj, dict):
        _populate_unique_dict(obj)
    else:
        _populate_entry(obj_name, obj)

    return uid_from_hashlib(str(unique_dict))


def uid_from_hashlib(unique_string: str) -> str:
    unique_string_bytes = unique_string.encode("utf-8")
    hash_object = hashlib.sha256(unique_string_bytes)
    uid = hash_object.hexdigest()
    return uid


def order_kwargs_by_annotations(f: Callable, kwargs: dict[str, Any]) -> None:
    """Order in-place the kwargs based on the annotations of the function."""
    annotations = f.__annotations__
    ordered_items = [(key, kwargs[key]) for key in annotations if key in kwargs]
    kwargs.clear()
    kwargs.update(ordered_items)


def dace_inhibitor(f: Callable):
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return f


@dace_inhibitor
def wait(comm_handle: Union[int, decomposition.ExchangeResult]):
    if isinstance(comm_handle, int):
        pass
    else:
        comm_handle.wait()


def build_compile_time_connectivities(
    offset_providers: dict[str, gtx.common.Connectivity],
) -> dict[str, gtx.common.Connectivity]:
    connectivities = {}
    for k, v in offset_providers.items():
        if hasattr(v, "table"):
            connectivities[k] = gtx.otf.arguments.CompileTimeConnectivity(
                v.max_neighbors, v.has_skip_values, v.origin_axis, v.neighbor_axis, v.table.dtype
            )
        else:
            connectivities[k] = v

    return connectivities


if dace:

    def to_dace_annotations(fuse_func: Callable) -> dict[str, Any]:
        """Translate the annotations of the function to DaCe-compatible ones."""
        dace_annotations = {}

        fuse_func_type_hints = get_type_hints(fuse_func)
        for param in inspect.signature(fuse_func).parameters.values():
            if param.name in fuse_func_type_hints:
                if hasattr(fuse_func_type_hints[param.name], "__origin__"):
                    if fuse_func_type_hints[param.name].__origin__ is gtx.Field:
                        dims_ = fuse_func_type_hints[param.name].__args__[0].__args__
                        dace_dims = [
                            orchestration_dtypes.gt4py_dim_to_dace_symbol(dim_) for dim_ in dims_
                        ]
                        dtype_ = fuse_func_type_hints[param.name].__args__[1]
                        dace_annotations[param.name] = dace.data.Array(
                            dtype=orchestration_dtypes.DACE_PRIMITIVE_DTYPES[
                                orchestration_dtypes.ICON4PY_PRIMITIVE_DTYPES.index(dtype_)
                            ],
                            shape=dace_dims,
                        )
                    else:
                        raise ValueError(
                            f"The type hint [{fuse_func_type_hints[param.name]}] is not supported."
                        )
                elif hasattr(fuse_func_type_hints[param.name], "__dataclass_fields__"):
                    dace_annotations[param.name] = dace.data.Structure(
                        orchestration_dtypes.dace_structure_dict(fuse_func_type_hints[param.name]),
                        name=fuse_func_type_hints[param.name].__name__,
                    )
                elif (
                    fuse_func_type_hints[param.name]
                    in orchestration_dtypes.ICON4PY_PRIMITIVE_DTYPES
                ):
                    dace_annotations[param.name] = orchestration_dtypes.DACE_PRIMITIVE_DTYPES[
                        orchestration_dtypes.ICON4PY_PRIMITIVE_DTYPES.index(
                            fuse_func_type_hints[param.name]
                        )
                    ]
                else:
                    raise ValueError(
                        f"The type hint [{fuse_func_type_hints[param.name]}] is not supported."
                    )
            else:
                dace_annotations[param.name] = dace.compiletime

        return dace_annotations

    def dev_type_from_gt4py_to_dace(device_type: core_defs.DeviceType) -> dace.dtypes.DeviceType:
        if device_type == core_defs.DeviceType.CPU:
            return dace.dtypes.DeviceType.CPU
        elif device_type == core_defs.DeviceType.CUDA:
            return dace.dtypes.DeviceType.GPU
        else:
            raise ValueError("The device type is not supported.")

    def parse_compile_cache_sdfg(
        unique_id: str,
        compiled_sdfgs: dict[str, dict[str, Any]],
        default_build_folder: Path,
        exchange_obj: Optional[decomposition.ExchangeRuntime],
        fuse_func: Callable,
        compile_time_args_kwargs: dict[str, Any],
        self_name: Optional[str] = None,
        simplify_fused_sdfg: bool = True,
    ) -> None:
        """Function that parses, compiles and caches the fused SDFG along with adding the halo exchanges."""
        if unique_id in compiled_sdfgs:
            return

        compiled_sdfgs[unique_id] = compiled_sdfg = {}

        with dace.config.temporary_config():
            device_type = configure_dace_temp_env(default_build_folder)

            compiled_sdfg["dace_program"] = dace_program = dace.program(
                auto_optimize=False,
                device=dev_type_from_gt4py_to_dace(device_type),
                distributed_compilation=False,
            )(fuse_func)

            # export DACE_ORCH_CACHE_FROM_DISK=(True or 1) if you want to activate it
            cache_from_disk = get_env_bool("DACE_ORCH_CACHE_FROM_DISK", default=False)
            dace_program_location = Path(
                default_build_folder
            ) / f"{fuse_func.__module__}.{fuse_func.__name__}".replace(".", "_")

            if cache_from_disk and (Path(dace_program_location) / "program.sdfg").exists():
                try:
                    if self_name:
                        self = compile_time_args_kwargs.pop(self_name)
                        compiled_sdfg["sdfg"], _ = dace_program.load_sdfg(
                            Path(dace_program_location) / "program.sdfg",
                            self,
                            **compile_time_args_kwargs,
                        )
                        (
                            compiled_sdfg["compiled_sdfg"],
                            _,
                        ) = dace_program.load_precompiled_sdfg(
                            dace_program_location, self, **compile_time_args_kwargs
                        )
                    else:
                        compiled_sdfg["sdfg"], _ = dace_program.load_sdfg(
                            Path(dace_program_location) / "program.sdfg",
                            **compile_time_args_kwargs,
                        )
                        (
                            compiled_sdfg["compiled_sdfg"],
                            _,
                        ) = dace_program.load_precompiled_sdfg(
                            dace_program_location, **compile_time_args_kwargs
                        )
                except:  # noqa: E722
                    raise ValueError(
                        f"Corrupted cache. Remove `{default_build_folder}` folder and re-run the program."
                    ) from None
            else:
                if self_name:
                    self = compile_time_args_kwargs.pop(self_name)
                    compiled_sdfg["sdfg"] = dace_program.to_sdfg(
                        self, **compile_time_args_kwargs, simplify=False, validate=True
                    )
                else:
                    compiled_sdfg["sdfg"] = dace_program.to_sdfg(
                        **compile_time_args_kwargs, simplify=False, validate=True
                    )
                sdfg = compiled_sdfg["sdfg"]

                if sdfg.name != f"{fuse_func.__module__}.{fuse_func.__name__}".replace(".", "_"):
                    raise ValueError(
                        "fused_SDFG.name != {fuse_func.__module__}.{fuse_func.__name__}"
                    )

                if simplify_fused_sdfg:
                    SimplifyPass(
                        validate=True,
                        validate_all=False,
                        verbose=False,
                        skip={
                            "InlineSDFGs",
                            "DeadDataflowElimination",
                        },
                    ).apply_pass(sdfg, {})
                    # Alternatively:
                    # sdfg.simplify(validate=True) # noqa: ERA001

                if exchange_obj:
                    exchange_obj.num_of_halo_tasklets = (
                        0  # reset the counter for the next fused SDFG
                    )

                sdfg.save(Path(dace_program_location) / "program.sdfg")

                with hooks.invoke_sdfg_call_hooks(sdfg) as sdfg_:
                    # TODO(kotsaloscv): Re-think the distributed compilation -all args need to be type annotated and not dace.compiletime-
                    compiled_sdfg["compiled_sdfg"] = sdfg_.compile(validate=dace_program.validate)

    def get_env_bool(env_var_name: str, default: bool = False) -> bool:
        value = os.getenv(env_var_name, str(default)).lower()
        return value in ("true", "1")

    def configure_dace_temp_env(default_build_folder: Path) -> core_defs.DeviceType:
        dace.config.Config.set("default_build_folder", value=default_build_folder)
        dace.config.Config.set(
            "compiler", "allow_view_arguments", value=True
        )  # Allow numpy views as arguments: If true, allows users to call DaCe programs with NumPy views (for example, “A[:,1]” or “w.T”)
        dace.config.Config.set(
            "optimizer", "automatic_simplification", value=False
        )  # simplifications & optimizations after placing halo exchanges -need a sequential structure of nested sdfgs-
        dace.config.Config.set("optimizer", "autooptimize", value=False)
        device_type = settings.backend.executor.otf_workflow.step.translation.device_type
        if device_type == core_defs.DeviceType.CPU:
            device = "cpu"
            compiler_args = dace.config.Config.get("compiler", "cpu", "args")
            compiler_args = compiler_args.replace("-std=c++14", "-std=c++17")  # needed for GHEX
            # disable finite-math-only in order to support isfinite/isinf/isnan builtins
            if "-ffast-math" in compiler_args:
                compiler_args += " -fno-finite-math-only"
            if "-ffinite-math-only" in compiler_args:
                compiler_args = compiler_args.replace("-ffinite-math-only", "")
        elif device_type == core_defs.DeviceType.CUDA:
            device = "cuda"
            compiler_args = dace.config.Config.get("compiler", "cuda", "args")
            compiler_args = compiler_args.replace("-Xcompiler", "-Xcompiler -std=c++17")
            compiler_args += " -std=c++17"
        else:
            raise ValueError("The device type is not supported.")
        dace.config.Config.set("compiler", device, "args", value=compiler_args)

        return device_type

    def get_stride_from_numpy_to_dace(numpy_array: np.ndarray, axis: int) -> int:
        # GHEX/NumPy strides: number of bytes to jump
        # DaCe strides: number of elements to jump
        return numpy_array.strides[axis] // numpy_array.itemsize

    def dace_specific_kwargs(
        exchange_obj: Optional[decomposition.ExchangeRuntime],
        offset_providers: dict[str, gtx.common.Connectivity],
    ) -> dict[str, Any]:
        """kwargs needed by the compiled SDFG.

        The additional kwargs are the connectivity tables (runtime tables) and the GHEX C++ pointers.
        """
        return {
            # connectivity tables at runtime
            **{
                connectivity_identifier(k): v.table
                for k, v in offset_providers.items()
                if hasattr(v, "table")
            },
            # GHEX C++ ptrs
            "__context_ptr": expose_cpp_ptr(exchange_obj._context)
            if not isinstance(exchange_obj, decomposition.SingleNodeExchange)
            else 0,
            "__comm_ptr": expose_cpp_ptr(exchange_obj._comm)
            if not isinstance(exchange_obj, decomposition.SingleNodeExchange)
            else 0,
            **{
                f"__pattern_{dim.value}Dim_ptr": expose_cpp_ptr(exchange_obj._patterns[dim])
                if not isinstance(exchange_obj, decomposition.SingleNodeExchange)
                else 0
                for dim in dims.global_dimensions.values()
            },
            **{
                f"__domain_descriptor_{dim.value}Dim_ptr": expose_cpp_ptr(
                    exchange_obj._domain_descriptors[dim].__wrapped__
                )
                if not isinstance(exchange_obj, decomposition.SingleNodeExchange)
                else 0
                for dim in dims.global_dimensions.values()
            },
        }

    def modified_orig_annotations(
        fuse_func: Callable, fuse_func_orig_annotations: dict[str, Any]
    ) -> dict[str, Any]:
        """Add None as annotation to the the non-annotated args/kwargs."""
        ii = 0
        modified_fuse_func_orig_annotations = {}
        for i, annotation_ in enumerate(fuse_func.__annotations__.values()):
            if annotation_ is dace.compiletime:
                modified_fuse_func_orig_annotations[
                    list(fuse_func.__annotations__.keys())[i]
                ] = None
            else:
                modified_fuse_func_orig_annotations[
                    list(fuse_func.__annotations__.keys())[i]
                ] = list(fuse_func_orig_annotations.values())[ii]
                ii += 1
        return modified_fuse_func_orig_annotations

    def dace_symbols_concretization(
        grid: icon_grid.IconGrid,
        fuse_func: Callable,
        fuse_func_orig_annotations: dict[str, Any],
        args: Any,
        kwargs: Any,
    ) -> dict[str, Any]:
        """Since the generated SDFGs are parametric, we need to concretize the symbols before calling the compiled SDFGs."""
        flattened_xargs_type_value = list(
            zip(
                [*fuse_func.__annotations__.values()],
                [*args, *kwargs.values()],
                strict=True,
            )
        )

        def _concretize_symbols_for_dace_structure(dace_cls, orig_cls):
            concretized_symbols = {}
            for k_v in flattened_xargs_type_value:
                if k_v[0] is not dace_cls:
                    continue
                for member in orig_cls.__dataclass_fields__.keys():
                    for stride in range(getattr(k_v[1], member).ndarray.ndim):
                        concretized_symbols[
                            orchestration_dtypes.stride_symbol_name_from_field(
                                orig_cls, member, stride
                            )
                        ] = get_stride_from_numpy_to_dace(getattr(k_v[1], member).ndarray, stride)
            return concretized_symbols

        modified_fuse_func_orig_annotations = modified_orig_annotations(
            fuse_func, fuse_func_orig_annotations
        )

        concretize_symbols_for_dace_structure = {}
        for annotation_, annotation_orig_ in zip(
            fuse_func.__annotations__.values(),
            modified_fuse_func_orig_annotations.values(),
            strict=True,
        ):
            if type(annotation_) is not dace.data.Structure:
                continue
            concretize_symbols_for_dace_structure.update(
                _concretize_symbols_for_dace_structure(annotation_, annotation_orig_)
            )

        return {
            **{
                "CellDim_sym": grid.offset_providers["C2E"].table.shape[0],
                "EdgeDim_sym": grid.offset_providers["E2C"].table.shape[0],
                "KDim_sym": grid.num_levels,
            },
            **concretize_symbols_for_dace_structure,
        }

    def mod_xargs_for_dace_structures(
        fuse_func: Callable, fuse_func_orig_annotations: dict[str, Any], args: Any, kwargs: Any
    ) -> tuple:
        """Modify the args/kwargs to support DaCe Structures, i.e., teach DaCe how to extract the data from the corresponding Python data classes"""
        flattened_xargs_type_value = list(
            zip(
                [*fuse_func.__annotations__.values()],
                [*args, *kwargs.values()],
                strict=True,
            )
        )
        # initialize
        orig_args_kwargs = list(args) + list(kwargs.values())
        mod_args_kwargs = list(args) + list(kwargs.values())

        def _mod_xargs_for_dace_structure(dace_cls, orig_cls):
            for i, k_v in enumerate(flattened_xargs_type_value):
                if k_v[0] is dace_cls:
                    mod_args_kwargs[i] = dace_cls.dtype._typeclass.as_ctypes()(
                        **{
                            member: getattr(k_v[1], member).data_ptr()
                            for member in orig_cls.__dataclass_fields__.keys()
                        }
                    )

        modified_fuse_func_orig_annotations = modified_orig_annotations(
            fuse_func, fuse_func_orig_annotations
        )

        for annotation_, annotation_orig_ in zip(
            fuse_func.__annotations__.values(),
            modified_fuse_func_orig_annotations.values(),
            strict=True,
        ):
            if type(annotation_) is not dace.data.Structure:
                continue
            # modify mod_args_kwargs in place
            _mod_xargs_for_dace_structure(
                annotation_,
                annotation_orig_,
            )

        for i, mod_arg_kwarg in enumerate(mod_args_kwargs):
            if hasattr(mod_arg_kwarg, "_fields_"):
                mod_arg_kwarg.descriptor = dace.data.Structure(
                    orchestration_dtypes.dace_structure_dict(orig_args_kwargs[i].__class__),
                    name=orig_args_kwargs[i].__class__.__name__,
                )

        return tuple(mod_args_kwargs[0 : len(args)]), {
            k: v for k, v in zip(kwargs.keys(), mod_args_kwargs[len(args) :], strict=True)
        }
