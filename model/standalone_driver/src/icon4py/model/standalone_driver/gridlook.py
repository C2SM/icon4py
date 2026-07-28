# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Export driver output for the gridlook web viewer.

Gridlook (https://github.com/d70-t/gridlook) renders zarr stores on a
globe directly in the browser. It cannot read the driver's output as
written: it needs the triangular mesh geometry inside the store, cell
fields in the undecomposed global cell order, and consolidated metadata,
while driver output carries no geometry, rank-block zarr stores (see
:mod:`icon4py.model.common.io.distributed`) are padded and ordered by
rank, and netCDF files are not zarr stores at all. The ``export`` command
bridges the gap for any driver output -- a zarr store (serial, gathered
or rank-block) or a netCDF file: it reorders every cell field into global
order, grafts the mesh geometry from the run's ICON grid file, and writes
a new self-contained zarr store. The ``serve`` command hosts the exported
store over HTTP with the CORS header gridlook requires (the viewer
fetches data over HTTP only; it has no local-file mode).
"""

import contextlib
import dataclasses
import functools
import http.server
import pathlib
import shutil
import sys
import uuid
import warnings
from collections.abc import Iterator
from typing import Annotated, Any, Final, TypeAlias

import netCDF4 as nc
import numpy as np
import typer
import zarr

from icon4py.model.common.io import writers


GRIDLOOK_APP_URL: Final[str] = "https://gridlook.pages.dev"

#: Cartesian vertex coordinates of the triangular mesh, copied verbatim from the
#: grid file when present (e.g. torus grids).
VERTEX_COORDINATE_VARIABLES: Final[tuple[str, ...]] = (
    "cartesian_x_vertices",
    "cartesian_y_vertices",
    "cartesian_z_vertices",
)
#: Spherical vertex coordinates (radians), the fallback the Cartesian coordinates
#: are derived from (icosahedral grid files carry only these).
SPHERICAL_VERTEX_COORDINATE_VARIABLES: Final[tuple[str, str]] = ("vlon", "vlat")
VERTEX_OF_CELL: Final[str] = "vertex_of_cell"

_GRID_UUID_ATTRIBUTE: Final[str] = "uuidOfHGrid"
_CELL_GLOBAL_INDEX: Final[str] = f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"

app = typer.Typer(no_args_is_help=True)


@dataclasses.dataclass(frozen=True)
class GridGeometry:
    """Triangular mesh of an ICON grid file, in the form gridlook consumes."""

    vertex_of_cell: np.ndarray
    vertex_coordinates: dict[str, np.ndarray]
    grid_uuid: str | None

    @property
    def num_cells(self) -> int:
        return int(self.vertex_of_cell.shape[1])


def _cartesian_from_spherical(vlon: np.ndarray, vlat: np.ndarray) -> dict[str, np.ndarray]:
    """Unit-sphere Cartesian vertex coordinates from spherical ones (radians)."""
    x_name, y_name, z_name = VERTEX_COORDINATE_VARIABLES
    return {
        x_name: np.cos(vlat) * np.cos(vlon),
        y_name: np.cos(vlat) * np.sin(vlon),
        z_name: np.sin(vlat),
    }


def read_grid_geometry(grid_file: pathlib.Path) -> GridGeometry:
    """Read the triangular mesh from an ICON grid file.

    Vertex positions are copied from the Cartesian coordinate variables when the
    grid file provides them (e.g. torus grids) and are otherwise derived on the
    unit sphere from the spherical ones (icosahedral grid files carry only those).
    """
    with nc.Dataset(grid_file, "r") as dataset:
        dataset.set_auto_maskandscale(False)
        if VERTEX_OF_CELL not in dataset.variables:
            raise ValueError(f"Grid file '{grid_file}' is missing variable '{VERTEX_OF_CELL}'.")
        vertex_of_cell = np.asarray(dataset.variables[VERTEX_OF_CELL][:], dtype=np.int64)
        if all(name in dataset.variables for name in VERTEX_COORDINATE_VARIABLES):
            vertex_coordinates = {
                name: np.asarray(dataset.variables[name][:], dtype=np.float64)
                for name in VERTEX_COORDINATE_VARIABLES
            }
        elif all(name in dataset.variables for name in SPHERICAL_VERTEX_COORDINATE_VARIABLES):
            vlon_name, vlat_name = SPHERICAL_VERTEX_COORDINATE_VARIABLES
            vertex_coordinates = _cartesian_from_spherical(
                np.asarray(dataset.variables[vlon_name][:], dtype=np.float64),
                np.asarray(dataset.variables[vlat_name][:], dtype=np.float64),
            )
        else:
            raise ValueError(
                f"Grid file '{grid_file}' provides neither Cartesian vertex coordinates "
                f"({', '.join(VERTEX_COORDINATE_VARIABLES)}) nor spherical ones "
                f"({', '.join(SPHERICAL_VERTEX_COORDINATE_VARIABLES)})."
            )
        grid_uuid = getattr(dataset, _GRID_UUID_ATTRIBUTE, None)

    if vertex_of_cell.ndim != 2 or vertex_of_cell.shape[0] != 3:
        raise ValueError(
            f"Variable '{VERTEX_OF_CELL}' in '{grid_file}' has shape "
            f"{vertex_of_cell.shape}: expected (3, num_cells)."
        )
    lengths = {coordinate.shape[0] for coordinate in vertex_coordinates.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Vertex coordinate variables in '{grid_file}' have inconsistent lengths: "
            f"{', '.join(str(coordinate.shape[0]) for coordinate in vertex_coordinates.values())}."
        )
    (num_vertices,) = lengths
    if vertex_of_cell.min() < 1 or vertex_of_cell.max() > num_vertices:
        raise ValueError(
            f"Variable '{VERTEX_OF_CELL}' must hold 1-based vertex indices in "
            f"1..{num_vertices}, got {vertex_of_cell.min()}..{vertex_of_cell.max()}."
        )
    return GridGeometry(
        # int32: gridlook does plain Number arithmetic on the indices; an int64
        # array reaches it as BigInt64Array and throws
        vertex_of_cell=vertex_of_cell.astype(np.int32),
        vertex_coordinates=vertex_coordinates,
        grid_uuid=None if grid_uuid is None else str(grid_uuid),
    )


def _normalized_uuid(value: str) -> str:
    """Canonical form of a UUID attribute, or the raw string if unparsable."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return value


def _json_safe(value: Any) -> Any:
    """Coerce a netCDF attribute value to a JSON-serializable one for zarr attrs.

    ICON4Py's own output carries only string attributes, but a foreign netCDF source
    may attach numeric ones, which ``netCDF4`` returns as numpy scalars or arrays and
    ``zarr`` then refuses to serialize. numpy scalars become Python scalars, numpy
    arrays become (nested) lists, and byte strings are decoded; anything already
    JSON-safe is passed through unchanged.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _netcdf_attributes(source: nc.Variable | nc.Dataset) -> dict[str, Any]:
    """JSON-safe copy of a netCDF variable's or dataset's attributes."""
    # NOTE: CF packing attributes (scale_factor, add_offset, _FillValue) are copied
    # verbatim, not decoded; ingesting packed foreign netCDF is out of scope (driver
    # output is never packed), so they are left for a future reader to interpret.
    return {name: _json_safe(source.getncattr(name)) for name in source.ncattrs()}


@dataclasses.dataclass(frozen=True)
class _NetcdfArray:
    """Read adapter presenting a netCDF variable with the surface the exporter reads."""

    variable: nc.Variable

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.variable.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.variable.dtype

    @property
    def attrs(self) -> dict[str, Any]:
        return _netcdf_attributes(self.variable)

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return tuple(self.variable.dimensions)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        return np.asarray(self.variable[index])


@dataclasses.dataclass(frozen=True)
class _NetcdfGroup:
    """Read adapter presenting a netCDF dataset with the surface the exporter reads."""

    dataset: nc.Dataset

    @property
    def attrs(self) -> dict[str, Any]:
        return _netcdf_attributes(self.dataset)

    def __contains__(self, name: str) -> bool:
        return name in self.dataset.variables

    def __getitem__(self, name: str) -> _NetcdfArray:
        return _NetcdfArray(self.dataset.variables[name])

    def arrays(self) -> Iterator[tuple[str, _NetcdfArray]]:
        for name, variable in self.dataset.variables.items():
            yield name, _NetcdfArray(variable)


#: Read side of an export: a zarr store group or an adapted netCDF dataset.
_SourceGroup: TypeAlias = zarr.Group | _NetcdfGroup  # noqa: UP040
_SourceArray: TypeAlias = zarr.Array | _NetcdfArray  # noqa: UP040


@contextlib.contextmanager
def _open_source(source: pathlib.Path) -> Iterator[_SourceGroup]:
    """Open a driver output source: a zarr store (a directory) or a netCDF file."""
    if source.is_dir():
        yield zarr.open_group(str(source), mode="r")
        return
    dataset = nc.Dataset(source, "r")
    try:
        # raw values: the exporter copies stored values verbatim, like the zarr path
        dataset.set_auto_maskandscale(False)
        yield _NetcdfGroup(dataset)
    finally:
        dataset.close()


def _data_array(group: _SourceGroup, name: str) -> _SourceArray:
    """Typed access to a store array (``zarr.Group.__getitem__`` may also yield groups)."""
    array = group[name]
    assert isinstance(array, (zarr.Array, _NetcdfArray))
    return array


def _dimension_names(array: _SourceArray) -> tuple[str, ...] | None:
    if isinstance(array, _NetcdfArray):
        return array.dimension_names
    names = getattr(array.metadata, "dimension_names", None)
    return None if names is None else tuple(names)


def _global_cell_positions(group: _SourceGroup) -> np.ndarray | None:
    """Store position of every global cell, or None for a store in global order."""
    if _CELL_GLOBAL_INDEX not in group:
        return None
    global_index = np.asarray(_data_array(group, _CELL_GLOBAL_INDEX)[:])
    owned_positions = np.nonzero(global_index >= 0)[0]
    owned = global_index[owned_positions]
    global_size = int(owned.shape[0])
    is_permutation = (
        global_size > 0
        and int(owned.max()) < global_size
        and bool(np.all(np.bincount(owned, minlength=global_size) == 1))
    )
    if not is_permutation:
        raise ValueError(
            f"Array '{_CELL_GLOBAL_INDEX}' does not enumerate the global grid: "
            f"expected each index in 0..{global_size - 1} exactly once."
        )
    positions = np.empty(global_size, dtype=np.int64)
    positions[owned] = owned_positions
    return positions


def _partition_data_variables(group: _SourceGroup) -> tuple[dict[str, _SourceArray], list[str]]:
    """Split time-dependent variables into cell fields and the rest."""
    cell_variables: dict[str, _SourceArray] = {}
    skipped: list[str] = []
    arrays: Iterator[tuple[str, _SourceArray]] = group.arrays()
    for name, array in sorted(arrays, key=lambda entry: entry[0]):
        dimensions = _dimension_names(array)
        if dimensions is None or len(dimensions) < 2 or dimensions[0] != writers.TIME:
            continue
        if dimensions[-1] == writers.CELL:
            cell_variables[name] = array
        else:
            skipped.append(name)
    return cell_variables, skipped


def _copy_coordinate(source: _SourceGroup, destination: zarr.Group, name: str) -> None:
    if name not in source:
        return
    array = _data_array(source, name)
    copy = destination.create_array(
        name,
        shape=array.shape,
        chunks=array.shape if array.shape[0] > 0 else (1,),
        dtype=array.dtype,
        dimension_names=[name],
    )
    copy.attrs.update(dict(array.attrs))
    if array.shape[0] > 0:
        copy[:] = array[:]


def _copy_cell_variable(
    array: _SourceArray,
    destination: zarr.Group,
    name: str,
    positions: np.ndarray | None,
    num_cells: int,
) -> None:
    """Copy one (time, ..., cell) field, reordering the cell axis to global order."""
    num_times = array.shape[0]
    middle_shape = array.shape[1:-1]
    copy = destination.create_array(
        name,
        shape=(num_times, *middle_shape, num_cells),
        # one chunk per displayed frame: gridlook fetches whole chunks for a
        # point selection of every dimension but the last
        chunks=(1, *(1,) * len(middle_shape), num_cells),
        dtype=array.dtype,
        fill_value=float("nan") if np.issubdtype(array.dtype, np.floating) else None,
        dimension_names=_dimension_names(array),
    )
    copy.attrs.update(dict(array.attrs))
    scratch = np.empty((*middle_shape, num_cells), dtype=array.dtype)
    for time_index in range(num_times):
        slab = array[time_index]
        if positions is None:
            copy[time_index] = slab
        else:
            np.take(slab, positions, axis=-1, out=scratch)
            copy[time_index] = scratch


def export_store(
    *, source: pathlib.Path, grid_file: pathlib.Path, output: pathlib.Path
) -> tuple[list[str], list[str]]:
    """Write the gridlook store for a driver output.

    Args:
        source: driver output: a zarr store (serial/gathered or rank-block) or a
            netCDF file (always in global cell order).
        grid_file: ICON grid file of the run the output was written by.
        output: path of the store to create; must not exist yet.

    Returns:
        Names of the exported cell variables and of the skipped variables.
    """
    if output.exists():
        raise ValueError(f"Output store '{output}' already exists: refusing to overwrite.")
    geometry = read_grid_geometry(grid_file)
    with _open_source(source) as source_group:
        return _write_gridlook_store(
            source_group, geometry, source=source, grid_file=grid_file, output=output
        )


def _write_gridlook_store(
    source_group: _SourceGroup,
    geometry: GridGeometry,
    *,
    source: pathlib.Path,
    grid_file: pathlib.Path,
    output: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Validate the opened source and write the gridlook store from it."""
    store_uuid = source_group.attrs.get(_GRID_UUID_ATTRIBUTE)
    if (
        store_uuid is not None
        and geometry.grid_uuid is not None
        and _normalized_uuid(str(store_uuid)) != _normalized_uuid(geometry.grid_uuid)
    ):
        raise ValueError(
            f"Grid file '{grid_file}' does not match the source: '{_GRID_UUID_ATTRIBUTE}' is "
            f"'{geometry.grid_uuid}' in the grid file but '{store_uuid}' in the source."
        )

    positions = _global_cell_positions(source_group)
    if positions is not None and positions.shape[0] != geometry.num_cells:
        raise ValueError(
            f"Grid file '{grid_file}' has {geometry.num_cells} cells, but the source "
            f"'{source}' holds fields on {positions.shape[0]} global cells."
        )
    store_cell_length = (
        _data_array(source_group, _CELL_GLOBAL_INDEX).shape[0]
        if positions is not None
        else geometry.num_cells
    )

    cell_variables, skipped = _partition_data_variables(source_group)
    if not cell_variables:
        raise ValueError(f"Source '{source}' contains no cell variables to export.")
    # validate every variable before the output store is created: a failure
    # below would leave a partial store that blocks reruns
    for name, array in cell_variables.items():
        if array.shape[-1] == store_cell_length:
            continue
        if positions is None:
            raise ValueError(
                f"Variable '{name}' has {array.shape[-1]} entries on the cell axis, "
                f"but the grid file '{grid_file}' has {geometry.num_cells} cells."
            )
        raise ValueError(
            f"Variable '{name}' has {array.shape[-1]} entries on the cell axis, "
            f"but the source's cell axis has {store_cell_length}."
        )

    destination = zarr.open_group(str(output), mode="w-", zarr_format=3)
    try:
        attributes = dict(source_group.attrs)
        attributes.setdefault("title", output.stem)
        destination.attrs.update(attributes)

        coordinates = [writers.TIME]
        for array in cell_variables.values():
            dimensions = _dimension_names(array)
            assert dimensions is not None
            coordinates.extend(name for name in dimensions[1:-1] if name not in coordinates)
        for name in coordinates:
            _copy_coordinate(source_group, destination, name)

        for name, array in cell_variables.items():
            _copy_cell_variable(array, destination, name, positions, geometry.num_cells)

        # the geometry arrays deliberately carry no dimension names: gridlook
        # hides arrays without them from its variable selector
        vertex_of_cell = destination.create_array(
            VERTEX_OF_CELL,
            shape=geometry.vertex_of_cell.shape,
            chunks=geometry.vertex_of_cell.shape,
            dtype=geometry.vertex_of_cell.dtype,
        )
        vertex_of_cell[:] = geometry.vertex_of_cell
        for name, values in geometry.vertex_coordinates.items():
            coordinate = destination.create_array(
                name, shape=values.shape, chunks=values.shape, dtype=values.dtype
            )
            coordinate[:] = values

        with warnings.catch_warnings():
            # consolidated metadata is mandatory for gridlook (it does not fall
            # back to listing the store) but is a zarr-python extension of the
            # format 3 specification; silence the spec-stability warning it emits
            warnings.filterwarnings("ignore", message="Consolidated metadata", category=UserWarning)
            zarr.consolidate_metadata(destination.store)
    except BaseException:
        # mode "w-" above guarantees this call created the store, so removing
        # it cannot touch pre-existing data
        shutil.rmtree(output, ignore_errors=True)
        raise
    return list(cell_variables), skipped


class _CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static-file handler allowing cross-origin reads (gridlook runs in the browser)."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def _serve_store(*, store: pathlib.Path, host: str, port: int) -> None:
    # the store itself is the document root: everything the viewer needs is
    # inside it, and the CORS header below makes anything served here readable
    # by every website in the user's browser
    directory = store.resolve()
    handler = functools.partial(_CORSHTTPRequestHandler, directory=str(directory))
    try:
        server = http.server.ThreadingHTTPServer((host, port), handler)
    except OSError as error:
        typer.echo(f"Cannot serve on '{host}:{port}': {error.strerror or error}.", err=True)
        raise typer.Exit(code=1) from error
    with server:
        # a wildcard bind address is not a reachable client-side URL
        url_host = "127.0.0.1" if host == "0.0.0.0" else host
        typer.echo(f"Serving '{directory}' at http://{url_host}:{port} (Ctrl+C to stop).")
        typer.echo(f"View the store at: {GRIDLOOK_APP_URL}/#http://{url_host}:{port}")
        if url_host in ("127.0.0.1", "localhost"):
            typer.echo(
                "Note: if the viewer cannot load the data, your browser is blocking the "
                "HTTPS viewer from fetching this plain-HTTP store. Use a Chromium-based "
                "(Chrome, Brave, Edge) or Firefox browser, which allow it for loopback; "
                "Safari blocks it with no override."
            )
        else:
            typer.echo(
                "Note: browsers block the HTTPS viewer from fetching plain-HTTP data on "
                "non-loopback hosts; use a locally running gridlook for this address."
            )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            typer.echo("Stopped.")


@app.command()
def export(
    *,
    store_path: Annotated[
        pathlib.Path,
        typer.Option(exists=True, help="Driver output: zarr store or netCDF file."),
    ],
    grid_file_path: Annotated[
        pathlib.Path,
        typer.Option(exists=True, dir_okay=False, help="ICON grid file (netCDF) of the run."),
    ],
    output_path: Annotated[
        pathlib.Path | None,
        typer.Option(help="Store to create. Defaults to '<store>_gridlook.zarr'."),
    ] = None,
    serve: Annotated[
        bool,
        typer.Option("--serve/--no-serve", help="Serve the exported store when done."),
    ] = False,
    host: Annotated[str, typer.Option(help="Interface to serve on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8000,
) -> None:
    """Convert driver output (a zarr store or a netCDF file) into a store gridlook can display."""
    output_store = (
        output_path
        if output_path is not None
        else store_path.with_name(f"{store_path.stem}_gridlook.zarr")
    )
    exported, skipped = export_store(
        source=store_path, grid_file=grid_file_path, output=output_store
    )
    for name in skipped:
        typer.echo(f"Skipped '{name}': gridlook renders cell fields only.")
    typer.echo(f"Exported to '{output_store}': {', '.join(exported)}.")
    if serve:
        _serve_store(store=output_store, host=host, port=port)
    else:
        typer.echo(f"Serve it with: icon4py-gridlook serve --store-path {output_store}")


@app.command()
def serve(
    *,
    store_path: Annotated[
        pathlib.Path,
        typer.Option(exists=True, file_okay=False, help="Exported gridlook store to host."),
    ],
    host: Annotated[str, typer.Option(help="Interface to serve on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8000,
) -> None:
    """Host an exported store over HTTP with CORS enabled for gridlook."""
    _serve_store(store=store_path, host=host, port=port)


if __name__ == "__main__":
    sys.exit(app())
