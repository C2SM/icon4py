# imports for generated wrapper code
import logging

import math
from square_plugin import ffi
import numpy as np

from numpy.typing import NDArray
from gt4py.next.iterator.embedded import np_as_located_field
from icon4pytools.py2fgen.settings import config

xp = config.array_ns
from icon4py.model.common import dimension as dims


# necessary imports when embedding a gt4py program directly
from gt4py.next import itir_python as run_roundtrip
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached, run_gtfn_gpu_cached
from icon4py.model.common.grid.simple import SimpleGrid

# We need a grid to pass offset providers to the embedded gt4py program (granules load their own grid at runtime)
grid = SimpleGrid()


# logger setup
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.ERROR, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")


import numpy as np

# embedded module imports
import cProfile
import pstats
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.decorator import program
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.simple import SimpleGrid
from icon4pytools.py2fgen.settings import backend


# embedded function imports
from icon4pytools.py2fgen.wrappers.simple import square


def unpack(ptr, *sizes: int) -> NDArray:
    """
    Converts a C pointer into a NumPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations requiring in-place modification of CPU data, enabling
    changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ptr (CData): A CFFI pointer to the beginning of the data array in CPU memory. This pointer
                     should reference a contiguous block of memory whose total size matches the product
                     of the specified dimensions.
        *sizes (int): Variable length argument list specifying the dimensions of the array.
                      These sizes determine the shape of the resulting NumPy array.

    Returns:
        np.ndarray: A NumPy array that provides a direct view of the data pointed to by `ptr`.
                    This array shares the underlying data with the original Fortran code, allowing
                    modifications made through the array to affect the original data.
    """
    length = math.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    # Map C data types to NumPy dtypes
    dtype_map: dict[str, np.dtype] = {
        "int": np.dtype(np.int32),
        "double": np.dtype(np.float64),
    }
    dtype = dtype_map.get(c_type, np.dtype(c_type))

    # Create a NumPy array from the buffer, specifying the Fortran order
    arr = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), dtype=dtype).reshape(  # type: ignore
        sizes, order="F"
    )
    return arr


def int_array_to_bool_array(int_array: NDArray) -> NDArray:
    """
    Converts a NumPy array of integers to a boolean array.
    In the input array, 0 represents False, and any non-zero value (1 or -1) represents True.

    Args:
        int_array: A NumPy array of integers.

    Returns:
        A NumPy array of booleans.
    """
    bool_array = int_array != 0
    return bool_array


@ffi.def_extern()
def square_wrapper(
    inp: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
    result: gtx.Field[[dims.CEDim, dims.KDim], gtx.float64],
    n_CE: gtx.int32,
    n_K: gtx.int32,
):
    try:

        # Unpack pointers into Ndarrays

        inp = unpack(inp, n_CE, n_K)

        result = unpack(result, n_CE, n_K)

        # Allocate GT4Py Fields

        inp = np_as_located_field(dims.CEDim, dims.KDim)(inp)

        result = np_as_located_field(dims.CEDim, dims.KDim)(result)

        square.with_backend(run_gtfn_cached)(inp, result, offset_provider=grid.offset_providers)

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0
