# flake8: noqa D104
import numpy as np
from gt4py.next.ffront.fbuiltins import float64, int32
from icon4pytools.py2f.wrappers.square_functions import square_output_param


# TODO [Magdalena]: generalize this and provide it as decorator to def_extern functions?
# TODO [Magdalena]: generalize for arbitrary number of input fields
def unpack(ptr, size_x, size_y) -> np.ndarray:
    """
    unpacks a 2d c/fortran field into a numpy array.

    :param ptr: c_pointer to the field
    :param size_x: col size (since its called from fortran)
    :param size_y: row size
    :return: a numpy array with shape=(size_y, size_x)
    and dtype = ctype of the pointer
    """
    # for now only 2d, invert for row/column precedence...
    shape = (size_y, size_x)
    length = np.prod(shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ar = np.frombuffer(
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=np.dtype(c_type),
        count=-1,
        offset=0,
    ).reshape(shape)
    return ar


def pack(ptr, arr: np.ndarray):
    """
    memcopies a numpy array into a pointer.

    :param ptr: c pointer
    :param arr: numpy array
    :return:
    """
    # for now only 2d
    length = np.prod(arr.shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ffi.memmove(ptr, np.ravel(arr), length * ffi.sizeof(c_type))


def square_wrapper(field_ptr: float64, nx: int32, ny: int32, result_ptr: float64):
    """
    simple python function that squares all entries of a field of
    size nx x ny and returns a pointer to the result.

    :param field_ptr:
    :param nx:
    :param ny:
    :param result_ptr:
    :return:
    """
    a = unpack(field_ptr, nx, ny)
    res = unpack(result_ptr, nx, ny)
    square_output_param(a, res)
