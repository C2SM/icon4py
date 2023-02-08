# icon4py py2f90

## Description

Python utilities for generating a C library and Fortran interface to call Python icon4py modules. The library [embeds python via CFFI ](https://cffi.readthedocs.io/en/latest/embedding.html)

### py2f90gen

Generates a C header file and a Fortran interface and compiles python functions into a C library embedding python.

The functions need to be decorated with `CffiMethod.register` and have a signature with scalar arguments of gt4py fields:

```
@CffiMethod.register
def foo(i:int, param:float, field1: Field[[VertexDim, KDim], float], field2: Field[CellDim, KDim], float])
```

## Installation instructions

Check `README.md` file in the root of the repository.
