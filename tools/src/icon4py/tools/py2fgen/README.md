# Py2fgen

A lighweight Fortran bindings generator for Python functions utilizing CFFI.

## Getting started

- Decorate your Python function with `py2fgen.export()`.
- Provide information about the function's parameters, in one of the following ways (see TODO for details)
  - full signature specification with TODO `_template.Func` class
  - provide a `ParameterDescriptor` via `Annotated` type hint for each parameter
  - add a hook that translates the type annotation to a `ParameterDescriptor`
- Optional: provide a hook on how to convert the raw arguments to custom types.
- Finally, run the py2fgen command line tool to generate:
  - the Fortran module
  - a compiled shared library containing the wrapper code around your function

## Documentation

### Parameter descriptors

TODO

### Raw argument conversion hooks

By default, array arguments are translated to Numpy or CuPy arrays. Alternatively, you can provide a hook that provides a translation function from an `ArrayInfo` to your custom type. Note, this translation function is executed on each call and should be as efficient as possible, e.g. by caching the result.

### Optimized Python

For production, make sure you set the environment variable `PYTHONOPTIMIZE` to `2` to enable Python optimizations, see https://docs.python.org/3/using/cmdline.html#envvar-PYTHONOPTIMIZE.

### Debugging the bindings

For debugging, you can set the environment variable `PY2FGEN_LOGGING` to TODO to print debug information.

Additionally, you can enable profiling by setting the environment variable `PY2FGEN_PROFILE` to `1`.

Note, that debugging and profiling is not available if Python is set to optimized mode.

### Known problems

- On the Fortran side we use standard 4-byte logicals to represent Python booleans. Currently, we do not create views of the boolean arrays, but instead copy the data to 1-byte boolean arrays on the Python side. Therefore, these arrays are read-only.
