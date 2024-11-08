[![Open in Gitpod](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/C2SM/icon4py)

# ICON4Py

ICON4Py hosts Python implementations of various components from the ICON climate and weather model. Additionally, it includes icon4pytools, a collection of command-line interfaces (CLIs), and utilities required for the integration of ICON4Py code into the ICON Fortran model. ICON4Py leverages [GT4Py](https://github.com/GridTools/gt4py) to ensure efficient and performance portable implementations of these components.

## Project Structure

The repository is organized into directories, each containing independent Python namespace packages for different ICON components or utility packages. These packages can be installed independently. Since these packages are not available from a package repository (yet), you need to specify the location of dependencies within this repository. This can be done by installing the required dependencies first. Refer to the [Installation instructions](#installation-instructions) below.

## License

ICON4Py is licensed under the terms of the BSD-3-Clause.

## Installation instructions
### Dependencies
A minimal installation of ICON4Py needs 
- Python 3.10
- boost >= 1.85.0

You can install all packages at once by using the provided `requirements.txt` or `requirements-dev.txt` files in the root of the repository. For example:
The `-dev.txt` file installs ICON4Py packages and GT4Py in editable mode, such that source changes are immediatly picked up and used in the virtual environment. 
```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Create a (Python 3.10) virtual environment (usually at `.venv`)
python3.10 -m venv .venv

# Activate the virtual environment and make sure that 'wheel' is installed
source .venv/bin/activate
pip install --upgrade wheel pip setuptools

# Install all the ICON4Py packages and its dependencies
# External dependencies would be checked out at './_external_src'
pip install --src _external_src -r requirements-dev.txt

# Finally, check that everything works
pytest -v
```

The `--src _external_src` option tells `pip` to use a specific folder as the base path for checked out sources, which is very convenient for development tasks involving changes in external dependencies like `gt4py`. For convenience, `./_external_src` has been already added to the repository `.gitignore`.

You can also use [tox](https://tox.wiki/en/latest/) for the automatic installation of all packages in development mode in a single step:


```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Use tox to create and set up a development environment (usually at `.venv`) in verbose mode
pip install tox
python -m tox -vv -e dev --devenv .venv

# Activate the virtual environment and check that everything works
source .venv/bin/activate
pytest -v
```



### Installation of specific subpackages

In case you only want to install a specific subpackage, use the actual subpackage `requirements.txt` or `requirements-dev.txt` files.

For example:

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Create a (Python 3.10) virtual environment (usually at `.venv`)
python3.10 -m venv .venv

# Activate the virtual environment and make sure that 'wheel' is installed
source .venv/bin/activate
pip install --upgrade wheel

# Install a specific ICON4Py subpackage and its dependencies
cd _SUBPACKAGE_  # where _SUBPACKAGE_ in model/atmosphere/dycore | tools | ...
pip install -r requirements-dev.txt

# or in the case of there being a pyproject.toml file
pip install .
```

## Development instructions

After following the installation instructions above using the development requirements (`*-dev.txt` files), an _editable_ installation of all the packages will be active in the virtual environment. In this mode, code changes are immediately visible since source files are imported directly by the Python interpreter.

### Code quality checks

[pre-commit](https://pre-commit.com/) is used to run several linting and checking tools. It should always be executed locally before opening a pull request. When executing pre-commit locally you can either run it for the `model` or `tools` folder:

For example to run code checks on all components in `icon4py.model` you can do:

```bash
# running precommit for all components in model
cd model/
pre-commit run --all-files
```

### Testing

[pytest](https://pytest.org/) is used for testing and it also comes with a command tool to easily run tests:

```bash
# Run all tests (verbose mode)
pytest -v

# Run only tests in `path/to/test/folder`
pytest -v path/to/test/folder
```

Nonetheless, we also recommended to use `tox` to run the complete test suite:

```bash
# Run test suite in the default environment
tox

# Run test suite in a specific environment (use `tox -a` to see list of envs)
tox -e py310
```

The default `tox` environment is configured to generate HTML test coverage reports in `_reports/coverage_html/`.

### Benchmarking

We use [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/) to benchmark the execution time of stencils in icon4py. To disable benchmarking during testing you can use `--benchmark-disable` when invoking `pytest`.

### Documentation

The documentation is at a very early stage given the constant state of development. Some effort is ongoing to document the dycore and can be compiled as follows.

You can install the required packages by using the provided `requirements-dev.txt` file in the root of the repository.

Then move to the dycore docs folder and build the html documentation with the provided makefile:

```bash
cd model/atmosphere/dycore/docs
make html
```

The documentation can then be accessed at `docs/_build/html/index.html`

### More Information

For more information please consult the package specific READMEs found in the `model` and `tools` folders.

TODO: @halungge dummy change to be removed again