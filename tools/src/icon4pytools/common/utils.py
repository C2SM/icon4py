import pathlib
import shutil
import subprocess
import sys

PYTHON_PATH = sys.executable


def write_string(string: str, outdir: pathlib.Path, fname: str) -> None:
    path = outdir / fname
    with open(path, "w") as f:
        f.write(string)


def format_fortran_code(source: str) -> str:
    """Format fortran code using fprettify.

    Try to find fprettify in PATH -> found by which
    otherwise look in PYTHONPATH
    """
    fprettify_path = shutil.which("fprettify")

    if fprettify_path is None:
        bin_path = pathlib.Path(PYTHON_PATH).parent
        fprettify_path = str(bin_path / "fprettify")
    args = [str(fprettify_path)]
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return p1.communicate(source.encode("UTF-8"))[0].decode("UTF-8").rstrip()
