from pathlib import Path

from icon4py.bindings.utils import write_string
from icon4py.diffusion.wrapper.binding import generate_c_header, \
    generate_f90_interface
from icon4py.diffusion.wrapper.decorators import compile_cffi_plugin
from icon4py.diffusion.wrapper.parsing import parse_functions_from_module

def main():
    module_name = "icon4py.diffusion.wrapper.diffusion_wrapper"
    python_src_file = "diffusion_wrapper.py"
    build_path = Path("./build")
    build_path.mkdir(exist_ok=True, parents=True)
    plugin = parse_functions_from_module(module_name, ["diffusion_init", "diffusion_run"])
    c_header = generate_c_header(plugin)
    f90_interface= generate_f90_interface(plugin)
    write_string(f90_interface, build_path, f"{plugin.name}.f90")

    compile_cffi_plugin(plugin.name, c_header, python_src_file , str(build_path))


if __name__ == '__main__':
    main()

