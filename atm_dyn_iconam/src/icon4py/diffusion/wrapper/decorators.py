def cffi_plugin(plugin_name):
    from plugin_name import ffi
    def cffi_plugin_decorator(func):
        def plugin_wrapper(*args, **kwargs):
            return ffi.def_extern(func(*args, **kwargs))

        return plugin_wrapper
