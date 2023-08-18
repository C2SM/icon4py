def builder(func):
    """Use as decorator on builder functions."""
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return wrapper
