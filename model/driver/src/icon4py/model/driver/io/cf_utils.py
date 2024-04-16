from typing import Final


LEVEL_NAME:Final[str] = 'model_level_number'
INTERFACE_LEVEL_NAME:Final[str] = 'interface_model_level_number'
DEFAULT_CALENDAR :Final[str]= "proleptic_gregorian"
DEFAULT_TIME_UNIT:Final[str] = 'seconds since 1970-01-01 00:00:00'

"""
CF conventions encourage to use the COARDS conventions for the order of the dimensions: `T` (time), `Z` (height or depth), `Y` (latitude), `X` (longitude).
In the unstructured case `Y` and `X`  combine to the horizontal dimension.
""" 
COARDS_T_POS:Final[int] = 0
COARDS_Z_POS:Final[int] = 1
HORIZONTAL_POS:Final[int] = 2


