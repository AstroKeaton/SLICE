from . import core

# Expose core functions at the top level
from .core import *

# Package version
__version__ = "0.1.0"

__all__ = [
    "core",        # allows `slice.core.catPeaks`
    "catPeaks",
    "cullLines",
    "keepLines",
    "lineID",
    "findLines",
    "findSpecies",
    "readLineID",
    "plotID",
    "__version__",
]
