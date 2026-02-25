from . import core

# Expose core functions at the top level
from .core import (
    catPeaks,
    cullLines,
    keepLines,
    lineID,
    findLines,
    findSpecies,
    readLineID,
    plotID,
    viewFit,
    combList,
    getLines,
)

# Package version
__version__ = "0.4.0"

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
    "viewFit",
    "combList",
    "getLines",
    "__version__",
]
