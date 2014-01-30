"""
.. note::
  This module is experimental. It's API may change between versions.

This module implements function for DTCWT-based image registration as outlined in
`[1] <http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5936113>`_.
These functions are 2D-only for the moment.

The functions in the top-level :py:mod:`dtcwt.registration` module are imported
as a convenience from :py:mod:`dtcwt.numpybackend`. You could also import
:py:mod:`dtcwt.numpybackend` directly to explicitly select backend.

"""

from .numpybackend import *

__all__ = [
    'estimatereg',
    'velocityfield',
    'warp',
    'warptransform',
]
