"""Functions for compatibility with MATLAB scripts. These functions are
intentionally similar in name and behaviour to the original functions from the
DTCWT MATLAB toolbox. They are included in the library to ease the porting of
MATLAB scripts but shouldn't be used in new projects.

.. note::

    The functionality of ``dtwavexfm2b`` and ``dtwaveifm2b`` has been folded
    into ``dtwavexfm2`` and ``dtwaveifm2``. For convenience of porting MATLAB
    scripts, the original function names are available in the :py:mod:`dtcwt`
    module as aliases but they should not be used in new code.

"""
from dtcwt.transform1d import dtwavexfm, dtwaveifm
from dtcwt.transform2d import dtwavexfm2, dtwaveifm2, dtwavexfm2b, dtwaveifm2b
from dtcwt.transform3d import dtwavexfm3, dtwaveifm3

__all__ = [
    'dtwavexfm',
    'dtwaveifm',

    'dtwavexfm2',
    'dtwaveifm2',
    'dtwavexfm2b',
    'dtwaveifm2b',

    'dtwavexfm3',
    'dtwaveifm3',
]

