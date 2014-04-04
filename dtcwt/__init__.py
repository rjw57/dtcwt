import os
import sys

__all__ = [
    '__version__',

    'Transform1d',
    'Transform2d',
    'Transform3d',
    'Pyramid',

    'backend_name',
    'push_backend',
    'pop_backend',
    'preserve_backend_stack',
]

from dtcwt._version import __version__

import dtcwt.numpy
import dtcwt.numpy.sampling
import dtcwt.opencl
import dtcwt.opencl.sampling

# An array of dictionaries. Each dictionary stores the top-level module
# variables for that backend.
_BACKEND_STACK = []

# Possible backends keyed by name
_AVAILABLE_BACKENDS = {
    'numpy': {
        'Transform1d': dtcwt.numpy.Transform1d,
        'Transform2d': dtcwt.numpy.Transform2d,
        'Transform3d': dtcwt.numpy.Transform3d,
        'Pyramid': dtcwt.numpy.Pyramid,
        '_sampling': dtcwt.numpy.sampling,
    },
    'opencl': {
        'Transform1d': dtcwt.numpy.Transform1d,
        'Transform2d': dtcwt.opencl.Transform2d,
        'Transform3d': dtcwt.numpy.Transform3d,
        'Pyramid': dtcwt.opencl.Pyramid,
        '_sampling': dtcwt.opencl.sampling,
    },
}

def _update_from_current_backend():
    # Update values from backend
    for k,v in _BACKEND_STACK[-1][1].items():
        setattr(dtcwt, k, v)
    dtcwt.backend_name = _BACKEND_STACK[-1][0]

class _BackendGuard(object):
    def __init__(self, stack):
        self._stack = list(stack)

    def __enter__(self):
        return _BACKEND_STACK

    def __exit__(self, exc_type, exc_value, exc_tb):
        dtcwt._BACKEND_STACK = self._stack
        # only re-raise if it's *not* the exception that was
        # passed to throw(), because __exit__() must not raise
        # an exception unless __exit__() itself failed.  But
        # throw() has to raise the exception to signal
        # propagation, so this fixes the impedance mismatch
        # between the throw() protocol and the __exit__()
        # protocol.
        #
        if sys.exc_info()[1] is not exc_value:
            raise

def preserve_backend_stack():
    """Return a generator object which can be used to preserve the backend
    stack even when an exception has been raise. For example:

    .. code-block:: python

        # current backend is NumPy
        assert dtcwt.backend_name == 'numpy'

        with dtcwt.preserve_backend_stack():
            dtcwt.push_backend('opencl')
            # ... things which may raise an exception

        # current backend is NumPy even if an exception was thrown
        assert dtcwt.backend_name == 'numpy'

    """
    return _BackendGuard(_BACKEND_STACK)

def push_backend(name):
    """Switch backend implementation to *name*. Push the previous backend onto
    the backend stack. The previous backend may be restored via
    :py:func:`dtcwt.pop_backend`.

    :param name: string identifying which backend to switch to
    :raises ValueError: if *name* does not correspond to a known backend

    *name* may take one of the following values:

    * ``numpy``: the default NumPy backend. See :py:mod:`dtcwt.numpy`.
    * ``opencl``: a backend which uses OpenCL where available. See
      :py:mod:`dtcwt.opencl`.

    """
    try:
        _BACKEND_STACK.append((name, _AVAILABLE_BACKENDS[name]))
    except KeyError:
        raise ValueError('No such backend: {0}'.format(name))
    _update_from_current_backend()

def pop_backend():
    """Restore the backend after a call to :py:func:`push_backend`. Calls to
    :py:func:`pop_backend` and :py:func:`pop_backend` may be nested. This
    function will undo the most recent call to :py:func:`push_backend`.

    :raise IndexError: if one attempts to pop more backends than one has pushed.

    """
    # It is an error to pop off the default backend
    if len(_BACKEND_STACK) <= 1:
        raise IndexError('Cannot pop default backend')

    _BACKEND_STACK.pop()
    _update_from_current_backend()

backend_name = None
"""A string providing a short human-readable name for the DTCWT backend
currently being used. This corresponds to the *name* parameter passed to
:py:func:`dtcwt.push_backend`. The *default* backend is ``numpy`` but can be
overridden by setting the DTCWT_BACKEND environment variable to a valid backend
name.
"""

# Default to the numpy backend unless DTCWT_BACKEND environment variable is
# set.
push_backend(os.getenv('DTCWT_BACKEND', 'numpy'))
