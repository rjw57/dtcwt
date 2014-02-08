from __future__ import absolute_import

from dtcwt.utils import asfarray

class Pyramid(object):
    """A representation of a transform domain signal.

    Backends are free to implement any class which respects this interface for
    storing transform-domain signals. The inverse transform may accept a
    backend-specific version of this class but should always accept any class
    which corresponds to this interface.

    .. py:attribute:: lowpass

        A NumPy-compatible array containing the coarsest scale lowpass signal.

    .. py:attribute:: highpasses

        A tuple where each element is the complex subband coefficients for
        corresponding scales finest to coarsest.

    .. py:attribute:: scales

        *(optional)* A tuple where each element is a NumPy-compatible array
        containing the lowpass signal for corresponding scales finest to
        coarsest. This is not required for the inverse and may be *None*.

    """
    def __init__(self, lowpass, highpasses, scales=None):
        self.lowpass = asfarray(lowpass)
        self.highpasses = tuple(asfarray(x) if x is not None else None for x in highpasses)
        self.scales = tuple(asfarray(x) for x in scales) if scales is not None else None

