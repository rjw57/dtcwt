from dtcwt.utils import asfarray
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT

class TransformDomainSignal(object):
    """A representation of a transform domain signal.

    Backends are free to implement any class which respects this interface for
    storing transform-domain signals. The inverse transform may accept a
    backend-specific version of this class but should always accept any class
    which corresponds to this interface.

    .. py:attribute:: lowpass
        
        A NumPy-compatible array containing the coarsest scale lowpass signal.

    .. py:attribute:: subbands
        
        A tuple where each element is the complex subband coefficients for
        corresponding scales finest to coarsest.

    .. py:attribute:: scales
        
        *(optional)* A tuple where each element is a NumPy-compatible array
        containing the lowpass signal for corresponding scales finest to
        coarsest. This is not required for the inverse and may be *None*.

    """
    def __init__(self, lowpass, subbands, scales=None):
        self.lowpass = asfarray(lowpass)
        self.subbands = tuple(asfarray(x) for x in subbands)
        self.scales = tuple(asfarray(x) for x in scales) if scales is not None else None

class ReconstructedSignal(object):
    """
    A representation of the reconstructed signal from the inverse transform. A
    backend is free to implement their own version of this class providing it
    corresponds to the interface documented.

    .. py:attribute:: value

        A NumPy-compatible array containing the reconstructed signal.

    """
    def __init__(self, value):
        self.value = asfarray(value)

class Transform2d(object):
    """
    An implementation of a 2D DT-CWT transformation. Backends must provide a
    transform class which provides an interface compatible with this base
    class.

    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    In some cases the tuples may have more elements. This is used to represent
    the :ref:`rot-symm-wavelets`.
    
    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        raise NotImplementedError()

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 2D real array
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.backend.TransformDomainSignal` compatible object representing the transform-domain signal

        """
        raise NotImplementedError()

    def inverse(self, td_signal, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
        reconstruction.

        :param td_signal: A :py:class:`dtcwt.backend.TransformDomainSignal`-like class holding the transform domain representation to invert.
        :param gain_mask: Gain to be applied to each subband.

        :returns: A :py:class:`dtcwt.backend.ReconstructedSignal` compatible instance with the reconstruction.

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with direction
        *d* at level *l*. If gain_mask[d,l] == 0, no computation is performed for
        band (d,l). Default *gain_mask* is all ones. Note that both *d* and *l* are
        zero-indexed.

        """
        raise NotImplementedError()

