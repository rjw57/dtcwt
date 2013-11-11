import numpy as np
import logging

from six.moves import xrange

from dtcwt import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.lowlevel import colfilter, coldfilt, colifilt
from dtcwt.utils import appropriate_complex_type_for, asfarray

from dtcwt.backend import TransformDomainSignal
from dtcwt.backend.backend_numpy.transform2d import Transform2dNumPy

def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False):
    """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

    :param X: 2D real array
    :param nlevels: Number of levels of wavelet decomposition
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.

    :returns Yl: The real lowpass image from the final level
    :returns Yh: A tuple containing the complex highpass subimages for each level.
    :returns Yscale: If *include_scale* is True, a tuple containing real lowpass coefficients for every scale.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    Example::

        # Performs a 3-level transform on the real image X using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Yl, Yh = dtwavexfm2(X, 3, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
    .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001

    """

    trans = Transform2dNumPy(biort, qshift)
    res = trans.forward(X, nlevels, include_scale)

    if include_scale:
        return res.lowpass, res.subbands, res.scales
    else:
        return res.lowpass, res.subbands

def dtwaveifm2(Yl,Yh,biort=DEFAULT_BIORT,qshift=DEFAULT_QSHIFT,gain_mask=None):
    """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
    reconstruction.

    :param Yl: The real lowpass subband from the final level
    :param Yh: A sequence containing the complex highpass subband for each level.
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.
    :param gain_mask: Gain to be applied to each subband.

    :returns Z: Reconstructed real array

    The (*d*, *l*)-th element of *gain_mask* is gain for subband with direction
    *d* at level *l*. If gain_mask[d,l] == 0, no computation is performed for
    band (d,l). Default *gain_mask* is all ones. Note that both *d* and *l* are
    zero-indexed.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    Example::

        # Performs a 3-level reconstruction from Yl,Yh using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Z = dtwaveifm2(Yl, Yh, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
    .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

    """
    trans = Transform2dNumPy(biort, qshift)
    res = trans.inverse(TransformDomainSignal(Yl, Yh), gain_mask=gain_mask)
    return res.value

# BACKWARDS COMPATIBILITY: add a dtwave{i,x}fm2b function which is a copy of
# dtwave{i,x}fm2b. The functionality of the ...b variant is rolled into the
# original.
dtwavexfm2b = dtwavexfm2
dtwaveifm2b = dtwaveifm2
