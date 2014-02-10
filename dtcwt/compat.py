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
from __future__ import absolute_import

from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.numpy import Transform1d, Transform2d, Transform3d, Pyramid

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

def dtwavexfm(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False):
    """Perform a *n*-level DTCWT decompostion on a 1D column vector *X* (or on
    the columns of a matrix *X*).

    :param X: 1D real array or 2D real array whose columns are to be transformed
    :param nlevels: Number of levels of wavelet decomposition
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.

    :returns Yl: The real lowpass image from the final level
    :returns Yh: A tuple containing the (N, M, 6) shape complex highpass subimages for each level.
    :returns Yscale: If *include_scale* is True, a tuple containing real lowpass coefficients for every scale.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    Example::

        # Performs a 5-level transform on the real image X using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Yl, Yh = dtwavexfm(X,5,'near_sym_b','qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
    .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

    """
    trans = Transform1d(biort, qshift)
    res = trans.forward(X, nlevels, include_scale)

    if include_scale:
        return res.lowpass, res.highpasses, res.scales
    else:
        return res.lowpass, res.highpasses

def dtwaveifm(Yl, Yh, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, gain_mask=None):
    """Perform an *n*-level dual-tree complex wavelet (DTCWT) 1D
    reconstruction.

    :param Yl: The real lowpass subband from the final level
    :param Yh: A sequence containing the complex highpass subband for each level.
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.
    :param gain_mask: Gain to be applied to each subband.

    :returns Z: Reconstructed real array.

    The *l*-th element of *gain_mask* is gain for wavelet subband at level l.
    If gain_mask[l] == 0, no computation is performed for band *l*. Default
    *gain_mask* is all ones. Note that *l* is 0-indexed.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    Example::

        # Performs a reconstruction from Yl,Yh using the 13,19-tap filters
        # for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Z = dtwaveifm(Yl, Yh, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
    .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

    """
    trans = Transform1d(biort, qshift)
    res = trans.inverse(Pyramid(Yl, Yh), gain_mask=gain_mask)
    return res

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

    trans = Transform2d(biort, qshift)
    res = trans.forward(X, nlevels, include_scale)

    if include_scale:
        return res.lowpass, res.highpasses, res.scales
    else:
        return res.lowpass, res.highpasses

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
    trans = Transform2d(biort, qshift)
    res = trans.inverse(Pyramid(Yl, Yh), gain_mask=gain_mask)
    return res

# BACKWARDS COMPATIBILITY: add a dtwave{i,x}fm2b function which is a copy of
# dtwave{i,x}fm2b. The functionality of the ...b variant is rolled into the
# original.
dtwavexfm2b = dtwavexfm2
dtwaveifm2b = dtwaveifm2

def dtwavexfm3(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT,
               include_scale=False, ext_mode=4, discard_level_1=False):
    """Perform a *n*-level DTCWT-3D decompostion on a 3D matrix *X*.

    :param X: 3D real array-like object
    :param nlevels: Number of levels of wavelet decomposition
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.
    :param ext_mode: Extension mode. See below.
    :param discard_level_1: True if level 1 high-pass bands are to be discarded.

    :returns Yl: The real lowpass image from the final level
    :returns Yh: A tuple containing the complex highpass subimages for each level.

    Each element of *Yh* is a 4D complex array with the 4th dimension having
    size 28. The 3D slice ``Yh[l][:,:,:,d]`` corresponds to the complex higpass
    coefficients for direction d at level l where d and l are both 0-indexed.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    There are two values for *ext_mode*, either 4 or 8. If *ext_mode* = 4,
    check whether 1st level is divisible by 2 (if not we raise a
    ``ValueError``). Also check whether from 2nd level onwards, the coefs can
    be divided by 4. If any dimension size is not a multiple of 4, append extra
    coefs by repeating the edges. If *ext_mode* = 8, check whether 1st level is
    divisible by 4 (if not we raise a ``ValueError``). Also check whether from
    2nd level onwards, the coeffs can be divided by 8. If any dimension size is
    not a multiple of 8, append extra coeffs by repeating the edges twice.

    If *discard_level_1* is True the highpass coefficients at level 1 will be
    discarded. (And, in fact, will never be calculated.) This turns the
    transform from being 8:1 redundant to being 1:1 redundant at the cost of
    no-longer allowing perfect reconstruction. If this option is selected then
    `Yh[0]` will be `None`. Note that :py:func:`dtwaveifm3` will accepts
    `Yh[0]` being `None` and will treat it as being zero.

    Example::

        # Performs a 3-level transform on the real 3D array X using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Yl, Yh = dtwavexfm3(X, 3, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Huizhong Chen, Jan 2009
    .. codeauthor:: Nick Kingsbury, Cambridge University, July 1999.

    """
    trans = Transform3d(biort, qshift, ext_mode)
    res = trans.forward(X, nlevels, include_scale, discard_level_1)

    if include_scale:
        return res.lowpass, res.highpasses, res.scales
    else:
        return res.lowpass, res.highpasses

def dtwaveifm3(Yl, Yh, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, ext_mode=4):
    """Perform an *n*-level dual-tree complex wavelet (DTCWT) 3D
    reconstruction.

    :param Yl: The real lowpass subband from the final level
    :param Yh: A sequence containing the complex highpass subband for each level.
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.
    :param ext_mode: Extension mode. See below.

    :returns Z: Reconstructed real image matrix.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    There are two values for *ext_mode*, either 4 or 8. If *ext_mode* = 4,
    check whether 1st level is divisible by 2 (if not we raise a
    ``ValueError``). Also check whether from 2nd level onwards, the coefs can
    be divided by 4. If any dimension size is not a multiple of 4, append extra
    coefs by repeating the edges. If *ext_mode* = 8, check whether 1st level is
    divisible by 4 (if not we raise a ``ValueError``). Also check whether from
    2nd level onwards, the coeffs can be divided by 8. If any dimension size is
    not a multiple of 8, append extra coeffs by repeating the edges twice.

    Example::

        # Performs a 3-level reconstruction from Yl,Yh using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Z = dtwaveifm3(Yl, Yh, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Huizhong Chen, Jan 2009
    .. codeauthor:: Nick Kingsbury, Cambridge University, July 1999.

    """
    trans = Transform3d(biort, qshift, ext_mode)
    res = trans.inverse(Pyramid(Yl, Yh))
    return res
