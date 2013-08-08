import numpy as np
import logging

from six.moves import xrange

from dtcwt import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.lowlevel import colfilter, coldfilt, colifilt

def dtwavexfm3(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, ext_mode=4):
    """Perform a *n*-level DTCWT-3D decompostion on a 3D matrix *X*.

    :param X: 3D real matrix/Image of shape (N, M)
    :param nlevels: Number of levels of wavelet decomposition
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.
    :param ext_mode: Extension mode. See below.

    :returns Yl: The real lowpass image from the final level
    :returns Yh: A tuple containing the (N, M, 7) shape complex highpass subimages for each level.

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

        # Performs a 3-level transform on the real image X using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Yl, Yh = dtwavexfm3(X, 3, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013

    """
    X = np.atleast_3d(X)

    # Try to load coefficients if biort is a string parameter
    try:
        h0o, g0o, h1o, g1o = _biort(biort)
    except TypeError:
        h0o, g0o, h1o, g1o = biort

    # Try to load coefficients if qshift is a string parameter
    try:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    except TypeError:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

    # Check value of ext_mode. TODO: this should really be an enum :S
    if ext_mode != 4 and ext_mode != 8:
        raise ValueError('ext_mode must be one of 4 or 8')

    Yl = X
    Yh = [None,] * nlevels

    # level is 0-indexed
    for level in xrange(nlevels):
        # Transform
        if level == 0:
            Yl, Yh[level] = _level1_xfm(Yl, h0o, h1o, ext_mode)

    return Yl, tuple(Yh)

def dtwaveifm3(Yl, Yh, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
    """Perform an *n*-level dual-tree complex wavelet (DTCWT) 3D
    reconstruction.

    :param Yl: The real lowpass subband from the final level
    :param Yh: A sequence containing the complex highpass subband for each level.
    :param biort: Level 1 wavelets to use. See :py:func:`biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.

    :returns Z: Reconstructed real image matrix.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
    interpreted as tuples of vectors giving filter coefficients. In the *biort*
    case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
    be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    Example::

        # Performs a 3-level reconstruction from Yl,Yh using the 13,19-tap
        # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
        Z = dtwaveifm3(Yl, Yh, 'near_sym_b', 'qshift_b')

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013

    """
    # Try to load coefficients if biort is a string parameter
    try:
        h0o, g0o, h1o, g1o = _biort(biort)
    except TypeError:
        h0o, g0o, h1o, g1o = biort

    # Try to load coefficients if qshift is a string parameter
    try:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    except TypeError:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

    X = Yl

    # level is 0-indexed
    nlevels = len(Yh)
    for level in xrange(nlevels):
        # Transform
        if level == 0:
            X = _level1_ifm(Yl, Yh[level], g0o, g1o)

    return X

def _level1_xfm(X, h0o, h1o, ext_mode):
    """Perform level 1 of the 3d transform.

    """
    # Check shape of input according to ext_mode. Note that shape of X is
    # double original input in each direction.
    if ext_mode == 4 and np.any(np.fmod(X.shape, 2) != 0):
        raise ValueError('Input shape should be a multiple of 2 in each direction when ext_mode == 4')
    elif ext_mode == 8 and np.any(np.fmod(X.shape, 4) != 0):
        raise ValueError('Input shape should be a multiple of 4 in each direction when ext_mode == 8')

    # Create work area
    work = np.zeros(np.asarray(X.shape) * 2, dtype=X.dtype)

    # Form some useful slices
    s0a = slice(None, work.shape[0] >> 1)
    s1a = slice(None, work.shape[1] >> 1)
    s2a = slice(None, work.shape[2] >> 1)
    s0b = slice(work.shape[0] >> 1, None)
    s1b = slice(work.shape[1] >> 1, None)
    s2b = slice(work.shape[2] >> 1, None)

    # Assign input
    work[s0a, s1a, s2a] = X

    # Loop over 2nd dimension extracting 2D slice from first and 3rd dimensions
    for f in xrange(work.shape[1] >> 1):
        # extract slice
        y = work[s0a, f, s2a].T

        # Do odd top-level filters on 3rd dim. The order here is important
        # since the second filtering will modify the elements of y as well
        # since y is merely a view onto work.
        work[s0a, f, s2b] = colfilter(y, h1o).T
        work[s0a, f, s2a] = colfilter(y, h0o).T

    # Loop over 3rd dimension extracting 2D slice from first and 2nd dimensions
    for f in xrange(work.shape[2]):
        # Do odd top-level filters on rows.
        y1 = work[s0a, s1a, f].T
        y2 = np.vstack((colfilter(y1, h0o), colfilter(y1, h1o))).T

        # Do odd top-level filters on columns.
        work[s0a, :, f] = colfilter(y2, h0o)
        work[s0b, :, f] = colfilter(y2, h1o)

    # Return appropriate slices of output
    return (work[s0a, s1a, s2a],                # LLL
        np.concatenate((
            work[s0a, s1b, s2a, np.newaxis],    # HLL
            work[s0b, s1a, s2a, np.newaxis],    # LHL
            work[s0b, s1b, s2a, np.newaxis],    # HHL
            work[s0a, s1a, s2b, np.newaxis],    # LLH
            work[s0a, s1b, s2b, np.newaxis],    # HLH
            work[s0b, s1a, s2b, np.newaxis],    # LHH
            work[s0b, s1b, s2b, np.newaxis],    # HLH
        ), axis=3))

def _level1_ifm(Yl, Yh, g0o, g1o):
    """Perform level 1 of the inverse 3d transform.

    """
    # Create work area
    work = np.zeros(np.asarray(Yl.shape) * 2, dtype=Yl.dtype)

    # Form some useful slices
    s0a = slice(None, work.shape[0] >> 1)
    s1a = slice(None, work.shape[1] >> 1)
    s2a = slice(None, work.shape[2] >> 1)
    s0b = slice(work.shape[0] >> 1, None)
    s1b = slice(work.shape[1] >> 1, None)
    s2b = slice(work.shape[2] >> 1, None)

    # Assign regions of work area
    work[s0a, s1a, s2a] = Yl
    work[s0a, s1b, s2a] = Yh[:,:,:,0]
    work[s0b, s1a, s2a] = Yh[:,:,:,1]
    work[s0b, s1b, s2a] = Yh[:,:,:,2]
    work[s0a, s1a, s2b] = Yh[:,:,:,3]
    work[s0a, s1b, s2b] = Yh[:,:,:,4]
    work[s0b, s1a, s2b] = Yh[:,:,:,5]
    work[s0b, s1b, s2b] = Yh[:,:,:,6]

    for f in xrange(work.shape[2]):
        # Do odd top-level filters on rows.
        y = colfilter(work[:, s1a, f].T, g0o) + colfilter(work[:, s1b, f].T, g1o)

        # Do odd top-level filters on columns.
        work[s0a, s1a, f] = colfilter(y[:, s0a].T, g0o) + colfilter(y[:, s0b].T, g1o)

    for f in xrange(work.shape[1]>>1):
        # Do odd top-level filters on 3rd dim.
        y = work[s0a, f, :].T
        work[s0a, f, s2a] = (colfilter(y[s2a, :], g0o) + colfilter(y[s2b, :], g1o)).T

    return work[s0a, s1a, s2a]

# vim:sw=4:sts=4:et
