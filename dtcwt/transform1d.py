from __future__ import absolute_import

import numpy as np
import logging

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.numpy.lowlevel import colfilter, coldfilt, colifilt
from dtcwt.utils import as_column_vector, asfarray

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
    # Need this because colfilter and friends assumes input is 2d
    X = asfarray(X)
    if len(X.shape) == 1:
       X = np.atleast_2d(X).T

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

    L = np.asanyarray(X.shape)

    # ensure that X is an even length, thus enabling it to be extended if needs be.
    if X.shape[0] % 2 != 0:
        raise ValueError('Size of input X must be a multiple of 2')

    if nlevels == 0:
        if include_scale:
            return X, (), ()
        else:
            return X, ()

    # initialise
    Yh = [None,] * nlevels
    if include_scale:
        # This is only required if the user specifies scales are to be outputted
        Yscale = [None,] * nlevels

    # Level 1.
    Hi = colfilter(X, h1o)  
    Lo = colfilter(X, h0o)
    Yh[0] = Hi[::2,:] + 1j*Hi[1::2,:] # Convert Hi to complex form.
    if include_scale:
        Yscale[0] = Lo

    # Levels 2 and above.
    for level in xrange(1, nlevels):
        # Check to see if height of Lo is divisable by 4, if not extend.
        if Lo.shape[0] % 4 != 0:
            Lo = np.vstack((Lo[0,:], Lo, Lo[-1,:]))

        Hi = coldfilt(Lo,h1b,h1a)
        Lo = coldfilt(Lo,h0b,h0a)

        Yh[level] = Hi[::2,:] + 1j*Hi[1::2,:] # Convert Hi to complex form.
        if include_scale:
            Yscale[level] = Lo

    Yl = Lo

    if include_scale:
        return Yl, Yh, Yscale
    else:
        return Yl, Yh

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
    a = len(Yh) # No of levels.

    if gain_mask is None:
        gain_mask = np.ones(a) # Default gain_mask.

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

    level = a-1   # No of levels = no of rows in L.
    if level < 0:
        # if there are no levels in the input, just return the Yl value
        return Yl

    Lo = Yl
    while level >= 1:  # Reconstruct levels 2 and above in reverse order.
       Hi = c2q1d(Yh[level]*gain_mask[level])
       Lo = colifilt(Lo, g0b, g0a) + colifilt(Hi, g1b, g1a)
       
       if Lo.shape[0] != 2*Yh[level-1].shape[0]:  # If Lo is not the same length as the next Yh => t1 was extended.
          Lo = Lo[1:-1,...]                       # Therefore we have to clip Lo so it is the same height as the next Yh.

       if np.any(np.asanyarray(Lo.shape) != np.asanyarray(Yh[level-1].shape * np.array((2,1)))):
          raise ValueError('Yh sizes are not valid for DTWAVEIFM')
       
       level -= 1

    if level == 0:  # Reconstruct level 1.
       Hi = c2q1d(Yh[level]*gain_mask[level])
       Z = colfilter(Lo,g0o) + colfilter(Hi,g1o)

    # Return a 1d vector or a column vector
    if Z.shape[1] == 1:
        return Z.flatten()
    else:
        return Z

#==========================================================================================
#                  **********      INTERNAL FUNCTION    **********
#==========================================================================================

def c2q1d(x):
    """An internal function to convert a 1D Complex vector back to a real
    array,  which is twice the height of x.

    """
    a, b = x.shape
    z = np.zeros((a*2, b), dtype=x.real.dtype)
    z[::2, :] = np.real(x)
    z[1::2, :] = np.imag(x)

    return z

# vim:sw=4:sts=4:et
