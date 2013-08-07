import numpy as np
import logging

from dtcwt import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.lowlevel import colfilter, coldfilt, colifilt, as_column_vector

def dtwavexfm(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False):
    """ Function to perform a n-level DTCWT decompostion on a 1-D column vector
    X (or on the columns of a matrix X).

    Yl, Yh = dtwavexfm(X, nlevels, biort, qshift)
    Yl, Yh, Yscale = dtwavexfm(X, nlevels, biort, qshift, include_scale=True)

        X -> real 1-D signal column vector (or matrix of vectors)

        nlevels -> No. of levels of wavelet decomposition

        biort ->  'antonini'   => Antonini 9,7 tap filters.
                  'legall'     => LeGall 5,3 tap filters.
                  'near_sym_a' => Near-Symmetric 5,7 tap filters.
                  'near_sym_b' => Near-Symmetric 13,19 tap filters.

        qshift -> 'qshift_06' => Quarter Sample Shift Orthogonal (Q-Shift) 10,10 tap filters, 
                                 (only 6,6 non-zero taps).
                  'qshift_a' =>  Q-shift 10,10 tap filters,
                                 (with 10,10 non-zero taps, unlike qshift_06).
                  'qshift_b' => Q-Shift 14,14 tap filters.
                  'qshift_c' => Q-Shift 16,16 tap filters.
                  'qshift_d' => Q-Shift 18,18 tap filters.
                  

        Yl     -> The real lowpass subband from the final level.
        Yh     -> A cell array containing the complex highpass subband for each level.
        Yscale -> This is an OPTIONAL output argument, that is a cell array containing 
                  real lowpass coefficients for every scale. Only returned if include_scale
                  is True.

    If biort or qshift are not strings, there are interpreted as tuples of
    vectors giving filter coefficients. In the biort case, this shold be (h0o,
    g0o, h1o, g1o). In the qshift case, this should be (h0a, h0b, g0a, g0b,
    h1a, h1b, g1a, g1b).

    Example: Yl, Yh = dtwavexfm(X,5,'near_sym_b','qshift_b')
    performs a 5-level transform on the real image X using the 13,19-tap filters 
    for level 1 and the Q-shift 14-tap filters for levels >= 2.

    Nick Kingsbury and Cian Shaffrey
    Cambridge University, May 2002

    """

    # Need this because colfilter and friends assumes input is 2d
    X = as_column_vector(X)

    # Try to load coefficients if biort is a string parameter
    if isinstance(biort, basestring):
        h0o, g0o, h1o, g1o = _biort(biort)
    else:
        h0o, g0o, h1o, g1o = biort

    # Try to load coefficients if qshift is a string parameter
    if isinstance(qshift, basestring):
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    else:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

    L = np.asarray(X.shape)

    # ensure that X is an even length, thus enabling it to be extended if needs be.
    if X.shape[0] % 2 != 0:
        raise ValueError('Size of input X must be a multiple of 2')

    if len(X.shape) > 1 and np.any(L[1:] != 1):
        raise ValueError('X must be one-dimensional')

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
    """Function to perform an n-level dual-tree complex wavelet (DTCWT)
    1-D reconstruction.

    Z = dtwaveifm(Yl, Yh, biort, qshift, [gain_mask])
       
        Yl -> The real lowpass subband from the final level
        Yh -> A cell array containing the complex highpass subband for each level.

        biort ->  'antonini'   => Antonini 9,7 tap filters.
                  'legall'     => LeGall 5,3 tap filters.
                  'near_sym_a' => Near-Symmetric 5,7 tap filters.
                  'near_sym_b' => Near-Symmetric 13,19 tap filters.

        qshift -> 'qshift_06' => Quarter Sample Shift Orthogonal (Q-Shift) 10,10 tap filters, 
                                 (only 6,6 non-zero taps).
                  'qshift_a' =>  Q-shift 10,10 tap filters,
                                 (with 10,10 non-zero taps, unlike qshift_06).
                  'qshift_b' => Q-Shift 14,14 tap filters.
                  'qshift_c' => Q-Shift 16,16 tap filters.
                  'qshift_d' => Q-Shift 18,18 tap filters.

        gain_mask -> Gain to be applied to each subband. 
                     gain_mask(l) is gain for wavelet subband at level l.
                     If gain_mask(l) == 0, no computation is performed for band (l).
                     Default gain_mask = ones(1,length(Yh)). Note that l is 0-indexed.

        Z -> Reconstructed real signal vector (or matrix).

    If biort or qshift are not strings, there are interpreted as tuples of
    vectors giving filter coefficients. In the biort case, this shold be (h0o,
    g0o, h1o, g1o). In the qshift case, this should be (h0a, h0b, g0a, g0b,
    h1a, h1b, g1a, g1b).

    For example:  Z = dtwaveifm(Yl,Yh,'near_sym_b','qshift_b');
    performs a reconstruction from Yl,Yh using the 13,19-tap filters 
    for level 1 and the Q-shift 14-tap filters for levels >= 2.

    Nick Kingsbury and Cian Shaffrey
    Cambridge University, May 2002

    """
    a = len(Yh) # No of levels.

    if gain_mask is None:
        gain_mask = np.ones(a) # Default gain_mask.

    # Try to load coefficients if biort is a string parameter
    if isinstance(biort, basestring):
        h0o, g0o, h1o, g1o = _biort(biort)
    else:
        h0o, g0o, h1o, g1o = biort

    # Try to load coefficients if qshift is a string parameter
    if isinstance(qshift, basestring):
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    else:
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

       if np.any(np.asarray(Lo.shape) != np.asarray(Yh[level-1].shape * np.array((2,1)))):
          raise ValueError('Yh sizes are not valid for DTWAVEIFM')
       
       level -= 1

    if level == 0:  # Reconstruct level 1.
       Hi = c2q1d(Yh[level]*gain_mask[level])
       Z = colfilter(Lo,g0o) + colfilter(Hi,g1o)

    return Z.flatten()

#==========================================================================================
#                  **********      INTERNAL FUNCTION    **********
#==========================================================================================

def c2q1d(x):
    """An internal function to convert a 1D Complex vector back to a real
    array,  which is twice the height of x.

    """
    a, b = x.shape
    z = np.zeros((a*2, b))
    z[::2, :] = np.real(x)
    z[1::2, :] = np.imag(x)

    return z

# vim:sw=4:sts=4:et
