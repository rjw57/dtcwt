import numpy as np
import logging

from dtcwt import biort as _biort, qshift as _qshift, colfilter, coldfilt

def dtwavexfm2(X, nlevels=3, biort='near_sym_a', qshift='qshift_a', include_scale=False):
    """Function to perform a n-level DTCWT-2D decompostion on a 2D matrix X

    Yl, Yh = dtwavexfm2(X, nlevels, biort, qshift)
    Yl, Yh, Yscale = dtwavexfm2(X, nlevels, biort, qshift, include_scale=True)

        X -> 2D real matrix/Image of shape (N, M)

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


        Yl     -> The real lowpass image from the final level
        Yh     -> A tuple containing the (N, M, 6) shape complex highpass subimages for each level.
        Yscale -> This is an OPTIONAL output argument, that is a tuple containing 
                  real lowpass coefficients for every scale. Only returned if include_scale
                  is True.

    If biort or qshift are not strings, there are interpreted as tuples of
    vectors giving filter coefficients. In the biort case, this shold be (h0o,
    g0o, h1o, g1o). In the qshift case, this should be (h0a, h0b, g0a, g0b,
    h1a, h1b, g1a, g1b).

    Example: Yl,Yh = dtwavexfm2(X,3,'near_sym_b','qshift_b')
    performs a 3-level transform on the real image X using the 13,19-tap filters 
    for level 1 and the Q-shift 14-tap filters for levels >= 2.

    Nick Kingsbury and Cian Shaffrey
    Cambridge University, Sept 2001

    """

    X = np.atleast_2d(X)

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

    original_size = X.shape
    
    if len(X.shape) >= 3:
        raise ValueError('The entered image is {0}, please enter each image slice separately.'.
                format('x'.join(list(str(s) for s in X.shape))))

    # The next few lines of code check to see if the image is odd in size, if so an extra ...
    # row/column will be added to the bottom/right of the image
    initial_row_extend = 0  #initialise
    initial_col_extend = 0
    if original_size[0] % 2 != 0:
        # if X.shape[0] is not divisable by 2 then we need to extend X by adding a row at the bottom
        X = np.vstack((X, X[-1,:]))  # Any further extension will be done in due course.
        initial_row_extend = 1;

    if original_size[1] % 2 != 0:
        # if X.shape[1] is not divisable by 2 then we need to extend X by adding a col to the left
        X = np.hstack((X, np.atleast_2d(X[:,-1]).T))
        initial_col_extend = 1
    
    extended_size = X.shape

    if nlevels == 0:
        if include_scale:
            return X, (), ()
        else:
            return X, ()

    # initialise
    Yh = [None,] * nlevels
    if include_scale:
        # this is only required if the user specifies a third output component.
        Yscale = [None,] * nlevels

    if nlevels >= 1:
        # Do odd top-level filters on cols.
        Lo = colfilter(X,h0o).T
        Hi = colfilter(X,h1o).T

        # Do odd top-level filters on rows.
        LoLo = colfilter(Lo,h0o).T
        Yh[0] = np.zeros((LoLo.shape[0]/2, LoLo.shape[1]/2, 6), dtype=np.complex64)

        Yh[0][:,:,[0, 5]] = q2c(colfilter(Hi,h0o).T)     # Horizontal pair
        Yh[0][:,:,[2, 3]] = q2c(colfilter(Lo,h1o).T)     # Vertical pair
        Yh[0][:,:,[1, 4]] = q2c(colfilter(Hi,h1o).T)     # Diagonal pair

        if include_scale:
            Yscale[0] = LoLo

    if nlevels >= 2:
        for level in xrange(1, nlevels):
            row_size, col_size = LoLo.shape
            if row_size % 4 != 0:
                # Extend by 2 rows if no. of rows of LoLo are not divisable by 4
                LoLo = np.vstack((LoLo[0,:], LoLo, LoLo[-1,:]))

            if col_size % 4 != 0:
                # Extend by 2 cols if no. of cols of LoLo are not divisable by 4
                LoLo = np.hstack((np.atleast_2d(LoLo[:,0]).T, LoLo, np.atleast_2d(LoLo[:,-1]).T))
         
            # Do even Qshift filters on rows.
            Lo = coldfilt(LoLo,h0b,h0a).T
            Hi = coldfilt(LoLo,h1b,h1a).T
          
            # Do even Qshift filters on columns.
            LoLo = coldfilt(Lo,h0b,h0a).T

            Yh[level] = np.zeros((LoLo.shape[0]/2, LoLo.shape[1]/2, 6), dtype=np.complex64)
            Yh[level][:,:,[0, 5]] = q2c(coldfilt(Hi,h0b,h0a).T)  # Horizontal
            Yh[level][:,:,[2, 3]] = q2c(coldfilt(Lo,h1b,h1a).T)  # Vertical
            Yh[level][:,:,[1, 4]] = q2c(coldfilt(Hi,h1b,h1a).T)  # Diagonal   

            if include_scale:
                Yscale[0] = LoLo

    Yl = LoLo
    
    if initial_row_extend == 1 and initial_col_extend == 1:
        logging.warn('The image entered is now a {0} NOT a {1}.'.format(
            'x'.join(list(str(s) for s in extended_size)),
            'x'.join(list(str(s) for s in original_size))))
        logging.warn(
            'The bottom row and rightmost column have been duplicated, prior to decomposition.')

    if initial_row_extend == 1 and initial_col_extend == 0:
        logging.warn('The image entered is now a {0} NOT a {1}.'.format(
            'x'.join(list(str(s) for s in extended_size)),
            'x'.join(list(str(s) for s in original_size))))
        logging.warn(
            'The bottom row has been duplicated, prior to decomposition.')

    if initial_row_extend == 0 and initial_col_extend == 1:
        logging.warn('The image entered is now a {0} NOT a {1}.'.format(
            'x'.join(list(str(s) for s in extended_size)),
            'x'.join(list(str(s) for s in original_size))))
        logging.warn(
            'The rightmost column has been duplicated, prior to decomposition.')

    if include_scale:
        return Yl, tuple(Yh), tuple(Yscale)
    else:
        return Yl, tuple(Yh)

#==========================================================================================
#                       **********    INTERNAL FUNCTION    **********
#==========================================================================================

def q2c(y):
    """Convert from quads in y to complex numbers in z.

    """
    j2 = np.sqrt(0.5) * np.array([1, 1j])

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d

    # Combine (a,b) and (d,c) to form two complex subimages. 
    p = y[0::2, 0::2]*j2[0] + y[0::2, 1::2]*j2[1]     # p = (a + jb) / sqrt(2)
    q = y[1::2, 1::2]*j2[0] - y[1::2, 0::2]*j2[1] # q = (d - jc) / sqrt(2)

    # Form the 2 subbands in z.
    z = np.dstack((p-q,p+q))

    return z

# vim:sw=4:sts=4:et
