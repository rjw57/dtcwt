from __future__ import division

import logging
import numpy as np
from six.moves import xrange

from dtcwt import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.lowlevel import appropriate_complex_type_for, asfarray
from dtcwt.opencl.lowlevel import colfilter, coldfilt, colifilt
from dtcwt.opencl.lowlevel import axis_convolve, axis_convolve_dfilter, q2c
from dtcwt.opencl.lowlevel import to_device, to_queue, to_array, empty

def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False, queue=None):
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
    queue = to_queue(queue)
    X = np.atleast_2d(asfarray(X))

    # Try to load coefficients if biort is a string parameter
    try:
        h0o, g0o, h1o, g1o = tuple(to_device(x) for x in _biort(biort))
    except TypeError:
        h0o, g0o, h1o, g1o = tuple(to_device(x) for x in biort)

    # Try to load coefficients if qshift is a string parameter
    try:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = tuple(to_device(x) for x in _qshift(qshift))
    except TypeError:
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = tuple(to_device(x) for x in qshift)

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
        X = np.vstack((X, X[[-1],:]))  # Any further extension will be done in due course.
        initial_row_extend = 1

    if original_size[1] % 2 != 0:
        # if X.shape[1] is not divisable by 2 then we need to extend X by adding a col to the left
        X = np.hstack((X, X[:,[-1]]))
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

    complex_dtype = appropriate_complex_type_for(X)

    if nlevels >= 1:
        # Do odd top-level filters on cols.
        Lo = axis_convolve(X,h0o,axis=0,queue=queue)
        Hi = axis_convolve(X,h1o,axis=0,queue=queue)

        # Do odd top-level filters on rows.
        LoLo = axis_convolve(Lo,h0o,axis=1)

        Yh[0] = to_array(q2c(
            axis_convolve(Hi,h0o,axis=1,queue=queue),
            axis_convolve(Lo,h1o,axis=1,queue=queue),
            axis_convolve(Hi,h1o,axis=1,queue=queue),
        ), queue=queue)

        if include_scale:
            Yscale[0] = to_array(LoLo)

    for level in xrange(1, nlevels):
        row_size, col_size = LoLo.shape
        if row_size % 4 != 0:
            # Extend by 2 rows if no. of rows of LoLo are not divisable by 4
            LoLo = to_array(LoLo)
            LoLo = np.vstack((LoLo[:1,:], LoLo, LoLo[-1:,:]))

        if col_size % 4 != 0:
            # Extend by 2 cols if no. of cols of LoLo are not divisable by 4
            LoLo = to_array(LoLo)
            LoLo = np.hstack((LoLo[:,:1], LoLo, LoLo[:,-1:]))

        # Do even Qshift filters on rows.
        Lo = axis_convolve_dfilter(LoLo,h0b,axis=0,queue=queue)
        Hi = axis_convolve_dfilter(LoLo,h1b,axis=0,queue=queue)

        # Do even Qshift filters on columns.
        LoLo = axis_convolve_dfilter(Lo,h0b,axis=1,queue=queue)

        Yh[level] = to_array(q2c(
            axis_convolve_dfilter(Hi,h0b,axis=1,queue=queue),
            axis_convolve_dfilter(Lo,h1b,axis=1,queue=queue),
            axis_convolve_dfilter(Hi,h1b,axis=1,queue=queue),
        ), queue=queue)

        if include_scale:
            Yscale[level] = to_array(LoLo)

    Yl = to_array(LoLo,queue=queue)

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

