import numpy as np
import logging

from six.moves import xrange

from dtcwt.numpy.common import Pyramid
from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import appropriate_complex_type_for, asfarray

from dtcwt.numpy.lowlevel import *

import pdb

class Transform3d(object):
    """
    An implementation of the 3D DT-CWT via NumPy. *biort* and *qshift* are the
    wavelets which parameterise the transform. Valid values are documented in
    :py:func:`dtcwt.dtwavexfm3`.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, ext_mode=4):
        # Load bi-orthogonal wavelets
        try:
            self.biort = _biort(biort)
        except TypeError:
            self.biort = biort

        # Load quarter sample shift wavelets
        try:
            self.qshift = _qshift(qshift)
        except TypeError:
            self.qshift = qshift

        self.ext_mode = ext_mode
            
    def forward(self, X, nlevels=3, include_scale=False, discard_level_1=False):
        """Perform a *n*-level DTCWT-3D decompostion on a 3D matrix *X*.
        
        :param X: 3D real array-like object
        :param nlevels: Number of levels of wavelet decomposition
        :param biort: Level 1 wavelets to use. See :py:func:`biort`.
        :param qshift: Level >= 2 wavelets to use. See :py:func:`qshift`.
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
        
        If *discard_level_1* is True the highpass coefficients at level 1 will not be
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
        X = np.atleast_3d(asfarray(X))

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')
        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift[:10]
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        # Check value of ext_mode. TODO: this should really be an enum :S
        if self.ext_mode != 4 and self.ext_mode != 8:
            raise ValueError('ext_mode must be one of 4 or 8')

        Yl = X
        Yh = [None,] * nlevels
        
        if include_scale:
            # this is only required if the user specifies a third output component.
            Yscale = [None,] * nlevels

        #pdb.set_trace()
        # level is 0-indexed
        for level in xrange(nlevels):
            # Transform
            if level == 0 and discard_level_1:
                Yl = _level1_xfm_no_highpass(Yl, h0o, h1o, self.ext_mode)
            elif level == 0 and not discard_level_1:
                Yl, Yh[level] = _level1_xfm(Yl, h0o, h1o, self.ext_mode)
            else:
                Yl, Yh[level] = _level2_xfm(Yl, h0a, h0b, h1a, h1b, self.ext_mode)
            if include_scale:
                Yscale[level] = Yl.copy()
        
                #Yh[nlevels+1]=1 #to throw an error for debugging in nose
        if include_scale:
            return Pyramid(Yl, tuple(Yh), tuple(Yscale))
        else: 
            return Pyramid(Yl, tuple(Yh))

    def inverse(self, td_signal):
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
        Yl = td_signal.lowpass
        Yh = td_signal.highpasses

        # Try to load coefficients if biort is a string parameter
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')
        
        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b, g2a, g2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        X = Yl

        nlevels = len(Yh)
        # level is 0-indexed but interpreted starting from the *last* level
        for level in xrange(nlevels):
            # Transform
            if level == nlevels-1: # non-obviously this is the 'first' level
                if Yh[-level-1] is None:
                    Yl = _level1_ifm_no_highpass(Yl, g0o, g1o)
                else:
                    Yl = _level1_ifm(Yl, Yh[-level-1], g0o, g1o)
            else:
                # Gracefully handle the Yh[0] is None case.
                if Yh[-level-2] is not None:
                    prev_shape = Yh[-level-2].shape
                else:
                    prev_shape = np.array(Yh[-level-1].shape) * 2
                    
                Yl = _level2_ifm(Yl, Yh[-level-1], g0a, g0b, g1a, g1b, self.ext_mode, prev_shape)

        return Yl

def _level1_xfm(X, h0o, h1o, ext_mode):
    """Perform level 1 of the 3d transform.

    """
    # Check shape of input according to ext_mode. Note that shape of X is
    # double original input in each direction.
    if ext_mode == 4 and np.any(np.fmod(X.shape, 2) != 0):
        raise ValueError('Input shape should be a multiple of 2 in each direction when self.ext_mode == 4')
    elif ext_mode == 8 and np.any(np.fmod(X.shape, 4) != 0):
        raise ValueError('Input shape should be a multiple of 4 in each direction when self.ext_mode == 8')

    # Create work area
    work_shape = np.asanyarray(X.shape) * 2

    # We need one extra row per octant if filter length is even
    if h0o.shape[0] % 2 == 0:
        work_shape += 2

    work = np.zeros(work_shape, dtype=X.dtype)

    # Form some useful slices
    s0a = slice(None, work.shape[0] >> 1)
    s1a = slice(None, work.shape[1] >> 1)
    s2a = slice(None, work.shape[2] >> 1)
    s0b = slice(work.shape[0] >> 1, None)
    s1b = slice(work.shape[1] >> 1, None)
    s2b = slice(work.shape[2] >> 1, None)

    x0a = slice(None, X.shape[0])
    x1a = slice(None, X.shape[1])
    x2a = slice(None, X.shape[2])
    x0b = slice(work.shape[0] >> 1, (work.shape[0] >> 1) + X.shape[0])
    x1b = slice(work.shape[1] >> 1, (work.shape[1] >> 1) + X.shape[1])
    x2b = slice(work.shape[2] >> 1, (work.shape[2] >> 1) + X.shape[2])

    # Assign input
    if h0o.shape[0] % 2 == 0:
        work[:X.shape[0], :X.shape[1], :X.shape[2]] = X

        # Copy last rows/cols/slices
        work[ X.shape[0], :X.shape[1], :X.shape[2]] = X[-1, :, :]
        work[:X.shape[0],  X.shape[1], :X.shape[2]] = X[:, -1, :]
        work[:X.shape[0], :X.shape[1],  X.shape[2]] = X[:, :, -1]
        work[X.shape[0], X.shape[1], X.shape[2]] = X[-1,-1,-1]
    else:
        work[s0a, s1a, s2a] = X

    # Loop over 2nd dimension extracting 2D slice from first and 3rd dimensions
    for f in xrange(work.shape[1] >> 1):
        # extract slice
        y = work[s0a, f, x2a].T
        # Do odd top-level filters on 3rd dim. The order here is important
        # since the second filtering will modify the elements of y as well
        # since y is merely a view onto work.
        work[s0a, f, s2b] = colfilter(y, h1o).T
        work[s0a, f, s2a] = colfilter(y, h0o).T

    # Loop over 3rd dimension extracting 2D slice from first and 2nd dimensions
    for f in xrange(work.shape[2]):
        # Do odd top-level filters on rows.
        y1 = work[x0a, x1a, f].T
        y2 = np.vstack((colfilter(y1, h0o), colfilter(y1, h1o))).T

        # Do odd top-level filters on columns.
        work[s0a, :, f] = colfilter(y2, h0o)
        work[s0b, :, f] = colfilter(y2, h1o)
        #if f==2:
        #work[:,:,work.shape[2]+1]=1 #to throw an error so we can inspect y in the unit test

    # Return appropriate slices of output
    return (
        work[s0a, s1a, s2a],                # LLL
        np.concatenate((
            cube2c(work[x0a, x1b, x2a]),    # HLL
            cube2c(work[x0b, x1a, x2a]),    # LHL
            cube2c(work[x0b, x1b, x2a]),    # HHL
            cube2c(work[x0a, x1a, x2b]),    # LLH
            cube2c(work[x0a, x1b, x2b]),    # HLH
            cube2c(work[x0b, x1a, x2b]),    # LHH
            cube2c(work[x0b, x1b, x2b]),    # HHH
            ), axis=3)
        )

def _level1_xfm_no_highpass(X, h0o, h1o, ext_mode):
    """Perform level 1 of the 3d transform discarding highpass subbands.

    """
    # Check shape of input according to ext_mode. Note that shape of X is
    # double original input in each direction.
    if ext_mode == 4 and np.any(np.fmod(X.shape, 2) != 0):
        raise ValueError('Input shape should be a multiple of 2 in each direction when self.ext_mode == 4')
    elif ext_mode == 8 and np.any(np.fmod(X.shape, 4) != 0):
        raise ValueError('Input shape should be a multiple of 4 in each direction when self.ext_mode == 8')

    out = np.zeros_like(X)

    # Loop over 2nd dimension extracting 2D slice from first and 3rd dimensions
    for f in xrange(X.shape[1]):
        # extract slice
        y = X[:, f, :].T
        out[:, f, :] = colfilter(y, h0o).T
        
  # Loop over 3rd dimension extracting 2D slice from first and 2nd dimensions
    for f in xrange(X.shape[2]):
        y = colfilter(out[:, :, f].T, h0o).T
        out[:, :, f] = colfilter(y, h0o)

    return out

def _level2_xfm(X, h0a, h0b, h1a, h1b, ext_mode):
    """Perform level 2 or greater of the 3d transform.

    """

    if ext_mode == 4:
        if X.shape[0] % 4 != 0:
            X = np.concatenate((X[[0],:,:], X, X[[-1],:,:]), 0)
        if X.shape[1] % 4 != 0:
            X = np.concatenate((X[:,[0],:], X, X[:,[-1],:]), 1)
        if X.shape[2] % 4 != 0:
            X = np.concatenate((X[:,:,[0]], X, X[:,:,[-1]]), 2)
    elif ext_mode == 8:
        if X.shape[0] % 8 != 0:
            X = np.concatenate((X[(0,0),:,:], X, X[(-1,-1),:,:]), 0)
        if X.shape[1] % 8 != 0:
            X = np.concatenate((X[:,(0,0),:], X, X[:,(-1,-1),:]), 1)
        if X.shape[2] % 8 != 0:
            X = np.concatenate((X[:,:,(0,0)], X, X[:,:,(-1,-1)]), 2)

    # Create work area
    work_shape = np.asanyarray(X.shape)
    work = np.zeros(work_shape, dtype=X.dtype)

    # Form some useful slices
    s0a = slice(None, work.shape[0] >> 1)
    s1a = slice(None, work.shape[1] >> 1)
    s2a = slice(None, work.shape[2] >> 1)
    s0b = slice(work.shape[0] >> 1, None)
    s1b = slice(work.shape[1] >> 1, None)
    s2b = slice(work.shape[2] >> 1, None)

    # Assign input
    work = X

    # Loop over 2nd dimension extracting 2D slice from first and 3rd dimensions
    for f in xrange(work.shape[1]):
        # extract slice (copy required because we overwrite the work array)
        y = work[:, f, :].T.copy()

        # Do even Qshift filters on 3rd dim.
        work[:, f, s2b] = coldfilt(y, h1b, h1a).T
        work[:, f, s2a] = coldfilt(y, h0b, h0a).T

    # Loop over 3rd dimension extracting 2D slice from first and 2nd dimensions
    for f in xrange(work.shape[2]):
        # Do even Qshift filters on rows.
        y1 = work[:, :, f].T
        y2 = np.vstack((coldfilt(y1, h0b, h0a), coldfilt(y1, h1b, h1a))).T
        
        # Do even Qshift filters on columns.
        work[s0a, :, f] = coldfilt(y2, h0b, h0a)
        work[s0b, :, f] = coldfilt(y2, h1b, h1a)

    # Return appropriate slices of output
    return (
        work[s0a, s1a, s2a],                # LLL
        np.concatenate((
            cube2c(work[s0a, s1b, s2a]),    # HLL
            cube2c(work[s0b, s1a, s2a]),    # LHL
            cube2c(work[s0b, s1b, s2a]),    # HHL
            cube2c(work[s0a, s1a, s2b]),    # LLH
            cube2c(work[s0a, s1b, s2b]),    # HLH
            cube2c(work[s0b, s1a, s2b]),    # LHH
            cube2c(work[s0b, s1b, s2b]),    # HHH
            ), axis=3)
        )

def _level1_ifm(Yl, Yh, g0o, g1o):
    """
    Perform level 1 of the inverse 3d transform.
    """
    # Create work area
    work = np.zeros(np.asanyarray(Yl.shape) * 2, dtype=Yl.dtype)

    # Work out shape of output
    Xshape = np.asanyarray(work.shape) >> 1
    if g0o.shape[0] % 2 == 0:
        # if we have an even length filter, we need to shrink the output by 1
        # to compensate for the addition of an extra row/column/slice in 
        # the forward transform
        Xshape -= 1

    # Form some useful slices
    s0a = slice(None, work.shape[0] >> 1)
    s1a = slice(None, work.shape[1] >> 1)
    s2a = slice(None, work.shape[2] >> 1)
    s0b = slice(work.shape[0] >> 1, None)
    s1b = slice(work.shape[1] >> 1, None)
    s2b = slice(work.shape[2] >> 1, None)
      
    x0a = slice(None, Xshape[0])
    x1a = slice(None, Xshape[1])
    x2a = slice(None, Xshape[2])
    x0b = slice(work.shape[0] >> 1, (work.shape[0] >> 1) + Xshape[0])
    x1b = slice(work.shape[1] >> 1, (work.shape[1] >> 1) + Xshape[1])
    x2b = slice(work.shape[2] >> 1, (work.shape[2] >> 1) + Xshape[2])

    # Assign regions of work area
    work[s0a, s1a, s2a] = Yl
    work[x0a, x1b, x2a] = c2cube(Yh[:,:,:, 0:4 ])
    work[x0b, x1a, x2a] = c2cube(Yh[:,:,:, 4:8 ])
    work[x0b, x1b, x2a] = c2cube(Yh[:,:,:, 8:12])
    work[x0a, x1a, x2b] = c2cube(Yh[:,:,:,12:16])
    work[x0a, x1b, x2b] = c2cube(Yh[:,:,:,16:20])
    work[x0b, x1a, x2b] = c2cube(Yh[:,:,:,20:24])
    work[x0b, x1b, x2b] = c2cube(Yh[:,:,:,24:28])

    for f in xrange(work.shape[2]):
        # Do odd top-level filters on rows.
        y = colfilter(work[:, x1a, f].T, g0o) + colfilter(work[:, x1b, f].T, g1o)

        # Do odd top-level filters on columns.
        work[s0a, s1a, f] = colfilter(y[:, x0a].T, g0o) + colfilter(y[:, x0b].T, g1o)

    for f in xrange(work.shape[1]>>1):
        # Do odd top-level filters on 3rd dim.
        y = work[s0a, f, :].T
        work[s0a, f, s2a] = (colfilter(y[x2a, :], g0o) + colfilter(y[x2b, :], g1o)).T

    if g0o.shape[0] % 2 == 0:
        return work[1:(work.shape[0]>>1), 1:(work.shape[1]>>1), 1:(work.shape[2]>>1)]
    else:
        return work[s0a, s1a, s2a]

def _level1_ifm_no_highpass(Yl, g0o, g1o):
    """Perform level 1 of the inverse 3d transform assuming highpass
    coefficients are zero.

    """
    # Create work area
    output = np.zeros_like(Yl)

    for f in xrange(Yl.shape[2]):
        y = colfilter(Yl[:, :, f].T, g0o)
        output[:, :, f] = colfilter(y.T, g0o)

    for f in xrange(Yl.shape[1]):
        y = output[:, f, :].T.copy()
        output[:, f, :] = colfilter(y, g0o)

    return output

def _level2_ifm(Yl, Yh, g0a, g0b, g1a, g1b, ext_mode, prev_level_size):
    """Perform level 2 or greater of the 3d inverse transform.

    """
    # Create work area
    work = np.zeros(np.asanyarray(Yl.shape)*2, dtype=Yl.dtype)

    # Form some useful slices
    s0a = slice(None, work.shape[0] >> 1)
    s1a = slice(None, work.shape[1] >> 1)
    s2a = slice(None, work.shape[2] >> 1)
    s0b = slice(work.shape[0] >> 1, None)
    s1b = slice(work.shape[1] >> 1, None)
    s2b = slice(work.shape[2] >> 1, None)

    # Assign regions of work area
    work[s0a, s1a, s2a] = Yl
    work[s0a, s1b, s2a] = c2cube(Yh[:,:,:, 0:4 ])
    work[s0b, s1a, s2a] = c2cube(Yh[:,:,:, 4:8 ])
    work[s0b, s1b, s2a] = c2cube(Yh[:,:,:, 8:12])
    work[s0a, s1a, s2b] = c2cube(Yh[:,:,:,12:16])
    work[s0a, s1b, s2b] = c2cube(Yh[:,:,:,16:20])
    work[s0b, s1a, s2b] = c2cube(Yh[:,:,:,20:24])
    work[s0b, s1b, s2b] = c2cube(Yh[:,:,:,24:28])

    for f in xrange(work.shape[2]):
        # Do even Qshift filters on rows.
        y = colifilt(work[:, s1a, f].T, g0b, g0a) + colifilt(work[:, s1b, f].T, g1b, g1a)

          # Do even Qshift filters on columns.
        work[:, :, f] = colifilt(y[:, s0a].T, g0b, g0a) + colifilt(y[:,s0b].T, g1b, g1a)

    for f in xrange(work.shape[1]):
        # Do even Qshift filters on 3rd dim.
        y = work[:, f, :].T
        work[:, f, :] = (colifilt(y[s2a, :], g0b, g0a) + colifilt(y[s2b, :], g1b, g1a)).T

    # Now check if the size of the previous level is exactly twice the size of
    # the current level. If YES, this means we have not done the extension in
    # the previous level. If NO, then we have to remove the appended row /
    # column / frame from the previous level DTCWT coefs.

    prev_level_size = np.asarray(prev_level_size)
    curr_level_size = np.asarray(Yh.shape)

    if ext_mode == 4:
        if curr_level_size[0] * 2 != prev_level_size[0]:
            # Discard the top and bottom rows
            work = work[1:-1,:,:]
        if curr_level_size[1] * 2 != prev_level_size[1]:
            # Discard the top and bottom rows
            work = work[:,1:-1,:]
        if curr_level_size[2] * 2 != prev_level_size[2]:
            # Discard the top and bottom rows
            work = work[:,:,1:-1]
    elif ext_mode == 8:
        if curr_level_size[0] * 2 != prev_level_size[0]:
        # Discard the top and bottom rows
            work = work[2:-2,:,:]
        if curr_level_size[1] * 2 != prev_level_size[1]:
        # Discard the top and bottom rows
            work = work[:,2:-2,:]
        if curr_level_size[2] * 2 != prev_level_size[2]:
        # Discard the top and bottom rows
            work = work[:,:,2:-2]

    return work

#==========================================================================================
#                       **********    INTERNAL FUNCTIONS    **********
#==========================================================================================

def cube2c(y):
    """Convert from octets in y to complex numbers in z.

    Arrange pixels from the corners of the quads into
    2 subimages of alternate real and imag pixels.

        e----f
       /|   /|
      a----b |
      | g- | h
      |/   |/
      c----d

    """

    # TODO: check this scaling
    j2 = 0.5 * np.array([1, 1j])

    # This is taken from:
    # Efficient Registration of Nonrigid 3-D Bodies, Huizhong Chen, and Nick Kingsbury, 2012
    # IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 21, NO. 1, JANUARY 2012
    # eqs. (6) to (9)

    A = y[0::2, 0::2, 0::2]
    B = y[0::2, 1::2, 0::2]
    C = y[1::2, 0::2, 0::2]
    D = y[1::2, 1::2, 0::2]
    E = y[0::2, 0::2, 1::2]
    F = y[0::2, 1::2, 1::2]
    G = y[1::2, 0::2, 1::2]
    H = y[1::2, 1::2, 1::2]

    # Combine to form subbands
    p = ( A-G-D-F) * j2[0] + ( B-H+C+E) * j2[1]
    q = ( A-G+D+F) * j2[0] + (-B+H+C+E) * j2[1]
    r = ( A+G+D-F) * j2[0] + ( B+H-C+E) * j2[1]
    s = ( A+G-D+F) * j2[0] + (-B-H-C+E) * j2[1]

    #j2[2]=1 #to throw an error
    # Form the 2 subbands in z.
    z = np.concatenate((
        p[:,:,:,np.newaxis],
        q[:,:,:,np.newaxis],
        r[:,:,:,np.newaxis],
        s[:,:,:,np.newaxis],
    ), axis=3)

    return z

def c2cube(z):
    """Convert from complex numbers octets in z to octets in y.

    Undoes cube2c().

        e----f
       /|   /|
      a----b |
      | g- | h
      |/   |/
      c----d

    """

    scale = 0.5

    p = z[:,:,:,0]
    q = z[:,:,:,1]
    r = z[:,:,:,2]
    s = z[:,:,:,3]

    pr, pi = p.real, p.imag #A,E
    qr, qi = q.real, q.imag #B,F
    rr, ri = r.real, r.imag #C,G
    sr, si = s.real, s.imag #D,H

    y = np.zeros(np.asanyarray(z.shape[:3])*2, dtype=z.real.dtype)

    y[0::2, 0::2, 0::2] = ( pr+qr+rr+sr)
    y[1::2, 0::2, 1::2] = (-pr-qr+rr+sr)
    y[1::2, 1::2, 0::2] = (-pr+qr+rr-sr)
    y[0::2, 1::2, 1::2] = (-pr+qr-rr+sr)

    y[0::2, 1::2, 0::2] = ( pi-qi+ri-si)
    y[1::2, 1::2, 1::2] = (-pi+qi+ri-si)
    y[1::2, 0::2, 0::2] = ( pi+qi-ri-si)
    y[0::2, 0::2, 1::2] = ( pi+qi+ri+si)

    return y * scale

# vim:sw=4:sts=4:et
