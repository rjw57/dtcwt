import numpy as np
import logging

from six.moves import xrange

from dtcwt.backend.base import TransformDomainSignal, ReconstructedSignal, Transform2d as Transform2dBase
from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import appropriate_complex_type_for, asfarray

from dtcwt.backend.backend_numpy.lowlevel import LowLevelBackendNumPy

# Use the NumPy low-level backend
_BACKEND = LowLevelBackendNumPy()

class Transform2d(Transform2dBase):
    """
    An implementation of the 2D DT-CWT via NumPy. *biort* and *qshift* are the
    wavelets which parameterise the transform. Valid values are documented in
    :py:func:`dtcwt.dtwavexfm2`.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
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

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 2D real array
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.backend.TransformDomainSignal` compatible object representing the transform-domain signal

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001

        """
        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 10 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 10:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 10 or 8 components.')

        X = np.atleast_2d(asfarray(X))
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
                return TransformDomainSignal(X, (), ())
            else:
                return TransformDomainSignal(X, ())

        # initialise
        Yh = [None,] * nlevels
        if include_scale:
            # this is only required if the user specifies a third output component.
            Yscale = [None,] * nlevels

        complex_dtype = appropriate_complex_type_for(X)

        if nlevels >= 1:
            # Do odd top-level filters on cols.
            Lo = _BACKEND.colfilter(X,h0o).T
            Hi = _BACKEND.colfilter(X,h1o).T
            if len(self.biort) >= 6:
                Ba = _BACKEND.colfilter(X,h2o).T

            # Do odd top-level filters on rows.
            LoLo = _BACKEND.colfilter(Lo,h0o).T
            Yh[0] = np.zeros((LoLo.shape[0] >> 1, LoLo.shape[1] >> 1, 6), dtype=complex_dtype)
            Yh[0][:,:,0:6:5] = q2c(_BACKEND.colfilter(Hi,h0o).T)     # Horizontal pair
            Yh[0][:,:,2:4:1] = q2c(_BACKEND.colfilter(Lo,h1o).T)     # Vertical pair
            if len(self.biort) >= 6:
                Yh[0][:,:,1:5:3] = q2c(_BACKEND.colfilter(Ba,h2o).T)     # Diagonal pair
            else:
                Yh[0][:,:,1:5:3] = q2c(_BACKEND.colfilter(Hi,h1o).T)     # Diagonal pair

            if include_scale:
                Yscale[0] = LoLo

        for level in xrange(1, nlevels):
            row_size, col_size = LoLo.shape
            if row_size % 4 != 0:
                # Extend by 2 rows if no. of rows of LoLo are not divisable by 4
                LoLo = np.vstack((LoLo[:1,:], LoLo, LoLo[-1:,:]))

            if col_size % 4 != 0:
                # Extend by 2 cols if no. of cols of LoLo are not divisable by 4
                LoLo = np.hstack((LoLo[:,:1], LoLo, LoLo[:,-1:]))

            # Do even Qshift filters on rows.
            Lo = _BACKEND.coldfilt(LoLo,h0b,h0a).T
            Hi = _BACKEND.coldfilt(LoLo,h1b,h1a).T
            if len(self.qshift) >= 10:
                Ba = _BACKEND.coldfilt(LoLo,h2b,h2a).T

            # Do even Qshift filters on columns.
            LoLo = _BACKEND.coldfilt(Lo,h0b,h0a).T

            Yh[level] = np.zeros((LoLo.shape[0]>>1, LoLo.shape[1]>>1, 6), dtype=complex_dtype)
            Yh[level][:,:,0:6:5] = q2c(_BACKEND.coldfilt(Hi,h0b,h0a).T)  # Horizontal
            Yh[level][:,:,2:4:1] = q2c(_BACKEND.coldfilt(Lo,h1b,h1a).T)  # Vertical
            if len(self.qshift) >= 10:
                Yh[level][:,:,1:5:3] = q2c(_BACKEND.coldfilt(Ba,h2b,h2a).T)  # Diagonal   
            else:
                Yh[level][:,:,1:5:3] = q2c(_BACKEND.coldfilt(Hi,h1b,h1a).T)  # Diagonal   

            if include_scale:
                Yscale[level] = LoLo

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
            return TransformDomainSignal(Yl, tuple(Yh), tuple(Yscale))
        else:
            return TransformDomainSignal(Yl, tuple(Yh))

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

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        Yl = td_signal.lowpass
        Yh = td_signal.subbands

        a = len(Yh) # No of levels.

        if gain_mask is None:
            gain_mask = np.ones((6,a)) # Default gain_mask.

        gain_mask = np.array(gain_mask)

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 10 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 10:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 10 or 8 components.')

        current_level = a
        Z = Yl

        while current_level >= 2: # this ensures that for level 1 we never do the following
            lh = c2q(Yh[current_level-1][:,:,[0, 5]], gain_mask[[0, 5], current_level-1])
            hl = c2q(Yh[current_level-1][:,:,[2, 3]], gain_mask[[2, 3], current_level-1])
            hh = c2q(Yh[current_level-1][:,:,[1, 4]], gain_mask[[1, 4], current_level-1])

            # Do even Qshift filters on columns.
            y1 = _BACKEND.colifilt(Z,g0b,g0a) + _BACKEND.colifilt(lh,g1b,g1a)
            y2 = _BACKEND.colifilt(hl,g0b,g0a) + _BACKEND.colifilt(hh,g1b,g1a)

            # Do even Qshift filters on rows.
            Z = (_BACKEND.colifilt(y1.T,g0b,g0a) + _BACKEND.colifilt(y2.T,g1b,g1a)).T

            # Check size of Z and crop as required
            [row_size, col_size] = Z.shape
            S = 2*np.array(Yh[current_level-2].shape)
            if row_size != S[0]:    # check to see if this result needs to be cropped for the rows
                Z = Z[1:-1,:]
            if col_size != S[1]:    # check to see if this result needs to be cropped for the cols
                Z = Z[:,1:-1]

            if np.any(np.array(Z.shape) != S[:2]):
                raise ValueError('Sizes of subbands are not valid for DTWAVEIFM2')
            
            current_level = current_level - 1

        if current_level == 1:
            lh = c2q(Yh[current_level-1][:,:,[0, 5]],gain_mask[[0, 5],current_level-1])
            hl = c2q(Yh[current_level-1][:,:,[2, 3]],gain_mask[[2, 3],current_level-1])
            hh = c2q(Yh[current_level-1][:,:,[1, 4]],gain_mask[[1, 4],current_level-1])

            # Do odd top-level filters on columns.
            y1 = _BACKEND.colfilter(Z,g0o) + _BACKEND.colfilter(lh,g1o)

            if len(self.qshift) >= 10:
                y2 = _BACKEND.colfilter(hl,g0o)
                y2bp = _BACKEND.colfilter(hh,g2o)

                # Do odd top-level filters on rows.
                Z = (_BACKEND.colfilter(y1.T,g0o) + _BACKEND.colfilter(y2.T,g1o) + _BACKEND.colfilter(y2bp.T, g2o)).T
            else:
                y2 = _BACKEND.colfilter(hl,g0o) + _BACKEND.colfilter(hh,g1o)

                # Do odd top-level filters on rows.
                Z = (_BACKEND.colfilter(y1.T,g0o) + _BACKEND.colfilter(y2.T,g1o)).T

        return ReconstructedSignal(Z)

#==========================================================================================
#                       **********    INTERNAL FUNCTIONS    **********
#==========================================================================================

def q2c(y):
    """Convert from quads in y to complex numbers in z.

    """
    j2 = (np.sqrt(0.5) * np.array([1, 1j])).astype(appropriate_complex_type_for(y))

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d

    # Combine (a,b) and (d,c) to form two complex subimages. 
    p = y[0::2, 0::2]*j2[0] + y[0::2, 1::2]*j2[1] # p = (a + jb) / sqrt(2)
    q = y[1::2, 1::2]*j2[0] - y[1::2, 0::2]*j2[1] # q = (d - jc) / sqrt(2)

    # Form the 2 subbands in z.
    z = np.dstack((p-q,p+q))

    return z

def c2q(w,gain):
    """Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 subbands
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """

    x = np.zeros((w.shape[0] << 1, w.shape[1] << 1), dtype=w.real.dtype)

    sc = np.sqrt(0.5) * gain
    P = w[:,:,0]*sc[0] + w[:,:,1]*sc[1]
    Q = w[:,:,0]*sc[0] - w[:,:,1]*sc[1]

    # Recover each of the 4 corners of the quads.
    x[0::2, 0::2] = P.real  # a = (A+C)*sc
    x[0::2, 1::2] = P.imag  # b = (B+D)*sc
    x[1::2, 0::2] = Q.imag  # c = (B-D)*sc
    x[1::2, 1::2] = -Q.real # d = (C-A)*sc

    return x

