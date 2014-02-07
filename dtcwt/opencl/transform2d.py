from __future__ import division, absolute_import

import logging
import numpy as np
from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import appropriate_complex_type_for, asfarray, memoize
from dtcwt.opencl.lowlevel import colfilter, coldfilt, colifilt
from dtcwt.opencl.lowlevel import axis_convolve, axis_convolve_dfilter, q2c
from dtcwt.opencl.lowlevel import to_device, to_queue, to_array, empty

from dtcwt.numpy import TransformDomainSignal, ReconstructedSignal
from dtcwt.numpy import Transform2d as Transform2dNumPy

try:
    from pyopencl.array import concatenate, Array as CLArray
except ImportError:
    # The lack of OpenCL will be caught by the low-level routines.
    pass

def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False, queue=None):
    t = Transform2d(biort=biort, qshift=qshift, queue=queue)
    r = t.forward(X, nlevels=nlevels, include_scale=include_scale)
    if include_scale:
        return r.lowpass, r.subbands, r.scales
    else:
        return r.lowpass, r.subbands

class TransformDomainSignal(object):
    """
    An interface-compatible version of
    :py:class:`dtcwt.TransformDomainSignal` where the initialiser
    arguments are assumed to by :py:class:`pyopencl.array.Array` instances.

    The attributes defined in :py:class:`dtcwt.TransformDomainSignal`
    are implemented via properties. The original OpenCL arrays may be accessed
    via the ``cl_...`` attributes.

    .. note::
    
        The copy from device to host is performed *once* and then memoized.
        This makes repeated access to the host-side attributes efficient but
        will mean that any changes to the device-side arrays will not be
        reflected in the host-side attributes after their first access. You
        should not be modifying the arrays once you return an instance of this
        class anyway but if you do, beware!

    .. py:attribute:: cl_lowpass

        The CL array containing the lowpass image.

    .. py:attribute:: cl_subbands

        A tuple of CL arrays containing the subband images.

    .. py:attribute:: cl_scales

        *(optional)* Either ``None`` or a tuple of lowpass images for each
        scale.

    """
    def __init__(self, lowpass, subbands, scales=None):
        self.cl_lowpass = lowpass
        self.cl_subbands = subbands
        self.cl_scales = scales

    @property
    def lowpass(self):
        if not hasattr(self, '_lowpass'):
            self._lowpass = to_array(self.cl_lowpass) if self.cl_lowpass is not None else None
        return self._lowpass

    @property
    def subbands(self):
        if not hasattr(self, '_subbands'):
            self._subbands = tuple(to_array(x) for x in self.cl_subbands) if self.cl_subbands is not None else None
        return self._subbands

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            self._scales = tuple(to_array(x) for x in self.cl_scales) if self.cl_scales is not None else None
        return self._scales

class Transform2d(Transform2dNumPy):
    """
    An implementation of the 2D DT-CWT via OpenCL. *biort* and *qshift* are the
    wavelets which parameterise the transform. Valid values are documented in
    :py:func:`dtcwt.dtwavexfm2`.

    If *queue* is non-*None* it is an instance of
    :py:class:`pyopencl.CommandQueue` which is used to compile and execute the
    OpenCL kernels which implement the transform. If it is *None*, the first
    available compute device is used.

    .. note::
        
        At the moment *only* the **forward** transform is accelerated. The
        inverse transform uses the NumPy backend.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, queue=None):
        super(Transform2d, self).__init__(biort=biort, qshift=qshift)
        self.queue = to_queue(queue)

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 2D real array
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.TransformDomainSignal` compatible object representing the transform-domain signal

        .. note::

            *X* may be a :py:class:`pyopencl.array.Array` instance which has
            already been copied to the device. In which case, it must be 2D.
            (I.e. a vector will not be auto-promoted.)

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001

        """
        queue = self.queue

        if isinstance(X, CLArray):
            if len(X.shape) != 2:
                raise ValueError('Input array must be two-dimensional')
        else:
            # If not an array, copy to device
            X = np.atleast_2d(asfarray(X))

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

        original_size = X.shape

        if len(X.shape) >= 3:
            raise ValueError('The entered image is {0}, please enter each image slice separately.'.
                    format('x'.join(list(str(s) for s in X.shape))))

        # The next few lines of code check to see if the image is odd in size, if so an extra ...
        # row/column will be added to the bottom/right of the image
        initial_row_extend = 0  #initialise
        initial_col_extend = 0
        if original_size[0] % 2 != 0:
            # if X.shape[0] is not divisible by 2 then we need to extend X by adding a row at the bottom
            X = to_array(X)
            X = np.vstack((X, X[[-1],:]))  # Any further extension will be done in due course.
            initial_row_extend = 1

        if original_size[1] % 2 != 0:
            # if X.shape[1] is not divisible by 2 then we need to extend X by adding a col to the left
            X = to_array(X)
            X = np.hstack((X, X[:,[-1]]))
            initial_col_extend = 1

        extended_size = X.shape

        # Copy X to the device if necessary
        X = to_device(X, queue=queue)

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

        complex_dtype = np.complex64

        if nlevels >= 1:
            # Do odd top-level filters on cols.
            Lo = axis_convolve(X,h0o,axis=0,queue=queue)
            Hi = axis_convolve(X,h1o,axis=0,queue=queue)
            if len(self.biort) >= 6:
                Ba = axis_convolve(X,h2o,axis=0,queue=queue)

            # Do odd top-level filters on rows.
            LoLo = axis_convolve(Lo,h0o,axis=1)

            if len(self.biort) >= 6:
                diag = axis_convolve(Ba,h2o,axis=1,queue=queue)
            else:
                diag = axis_convolve(Hi,h1o,axis=1,queue=queue)

            Yh[0] = q2c(
                axis_convolve(Hi,h0o,axis=1,queue=queue),
                axis_convolve(Lo,h1o,axis=1,queue=queue),
                diag,
            )

            if include_scale:
                Yscale[0] = LoLo

        for level in xrange(1, nlevels):
            row_size, col_size = LoLo.shape

            if row_size % 4 != 0:
                # Extend by 2 rows if no. of rows of LoLo are not divisible by 4
                LoLo = to_array(LoLo)
                LoLo = np.vstack((LoLo[:1,:], LoLo, LoLo[-1:,:]))

            if col_size % 4 != 0:
                # Extend by 2 cols if no. of cols of LoLo are not divisible by 4
                LoLo = to_array(LoLo)
                LoLo = np.hstack((LoLo[:,:1], LoLo, LoLo[:,-1:]))

            # Do even Qshift filters on rows.
            Lo = axis_convolve_dfilter(LoLo,h0b,axis=0,queue=queue)
            Hi = axis_convolve_dfilter(LoLo,h1b,axis=0,queue=queue)
            if len(self.qshift) >= 12:
                Ba = axis_convolve_dfilter(LoLo,h2b,axis=0,queue=queue)

            # Do even Qshift filters on columns.
            LoLo = axis_convolve_dfilter(Lo,h0b,axis=1,queue=queue)

            if len(self.qshift) >= 12:
                diag = axis_convolve_dfilter(Ba,h2b,axis=1,queue=queue)
            else:
                diag = axis_convolve_dfilter(Hi,h1b,axis=1,queue=queue)

            Yh[level] = q2c(
                axis_convolve_dfilter(Hi,h0b,axis=1,queue=queue),
                axis_convolve_dfilter(Lo,h1b,axis=1,queue=queue),
                diag,
            )

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
