"""2D DTCWT implemented using OpenCL."""

import logging
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
    _HAVE_OPENCL = True
except ImportError:
    _HAVE_OPENCL = False

from dtcwt.numpy.common import Pyramid as NumPyPyramid
from dtcwt.numpy.transform2d import DEFAULT_BIORT, DEFAULT_QSHIFT

import dtcwt.opencl.convolve as convolve
import dtcwt.opencl.coeffs as coeffs

log = logging.getLogger()

class Pyramid(object):
    """OpenCL transform pyramid.

    The OpenCL transform implementation tries to keep the results on the device
    if possible. Accessing the :py:attr:`lowpass` or :py:attr:`highpasses`
    attribute on instances of this class will transparently copy the data over
    to the host.

    .. py:attribute:: lowpass

        A NumPy-compatible array containing the coarsest scale lowpass signal.

        .. note::

            Accessing this attribute causes an implicit copy from the device to the
            host.

    .. py:attribute:: highpasses

        A tuple where each element is the complex subband coefficients for
        corresponding scales finest to coarsest.

        .. note::

            Accessing this attribute causes an implicit copy from the device to the
            host.

    .. py:attribute:: event

        A PyOpenCL event which can be waited for to ensure that the Pyramid has
        been fully computed.

    .. py:attribute:: cl_lowpass

        A :py:class:`pyopencl.array.Array` containing the coarsest scale
        lowpass signal.

    .. py:attribute:: cl_highpasses

        A tuple where each element is the complex subband coefficients for
        corresponding scales finest to coarsest. The subband coefficients are
        stored in a :py:class:`pyopencl.array.Array` instance.

    """

    def __init__(self, lowpass, highpasses, event):
        self.cl_lowpass = lowpass
        self.cl_highpasses = highpasses
        self.event = event

    def get(self):
        """Return a :py:class:`dtcwt.numpy.Pyramid` object initialised by
        copying the results from the device. This is useful if you want to get
        hold of a host-side copy of the results and then let the device-side
        results be garbage collected.

        """
        return NumPyPyramid(self.lowpass, self.highpasses)

    @property
    def lowpass(self):
        return self.cl_lowpass.get()

    @property
    def highpasses(self):
        return tuple(x.get() for x in self.cl_highpasses)

class Transform2d(object):
    """OpenCL implementation of 2D DTCWT.

    The :py:class:`pyopencl.CommandQueue` associated with the transform is
    specified as *queue*.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, queue=None):
        if queue is None:
            log.warn('Queue not specified for OpenCL transform. Using first available.')
            ctx = cl.create_some_context(interactive=False)
            queue = cl.CommandQueue(ctx)

        self._queue = queue
        self._biort_coeffs = coeffs.biort(biort)
        self._input_dtype = np.float32
        self._l1_convolution = convolve.Convolution1D(
            self._queue, self._biort_coeffs, self._input_dtype)

    def forward(self, X, nlevels=3, include_scale=False, wait_for=None):
        if include_scale:
            raise NotImplementedError(
                'Setting include_scale is not implemented for the OpenCL transform')
        if nlevels != 1:
            raise NotImplementedError(
                'Setting nlevels > 1 is not implemented for the OpenCL transform')

        if not isinstance(X, cla.Array):
            X = np.atleast_2d(X)
            if X.dtype != self._input_dtype:
                log.warn('OpenCL operates only on arrays with dtype {0}.'.format(self._input_dtype))
                log.warn('The input will be converted.')
                X = np.asarray(X, dtype=self._input_dtype)
            X = cla.to_device(self._queue, X)
        elif len(X.shape) < 2:
            raise ValueError('OpenCL arrays need to be at least 2 dimensional.')
        elif X.dtype != self._input_dtype:
            raise ValueError('OpenCL arrays need to have dtype {0}. '.format(self._input_dtype) +
                '(Passed array has {0}.)'.format(X.dtype))

        if len(X.shape) > 2:
            raise NotImplementedError(
                'Input with 3 or greater dimensions not supported in OpenCL transform')

        # Do level 1 of transform
        lp, hp, evt = self._level1_transform(X, wait_for)

        # Return result
        return Pyramid(lp, (hp,), evt)

    def inverse(self, pyramid, gain_mask=None):
        if gain_mask is not None:
            raise NotImplementedError('Setting gain_mask is not supported in the OpenCL backend.')
        raise NotImplementedError()

    def _level1_transform(self, X, wait_for):
        if np.any(np.asarray(X.shape[:2]) % 2 == 1):
            raise ValueError('Input must have even rows and columns')

        # Do the first dimension convolution
        lohi = cla.empty(self._queue, X.shape, dtype=cla.vec.float2)
        evt = self._l1_convolution(X, lohi, wait_for=wait_for)

        lp = cla.empty(self._queue, X.shape, dtype=np.float32)
        hp = cla.empty(self._queue, X.shape + (6,), dtype=np.complex64)
        return lp, hp, evt
