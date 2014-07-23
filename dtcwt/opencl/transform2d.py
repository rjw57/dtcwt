from __future__ import division, absolute_import

import logging
import numpy as np
from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import appropriate_complex_type_for, asfarray, memoize
from dtcwt.opencl.lowlevel import axis_convolve, axis_convolve_dfilter, q2c
from dtcwt.opencl.lowlevel import to_device, to_queue, to_array, empty

from dtcwt.numpy import Pyramid
from dtcwt.numpy import Transform2d as Transform2dNumPy

try:
    from pyopencl.array import concatenate, Array as CLArray
    import pyopencl as cl
    import pyopencl.array as cla
except ImportError:
    # The lack of OpenCL will be caught by the low-level routines.
    pass

def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False, queue=None):
    t = Transform2d(biort=biort, qshift=qshift, queue=queue)
    r = t.forward(X, nlevels=nlevels, include_scale=include_scale)
    if include_scale:
        return r.lowpass, r.highpasses, r.scales
    else:
        return r.lowpass, r.highpasses

class Pyramid(object):
    """
    An interface-compatible version of
    :py:class:`dtcwt.Pyramid` where the initialiser
    arguments are assumed to by :py:class:`pyopencl.array.Array` instances.

    The attributes defined in :py:class:`dtcwt.Pyramid`
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

    .. py:attribute:: cl_highpasses

        A tuple of CL arrays containing the subband images.

    .. py:attribute:: cl_scales

        *(optional)* Either ``None`` or a tuple of lowpass images for each
        scale.

    """
    def __init__(self, lowpass, highpasses, scales=None):
        self.cl_lowpass = lowpass
        self.cl_highpasses = highpasses
        self.cl_scales = scales

    @property
    def lowpass(self):
        if not hasattr(self, '_lowpass'):
            self._lowpass = to_array(self.cl_lowpass) if self.cl_lowpass is not None else None
        return self._lowpass

    @property
    def highpasses(self):
        if not hasattr(self, '_highpasses'):
            self._highpasses = tuple(to_array(x) for x in self.cl_highpasses) if self.cl_highpasses is not None else None
        return self._highpasses

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            self._scales = tuple(to_array(x) for x in self.cl_scales) if self.cl_scales is not None else None
        return self._scales

class Transform2d(Transform2dNumPy):
    """
    An implementation of the 2D DT-CWT via OpenCL. *biort* and *qshift* are the
    wavelets which parameterise the transform.

    If *queue* is non-*None* it is an instance of
    :py:class:`pyopencl.CommandQueue` which is used to compile and execute the
    OpenCL kernels which implement the transform. If it is *None*, the first
    available compute device is used.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`dtcwt.coeffs.biort` or :py:func:`dtcwt.coeffs.qshift` functions.
    Otherwise, they are interpreted as tuples of vectors giving filter
    coefficients. In the *biort* case, this should be (h0o, g0o, h1o, g1o). In
    the *qshift* case, this should be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    .. note::

        At the moment *only* the **forward** transform is accelerated. The
        inverse transform uses the NumPy backend.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, queue=None):
        super(Transform2d, self).__init__(biort=biort, qshift=qshift)
        self.queue = to_queue(queue)
        self._build_cl_resources()

    def _build_cl_resources(self):
        # Some complexities here because all of these are in terms of width, height

        local_dim = int(np.floor(np.sqrt(self.queue.device.max_work_group_size)))
        local_shape = (local_dim, local_dim)

        if len(self.biort) != 4:
            raise NotImplementedError('OpenCL transform not implemented for modified wavelets')

        # Work out maximum apron size
        h0o, _, h1o = self.biort[:3]
        max_apron_size = max((h0o.shape[0]-1)>>1, (h1o.shape[0]-1)>>1)

        # Therefore work out image chunk shape minus apron
        chunk_shape = tuple(x-2*max_apron_size for x in local_shape)
        if not np.all(np.asarray(chunk_shape) > 0):
            raise RuntimeError('OpenCL device does not have large enough work group size')

        # Check local memory can handle it. This is an approximate calculation
        local_mem_elements = local_shape[0]*local_shape[1]
        if self.queue.device.local_mem_size < (1+2)*4*local_mem_elements:
            raise RuntimeError('OpenCL device does not have enough local memory')

        self._program = cl.Program(self.queue.context, FORWARD_TRANSFORM_PROGRAM).build(
                ('-D LOCAL_WIDTH={0[0]} -D LOCAL_HEIGHT={0[1]} ' +
                 '-D APRON_SIZE={1} -D CHUNK_WIDTH={2[0]} -D CHUNK_HEIGHT={2[1]}').format(
                     local_shape, max_apron_size, chunk_shape
        ))

        self._local_shape = local_shape
        self._max_apron_size = max_apron_size
        self._chunk_shape = chunk_shape

        # Now cache the biort kernels
        # Interleave lo and hi kernels into single float2 buffer
        hilo_kernel = np.zeros(((2*max_apron_size)+1,), cla.vec.float2)
        lo_half_width, hi_half_width = (h0o.shape[0]-1)>>1, (h1o.shape[0]-1)>>1
        for d in range(-max_apron_size, max_apron_size+1):
            lo = h0o[d+lo_half_width] if d >= -lo_half_width and d <= lo_half_width else 0
            hi = h1o[d+hi_half_width] if d >= -hi_half_width and d <= hi_half_width else 0

            hilo_kernel[d+max_apron_size] = (hi,lo)
        self._hilo_kernel_cla = cla.to_device(self.queue, hilo_kernel)

    def _level1_forward(self, input_cla):
        """input_cla should be an OpenCL array in C order.

        Returns event, lowpass, subbands where, after event has happened, lowpass and subbands
        contain the level 1 result."""

        # In this function, shape is (width, height)
        input_shape = input_cla.shape[1::-1]

        # Compute output image shape
        output_shape = tuple(int(2*np.ceil(x/2.0)) for x in input_shape)

        # Compute subbands output shape
        subbands_shape = tuple(int(x>>1) for x in output_shape) + (6,)

        # Create output arrays (TODO: re-use temp_cla array)
        temp_cla = cla.Array(self.queue, output_shape[::-1], dtype=cla.vec.float4)
        lowpass_cla = cla.Array(self.queue, output_shape[::-1], dtype=np.float32)
        subbands_cla = cla.Array(self.queue, subbands_shape[1::-1] + subbands_shape[2:],
                dtype=np.complex64)

        # Work out global shapes
        conv_global_shape = tuple(z*int(np.ceil(x/float(y)))
                for x, y, z in zip(output_shape, self._chunk_shape, self._local_shape))
        extract_global_shape = tuple(y*int(np.ceil(x/float(y)))
                for x, y in zip(subbands_shape, self._local_shape))

        # Convolve
        prg = self._program
        evt = prg.l1_convolve_kernel(self.queue, conv_global_shape, self._local_shape,
                                    input_cla.data, cla.vec.make_int2(*input_shape), temp_cla.data,
                                    self._hilo_kernel_cla.data,
                                    np.int32((self._hilo_kernel_cla.shape[0]-1)>>1))
        evt = prg.l1_extract_subbands_kernel(self.queue, extract_global_shape, self._local_shape,
                                             temp_cla.data, cla.vec.make_int2(*subbands_shape[:2]),
                                             lowpass_cla.data, subbands_cla.data,
                                             wait_for=[evt,])

        return evt, lowpass_cla, subbands_cla

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 2D real array
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.Pyramid` compatible object representing the transform-domain signal

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
                return Pyramid(X, (), ())
            else:
                return Pyramid(X, ())

        # initialise
        Yh = [None,] * nlevels
        if include_scale:
            # this is only required if the user specifies a third output component.
            Yscale = [None,] * nlevels

        complex_dtype = np.complex64

        if nlevels >= 1:
            # Do level 1 transform
            l1_evt, LoLo, Yh[0] = self._level1_forward(X)
            l1_evt.wait()

#            # Do odd top-level filters on cols.
#            Lo = axis_convolve(X,h0o,axis=0,queue=queue)
#            Hi = axis_convolve(X,h1o,axis=0,queue=queue)
#            if len(self.biort) >= 6:
#                Ba = axis_convolve(X,h2o,axis=0,queue=queue)
#
#            # Do odd top-level filters on rows.
#            LoLo = axis_convolve(Lo,h0o,axis=1,queue=queue)
#
#            if len(self.biort) >= 6:
#                diag = axis_convolve(Ba,h2o,axis=1,queue=queue)
#            else:
#                diag = axis_convolve(Hi,h1o,axis=1,queue=queue)
#
#            Yh[0] = q2c(
#                axis_convolve(Hi,h0o,axis=1,queue=queue),
#                axis_convolve(Lo,h1o,axis=1,queue=queue),
#                diag,
#                queue=queue
#            )

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
                queue=queue
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
            return Pyramid(Yl, tuple(Yh), tuple(Yscale))
        else:
            return Pyramid(Yl, tuple(Yh))

FORWARD_TRANSFORM_PROGRAM = '''
    // reflect p about edges of image on [0, shape.x) x [0, shape.y)
    int2 reflect(int2 p, int2 shape) {
        // Reflect coord to be in +ve quarter-plane being careful about -1 -> 0
        p = select(p, -p-1, p < 0);

        // Restrict p to be within first 2*shape pixels
        p = p % (2*shape);

        // Reflect if outside of first shape pixels.
        return select(p, 2*shape-p-1, p >= shape);
    }

    // sample from input at p using edge reflection
    float sample(__global const float* input, int2 input_shape, int2 p) {
        // Reflect p to be within shape
        p = reflect(p, input_shape);
        return input[p.x + p.y*input_shape.x];
    }

    // Output shape must be the same as input shape rounded up to nearest even number.
    // Each work group operates on a sub-chunk.
    __kernel void l1_convolve_kernel(
        __global const float* input, int2 input_shape, __global float4* output,
        __constant float2* hilo_kernel, int hilo_kernel_half_width)
    {
        // Each kernel works on one chunk in local memory which includes the apron.

        // Output is input rounded up to nearest even
        int2 output_shape = (input_shape+(int2)(1,1)) & ~(int2)(1,1);

        // Work out which input point this thread corresponds to.
        int2 chunk_shape = (int2)(CHUNK_WIDTH, CHUNK_HEIGHT);
        int2 chunk_idx = (int2)(get_group_id(0), get_group_id(1));
        int2 intra_chunk_idx = (int2)(get_local_id(0)-APRON_SIZE, get_local_id(1)-APRON_SIZE);

        // N.B. output_pt is only valid outside of the apron
        int2 output_pt = chunk_idx * chunk_shape + intra_chunk_idx;
        int2 input_pt = output_pt;

        // Allocate local memory for image cache
        __local float input_cache[LOCAL_WIDTH*LOCAL_HEIGHT];
        int im_cache_idx = get_local_id(0) + get_local_id(1)*LOCAL_WIDTH;

        // Read from input into cache
        input_cache[im_cache_idx] = sample(input, input_shape, input_pt);

        // Wait for all threads to finish writing to local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // Allocate local memory for row-wise convolution cache
        __local float2 conv_cache[CHUNK_WIDTH*LOCAL_HEIGHT];
        int conv_cache_idx = (get_local_id(0)-APRON_SIZE) + get_local_id(1)*CHUNK_WIDTH;

        // Only compute row-wise convolution if we're within apron
        if((get_local_id(0) < APRON_SIZE) || (get_local_id(0) >= get_local_size(0) - APRON_SIZE))
            return;

        float2 conv_result = (float2)(0.f, 0.f);
        for(int d=-hilo_kernel_half_width; d<=hilo_kernel_half_width; ++d) {
            float2 hilo = hilo_kernel[d+hilo_kernel_half_width];
            conv_result += hilo * input_cache[im_cache_idx + d];
        }
        conv_cache[conv_cache_idx] = conv_result;

        // Wait for all threads to finish writing to local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // Only compute final output if we're within apron *and* within output
        if((get_local_id(1) < APRON_SIZE) || (get_local_id(1) >= get_local_size(1) - APRON_SIZE))
            return;
        if(any(output_pt < (int2)(0,0)) || any(output_pt >= output_shape))
            return;

        // Output corresponds to (col_row):
        // (lo_lo, hi_lo, lo_hi, hi_hi)

        float4 result = (float4)(0.f, 0.f, 0.f, 0.f);
        for(int d=-hilo_kernel_half_width; d<=hilo_kernel_half_width; ++d) {
            float2 hilo = hilo_kernel[d+hilo_kernel_half_width];
            result += conv_cache[conv_cache_idx + d*CHUNK_WIDTH].yyxx * hilo.yxyx;
        }

        output[output_pt.x + output_pt.y*output_shape.x] = result;
    }

    // The input must have even size in each dimension.
    // output_shape is half the input's shape
    // The lowpass output must be the same shape as the input.
    // The subbands output should be half the output shape in x- and y- and have
    // six elements in the z-direction. The z-direction is fastest moving.
    __kernel void l1_extract_subbands_kernel(
        __global float4* input, int2 output_shape,
        __global float* lowpass_output,
        __global float2* subbands_output)
    {
        int2 output_pt = (int2)(get_global_id(0), get_global_id(1));
        if(any(output_pt > output_shape)) { return; }

        int2 input_shape = 2*output_shape;
        int2 input_pt = 2*output_pt;
        int input_idx = input_pt.x + input_pt.y*input_shape.x;

        // Get input quad for this pixel
        // a--b
        // |  |
        // c--d
        float4 a = input[input_idx], b = input[input_idx+1];
        float4 c = input[input_idx+input_shape.x], d = input[input_idx+1+input_shape.x];

        // Copy lowpass output
        lowpass_output[input_idx] = a.x;
        lowpass_output[input_idx+1] = b.x;
        lowpass_output[input_idx+input_shape.x] = c.x;
        lowpass_output[input_idx+1+input_shape.x] = d.x;

        float sqrt_half = sqrt(0.5f);

        // Form complex parts
        float4 z1_real = sqrt_half * (a-d), z1_imag = sqrt_half * (b+c);
        float4 z2_real = sqrt_half * (a+d), z2_imag = sqrt_half * (b-c);

        int subband_idx = 6 * (output_pt.x + output_pt.y*output_shape.x);

        // Compute subbands
        subbands_output[subband_idx+0] = (float2)(z1_real.y, z1_imag.y);
        subbands_output[subband_idx+1] = (float2)(z1_real.w, z1_imag.w);
        subbands_output[subband_idx+2] = (float2)(z1_real.z, z1_imag.z);
        subbands_output[subband_idx+3] = (float2)(z2_real.z, z2_imag.z);
        subbands_output[subband_idx+4] = (float2)(z2_real.w, z2_imag.w);
        subbands_output[subband_idx+5] = (float2)(z2_real.y, z2_imag.y);
    }
'''
