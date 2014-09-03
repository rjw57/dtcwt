"""Low-level support for convolution using OpenCL."""

import os
import numpy as np
import dtcwt.coeffs

try:
    import pyopencl as cl
    import pyopencl.array as cla
    _HAVE_OPENCL = True
except ImportError:
    _HAVE_OPENCL = False

_PROGRAM_PATH=os.path.join(os.path.dirname(__file__), 'convolve.cl')
_DEFAULT_CHUNK_SIZE=32

def _array_to_spec(array):
    data = array.base_data
    offset = np.int32(array.offset // array.dtype.itemsize)
    strides = cla.vec.make_int2(*np.divide(array.strides, array.dtype.itemsize))
    shape = cla.vec.make_int2(*array.shape)
    return data, offset, strides, shape

@np.vectorize
def _ceil_multiple(x, m):
        return m * ((x+(m-1)) // m)

def _global_and_local_size(output_shape, chunk_size):
    local_size = tuple(int(x) for x in (chunk_size, chunk_size))
    global_size = tuple(int(x) for x in _ceil_multiple(output_shape[:2], local_size))
    return global_size, local_size

def _good_chunk_size_for_queue(queue):
    # By default use _DEFAULT_CHUNK_SIZE for chunk size but reduce it if
    # the device does not support a work group that big.
    chunk_size = _DEFAULT_CHUNK_SIZE
    if chunk_size * chunk_size > queue.device.max_work_group_size:
        # Set chunk size to largest which will fit
        chunk_size = int(np.floor(np.sqrt(queue.device.max_work_group_size)))
    return chunk_size

def _build_program(queue, kernel_half_width, chunk_size):
    """Load and build the convolution kernel program for a specified kernel
    half width and chunk size. Returns a cl.Program object on success.

    """
    constants = {
        'KERNEL_HALF_WIDTH': kernel_half_width,
        'CHUNK_SIZE': chunk_size,
    }
    options = list('-D{0}={1}'.format(k,v) for k,v in constants.items())
    program = cl.Program(queue.context, open(_PROGRAM_PATH).read())
    program.build(options)
    return program

def _write_input_pixel_test_image(queue, output_array, input_offset, input_shape, wait_for=None):
    chunk_size = _good_chunk_size_for_queue(queue)
    program = _build_program(queue, 0, chunk_size)
    out_data, out_offset, out_strides, out_shape = _array_to_spec(output_array)
    global_size, local_size = _global_and_local_size(output_array.shape, chunk_size)
    return program.test_edge_reflect(queue, global_size, local_size,
            cla.vec.make_int2(*input_offset), cla.vec.make_int2(*input_shape),
            out_data, out_offset, out_strides, out_shape, wait_for=wait_for)

class Convolution1D(object):
    def __init__(self, queue, kernel_coeffs):
        if kernel_coeffs.dtype != cla.vec.float2 or len(kernel_coeffs.shape) != 1:
            raise ValueError('Kernel coefficients must be a 1d vector of float2')
        if kernel_coeffs.shape[0] % 2 != 1:
            raise ValueError('Kernel coefficients vector must have odd length')

        # Remember which queue we work on
        self._queue = queue

        # Copy kernel to device if necessary
        self._kernel_coeffs = cla.to_device(queue, kernel_coeffs)

        # Compute chunk size, kernel half width and build device program
        self._chunk_size = _good_chunk_size_for_queue(queue)
        self._kernel_half_width = (kernel_coeffs.shape[0] - 1)>>1
        self._program = _build_program(queue, self._kernel_half_width, self._chunk_size)

        # Fetch kernels
        self._convolve_scalar = self._program.convolve_scalar

    def __call__(self, input_array, output_array, wait_for=None):
        in_data, in_offset, in_strides, in_shape = _array_to_spec(input_array)
        out_data, out_offset, out_strides, out_shape = _array_to_spec(output_array)
        global_size, local_size = _global_and_local_size(output_array.shape, self._chunk_size)
        return self._convolve_scalar(self._queue, global_size, local_size,
                self._kernel_coeffs.data,
                in_data, in_offset, in_strides, in_shape,
                out_data, out_offset, out_strides, out_shape, wait_for=wait_for)
