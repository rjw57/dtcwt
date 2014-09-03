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
    program = _build_program(queue, 0, _DEFAULT_CHUNK_SIZE)
    out_data, out_offset, out_strides, out_shape = _array_to_spec(output_array)
    global_size, local_size = _global_and_local_size(output_array.shape, _DEFAULT_CHUNK_SIZE)
    return program.test_edge_reflect(queue, global_size, local_size,
            cla.vec.make_int2(*input_offset), cla.vec.make_int2(*input_shape),
            out_data, out_offset, out_strides, out_shape, wait_for=wait_for)

def biort(name):
    """Return a host array of kernel coefficients for the named biorthogonal
    level 1 wavelet. The lowpass coefficients are loaded into the 'x' field of
    the returned array and the highpass are loaded into the 'y' field. The
    returned array is suitable for passing to Convolution1D().
    """
    low, _, high, _ = dtcwt.coeffs.biort(name)
    assert low.shape[0] % 2 == 1
    assert high.shape[0] % 2 == 1

    kernel_coeffs = np.zeros((max(low.shape[0], high.shape[0]),), cla.vec.float2)

    low_offset = (kernel_coeffs.shape[0] - low.shape[0])>>1
    kernel_coeffs['x'][low_offset:low_offset+low.shape[0]] = low.flatten()
    high_offset = (kernel_coeffs.shape[0] - high.shape[0])>>1
    kernel_coeffs['y'][high_offset:high_offset+high.shape[0]] = high.flatten()

    return kernel_coeffs

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
        self._chunk_size = _DEFAULT_CHUNK_SIZE
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
