import os
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
    _HAVE_OPENCL = True
except ImportError:
    _HAVE_OPENCL = False

_PROGRAM_PATH=os.path.join(os.path.dirname(__file__), 'convolve.cl')
_DEFAULT_CHUNK_SIZE=32

def _array_to_spec(array):
    data = array.base_data[array.offset:]
    strides = cla.vec.make_int2(*np.divide(array.strides, array.dtype.itemsize))
    shape = cla.vec.make_int2(*array.shape)
    return data, strides, shape

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
    out_data, out_strides, out_shape = _array_to_spec(output_array)
    global_size, local_size = _global_and_local_size(output_array.shape, _DEFAULT_CHUNK_SIZE)
    return program.test_edge_reflect(queue, global_size, local_size,
            cla.vec.make_int2(*input_offset), cla.vec.make_int2(*input_shape),
            out_data, out_strides, out_shape, wait_for=wait_for)
