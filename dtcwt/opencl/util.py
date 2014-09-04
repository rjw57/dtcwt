import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
    _HAVE_OPENCL = True
except ImportError:
    _HAVE_OPENCL = False

_DEFAULT_CHUNK_SIZE=32

def array_to_spec(array):
    data = array.base_data
    offset = np.int32(array.offset // array.dtype.itemsize)
    strides = cla.vec.make_int2(*np.divide(array.strides, array.dtype.itemsize))
    shape = cla.vec.make_int2(*array.shape)
    return data, offset, strides, shape

@np.vectorize
def _ceil_multiple(x, m):
        return m * ((x+(m-1)) // m)

def global_and_local_size(output_shape, chunk_size):
    local_size = tuple(int(x) for x in (chunk_size, chunk_size))
    global_size = tuple(int(x) for x in _ceil_multiple(output_shape[:2], local_size))
    return global_size, local_size

def good_chunk_size_for_queue(queue):
    # By default use _DEFAULT_CHUNK_SIZE for chunk size but reduce it if
    # the device does not support a work group that big.
    chunk_size = _DEFAULT_CHUNK_SIZE
    if chunk_size * chunk_size > queue.device.max_work_group_size:
        # Set chunk size to largest which will fit
        chunk_size = int(np.floor(np.sqrt(queue.device.max_work_group_size)))
    return chunk_size
