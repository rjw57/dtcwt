from __future__ import division, absolute_import

import struct

import numpy as np
import pyopencl as cl
from six.moves import xrange

from dtcwt.utils import reflect, memoize
from dtcwt.opencl.lowlevel import to_queue, to_device, to_array, CL_ARRAY_HEADER

def _sample_clipped(im, xs, ys):
    """Truncated and symmetric sampling."""
    sym_xs = reflect(xs, -0.5, im.shape[1]-0.5).astype(np.int)
    sym_ys = reflect(ys, -0.5, im.shape[0]-0.5).astype(np.int)
    return im[sym_ys, sym_xs, ...]

def _sample_opencl_array(image, xs, ys, output, queue=None):
    assert isinstance(image, cl.array.Array)
    assert isinstance(xs, cl.array.Array)
    assert isinstance(ys, cl.array.Array)
    assert isinstance(output, cl.array.Array)

    assert image.dtype == np.float32
    assert xs.shape == ys.shape
    assert output.shape == output.shape
    assert output.dtype == np.float32

    image_strides = struct.pack('iiii',
        *(tuple(s//image.dtype.itemsize for s in image.strides) + (0,0,0,0))[:4])
    image_offset = np.int32(image.offset)
    image_shape = struct.pack('iiii', *(tuple(image.shape) + (1,1,1,1))[:4])

    xs_strides = struct.pack('iiii',
        *(tuple(s//xs.dtype.itemsize for s in xs.strides) + (0,0,0,0))[:4])
    xs_offset = np.int32(xs.offset)

    ys_strides = struct.pack('iiii',
        *(tuple(s//ys.dtype.itemsize for s in ys.strides) + (0,0,0,0))[:4])
    ys_offset = np.int32(ys.offset)

    output_strides = struct.pack('iiii',
        *(tuple(s//output.dtype.itemsize for s in output.strides) + (0,0,0,0))[:4])
    output_offset = np.int32(output.offset)
    output_shape = struct.pack('iiii', *(tuple(output.shape) + (1,1,1,1))[:4])

    # Create kernel
    kern = _sample_kernel_for_context(queue.context)

    # Work out size of work group taking into account element step
    work_shape = np.array(output.shape[:3])

    # Work out optimum group size
    if work_shape.shape[0] >= 2 and np.all(work_shape[:2] > 1):
        local_shape = (int(np.floor(np.sqrt(queue.device.max_work_group_size))),) * 2 + (1,1,)
    else:
        local_shape = (queue.device.max_work_group_size, 1, 1)
    local_shape = local_shape[:len(work_shape)]
    global_shape = list(int(np.ceil(x/float(y))*y) for x, y in zip(work_shape, local_shape))

    kern(queue, global_shape, local_shape,
            image.base_data, image_strides, image_shape, image_offset,
            xs.base_data, xs_strides, xs_offset,
            ys.base_data, ys_strides, ys_offset,
            output.base_data, output_strides, output_shape, output_offset)

@memoize
def _sample_kernel_for_context(context):
    """Return a kernel for convolution suitable for use with *context*. The
    return values are memoized.

    """
    kern_prog = cl.Program(context, CL_ARRAY_HEADER + SAMPLE_KERNEL)
    kern_prog.build()
    return kern_prog.sample_kernel

SAMPLE_KERNEL = '''
void __kernel sample_kernel(
    const __global float* image, int4 image_strides, int4 image_shape, int image_offset,
    const __global float* xs, int4 xs_strides, int xs_offset,
    const __global float* ys, int4 ys_strides, int ys_offset,
    __global float* output, int4 output_strides, int4 output_shape, int output_offset)
{
    // NOTE: xs and ys has the same shape as output

    int4 output_coord = { get_global_id(0), get_global_id(1), get_global_id(2), 0 };
    struct array_spec image_spec = {
        .strides = image_strides, .shape = image_shape, .offset = image_offset };
    struct array_spec xs_spec = {
        .strides = xs_strides, .shape = output_shape, .offset = xs_offset };
    struct array_spec ys_spec = {
        .strides = ys_strides, .shape = output_shape, .offset = ys_offset };
    struct array_spec output_spec = {
        .strides = output_strides, .shape = output_shape, .offset = output_offset };

    if(any(output_coord >= output_spec.shape))
        return;

    int4 sample_coord = (int4)(
        round(ys[coord_to_offset(output_coord, ys_spec)]),
        round(xs[coord_to_offset(output_coord, xs_spec)]),
        0, 0
    );

    sample_coord = reflect((int4)(sample_coord), 0, image_shape);
    sample_coord.z = output_coord.z;

    output[coord_to_offset(output_coord, output_spec)] =
            image[coord_to_offset(sample_coord, image_spec)];
}
'''

def sample_nearest(im, xs, ys, queue=None):
    queue = to_queue(queue)
    im = to_device(im, queue=queue)
    xs = to_device(xs, queue=queue)
    ys = to_device(ys, queue=queue)

    # Make sure im is the correct shape
    if len(im.shape) != 3 and len(im.shape) != 2:
        raise ValueError('Input array must be 2- or 3-dimensional')

    # Make sure xs and ys match in shape
    if xs.shape != ys.shape:
        raise ValueError('Shape of xs and ys must match')

    # Create output
    out_shape = xs.shape + im.shape[2:]
    output = cl.array.zeros(queue, out_shape, dtype=np.float32)

    # Create sampler
    _sample_opencl_array(im, xs, ys, output, queue=queue)

    return to_array(output, queue=queue)

def sample_bilinear(im, xs, ys):
    # Convert arguments
    xs = np.asanyarray(xs)
    ys = np.asanyarray(ys)
    im = np.atleast_2d(np.asanyarray(im))

    if xs.shape != ys.shape:
        raise ValueError('Shape of xs and ys must match')

    # Split sample co-ords into floor and fractional part.
    floor_xs, floor_ys = np.floor(xs), np.floor(ys)
    frac_xs, frac_ys = xs - floor_xs, ys - floor_ys

    while len(im.shape) != len(frac_xs.shape):
        frac_xs = np.repeat(frac_xs[...,np.newaxis], im.shape[len(frac_xs.shape)], len(frac_xs.shape))
        frac_ys = np.repeat(frac_ys[...,np.newaxis], im.shape[len(frac_ys.shape)], len(frac_ys.shape))

    # Do x-wise sampling
    lower = (1.0 - frac_xs) * _sample_clipped(im, floor_xs, floor_ys) + frac_xs * _sample_clipped(im, floor_xs+1, floor_ys)
    upper = (1.0 - frac_xs) * _sample_clipped(im, floor_xs, floor_ys+1) + frac_xs * _sample_clipped(im, floor_xs+1, floor_ys+1)

    return ((1.0 - frac_ys) * lower + frac_ys * upper).astype(im.dtype)

def sample_lanczos(im, xs, ys):
    # Convert arguments
    xs = np.asanyarray(xs)
    ys = np.asanyarray(ys)
    im = np.atleast_2d(np.asanyarray(im))

    if xs.shape != ys.shape:
        raise ValueError('Shape of xs and ys must match')

    # Split sample co-ords into floor part
    floor_xs, floor_ys = np.floor(xs), np.floor(ys)
    frac_xs, frac_ys = xs - floor_xs, ys - floor_ys

    a = 3.0

    def _l(x):
        # Note: NumPy's sinc function returns sin(pi*x) / (pi*x)
        return np.sinc(x) * np.sinc(x/a)

    S = None
    for dx in np.arange(-a+1, a+1):
        Lx = _l(frac_xs - dx)
        for dy in np.arange(-a+1, a+1):
            Ly = _l(frac_ys - dy)

            weight = Lx * Ly
            while len(im.shape) != len(weight.shape):
                weight = np.repeat(weight[...,np.newaxis], im.shape[len(weight.shape)], len(weight.shape))

            contrib = weight * _sample_clipped(im, floor_xs+dx, floor_ys+dy)
            if S is None:
                S = contrib
            else:
                S += contrib

    return S

