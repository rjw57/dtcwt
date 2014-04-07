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

def _sample_opencl_array(kern, image, xs, ys, output, queue=None):
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

_SAMPLE_HEADER = CL_ARRAY_HEADER + '''
float sample(int4 sample_coord,
    const __global float* image_data,
    const struct array_spec* image_spec)
{
    sample_coord = reflect((int4)(sample_coord), 0, image_spec->shape);
    return image_data[coord_to_offset(sample_coord, *image_spec)];
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
    output = cl.array.empty(queue, out_shape, dtype=np.float32)

    # Create kernel
    kern = _sample_nearest_kernel_for_context(queue.context)

    # Create sampler
    _sample_opencl_array(kern, im, xs, ys, output, queue=queue)

    return to_array(output, queue=queue)

@memoize
def _sample_nearest_kernel_for_context(context):
    """Return a kernel for convolution suitable for use with *context*. The
    return values are memoized.

    """
    kern_prog = cl.Program(context, _SAMPLE_HEADER + _SAMPLE_NEAREST_KERNEL)
    kern_prog.build()
    return kern_prog.sample_kernel

_SAMPLE_NEAREST_KERNEL = '''
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
        output_coord.z,
        0
    );

    output[coord_to_offset(output_coord, output_spec)] =
        sample(sample_coord, image, &image_spec);
}
'''

def sample_bilinear(im, xs, ys, queue=None):
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
    output = cl.array.empty(queue, out_shape, dtype=np.float32)

    # Create kernel
    kern = _sample_bilinear_kernel_for_context(queue.context)

    # Create sampler
    _sample_opencl_array(kern, im, xs, ys, output, queue=queue)

    return to_array(output, queue=queue)

@memoize
def _sample_bilinear_kernel_for_context(context):
    """Return a kernel for convolution suitable for use with *context*. The
    return values are memoized.

    """
    kern_prog = cl.Program(context, _SAMPLE_HEADER + _SAMPLE_BILINEAR_KERNEL)
    kern_prog.build()
    return kern_prog.sample_kernel

_SAMPLE_BILINEAR_KERNEL = '''
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

    float2 sample_coord_f = (float2)(
        ys[coord_to_offset(output_coord, ys_spec)],
        xs[coord_to_offset(output_coord, xs_spec)]
    );

    float2 floor_coord;
    float2 frac_coord = fract(sample_coord_f, &floor_coord);

    int4 sample_coord = (int4)(
        floor_coord.x,
        floor_coord.y,
        output_coord.z,
        0
    );

    float4 samples = (float4)(
        sample(sample_coord, image, &image_spec),
        sample(sample_coord + (int4)(1,0,0,0), image, &image_spec),
        sample(sample_coord + (int4)(0,1,0,0), image, &image_spec),
        sample(sample_coord + (int4)(1,1,0,0), image, &image_spec)
    );

    //output[coord_to_offset(output_coord, output_spec)] = samples.s0;

    // Do x-wise sampling
    float lower = (1.0 - frac_coord.y) * samples.s0 + frac_coord.y * samples.s2;
    float upper = (1.0 - frac_coord.y) * samples.s1 + frac_coord.y * samples.s3;

    output[coord_to_offset(output_coord, output_spec)] =
        (1.0 - frac_coord.x) * lower + frac_coord.x * upper;
}
'''

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

