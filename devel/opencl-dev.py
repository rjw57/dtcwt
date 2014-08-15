#!/usr/bin/env python

import logging
import os
import sys

import matplotlib

from pylab import *
rcParams['image.cmap'] = 'gray'
rcParams['image.interpolation'] = 'none'

import pyopencl as cl
import pyopencl.array as cla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tests.datasets as datasets

logging.basicConfig(level=logging.DEBUG)

from dtcwt.opencl._convolve import Convolution, Region

#if 'PYOPENCL_CTX' not in os.environ:
#    os.environ['PYOPENCL_CTX']='1'
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

FILTER_WIDTH=17

def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    conv = Convolution(ctx, FILTER_WIDTH, 4)

    print('OpenCL devices:')
    [print('    {0}'.format(d.name)) for d in ctx.devices]

    print('Loading images')
    test_image_rgb = datasets.traffic_hd_rgb()
    test_image_gray = datasets.traffic_hd()
    test_image_rgba = np.dstack((test_image_rgb, np.ones_like(test_image_rgb[:,:,0])))
    print('Input shape is {0}'.format(test_image_rgba.shape))

    # Swizzle RGB image into a form friendly to transform and upload to device
    input_buffer = np.asanyarray(test_image_rgba, np.float32, 'C')
    print('Input buffer size is {0}, strides {1}'.format(input_buffer.nbytes, input_buffer.strides))
    input_array = cla.empty(queue, test_image_rgba.shape[:2], cla.vec.float3, order='C')
    cl.enqueue_copy(queue, input_array.data, input_buffer)

    figure()
    input_array_host = input_array.get()
    imshow(np.dstack(tuple(input_array_host[d] for d in 'xyz')))
    title('Input')

    # Create output array like input
    output_array = cla.zeros_like(input_array)

    # Copy with sampling
    input_region = Region(input_array.data, input_array.shape,
            (-100,-100), (2,3),
            tuple(int(x/input_array.dtype.itemsize) for x in input_array.strides))
    output_region = Region(output_array.data, output_array.shape,
            (0,0), (1,1),
            tuple(int(x/output_array.dtype.itemsize) for x in output_array.strides))
    conv._copy_region(queue, output_array.shape, input_region, output_region)

    figure()
    output_array_host = output_array.get()
    imshow(np.dstack(tuple(output_array_host[d] for d in 'xyz')))
    title('Shifted')

    input_region = Region(input_array.data, input_array.shape,
            (0,0), (1,1),
            tuple(int(x/input_array.dtype.itemsize) for x in input_array.strides))
    output_region = Region(output_array.data, output_array.shape,
            (0,0), (1,1),
            tuple(int(x/output_array.dtype.itemsize) for x in output_array.strides))

    # Form filter
    filter_kernel = np.sin(np.linspace(0, np.pi, FILTER_WIDTH))
    filter_kernel /= np.sum(filter_kernel)

    # Call kernel
    conv.set_filter_kernel(queue, filter_kernel)
    pixel_count = (output_array.shape[0], output_array.shape[1])
    evt = conv._unchecked_convolve(queue, pixel_count, input_region, output_region)
    evt.wait()

    # Now the other direction
    output_region = Region(input_array.data, input_array.shape[::-1],
                 (0,0), (1,1),
                 tuple(int(x/input_array.dtype.itemsize) for x in input_array.strides[::-1]))
    input_region = Region(output_array.data, output_array.shape[::-1],
                 (0,0), (1,1),
                 tuple(int(x/output_array.dtype.itemsize) for x in output_array.strides[::-1]))

    evt = conv._unchecked_convolve(queue, pixel_count[::-1], input_region, output_region)
    evt.wait()

    figure()
    input_array_host = input_array.get()
    imshow(np.dstack(tuple(input_array_host[d] for d in 'xyz')))
    title('Output')

if __name__ == '__main__':
    main()
    show()
