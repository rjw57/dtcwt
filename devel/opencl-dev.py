#!/usr/bin/env python

import os
import sys

import matplotlib

from pylab import *
rcParams['image.cmap'] = 'gray'

import pyopencl as cl
import pyopencl.array as cla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tests.datasets as datasets

os.environ['PYOPENCL_CTX']='1'
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

FILTER_WIDTH=11
LOCAL_SIZE=(16,16,1)
convolve_prog = cl.Program(ctx,
    open(os.path.join(os.path.dirname(__file__), 'convolve.cl')).read()).build([
        '-DINPUT_TYPE=float4', '-DFILTER_WIDTH={0}'.format(FILTER_WIDTH),
        '-DLOCAL_SIZE_0={0}'.format(LOCAL_SIZE[0]),
        '-DLOCAL_SIZE_REST={0}'.format(np.product(LOCAL_SIZE[1:])),
    ])

def main():
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
    input_array = cla.empty(queue, test_image_rgba.shape[:2] + (1,), cla.vec.float3, order='C')
    cl.enqueue_copy(queue, input_array.data, input_buffer)

    # Create output array like input
    output_array = cla.zeros_like(input_array)

    # Process a small area of input into output
    input_strides = cla.vec.make_int4(
        input_array.strides[0]/input_array.dtype.itemsize,
        input_array.strides[1]/input_array.dtype.itemsize,
        input_array.size, input_array.size,
    )
    print('Passing input strides {0}'.format(input_strides))

    input_offset = cla.vec.make_int4(100, 100, 0, 0)
    input_skip = cla.vec.make_int4(1,1,1,1)
    print('Passing input offset {0} and skip {1}'.format(input_offset, input_skip))

    # Process a small area of output into output
    output_shape = cla.vec.make_int4(800,600,1,1)
    output_strides = cla.vec.make_int4(
        output_array.strides[0]/output_array.dtype.itemsize,
        output_array.strides[1]/output_array.dtype.itemsize,
        output_array.size, output_array.size,
    )
    print('Passing output shape {0} and strides {1}'.format(output_shape, output_strides))

    output_offset = cla.vec.make_int4(100, 100, 0, 0)
    output_skip = cla.vec.make_int4(1,1,1,1)
    print('Passing output offset {0} and skip {1}'.format(output_offset, output_skip))

    # Form filter
    filter_kernel = np.linspace(0, np.pi, FILTER_WIDTH)
    filter_kernel /= np.sum(filter_kernel)
    filter_kernel = cla.to_device(queue, np.asanyarray(filter_kernel, np.float32, 'C'))

    local_size = LOCAL_SIZE
    output_shape_tup = (output_shape['x'], output_shape['y'], output_shape['z'], output_shape['w'])
    global_size = tuple(y * int(np.ceil(x/y)) for x, y in zip(output_shape_tup, local_size))
    print('Global and local sizes: {0} and {1}'.format(global_size, local_size))

    global_size_arr = np.array(global_size)
    fw_arr = np.array(((FILTER_WIDTH-1)>>1, 0, 0))

    # Check for valid regions
    input_skip_array = np.array((input_skip['x'], input_skip['y'], input_skip['z']))
    input_offset_array = np.array((input_offset['x'], input_offset['y'], input_offset['z']))
    assert not np.any(input_offset_array - fw_arr*input_skip_array < 0)
    assert not np.any(input_offset_array +
            (fw_arr+global_size_arr)*input_skip_array > np.array(input_array.shape))

    output_shape_array = np.array((output_shape['x'], output_shape['y'], output_shape['z']))
    output_skip_array = np.array((output_skip['x'], output_skip['y'], output_skip['z']))
    output_offset_array = np.array((output_offset['x'], output_offset['y'], output_offset['z']))
    assert not np.any(output_offset_array < 0)
    assert not np.any(output_offset_array + global_size_arr*output_skip_array >
            np.array(output_array.shape))

    # Call kernel
    evt = convolve_prog.convolve(queue, global_size, local_size,
            input_array.data, input_offset, input_skip, input_strides,
            filter_kernel.data,
            output_array.data, output_offset, output_skip, output_shape, output_strides)
    evt.wait()

    figure()
    input_array_host = input_array.get()
    imshow(np.dstack(tuple(input_array_host[d] for d in 'xyz')))
    title('Input')

    figure()
    output_array_host = output_array.get()
    imshow(np.dstack(tuple(output_array_host[d] for d in 'xyz')))
    title('Output')

if __name__ == '__main__':
    main()
    show()
