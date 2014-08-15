#!/usr/bin/env python
from __future__ import division

import logging
import os
import sys
import time

import matplotlib

from pylab import *
rcParams['image.cmap'] = 'gray'
rcParams['image.interpolation'] = 'none'

import pyopencl as cl
import pyopencl.array as cla

import dtcwt.numpy as dnp
from dtcwt.numpy.transform2d import q2c
import dtcwt.coeffs as coeffs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tests.datasets as datasets

logging.basicConfig(level=logging.DEBUG)

from dtcwt.opencl._convolve import Convolution, Region

#if 'PYOPENCL_CTX' not in os.environ:
#    os.environ['PYOPENCL_CTX']='1'
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

FILTER_WIDTH=17

def climshow(image):
    if isinstance(image, cla.Array):
        image = image.get()
    if image.dtype is cla.vec.float4:
        imshow(np.dstack(tuple(image[d] for d in 'xyzw')))
    else:
        imshow(image, cmap=cm.gray, clim=(0,1))

class Level1(object):
    def __init__(self, queue, biort, input_shape):
        self.queue = queue
        if len(input_shape) != 2:
            raise ValueError('Input shape must be 2 dimensional')
        self.input_shape = input_shape
        if np.any(tuple(x % 2 != 0 for x in input_shape)):
            raise ValueError('Input shape must be even in each dimension')

        # Construct the filter kernel
        h0o, g0o, h1o, g1o = biort
        h0o, h1o = h0o.flatten(), h1o.flatten() # to 1d

        assert h0o.shape[0] % 2 == 1
        assert h1o.shape[0] % 2 == 1

        filter_width = max(h0o.shape[0], h1o.shape[0])
        self.filter_width = filter_width

        filter_kernel = np.zeros((2, filter_width))
        h0o_offset = (filter_width-h0o.shape[0])>>1
        filter_kernel[0,h0o_offset:(h0o.shape[0]+h0o_offset)] = h0o
        h1o_offset = (filter_width-h1o.shape[0])>>1
        filter_kernel[1,h1o_offset:(h1o.shape[0]+h1o_offset)] = h1o

        # Prepare the convolution
        self.conv = Convolution(queue.context, filter_width, 1)
        self.conv.set_filter_kernel(queue, filter_kernel)

        # Calculate size of workspace: 2 output pixels for every input for
        # column wise convolution and a further 4 output pixels for rowwise.
        # Alignment is to 4 pixel boundaries
        input_size = np.product(input_shape)
        pixel_size = 1 * 4 # bytes
        alignment = 4

        # Aligned sizes of various sub-bits of workspace in units of input data type
        self._ws_sizes = [
                alignment*((2*input_size+(alignment-1))//alignment),
                alignment*((4*input_size+(alignment-1))//alignment)
        ]
        self.workspace_size = int(np.sum(self._ws_sizes) * pixel_size)

        # Output is full-res lolo image plus 6 half-res complex images => four
        # times redundancy.
        self.output_sizes = [
                int(np.product(input_shape) * pixel_size),
                int(3*np.product(input_shape) * pixel_size)
        ]

    def transform(self, input_array, workspace, wait_for=None):
        """Workspace and output must have been created by create_{...} methods.

        """
        wait_for = wait_for or []

        if input_array.shape != self.input_shape:
            raise ValueError('Input array does not have the shape prepared for')
        if input_array.dtype != np.float32:
            raise ValueError('Input array does not have float32 dtype')
        if workspace.size < self.workspace_size:
            raise ValueError('Workspace is too small ({0} vs {1})'.format(
                workspace.size, self.workspace_size))

        # Create workspace and output array views on workspace
        workspace_array = cla.Array(self.queue,
                input_array.shape, cla.vec.float2,
                data=workspace, offset=0)
        output_array = cla.Array(self.queue,
                input_array.shape, cla.vec.float4,
                data=workspace, offset=(workspace_array.size+1)>>1)
        output_array = cla.empty(self.queue, self.input_shape, cla.vec.float4)

        # output_array, workspace_array = output, workspace

        # C-style ordering where first index is horizontal
        input_array_strides = (1, input_array.shape[1],
                input_array.shape[1]*input_array.shape[0])

        # C-style ordering where first index is horizontal
        workspace_array_strides = (1, workspace_array.shape[1],
                workspace_array.shape[1] * workspace_array.shape[0])

        # Perform column convolution
        input_region = Region(input_array.base_data,
                input_array.offset,
                input_array.shape[::-1],
                (0,0), (1,1), input_array_strides)
        output_region = Region(workspace_array.base_data,
                workspace_array.offset*2,
                workspace_array.shape[::-1],
                (0,0), (1,1), workspace_array_strides)
        colconv_evt = self.conv._unchecked_convolve(self.queue, workspace_array.shape[::-1],
                input_region, output_region, wait_for=wait_for)

        # Output will store lo-lo, lo-hi, hi-lo and hi-hi column-then-row-wise
        # convolution as four interleaved pixels. We will write to it assuming
        # float2 pixels and so it has double the number of columns we specify.
        effective_output_shape = (output_array.shape[0], 2*output_array.shape[1])
        output_array_strides = (effective_output_shape[1], 1,
                effective_output_shape[0] * effective_output_shape[1])

        # Now we view workspace as a double-wide version of the input with floating
        # point pixels where pixels alternate lo-hi
        effective_workspace_shape = (workspace_array.shape[0], 2*workspace_array.shape[1])
        workspace_array_strides = (effective_workspace_shape[1], 1,
                effective_workspace_shape[0]*effective_workspace_shape[1])
        input_region = Region(workspace_array.base_data,
                workspace_array.offset*2,
                effective_workspace_shape,
                (0,0), (1,2), workspace_array_strides)
        output_region = Region(output_array.base_data,
                output_array.offset*4,
                effective_output_shape,
                (0,0), (1,2), output_array_strides)
        rowloconv_evt = self.conv._unchecked_convolve(self.queue, output_array.shape,
                input_region, output_region, wait_for=[colconv_evt])

        input_region = Region(workspace_array.base_data,
                workspace_array.offset*2,
                effective_workspace_shape,
                (0,1), (1,2), workspace_array_strides)
        output_region = Region(output_array.base_data,
                output_array.offset*4,
                effective_output_shape,
                (0,1), (1,2), output_array_strides)
        rowhiconv_evt = self.conv._unchecked_convolve(self.queue, output_array.shape,
                input_region, output_region, wait_for=[colconv_evt])

        return output_array, cl.enqueue_marker(self.queue, wait_for=[rowloconv_evt, rowhiconv_evt])

    def format_output(self, output, wait_for=None):
        wait_for = wait_for or []
        cl.wait_for_events(wait_for)

        output = output.get()
        lolo, lohi, hilo, hihi = tuple(output[d] for d in 'xyzw')
        highpasses = np.empty(tuple(x>>1 for x in output.shape) + (6,), np.complex64)
        highpasses[:,:,0:6:5] = q2c(lohi)
        highpasses[:,:,2:4:1] = q2c(hilo)
        highpasses[:,:,1:5:3] = q2c(hihi)

        return lolo, highpasses

def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # Create a "gold" transform
    biort, qshift = coeffs.biort('near_sym_b'), coeffs.qshift('qshift_b')
    gold = dnp.Transform2d(biort=biort, qshift=qshift)

    print('OpenCL devices:')
    [print('    {0}'.format(d.name)) for d in ctx.devices]

    print('Loading images')
    test_image_rgb = datasets.traffic_hd_rgb()
    test_image_gray = datasets.traffic_hd()
    test_image_rgba = np.dstack((test_image_rgb, np.ones_like(test_image_rgb[:,:,0])))
    print('Input shape is {0}'.format(test_image_rgba.shape))

    gold_pyramid = gold.forward(test_image_gray, 1)
    figure()
    subplot(2,1,1)
    imshow(gold_pyramid.lowpass)
    title('CPU transform lowpass')
    subplot(2,1,2)
    imshow(np.abs(gold_pyramid.highpasses[0][:,:,0]), cmap=cm.jet)
    title('CPU transform highpass')

    # Swizzle RGB image into a form friendly to transform and upload to device
    input_buffer = np.asanyarray(test_image_gray, np.float32, 'C')
    print('Input buffer size is {0}, strides {1}'.format(input_buffer.nbytes, input_buffer.strides))
    input_array = cla.empty(queue, test_image_gray.shape[:2], np.float32, order='C')
    cl.enqueue_copy(queue, input_array.data, input_buffer)

    l1 = Level1(queue, biort, input_array.shape)
    print('Level 1 requires workspace of size {0:.2f}MB'.format(l1.workspace_size/(1<<20)))
    print('Level 1 has output of sizes {0[0]:.2f}MB (lowpass) and {0[1]:.2f}MB (highpass)'.format(
        tuple(x/(1<<20) for x in l1.output_sizes)))
    l1_workspace = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, l1.workspace_size)
    l1_outputs = tuple(cl.Buffer(ctx, cl.mem_flags.READ_WRITE, s) for s in l1.output_sizes)

    runs, start, end = 0, time.time(), time.time()
    while end - start < 5:
        output, evt = l1.transform(input_array, l1_workspace)
        cl.wait_for_events([evt])
        runs += 1
        end = time.time()
    print('{0} runs, {1:.2g} per run ({2:.2f}Hz)'.format(runs, (end-start)/runs, runs/(end-start)))

    lolo, highpasses = l1.format_output(output, [evt])

    figure()
    subplot(2,1,1)
    imshow(lolo)
    title('OpenCL transform lowpass')
    subplot(2,1,2)
    imshow(np.abs(highpasses[:,:,0]), cmap=cm.jet)
    title('OpenCL transform highpass')

    print('Lowpass max abs diff', np.abs(lolo - gold_pyramid.lowpass).max())
    print('Highpass max abs diff', np.abs(highpasses - gold_pyramid.highpasses[0]).max())

    #figure()
    #climshow(input_array)
    #title('Input')

if __name__ == '__main__':
    main()
    show()
