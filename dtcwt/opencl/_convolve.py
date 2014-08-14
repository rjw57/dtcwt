import os
import logging

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

class Convolution(object):
    def __init__(self, ctx, filter_width, input_components=1):
        if filter_width % 2 != 1:
            raise ValueError('Filter width must be odd')
        if filter_width <= 1:
            raise ValueError('Filter width must be > 1')
        if input_components not in [1, 4]:
            raise ValueError('Number of input components must be 1 or 4')

        self.filter_width = filter_width
        self.ctx = ctx

        if input_components == 1:
            self.input_type = 'float'
            self.input_dtype = np.float32
            self.input_itemsize = 4
        elif input_components == 4:
            self.input_type = 'float4'
            self.input_dtype = cla.vec.float4
            self.input_itemsize = 4

        # Compute best work group size for each device
        optimal_local_size = None
        for d in ctx.devices:
            device_optimal_local_size = self._optimal_local_size_for_device(d)
            logging.debug('Optimal local size for {0} is {1}'.format(d.name,
                device_optimal_local_size))
            if optimal_local_size is not None and \
                    np.product(optimal_local_size) > np.product(device_optimal_local_size):
                optimal_local_size = device_optimal_local_size
            elif optimal_local_size is None:
                optimal_local_size = device_optimal_local_size

        logging.debug('Overall optimal local size is {0}'.format(optimal_local_size))
        self.local_size = optimal_local_size
        self.program = self._build_program()
        self.filter_kernel = None

    def set_filter_kernel(self, queue, filter_kernel):
        if filter_kernel.shape[0] != self.filter_width:
            raise ValueError('Filter kernel has length {0}, expected {1}'.format(
                filter_kernel.shape[0], self.filter_width))
        self.filter_kernel = cla.to_device(queue, np.asanyarray(filter_kernel, np.float32, 'C'))

    def _checked_convolve(self,
            input_array, input_offset, input_skip, input_strides,
            output_array, output_offset, output_skip, output_strides, output_shape):
        """{input,output}_{offset,skip} and output_shape must be
        pyopencl.array.vec.float4 instances.

        """
        if self.filter_kernel is None:
            raise RuntimeError('No filter kernel set')

        output_shape_tup = (output_shape['x'], output_shape['y'])
        global_size = tuple(y * int(np.ceil(x/y))
                            for x, y in zip(output_shape_tup, self.local_size))
        global_size_arr = np.array(global_size)
        fw_arr = np.array(((self.filter_width-1)>>1, 0))

        # Firstly, just check that this convolution won't scribble in la-la land
        input_skip_array = np.array((input_skip['x'], input_skip['y']))
        input_offset_array = np.array((input_offset['x'], input_offset['y']))
        if np.any(input_offset_array - fw_arr*input_skip_array < 0):
            raise ValueError('Input array will be read from before start')
        if np.any(input_offset_array +
                (fw_arr+global_size_arr)*input_skip_array > np.array(input_array.shape)):
            raise ValueError('Input array will be read from after end')

        output_shape_array = np.array((output_shape['x'], output_shape['y']))
        output_skip_array = np.array((output_skip['x'], output_skip['y']))
        output_offset_array = np.array((output_offset['x'], output_offset['y']))
        if np.any(output_offset_array < 0):
            raise ValueError('Output array will be written to before start')
        if np.any(output_offset_array + global_size_arr*output_skip_array >
                np.array(output_array.shape)):
            raise ValueError('Output array will be written to after end')

        return self._unchecked_convolve(
            input_array, input_offset, input_skip, input_strides,
            output_array, output_offset, output_skip, output_strides, output_shape)

    def _unchecked_convolve(self,
            input_array, input_offset, input_skip, input_strides,
            output_array, output_offset, output_skip, output_strides, output_shape):
        assert input_array.queue is output_array.queue

        output_shape_tup = (output_shape['x'], output_shape['y'])
        global_size = tuple(y * int(np.ceil(x/y))
                            for x, y in zip(output_shape_tup, self.local_size))

        return self.program.convolve(output_array.queue, global_size, self.local_size,
            input_array.data, input_offset, input_skip, input_strides,
            self.filter_kernel.data,
            output_array.data, output_offset, output_skip, output_shape, output_strides)

    def _optimal_local_size_for_device(self, device):
        if device.max_work_item_dimensions < 2:
            raise ValueError('Device {0} does not support at least 2d work items'.format(d.name))

        # Sanity check: never go above this size no matter what device claims
        hard_max_N, hard_max_M = 32, 32

        # Let local size be N, M. With a filter width of W, the amount of
        # redundancy is (W-1)*M. If we set M=1, however, then we may spend more
        # time than is necessary padding small input. Start with as square an
        # input as we can manage.

        N = int(np.ceil(np.sqrt(device.max_work_group_size)))
        M = int(np.floor(np.sqrt(device.max_work_group_size)))

        # How much local memory will this require?
        local_mem_required = self.input_itemsize * (N+self.filter_width-1) * M
        logging.debug('Size {0} requires {1} local memory out of {2} on {3}'.format(
            (N,M), local_mem_required, device.local_mem_size, device.name))

        # If this is supported by the device, excellent
        if local_mem_required <= device.local_mem_size:
            return (min(hard_max_N, N), min(hard_max_M, M))

        # Otherwise, keep N as big as possible and choose M
        max_n = int(np.floor((device.local_mem_size / self.input_itemsize)
            - self.filter_width + 1))
        if max_n < 1:
            raise ValueError('Device has too little local memory')
        N = min(N, max_n)
        M = int(np.floor(device.local_mem_size /
                (self.input_itemsize * float(N + self.filter_width - 1))))
        local_mem_required = self.input_itemsize * (N+self.filter_width-1) * M
        logging.debug('Size {0} requires {1} local memory out of {2} on {3}'.format(
            (N,M), local_mem_required, device.local_mem_size, device.name))

        return (min(hard_max_N, N), min(hard_max_M, M))

    def _build_program(self):
        return cl.Program(self.ctx,
            open(os.path.join(os.path.dirname(__file__), 'convolve.cl')).read()).build([
                '-DINPUT_TYPE={0}'.format(self.input_type),
                '-DFILTER_WIDTH={0}'.format(self.filter_width),
                '-DLOCAL_SIZE_0={0}'.format(self.local_size[0]),
                '-DLOCAL_SIZE_REST={0}'.format(np.product(self.local_size[1:])),
            ])
