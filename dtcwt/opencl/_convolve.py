import collections
import os
import logging

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

Region = collections.namedtuple('Region', ['data', 'shape', 'offset', 'skip', 'strides'])

def as_int4(sequence, fill_value=0):
    rv = cla.vec.make_int4(*(fill_value,)*4)
    for k, v in zip('xyzw', sequence):
        rv[k] = v
    return rv;

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
        filter_kernel = np.asanyarray(np.atleast_2d(filter_kernel), dtype=np.float32, order='C')
        if filter_kernel.shape[1] != self.filter_width:
            raise ValueError('Filter kernel has width {0}, expected {1}'.format(
                filter_kernel.shape[1], self.filter_width))
        self.filter_kernel = cla.to_device(queue, filter_kernel)

    def _copy_region(self, queue, output_shape, input_region, output_region):
        global_size = tuple(y * int(np.ceil(x/y))
                            for x, y in zip(output_shape, self.local_size))
        input_total_stride = np.product(input_region.shape)
        output_total_stride = np.product(output_region.shape)

        return self.program.copy_with_sampling(queue, global_size, self.local_size,
            as_int4(output_shape,1),
            input_region.data, as_int4(input_region.offset, 0),
            as_int4(input_region.shape, 1),
            as_int4(input_region.skip, 1), as_int4(input_region.strides, input_total_stride),
            output_region.data, as_int4(output_region.offset, 0),
            as_int4(output_region.shape, 1),
            as_int4(output_region.skip, 1), as_int4(output_region.strides, output_total_stride))

    def _unchecked_convolve(self, queue, output_shape, input_region, output_region):
        global_size = tuple(y * int(np.ceil(x/y))
                            for x, y in zip(output_shape, self.local_size))
        input_total_stride = np.product(input_region.shape)
        output_total_stride = np.product(output_region.shape)

        return self.program.convolve(queue, global_size, self.local_size,
            self.filter_kernel.data, np.int32(self.filter_kernel.shape[0]), as_int4(output_shape,1),
            input_region.data, as_int4(input_region.offset, 0),
            as_int4(input_region.shape, 1),
            as_int4(input_region.skip, 1), as_int4(input_region.strides, input_total_stride),
            output_region.data, as_int4(output_region.offset, 0),
            as_int4(output_region.shape, 1),
            as_int4(output_region.skip, 1), as_int4(output_region.strides, output_total_stride))

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
