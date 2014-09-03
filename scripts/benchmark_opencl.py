"""
A simple script to benchmark the OpenCL low-level routines in comparison to the
CPU ones.

"""

from __future__ import print_function, division

import os
import timeit

import numpy as np
from PIL import Image

try:
    import pyopencl as cl
    HAVE_OPENCL = True
except ImportError:
    HAVE_OPENCL = False

lena = np.load(os.path.join(os.path.dirname(__file__), '..', 'tests', 'lena.npz'))['lena']
traffic_rgb = Image.open(os.path.join(os.path.dirname(__file__), '..', 'tests', 'traffic_hd.jpg'))
traffic_rgb = np.asarray(traffic_rgb, np.float32) / 255
traffic_g = np.copy(traffic_rgb[:,:,1])

def format_time(t):
    units = (
        (60*60, 'hr'), (60, 'min'), (1, 's'), (1e-3, 'ms')
    )

    for scale, unit in units:
        if t >= scale:
            return '{0:.2f} {1}'.format(t/scale, unit)

    return '{0:.2f} {1}'.format(t*1e6, 'us')

def benchmark(statement='pass', setup='pass'):
    number, repeat = (1, 3)
    min_time = 0

    while min_time < 0.2:
        number *= 10
        times = timeit.repeat(statement, setup, repeat=repeat, number=number)
        min_time = min(times)

    t = min_time / number
    print('{0} loops, best of {1}: {2}'.format(number, repeat, format_time(t)))

    return t

class NumpyBenchmark(object):
    def __init__(self, *args, **kwargs):
        from dtcwt.numpy.transform2d import Transform2d
        self.t = Transform2d()
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.t.forward(*self.args, **self.kwargs)

class OpenCLBenchmark(object):
    def __init__(self, *args, **kwargs):
        from dtcwt.opencl.transform2d import Transform2d
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.t = Transform2d(queue=self.queue)
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.t.forward(*self.args, **self.kwargs).event.wait()

def main():
    if not HAVE_OPENCL:
        print('Skipping OpenCL benchmark since OpenCL is not present')
        return

    print('Running 1 level Lena...')
    print('  NumPy')
    a = benchmark(
        'bm.run()',
        'from __main__ import lena, NumpyBenchmark; bm = NumpyBenchmark(lena, 1)')
    print('  OpenCL')
    b = benchmark(
        'bm.run()',
        'from __main__ import lena, OpenCLBenchmark; bm = OpenCLBenchmark(lena, 1)')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

    print('Running 1 level traffic (green channel)...')
    a = benchmark(
        'bm.run()',
        '''from __main__ import traffic_g, NumpyBenchmark; bm = NumpyBenchmark(traffic_g, 1)''')
    b = benchmark(
        'bm.run()',
        '''from __main__ import traffic_g, OpenCLBenchmark; bm = OpenCLBenchmark(traffic_g, 1)''')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

if __name__ == '__main__':
    main()
