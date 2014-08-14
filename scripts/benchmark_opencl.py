"""
A simple script to benchmark the OpenCL low-level routines in comparison to the
CPU ones.

"""

from __future__ import print_function, division

import os
import timeit
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dtcwt.coeffs import biort, qshift
from dtcwt.opencl.lowlevel import NoCLPresentError, get_default_queue
from tests import datasets

testimage = datasets.traffic_hd()[:,:,0]
h0o, g0o, h1o, g1o = biort('near_sym_b')
h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')

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

    try:
        while min_time < 0.2:
            number *= 10
            times = timeit.repeat(statement, setup, repeat=repeat, number=number)
            min_time = min(times)
    except NoCLPresentError:
        print('Skipping benchmark since OpenCL is not present')
        return 1

    t = min_time / number
    print('{0} loops, best of {1}: {2}'.format(number, repeat, format_time(t)))

    return t

def main():
    try:
        queue = get_default_queue()
        print('Using context: {0}'.format(queue.context))
    except NoCLPresentError:
        print('Skipping OpenCL benchmark since OpenCL is not present')

    print('Running NumPy dtwavexfm2 (one level)...')
    a = benchmark('dtwavexfm2(testimage, nlevels=1)',
            'from dtcwt.compat import dtwavexfm2; from __main__ import testimage')
    print('Running OpenCL dtwavexfm2 (one level)...')
    b = benchmark('dtwavexfm2(testimage, nlevels=1)',
            'from dtcwt.opencl.transform2d import dtwavexfm2; from __main__ import testimage')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

    print('Running NumPy pre-planned (one level)...')
    a = benchmark('t.forward(testimage, nlevels=1)',
            'from dtcwt.numpy import Transform2d; t=Transform2d(); from __main__ import testimage')
    print('Running OpenCL pre-planned (one level)...')
    b = benchmark('t.forward(testimage, nlevels=1).lowpass',
            'from dtcwt.opencl import Transform2d; t=Transform2d(); from __main__ import testimage')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

    print('Running NumPy dtwavexfm2...')
    a = benchmark('dtwavexfm2(testimage)',
            'from dtcwt.compat import dtwavexfm2; from __main__ import testimage')
    print('Running OpenCL dtwavexfm2...')
    b = benchmark('dtwavexfm2(testimage)',
            'from dtcwt.opencl.transform2d import dtwavexfm2; from __main__ import testimage')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

    print('Running NumPy pre-planned...')
    a = benchmark('t.forward(testimage)',
            'from dtcwt.numpy import Transform2d; t=Transform2d(); from __main__ import testimage')
    print('Running OpenCL pre-planned...')
    b = benchmark('t.forward(testimage).lowpass',
            'from dtcwt.opencl import Transform2d; t=Transform2d(); from __main__ import testimage')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

    print('Running NumPy dtwavexfm2 (non-POT)...')
    a = benchmark('dtwavexfm2(testimage[:510,:480])',
            'from dtcwt.compat import dtwavexfm2; from __main__ import testimage')
    print('Running OpenCL dtwavexfm2 (non-POT)...')
    b = benchmark('dtwavexfm2(testimage[:510,:480])',
            'from dtcwt.opencl.transform2d import dtwavexfm2; from __main__ import testimage')
    print('Speed up: x{0:.2f}'.format(a/b))
    print('=====')

if __name__ == '__main__':
    main()
