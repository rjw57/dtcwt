"""
A simple script to benchmark the OpenCL low-level routines in comparison to the
CPU ones.

"""

from __future__ import print_function, division

import os
import timeit

import numpy as np

from dtcwt.coeffs import biort, qshift
from dtcwt.opencl.lowlevel import NoCLPresentError

lena = np.load(os.path.join(os.path.dirname(__file__), '..', 'tests', 'lena.npz'))['lena']
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
        return 0

    t = min_time / number
    print('{0} loops, best of {1}: {2}'.format(number, repeat, format_time(t)))

    return t

def main():
    print('Running NumPy colfilter...')
    a = benchmark('colfilter(lena, h1o)',
            'from dtcwt.lowlevel import colfilter; from __main__ import lena, h1o')
    print('Running OpenCL colfilter...')
    b = benchmark('colfilter(lena, h1o)',
            'from dtcwt.opencl.lowlevel import colfilter; from __main__ import lena, h1o')
    print('Percentage speed up: {0:.0f}%'.format(1e2*a/b))
    print('=====')

    print('Running NumPy coldfilt...')
    a = benchmark('coldfilt(lena, h0b, h0a)',
            'from dtcwt.lowlevel import coldfilt; from __main__ import lena, h0b, h0a')
    print('Running OpenCL coldfilt...')
    b = benchmark('coldfilt(lena, h0b, h0a)',
            'from dtcwt.opencl.lowlevel import coldfilt; from __main__ import lena, h0b, h0a')
    print('Percentage speed up: {0:.0f}%'.format(1e2*a/b))
    print('=====')

    print('Running NumPy colifilt...')
    a = benchmark('colifilt(lena, h0b, h0a)',
            'from dtcwt.lowlevel import colifilt; from __main__ import lena, h0b, h0a')
    print('Running OpenCL colifilt...')
    b = benchmark('colifilt(lena, h0b, h0a)',
            'from dtcwt.opencl.lowlevel import colifilt; from __main__ import lena, h0b, h0a')
    print('Percentage speed up: {0:.0f}%'.format(1e2*a/b))
    print('=====')

if __name__ == '__main__':
    main()
