#!/usr/bin/env python
"""
A simple script to benchmark the sampling routines from DTCWT.

"""

from __future__ import print_function, division

import os
import sys
import textwrap
import timeit

# Add project root to path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    setup_numpy=textwrap.dedent('''
        import numpy as np
        import dtcwt
        import dtcwt.sampling
        from dtcwt.opencl.lowlevel import to_device
        import tests.datasets as datasets

        dtcwt.push_backend('numpy')

        # Create random image
        X = np.random.rand(720,1280,3).astype(np.float32)

        # Generate random sampling co-ords
        sample_shape = (200,200)
        ys = (4 * np.random.rand(*sample_shape).astype(np.float32) - 2) * X.shape[0]
        xs = (4 * np.random.rand(*sample_shape).astype(np.float32) - 2) * X.shape[1]

        print(dtcwt._sampling)
    ''')

    setup_cl=textwrap.dedent('''
        import numpy as np
        import dtcwt
        import dtcwt.sampling
        from dtcwt.opencl.lowlevel import to_device
        import tests.datasets as datasets

        dtcwt.push_backend('opencl')

        # Create random image
        X = np.random.rand(720,1280,3).astype(np.float32)

        # Generate random sampling co-ords
        sample_shape = (200,200)
        ys = (4 * np.random.rand(*sample_shape).astype(np.float32) - 2) * X.shape[0]
        xs = (4 * np.random.rand(*sample_shape).astype(np.float32) - 2) * X.shape[1]

        X = to_device(X)
        xs = to_device(xs)
        ys = to_device(ys)
    ''')

    number_nearest, number_bilinear, number_lanczos = 500, 500, 2

    print('OpenCL:')
    print('Nearest-neighbour:')
    t = timeit.Timer(setup=setup_cl,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'nearest')")
    try:
        secs = t.timeit(number_nearest)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_nearest, secs / number_nearest
            ))
    except:
        t.print_exc()

    print('Bilinear:')
    t = timeit.Timer(setup=setup_cl,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'bilinear')")
    try:
        secs = t.timeit(number_bilinear)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_bilinear, secs / number_bilinear
            ))
    except:
        t.print_exc()

#    print('Lanczos:')
#    t = timeit.Timer(setup=setup_cl,
#        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'lanczos')")
#    try:
#        secs = t.timeit(number_lanczos)
#        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
#                secs, number_lanczos, secs / number_lanczos
#            ))
#    except:
#        t.print_exc()

    number_nearest, number_bilinear, number_lanczos = 500, 30, 2

    print('NUMPY:')
    print('Nearest-neighbour:')
    t = timeit.Timer(setup=setup_numpy,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'nearest')")
    try:
        secs = t.timeit(number_nearest)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_nearest, secs / number_nearest
            ))
    except:
        t.print_exc()

    print('Bilinear:')
    t = timeit.Timer(setup=setup_numpy,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'bilinear')")
    try:
        secs = t.timeit(number_bilinear)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_bilinear, secs / number_bilinear
            ))
    except:
        t.print_exc()

#    print('Lanczos:')
#    t = timeit.Timer(setup=setup_numpy,
#        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'lanczos')")
#    try:
#        secs = t.timeit(number_lanczos)
#        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
#                secs, number_lanczos, secs / number_lanczos
#            ))
#    except:
#        t.print_exc()

if __name__ == '__main__':
    main()

