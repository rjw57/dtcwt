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
    setup=textwrap.dedent('''
        import numpy as np
        import dtcwt
        import dtcwt.sampling
        import tests.datasets as datasets

        # Create random image
        X = np.random.rand(720,1280,3)

        # Generate random sampling co-ords
        sample_shape = (32, 32)
        ys = (4 * np.random.rand(*sample_shape) - 2) * X.shape[0]
        xs = (4 * np.random.rand(*sample_shape) - 2) * X.shape[1]
    ''')

    number_nearest, number_bilinear, number_lanczos = 10000, 1000, 100

    print('Nearest-neighbour:')
    t = timeit.Timer(setup=setup,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'nearest')")
    try:
        secs = t.timeit(number_nearest)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_nearest, secs / number_nearest
            ))
    except:
        t.print_exc()

    print('Bilinear:')
    t = timeit.Timer(setup=setup,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'bilinear')")
    try:
        secs = t.timeit(number_bilinear)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_bilinear, secs / number_bilinear
            ))
    except:
        t.print_exc()

    print('Lanczos:')
    t = timeit.Timer(setup=setup,
        stmt="Xs = dtcwt.sampling.sample(X, xs, ys, 'lanczos')")
    try:
        secs = t.timeit(number_lanczos)
        print('{0:.2e}s for {1} iterations => {2:.2e}s/iteration'.format(
                secs, number_lanczos, secs / number_lanczos
            ))
    except:
        t.print_exc()

if __name__ == '__main__':
    main()

